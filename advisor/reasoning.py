"""Recommendation logic.

Inputs:
  - a HardwareProfile (from advisor.hardware)
  - a ModelCatalog (from advisor.model_catalog)

Outputs:
  - A Recommendation: one primary local pick, one primary free cloud pick,
    and a ranked list of viable alternates (other locals + other free clouds).

We do not hardcode "if RAM > 32 pick 70B" rules. Instead we:

  1. Filter the catalog: a model is "viable" if it can plausibly run on
     this machine. The definition of "plausibly" depends on the kind of
     memory pool available (Apple Silicon unified vs. dedicated VRAM vs.
     plain system RAM). We use a conservative headroom model from the
     catalog.

  2. Score each viable model on a weighted combination of:
        - fit_with_hardware  (does it run comfortably, or is it a tight squeeze?)
        - coding_quality     (catalog's coding_score)
        - agentic_quality    (instruction following, tool use)
        - speed              (how snappy will it feel on this hardware?)
        - context_window     (long context is genuinely useful for a coding agent)

     Weights shift depending on the detected hardware. A machine with
     abundant RAM/VRAM weights quality higher; a constrained machine
     weights fit-with-hardware higher.

  3. Pick the highest-scoring local and the highest-scoring free cloud
     model as the two defaults. If the local side is empty (e.g. we
     detected essentially nothing about the machine), we fall back to the
     smallest model in the catalog as a guaranteed-runs-anywhere pick.

  4. Optionally ask an LLM (Claude) to write a richer narrative explanation
     of the picks. The LLM never sees the catalog or hardware directly —
     it only sees a compact summary the local logic already produced, so
     the local logic is the source of truth. If no API key is configured,
     we fall back to a deterministic templated explanation.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass, field
from typing import Optional

from . import llm
from .hardware import HardwareProfile
from .model_catalog import (
    BYTES_PER_PARAM,
    DEFAULT_QUANT,
    ModelCatalog,
    ModelEntry,
    RAM_HEADROOM_GB,
    VRAM_HEADROOM_GB,
)


# ---------------------------------------------------------------------------
# Result types
# ---------------------------------------------------------------------------

@dataclass
class RankedModel:
    """A model in the final ranked list, with enough context for the CLI
    to explain why it's where it is."""
    model: ModelEntry
    score: float
    fit: str            # "comfortable" | "tight" | "squeezed"
    reasons: list[str] = field(default_factory=list)


@dataclass
class Recommendation:
    profile: HardwareProfile
    primary_local: Optional[RankedModel]
    primary_cloud: Optional[RankedModel]
    alternates: list[RankedModel]
    explanation: str            # human-readable, specific to this hardware


# ---------------------------------------------------------------------------
# Viability + scoring
# ---------------------------------------------------------------------------

def _effective_memory_gb(profile: HardwareProfile) -> float:
    """How much memory can a model actually use on this machine, with the
    OS / runtime overhead already subtracted.

    Three regimes:

      1. Apple Silicon (unified memory): the entire system RAM is the model
         pool, minus OS/runtime headroom. Apple Silicon is genuinely good at
         this; we use a small fixed headroom.

      2. Dedicated NVIDIA/AMD GPU present: we treat the pool as VRAM plus a
         generous slice of system RAM. llama.cpp routinely runs models that
         spill beyond VRAM by offloading most layers to the GPU and a
         handful to the CPU — that's a normal, supported config, not a
         footgun. So the pool is roughly: (VRAM - small VRAM headroom) +
         (system RAM - OS headroom), where the system-RAM slice is capped
         to avoid recommending something the user can't actually run while
         their IDE + browser are open.

      3. CPU-only: system RAM minus a sensible headroom. The 50% cap keeps
         us honest on machines with a lot of RAM (don't tell someone with
         128 GB they can run a 100 GB model — that's still gonna swap).

    Returns -1.0 when we have no useful information about the host (no RAM
    detected and no GPU detected). Callers should treat that as "fall back
    to the smallest model" rather than a zero-sized pool.
    """
    ram = profile.ram_total_gb or 0.0
    vram = profile.best_dedicated_vram_gb

    os_key = "apple_silicon" if profile.is_apple_silicon else (
        (profile.os_name or "unknown").lower()
    )
    absolute_headroom = RAM_HEADROOM_GB.get(os_key, RAM_HEADROOM_GB["unknown"])

    if profile.is_apple_silicon:
        # Unified memory: whole RAM minus a small headroom.
        headroom = max(2.0, min(absolute_headroom, ram * 0.5))
        pool = max(0.0, ram - headroom)
        return pool if pool > 0 else -1.0

    if vram > 0 and ram > 0:
        # Dedicated-GPU box: VRAM (for the GPU-resident layers) plus a
        # generous slice of system RAM (for offloaded CPU layers).
        vram_usable = max(0.0, vram - VRAM_HEADROOM_GB)
        # Allow up to 70% of system RAM to participate in offload. The OS
        # + dev tools usually sit comfortably in the remaining 30% even
        # while a 20 GB model is loading.
        ram_usable = max(0.0, ram - max(2.0, absolute_headroom))
        ram_offload_allowance = ram_usable * 0.70
        pool = vram_usable + ram_offload_allowance
        return pool

    if vram > 0:
        # GPU present but no RAM detected: fall back to VRAM alone, with
        # a small headroom. This shouldn't happen in practice but we
        # don't want to crash.
        return max(0.0, vram - VRAM_HEADROOM_GB)

    if ram > 0:
        # CPU-only: keep the 50% cap so we don't recommend 100 GB on a
        # 128 GB box.
        headroom = max(2.0, min(absolute_headroom, ram * 0.5))
        pool = max(0.0, ram - headroom)
        return pool if pool > 0 else -1.0

    return -1.0  # no useful info


def _model_memory_need_gb(model: ModelEntry) -> float:
    """How much RAM/VRAM this model needs at its default quant."""
    if model.kind == "cloud":
        return 0.0
    if model.ram_q4_gb is not None and model.default_quant == DEFAULT_QUANT:
        return model.ram_q4_gb
    if model.params_b is None:
        return 0.0
    bpp = BYTES_PER_PARAM.get(model.default_quant, BYTES_PER_PARAM[DEFAULT_QUANT])
    return round(model.params_b * bpp / 8.0, 2)


def _classify_fit(profile: HardwareProfile, model: ModelEntry, pool_gb: float) -> str:
    """How comfortable is the model on this machine?

    comfortable: model uses <= 55% of the usable memory pool
    tight:       55..80%
    squeezed:    80..100% (it will run, but the user will feel it)
    """
    need = _model_memory_need_gb(model)
    if need <= 0 or pool_gb <= 0:
        return "unknown"
    ratio = need / pool_gb
    if ratio <= 0.55:
        return "comfortable"
    if ratio <= 0.80:
        return "tight"
    if ratio <= 1.0:
        return "squeezed"
    return "too-big"


def _viable(profile: HardwareProfile, model: ModelEntry, pool_gb: float) -> bool:
    """A model is viable if it fits in the usable memory pool with at
    least a small margin. We allow 'squeezed' but not 'too-big'.

    Special case: server_class models (e.g. Qwen3-Coder-480B, DeepSeek
    V4 Pro) require multi-GPU / huge unified memory. They are only viable
    if the effective memory pool is at least 150 GB, regardless of fit
    classification. This keeps us from quietly recommending something
    that won't actually load on a consumer machine.
    """
    if model.server_class:
        return pool_gb >= 150.0
    if pool_gb < 0:
        # Unknown host: only trust models <= 4 GB of RAM need.
        return _model_memory_need_gb(model) <= 4.0
    fit = _classify_fit(profile, model, pool_gb)
    return fit in ("comfortable", "tight", "squeezed")


def _score_local(
    profile: HardwareProfile,
    model: ModelEntry,
    pool_gb: float,
) -> RankedModel:
    """Score a local model for this machine. Returns a RankedModel with
    reasons so the CLI can show the user *why* this model landed where
    it did."""
    reasons: list[str] = []
    fit = _classify_fit(profile, model, pool_gb)
    need = _model_memory_need_gb(model)

    # 1. Fit-with-hardware (0..10). Penalize tight/squeezed fits.
    if fit == "comfortable":
        fit_score = 10.0
        reasons.append(
            f"It will run comfortably on your machine "
            f"(~{need:.0f} GB needed vs. ~{pool_gb:.0f} GB available)."
        )
    elif fit == "tight":
        fit_score = 7.0
        reasons.append(
            f"It will fit on your machine but uses most of the available memory "
            f"(~{need:.0f} GB needed vs. ~{pool_gb:.0f} GB available); close other apps for best speed."
        )
    elif fit == "squeezed":
        fit_score = 4.5
        reasons.append(
            f"It will technically run but will use nearly all of your memory "
            f"(~{need:.0f} GB needed vs. ~{pool_gb:.0f} GB available); expect slowdowns."
        )
    else:
        fit_score = 3.0
        reasons.append("Memory situation is unclear; this is a conservative pick.")

    # 2. Speed
    speed = model.speed_score
    vram = profile.best_dedicated_vram_gb
    model_fits_in_vram = (vram > 0) and (need <= vram * 0.85)
    if vram >= 8 and model_fits_in_vram:
        # Real GPU, model fits in VRAM: bump perceived speed a little.
        speed = min(10.0, speed + 0.5)
        if model.needs_dedicated_gpu in ("yes", "sometimes"):
            reasons.append("Your dedicated GPU will give it a meaningful speed boost.")
    elif vram >= 8 and not model_fits_in_vram:
        # Real GPU but the model is bigger than VRAM — partial CPU offload.
        speed = max(0.0, speed - 1.5)
        reasons.append(
            "Your GPU is real but smaller than this model; llama.cpp will offload some "
            "layers to your CPU. Still much faster than CPU-only, but slower than a fully GPU-resident run."
        )
    elif vram > 0:
        if model.needs_dedicated_gpu == "yes":
            speed = max(0.0, speed - 2.0)
            reasons.append("Your GPU is small for this model; CPU offload will slow it down.")
    else:
        if model.needs_dedicated_gpu == "yes":
            speed = max(0.0, speed - 3.0)
            reasons.append("This model really wants a dedicated GPU; on CPU it will be slow.")
        elif model.needs_dedicated_gpu == "sometimes":
            speed = max(0.0, speed - 1.0)
            reasons.append("No dedicated GPU detected; this model will run on CPU.")

    # 3. Coding quality (use the catalog score, lightly downweighted on
    #    very constrained machines where reliability matters more than
    #    ceiling quality).
    coding = model.coding_score
    if fit in ("squeezed", "tight") and pool_gb < 8:
        coding = coding * 0.9

    # 4. Context window (bonus for long context)
    ctx_bonus = 0.0
    if model.context_window >= 100000:
        ctx_bonus = 0.6
        reasons.append(f"Very long context window ({model.context_window // 1000}k tokens).")
    elif model.context_window >= 30000:
        ctx_bonus = 0.3
        reasons.append(f"Long context window ({model.context_window // 1000}k tokens).")

    # 5. Agentic score
    agentic = model.agentic_score

    # Weighting. On abundant machines quality dominates; on tight
    # machines fit dominates.
    if fit == "comfortable" and pool_gb >= 16:
        w_fit, w_coding, w_agentic, w_speed = 1.5, 2.5, 2.0, 1.0
    elif fit in ("tight", "squeezed") or pool_gb < 8:
        w_fit, w_coding, w_agentic, w_speed = 3.0, 1.5, 1.5, 1.0
    else:
        w_fit, w_coding, w_agentic, w_speed = 2.0, 2.0, 1.5, 1.0

    score = (
        w_fit * fit_score
        + w_coding * coding
        + w_agentic * agentic
        + w_speed * speed
        + ctx_bonus
    )

    return RankedModel(model=model, score=score, fit=fit, reasons=reasons)


def _score_cloud(profile: HardwareProfile, model: ModelEntry) -> RankedModel:
    reasons: list[str] = []
    reasons.append("Cloud model — runs on the provider's hardware, so no local resource limits apply.")
    if model.context_window >= 200000:
        reasons.append(f"Very long context window ({model.context_window // 1000}k tokens).")
    elif model.context_window >= 100000:
        reasons.append(f"Long context window ({model.context_window // 1000}k tokens).")
    if not model.free_tier:
        reasons.append("Not on a free tier — listed as a paid upgrade path only.")
    if model.family == "claude":
        reasons.append("Claude models are particularly strong at agentic / tool-use workflows.")

    # Clouds are always "comfortable" — they don't use local resources.
    # Score = quality-weighted.
    coding = model.coding_score
    agentic = model.agentic_score
    speed = model.speed_score
    free_bonus = 1.5 if model.free_tier else 0.0
    score = 2.5 * coding + 2.0 * agentic + 1.0 * speed + free_bonus
    return RankedModel(model=model, score=score, fit="cloud", reasons=reasons)


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def recommend(
    profile: HardwareProfile,
    catalog: Optional[ModelCatalog] = None,
    use_llm: bool = True,
) -> Recommendation:
    """Build a Recommendation for the given hardware profile."""
    catalog = catalog or ModelCatalog()
    pool_gb = _effective_memory_gb(profile)

    # Score + filter locals
    local_ranked: list[RankedModel] = []
    for m in catalog.local:
        if not _viable(profile, m, pool_gb):
            continue
        local_ranked.append(_score_local(profile, m, pool_gb))
    local_ranked.sort(key=lambda r: r.score, reverse=True)

    # Score clouds. For the *primary* pick we restrict to free-tier models
    # only — the spec is explicitly "best free cloud option", and surfacing
    # a paid model as the default would violate that. Non-free models are
    # still scored and shown in the alternates list with a clear note.
    cloud_ranked: list[RankedModel] = []
    for m in catalog.cloud:
        cloud_ranked.append(_score_cloud(profile, m))
    cloud_ranked.sort(key=lambda r: r.score, reverse=True)
    free_cloud_ranked = [r for r in cloud_ranked if r.model.free_tier]

    # Pick primaries
    primary_local = local_ranked[0] if local_ranked else None
    primary_cloud = free_cloud_ranked[0] if free_cloud_ranked else (
        cloud_ranked[0] if cloud_ranked else None
    )

    # If the local side is empty (or viability was so strict nothing fit),
    # we still want *some* local recommendation so the user has a path
    # forward. Pick the smallest model in the catalog — it will basically
    # always run.
    if primary_local is None:
        fallback = min(catalog.local, key=lambda m: _model_memory_need_gb(m))
        primary_local = RankedModel(
            model=fallback,
            score=0.0,
            fit="fallback",
            reasons=[
                "We couldn't confidently size your hardware, so this is the "
                "smallest model in the catalog — it should run on essentially any machine.",
            ],
        )
        # Ensure it shows up in alternates too
        if not any(r.model.id == fallback.id for r in local_ranked):
            local_ranked.append(primary_local)

    # Build the alternates list:
    #   - everything in local_ranked except the primary
    #   - everything in cloud_ranked except the primary cloud
    #   - de-duplicated by model id
    alts: list[RankedModel] = []
    seen: set[str] = set()

    if primary_local:
        seen.add(primary_local.model.id)
    if primary_cloud:
        seen.add(primary_cloud.model.id)

    for r in local_ranked:
        if r.model.id in seen:
            continue
        seen.add(r.model.id)
        alts.append(r)
    for r in cloud_ranked:
        if r.model.id in seen:
            continue
        seen.add(r.model.id)
        alts.append(r)

    alts.sort(key=lambda r: r.score, reverse=True)

    # Build the explanation: the LLM is the narrator when asked for (and
    # reachable); otherwise we fall back to a short, honest, fact-specific
    # message. Either way the local logic already made the picks.
    if use_llm:
        explanation = _maybe_llm_explain(profile, primary_local, primary_cloud, alts)
    else:
        explanation = ""
    if not explanation:
        explanation = _minimal_explain(profile, primary_local, primary_cloud, alts)

    return Recommendation(
        profile=profile,
        primary_local=primary_local,
        primary_cloud=primary_cloud,
        alternates=alts,
        explanation=explanation,
    )


# ---------------------------------------------------------------------------
# Explanations
# ---------------------------------------------------------------------------

def _profile_summary(profile: HardwareProfile, pool_gb: float) -> str:
    parts: list[str] = []
    if profile.os_name:
        s = profile.os_name
        if profile.os_version:
            s += f" {profile.os_version}"
        if profile.arch:
            s += f" ({profile.arch})"
        parts.append(s)
    if profile.cpu_model:
        cpu = profile.cpu_model
        if profile.cpu_cores_logical:
            cpu += f" ({profile.cpu_cores_logical} threads)"
        parts.append(cpu)
    if profile.ram_total_gb:
        parts.append(f"{profile.ram_total_gb:.0f} GB RAM")
    if profile.gpus:
        g = profile.gpus[0]
        gpu_str = g.name
        if g.vram_gb:
            gpu_str += f" ({g.vram_gb:.0f} GB VRAM)"
        elif g.notes:
            gpu_str += f" ({g.notes})"
        parts.append(gpu_str)
    if profile.disk_free_gb:
        parts.append(f"{profile.disk_free_gb:.0f} GB free disk")
    parts.append(f"~{pool_gb:.0f} GB usable for models")
    return " · ".join(parts) or "unknown hardware"


def _fit_clause(fit: str) -> str:
    """A short, varied clause describing how comfortably a model fits. Kept
    as one tail of a sentence whose opening already varies by machine class,
    so the overall message is not a single reusable template."""
    if fit == "comfortable":
        return "fits here with room to spare"
    if fit == "tight":
        return "fits, but will leave little headroom while it runs"
    if fit == "squeezed":
        return "will run, though it uses just about all of the available memory"
    return "should run on almost any machine"


def _minimal_explain(
    profile: HardwareProfile,
    primary_local: Optional[RankedModel],
    primary_cloud: Optional[RankedModel],
    alts: list[RankedModel],
) -> str:
    """Short, honest, fact-specific fallback used when no LLM is reachable.

    The opening sentence is shaped by what is actually notable about the
    detected machine — Apple Silicon unified memory, a dedicated GPU as a
    VRAM ceiling, a small no-GPU box, a CPU-only box, or an unknown one — so
    it does not read as a boilerplate with numbers swapped in. Every
    number/name comes straight from the profile.
    """
    pool = _effective_memory_gb(profile)
    sentences: list[str] = []

    # 1. The single most defining fact about THIS machine.
    if profile.is_apple_silicon:
        chip = profile.cpu_model or "Apple Silicon Mac"
        ram = f"{profile.ram_total_gb:.0f}" if profile.ram_total_gb else "?"
        sentences.append(
            f"Your {chip} carries {ram} GB of unified memory — macOS and the "
            f"runtime keep a slice of it, leaving roughly {pool:.0f} GB for a "
            f"model to actually use."
        )
    elif profile.best_dedicated_vram_gb > 0:
        gpu = next((g for g in profile.gpus if g.vram_gb), None)
        gpu_name = gpu.name if (gpu and gpu.name) else "your GPU"
        sentences.append(
            f"The hard ceiling here is {gpu_name}: {profile.best_dedicated_vram_gb:.0f} GB "
            f"of VRAM for the GPU-resident layers, plus whatever spills over to "
            f"your CPU, for about {pool:.0f} GB to work with."
        )
    elif profile.ram_total_gb and profile.ram_total_gb < 12:
        sentences.append(
            f"This box has only {profile.ram_total_gb:.0f} GB of RAM and no "
            f"dedicated GPU, which caps a local model at roughly {pool:.0f} GB "
            f"— big flagships are simply out of the question here."
        )
    elif profile.ram_total_gb:
        sentences.append(
            f"This is a CPU-only machine with {profile.ram_total_gb:.0f} GB of RAM, "
            f"so a local model can draw on about {pool:.0f} GB but will run on "
            f"the processor."
        )
    else:
        sentences.append(
            "We couldn't learn much about this machine, so the picks below are "
            "deliberately conservative."
        )

    # 2. The local pick, tied to the opening above.
    if primary_local:
        m = primary_local.model
        if m.kind == "local":
            need = _model_memory_need_gb(m)
            sentences.append(
                f"{m.id} is the local pick — at {m.default_quant} it needs "
                f"~{need:.0f} GB, which {_fit_clause(primary_local.fit)}."
            )

    # 3. The free cloud pick — it avoids the local memory problem entirely.
    if primary_cloud:
        m = primary_cloud.model
        if m.free_tier:
            sentences.append(
                f"If you'd rather not spend local RAM at all, {m.id} is the "
                f"free cloud fallback — Anthropic runs it on their own "
                f"hardware, costing your machine nothing."
            )

    return " ".join(sentences)


# ---------------------------------------------------------------------------
# Optional LLM-narrated explanation
# ---------------------------------------------------------------------------

def _maybe_llm_explain(
    profile: HardwareProfile,
    primary_local: Optional[RankedModel],
    primary_cloud: Optional[RankedModel],
    alts: list[RankedModel],
) -> str:
    """If ANTHROPIC_API_KEY is set, call Claude to write the natural-language
    explanation. Returns "" on any failure so the caller can fall back to
    the minimal fact-specific message.

    We deliberately keep the LLM's job small: it only writes the words. The
    picks are already made by the local scoring logic; the LLM is given a
    compact, real summary of the hardware and those picks and asked to talk
    about this exact machine — never to second-guess the choices.
    """
    if not llm.api_key():
        return ""

    pool_gb = _effective_memory_gb(profile)

    def _r_to_dict(r: Optional[RankedModel]) -> Optional[dict]:
        if r is None:
            return None
        return {
            "id": r.model.id,
            "kind": r.model.kind,
            "score": round(r.score, 2),
            "fit": r.fit,
            "params_b": r.model.params_b,
            "ram_q4_gb": r.model.ram_q4_gb,
            "context_window": r.model.context_window,
            "coding_score": r.model.coding_score,
            "agentic_score": r.model.agentic_score,
            "reasons": r.reasons,
        }

    payload = {
        "hardware": {
            "summary": _profile_summary(profile, pool_gb),
            "os_name": profile.os_name,
            "cpu_model": profile.cpu_model,
            "ram_total_gb": profile.ram_total_gb,
            "best_dedicated_vram_gb": profile.best_dedicated_vram_gb,
            "is_apple_silicon": profile.is_apple_silicon,
            "gpus": [{"vendor": g.vendor, "name": g.name, "vram_gb": g.vram_gb} for g in profile.gpus],
            "disk_free_gb": profile.disk_free_gb,
            "notes": profile.notes,
        },
        "primary_local": _r_to_dict(primary_local),
        "primary_cloud": _r_to_dict(primary_cloud),
        "top_alternates": [_r_to_dict(r) for r in alts[:5] if r is not None],
    }

    system_prompt = (
        "You are the explanation layer of OliveCode, a tool that recommends "
        "AI coding models based on the user's hardware. The engine has already "
        "chosen the picks — you only write the words. Write one short, warm "
        "paragraph (~100-160 words) explaining why these picks suit THIS "
        "machine.\n\n"
        "Requirements:\n"
        "- Reference the actual detected hardware by name and number: the exact "
        "CPU/chip model, the exact RAM or VRAM figure, the exact model ids of "
        "the two picks. Never write 'your device' or 'your machine' without a "
        "concrete spec attached.\n"
        "- Shape the message around what is genuinely notable here: e.g. unified "
        "memory on Apple Silicon, a GPU smaller than the model, a huge memory "
        "pool, or an old box with no GPU. Vary your sentence structure to fit "
        "that reality; do not reuse a stock paragraph with numbers swapped in.\n"
        "- Do not recommend different models and do not hedge with 'you could "
        "also try' lists.\n"
        "- Never invent facts. If a number is not in the input, do not make one up.\n"
        "- Output plain prose only — no bullet points, no JSON, no headers."
    )

    user_prompt = (
        "Here is the detected hardware and the engine's picks. Write the "
        "explanation.\n\n" + json.dumps(payload, indent=2)
    )

    return llm.anthropic_completion(
        system=system_prompt,
        user=user_prompt,
        max_tokens=600,
    )


__all__ = [
    "recommend",
    "Recommendation",
    "RankedModel",
]
