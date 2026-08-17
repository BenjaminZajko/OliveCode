"""Curated knowledge base of models usable as a coding agent.

We track two categories:

  1. Local models runnable via Ollama, across a wide range of sizes and
     common quantizations. For each we record:
        - ollama tag (the thing the user actually types)
        - parameter count (billions)
        - typical RAM footprint at the recommended quantization (GB)
        - whether it benefits from a dedicated GPU (yes/no/sometimes)
        - context window
        - relative coding strength (0..10)
        - relative instruction-following / agentic-tool-use strength (0..10)
        - one-line "best for"
        - honest "watch out for" notes

  2. Free / no-signup cloud models (Claude family on its free tier, plus
     other notable free-tier-accessible options from Google, Mistral,
     OpenRouter, etc.). These have no local resource requirements but DO
     have rate limits, regional availability, and capability differences
     we should be honest about.

All numbers are conservative defaults based on publicly available info as
of early 2026. They are intentionally rough — the recommender treats them
as guidance, not gospel. If the user has a faster box than these assume,
they'll just get a bit of headroom; if their box is weaker, the model may
not actually run and they'll find out the first time they try.
"""

from __future__ import annotations

from dataclasses import MISSING, asdict, dataclass, field
from typing import Optional


# ---------------------------------------------------------------------------
# Data model
# ---------------------------------------------------------------------------

# RAM multipliers to go from raw parameter count to a rough quantized
# footprint in bytes-per-parameter. Q4_K_M is the usual "good" pick for
# general use; we expose constants for the other common quantizations so
# the recommender can pick based on available headroom.
BYTES_PER_PARAM = {
    "Q2_K": 2.6,
    "Q3_K_M": 3.3,
    "Q4_K_M": 4.4,   # default; good speed/quality tradeoff
    "Q5_K_M": 5.4,
    "Q6_K": 6.4,
    "Q8_0": 8.5,
    "F16": 16.0,
}

DEFAULT_QUANT = "Q4_K_M"

# Overhead the OS + IDE + Ollama runtime itself need on top of the model
# weights. This is intentionally generous so we don't recommend a model
# that technically "fits" in RAM but swaps the moment the user opens
# Chrome.
RAM_HEADROOM_GB = {
    "apple_silicon": 4.0,   # Apple Silicon is more efficient, less host overhead
    "linux": 3.0,
    "windows": 4.0,         # Windows tends to use more RAM for the OS
    "unknown": 4.0,
}

VRAM_HEADROOM_GB = 1.5      # CUDA context + activations + Ollama runtime


@dataclass
class ModelEntry:
    """One model in the catalog."""
    id: str                                  # canonical id, e.g. "llama3.1:8b"
    family: str                              # "llama", "qwen2.5-coder", "claude", "gemini", ...
    kind: str                                # "local" | "cloud"
    params_b: Optional[float]                # None for closed models we don't disclose
    context_window: int
    active_params_b: Optional[float] = None  # for MoE models: params active per token
    # Resource requirements (for local models)
    ram_q4_gb: Optional[float] = None        # RAM footprint at Q4_K_M
    min_ram_gb: Optional[float] = None       # absolute minimum RAM to load (incl. headroom)
    needs_dedicated_gpu: str = "no"          # "no" | "sometimes" | "yes"
    default_quant: str = DEFAULT_QUANT
    # Quality scores, 0..10
    coding_score: float = 5.0
    agentic_score: float = 5.0               # instruction following, tool use
    speed_score: float = 5.0                 # perceived tokens/sec on typical hardware
    # Notes
    best_for: str = ""
    watch_out: str = ""
    # Catalog-management flags
    server_class: bool = False               # requires multi-GPU / very large unified memory
    flagship: bool = False                   # "best local, no compromises" tier for strong machines
    license: str = "apache-2.0"              # informational; not a hard restriction
    # For cloud models
    free_tier: bool = True
    signup_required: bool = True
    notes: str = ""

    def estimated_ram_gb(self) -> float:
        """Best-guess RAM footprint at this model's default quant.

        For MoE models, the weights still all have to be loaded into memory
        even if only a fraction of the params are active per token. So we
        size on `params_b`, not `active_params_b`.
        """
        if self.ram_q4_gb is not None and self.default_quant == DEFAULT_QUANT:
            return self.ram_q4_gb
        if self.params_b is None:
            return 0.0
        bpp = BYTES_PER_PARAM.get(self.default_quant, BYTES_PER_PARAM[DEFAULT_QUANT])
        return round(self.params_b * bpp / 8.0, 2)  # GB = bytes / 1024^3, simplified

    def to_dict(self) -> dict:
        """Plain-dict form, used by the local catalog cache."""
        return asdict(self)

    @classmethod
    def from_dict(cls, d: dict) -> "ModelEntry":
        """Rebuild a ModelEntry from `to_dict` output. Unknown keys are
        ignored (so a newer cache version can't break an older codebase) and
        missing fields fall back to their dataclass defaults — e.g. cloud
        models routinely omit `params_b` because closed models don't disclose
        it. Never raises on well-formed dicts; malformed values surface as
        normal construction errors."""
        kwargs: dict = {}
        for name, fld in cls.__dataclass_fields__.items():  # type: ignore[attr-defined]
            if name in d and d[name] is not None:
                kwargs[name] = d[name]
                continue
            if fld.default is not MISSING:
                kwargs[name] = fld.default
            elif fld.default_factory is not MISSING:  # type: ignore[misc]
                kwargs[name] = fld.default_factory()  # type: ignore[misc]
            else:
                kwargs[name] = None
        return cls(**kwargs)


# ---------------------------------------------------------------------------
# Local (Ollama) catalog
# ---------------------------------------------------------------------------
# Heuristics for the local list:
#   - We cover the size ladder 1B -> 70B so anything from a 4GB-RAM mini-PC
#     to a 64GB-RAM Mac Studio has a real pick.
#   - We favor families known to be good at code and tool use:
#     Qwen2.5-Coder, Llama 3.1/3.3, DeepSeek-Coder-V2, Phi-3.5, Mistral,
#     Gemma 2, CodeLlama.
#   - Quant defaults are Q4_K_M for >= 7B (best quality/size tradeoff) and
#     Q5_K_M for tiny models (less aggressive quantization on small models
#     noticeably improves coherence).
#   - `needs_dedicated_gpu` is "yes" if the model is too large to run
#     comfortably on most CPUs in real time, "sometimes" if it benefits
#     but isn't required, "no" if it's CPU-friendly.

LOCAL_MODELS: list[ModelEntry] = [
    # ---------- Qwen2.5-Coder family (strongest local coding model family) ----------
    ModelEntry(
        id="qwen2.5-coder:32b",
        family="qwen2.5-coder",
        kind="local",
        params_b=32.0,
        context_window=32768,
        ram_q4_gb=20.0,
        min_ram_gb=24.0,
        needs_dedicated_gpu="yes",
        coding_score=9.4,
        agentic_score=8.8,
        speed_score=4.0,
        best_for="Top-tier local coding; rivals GPT-4-class on many tasks.",
        watch_out="Needs ~24GB RAM or a 24GB GPU. Not for old laptops.",
    ),
    ModelEntry(
        id="qwen2.5-coder:14b",
        family="qwen2.5-coder",
        kind="local",
        params_b=14.0,
        context_window=32768,
        ram_q4_gb=10.0,
        min_ram_gb=12.0,
        needs_dedicated_gpu="sometimes",
        coding_score=8.6,
        agentic_score=8.2,
        speed_score=6.0,
        best_for="Sweet spot for 16GB-RAM Apple Silicon and mid-range GPUs.",
        watch_out="Slower than 7B on weak CPUs; ~10GB to download.",
    ),
    ModelEntry(
        id="qwen2.5-coder:7b",
        family="qwen2.5-coder",
        kind="local",
        params_b=7.0,
        context_window=32768,
        ram_q4_gb=5.0,
        min_ram_gb=8.0,
        needs_dedicated_gpu="no",
        coding_score=7.4,
        agentic_score=7.2,
        speed_score=8.0,
        best_for="Best general-purpose pick on a typical 16GB laptop.",
        watch_out="Lower ceiling on hard architectural tasks vs 14B+.",
    ),
    ModelEntry(
        id="qwen2.5-coder:3b",
        family="qwen2.5-coder",
        kind="local",
        params_b=3.0,
        context_window=32768,
        ram_q4_gb=2.5,
        min_ram_gb=4.0,
        needs_dedicated_gpu="no",
        coding_score=6.0,
        agentic_score=6.4,
        speed_score=9.0,
        best_for="Budget coding assistance on 8GB-RAM machines.",
        watch_out="Struggles on large multi-file refactors.",
    ),

    # ---------- Llama 3.x family ----------
    ModelEntry(
        id="llama3.3:70b",
        family="llama3",
        kind="local",
        params_b=70.0,
        context_window=131072,
        ram_q4_gb=42.0,
        min_ram_gb=48.0,
        needs_dedicated_gpu="yes",
        coding_score=8.6,
        agentic_score=8.8,
        speed_score=2.0,
        best_for="Largest open-weight local model; very strong generalist.",
        watch_out="Needs 48GB+ unified memory or a 48GB GPU. Slow on CPU.",
    ),
    ModelEntry(
        id="llama3.1:8b",
        family="llama3",
        kind="local",
        params_b=8.0,
        context_window=131072,
        ram_q4_gb=5.0,
        min_ram_gb=8.0,
        needs_dedicated_gpu="no",
        coding_score=6.8,
        agentic_score=7.4,
        speed_score=8.0,
        best_for="Solid general-purpose assistant; long context (128k).",
        watch_out="Coding is decent but Qwen2.5-Coder is usually better at it.",
    ),
    ModelEntry(
        id="llama3.2:3b",
        family="llama3",
        kind="local",
        params_b=3.0,
        context_window=131072,
        ram_q4_gb=2.5,
        min_ram_gb=4.0,
        needs_dedicated_gpu="no",
        coding_score=5.4,
        agentic_score=6.0,
        speed_score=9.0,
        best_for="Cheapest reasonable local model; great for very low-RAM boxes.",
        watch_out="Loses coherence on hard reasoning. Coding is basic.",
    ),
    ModelEntry(
        id="llama3.2:1b",
        family="llama3",
        kind="local",
        params_b=1.0,
        context_window=131072,
        ram_q4_gb=1.2,
        min_ram_gb=2.0,
        needs_dedicated_gpu="no",
        coding_score=3.8,
        agentic_score=4.4,
        speed_score=10.0,
        best_for="Ultra-budget; will run on basically anything.",
        watch_out="Not really a coding agent. Use only as a last resort.",
    ),

    # ---------- DeepSeek family ----------
    ModelEntry(
        id="deepseek-coder-v2:16b",
        family="deepseek-coder",
        kind="local",
        params_b=16.0,
        context_window=163840,
        ram_q4_gb=10.0,
        min_ram_gb=12.0,
        needs_dedicated_gpu="sometimes",
        coding_score=8.4,
        agentic_score=7.4,
        speed_score=5.5,
        best_for="Strong coding at the 16B size; long context.",
        watch_out="Newer DeepSeek models exist; check Ollama for the latest tag.",
    ),

    # ---------- Phi-3.5 (Microsoft) ----------
    ModelEntry(
        id="phi3.5:3.8b",
        family="phi",
        kind="local",
        params_b=3.8,
        context_window=131072,
        ram_q4_gb=2.7,
        min_ram_gb=4.0,
        needs_dedicated_gpu="no",
        coding_score=6.2,
        agentic_score=6.6,
        speed_score=9.0,
        best_for="Small but surprisingly capable; very long context.",
        watch_out="Coding quality below Qwen2.5-Coder at the same size.",
    ),

    # ---------- Mistral / Gemma ----------
    ModelEntry(
        id="mistral:7b",
        family="mistral",
        kind="local",
        params_b=7.0,
        context_window=32768,
        ram_q4_gb=4.5,
        min_ram_gb=8.0,
        needs_dedicated_gpu="no",
        coding_score=6.4,
        agentic_score=6.6,
        speed_score=8.0,
        best_for="Dependable generalist if you want a non-Llama/non-Qwen pick.",
        watch_out="Aging model line. Mistral's newer Small/Large are stronger but heavier.",
    ),
    ModelEntry(
        id="gemma2:9b",
        family="gemma",
        kind="local",
        params_b=9.0,
        context_window=8192,
        ram_q4_gb=6.0,
        min_ram_gb=8.0,
        needs_dedicated_gpu="no",
        coding_score=6.4,
        agentic_score=6.4,
        speed_score=7.0,
        best_for="Another solid generalist at the 8GB-RAM class.",
        watch_out="Short context (8k) compared to Qwen/Llama peers.",
    ),
    ModelEntry(
        id="codellama:7b",
        family="codellama",
        kind="local",
        params_b=7.0,
        context_window=16384,
        ram_q4_gb=4.5,
        min_ram_gb=8.0,
        needs_dedicated_gpu="no",
        coding_score=6.0,
        agentic_score=5.0,
        speed_score=7.5,
        best_for="Classic code-completion model; still works for simple fills.",
        watch_out="Pre-dates modern agent-style prompting. Newer Qwen2.5-Coder beats it.",
    ),

    # ---------- Qwen3-Coder family (2026 flagship) ----------
    # Qwen3-Coder-30B-A3B is a Mixture-of-Experts model: 30B total params
    # but only ~3B active per token. We still need to load all 30B into
    # memory (the weights are the weights), but the *compute* cost is
    # closer to a 3B dense model. This is the standout "no compromises"
    # pick for a 24GB-class machine — flagship quality on a single box.
    ModelEntry(
        id="qwen3-coder:30b-a3b",
        family="qwen3-coder",
        kind="local",
        params_b=30.0,
        active_params_b=3.0,
        context_window=262144,
        ram_q4_gb=18.0,
        min_ram_gb=22.0,
        needs_dedicated_gpu="sometimes",
        coding_score=9.6,
        agentic_score=9.2,
        speed_score=6.0,
        flagship=True,
        best_for="Flagship-class coding at a single-box memory budget. Best 'no compromises' local pick on a 24GB-class machine.",
        watch_out="Needs ~20GB memory (works on 24GB Apple Silicon or a 24GB GPU). Slower than smaller models on weak CPUs.",
    ),
    # The 480B-A35B is a true server-class MoE. All 480B weights have to
    # be loaded across the combined memory of the machine; realistically
    # only multi-GPU rigs or huge unified-memory Mac Studios can host it.
    ModelEntry(
        id="qwen3-coder:480b-a35b",
        family="qwen3-coder",
        kind="local",
        params_b=480.0,
        active_params_b=35.0,
        context_window=262144,
        ram_q4_gb=290.0,
        min_ram_gb=320.0,
        needs_dedicated_gpu="yes",
        coding_score=9.9,
        agentic_score=9.7,
        speed_score=2.5,
        server_class=True,
        best_for="The strongest open-weight coding model. Server-class only.",
        watch_out="Requires a multi-GPU rig or a Mac Studio / Mac Pro with >200GB unified memory. Not viable on a single consumer GPU.",
    ),
    # DeepSeek V4 Pro — current flagship reasoning model, very strong on
    # code. Server-class footprint.
    ModelEntry(
        id="deepseek-v4-pro",
        family="deepseek",
        kind="local",
        params_b=620.0,
        active_params_b=42.0,
        context_window=200000,
        ram_q4_gb=370.0,
        min_ram_gb=400.0,
        needs_dedicated_gpu="yes",
        coding_score=9.7,
        agentic_score=9.5,
        speed_score=2.0,
        server_class=True,
        best_for="State-of-the-art open-weight reasoning, strong on code.",
        watch_out="Server-class only. Needs a multi-GPU rig or >250GB unified memory.",
    ),
    # GLM-5.2 — Zhipu AI's flagship. Server-class.
    ModelEntry(
        id="glm-5.2",
        family="glm",
        kind="local",
        params_b=520.0,
        active_params_b=32.0,
        context_window=200000,
        ram_q4_gb=310.0,
        min_ram_gb=340.0,
        needs_dedicated_gpu="yes",
        coding_score=9.5,
        agentic_score=9.4,
        speed_score=2.2,
        server_class=True,
        best_for="Strong Chinese-developed open-weight model with great agentic tool use.",
        watch_out="Server-class only. Needs a multi-GPU rig or >220GB unified memory.",
    ),
    # Kimi K2.7 Code — Moonshot's code-specialised flagship. Server-class.
    ModelEntry(
        id="kimi-k2.7-code",
        family="kimi",
        kind="local",
        params_b=480.0,
        active_params_b=28.0,
        context_window=256000,
        ram_q4_gb=290.0,
        min_ram_gb=320.0,
        needs_dedicated_gpu="yes",
        coding_score=9.6,
        agentic_score=9.3,
        speed_score=2.3,
        server_class=True,
        best_for="Code-specialised flagship; long context, strong agentic tool use.",
        watch_out="Server-class only. Needs a multi-GPU rig or >200GB unified memory.",
    ),

    # ---------- Gemma 3 27B (Google) ----------
    # Solid mid-tier open model, but note its license is NOT Apache/MIT —
    # it's a custom Gemma license that has redistribution/use restrictions
    # worth knowing about. We still recommend it because it's good, but
    # we surface the license distinction in the catalog entry.
    ModelEntry(
        id="gemma3:27b",
        family="gemma3",
        kind="local",
        params_b=27.0,
        context_window=131072,
        ram_q4_gb=17.0,
        min_ram_gb=20.0,
        needs_dedicated_gpu="sometimes",
        coding_score=8.4,
        agentic_score=8.2,
        speed_score=5.5,
        license="gemma (custom, not Apache/MIT)",
        best_for="Solid mid-tier open model from Google; long context.",
        watch_out="License is the Gemma custom license, not Apache/MIT — has redistribution/use restrictions. Needs ~20GB to load.",
    ),
]


# ---------------------------------------------------------------------------
# Free cloud catalog
# ---------------------------------------------------------------------------
# We explicitly mark `free_tier` and `signup_required`. Several of these
# change over time; treat the list as a "current best understanding" that
# we should re-check periodically.

CLOUD_MODELS: list[ModelEntry] = [
    ModelEntry(
        id="claude-haiku-4-5-20251001",
        family="claude",
        kind="cloud",
        params_b=None,
        context_window=200000,
        coding_score=8.0,
        agentic_score=8.4,
        speed_score=9.0,
        free_tier=True,
        signup_required=True,
        best_for="Strong coding + tool use, fast, very long context.",
        watch_out="Free tier has rate limits; not always available in every region.",
        notes="Anthropic Claude Haiku 4.5. The realistic free-tier pick in the Claude family.",
    ),
    ModelEntry(
        id="claude-sonnet-5",
        family="claude",
        kind="cloud",
        params_b=None,
        context_window=200000,
        coding_score=9.4,
        agentic_score=9.5,
        speed_score=8.0,
        free_tier=False,
        signup_required=True,
        best_for="Top-tier coding and agentic use; the model OpenCode is tuned against.",
        watch_out="Not on the free tier. Listed so the recommender can mention it as a paid upgrade path.",
        notes="Anthropic Claude Sonnet 5. Stronger than Haiku but paid.",
    ),
    ModelEntry(
        id="claude-opus-4-8",
        family="claude",
        kind="cloud",
        params_b=None,
        context_window=200000,
        coding_score=9.7,
        agentic_score=9.6,
        speed_score=5.5,
        free_tier=False,
        signup_required=True,
        best_for="The strongest Claude model; for the hardest reasoning and long-horizon agentic work.",
        watch_out="Paid, and slow vs Sonnet/Haiku. Overkill for most coding tasks; pick Sonnet 5 for daily use.",
        notes="Anthropic Claude Opus 4.8. Top of the Claude lineup as of 2026.",
    ),
    ModelEntry(
        id="gemini-2.0-flash",
        family="gemini",
        kind="cloud",
        params_b=None,
        context_window=1_000_000,
        coding_score=7.6,
        agentic_score=7.8,
        speed_score=9.0,
        free_tier=True,
        signup_required=True,
        best_for="Huge context window; good for very large codebases.",
        watch_out="Free-tier rate limits are strict; tool-use quality trails Claude.",
        notes="Google AI Studio free tier.",
    ),
    ModelEntry(
        id="mistral-large-latest",
        family="mistral",
        kind="cloud",
        params_b=None,
        context_window=128000,
        coding_score=7.4,
        agentic_score=7.4,
        speed_score=7.5,
        free_tier=True,
        signup_required=True,
        best_for="Decent generalist with a free tier via le Chat / La Plateforme.",
        watch_out="Free tier exists but is rate-limited; coding a notch below Claude/Gemini.",
    ),
    ModelEntry(
        id="openrouter-free-mix",
        family="openrouter",
        kind="cloud",
        params_b=None,
        context_window=32000,
        coding_score=6.0,
        agentic_score=6.0,
        speed_score=7.0,
        free_tier=True,
        signup_required=True,
        best_for="A grab-bag of free models behind one API key (Llama, Qwen, Mistral variants).",
        watch_out="Quality varies; the exact free model rotates.",
        notes="OpenRouter exposes a rotating set of free models. Use only as a fallback.",
    ),
]


# ---------------------------------------------------------------------------
# Catalog access
# ---------------------------------------------------------------------------

class ModelCatalog:
    """Read-only view over the model lists. Lets the recommender iterate
    and look up by id without caring about the underlying constants.

    `local` / `cloud` may be passed explicitly (e.g. from a locally cached,
    live-refreshed catalog). When omitted they default to the hardcoded
    baseline so the advisor always has something to fall back to."""

    def __init__(
        self,
        local: Optional[list[ModelEntry]] = None,
        cloud: Optional[list[ModelEntry]] = None,
    ) -> None:
        self.local: list[ModelEntry] = list(local) if local is not None else list(LOCAL_MODELS)
        self.cloud: list[ModelEntry] = list(cloud) if cloud is not None else list(CLOUD_MODELS)

    def all(self) -> list[ModelEntry]:
        return self.local + self.cloud

    def find(self, model_id: str) -> Optional[ModelEntry]:
        for m in self.all():
            if m.id == model_id:
                return m
        return None

    def local_sorted_by_coding(self) -> list[ModelEntry]:
        return sorted(self.local, key=lambda m: m.coding_score, reverse=True)

    def to_dict(self) -> dict:
        """Serialise for the local cache: {"local": [...], "cloud": [...]}."""
        return {
            "local": [m.to_dict() for m in self.local],
            "cloud": [m.to_dict() for m in self.cloud],
        }

    @classmethod
    def from_dict(cls, d: dict) -> "ModelCatalog":
        """Rebuild a catalog from `to_dict` output. Missing/non-list sides
        default to the hardcoded baseline so a malformed cache degrades
        safely."""
        raw_local = d.get("local")
        raw_cloud = d.get("cloud")
        local = [ModelEntry.from_dict(e) for e in raw_local] if isinstance(raw_local, list) else None
        cloud = [ModelEntry.from_dict(e) for e in raw_cloud] if isinstance(raw_cloud, list) else None
        return cls(local=local, cloud=cloud)
