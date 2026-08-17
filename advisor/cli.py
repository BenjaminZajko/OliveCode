"""Interactive CLI for the advisor.

Flow:
  1. Print what we detected about the user's machine.
  2. Print the two-pick recommendation: one local (Ollama), one free cloud.
  3. Always offer a "show me other options" path that lists alternates,
     each with a short reason.
  4. If stdin isn't a TTY (e.g. piped, or `--json` was passed), just
     print everything once and exit; no interactive prompt.

Stdout-only. All prompt logic uses `input()` with safe fallbacks.
"""

from __future__ import annotations

import json
import os
import sys
from typing import Optional

from .hardware import HardwareProfile
from .reasoning import RankedModel, Recommendation, _effective_memory_gb


# ---------------------------------------------------------------------------
# Pretty printing
# ---------------------------------------------------------------------------

def _fmt_gb(v: Optional[float]) -> str:
    if v is None:
        return "unknown"
    return f"{v:.0f} GB"


def print_hardware(profile: HardwareProfile) -> None:
    print("Detected hardware")
    print("-----------------")
    rows: list[tuple[str, str]] = []
    if profile.os_name:
        os_str = profile.os_name
        if profile.os_version:
            os_str += f" {profile.os_version}"
        if profile.arch:
            os_str += f" ({profile.arch})"
        rows.append(("OS", os_str))
    else:
        rows.append(("OS", "unknown"))

    if profile.cpu_model:
        cpu_str = profile.cpu_model
        if profile.cpu_cores_logical:
            cpu_str += f" ({profile.cpu_cores_logical} threads"
            if profile.cpu_cores_physical:
                cpu_str += f", {profile.cpu_cores_physical} physical"
            cpu_str += ")"
        rows.append(("CPU", cpu_str))
    elif profile.cpu_cores_logical:
        rows.append(("CPU", f"{profile.cpu_cores_logical} threads"))
    else:
        rows.append(("CPU", "unknown"))

    rows.append(("RAM", _fmt_gb(profile.ram_total_gb)))

    if profile.gpus:
        for i, g in enumerate(profile.gpus):
            label = "GPU" if len(profile.gpus) == 1 else f"GPU {i+1}"
            vram = f" · {_fmt_gb(g.vram_gb)} VRAM" if g.vram_gb else ""
            drv = f" · driver {g.driver}" if g.driver else ""
            note = f" · {g.notes}" if g.notes else ""
            rows.append((label, f"{g.vendor}: {g.name}{vram}{drv}{note}"))
    else:
        rows.append(("GPU", "none detected"))

    if profile.disk_free_gb is not None:
        rows.append(("Disk free", _fmt_gb(profile.disk_free_gb)))
    if profile.disk_total_gb is not None:
        rows.append(("Disk total", _fmt_gb(profile.disk_total_gb)))

    pool = _effective_memory_gb(profile)
    rows.append(("Usable for models", f"~{pool:.0f} GB"))

    width = max(len(k) for k, _ in rows)
    for k, v in rows:
        print(f"  {k.ljust(width)} : {v}")

    if profile.notes:
        print()
        print("Notes from detection:")
        for n in profile.notes:
            print(f"  - {n}")


def _print_pick(label: str, r: RankedModel) -> None:
    m = r.model
    print(f"\n  {label}: {m.id}")
    if m.kind == "local":
        bits: list[str] = []
        if m.params_b:
            bits.append(f"{m.params_b:g}B params")
        if m.ram_q4_gb:
            bits.append(f"~{m.ram_q4_gb:.0f} GB RAM at {m.default_quant}")
        if m.context_window:
            bits.append(f"{m.context_window // 1000}k context")
        if bits:
            print(f"    ({', '.join(bits)})")
    else:
        if m.context_window:
            print(f"    (context {m.context_window // 1000}k, free={m.free_tier})")
    if m.best_for:
        print(f"    Why: {m.best_for}")
    for reason in r.reasons[:3]:
        print(f"    - {reason}")


def _print_recommendation(rec: Recommendation) -> None:
    print()
    print("Our recommendation for your machine")
    print("-----------------------------------")
    if rec.primary_local:
        _print_pick("Local (Ollama)", rec.primary_local)
    if rec.primary_cloud:
        _print_pick("Free cloud", rec.primary_cloud)

    print()
    print("Why these?")
    print("----------")
    print(rec.explanation)


def _print_alternates(rec: Recommendation) -> None:
    if not rec.alternates:
        print("\nNo other viable models for this machine.")
        return
    print()
    print("Other viable options, ranked best-to-worst for your machine")
    print("-----------------------------------------------------------")
    for i, r in enumerate(rec.alternates, 1):
        m = r.model
        kind = "local" if m.kind == "local" else "cloud"
        fit = r.fit if m.kind == "local" else "cloud"
        score = f"{r.score:.1f}"
        if m.kind == "local":
            head = f"{i:>2}. [{kind}/{fit}] {m.id}  (score {score})"
        else:
            head = f"{i:>2}. [{kind}] {m.id}  (score {score})"
        print(head)
        if m.best_for:
            print(f"     Best for: {m.best_for}")
        if r.reasons:
            # one short reason
            print(f"     Why here: {r.reasons[0]}")


# ---------------------------------------------------------------------------
# Interactive flow
# ---------------------------------------------------------------------------

def _safe_input(prompt: str) -> str:
    try:
        return input(prompt)
    except (EOFError, KeyboardInterrupt):
        print()
        return ""


def run_interactive(rec: Recommendation) -> int:
    print_hardware(rec.profile)
    _print_recommendation(rec)

    if not sys.stdin.isatty() or not sys.stdout.isatty():
        # Non-interactive: dump alternates too and exit.
        _print_alternates(rec)
        return 0

    print()
    # Offer to install the recommended local model. This is the "yes,
    # set it up for me" moment that turns the advisor into a real
    # first-run wizard.
    if rec.primary_local is not None:
        if _ask_yes_no(
            f"Install {rec.primary_local.model.id} now? [Y/n] ",
            default_yes=True,
        ):
            do_install(rec.primary_local.model.id, rec.profile.os_name)

    while True:
        ans = _safe_input(
            "Show other viable options? [y/N] (or 'q' to quit): "
        ).strip().lower()
        if ans in ("q", "quit", "exit"):
            print("Bye.")
            return 0
        if ans in ("y", "yes"):
            _print_alternates(rec)
            print()
            continue
        # default = no
        return 0


# ---------------------------------------------------------------------------
# Install-prompt helpers
# ---------------------------------------------------------------------------

def _ask_yes_no(prompt: str, default_yes: bool = True) -> bool:
    """Prompt until the user gives a recognisable yes/no answer, or until
    stdin closes. Returns `default_yes` on EOF / empty line."""
    suffix = "[Y/n]" if default_yes else "[y/N]"
    while True:
        ans = _safe_input(f"{prompt}{suffix} ").strip().lower()
        if ans == "":
            return default_yes
        if ans in ("y", "yes"):
            return True
        if ans in ("n", "no"):
            return False
        # Anything else: re-prompt so we don't accidentally act on a typo.
        print("Please answer y or n.")


def do_install(model_id: str, os_name: Optional[str]) -> None:
    """Run the install for `model_id`, print the result, and return."""
    # Import inside the function so the cli module stays subprocess-free
    # at import time. installer is the only module allowed to shell out.
    from .installer import install_model, ollama_install_instructions

    print()
    print(f"Installing {model_id}")
    print("-" * (len("Installing ") + len(model_id)))

    result = install_model(model_id)
    print()
    if result.ok:
        print(f"[install] {result.message}")
    elif result.kind == "ollama_missing":
        print("[install] Ollama isn't installed on this machine yet.")
        print()
        print(ollama_install_instructions(os_name))
    elif result.kind == "daemon_not_running":
        print(f"[install] {result.message}")
        if result.detail:
            print()
            print(f"  (Ollama said: {result.detail})")
    elif result.kind == "pull_cancelled":
        print(f"[install] {result.message}")
        print("  Run `python3 -m advisor --install` again to resume.")
    elif result.kind == "failed":
        print(f"[install] {result.message}")
        if result.detail:
            print()
            print(f"  (Ollama said: {result.detail})")
    else:
        print(f"[install] {result.message}")
    print()


def print_json(
    rec: Recommendation,
    catalog_status: Optional[str] = None,
    catalog_refreshed_at: Optional[str] = None,
) -> None:
    """Machine-readable output for the --json flag."""
    out = {
        "hardware": rec.profile.to_dict(),
        "catalog_status": catalog_status,
        "catalog_refreshed_at": catalog_refreshed_at,
        "effective_memory_gb": _effective_memory_gb(rec.profile),
        "primary_local": _serialise_ranked(rec.primary_local),
        "primary_cloud": _serialise_ranked(rec.primary_cloud),
        "alternates": [_serialise_ranked(r) for r in rec.alternates],
        "explanation": rec.explanation,
    }
    print(json.dumps(out, indent=2, default=str))


def _serialise_ranked(r: Optional[RankedModel]) -> Optional[dict]:
    if r is None:
        return None
    m = r.model
    return {
        "id": m.id,
        "kind": m.kind,
        "family": m.family,
        "params_b": m.params_b,
        "context_window": m.context_window,
        "ram_q4_gb": m.ram_q4_gb,
        "needs_dedicated_gpu": m.needs_dedicated_gpu,
        "free_tier": m.free_tier,
        "coding_score": m.coding_score,
        "agentic_score": m.agentic_score,
        "speed_score": m.speed_score,
        "best_for": m.best_for,
        "watch_out": m.watch_out,
        "score": round(r.score, 2),
        "fit": r.fit,
        "reasons": r.reasons,
    }


__all__ = ["run_interactive", "print_json", "print_hardware", "do_install"]
