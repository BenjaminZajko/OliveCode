"""Entry point for the OliveCode hardware & model advisor.

Run as:
  python3 -m advisor                    # interactive
  python3 -m advisor --json             # machine-readable
  python3 -m advisor --no-llm           # skip the optional LLM explanation
  python3 -m advisor --hardware-only    # just dump detected hardware and exit
  python3 -m advisor --refresh-catalog  # force a live catalog refresh
  python3 -m advisor --install          # detect + recommend + install recommended local model

The module is designed so this file does as little work as possible: it
wires together detection, the catalog (+ live refresh/caching), the
reasoning engine, the installer, and the CLI.
"""

from __future__ import annotations

import argparse
import sys

from . import cli
from .hardware import detect_hardware
from .catalog_refresh import load_catalog
from .reasoning import recommend


def _build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(
        prog="advisor",
        description=(
            "OliveCode hardware & model advisor. Detects your machine and "
            "recommends the best free local (Ollama) and free cloud model "
            "for a coding agent."
        ),
    )
    p.add_argument(
        "--json",
        action="store_true",
        help="Print a single machine-readable JSON document and exit.",
    )
    p.add_argument(
        "--no-llm",
        action="store_true",
        help="Skip the optional LLM-narrated explanation; use the local message only.",
    )
    p.add_argument(
        "--hardware-only",
        action="store_true",
        help="Only print detected hardware, then exit.",
    )
    p.add_argument(
        "--no-interactive",
        action="store_true",
        help="Don't prompt for the 'show other options' follow-up.",
    )
    p.add_argument(
        "--refresh-catalog",
        action="store_true",
        help="Force a live catalog refresh from the web (skips the 7-day staleness check).",
    )
    p.add_argument(
        "--install",
        action="store_true",
        help=(
            "Show the recommendation, then immediately download the "
            "recommended local model via Ollama (no extra prompt)."
        ),
    )
    p.add_argument(
        "--install-model",
        type=str,
        default=None,
        metavar="MODEL_ID",
        help=(
            "Install a specific Ollama model (e.g. 'llama3.2:1b') instead "
            "of the recommended one. Useful for testing the download path "
            "with a small model. Implies --install."
        ),
    )
    return p


def _catalog_status_line(status: str, refreshed_at: object | None) -> str:
    """One short, non-intrusive line telling the user where the model
    catalog came from."""
    if status == "fresh":
        return "[catalog] refreshed live just now."
    if status == "cached":
        return "[catalog] using recent local cache (no refresh needed)."
    if status == "cached-stale":
        return (
            "[catalog] local cache older than 7 days and a live refresh "
            "wasn't possible; using the cached catalog anyway."
        )
    return "[catalog] offline defaults (no cache; refresh unavailable)."


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # Load the catalog: this may refresh from the web, but NEVER blocks or
    # crashes — it falls back to cache, then hardcoded defaults.
    catalog, status, refreshed_at = load_catalog(force_refresh=args.refresh_catalog)

    if args.json:
        # Machine-readable: bake the catalog status in instead of printing a
        # separate (noisy) line. We always skip the LLM explanation for
        # determinism.
        rec = recommend(profile=detect_hardware(), catalog=catalog, use_llm=False)
        cli.print_json(rec, catalog_status=status,
                       catalog_refreshed_at=refreshed_at.isoformat() if refreshed_at else None)
        return 0

    print(_catalog_status_line(status, refreshed_at))

    # 1. Detect hardware. By design, this cannot raise.
    profile = detect_hardware()

    if args.hardware_only:
        cli.print_hardware(profile)
        return 0

    # 2. Get a recommendation. LLM narration is opt-in via env var; we
    #    turn it off automatically when not in interactive mode and when
    #    --install is set (the install flag is the "make it so" signal).
    use_llm = (
        (not args.no_llm)
        and (not args.no_interactive)
        and (not args.install)
        and sys.stdin.isatty()
    )
    rec = recommend(profile, catalog=catalog, use_llm=use_llm)

    # 3. Install path. We always print the recommendation first so the
    # user sees what we're about to download, then immediately proceed.
    if args.install or args.install_model:
        cli.print_hardware(profile)
        cli._print_recommendation(rec)
        if args.install_model:
            cli.do_install(args.install_model, profile.os_name)
        elif rec.primary_local is not None:
            cli.do_install(rec.primary_local.model.id, profile.os_name)
        else:
            print("\n[install] No local model in the recommendation — nothing to install.")
        return 0

    # 4. Non-interactive: show everything, then exit.
    if args.no_interactive:
        cli.print_hardware(profile)
        cli._print_recommendation(rec)
        cli._print_alternates(rec)
        return 0

    # 5. Interactive flow (run_interactive handles the install prompt itself).
    return cli.run_interactive(rec)


if __name__ == "__main__":
    raise SystemExit(main())
