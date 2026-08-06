"""Entry point for the OliveCode hardware & model advisor.

Run as:
  python3 -m advisor                    # interactive
  python3 -m advisor --json             # machine-readable
  python3 -m advisor --no-llm           # skip the optional LLM explanation
  python3 -m advisor --hardware-only    # just dump detected hardware and exit

The module is designed so this file does as little work as possible: it
wires together detection, the catalog, the reasoning engine, and the CLI.
"""

from __future__ import annotations

import argparse
import json
import sys

from . import cli
from .hardware import detect_hardware
from .model_catalog import ModelCatalog
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
        help="Skip the optional LLM-narrated explanation; use the local template only.",
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
    return p


def main(argv: list[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)

    # 1. Detect hardware. By design, this cannot raise.
    profile = detect_hardware()

    if args.hardware_only:
        cli.print_hardware(profile)
        return 0

    if args.json:
        # In --json mode we always skip the LLM call to keep output deterministic.
        rec = recommend(profile, catalog=ModelCatalog(), use_llm=False)
        cli.print_json(rec)
        return 0

    # 2. Get a recommendation. LLM narration is opt-in via env var; we
    #    turn it off automatically when not in interactive mode.
    use_llm = (not args.no_llm) and (not args.no_interactive) and sys.stdin.isatty()
    rec = recommend(profile, catalog=ModelCatalog(), use_llm=use_llm)

    # 3. Show it.
    if args.no_interactive:
        cli.print_hardware(profile)
        cli._print_recommendation(rec)
        cli._print_alternates(rec)
        return 0

    return cli.run_interactive(rec)


if __name__ == "__main__":
    raise SystemExit(main())
