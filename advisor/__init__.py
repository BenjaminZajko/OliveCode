"""OliveCode hardware & model advisor.

A modular advisor that:
  1. Detects the user's hardware (OS, CPU, RAM, GPU, disk) in a way that
     never crashes on unusual or constrained machines.
  2. Uses a curated knowledge base of local (Ollama) and free cloud models
     to reason about which model is the best fit for the detected hardware
     when used as a coding agent.
  3. Recommends a primary local pick and a primary free cloud pick, with
     a ranked list of viable alternatives on demand.

The package is intentionally split into:
  - hardware:          detection
  - model_catalog:     the knowledge base
  - reasoning:         scoring + recommendation logic (+ optional LLM narration)
  - catalog_refresh:   live catalog refresh with local caching
  - llm:               shared Anthropic API wrapper
  - installer:         the only module allowed to shell out (Ollama pulls)
  - cli:               the user-facing interactive flow
  - main:              the entry point
"""

from .hardware import detect_hardware, HardwareProfile
from .model_catalog import ModelCatalog, ModelEntry
from .reasoning import recommend, Recommendation, RankedModel
from . import cli, main, catalog_refresh, installer

__all__ = [
    "detect_hardware",
    "HardwareProfile",
    "ModelCatalog",
    "ModelEntry",
    "recommend",
    "Recommendation",
    "RankedModel",
    "cli",
    "main",
    "installer",
]
