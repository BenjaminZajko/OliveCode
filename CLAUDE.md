# OliveCode

OliveCode is a fork of [OpenCode](https://github.com/sst/opencode), the
open-source Claude Code alternative. OpenCode is great but expects the
user to already know which model they want, sign up for the right
provider, pay for API access, and tune configuration files by hand.
OliveCode's purpose is to make that entire setup step disappear.

## What OliveCode is

The core value proposition is **convenience**:

- A one-click, zero-hassle first-run experience.
- The app detects the user's hardware automatically and recommends the
  best *free* model for that machine — either a local model runnable via
  Ollama, or a free cloud option — so the user doesn't have to research
  quantizations, VRAM requirements, or free-tier availability themselves.
- The recommended model is configured automatically; the user doesn't
  edit a config file on day one.

A secondary, planned feature is an **ad / revenue-share** model: ads
displayed in the app that pay the user a share of the revenue. This is a
bonus, not the core value. Don't optimise the codebase for the
ad-system at the expense of the setup experience.

A possible future bonus feature is **automatic model switching** based
on the task (e.g. use a small fast local model for completions, a
larger cloud model for hard architectural questions). This is not built
yet and shouldn't be added speculatively.

## Current stage: the advisor module (Step 1)

The very first building block is a standalone **hardware & model
advisor** that lives in the `advisor/` subfolder. It is being built and
tested *before* being wired into the forked OpenCode itself, so that
the recommendation logic and the model catalog can mature independently
of the rest of the app.

When the advisor is wired in, it will become a first-run setup wizard:
"Here's what we recommend for your machine — no need to choose anything
yourself."

The advisor itself must:

- Detect the user's hardware thoroughly (OS, CPU, RAM, GPU, disk) and
  **never crash**, no matter how unusual the host is. Every detection
  step is wrapped so a failure on one field never takes down the rest.
- Reason about that hardware against a curated knowledge base of local
  (Ollama) and free cloud models, weighing real trade-offs (fit on
  available memory, GPU vs. CPU, coding quality, context window, etc.)
  rather than hardcoded `if/else` thresholds.
- Output exactly two defaults — one best free local model, one best
  free cloud model — plus an "I don't want either of these, show me
  other options" path that lists alternates ranked for the specific
  machine.
- Optionally call out to an LLM to write a richer narrative explanation
  of the recommendation, when `ANTHROPIC_API_KEY` is set. The LLM is
  the narrator, never the source of truth: the local scoring logic
  always picks the models first.

## Code conventions

- **Multiple well-separated files, not monolithic scripts.** Each module
  should have a single clear concern.
- **Robust error handling everywhere.** Anything that touches the
  outside world (subprocess, filesystem, network, hardware probes) must
  be wrapped so a failure on one input doesn't crash the program.
  "Could not detect X" is always a valid answer; throwing is not.
- **English throughout** — code, comments, user-facing strings.
- **No silent fallbacks that change the recommendation's meaning.** If
  detection produced nothing useful, say so explicitly; the user should
  be able to see what was known and what wasn't.

## Running the advisor

From the project root:

```
python3 -m advisor                 # interactive
python3 -m advisor --no-llm        # skip the optional LLM explanation
python3 -m advisor --json          # machine-readable
python3 -m advisor --hardware-only # just print detected hardware
```

To enable the LLM-narrated explanation, set `ANTHROPIC_API_KEY` in the
environment before running.

## Wiring into OpenCode (later)

The advisor is *not* yet wired into the forked OpenCode. That work
should only start after this module is reviewed and the model catalog
is stable. The integration shape is deliberately not committed to yet
— don't scaffold the integration speculatively.
