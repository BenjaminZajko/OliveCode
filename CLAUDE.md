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

## Architecture (decided)

The product's core promise is *100% free, unlimited, and zero-setup* for
the end user. That shapes how every optional feature has to be
architected: never require the regular user to have an account, an API
key, or to pay.

1. **Default mode (everyone, zero setup).** The app auto-downloads and
   configures the best local model for the user's hardware (via Ollama)
   on first run. No account, no key, no limits, works offline forever.
   This is the only mode most users will ever need.

2. **Optional "Connect" mode (opt-in, for cloud-class quality).** A
   one-click OAuth PKCE flow against OpenRouter: opens the browser, the
   user logs in or signs up in ~1–2 clicks, approves, redirects back with
   an authorization code, the app exchanges it for the user's OWN
   OpenRouter API key, and stores it locally. This is the user's own
   account/quota — not a shared key we control — so it avoids provider
   ToS problems and avoids us running/paying for a shared backend. Live
   web search and a small task-based model-switching logic (cheap/fast
   model for simple tasks, stronger model for complex ones) will live in
   this mode only, using the user's own OpenRouter key. **Not built
   yet.**

3. **Catalog freshness.** The model catalog cannot be refreshed by LLM
   + web search on each end user's install (that would require every
   user to have an API key). Instead: the developer periodically
   refreshes the catalog themselves — locally, with their own key, as
   often as they want — and publishes the result to a fixed public URL
   (a raw `catalog.json` in the repo, once it's pushed to GitHub). Each
   user's install does a plain HTTPS GET of that URL to check for a
   newer catalog, with a local cache (≥7 days) and silent offline
   fallback. **No LLM call on the user's machine for this path.**

4. **App updates.** The app itself (not just the catalog) will later get
   a simple auto-update mechanism via GitHub Releases. **Not built yet.**

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
- Pull a curated catalog from the developer's published catalog.json
  (plain HTTPS GET, no LLM). Fall back silently to a local cache (≥7
  days), then to the hardcoded baseline if there's no cache yet. Never
  crash, never block, never require an API key on the user's machine.
- Optionally call out to an LLM to write a richer narrative explanation
  of the recommendation, when `ANTHROPIC_API_KEY` is set. The LLM is
  the narrator, never the source of truth: the local scoring logic
  always picks the models first. **Regular end users will never have
  `ANTHROPIC_API_KEY` set** — this enhancement is for developer
  convenience only, and the path must keep working fully without it.

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
- **No user-side API key dependencies.** Per the architecture above,
  every code path that runs on a regular end user's machine must work
  with no `ANTHROPIC_API_KEY` (or any other key) set. The LLM-narrated
  explanation is an optional developer-only enhancement and must
  degrade gracefully when no key is present.

## Running the advisor

From the project root:

```
python3 -m advisor                  # interactive
python3 -m advisor --no-llm         # skip the optional LLM explanation
python3 -m advisor --json           # machine-readable
python3 -m advisor --hardware-only  # just dump detected hardware
python3 -m advisor --refresh-catalog # force a catalog refresh from the public URL
```

The model catalog is published by the developer as a static
`catalog.json` at a fixed URL (see `advisor/catalog_refresh.py` for the
current URL). Each user's install fetches that on startup if the local
cache is missing or older than 7 days, then caches to
`~/.cache/olivecode/catalog.json`. Refreshes only ever add or update on
top of the hardcoded baseline; with no network it silently uses the
last good cache, or the built-in defaults if there's no cache yet.
**No API key is required for this path.**

The LLM-narrated explanation (still used by the reasoning module for
now) is an optional developer-only enhancement. It activates when
`ANTHROPIC_API_KEY` is set in the environment; without it, the
explanation falls back to a local deterministic message.

## Wiring into OpenCode (later)

The advisor is *not* yet wired into the forked OpenCode. That work
should only start after this module is reviewed and the model catalog
is stable. The integration shape is deliberately not committed to yet
— don't scaffold the integration speculatively.
