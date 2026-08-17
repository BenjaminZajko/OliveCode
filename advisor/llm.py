"""Shared wrapper around the Anthropic Messages API.

Both the narrative explanation and the live catalog refresh need to talk
to Claude over HTTP. This module keeps that call — and its failure
handling — in one place. Every function here NEVER raises and returns a
safe empty value on any failure, so callers can fall back to their
deterministic path without thinking about it.

The endpoint can be overridden with the OLIVECODE_API_BASE env var. That
is mostly for tests (point it at a dead port to exercise the offline
path) and for anyone who wants to proxy the request.
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from typing import Optional

DEFAULT_MODEL = "claude-haiku-4-5-20251001"
API_BASE = "https://api.anthropic.com/v1/messages"

# Native Anthropic web-search tool. The API runs the search server-side and
# returns the results in the same response; we never have to execute it.
WEB_SEARCH_TOOL = {
    "type": "web_search_20250305",
    "name": "web_search",
    "max_uses": 5,
}


def api_key() -> Optional[str]:
    """The configured Anthropic key, or None (never raises)."""
    try:
        return os.environ.get("ANTHROPIC_API_KEY") or None
    except Exception:
        return None


def _endpoint() -> str:
    try:
        return os.environ.get("OLIVECODE_API_BASE") or API_BASE
    except Exception:
        return API_BASE


def _post(body: dict, timeout: float) -> Optional[dict]:
    """One raw POST to the Messages API. Returns the parsed JSON response,
    or None on any failure. Never raises."""
    key = api_key()
    if not key:
        return None
    req = urllib.request.Request(
        _endpoint(),
        data=json.dumps(body).encode("utf-8"),
        headers={
            "Content-Type": "application/json",
            "x-api-key": key,
            "anthropic-version": "2023-06-01",
        },
        method="POST",
    )
    try:
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            return json.loads(resp.read().decode("utf-8"))
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        json.JSONDecodeError,
        OSError,
    ):
        return None
    except Exception:
        return None


def _extract_text(content: list) -> str:
    try:
        return "\n".join(
            b.get("text", "") for b in content if b.get("type") == "text"
        ).strip()
    except Exception:
        return ""


def anthropic_completion(
    system: str,
    user: str,
    model: str = DEFAULT_MODEL,
    max_tokens: int = 800,
    tools: Optional[list[dict]] = None,
    timeout: float = 40.0,
    max_rounds: int = 4,
) -> str:
    """Run one LLM turn and return the final text content.

    When `tools` includes the server-side web search tool, the API runs the
    searches for us. A response can come back with `stop_reason:
    "pause_turn"`, which means we must echo the assistant message back
    unchanged and let the API continue — we loop on that until the model
    finishes or we hit `max_rounds`.

    Returns "" on any failure. Never raises.
    """
    if not api_key():
        return ""

    messages: list[dict] = [{"role": "user", "content": user}]
    last_text = ""
    for _ in range(max_rounds):
        body: dict = {
            "model": model,
            "max_tokens": max_tokens,
            "system": system,
            "messages": messages,
        }
        if tools:
            body["tools"] = tools
        data = _post(body, timeout)
        if data is None:
            return last_text

        content = data.get("content", [])
        if isinstance(content, list):
            last_text = _extract_text(content)

        stop = data.get("stop_reason")
        if stop != "pause_turn":
            return last_text
        # Server-side web search needs more time: hand the assistant's
        # partial message back and let the API continue the search loop.
        messages.append({"role": "assistant", "content": content})

    return last_text


__all__ = [
    "DEFAULT_MODEL",
    "WEB_SEARCH_TOOL",
    "api_key",
    "anthropic_completion",
]
