"""Live catalog refresh with local caching.

The hardcoded baseline in `model_catalog` goes stale as the model landscape
moves. This module lets the advisor periodically pull a curated, develop-
maintained catalog from a fixed public URL, then cache the merged result
locally so we don't hit the network on every run.

Important architectural note: refresh is a plain HTTPS GET of a static
catalog.json. There is NO LLM call on the user's machine and no API key
required — the catalog is curated by the project developer (with their own
key, offline, whenever they want) and published to a fixed URL. Each end
user's install just fetches that JSON.

Contract:
  - `load_catalog()` NEVER raises and never waits more than the HTTP
    timeout. It always returns a usable catalog plus a status string, so
    hardware detection and recommendation keep working fully offline.
  - A refresh only ever ADDS or UPDATES entries on top of the hardcoded
    baseline — it never removes or replaces it wholesale, so a partial or
    failed refresh can't leave the catalog empty or broken.
  - With no internet, DNS failure, HTTP error, timeout, malformed JSON,
    or any other failure we fall back to the last good cache, or to the
    hardcoded defaults if no cache exists yet.

Status strings returned by `load_catalog`:
  - "fresh"            just successfully fetched the static catalog.json
  - "cached"           using a local cache that is still younger than 7 days
  - "cached-stale"     using a cache older than 7 days (refresh unavailable)
  - "offline-default"  no cache at all; using the hardcoded baseline
"""

from __future__ import annotations

import json
import os
import urllib.error
import urllib.request
from datetime import datetime, timedelta, timezone
from pathlib import Path
from typing import Optional

from .model_catalog import ModelCatalog, ModelEntry, LOCAL_MODELS, CLOUD_MODELS


# ---------------------------------------------------------------------------
# Public constants
# ---------------------------------------------------------------------------

# The static catalog file published by the developer. Once the repo is on
# GitHub this should point at the real raw URL, e.g.:
#   https://raw.githubusercontent.com/<owner>/OliveCode/main/advisor/catalog.json
# Until then, the placeholder below will fail to resolve; the refresh call
# will silently fall back to the cache / hardcoded baseline, which is the
# intended behaviour.
CATALOG_URL = (
    "https://raw.githubusercontent.com/<owner>/OliveCode/main/advisor/catalog.json"
)

CACHE_ENV = "OLIVECODE_CACHE_DIR"
DEFAULT_CACHE_DIR = "~/.cache/olivecode"
MAX_AGE_DAYS = 7
CACHE_VERSION = 1
HTTP_TIMEOUT_SECONDS = 8.0


# ---------------------------------------------------------------------------
# Paths + basic cache I/O
# ---------------------------------------------------------------------------

def catalog_cache_path() -> Path:
    """Filesystem location of the catalog cache: ~/.cache/olivecode/catalog.json
    (overridable via OLIVECODE_CACHE_DIR)."""
    try:
        base = os.environ.get(CACHE_ENV) or DEFAULT_CACHE_DIR
        return Path(base).expanduser() / "catalog.json"
    except Exception:
        return Path(DEFAULT_CACHE_DIR).expanduser() / "catalog.json"


def read_cached(path: Optional[Path] = None) -> tuple[Optional[ModelCatalog], Optional[datetime]]:
    """Read the cache. Returns (catalog, refreshed_at); both are None if the
    cache is missing, malformed, or the wrong version. Never raises."""
    path = path or catalog_cache_path()
    try:
        if not path.exists():
            return None, None
        data = json.loads(path.read_text(encoding="utf-8", errors="replace"))
        if data.get("version") != CACHE_VERSION:
            return None, None
        cat = ModelCatalog.from_dict(data.get("catalog") or {})
        refreshed = None
        try:
            refreshed = datetime.fromisoformat(data.get("refreshed_at", "")).astimezone(timezone.utc)
        except Exception:
            refreshed = None
        return cat, refreshed
    except Exception:
        return None, None


def write_cache(path: Optional[Path], catalog: ModelCatalog) -> bool:
    """Persist the merged catalog with a refresh timestamp. Returns False on
    any error (disk full, permission, etc.) — the caller treats a cache we
    can't write as a non-fatal, non-blocking condition. Never raises."""
    path = path or catalog_cache_path()
    try:
        path.parent.mkdir(parents=True, exist_ok=True)
        payload = {
            "version": CACHE_VERSION,
            "refreshed_at": datetime.now(timezone.utc).isoformat(),
            "catalog": catalog.to_dict(),
        }
        path.write_text(
            json.dumps(payload, indent=2, default=str), encoding="utf-8"
        )
        return True
    except Exception:
        return False


def is_stale(refreshed_at: Optional[datetime], max_age_days: int = MAX_AGE_DAYS) -> bool:
    """True if the refresh timestamp is absent or older than `max_age_days`."""
    if refreshed_at is None:
        return True
    try:
        now = datetime.now(timezone.utc)
        cutoff = now - timedelta(days=max_age_days)
        return refreshed_at < cutoff
    except Exception:
        return True


def _baseline() -> ModelCatalog:
    return ModelCatalog(local=list(LOCAL_MODELS), cloud=list(CLOUD_MODELS))


# ---------------------------------------------------------------------------
# Merge (add / update only — never remove)
# ---------------------------------------------------------------------------

def merge(*catalogs: Optional[ModelCatalog]) -> ModelCatalog:
    """Fuse several catalogs into one by model id. Later catalogs override
    earlier ones for a given id; the union of all ids is kept. This can only
    ADD or UPDATE — it can never remove an entry that already exists."""
    local: dict[str, ModelEntry] = {}
    cloud: dict[str, ModelEntry] = {}
    for cat in catalogs:
        if cat is None:
            continue
        for m in cat.local:
            if m.kind == "local":
                local[m.id] = m
        for m in cat.cloud:
            if m.kind == "cloud":
                cloud[m.id] = m
    return ModelCatalog(local=list(local.values()), cloud=list(cloud.values()))


# ---------------------------------------------------------------------------
# Static-URL refresh
# ---------------------------------------------------------------------------

def _fetch_catalog_json(url: str = CATALOG_URL, timeout: float = HTTP_TIMEOUT_SECONDS) -> Optional[ModelCatalog]:
    """Plain HTTPS GET of the developer's published catalog.json. Returns
    a ModelCatalog on success, or None on any failure (DNS, timeout, bad
    URL, malformed JSON, unexpected schema, etc.). Never raises.

    The on-the-wire format is exactly the same JSON shape that
    `ModelCatalog.to_dict()` produces — i.e. `{"local": [...], "cloud": [...]}` —
    so the cache file and the published file are interchangeable.
    """
    try:
        req = urllib.request.Request(
            url,
            headers={
                "Accept": "application/json",
                "User-Agent": "OliveCode-advisor/1.0",
            },
        )
        with urllib.request.urlopen(req, timeout=timeout) as resp:
            raw = resp.read()
    except (
        urllib.error.URLError,
        urllib.error.HTTPError,
        TimeoutError,
        OSError,
        ValueError,
    ):
        return None
    except Exception:
        return None

    # Parse + validate. ANY failure here returns None so the caller
    # falls back to cache / baseline without surfacing the error.
    try:
        data = json.loads(raw.decode("utf-8", errors="replace"))
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    try:
        cat = ModelCatalog.from_dict(data)
    except Exception:
        return None
    # Defensive: if the publisher gave us an empty/broken catalog, treat
    # that as a failure rather than overwriting a good cache with nothing.
    if not cat.local and not cat.cloud:
        return None
    return cat


def refresh_via_url(
    baseline: ModelCatalog,
    cache: Optional[ModelCatalog],
    url: str = CATALOG_URL,
) -> Optional[ModelCatalog]:
    """Fetch the developer's published catalog.json and merge it on top of
    the baseline (+ existing cache). Returns the merged catalog on success,
    or None on any failure so the caller falls back. Never raises.

    We merge (rather than replace) so that a published catalog with fewer
    entries than the baseline cannot remove the local fallbacks. The
    `merge()` function only adds or updates — it never deletes.
    """
    fresh = _fetch_catalog_json(url=url)
    if fresh is None:
        return None
    merged = merge(baseline, cache, fresh)
    if not merged.local and not merged.cloud:
        return None
    return merged


# ---------------------------------------------------------------------------
# Public entry point
# ---------------------------------------------------------------------------

def load_catalog(
    force_refresh: bool = False,
    cache_path: Optional[Path] = None,
) -> tuple[ModelCatalog, str, Optional[datetime]]:
    """Return (catalog, status, refreshed_at). NEVER raises.

    Refreshes from the static URL only when the cache is missing / stale /
    force-refresh requested. On any refresh failure (bad URL, no internet,
    timeout, malformed JSON, ...) we drop straight back to the last good
    cache, or to the hardcoded baseline if there's no cache yet.

    No API key is required for any path — the catalog is plain JSON.
    """
    path = cache_path or catalog_cache_path()
    baseline = _baseline()

    cache, refreshed_at = read_cached(path)

    need_refresh = (
        force_refresh
        or cache is None
        or refreshed_at is None
        or is_stale(refreshed_at)
    )
    if need_refresh:
        try:
            fresh = refresh_via_url(baseline, cache)
            if fresh is not None:
                write_cache(path, fresh)
                return fresh, "fresh", datetime.now(timezone.utc)
        except Exception:
            pass  # fall through to cache / baseline

    if cache is not None:
        stale = refreshed_at is None or is_stale(refreshed_at)
        return cache, ("cached-stale" if stale else "cached"), refreshed_at

    return baseline, "offline-default", None


__all__ = [
    "catalog_cache_path",
    "load_catalog",
    "merge",
    "is_stale",
    "CATALOG_URL",
    "CACHE_VERSION",
    "MAX_AGE_DAYS",
    "refresh_via_url",
]
