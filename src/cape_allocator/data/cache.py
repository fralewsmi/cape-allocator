"""
Disk cache for market data fetches.

Stores serialised JSON under ~/.cache/cape_allocator/ (or CAPE_CACHE_DIR env).
Each cache entry is a dict with keys "fetched_at" (ISO timestamp) and "data".
TTL is checked on read; stale entries are treated as misses.

No third-party cache library is used — plain JSON files per key are trivial
to inspect and debug.
"""

from __future__ import annotations

import json
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any


def _cache_dir() -> Path:
    raw = os.environ.get("CAPE_CACHE_DIR", "~/.cache/cape_allocator")
    path = Path(raw).expanduser()
    path.mkdir(parents=True, exist_ok=True)
    return path


def _ttl_hours() -> float:
    return float(os.environ.get("CAPE_CACHE_TTL_HOURS", "24"))


def _cache_path(key: str) -> Path:
    safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
    return _cache_dir() / f"{safe}.json"


def cache_get(key: str) -> Any | None:
    """
    Return cached value for *key*, or None if missing / expired.

    Parameters
    ----------
    key:
        Logical cache key, e.g. ``"fred_dfii10"``.
    """
    path = _cache_path(key)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text())
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        if datetime.now(tz=UTC) - fetched_at > timedelta(hours=_ttl_hours()):
            return None
        return payload["data"]
    except (KeyError, ValueError, json.JSONDecodeError):
        return None  # Corrupt entry — treat as miss


def cache_set(key: str, data: Any) -> None:
    """
    Persist *data* under *key* with the current UTC timestamp.

    Parameters
    ----------
    key:
        Logical cache key.
    data:
        JSON-serialisable value.
    """
    payload = {
        "fetched_at": datetime.now(tz=UTC).isoformat(),
        "data": data,
    }
    _cache_path(key).write_text(json.dumps(payload, indent=2))


def cache_clear(key: str | None = None) -> None:
    """Remove one cache entry (or all entries if *key* is None)."""
    if key is not None:
        p = _cache_path(key)
        if p.exists():
            p.unlink()
    else:
        for f in _cache_dir().glob("*.json"):
            f.unlink()
