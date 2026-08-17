"""
Cache for market data fetches.

Backend is selected from the ``CAPE_CACHE_URL`` environment variable:

- ``s3://bucket-name``         → S3 backend (production)
- ``/path/to/dir`` or ``~/…``  → file backend (local dev)
- unset                        → file backend at ``~/.cache/cape_allocator``

To add a new backend: implement the ``CacheBackend`` protocol and add a
branch to ``_backend()``.

Each cache entry is a JSON object:
    ``fetched_at``  ISO-8601 UTC timestamp
    ``data``        any JSON-serialisable payload

TTL is checked on read; stale entries are treated as misses.
"""

from __future__ import annotations

import json
import logging
import os
from datetime import UTC, datetime, timedelta
from pathlib import Path
from typing import Any, Protocol

logger = logging.getLogger(__name__)


def _ttl_hours() -> float:
    return float(os.environ.get("CAPE_CACHE_TTL_HOURS", "24"))


def _wrap(data: Any) -> str:
    return json.dumps({"fetched_at": datetime.now(tz=UTC).isoformat(), "data": data})


def _unwrap(raw: str) -> Any | None:
    """Return data if the payload is fresh, else None."""
    try:
        payload = json.loads(raw)
        fetched_at = datetime.fromisoformat(payload["fetched_at"])
        if datetime.now(tz=UTC) - fetched_at <= timedelta(hours=_ttl_hours()):
            return payload["data"]
        return None
    except (KeyError, ValueError, json.JSONDecodeError):
        return None


class CacheBackend(Protocol):
    def get(self, key: str) -> Any | None: ...
    def set(self, key: str, data: Any) -> None: ...
    def clear(self, key: str | None) -> None: ...
    def oldest_age_hours(self) -> float | None: ...


class FileBackend:
    def __init__(self, cache_dir: str | None = None) -> None:
        raw = cache_dir or os.environ.get("CAPE_CACHE_DIR", "~/.cache/cape_allocator")
        self._dir = Path(raw).expanduser()
        self._dir.mkdir(parents=True, exist_ok=True)

    def _path(self, key: str) -> Path:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return self._dir / f"{safe}.json"

    def get(self, key: str) -> Any | None:
        path = self._path(key)
        if not path.exists():
            return None
        try:
            return _unwrap(path.read_text())
        except Exception:  # noqa: BLE001
            return None

    def set(self, key: str, data: Any) -> None:
        try:
            self._path(key).write_text(_wrap(data))
        except Exception as e:  # noqa: BLE001
            logger.warning("File cache set error for key %r: %s", key, e)

    def clear(self, key: str | None) -> None:
        if key is not None:
            p = self._path(key)
            if p.exists():
                p.unlink()
        else:
            for f in self._dir.glob("*.json"):
                f.unlink()

    def oldest_age_hours(self) -> float | None:
        files = list(self._dir.glob("*.json"))
        if not files:
            return None
        now = datetime.now(tz=UTC)
        ages: list[float] = []
        for f in files:
            try:
                payload = json.loads(f.read_text())
                fetched_at = datetime.fromisoformat(payload["fetched_at"])
                ages.append((now - fetched_at).total_seconds() / 3600)
            except Exception:  # noqa: BLE001
                continue
        return max(ages) if ages else None


class S3Backend:
    def __init__(self, bucket: str) -> None:
        self._bucket = bucket

    def _key(self, key: str) -> str:
        safe = "".join(c if c.isalnum() or c in "-_." else "_" for c in key)
        return f"cache/{safe}.json"

    def get(self, key: str) -> Any | None:
        import boto3
        from botocore.exceptions import ClientError

        try:
            obj = boto3.client("s3").get_object(Bucket=self._bucket, Key=self._key(key))
            return _unwrap(obj["Body"].read().decode())
        except ClientError as e:
            if e.response["Error"]["Code"] in ("NoSuchKey", "404"):
                return None
            logger.warning("S3 cache get error for key %r: %s", key, e)
            return None
        except Exception as e:  # noqa: BLE001
            logger.warning("S3 cache get error for key %r: %s", key, e)
            return None

    def set(self, key: str, data: Any) -> None:
        import boto3

        try:
            boto3.client("s3").put_object(
                Bucket=self._bucket,
                Key=self._key(key),
                Body=_wrap(data).encode(),
                ContentType="application/json",
            )
        except Exception as e:  # noqa: BLE001
            logger.warning("S3 cache set error for key %r: %s", key, e)

    def clear(self, key: str | None) -> None:
        import boto3

        s3 = boto3.client("s3")
        try:
            if key is not None:
                s3.delete_object(Bucket=self._bucket, Key=self._key(key))
            else:
                paginator = s3.get_paginator("list_objects_v2")
                for page in paginator.paginate(Bucket=self._bucket, Prefix="cache/"):
                    objects = [{"Key": o["Key"]} for o in page.get("Contents", [])]
                    if objects:
                        s3.delete_objects(
                            Bucket=self._bucket, Delete={"Objects": objects}
                        )
        except Exception as e:  # noqa: BLE001
            logger.warning("S3 cache clear error: %s", e)

    def oldest_age_hours(self) -> float | None:
        import boto3

        s3 = boto3.client("s3")
        try:
            paginator = s3.get_paginator("list_objects_v2")
            now = datetime.now(tz=UTC)
            ages: list[float] = []
            for page in paginator.paginate(Bucket=self._bucket, Prefix="cache/"):
                for obj in page.get("Contents", []):
                    ages.append((now - obj["LastModified"]).total_seconds() / 3600)
            return max(ages) if ages else None
        except Exception as e:  # noqa: BLE001
            logger.warning("S3 cache age check error: %s", e)
            return None


def _backend() -> CacheBackend:
    url = os.environ.get("CAPE_CACHE_URL", "")
    if url.startswith("s3://"):
        bucket = url[len("s3://") :]
        return S3Backend(bucket)
    return FileBackend(url or None)


def cache_get(key: str) -> Any | None:
    """Return cached value for *key*, or None if missing/expired."""
    return _backend().get(key)


def cache_set(key: str, data: Any) -> None:
    """Persist *data* under *key* with the current UTC timestamp."""
    _backend().set(key, data)


def cache_clear(key: str | None = None) -> None:
    """Remove one cache entry (or all entries if *key* is None)."""
    _backend().clear(key)


def get_cache_age_hours(cache_dir: str | None = None) -> float | None:
    """
    Return the age in hours of the oldest cache entry, or None if empty.

    The *cache_dir* parameter is used only by the file backend.
    """
    backend = _backend()
    if isinstance(backend, FileBackend) and cache_dir:
        return FileBackend(cache_dir).oldest_age_hours()
    return backend.oldest_age_hours()
