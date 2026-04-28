"""
FRED data fetcher — 10-year TIPS real yield.

Series used:
    DFII10  Market Yield on U.S. Treasury Securities at 10-Year Constant
            Maturity, Inflation-Indexed (Daily)
            https://fred.stlouisfed.org/series/DFII10

    WFII10  Weekly version, used as fallback if daily is unavailable.
            https://fred.stlouisfed.org/series/WFII10

FRED values are in percent (e.g. 2.20 = 2.20%).  We convert to decimal
before returning (e.g. 0.0220).

Uses the FRED REST API (``series/observations``).  Requires FRED_API_KEY in
``.env``.  Free registration:
    https://fred.stlouisfed.org/docs/api/api_key.html
"""

from __future__ import annotations

import asyncio
import logging
import os

import requests
from dotenv import load_dotenv

from cape_allocator.data.cache import cache_get, cache_set

load_dotenv()

logger = logging.getLogger(__name__)

_FRED_OBSERVATIONS = "https://api.stlouisfed.org/fred/series/observations"
_TIMEOUT_SECONDS = 10
_DAILY_SERIES = "DFII10"
_WEEKLY_SERIES = "WFII10"
_CACHE_KEY_DAILY = "fred_dfii10_daily"
_CACHE_KEY_WEEKLY = "fred_wfii10_weekly"


def _fred_api_key() -> str:
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key or api_key == "your_fred_api_key_here":
        raise OSError(
            "FRED_API_KEY is not set.  Add it to your .env file.\n"
            "Free registration: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return api_key


async def check_fred_connectivity(api_key: str) -> bool:
    """
    Check if FRED API is reachable by fetching a small amount of data.
    """
    loop = asyncio.get_event_loop()
    try:
        response = await loop.run_in_executor(
            None,
            lambda: requests.get(
                _FRED_OBSERVATIONS,
                params={
                    "series_id": _DAILY_SERIES,
                    "api_key": api_key,
                    "limit": 1,
                    "sort_order": "desc",
                    "file_type": "json",
                },
                timeout=5,
            ),
        )
        return response.status_code == 200
    except Exception:
        return False


def _fetch_fred_series(
    series_id: str,
    api_key: str,
    *,
    limit: int = 10,
    sort_order: str = "desc",
    observation_start: str | None = None,
    observation_end: str | None = None,
    offset: int = 0,
) -> list[dict]:
    """Return raw FRED observations as a list of dicts (``date``, ``value``, …)."""
    params: dict[str, str | int] = {
        "series_id": series_id,
        "api_key": api_key,
        "file_type": "json",
        "sort_order": sort_order,
        "limit": limit,
        "offset": offset,
    }
    if observation_start is not None:
        params["observation_start"] = observation_start
    if observation_end is not None:
        params["observation_end"] = observation_end

    logger.debug(
        "FRED API: observations series_id=%s limit=%s sort_order=%s",
        series_id,
        limit,
        sort_order,
    )
    response = requests.get(_FRED_OBSERVATIONS, params=params, timeout=_TIMEOUT_SECONDS)
    response.raise_for_status()
    return response.json().get("observations", [])


def fetch_tips_yield() -> tuple[float, str]:
    """
    Return the most recent 10-year TIPS real yield as a decimal and the
    series ID that was actually used.

    Returns
    -------
    tips_yield : float
        Real yield as a decimal, e.g. 0.0220 for 2.20%.
    source_series : str
        ``"DFII10"`` (daily) or ``"WFII10"`` (weekly fallback).

    Raises
    ------
    EnvironmentError
        If FRED_API_KEY is missing.
    RuntimeError
        If neither series returns a valid observation.
    """
    cached = cache_get(_CACHE_KEY_DAILY)
    if cached is not None:
        logger.info(
            "FRED TIPS: cache hit (%s, %.3f%% as decimal)",
            cached["series"],
            cached["yield"] * 100,
        )
        return cached["yield"], cached["series"]

    api_key = _fred_api_key()

    for series_id, cache_key in [
        (_DAILY_SERIES, _CACHE_KEY_DAILY),
        (_WEEKLY_SERIES, _CACHE_KEY_WEEKLY),
    ]:
        try:
            logger.info("FRED TIPS: requesting latest observation for %s…", series_id)
            observations = _fetch_fred_series(
                series_id,
                api_key,
                limit=30,
                sort_order="desc",
            )
            raw_pct: float | None = None
            for row in observations:
                v = row.get("value")
                if v is None or v == ".":
                    continue
                raw_pct = float(v)
                break
            if raw_pct is None:
                continue
            tips_yield = raw_pct / 100.0  # FRED returns percent; convert to decimal
            result = {"yield": tips_yield, "series": series_id}
            cache_set(cache_key, result)
            logger.info(
                "FRED TIPS: using %s (latest %.2f%% → %.3f decimal)",
                series_id,
                raw_pct,
                tips_yield,
            )
            return tips_yield, series_id
        except Exception:  # noqa: BLE001
            continue

    raise RuntimeError(
        f"Could not fetch a valid TIPS yield from FRED series "
        f"{_DAILY_SERIES} or {_WEEKLY_SERIES}.  "
        "Check your FRED_API_KEY and network connection."
    )
