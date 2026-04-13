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

Requires FRED_API_KEY in the .env file.  Free registration:
    https://fred.stlouisfed.org/docs/api/api_key.html
"""

from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from cape_allocator.data.cache import cache_get, cache_set

load_dotenv()

_DAILY_SERIES = "DFII10"
_WEEKLY_SERIES = "WFII10"
_CACHE_KEY_DAILY = "fred_dfii10_daily"
_CACHE_KEY_WEEKLY = "fred_wfii10_weekly"


def _get_fred_client() -> Fred:
    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key or api_key == "your_fred_api_key_here":
        raise EnvironmentError(
            "FRED_API_KEY is not set.  Add it to your .env file.\n"
            "Free registration: https://fred.stlouisfed.org/docs/api/api_key.html"
        )
    return Fred(api_key=api_key)


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
    # Try daily first
    cached = cache_get(_CACHE_KEY_DAILY)
    if cached is not None:
        return cached["yield"], cached["series"]

    fred = _get_fred_client()

    for series_id, cache_key in [
        (_DAILY_SERIES, _CACHE_KEY_DAILY),
        (_WEEKLY_SERIES, _CACHE_KEY_WEEKLY),
    ]:
        try:
            series: pd.Series = fred.get_series(series_id)
            series = series.dropna()
            if series.empty:
                continue
            raw_pct: float = float(series.iloc[-1])
            tips_yield = raw_pct / 100.0  # FRED returns percent; convert to decimal
            result = {"yield": tips_yield, "series": series_id}
            cache_set(cache_key, result)
            return tips_yield, series_id
        except Exception:  # noqa: BLE001
            continue

    raise RuntimeError(
        f"Could not fetch a valid TIPS yield from FRED series "
        f"{_DAILY_SERIES} or {_WEEKLY_SERIES}.  "
        "Check your FRED_API_KEY and network connection."
    )
