"""
U.S. CPI fetcher used to convert nominal EPS to real terms.

Series: CPIAUCSL — Consumer Price Index for All Urban Consumers: All Items
        https://fred.stlouisfed.org/series/CPIAUCSL

This is the same CPI series used by Robert Shiller in his online dataset
to adjust earnings and prices to a common price level.

Requires FRED_API_KEY in .env.
"""

from __future__ import annotations

import os

import pandas as pd
from dotenv import load_dotenv
from fredapi import Fred

from cape_allocator.data.cache import cache_get, cache_set

load_dotenv()

_CPI_SERIES = "CPIAUCSL"
_CACHE_KEY = "fred_cpiaucsl"


def fetch_cpi_index() -> pd.Series:
    """
    Return monthly U.S. CPI (CPIAUCSL) as a pandas Series indexed by date.

    Values are the raw index level (not percent change).  The caller
    divides historical earnings by the historical CPI and multiplies by
    the current CPI to obtain real values at today's price level —
    the same approach used in Shiller's dataset.

    Returns
    -------
    pd.Series
        Monthly CPI index, DatetimeIndex, sorted ascending.

    Raises
    ------
    EnvironmentError
        If FRED_API_KEY is not set.
    RuntimeError
        If the series cannot be fetched.
    """
    cached = cache_get(_CACHE_KEY)
    if cached is not None:
        return pd.Series(
            cached["values"],
            index=pd.to_datetime(cached["index"]),
            name="CPIAUCSL",
        )

    api_key = os.environ.get("FRED_API_KEY", "")
    if not api_key or api_key == "your_fred_api_key_here":
        raise EnvironmentError(
            "FRED_API_KEY is not set.  Add it to your .env file.\n"
            "Free registration: https://fred.stlouisfed.org/docs/api/api_key.html"
        )

    fred = Fred(api_key=api_key)
    try:
        series: pd.Series = fred.get_series(_CPI_SERIES).dropna().sort_index()
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not fetch CPI series {_CPI_SERIES} from FRED: {exc}"
        ) from exc

    cache_set(
        _CACHE_KEY,
        {
            "index": [str(d) for d in series.index],
            "values": series.tolist(),
        },
    )
    return series
