"""
U.S. CPI fetcher used to convert nominal EPS to real terms.

Series: CPIAUCSL — Consumer Price Index for All Urban Consumers: All Items
        https://fred.stlouisfed.org/series/CPIAUCSL

This is the same CPI series used by Robert Shiller in his online dataset
to adjust earnings and prices to a common price level.

Uses the FRED REST API via ``fred._fetch_fred_series`` (same as TIPS data).
Requires FRED_API_KEY in .env.
"""

from __future__ import annotations

import logging

import pandas as pd

from cape_allocator.data.cache import cache_get, cache_set
from cape_allocator.data.fred import _fetch_fred_series, _fred_api_key

logger = logging.getLogger(__name__)

_CPI_SERIES = "CPIAUCSL"
_CACHE_KEY = "fred_cpiaucsl"
_CPI_FETCH_START = "1900-01-01"
# Below FRED's per-request cap; CPIAUCSL has on the order of 1e3 monthly points.
_CPI_OBS_LIMIT = 100_000


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
        s = pd.Series(
            cached["values"],
            index=pd.to_datetime(cached["index"]),
            name="CPIAUCSL",
        )
        logger.info("FRED CPI: cache hit (CPIAUCSL, %s months)", len(s))
        return s

    logger.info("FRED CPI: downloading CPIAUCSL from FRED API…")
    try:
        observations = _fetch_fred_series(
            _CPI_SERIES,
            _fred_api_key(),
            limit=_CPI_OBS_LIMIT,
            sort_order="asc",
            observation_start=_CPI_FETCH_START,
        )
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not fetch CPI series {_CPI_SERIES} from FRED: {exc}"
        ) from exc

    dates: list[pd.Timestamp] = []
    values: list[float] = []
    for row in observations:
        v = row.get("value")
        if v is None or v == ".":
            continue
        dates.append(pd.to_datetime(row["date"]))
        values.append(float(v))

    if not values:
        raise RuntimeError(
            f"Could not fetch CPI series {_CPI_SERIES} from FRED: empty series"
        )

    series = pd.Series(values, index=pd.DatetimeIndex(dates), name="CPIAUCSL")
    series = series.sort_index()
    logger.info("FRED CPI: loaded %s monthly CPIAUCSL points", len(series))

    cache_set(
        _CACHE_KEY,
        {
            "index": [str(d) for d in series.index],
            "values": series.tolist(),
        },
    )
    return series
