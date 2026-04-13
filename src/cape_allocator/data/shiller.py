"""
Shiller aggregate CAPE fallback.

Fetches Robert Shiller's publicly available S&P 500 data from Yale:
    http://www.econ.yale.edu/~shiller/data/ie_data.xls

The spreadsheet's "Data" sheet contains monthly observations of the
Shiller CAPE (column "CAPE") alongside price, earnings, dividends, and CPI.

No API key required.

This is used as the AGGREGATE_10Y variant and as a fallback when
yfinance constituent coverage drops below the 80% threshold.
"""

from __future__ import annotations

import io

import pandas as pd
import requests

from cape_allocator.data.cache import cache_get, cache_set

_SHILLER_URL = "http://www.econ.yale.edu/~shiller/data/ie_data.xls"
_CACHE_KEY = "shiller_aggregate_cape"
_SHEET_NAME = "Data"
_HEADER_ROW = 7  # Zero-indexed; Shiller's XLS has 8 header rows


def fetch_aggregate_cape() -> tuple[float, str]:
    """
    Return the most recent Shiller aggregate CAPE and the data source URL.

    Returns
    -------
    cape : float
        Most recent monthly Shiller CAPE (aggregate, 10-year earnings).
    source : str
        URL from which the data was fetched.

    Raises
    ------
    RuntimeError
        If the data cannot be fetched or parsed.
    """
    cached = cache_get(_CACHE_KEY)
    if cached is not None:
        return cached["cape"], cached["source"]

    try:
        response = requests.get(_SHILLER_URL, timeout=30)
        response.raise_for_status()
    except requests.RequestException as exc:
        raise RuntimeError(
            f"Could not download Shiller data from {_SHILLER_URL}: {exc}"
        ) from exc

    try:
        df = pd.read_excel(
            io.BytesIO(response.content),
            sheet_name=_SHEET_NAME,
            header=_HEADER_ROW,
        )
        # The CAPE column is labelled "CAPE" in recent versions of the file.
        # Drop rows where CAPE is NaN (trailing empty rows in the spreadsheet).
        cape_col = df["CAPE"].dropna()
        cape = float(cape_col.iloc[-1])
    except Exception as exc:  # noqa: BLE001
        raise RuntimeError(
            f"Could not parse Shiller XLS from {_SHILLER_URL}: {exc}"
        ) from exc

    result = {"cape": cape, "source": _SHILLER_URL}
    cache_set(_CACHE_KEY, result)
    return cape, _SHILLER_URL
