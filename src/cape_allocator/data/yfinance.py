"""
Component CAPE fetcher using yfinance.

Methodology
-----------
Implements the constituent-level CAPE aggregation described in
Ma et al. (2026), equations (12)-(16):

    For each S&P 500 constituent i:
        CAPE_i = Price_i / mean_real_EPS_i(t-window, t)

    Component CAPE = sum_i( w_i * CAPE_i )

where w_i is each stock's share of total S&P 500 market capitalisation
(value-weighting, per the paper's primary specification).

Data sources
------------
Constituent list : Wikipedia S&P 500 table (no API key required).
                   https://en.wikipedia.org/wiki/List_of_S%26P_500_companies
Prices / EPS     : Yahoo Finance via yfinance.
                   Trailing twelve-month EPS (info["trailingEps"]) is used
                   as the single most recent annual earnings figure.
                   We fetch up to `window_years` years of annual EPS from
                   yfinance's financials table where available.
CPI adjustment   : yfinance does not provide CPI.  We apply a simple
                   normalisation using the U.S. CPI series from FRED
                   (CPIAUCSL) to convert historical EPS to real terms,
                   consistent with Shiller's methodology.

Limitations
-----------
- yfinance earnings history is patchy for many constituents (typically
  2-5 years rather than 10).  Short histories increase estimation noise.
- Coverage below 80% of S&P 500 market cap triggers fallback to the
  Shiller aggregate CAPE (caller's responsibility — this module returns
  coverage and raises no exception).
- This is not a reproduction of the Ma et al. dataset (which uses
  Compustat/Siblis).  Results will differ, particularly the absolute
  level of the Component CAPE.  The methodology is faithfully applied
  to the best freely available data.

Attribution
-----------
Ma, R., Marshall, B. R., Nguyen, N. H., & Visaltanachoti, N. (2026).
"CAPE Ratios and Long-Term Returns." SSRN 6060895.
"""

from __future__ import annotations

import io
import logging
import warnings
from concurrent.futures import ThreadPoolExecutor, as_completed
from dataclasses import dataclass, field

import numpy as np
import pandas as pd
import requests
import yfinance as yf

from cape_allocator.data.cache import cache_get, cache_set
from cape_allocator.data.cpi import fetch_cpi_index

logger = logging.getLogger(__name__)

_SP500_WIKI_URL = "https://en.wikipedia.org/wiki/List_of_S%26P_500_companies"
_CACHE_KEY_TICKERS = "sp500_tickers"
_CACHE_KEY_COMPONENT_CAPE = "component_cape_{variant}_{window}y"
_MIN_COVERAGE_THRESHOLD = 0.80
_WINSORISE_CAPE_MAX = 200.0  # Per Ma et al. (2026): outliers are winsorised
# Parallel Yahoo requests; keep moderate to reduce rate-limit risk.
_CONSTITUENT_FETCH_WORKERS = 12


@dataclass
class ConstituentResult:
    """Per-constituent CAPE computation result."""

    ticker: str
    cape: float
    market_cap: float
    weight: float = 0.0  # Populated after total market cap is known
    years_of_data: int = 0
    used_ttm_only: bool = False  # True if only one year of EPS was available


@dataclass
class ComponentCapeResult:
    """Aggregate result of a Component CAPE computation pass."""

    cape: float
    coverage: float  # Fraction of tickers successfully computed
    constituent_results: list[ConstituentResult] = field(default_factory=list)
    tickers_attempted: int = 0
    tickers_succeeded: int = 0


def fetch_sp500_tickers() -> list[str]:
    """
    Return the current S&P 500 ticker list from Wikipedia.

    Returns
    -------
    list[str]
        Ticker symbols as they appear on Yahoo Finance (e.g. "BRK.B" → "BRK-B").
    """
    cached = cache_get(_CACHE_KEY_TICKERS)
    if cached is not None:
        logger.info("Wikipedia: using cached S&P 500 list (%s tickers)", len(cached))
        return cached

    logger.info("Wikipedia: downloading S&P 500 constituent table…")
    try:
        headers = {
            "User-Agent": (
                "Mozilla/5.0 (compatible; cape-allocator/0.1; "
                "+https://github.com/cape-allocator)"
            )
        }
        response = requests.get(_SP500_WIKI_URL, headers=headers, timeout=15)
        response.raise_for_status()
        tables = pd.read_html(io.StringIO(response.text), attrs={"id": "constituents"})
        df = tables[0]
        tickers: list[str] = df["Symbol"].str.replace(".", "-", regex=False).tolist()
    except Exception as exc:
        raise RuntimeError(
            f"Could not fetch S&P 500 constituent list from Wikipedia: {exc}"
        ) from exc

    cache_set(_CACHE_KEY_TICKERS, tickers)
    logger.info("Wikipedia: loaded %s tickers", len(tickers))
    return tickers


def _real_eps_series(
    ticker_obj: yf.Ticker,
    cpi_index: pd.Series,
    window_years: int,
    *,
    info: dict | None = None,
) -> tuple[list[float], int, bool]:
    """
    Build a list of real (CPI-adjusted) annual EPS values for one ticker.

    Attempts to use yfinance annual income statement data for up to
    `window_years` years, falling back to trailing-twelve-month EPS only.

    Returns
    -------
    real_eps : list[float]
        Real EPS values, oldest first, length <= window_years.
    years_available : int
        How many years of data were actually obtained.
    used_ttm_only : bool
        True if the income statement was unavailable and TTM EPS was used.
    """
    info = info if info is not None else (ticker_obj.info or {})
    ttm_eps: float | None = info.get("trailingEps")

    # Attempt to read annual income statement
    try:
        financials = ticker_obj.financials  # columns = dates, rows = line items
        if financials is not None and not financials.empty:
            eps_row = None
            for candidate in ("Basic EPS", "Diluted EPS", "EPS"):
                if candidate in financials.index:
                    eps_row = financials.loc[candidate]
                    break

            if eps_row is not None:
                eps_series = (
                    eps_row.dropna()
                    .sort_index()  # oldest first
                    .tail(window_years)
                )
                if len(eps_series) >= 1:
                    # CPI-adjust each year's EPS to current price level
                    current_cpi = float(cpi_index.iloc[-1])
                    real_eps_vals: list[float] = []
                    for dt, raw_eps in eps_series.items():
                        year = int(pd.Timestamp(dt).year)
                        year_cpi = _cpi_for_year(cpi_index, year)
                        if year_cpi and year_cpi > 0:
                            real_eps_vals.append(
                                float(raw_eps) * current_cpi / year_cpi
                            )
                        else:
                            real_eps_vals.append(float(raw_eps))
                    if real_eps_vals:
                        return real_eps_vals, len(real_eps_vals), False
    except Exception:  # noqa: BLE001
        pass  # Fall through to TTM-only path

    # Fallback: TTM EPS only (no CPI adjustment needed — it's already current)
    if ttm_eps is not None and ttm_eps > 0:
        return [float(ttm_eps)], 1, True

    return [], 0, False


def _cpi_for_year(cpi_index: pd.Series, year: int) -> float | None:
    """Return the mean CPI for a given calendar year, or None if unavailable."""
    try:
        mask = np.array(
            [pd.Timestamp(x).year == year for x in cpi_index.index],
            dtype=np.bool_,
        )
        annual = cpi_index[mask]
        if annual.empty:
            return None
        return float(annual.mean())
    except Exception:  # noqa: BLE001
        return None


def _compute_constituent_cape(
    ticker: str,
    cpi_index: pd.Series,
    window_years: int,
) -> ConstituentResult | None:
    """
    Compute a single constituent's CAPE ratio.

    Returns None if the ticker cannot be processed (missing price, negative
    or zero earnings, etc.).
    """
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            t = yf.Ticker(ticker)

        info = t.info or {}
        price: float | None = info.get("currentPrice") or info.get("regularMarketPrice")
        market_cap: float | None = info.get("marketCap")

        if price is None or price <= 0:
            return None
        if market_cap is None or market_cap <= 0:
            return None

        real_eps, years_available, used_ttm = _real_eps_series(
            t, cpi_index, window_years, info=info
        )

        if not real_eps:
            return None

        # Filter out non-positive EPS years before averaging
        # (Ma et al. winsorise but do not exclude; we exclude to avoid
        # negative or zero denominators with patchy free data)
        positive_eps = [e for e in real_eps if e > 0]
        if not positive_eps:
            return None

        mean_real_eps = float(np.mean(positive_eps))

        # Individual CAPE_i = Price_i / mean_real_EPS_i
        # Ma et al. (2026) eq. (12)
        raw_cape = price / mean_real_eps

        # Winsorise at upper bound (Ma et al. 2026 apply winsorisation)
        cape_i = min(raw_cape, _WINSORISE_CAPE_MAX)

        return ConstituentResult(
            ticker=ticker,
            cape=cape_i,
            market_cap=market_cap,
            years_of_data=years_available,
            used_ttm_only=used_ttm,
        )

    except Exception:  # noqa: BLE001
        return None


def fetch_component_cape(window_years: int = 10) -> ComponentCapeResult:
    """
    Compute the Component CAPE for the S&P 500.

    The Component CAPE is the market-cap-weighted average of individual
    constituent CAPE ratios, as per Ma et al. (2026) equations (12)-(16).

    Parameters
    ----------
    window_years : int
        Earnings averaging window (10 for COMPONENT_10Y, 5 for COMPONENT_5Y).

    Returns
    -------
    ComponentCapeResult
        Contains the aggregate CAPE, coverage fraction, and per-constituent
        detail.
    """
    cache_key = _CACHE_KEY_COMPONENT_CAPE.format(
        variant="component", window=window_years
    )
    cached = cache_get(cache_key)
    if cached is not None:
        logger.info(
            "Yahoo Finance: using cached component CAPE (%s-year window, "
            "%s/%s tickers last run)",
            window_years,
            cached["tickers_succeeded"],
            cached["tickers_attempted"],
        )
        # Reconstruct lightweight result from cache (no constituent detail)
        return ComponentCapeResult(
            cape=cached["cape"],
            coverage=cached["coverage"],
            tickers_attempted=cached["tickers_attempted"],
            tickers_succeeded=cached["tickers_succeeded"],
        )

    logger.info(
        "Yahoo Finance: building component CAPE — CPI + tickers (parallel), "
        "then %s workers for quotes…",
        _CONSTITUENT_FETCH_WORKERS,
    )

    with ThreadPoolExecutor(max_workers=2) as _io_pool:
        cpi_future = _io_pool.submit(fetch_cpi_index)
        tickers_future = _io_pool.submit(fetch_sp500_tickers)
        cpi_index = cpi_future.result()
        tickers = tickers_future.result()

    logger.info(
        "Yahoo Finance: fetching price, market cap, and EPS for %s tickers…",
        len(tickers),
    )

    results: list[ConstituentResult] = []
    total = len(tickers)
    done = 0
    with ThreadPoolExecutor(max_workers=_CONSTITUENT_FETCH_WORKERS) as pool:
        futures = [
            pool.submit(_compute_constituent_cape, sym, cpi_index, window_years)
            for sym in tickers
        ]
        for fut in as_completed(futures):
            try:
                result = fut.result()
            except Exception:  # noqa: BLE001
                pass
            else:
                if result is not None:
                    results.append(result)
            done += 1
            if done % 50 == 0 or done == total:
                logger.info("Yahoo Finance: progress %s/%s tickers", done, total)

    tickers_attempted = len(tickers)
    tickers_succeeded = len(results)

    if not results:
        raise RuntimeError(
            "Could not compute CAPE for any S&P 500 constituent. "
            "Check network connectivity and yfinance availability."
        )

    # Assign market-cap weights
    total_mcap = sum(r.market_cap for r in results)
    for r in results:
        r.weight = r.market_cap / total_mcap

    # Component CAPE = Σ_i( w_i * CAPE_i )
    # Ma et al. (2026) eq. (16): value-weighted average of individual CAPEs
    component_cape = float(sum(r.weight * r.cape for r in results))
    coverage = tickers_succeeded / tickers_attempted

    cache_set(
        cache_key,
        {
            "cape": component_cape,
            "coverage": coverage,
            "tickers_attempted": tickers_attempted,
            "tickers_succeeded": tickers_succeeded,
        },
    )

    logger.info(
        "Yahoo Finance: component CAPE done (coverage %s, %.1f×)",
        f"{coverage:.0%}",
        component_cape,
    )

    return ComponentCapeResult(
        cape=component_cape,
        coverage=coverage,
        constituent_results=results,
        tickers_attempted=tickers_attempted,
        tickers_succeeded=tickers_succeeded,
    )
