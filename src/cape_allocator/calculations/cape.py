"""
Pure CAPE calculation functions.

All functions here are stateless and free of I/O.  They operate on
plain Python floats or numpy arrays and are fully unit-testable.

Attribution
-----------
Component CAPE weighting scheme:
    Ma, R., Marshall, B. R., Nguyen, N. H., & Visaltanachoti, N. (2026).
    "CAPE Ratios and Long-Term Returns." SSRN 6060895.
    Equations (12)-(16).

    Earnings-yield conversion:
    Campbell, J. Y. & Shiller, R. J. (1988).
    "Stock Prices, Earnings, and Expected Dividends."
    Journal of Finance, 43(3), 661-676.

Constant-slope forecast coefficient β = -1/50:
    Li, K., Li, Y., Lyu, C., & Yu, J. (2025).
    "How to Dominate the Historical Average."
    Review of Financial Studies, 38(10), 3086-3116.
    Ma et al. (2026) adopt this coefficient in their primary results
    (Table 2 and Table 3).
"""

from __future__ import annotations

import numpy as np

from cape_allocator.models.inputs import (
    CapeVariant,
    HISTORICAL_MEAN_CAPE,
)

# ── Constant-slope coefficient ─────────────────────────────────────────────
# β = -1/50, per Li et al. (2025) and adopted by Ma et al. (2026).
# This fixed slope replaces the OLS-estimated β in the predictive regression:
#     r_{t:t+10} = α + β * CAPE_t + ε
# Using a constant slope reduces estimation variance at the cost of a small
# bias, yielding lower mean-squared forecast error out-of-sample.
CONSTANT_SLOPE_BETA: float = -1.0 / 50.0


def compute_earnings_yield(cape: float) -> float:
    """
    Convert a CAPE ratio to a cyclically-adjusted earnings yield.

    EY = 1 / CAPE

    The earnings yield is treated as an estimate of the long-run real
    arithmetic return on equities (Campbell & Shiller, 1988).

    Parameters
    ----------
    cape : float
        CAPE ratio, must be strictly positive.

    Returns
    -------
    float
        Earnings yield as a decimal (e.g. 0.0357 for a CAPE of 28×).

    Raises
    ------
    ValueError
        If *cape* is not strictly positive.
    """
    if cape <= 0:
        raise ValueError(f"CAPE must be strictly positive; got {cape}.")
    return 1.0 / cape


def compute_component_cape(
    prices: np.ndarray,
    mean_real_eps: np.ndarray,
    market_caps: np.ndarray,
) -> float:
    """
    Compute the market-cap-weighted Component CAPE from constituent arrays.

    Implements Ma et al. (2026), equations (12)-(16):

        CAPE_i     = Price_i / mean_real_EPS_i               eq. (12)
        w_i        = MarketCap_i / sum(MarketCap)             eq. (16)
        ComponentCAPE = sum_i( w_i * CAPE_i )                eq. (16)

    Parameters
    ----------
    prices : np.ndarray, shape (N,)
        Current price per constituent.
    mean_real_eps : np.ndarray, shape (N,)
        Mean real EPS over the earnings window per constituent.
        Must be strictly positive for all entries.
    market_caps : np.ndarray, shape (N,)
        Market capitalisation per constituent (used for value-weighting).

    Returns
    -------
    float
        Market-cap-weighted Component CAPE.

    Raises
    ------
    ValueError
        If arrays have mismatched lengths, or any market cap / EPS is <= 0.
    """
    if not (len(prices) == len(mean_real_eps) == len(market_caps)):
        raise ValueError(
            "prices, mean_real_eps, and market_caps must all have the same length."
        )
    if len(prices) == 0:
        raise ValueError("At least one constituent is required.")
    if np.any(market_caps <= 0):
        raise ValueError("All market_caps must be strictly positive.")
    if np.any(mean_real_eps <= 0):
        raise ValueError("All mean_real_eps must be strictly positive.")

    individual_capes = prices / mean_real_eps          # eq. (12)
    weights = market_caps / market_caps.sum()          # eq. (16) weights
    return float(np.dot(weights, individual_capes))    # eq. (16) aggregation


def forecast_10y_return(cape: float) -> float:
    """
    Forecast the annualised 10-year log equity market return from the CAPE.

    Uses the constant-slope specification of Li et al. (2025) as adopted
    by Ma et al. (2026) in their primary results:

        r_{t:t+10} ≈ α + β * CAPE_t,   β = -1/50

    The intercept α is not used here; only the marginal contribution of
    the CAPE level relative to the historical mean is computed.  This is
    used diagnostically — the Merton Rule allocation uses the *excess
    earnings yield* (a levels-based signal) rather than the regression
    forecast directly.

    Parameters
    ----------
    cape : float
        Current CAPE ratio.

    Returns
    -------
    float
        Predicted annualised 10-year log return (decimal).
    """
    if cape <= 0:
        raise ValueError(f"CAPE must be strictly positive; got {cape}.")
    return CONSTANT_SLOPE_BETA * cape


def cape_percentile_vs_history(cape: float, variant: CapeVariant) -> float:
    """
    Return a rough indication of how elevated the CAPE is versus its
    historical mean from Ma et al. (2026) Table 1 (1964-2024).

    Parameters
    ----------
    cape : float
        Current CAPE ratio.
    variant : CapeVariant
        Which variant's historical mean to compare against.

    Returns
    -------
    float
        Percentage deviation from historical mean
        (positive = above mean, negative = below).
    """
    mean = HISTORICAL_MEAN_CAPE[variant]
    return (cape - mean) / mean * 100.0
