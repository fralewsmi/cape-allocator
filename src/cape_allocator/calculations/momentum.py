"""
Momentum overlay calculations.

Implements the 12-month momentum signal following Haghani & White (2022)
and Asness et al. (2013).

The signal: 12-month price return of the S&P 500, excluding the most recent
month (so months t-12 to t-1). Excluding the last month avoids the well-
documented short-term reversal effect.

References:
    Haghani, V. & White, J. (2022). "Man Doth Not Invest By Earnings Yield Alone."
    Elm Wealth. https://elmwealth.com/earnings-yield-dynamic-allocation/

    Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013).
    "Value and Momentum Everywhere." The Journal of Finance, 68(3), 929-985.
"""

from __future__ import annotations

import pandas as pd


def compute_momentum_signal(sp500_prices: pd.Series) -> float:
    """
    Compute the 12-month momentum signal for S&P 500.

    The signal is the price return from month t-12 to t-1 (excluding the
    most recent month to avoid short-term reversal effects).

    Parameters
    ----------
    sp500_prices : pd.Series
        Monthly closing prices of S&P 500 (^GSPC), indexed by date.
        Must have at least 13 months of data.

    Returns
    -------
    float
        12-month momentum return (decimal). Positive values indicate
        momentum favors equities.

    Raises
    ------
    ValueError
        If insufficient price history (< 13 months).
    """
    if len(sp500_prices) < 13:
        raise ValueError(
            f"Need at least 13 months of S&P 500 prices; got {len(sp500_prices)}"
        )

    # Sort by date (most recent first) and get the last 13 prices
    prices = sp500_prices.sort_index(ascending=False).iloc[:13]

    # Price 12 months ago (t-12)
    price_t12 = prices.iloc[12]
    # Price 1 month ago (t-1)
    price_t1 = prices.iloc[1]

    # Return from t-12 to t-1
    momentum = (price_t1 - price_t12) / price_t12

    return float(momentum)


def blend_signals(
    merton_allocation: float,
    momentum_signal: float,
    momentum_weight: float,
) -> float:
    """
    Blend Merton and momentum allocations using equal weights.

    f_blended = (1 - w) * f_merton + w * f_momentum

    where f_momentum = 1.0 if momentum_signal > 0, else 0.0

    Parameters
    ----------
    merton_allocation : float
        Unconstrained Merton share (can be negative or > 1.0).
    momentum_signal : float
        12-month momentum return (decimal).
    momentum_weight : float
        Weight given to momentum signal (0.0 = pure Merton, 1.0 = pure momentum).

    Returns
    -------
    float
        Blended equity allocation.
    """
    f_momentum = 1.0 if momentum_signal > 0 else 0.0
    f_blended = (1 - momentum_weight) * merton_allocation + momentum_weight * f_momentum
    return f_blended
