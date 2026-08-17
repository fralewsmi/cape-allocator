"""
Momentum overlay calculations.

Implements the 12-month momentum signal following Haghani & White (2022)
and Asness et al. (2013).

Signal: 12-month S&P 500 price return from t-12 to t-1 (excluding the most
recent month to avoid the short-term reversal effect).

References:
    Haghani, V. & White, J. (2022). "Man Doth Not Invest By Earnings Yield Alone."
    Elm Wealth. https://elmwealth.com/earnings-yield-dynamic-allocation/

    Asness, C. S., Moskowitz, T. J., & Pedersen, L. H. (2013).
    "Value and Momentum Everywhere." The Journal of Finance, 68(3), 929–985.
"""

from __future__ import annotations

import pandas as pd


def compute_momentum_signal(sp500_prices: pd.Series) -> float:
    """
    Compute the 12-month momentum signal for S&P 500.

    Returns the price return from t-12 to t-1, excluding the most recent
    month to avoid the short-term reversal effect.

    Parameters
    ----------
    sp500_prices : pd.Series
        Monthly closing prices of S&P 500 (^GSPC), indexed by date.
        Must have at least 13 months of data.

    Returns
    -------
    float
        12-month momentum return (decimal). Positive = momentum favors equities.

    Raises
    ------
    ValueError
        If fewer than 13 months of price history are provided.
    """
    if len(sp500_prices) < 13:
        raise ValueError(
            f"Need at least 13 months of S&P 500 prices; got {len(sp500_prices)}"
        )

    # Sort descending and take the 13 most recent prices
    prices = sp500_prices.sort_index(ascending=False).iloc[:13]

    # t-12 and t-1 prices
    price_t12 = prices.iloc[12]
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
    Blend Merton and momentum allocations.

    f_blended = (1 - w) * f_merton + w * f_momentum

    where f_momentum = 1.0 if momentum_signal > 0, else 0.0.

    Parameters
    ----------
    merton_allocation : float
        Unconstrained Merton share (can be negative or > 1.0).
    momentum_signal : float
        12-month momentum return (decimal).
    momentum_weight : float
        Weight on momentum (0.0 = pure Merton, 1.0 = pure momentum).

    Returns
    -------
    float
        Blended equity allocation.
    """
    f_momentum = 1.0 if momentum_signal > 0 else 0.0
    f_blended = (1 - momentum_weight) * merton_allocation + momentum_weight * f_momentum
    return f_blended
