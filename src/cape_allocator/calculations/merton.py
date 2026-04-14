"""
Merton Rule portfolio allocation functions.

All functions are pure (stateless, no I/O) and operate on plain Python
floats.  They implement the optimal risky-asset weight for a CRRA investor
under the continuous-time framework of Merton (1971).

Attribution
-----------
Merton Rule formula:
    Merton, R. C. (1971).
    "Optimum Consumption and Portfolio Rules in a Continuous-Time Model."
    Journal of Economic Theory, 3(4), 373-413.

    f* = μ / (γ · σ²)

    where:
        f*  = optimal fraction of wealth allocated to the risky asset
        μ   = expected excess return of the risky asset over the risk-free rate
        γ   = coefficient of relative risk aversion (CRRA)
        σ   = expected annualised volatility of the risky asset

Application to Excess Earnings Yield:
    Haghani, V. & White, J. (2022).
    "Man Doth Not Invest By Earnings Yield Alone." Elm Wealth.
    https://elmwealth.com/earnings-yield-dynamic-allocation/

    μ is taken as the Excess Earnings Yield:
        μ = (1 / CAPE) - TIPS_real_yield

    σ = 0.18 (18%) is Haghani & White's recommended constant, matching
    the AllocateSmartly implementation (footnote 2 of their strategy page).

CER definition used in Ma et al. (2026), eq. (17):
    Campbell, J. Y. & Thompson, S. B. (2008).
    "Predicting Excess Stock Returns Out of Sample."
    Review of Financial Studies, 21(4), 1509-1531.
"""

from __future__ import annotations


def compute_excess_earnings_yield(
    earnings_yield: float,
    tips_yield: float,
) -> float:
    """
    Compute the Excess Earnings Yield (equity risk premium over TIPS).

    μ = EY - TIPS_yield

    where EY = 1/CAPE is the cyclically-adjusted earnings yield.

    Using TIPS rather than nominal Treasuries is deliberate: earnings yield
    is a real return estimate, so the appropriate comparison is the real
    risk-free rate (10-year TIPS yield), not the nominal Treasury yield.
    This avoids the unit-mixing error of the "Fed Model".

    Reference:
        Haghani & White (2022) — "Man Doth Not Invest By Earnings Yield Alone"
        Asness, C. S. (2003) — "Fight the Fed Model"
        Journal of Portfolio Management, 29(3).

    Parameters
    ----------
    earnings_yield : float
        1/CAPE — real equity return estimate (decimal).
    tips_yield : float
        10-year TIPS real yield (decimal), e.g. 0.022 for 2.2%.

    Returns
    -------
    float
        Excess earnings yield (decimal).  Negative values mean TIPS offer
        a higher expected real return than equities.
    """
    return earnings_yield - tips_yield


def compute_merton_share(
    mu: float,
    gamma: float,
    sigma: float,
) -> float:
    """
    Compute the unconstrained optimal equity allocation (Merton share).

    f* = μ / (γ · σ²)

    This is the fraction of wealth a CRRA investor with risk aversion γ
    should hold in equities, given expected excess return μ and volatility σ.

    Parameters
    ----------
    mu : float
        Expected excess return of equities over TIPS (decimal).
    gamma : float
        Coefficient of relative risk aversion (CRRA).  Must be > 0.
    sigma : float
        Expected annualised equity volatility (decimal).  Must be > 0.

    Returns
    -------
    float
        Unconstrained optimal equity weight.  Can be negative (short) or
        above 1.0 (leveraged) before bound application.

    Raises
    ------
    ValueError
        If *gamma* or *sigma* are not strictly positive.

    Examples
    --------
    Haghani & White (2022) calibration check:
        μ = 0.05, γ = 2, σ = 0.20 → f* = 0.05 / (2 * 0.04) = 0.625 (62.5%)
    """
    if gamma <= 0:
        raise ValueError(f"gamma must be strictly positive; got {gamma}.")
    if sigma <= 0:
        raise ValueError(f"sigma must be strictly positive; got {sigma}.")
    return mu / (gamma * sigma**2)


def apply_allocation_bounds(
    merton_share: float,
    min_equity: float,
    max_equity: float,
) -> float:
    """
    Clamp the Merton share to the investor's allocation bounds.

    Parameters
    ----------
    merton_share : float
        Unconstrained Merton share (from ``compute_merton_share``).
    min_equity : float
        Minimum equity allocation (floor), as a decimal.
    max_equity : float
        Maximum equity allocation (cap), as a decimal.

    Returns
    -------
    float
        Constrained equity allocation in [min_equity, max_equity].
    """
    return float(max(min_equity, min(max_equity, merton_share)))


def compute_cer(
    equity_allocation: float,
    excess_earnings_yield: float,
    gamma: float,
    sigma: float,
) -> float:
    """
    Compute the Certainty Equivalent Return (CER) for the allocation.

    CER = f·μ − (γ/2)·(f·σ)²

    The CER is the risk-free return the investor would be indifferent to
    receiving instead of the risky portfolio.  It is used by Ma et al.
    (2026) in Table 8 to compare the asset allocation value of different
    CAPE variants.

    Reference:
        Ma et al. (2026), eq. (17)
        Campbell & Thompson (2008), eq. for utility gain

    Parameters
    ----------
    equity_allocation : float
        Constrained equity allocation f (decimal).
    excess_earnings_yield : float
        μ — expected excess return over TIPS (decimal).
    gamma : float
        Coefficient of relative risk aversion.
    sigma : float
        Expected annualised equity volatility (decimal).

    Returns
    -------
    float
        Certainty equivalent return (decimal).
    """
    return (
        equity_allocation * excess_earnings_yield
        - (gamma / 2.0) * (equity_allocation * sigma) ** 2
    )
