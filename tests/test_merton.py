"""
Tests for Merton Rule calculation functions.

The primary deterministic anchor is the Haghani & White (2022) calibration:
    μ = 5%, γ = 2, σ = 20%  →  f* = 62.5%

Source:
    Haghani, V. & White, J. (2022).
    "Man Doth Not Invest By Earnings Yield Alone." Elm Wealth.
    Footnote 10: "Using the Merton Rule with γ=2 we get:
                  k* = 62.5% = 5% / (2 * 20%²)."
"""

from __future__ import annotations

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cape_allocator.calculations.merton import (
    apply_allocation_bounds,
    compute_cer,
    compute_excess_earnings_yield,
    compute_merton_share,
)


# ── Deterministic tests ───────────────────────────────────────────────────────

class TestComputeExcessEarningsYield:
    def test_positive_excess(self) -> None:
        assert compute_excess_earnings_yield(0.055, 0.022) == pytest.approx(0.033)

    def test_zero_excess_when_equal(self) -> None:
        assert compute_excess_earnings_yield(0.04, 0.04) == pytest.approx(0.0)

    def test_negative_excess_when_tips_higher(self) -> None:
        """High TIPS yield relative to earnings yield → negative EEY (avoid equities)."""
        eey = compute_excess_earnings_yield(0.02, 0.03)
        assert eey == pytest.approx(-0.01)

    def test_negative_tips_increases_eey(self) -> None:
        """Negative real TIPS yield (as in 2020-2022) makes equities more attractive."""
        eey = compute_excess_earnings_yield(0.03, -0.01)
        assert eey == pytest.approx(0.04)


class TestComputeMertonShare:
    def test_haghani_white_calibration(self) -> None:
        """
        Haghani & White (2022), footnote 10:
        f* = 5% / (2 * 20%²) = 0.05 / (2 * 0.04) = 0.625
        """
        f_star = compute_merton_share(mu=0.05, gamma=2.0, sigma=0.20)
        assert f_star == pytest.approx(0.625, rel=1e-6)

    def test_zero_eey_gives_zero_allocation(self) -> None:
        """If equities offer no excess return, optimal allocation is zero."""
        f_star = compute_merton_share(mu=0.0, gamma=2.0, sigma=0.18)
        assert f_star == pytest.approx(0.0)

    def test_negative_eey_gives_negative_share(self) -> None:
        """Negative excess return → short equities (before bounds applied)."""
        f_star = compute_merton_share(mu=-0.02, gamma=2.0, sigma=0.18)
        assert f_star < 0.0

    def test_higher_gamma_reduces_allocation(self) -> None:
        """More risk-averse investor holds less equity for the same expected return."""
        f_low = compute_merton_share(mu=0.03, gamma=2.0, sigma=0.18)
        f_high = compute_merton_share(mu=0.03, gamma=5.0, sigma=0.18)
        assert f_low > f_high

    def test_higher_sigma_reduces_allocation(self) -> None:
        """Higher volatility reduces the optimal equity weight."""
        f_low_vol = compute_merton_share(mu=0.03, gamma=2.0, sigma=0.15)
        f_high_vol = compute_merton_share(mu=0.03, gamma=2.0, sigma=0.25)
        assert f_low_vol > f_high_vol

    def test_zero_gamma_raises(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            compute_merton_share(mu=0.05, gamma=0.0, sigma=0.18)

    def test_negative_gamma_raises(self) -> None:
        with pytest.raises(ValueError, match="gamma"):
            compute_merton_share(mu=0.05, gamma=-1.0, sigma=0.18)

    def test_zero_sigma_raises(self) -> None:
        with pytest.raises(ValueError, match="sigma"):
            compute_merton_share(mu=0.05, gamma=2.0, sigma=0.0)

    def test_end_2024_inputs(self) -> None:
        """
        At Component CAPE ≈ 56×, EY ≈ 1.79%; TIPS ≈ 2.0%; EEY ≈ -0.21%.
        With γ=2, σ=18%: Merton share should be negative (→ 0% after floor).
        """
        ey = 1 / 56.0
        eey = compute_excess_earnings_yield(ey, 0.020)
        f_star = compute_merton_share(eey, gamma=2.0, sigma=0.18)
        assert f_star < 0.0  # Before bounds: short equities

    def test_ma_et_al_gamma_5(self) -> None:
        """
        At historical mean Component CAPE of 29.74×, TIPS ~2%:
        EY ≈ 3.36%, EEY ≈ 1.36%, γ=5, σ=18%:
        f* = 0.0136 / (5 * 0.0324) ≈ 8.4%
        """
        ey = 1 / 29.74
        eey = compute_excess_earnings_yield(ey, 0.020)
        f_star = compute_merton_share(eey, gamma=5.0, sigma=0.18)
        expected = eey / (5.0 * 0.18**2)
        assert f_star == pytest.approx(expected, rel=1e-9)


class TestApplyAllocationBounds:
    def test_within_bounds_unchanged(self) -> None:
        assert apply_allocation_bounds(0.6, 0.0, 1.0) == pytest.approx(0.6)

    def test_below_floor_clamped_to_floor(self) -> None:
        assert apply_allocation_bounds(-0.1, 0.0, 1.0) == pytest.approx(0.0)

    def test_above_cap_clamped_to_cap(self) -> None:
        assert apply_allocation_bounds(1.5, 0.0, 1.0) == pytest.approx(1.0)

    def test_leverage_cap_respected(self) -> None:
        assert apply_allocation_bounds(2.0, 0.0, 1.5) == pytest.approx(1.5)

    def test_exact_floor_is_not_clamped(self) -> None:
        assert apply_allocation_bounds(0.0, 0.0, 1.0) == pytest.approx(0.0)

    def test_exact_cap_is_not_clamped(self) -> None:
        assert apply_allocation_bounds(1.0, 0.0, 1.0) == pytest.approx(1.0)


class TestComputeCer:
    def test_haghani_white_cer(self) -> None:
        """
        With f=0.625, μ=0.05, γ=2, σ=0.20:
        CER = 0.625 * 0.05 - (2/2) * (0.625 * 0.20)² = 0.03125 - 0.015625 = 0.015625
        """
        cer = compute_cer(0.625, 0.05, 2.0, 0.20)
        assert cer == pytest.approx(0.015625, rel=1e-6)

    def test_zero_allocation_gives_zero_cer(self) -> None:
        """No equity exposure → no risk premium and no risk cost."""
        cer = compute_cer(0.0, 0.05, 2.0, 0.18)
        assert cer == pytest.approx(0.0)

    def test_negative_eey_penalises_equity_holding(self) -> None:
        """Holding equities when EEY is negative should give negative CER."""
        cer = compute_cer(0.5, -0.02, 2.0, 0.18)
        assert cer < 0.0

    def test_cer_is_maximised_at_unconstrained_merton_share(self) -> None:
        """
        The Merton share is the allocation that maximises CER analytically.
        Verify numerically that nearby allocations give lower CER.
        """
        mu, gamma, sigma = 0.04, 2.0, 0.18
        f_optimal = compute_merton_share(mu, gamma, sigma)
        cer_optimal = compute_cer(f_optimal, mu, gamma, sigma)
        for delta in [-0.05, 0.05]:
            cer_perturbed = compute_cer(f_optimal + delta, mu, gamma, sigma)
            assert cer_optimal > cer_perturbed


# ── Property-based tests (Hypothesis) ────────────────────────────────────────

_GAMMA_ST = st.floats(min_value=0.5, max_value=20.0, allow_nan=False)
_SIGMA_ST = st.floats(min_value=0.05, max_value=0.60, allow_nan=False)
_MU_ST    = st.floats(min_value=-0.20, max_value=0.30, allow_nan=False)
_F_ST     = st.floats(min_value=0.0, max_value=1.5, allow_nan=False)


@given(mu=_MU_ST, gamma=_GAMMA_ST, sigma=_SIGMA_ST)
def test_merton_share_linear_in_mu(
    mu: float, gamma: float, sigma: float
) -> None:
    """f*(2μ) = 2 * f*(μ): the Merton share scales linearly with excess return."""
    f1 = compute_merton_share(mu, gamma, sigma)
    f2 = compute_merton_share(2 * mu, gamma, sigma)
    assert f2 == pytest.approx(2 * f1, rel=1e-9)


@given(mu=_MU_ST, gamma=_GAMMA_ST, sigma=_SIGMA_ST)
def test_merton_share_inversely_proportional_to_gamma(
    mu: float, gamma: float, sigma: float
) -> None:
    """f*(γ) * γ is constant: doubling risk aversion halves the allocation."""
    f1 = compute_merton_share(mu, gamma, sigma)
    f2 = compute_merton_share(mu, 2 * gamma, sigma)
    assert f1 == pytest.approx(2 * f2, rel=1e-9)


@given(
    f=_F_ST,
    min_eq=st.floats(min_value=0.0, max_value=0.4, allow_nan=False),
    max_eq=st.floats(min_value=0.6, max_value=1.5, allow_nan=False),
)
def test_bounded_allocation_always_in_range(
    f: float, min_eq: float, max_eq: float
) -> None:
    """The clamped allocation is always within [min_equity, max_equity]."""
    result = apply_allocation_bounds(f, min_eq, max_eq)
    assert min_eq <= result <= max_eq


@given(mu=_MU_ST, gamma=_GAMMA_ST, sigma=_SIGMA_ST)
@settings(max_examples=300)
def test_cer_maximised_at_merton_share(
    mu: float, gamma: float, sigma: float
) -> None:
    """
    The unconstrained Merton share maximises CER for any valid (μ, γ, σ).
    Any perturbation should not increase CER.
    """
    f_opt = compute_merton_share(mu, gamma, sigma)
    cer_opt = compute_cer(f_opt, mu, gamma, sigma)
    for delta in [-0.01, 0.01]:
        cer_perturbed = compute_cer(f_opt + delta, mu, gamma, sigma)
        assert cer_opt >= cer_perturbed - 1e-12  # Numerical tolerance
