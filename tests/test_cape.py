"""
Tests for CAPE calculation functions.

Deterministic cases are anchored to values from:
    Ma et al. (2026) Table A2  — toy 5-stock Component CAPE example
    Ma et al. (2026) Table 1   — historical summary statistics

Property-based tests use Hypothesis to verify mathematical invariants
that must hold for all valid inputs.
"""

from __future__ import annotations

import numpy as np
import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cape_allocator.calculations.cape import (
    CONSTANT_SLOPE_BETA,
    cape_percentile_vs_history,
    compute_component_cape,
    compute_earnings_yield,
    forecast_10y_return,
)
from cape_allocator.models.inputs import CapeVariant, HISTORICAL_MEAN_CAPE


# ── Deterministic tests ───────────────────────────────────────────────────────

class TestComputeEarningsYield:
    def test_cape_of_28_gives_approximately_357_bps(self) -> None:
        """1/28 ≈ 3.571%"""
        assert compute_earnings_yield(28.0) == pytest.approx(1 / 28.0)

    def test_shiller_historical_mean_cape(self) -> None:
        """At the aggregate historical mean CAPE of 21.65, EY ≈ 4.62%."""
        mean_cape = HISTORICAL_MEAN_CAPE[CapeVariant.AGGREGATE_10Y]
        ey = compute_earnings_yield(mean_cape)
        assert ey == pytest.approx(1 / mean_cape, rel=1e-6)

    def test_component_10y_historical_mean_cape(self) -> None:
        """At Component 10Y mean CAPE of 29.74, EY ≈ 3.36%."""
        mean_cape = HISTORICAL_MEAN_CAPE[CapeVariant.COMPONENT_10Y]
        ey = compute_earnings_yield(mean_cape)
        assert ey == pytest.approx(1 / 29.74, rel=1e-5)

    def test_end_2024_component_cape(self) -> None:
        """
        At the end-2024 Component CAPE of ~56×,
        EY ≈ 1.79%  (Ma et al. 2026 communicated to Globe and Mail, March 2026).
        """
        ey = compute_earnings_yield(56.0)
        assert ey == pytest.approx(1 / 56.0, rel=1e-6)
        assert ey < 0.02  # Below 2% — historically thin

    def test_zero_cape_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            compute_earnings_yield(0.0)

    def test_negative_cape_raises(self) -> None:
        with pytest.raises(ValueError, match="strictly positive"):
            compute_earnings_yield(-10.0)


class TestComputeComponentCape:
    """
    Verify the aggregation formula against Ma et al. (2026) Table A2.

    Table A2 presents a toy example with five stocks A-E.
    The paper states:
        Aggregate CAPE        = 13.26
        Earnings-weighted CAPE = 13.26  (same, as expected from eq. 15)
        MV-weighted Component CAPE = 15.01

    We test the MV-weighted computation (eq. 16), which is the paper's
    primary specification.
    """

    # Values from Ma et al. (2026) Table A2
    _PRICES    = np.array([10_000.0, 20_000.0, 30_000.0, 40_000.0, 50_000.0])
    _MEAN_EPS  = np.array([1_146.0, 1_201.0, 3_825.0, 2_000.0, 3_139.0])
    _MCAPS     = np.array([10_000.0, 20_000.0, 30_000.0, 40_000.0, 50_000.0])

    def test_component_cape_matches_paper_table_a2(self) -> None:
        """
        Ma et al. (2026) Table A2: MV-weighted Component CAPE = 15.01.
        """
        result = compute_component_cape(self._PRICES, self._MEAN_EPS, self._MCAPS)
        assert result == pytest.approx(15.01, rel=0.01)  # 1% tolerance

    def test_individual_capes_computed_correctly(self) -> None:
        """
        From Table A2, individual CAPEs are approximately:
        A=8.72, B=16.65, C=7.84, D=20.00, E=15.93
        """
        expected_individual = np.array([8.72, 16.65, 7.84, 20.00, 15.93])
        actual_individual = self._PRICES / self._MEAN_EPS
        assert actual_individual == pytest.approx(expected_individual, rel=0.01)

    def test_mismatched_lengths_raises(self) -> None:
        with pytest.raises(ValueError, match="same length"):
            compute_component_cape(
                np.array([100.0, 200.0]),
                np.array([10.0]),
                np.array([1000.0, 2000.0]),
            )

    def test_empty_arrays_raises(self) -> None:
        with pytest.raises(ValueError):
            compute_component_cape(
                np.array([]), np.array([]), np.array([])
            )

    def test_zero_market_cap_raises(self) -> None:
        with pytest.raises(ValueError, match="market_caps"):
            compute_component_cape(
                np.array([100.0]),
                np.array([10.0]),
                np.array([0.0]),
            )

    def test_zero_eps_raises(self) -> None:
        with pytest.raises(ValueError, match="mean_real_eps"):
            compute_component_cape(
                np.array([100.0]),
                np.array([0.0]),
                np.array([1000.0]),
            )

    def test_single_stock_equals_individual_cape(self) -> None:
        """With one constituent, Component CAPE == individual CAPE."""
        result = compute_component_cape(
            np.array([100.0]),
            np.array([5.0]),
            np.array([50_000.0]),
        )
        assert result == pytest.approx(20.0)

    def test_equal_weights_gives_simple_mean(self) -> None:
        """With equal market caps, Component CAPE = simple mean of individual CAPEs."""
        prices = np.array([100.0, 200.0, 300.0])
        eps    = np.array([10.0,  10.0,  10.0])   # Individual CAPEs: 10, 20, 30
        mcaps  = np.array([1.0,   1.0,   1.0])
        result = compute_component_cape(prices, eps, mcaps)
        assert result == pytest.approx(20.0)


class TestConstantSlopeBeta:
    def test_beta_equals_minus_one_fiftieth(self) -> None:
        """
        Li et al. (2025) and Ma et al. (2026) use β = -1/50 = -0.02.
        """
        assert CONSTANT_SLOPE_BETA == pytest.approx(-0.02)

    def test_forecast_returns_negative_for_high_cape(self) -> None:
        """Higher CAPE → lower (more negative) forecast contribution."""
        assert forecast_10y_return(50.0) < forecast_10y_return(20.0)

    def test_forecast_zero_cape_raises(self) -> None:
        with pytest.raises(ValueError):
            forecast_10y_return(0.0)


class TestCapePercentileVsHistory:
    def test_at_mean_gives_zero_deviation(self) -> None:
        mean = HISTORICAL_MEAN_CAPE[CapeVariant.COMPONENT_10Y]
        pct = cape_percentile_vs_history(mean, CapeVariant.COMPONENT_10Y)
        assert pct == pytest.approx(0.0, abs=1e-9)

    def test_above_mean_is_positive(self) -> None:
        mean = HISTORICAL_MEAN_CAPE[CapeVariant.COMPONENT_10Y]
        pct = cape_percentile_vs_history(mean * 1.5, CapeVariant.COMPONENT_10Y)
        assert pct == pytest.approx(50.0, rel=1e-6)

    def test_end_2024_component_cape_is_far_above_mean(self) -> None:
        """
        At 56×, the Component CAPE is ~88% above its historical mean of 29.74×.
        """
        pct = cape_percentile_vs_history(56.0, CapeVariant.COMPONENT_10Y)
        assert pct == pytest.approx((56.0 - 29.74) / 29.74 * 100, rel=1e-4)
        assert pct > 80.0


# ── Property-based tests (Hypothesis) ────────────────────────────────────────

_POSITIVE_FLOAT = st.floats(min_value=0.001, max_value=1000.0, allow_nan=False)
_CAPE_RANGE     = st.floats(min_value=1.0, max_value=500.0, allow_nan=False)
_ARRAY_3        = st.lists(
    st.floats(min_value=0.01, max_value=1e9, allow_nan=False),
    min_size=3, max_size=3,
)


@given(cape=_CAPE_RANGE)
def test_earnings_yield_always_positive(cape: float) -> None:
    assert compute_earnings_yield(cape) > 0.0


@given(cape=_CAPE_RANGE)
def test_earnings_yield_monotonically_decreasing(cape: float) -> None:
    """Higher CAPE always means lower earnings yield."""
    if cape < 499.0:
        assert compute_earnings_yield(cape) > compute_earnings_yield(cape + 1.0)


@given(cape=_CAPE_RANGE)
def test_earnings_yield_bounded_by_one(cape: float) -> None:
    """EY = 1/CAPE ≤ 1 for all CAPE ≥ 1."""
    assert compute_earnings_yield(cape) <= 1.0


@given(
    prices=_ARRAY_3,
    mean_eps_raw=_ARRAY_3,
    mcaps=_ARRAY_3,
)
@settings(max_examples=200)
def test_component_cape_positive(
    prices: list[float],
    mean_eps_raw: list[float],
    mcaps: list[float],
) -> None:
    """Component CAPE is always positive for positive inputs."""
    p = np.array(prices)
    e = np.array(mean_eps_raw)
    m = np.array(mcaps)
    result = compute_component_cape(p, e, m)
    assert result > 0.0


@given(
    prices=_ARRAY_3,
    mean_eps_raw=_ARRAY_3,
    scale=st.floats(min_value=0.01, max_value=100.0, allow_nan=False),
)
@settings(max_examples=200)
def test_component_cape_invariant_to_mcap_scale(
    prices: list[float],
    mean_eps_raw: list[float],
    scale: float,
) -> None:
    """
    Scaling all market caps by a constant leaves the Component CAPE unchanged
    (weights are normalised, so only relative sizes matter).
    """
    p = np.array(prices)
    e = np.array(mean_eps_raw)
    m = np.ones(3)
    result_base = compute_component_cape(p, e, m)
    result_scaled = compute_component_cape(p, e, m * scale)
    assert result_base == pytest.approx(result_scaled, rel=1e-9)
