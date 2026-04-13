"""
Tests for the top-level compute_allocation orchestrator.

All external data fetching is mocked so tests run offline.
Known values from the papers are used to verify the full calculation chain.
"""

from __future__ import annotations

from datetime import date

import pytest
from hypothesis import given, settings
from hypothesis import strategies as st

from cape_allocator.calculations.allocator import compute_allocation
from cape_allocator.models.inputs import CapeVariant, InvestorParams, MarketInputs

# ── Deterministic end-to-end tests ───────────────────────────────────────────

class TestComputeAllocationDeterministic:
    def test_haghani_white_calibration_full_chain(
        self,
        haghani_white_params: InvestorParams,
        haghani_white_market: MarketInputs,
    ) -> None:
        """
        Haghani & White (2022) example:
        EY=5.5%, TIPS=0.5% → EEY=5.0%, γ=2, σ=20% → f*=62.5%.
        CER = 0.625*0.05 - 1*(0.625*0.20)² = 0.03125 - 0.015625 = 0.015625
        """
        result = compute_allocation(haghani_white_params, haghani_white_market)
        assert result.excess_earnings_yield == pytest.approx(0.05, rel=1e-4)
        assert result.merton_share_unconstrained == pytest.approx(0.625, rel=1e-4)
        assert result.equity_allocation == pytest.approx(0.625, rel=1e-4)
        assert result.tips_allocation == pytest.approx(0.375, rel=1e-4)
        assert result.cer == pytest.approx(0.015625, rel=1e-4)

    def test_end_2024_component_cape_gives_low_equity(
        self,
        standard_investor: InvestorParams,
        current_market_approx: MarketInputs,
    ) -> None:
        """
        At Component CAPE ~56× and TIPS ~2%, EEY ≈ -0.21%.
        The Merton share is negative; after floor it should be 0%.
        """
        result = compute_allocation(standard_investor, current_market_approx)
        assert result.excess_earnings_yield < 0
        assert result.equity_allocation == pytest.approx(0.0)
        assert result.tips_allocation == pytest.approx(1.0)

    def test_allocation_echoes_inputs(
        self,
        standard_investor: InvestorParams,
        current_market_approx: MarketInputs,
    ) -> None:
        """All input values should be faithfully echoed in the result."""
        result = compute_allocation(standard_investor, current_market_approx)
        assert result.cape_value == current_market_approx.cape_value
        assert result.tips_yield == current_market_approx.tips_yield
        assert result.gamma == standard_investor.gamma
        assert result.sigma == standard_investor.sigma

    def test_negative_eey_attaches_warning(
        self,
        standard_investor: InvestorParams,
        current_market_approx: MarketInputs,
    ) -> None:
        result = compute_allocation(standard_investor, current_market_approx)
        codes = [w.code for w in result.warnings]
        assert "NEGATIVE_EXCESS_EARNINGS_YIELD" in codes

    def test_high_cape_attaches_info_warning(
        self,
        standard_investor: InvestorParams,
        current_market_approx: MarketInputs,
    ) -> None:
        result = compute_allocation(standard_investor, current_market_approx)
        codes = [w.code for w in result.warnings]
        assert "CAPE_SIGNIFICANTLY_ABOVE_MEAN" in codes

    def test_constrained_allocation_attaches_info_warning(self) -> None:
        """When the Merton share is above the cap, a warning is attached."""
        investor = InvestorParams(
            gamma=2.0, sigma=0.18, min_equity=0.0, max_equity=0.30
        )
        market = MarketInputs(
            cape_value=10.0,   # High EY → large Merton share → will hit cap
            tips_yield=-0.01,
            cape_variant=CapeVariant.AGGREGATE_10Y,
            as_of_date=date(2024, 1, 1),
        )
        result = compute_allocation(investor, market)
        assert result.equity_allocation == pytest.approx(0.30)
        assert result.allocation_is_constrained is True
        codes = [w.code for w in result.warnings]
        assert "ALLOCATION_CONSTRAINED" in codes

    def test_low_coverage_attaches_warning(self) -> None:
        investor = InvestorParams()
        market = MarketInputs(
            cape_value=30.0,
            tips_yield=0.02,
            cape_variant=CapeVariant.COMPONENT_10Y,
            constituent_coverage=0.65,  # Below 0.80 threshold
            as_of_date=date(2024, 1, 1),
        )
        result = compute_allocation(investor, market)
        codes = [w.code for w in result.warnings]
        assert "LOW_CONSTITUENT_COVERAGE" in codes

    def test_no_errors_on_clean_inputs(self) -> None:
        investor = InvestorParams()
        market = MarketInputs(
            cape_value=20.0,
            tips_yield=0.01,
            cape_variant=CapeVariant.AGGREGATE_10Y,
            as_of_date=date(2023, 1, 1),
        )
        result = compute_allocation(investor, market)
        assert not result.has_errors()

    def test_tips_allocation_sums_to_one_minus_equity(
        self,
        standard_investor: InvestorParams,
        haghani_white_market: MarketInputs,
    ) -> None:
        result = compute_allocation(standard_investor, haghani_white_market)
        assert result.equity_allocation + result.tips_allocation == pytest.approx(1.0)

    def test_computed_fields_are_correct(
        self,
        standard_investor: InvestorParams,
        current_market_approx: MarketInputs,
    ) -> None:
        result = compute_allocation(standard_investor, current_market_approx)
        expected_mean = 29.74
        assert result.historical_mean_cape == pytest.approx(expected_mean, rel=1e-5)
        expected_pct = (56.0 - 29.74) / 29.74 * 100
        assert result.cape_vs_mean_pct == pytest.approx(expected_pct, rel=1e-4)

    def test_ma_investor_gamma_5(self, ma_investor: InvestorParams) -> None:
        """
        Ma et al. (2026) Table 8 uses γ=5.
        At historical mean Component CAPE (29.74×) with TIPS ~2%:
        EEY ≈ 1.36%, f* = 0.0136 / (5 * 0.0324) ≈ 8.4%
        CER > 0.
        """
        market = MarketInputs(
            cape_value=29.74,
            tips_yield=0.02,
            cape_variant=CapeVariant.COMPONENT_10Y,
            as_of_date=date(2024, 1, 1),
        )
        result = compute_allocation(ma_investor, market)
        assert result.excess_earnings_yield > 0
        assert result.equity_allocation > 0
        assert result.cer > 0


# ── Property-based tests (Hypothesis) ────────────────────────────────────────

_CAPE_ST   = st.floats(min_value=1.0, max_value=500.0, allow_nan=False)
_TIPS_ST   = st.floats(min_value=-0.05, max_value=0.10, allow_nan=False)
_GAMMA_ST  = st.floats(min_value=0.5, max_value=20.0, allow_nan=False)
_SIGMA_ST  = st.floats(min_value=0.05, max_value=0.60, allow_nan=False)
_BOUND_ST  = st.floats(min_value=0.0, max_value=0.45, allow_nan=False)


@given(
    cape=_CAPE_ST,
    tips=_TIPS_ST,
    gamma=_GAMMA_ST,
    sigma=_SIGMA_ST,
    min_eq=_BOUND_ST,
)
@settings(max_examples=500)
def test_allocation_always_within_bounds(
    cape: float, tips: float, gamma: float, sigma: float, min_eq: float
) -> None:
    """
    For any valid combination of inputs, the equity allocation must be
    within [min_equity, max_equity].
    """
    max_eq = min_eq + 0.55  # Ensure max > min
    investor = InvestorParams(
        gamma=gamma,
        sigma=sigma,
        min_equity=min_eq,
        max_equity=max_eq,
    )
    market = MarketInputs(
        cape_value=cape,
        tips_yield=tips,
        cape_variant=CapeVariant.AGGREGATE_10Y,
        as_of_date=date(2024, 1, 1),
    )
    result = compute_allocation(investor, market)
    assert min_eq - 1e-10 <= result.equity_allocation <= max_eq + 1e-10


@given(
    cape=_CAPE_ST,
    tips=_TIPS_ST,
    gamma=_GAMMA_ST,
    sigma=_SIGMA_ST,
)
@settings(max_examples=300)
def test_equity_plus_tips_always_sums_to_one(
    cape: float, tips: float, gamma: float, sigma: float
) -> None:
    """equity_allocation + tips_allocation must always equal 1.0."""
    investor = InvestorParams(gamma=gamma, sigma=sigma)
    market = MarketInputs(
        cape_value=cape,
        tips_yield=tips,
        cape_variant=CapeVariant.AGGREGATE_10Y,
        as_of_date=date(2024, 1, 1),
    )
    result = compute_allocation(investor, market)
    assert result.equity_allocation + result.tips_allocation == pytest.approx(1.0)
