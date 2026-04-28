"""
Tests for Pydantic input/output models.

Covers domain validation, cross-field constraints, and round-trip
serialisation.
"""

from __future__ import annotations

import pytest
from pydantic import ValidationError

from cape_allocator.models.inputs import CapeVariant, InvestorParams, MarketInputs
from cape_allocator.models.outputs import DataWarning, WarningSeverity


class TestInvestorParams:
    def test_defaults_match_haghani_white(self) -> None:
        """Default γ=2.0, σ=0.18 per Haghani & White (2022)."""
        p = InvestorParams()
        assert p.gamma == 2.0
        assert p.sigma == 0.18
        assert p.momentum_weight == 0.0
        assert p.cape_variant == CapeVariant.COMPONENT_10Y

    def test_valid_custom_params(self) -> None:
        p = InvestorParams(gamma=5.0, sigma=0.20, momentum_weight=0.5)
        assert p.gamma == 5.0
        assert p.momentum_weight == 0.5

    def test_gamma_too_low_rejected(self) -> None:
        with pytest.raises(ValidationError, match="gamma"):
            InvestorParams(gamma=0.0)

    def test_gamma_too_high_rejected(self) -> None:
        with pytest.raises(ValidationError, match="gamma"):
            InvestorParams(gamma=25.0)

    def test_sigma_too_low_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sigma"):
            InvestorParams(sigma=0.01)

    def test_sigma_too_high_rejected(self) -> None:
        with pytest.raises(ValidationError, match="sigma"):
            InvestorParams(sigma=0.99)

    def test_json_round_trip(self) -> None:
        p = InvestorParams(gamma=3.0, sigma=0.15)
        p2 = InvestorParams.model_validate_json(p.model_dump_json())
        assert p == p2


class TestMarketInputs:
    def test_valid_construction(self) -> None:
        m = MarketInputs(cape_value=29.74, tips_yield=0.022)
        assert m.cape_value == 29.74
        assert m.tips_yield == 0.022

    def test_zero_cape_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cape_value"):
            MarketInputs(cape_value=0.0, tips_yield=0.02)

    def test_negative_cape_rejected(self) -> None:
        with pytest.raises(ValidationError, match="cape_value"):
            MarketInputs(cape_value=-5.0, tips_yield=0.02)

    def test_negative_tips_allowed_within_bounds(self) -> None:
        """Negative TIPS yields are historically valid (e.g. 2020–2022)."""
        m = MarketInputs(cape_value=38.0, tips_yield=-0.01)
        assert m.tips_yield == -0.01

    def test_tips_too_extreme_rejected(self) -> None:
        with pytest.raises(ValidationError, match="tips_yield"):
            MarketInputs(cape_value=20.0, tips_yield=0.50)

    def test_coverage_out_of_range_rejected(self) -> None:
        with pytest.raises(ValidationError, match="constituent_coverage"):
            MarketInputs(cape_value=20.0, tips_yield=0.02, constituent_coverage=1.5)

    def test_coverage_none_is_valid(self) -> None:
        m = MarketInputs(cape_value=20.0, tips_yield=0.02, constituent_coverage=None)
        assert m.constituent_coverage is None


class TestCapeVariant:
    def test_all_variants_have_historical_mean(self) -> None:
        from cape_allocator.models.inputs import HISTORICAL_MEAN_CAPE

        for v in CapeVariant:
            assert v in HISTORICAL_MEAN_CAPE
            assert HISTORICAL_MEAN_CAPE[v] > 0

    def test_component_10y_mean_matches_paper(self) -> None:
        """Ma et al. (2026) Table 1: Component 10Y mean = 29.74."""
        from cape_allocator.models.inputs import HISTORICAL_MEAN_CAPE

        assert HISTORICAL_MEAN_CAPE[CapeVariant.COMPONENT_10Y] == pytest.approx(29.74)

    def test_aggregate_10y_mean_matches_paper(self) -> None:
        """Ma et al. (2026) Table 1: Aggregate 10Y mean = 21.65."""
        from cape_allocator.models.inputs import HISTORICAL_MEAN_CAPE

        assert HISTORICAL_MEAN_CAPE[CapeVariant.AGGREGATE_10Y] == pytest.approx(21.65)


class TestDataWarning:
    def test_severity_enum_values(self) -> None:
        assert WarningSeverity.INFO.value == "INFO"
        assert WarningSeverity.WARN.value == "WARN"
        assert WarningSeverity.ERROR.value == "ERROR"

    def test_warning_construction(self) -> None:
        w = DataWarning(
            severity=WarningSeverity.WARN,
            code="TEST_CODE",
            message="Test message",
        )
        assert w.severity == WarningSeverity.WARN
        assert w.code == "TEST_CODE"
