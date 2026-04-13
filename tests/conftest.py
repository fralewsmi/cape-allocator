"""
Shared test fixtures.

Fixed values are sourced directly from the academic papers to give the
tests a clear, attributable anchor:

    Ma et al. (2026) Table 1   — historical mean CAPEs
    Ma et al. (2026) Table A2  — toy 5-stock example (used to verify
                                  component CAPE aggregation)
    Haghani & White (2022)     — calibration example: μ=5%, γ=2, σ=20%
                                  → f*=62.5%
"""

from __future__ import annotations

from datetime import date

import numpy as np
import pytest

from cape_allocator.models.inputs import CapeVariant, InvestorParams, MarketInputs


# ── Haghani & White (2022) calibration fixture ───────────────────────────────
# "Using the Merton Rule with γ=2 we get: k* = 62.5% = 5% / (2 * 20%²)"
# Source: Haghani & White (2022), footnote 10.

@pytest.fixture
def haghani_white_params() -> InvestorParams:
    """Haghani & White (2022) reference parameters."""
    return InvestorParams(gamma=2.0, sigma=0.20, min_equity=0.0, max_equity=1.0)


@pytest.fixture
def haghani_white_market() -> MarketInputs:
    """
    Market inputs that produce μ = 5% in Haghani & White (2022).
    We use EY = 5.5% and TIPS = 0.5% so that EEY = 5.0%.
    """
    return MarketInputs(
        cape_value=1.0 / 0.055,   # EY = 5.5%  → CAPE ≈ 18.18×
        tips_yield=0.005,          # TIPS = 0.5%  → EEY = 5.0%
        cape_variant=CapeVariant.AGGREGATE_10Y,
        as_of_date=date(2022, 1, 1),
    )


# ── Ma et al. (2026) Table A2 toy example ────────────────────────────────────
# Five stocks A–E with known prices, earnings, and market caps.
# The paper shows: Aggregate CAPE = 13.26, MV-weighted Component CAPE = 15.01.

@pytest.fixture
def ma_table_a2_prices() -> np.ndarray:
    return np.array([10_000.0, 20_000.0, 30_000.0, 40_000.0, 50_000.0])


@pytest.fixture
def ma_table_a2_mean_eps() -> np.ndarray:
    """Average 10-year real earnings per stock (from Table A2)."""
    return np.array([1_146.0, 1_201.0, 3_825.0, 2_000.0, 3_139.0])


@pytest.fixture
def ma_table_a2_market_caps() -> np.ndarray:
    """Market caps equal to prices in the toy example (one share each)."""
    return np.array([10_000.0, 20_000.0, 30_000.0, 40_000.0, 50_000.0])


# ── Current market fixture (end-2024 approximate values) ─────────────────────
# Component 10Y CAPE at end-2024 ≈ 56× (Ma et al. 2026, communicated to
# Globe and Mail, March 2026).  TIPS yield approx 2.0%.

@pytest.fixture
def current_market_approx() -> MarketInputs:
    return MarketInputs(
        cape_value=56.0,
        tips_yield=0.020,
        cape_variant=CapeVariant.COMPONENT_10Y,
        constituent_coverage=0.92,
        as_of_date=date(2024, 12, 31),
    )


@pytest.fixture
def standard_investor() -> InvestorParams:
    """Haghani & White (2022) default investor."""
    return InvestorParams(gamma=2.0, sigma=0.18, min_equity=0.0, max_equity=1.0)


@pytest.fixture
def ma_investor() -> InvestorParams:
    """Ma et al. (2026) Table 8 investor (γ=5)."""
    return InvestorParams(gamma=5.0, sigma=0.18, min_equity=0.0, max_equity=1.0)
