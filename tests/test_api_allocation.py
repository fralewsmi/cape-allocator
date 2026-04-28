"""
Tests for allocation endpoints.

Ma et al. (2026) SSRN 6060895 — Component CAPE
Haghani & White (2022) Elm Wealth — Excess Earnings Yield + Merton Rule
Merton (1971) Journal of Economic Theory — Merton Rule formula
"""

from datetime import date
from unittest.mock import patch

import pytest
from fastapi.testclient import TestClient

from api.main import app
from cape_allocator.models.inputs import CapeVariant
from cape_allocator.models.outputs import AllocationResult

client = TestClient(app)


def test_post_allocation_manual() -> None:
    """Test POST /api/allocation/manual with known values."""
    # Haghani & White (2022) calibration: μ=5%, γ=2, σ=20% → f*=62.5%
    response = client.post(
        "/api/allocation/manual",
        json={
            "gamma": 2.0,
            "sigma": 0.20,
            "cape_value": 1 / 0.055,  # EY = 5.5%
            "tips_yield": 0.005,  # EEY = 5.0%
            "cape_variant": "aggregate_10y",
        },
    )
    assert response.status_code == 200
    data = response.json()
    assert data["equity_allocation"] == pytest.approx(0.625, abs=0.01)
    assert data["gamma"] == 2.0
    assert data["sigma"] == 0.20


@patch("api.routers.allocation.fetch_market_inputs_and_allocate")
def test_post_allocation(mock_fetch) -> None:
    """Test POST /api/allocation with mocked fetch."""
    mock_result = AllocationResult(
        cape_value=30.0,
        cape_variant=CapeVariant.COMPONENT_10Y,
        tips_yield=0.02,
        gamma=2.0,
        sigma=0.18,
        momentum_weight=0.0,
        as_of_date=date.today(),
        constituent_coverage=0.95,
        earnings_yield=1 / 30,
        excess_earnings_yield=1 / 30 - 0.02,
        merton_share_unconstrained=0.5,
        momentum_signal=0.0,
        f_momentum=0.0,
        equity_allocation=0.5,
        tips_allocation=0.5,
        cer=0.01,
    )
    mock_fetch.return_value = mock_result

    response = client.post(
        "/api/allocation",
        json={"gamma": 2.0, "sigma": 0.18},
    )
    assert response.status_code == 200
    data = response.json()
    assert data["equity_allocation"] == 0.5
