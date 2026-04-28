"""
Tests for sensitivity endpoint.

Ma et al. (2026) SSRN 6060895 — Component CAPE
Haghani & White (2022) Elm Wealth — Excess Earnings Yield + Merton Rule
Merton (1971) Journal of Economic Theory — Merton Rule formula
"""

import json

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


def test_sensitivity() -> None:
    """Test GET /api/sensitivity returns valid NDJSON."""
    response = client.get(
        "/api/sensitivity?gamma_min=1&gamma_max=1&cape_min=10&cape_max=10&cape_step=1&tips_yield=0.02&sigma=0.18"
    )
    assert response.status_code == 200
    assert response.headers["content-type"] == "application/x-ndjson"

    lines = response.text.strip().split("\n")
    assert len(lines) == 1
    data = json.loads(lines[0])
    assert "gamma" in data
    assert "cape" in data
    assert "equity_allocation" in data
    assert "cer" in data
    assert data["gamma"] == 1
    assert data["cape"] == 10.0
