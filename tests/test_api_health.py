"""
Tests for health endpoint.
"""

from unittest.mock import patch

from fastapi.testclient import TestClient

from api.main import app

client = TestClient(app)


@patch("api.routers.health.get_cache_age_hours")
@patch("api.routers.health.check_fred_connectivity")
def test_health_check(mock_fred, mock_cache) -> None:
    """Test GET /health returns correct structure."""
    mock_fred.return_value = True
    mock_cache.return_value = 2.5

    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert "cache_age_hours" in data
    assert "fred_reachable" in data
    assert "as_of" in data
    assert data["fred_reachable"] is True
    assert data["cache_age_hours"] == 2.5
