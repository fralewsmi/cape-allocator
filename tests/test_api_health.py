from unittest.mock import AsyncMock, patch

from fastapi.testclient import TestClient

from api.dependencies import get_cache_dir, get_cache_ttl_hours, get_fred_api_key
from api.main import app

client = TestClient(app)

app.dependency_overrides[get_fred_api_key] = lambda: "test-key"
app.dependency_overrides[get_cache_dir] = lambda: "/tmp/cache"
app.dependency_overrides[get_cache_ttl_hours] = lambda: 24.0


@patch("api.routers.health.get_cache_age_hours", return_value=2.5)
@patch("api.routers.health.check_fred_connectivity", new_callable=AsyncMock)
def test_health_check(mock_fred, mock_cache) -> None:
    """Test GET /health returns correct structure."""
    mock_fred.return_value = True
    response = client.get("/health")
    assert response.status_code == 200
    data = response.json()
    assert "status" in data
    assert data["fred_reachable"] is True
    assert data["cache_age_hours"] == 2.5
