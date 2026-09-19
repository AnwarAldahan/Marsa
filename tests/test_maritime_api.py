"""Real-data Maritime HTTP contract tests without live Gemini calls."""

import pytest
from fastapi.testclient import TestClient

from marsa.agents.maritime_agent import MaritimeInvestigationAgent, MaritimeResult
from marsa.api.dependencies import get_maritime_agent
from marsa.api.main import app


@pytest.fixture
def client():
    app.dependency_overrides[get_maritime_agent] = lambda: MaritimeInvestigationAgent()
    with TestClient(app) as http:
        yield http
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("hour", "status"),
    [
        ("2025-01-02T03:00:00Z", "NORMAL"),
        ("2025-01-01T00:00:00Z", "ELEVATED"),
        ("2025-01-01T03:00:00Z", "CONGESTED"),
    ],
)
def test_real_api_examples(client, hour, status):
    response = client.post("/api/agents/maritime/investigate", json={"timestamp_utc": hour})
    assert response.status_code == 200
    result = MaritimeResult.model_validate(response.json())
    assert result.maritime_status == status
    assert "average_speed_knots" not in result.evidence.model_dump()


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        ("nonsense", 422),
        ("2025-01-01T00:00:00", 422),
        ("2025-01-01T00:01:00Z", 422),
        ("2024-12-31T23:00:00Z", 422),
        ("2025-12-05T04:00:00Z", 404),
    ],
)
def test_api_hour_errors(client, hour, expected):
    response = client.post("/api/agents/maritime/investigate", json={"timestamp_utc": hour})
    assert response.status_code == expected


def test_partial_forecast_is_422(client):
    response = client.post(
        "/api/agents/maritime/investigate",
        json={"timestamp_utc": "2025-01-01T00:00:00Z", "forecast_risk": "high"},
    )
    assert response.status_code == 422
