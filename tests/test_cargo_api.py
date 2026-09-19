"""Real CSV Cargo API contract tests without live Gemini calls."""

import pytest
from fastapi.testclient import TestClient

from marsa.agents.cargo_agent import CargoInvestigationAgent, CargoResult
from marsa.api.dependencies import get_cargo_agent
from marsa.api.main import app


@pytest.fixture
def client():
    app.dependency_overrides[get_cargo_agent] = lambda: CargoInvestigationAgent()
    with TestClient(app) as http:
        yield http
    app.dependency_overrides.clear()


@pytest.mark.parametrize(
    ("hour", "status"),
    [
        ("2025-01-01T00:00:00Z", "NORMAL"),
        ("2025-01-02T02:00:00Z", "MODERATE"),
        ("2025-01-08T22:00:00Z", "ELEVATED"),
        ("2025-07-06T05:00:00Z", "CRITICAL"),
    ],
)
def test_real_api_statuses(client, hour, status):
    response = client.post("/api/agents/cargo/investigate", json={"timestamp_utc": hour})
    assert response.status_code == 200
    result = CargoResult.model_validate(response.json())
    assert result.cargo_status == status
    assert result.reasoning_mode == "deterministic"
    assert result.agent == "cargo"


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
    response = client.post("/api/agents/cargo/investigate", json={"timestamp_utc": hour})
    assert response.status_code == expected


def test_partial_forecast_is_422(client):
    response = client.post(
        "/api/agents/cargo/investigate",
        json={"timestamp_utc": "2025-01-01T00:00:00Z", "forecast_risk": "high"},
    )
    assert response.status_code == 422


def test_api_forecast_does_not_change_evidence(client):
    base = client.post("/api/agents/cargo/investigate", json={
        "timestamp_utc": "2025-01-01T00:00:00Z"
    }).json()
    for risk in ("high", "low"):
        response = client.post("/api/agents/cargo/investigate", json={
            "timestamp_utc": "2025-01-01T00:00:00Z", "forecast_risk": risk,
            "horizon_hours": 6,
        })
        assert response.status_code == 200
        contextual = response.json()
        for field in ("cargo_status", "evidence", "findings", "limitations"):
            assert contextual[field] == base[field]
        assert contextual["forecast_context"] == {"horizon_hours": 6, "risk_level": risk}
