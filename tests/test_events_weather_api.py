from __future__ import annotations

from datetime import UTC, datetime

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marsa.agents.events_weather import EventsWeatherContextBuilder
from marsa.agents.events_weather_agent import (
    EventsWeatherAgent,
    EventsWeatherResult,
    ReasoningMode,
)
from marsa.api.dependencies import get_events_weather_agent
from marsa.api.main import app
from marsa.common.exceptions import DataNotReadyError


def canonical_row():
    return {
        "hour_key": pd.Timestamp("2025-01-05 10:00:00"),
        "wind_speed_10m": 8.0,
        "wave_height": 1.4,
        "weather_code": 95.0,
        "weather_pressure_index": 0.8,
        "event_active": False,
        "event_id": pd.NA,
        "event_type": pd.NA,
        "event_severity": pd.NA,
        "event_start": pd.NaT,
        "event_end": pd.NaT,
        "event_duration_hours": pd.NA,
        "event_represented_hours": pd.NA,
    }


class StaticSource:
    def __init__(self):
        self.context = EventsWeatherContextBuilder().build(canonical_row())

    def get_context(self, timestamp_utc):
        return self.context


class MissingSource:
    def get_context(self, timestamp_utc):
        raise DataNotReadyError(
            f"Timestamp unavailable in authoritative timeline: {timestamp_utc.isoformat()}"
        )


class BrokenAgent:
    def investigate(self, request):
        raise RuntimeError("internal failure with GEMINI_API_KEY=secret-value")


class UnavailableProvider:
    def generate_structured(self, **kwargs):
        raise RuntimeError("provider unavailable")


@pytest.fixture
def client():
    app.dependency_overrides[get_events_weather_agent] = lambda: EventsWeatherAgent(
        context_source=StaticSource()
    )
    with TestClient(app) as test_client:
        yield test_client
    app.dependency_overrides.clear()


def post(client, **updates):
    body = {"timestamp_utc": "2025-01-05T10:00:00Z"}
    body.update(updates)
    return client.post("/api/agents/events-weather/investigate", json=body)


def test_valid_request_returns_complete_typed_result(client):
    response = post(client, investigation_reason="Investigate contextual pressure")
    assert response.status_code == 200
    result = EventsWeatherResult.model_validate(response.json())
    assert result.agent == "events_weather"
    assert result.reasoning_mode is ReasoningMode.DETERMINISTIC


def test_invalid_timestamp_format_returns_422(client):
    response = post(client, timestamp_utc="not-a-timestamp")
    assert response.status_code == 422


def test_naive_timestamp_returns_422(client):
    response = post(client, timestamp_utc="2025-01-05T10:00:00")
    assert response.status_code == 422


def test_missing_authoritative_timestamp_returns_404():
    app.dependency_overrides[get_events_weather_agent] = lambda: EventsWeatherAgent(
        context_source=MissingSource()
    )
    try:
        with TestClient(app) as client:
            response = post(client, timestamp_utc="2025-12-05T04:00:00Z")
        assert response.status_code == 404
        assert "Timestamp unavailable" in response.json()["detail"]
    finally:
        app.dependency_overrides.clear()


def test_forecast_context_does_not_change_classification(client):
    baseline = post(client).json()
    contextual = post(client, forecast_risk="high", horizon_hours=6).json()
    assert baseline["contextual_pressure"] == contextual["contextual_pressure"]
    assert baseline["weather_assessment"] == contextual["weather_assessment"]


def test_partial_forecast_context_returns_422(client):
    response = post(client, forecast_risk="high")
    assert response.status_code == 422


def test_agent_provider_fallback_still_returns_200():
    fallback_agent = EventsWeatherAgent(
        context_source=StaticSource(),
        llm_provider=UnavailableProvider(),
    )
    app.dependency_overrides[get_events_weather_agent] = lambda: fallback_agent
    try:
        with TestClient(app) as client:
            response = post(client)
        assert response.status_code == 200
        assert response.json()["reasoning_mode"] == "deterministic_fallback"
    finally:
        app.dependency_overrides.clear()


def test_unexpected_error_is_sanitized_and_contains_no_api_key():
    app.dependency_overrides[get_events_weather_agent] = lambda: BrokenAgent()
    try:
        with TestClient(app) as client:
            response = post(client)
        assert response.status_code == 500
        body = response.text
        assert "GEMINI_API_KEY" not in body
        assert "secret-value" not in body
        assert "traceback" not in body.lower()
    finally:
        app.dependency_overrides.clear()


def test_request_timestamp_serializes_as_utc_z(client):
    response = post(client)
    assert response.json()["timestamp_utc"] == "2025-01-05T10:00:00Z"


def test_request_model_accepts_optional_forecast_pair(client):
    response = post(
        client,
        investigation_reason="Investigate elevated forecast context",
        forecast_risk="high",
        horizon_hours=6,
    )
    assert response.status_code == 200
    assert EventsWeatherResult.model_validate(response.json()).timestamp_utc == datetime(
        2025, 1, 5, 10, tzinfo=UTC
    )
