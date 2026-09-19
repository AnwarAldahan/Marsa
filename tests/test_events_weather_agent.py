from __future__ import annotations

import json
from datetime import UTC, datetime

import pandas as pd
import pytest

from marsa.agents.events_weather import EventsWeatherContextBuilder
from marsa.agents.events_weather_agent import (
    EventsWeatherAgent,
    EventsWeatherInvestigationRequest,
    EventsWeatherNarrative,
    ForecastContext,
    PressureLevel,
    ReasoningMode,
)
from marsa.agents.tools.events_weather import EventsWeatherContextSource
from marsa.common.exceptions import DataNotReadyError
from marsa.provenance.types import DataProvenance


def row_at(timestamp: str, **updates):
    row = {
        "hour_key": pd.Timestamp(timestamp),
        "wind_speed_10m": 5.0,
        "wave_height": 0.8,
        "weather_code": 95.0,
        "weather_pressure_index": 0.1,
        "event_active": False,
        "event_id": pd.NA,
        "event_type": pd.NA,
        "event_severity": pd.NA,
        "event_start": pd.NaT,
        "event_end": pd.NaT,
        "event_duration_hours": pd.NA,
        "event_represented_hours": pd.NA,
    }
    row.update(updates)
    return row


def request_at(timestamp: str = "2025-01-15T12:00:00Z", **updates):
    values = {"timestamp_utc": datetime.fromisoformat(timestamp.replace("Z", "+00:00"))}
    values.update(updates)
    return EventsWeatherInvestigationRequest(**values)


class StaticSource:
    def __init__(self, row):
        self.context = EventsWeatherContextBuilder().build(row)

    def get_context(self, timestamp_utc):
        return self.context


class MockProvider:
    def __init__(self, response=None, error: Exception | None = None):
        self.response = response
        self.error = error
        self.payload = None
        self.system_prompt = None

    def generate_structured(self, *, system_prompt, payload, response_model):
        self.system_prompt = system_prompt
        self.payload = payload
        if self.error:
            raise self.error
        return self.response


def narrative(**updates):
    values = {
        "weather_summary": "Available weather context is low relative to the baseline.",
        "event_summary": "No synthetic external event is active.",
        "calendar_summary": "No weekend or federal holiday context is active.",
        "possible_operational_contribution": "No significant contextual pressure was identified.",
    }
    values.update(updates)
    return values


def active_row(severity="minor", pressure=0.1):
    return row_at(
        "2025-08-24 14:00:00",
        weather_pressure_index=pressure,
        event_active=True,
        event_id="EVT_TEST",
        event_type="logistics_disruption",
        event_severity=severity,
        event_start=pd.Timestamp("2025-08-24 14:00:00"),
        event_end=pd.Timestamp("2025-08-24 20:00:00"),
        event_duration_hours=6,
        event_represented_hours=6,
    )


def test_normal_weather_without_event_is_low():
    result = EventsWeatherAgent(StaticSource(row_at("2025-01-15 12:00:00"))).investigate(
        request_at()
    )
    assert result.contextual_pressure.level is PressureLevel.LOW
    assert result.weather_assessment.level is PressureLevel.LOW
    assert "No significant" in result.possible_operational_contribution


def test_elevated_weather_without_event():
    source = StaticSource(row_at("2025-01-15 12:00:00", weather_pressure_index=0.8))
    result = EventsWeatherAgent(source).investigate(request_at())
    assert result.contextual_pressure.level is PressureLevel.ELEVATED
    assert result.weather_assessment.level is PressureLevel.ELEVATED


def test_active_minor_event_produces_moderate_pressure():
    result = EventsWeatherAgent(StaticSource(active_row("minor"))).investigate(
        request_at("2025-08-24T14:00:00Z")
    )
    assert result.contextual_pressure.level is PressureLevel.MODERATE


def test_active_severe_event_produces_elevated_pressure():
    result = EventsWeatherAgent(StaticSource(active_row("severe"))).investigate(
        request_at("2025-08-24T14:00:00Z")
    )
    assert result.contextual_pressure.level is PressureLevel.ELEVATED


def test_weather_and_active_event_are_both_findings():
    result = EventsWeatherAgent(StaticSource(active_row("moderate", pressure=0.8))).investigate(
        request_at("2025-08-24T14:00:00Z")
    )
    assert len(result.findings) >= 2
    assert {p for finding in result.findings for p in finding.provenance} >= {
        DataProvenance.OBSERVED,
        DataProvenance.DERIVED,
        DataProvenance.SYNTHETIC,
    }


def test_weekend_local_context():
    result = EventsWeatherAgent(StaticSource(row_at("2025-01-04 08:00:00"))).investigate(
        request_at("2025-01-04T08:00:00Z")
    )
    assert result.calendar_context.is_weekend_local is True


def test_holiday_local_context():
    result = EventsWeatherAgent(StaticSource(row_at("2025-07-04 07:00:00"))).investigate(
        request_at("2025-07-04T07:00:00Z")
    )
    assert result.calendar_context.is_public_holiday_local is True


def test_missing_weather_is_unknown_and_reduces_confidence():
    source = StaticSource(
        row_at(
            "2025-01-15 12:00:00",
            wind_speed_10m=float("nan"),
            wave_height=float("nan"),
            weather_pressure_index=float("nan"),
        )
    )
    result = EventsWeatherAgent(source).investigate(request_at())
    assert result.weather_assessment.level is PressureLevel.UNKNOWN
    assert result.contextual_pressure.level is PressureLevel.UNKNOWN
    assert result.contextual_pressure.confidence == 0.45
    assert any("incomplete" in item.lower() for item in result.limitations)


def test_partial_weather_is_unknown_even_if_pressure_value_is_present():
    source = StaticSource(
        row_at(
            "2025-01-15 12:00:00",
            wind_speed_10m=float("nan"),
            weather_pressure_index=0.8,
        )
    )
    result = EventsWeatherAgent(source).investigate(request_at())
    assert result.weather_assessment.level is PressureLevel.UNKNOWN


def test_missing_authoritative_timestamp_fails_cleanly(tmp_path):
    frame = pd.DataFrame([row_at("2025-01-01 00:00:00")])
    path = tmp_path / "state.parquet"
    frame.to_parquet(path, index=False)
    source = EventsWeatherContextSource(path)
    with pytest.raises(DataNotReadyError, match="Timestamp unavailable"):
        source.get_context(datetime(2025, 1, 1, 1, tzinfo=UTC))


def test_synthetic_provenance_is_preserved():
    result = EventsWeatherAgent(StaticSource(active_row())).investigate(
        request_at("2025-08-24T14:00:00Z")
    )
    assert result.external_event_assessment.provenance is DataProvenance.SYNTHETIC
    event_evidence = next(item for item in result.evidence_provenance if item.field == "external_event")
    assert event_evidence.provenance is DataProvenance.SYNTHETIC


def test_weather_code_never_reaches_llm_payload():
    provider = MockProvider(narrative())
    EventsWeatherAgent(StaticSource(row_at("2025-01-15 12:00:00")), provider).investigate(
        request_at()
    )
    assert "weather_code" not in json.dumps(provider.payload)


def test_unverified_units_from_llm_are_rejected():
    provider = MockProvider(narrative(weather_summary="Wind reached 20 knots."))
    result = EventsWeatherAgent(
        StaticSource(row_at("2025-01-15 12:00:00")), provider
    ).investigate(request_at())
    assert result.reasoning_mode is ReasoningMode.DETERMINISTIC_FALLBACK
    assert "knots" not in result.weather_assessment.summary


def test_causal_llm_language_is_rejected():
    provider = MockProvider(
        narrative(possible_operational_contribution="Weather caused congestion.")
    )
    result = EventsWeatherAgent(
        StaticSource(row_at("2025-01-15 12:00:00")), provider
    ).investigate(request_at())
    assert result.reasoning_mode is ReasoningMode.DETERMINISTIC_FALLBACK
    assert "caused" not in result.possible_operational_contribution.lower()


def test_valid_structured_llm_output_is_used():
    provider = MockProvider(EventsWeatherNarrative(**narrative()))
    result = EventsWeatherAgent(
        StaticSource(row_at("2025-01-15 12:00:00")), provider
    ).investigate(request_at())
    assert result.reasoning_mode is ReasoningMode.LLM_ASSISTED


def test_invalid_llm_output_uses_fallback():
    provider = MockProvider({"weather_summary": "Incomplete object"})
    result = EventsWeatherAgent(
        StaticSource(row_at("2025-01-15 12:00:00")), provider
    ).investigate(request_at())
    assert result.reasoning_mode is ReasoningMode.DETERMINISTIC_FALLBACK


def test_unavailable_llm_uses_fallback():
    provider = MockProvider(error=RuntimeError("provider unavailable"))
    result = EventsWeatherAgent(
        StaticSource(row_at("2025-01-15 12:00:00")), provider
    ).investigate(request_at())
    assert result.reasoning_mode is ReasoningMode.DETERMINISTIC_FALLBACK


def test_utc_timestamp_serializes_with_z():
    result = EventsWeatherAgent(StaticSource(row_at("2025-01-15 12:00:00"))).investigate(
        request_at()
    )
    payload = json.loads(result.model_dump_json())
    assert payload["timestamp_utc"] == "2025-01-15T12:00:00Z"


def test_forecast_context_does_not_change_domain_classification():
    source = StaticSource(row_at("2025-01-15 12:00:00", weather_pressure_index=0.1))
    agent = EventsWeatherAgent(source)
    low_forecast = agent.investigate(
        request_at(forecast_context=ForecastContext(horizon_hours=6, risk_level="low"))
    )
    high_forecast = agent.investigate(
        request_at(forecast_context=ForecastContext(horizon_hours=6, risk_level="high"))
    )
    assert low_forecast.contextual_pressure == high_forecast.contextual_pressure
    assert low_forecast.weather_assessment == high_forecast.weather_assessment


def test_active_llm_summary_must_disclose_synthetic_event():
    provider = MockProvider(narrative(event_summary="A logistics disruption is active."))
    result = EventsWeatherAgent(StaticSource(active_row()), provider).investigate(
        request_at("2025-08-24T14:00:00Z")
    )
    assert result.reasoning_mode is ReasoningMode.DETERMINISTIC_FALLBACK
    assert "synthetic" in result.external_event_assessment.summary.lower()


@pytest.mark.parametrize(
    "unsupported_contribution",
    [
        "Weekend timing may contribute to operational pressure.",
        "The public holiday may contribute to congestion pressure.",
        "Local calendar status may contribute to operational pressure.",
        "The day of week may contribute to operational pressure.",
    ],
)
def test_calendar_context_cannot_become_pressure_contribution(
    unsupported_contribution,
):
    provider = MockProvider(
        narrative(possible_operational_contribution=unsupported_contribution)
    )
    result = EventsWeatherAgent(
        StaticSource(row_at("2025-01-05 10:00:00")), provider
    ).investigate(request_at("2025-01-05T10:00:00Z"))
    assert result.reasoning_mode is ReasoningMode.DETERMINISTIC_FALLBACK
    assert "weekend" not in result.possible_operational_contribution.lower()
    assert "holiday" not in result.possible_operational_contribution.lower()
    assert result.calendar_context.is_weekend_local is True


def test_calendar_fact_is_allowed_when_contribution_is_weather_only():
    provider = MockProvider(
        narrative(
            calendar_summary="The local operational period is a weekend.",
            possible_operational_contribution=(
                "Elevated weather may contribute to operational pressure; causality is not established."
            ),
        )
    )
    result = EventsWeatherAgent(
        StaticSource(row_at("2025-01-05 10:00:00", weather_pressure_index=0.8)),
        provider,
    ).investigate(request_at("2025-01-05T10:00:00Z"))
    assert result.reasoning_mode is ReasoningMode.LLM_ASSISTED
    assert result.calendar_context.is_weekend_local is True
    assert "weekend" in result.calendar_context.summary.lower()
    assert "weekend" not in result.possible_operational_contribution.lower()


def test_provider_receives_response_model_and_operational_prompt():
    provider = MockProvider(narrative())
    EventsWeatherAgent(StaticSource(row_at("2025-01-15 12:00:00")), provider).investigate(
        request_at()
    )
    assert "must not predict congestion" in provider.system_prompt.lower()
    assert "do not mention raw wind or wave numeric values" in provider.system_prompt.lower()
    assert provider.payload["deterministic_assessment"]["weather_level"] == "low"


@pytest.mark.parametrize(
    ("unsafe_text", "category"),
    [
        ("The port should deploy additional resources.", "strategy_recommendation"),
        ("The model predicts congestion at this hour.", "congestion_prediction"),
    ],
)
def test_strategy_and_congestion_claims_trigger_categorized_fallback(
    unsafe_text, category, caplog
):
    provider = MockProvider(narrative(possible_operational_contribution=unsafe_text))
    with caplog.at_level("INFO"):
        result = EventsWeatherAgent(
            StaticSource(row_at("2025-01-15 12:00:00")), provider
        ).investigate(request_at())
    assert result.reasoning_mode is ReasoningMode.DETERMINISTIC_FALLBACK
    assert f"Fallback reason: {category}" in caplog.text


def test_provider_error_logging_never_exposes_secret(caplog):
    provider = MockProvider(error=RuntimeError("failure with secret-value"))
    with caplog.at_level("INFO"):
        result = EventsWeatherAgent(
            StaticSource(row_at("2025-01-15 12:00:00")), provider
        ).investigate(request_at())
    assert result.reasoning_mode is ReasoningMode.DETERMINISTIC_FALLBACK
    assert "Fallback reason: provider_error" in caplog.text
    assert "secret-value" not in caplog.text
