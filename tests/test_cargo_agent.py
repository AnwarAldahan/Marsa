"""Cargo source and narrative tests against the frozen teammate CSV."""

from datetime import UTC, datetime

import pandas as pd
import pytest
from pydantic import ValidationError

from marsa.agents.cargo_agent import (
    CONTRIBUTION,
    CargoForecastContext,
    CargoInvestigationAgent,
    CargoInvestigationRequest,
)
from marsa.agents.tools.cargo_status import CargoStatusSource
from marsa.common.exceptions import DataNotReadyError
from marsa.provenance.types import DataProvenance


def request(hour, forecast=None):
    return CargoInvestigationRequest(
        timestamp_utc=datetime.fromisoformat(hour.replace("Z", "+00:00")),
        forecast_context=forecast,
    )


class MockProvider:
    def __init__(self, response=None, error=None):
        self.response = response
        self.error = error

    def generate_structured(self, **kwargs):
        if self.error:
            raise self.error
        return self.response


SAFE = {
    "finding": "The supplied synthetic yard occupancy is 60.49% with current status NORMAL.",
    "evidence_fields": ["yard_occupancy_percent", "cargo_status"],
    "possible_operational_contribution": CONTRIBUTION,
}


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        ("2025-01-01T00:00:00Z", "NORMAL"),
        ("2025-01-02T02:00:00Z", "MODERATE"),
        ("2025-01-08T22:00:00Z", "ELEVATED"),
        ("2025-07-06T05:00:00Z", "CRITICAL"),
        ("2025-08-12T22:00:00Z", "MODERATE"),
        ("2025-09-24T02:00:00Z", "NORMAL"),
    ],
)
def test_stored_status_and_exact_row(hour, expected):
    source = CargoStatusSource()
    result = CargoInvestigationAgent(source=source).investigate(request(hour))
    frame = pd.read_csv(source.dataset_path)
    row = frame.loc[frame.timestamp.eq(str(result.timestamp_utc))].iloc[0]
    assert result.cargo_status == row.operations_status == expected
    assert result.evidence.containers_in_yard == row.containers_in_yard
    assert result.evidence.yard_capacity == row.yard_capacity
    assert result.evidence.container_arrivals_last_1h == row.container_arrivals
    assert result.evidence.container_departures_last_1h == row.container_departures
    assert result.evidence.gate_throughput_last_1h == row.gate_throughput
    assert result.reasoning_mode == "deterministic"


@pytest.mark.parametrize(
    "hour",
    ["2025-01-01T00:30:00Z", "2025-01-01T00:00:01Z", "2024-12-31T23:00:00Z",
     "2026-01-01T00:00:00Z"],
)
def test_invalid_hour_rejected(hour):
    with pytest.raises(ValueError):
        CargoInvestigationAgent().investigate(request(hour))


def test_naive_hour_rejected():
    with pytest.raises(ValidationError):
        CargoInvestigationRequest(timestamp_utc=datetime(2025, 1, 1))


def test_missing_authoritative_hour_rejected():
    with pytest.raises(DataNotReadyError):
        CargoInvestigationAgent().investigate(request("2025-12-05T04:00:00Z"))


def test_timezone_offset_resolves_exact_utc_hour():
    result = CargoInvestigationAgent().investigate(
        request("2024-12-31T16:00:00-08:00")
    )
    assert result.timestamp_utc == datetime(2025, 1, 1, tzinfo=UTC)


def test_forecast_invariance_and_provenance():
    agent = CargoInvestigationAgent()
    baseline = agent.investigate(request("2025-01-01T00:00:00Z"))
    for risk in ("high", "low"):
        contextual = agent.investigate(request(
            "2025-01-01T00:00:00Z", CargoForecastContext(risk_level=risk, horizon_hours=6)
        ))
        assert contextual.cargo_status == baseline.cargo_status == "NORMAL"
        assert contextual.evidence == baseline.evidence
        assert contextual.findings == baseline.findings
        assert contextual.limitations == baseline.limitations
        assert contextual.evidence_provenance[:-1] == baseline.evidence_provenance
        assert contextual.evidence_provenance[-1].provenance == DataProvenance.PREDICTED


def test_all_field_provenance_is_explicit():
    result = CargoInvestigationAgent().investigate(request("2025-01-01T00:00:00Z"))
    mapping = {item.field: item for item in result.evidence_provenance}
    assert set(mapping) == set(type(result.evidence).model_fields) | {"cargo_status"}
    assert all(item.provenance != DataProvenance.OBSERVED for item in mapping.values())
    for field in ("yard_occupancy_percent", "cargo_flow_ratio", "cargo_status"):
        assert mapping[field].provenance == DataProvenance.DERIVED
        assert "synthetic" in mapping[field].semantic_limit.lower()
    for field in set(type(result.evidence).model_fields) - {"yard_occupancy_percent", "cargo_flow_ratio"}:
        assert mapping[field].provenance == DataProvenance.SYNTHETIC


def test_valid_gemini_only_adds_cited_narrative():
    baseline = CargoInvestigationAgent().investigate(request("2025-01-01T00:00:00Z"))
    result = CargoInvestigationAgent(llm_provider=MockProvider(SAFE)).investigate(
        request("2025-01-01T00:00:00Z")
    )
    assert result.reasoning_mode == "llm_assisted"
    assert result.cargo_status == baseline.cargo_status
    assert result.evidence == baseline.evidence
    assert result.findings[:-1] == baseline.findings
    assert result.findings[-1].provenance == (DataProvenance.AGENT_GENERATED,)
    assert result.evidence_provenance == baseline.evidence_provenance
    assert result.limitations == baseline.limitations


def test_grounded_unit_and_count_remain_llm_assisted():
    narrative = {
        **SAFE,
        "finding": "The synthetic model reports 6048.64 units in yard.",
        "evidence_fields": ["containers_in_yard"],
    }
    result = CargoInvestigationAgent(llm_provider=MockProvider(narrative)).investigate(
        request("2025-01-01T00:00:00Z")
    )
    assert result.reasoning_mode == "llm_assisted"


@pytest.mark.parametrize(
    "response",
    [
        {**SAFE, "finding": "The synthetic yard occupancy is 42%."},
        {**SAFE, "finding": "The synthetic truck waiting time is 15.12 hours.",
         "evidence_fields": ["truck_waiting_time_minutes"]},
        {**SAFE, "finding": "The synthetic yard occupancy is 60.49 minutes.",
         "evidence_fields": ["yard_occupancy_percent"]},
        {**SAFE, "finding": "The synthetic yard occupancy is 60.49%.", "evidence_fields": ["cargo_status"]},
        {**SAFE, "finding": "The observed terminal yard occupancy is 60.49%."},
        {**SAFE, "finding": "The synthetic status is CRITICAL."},
        {**SAFE, "finding": "The synthetic status is NORMAL because vessels caused congestion."},
        {**SAFE, "finding": "The synthetic status is NORMAL; open more gates."},
        {**SAFE, "finding": "The synthetic status is NORMAL and predicts high future congestion."},
        {**SAFE, "finding": "The synthetic status is NORMAL with a 90% forecast risk."},
        {**SAFE, "finding": "The live synthetic yard state is NORMAL."},
        {**SAFE, "finding": "The synthetic terminal has an incident and berth bottleneck."},
        {**SAFE, "possible_operational_contribution": "Synthetic evidence proves future congestion."},
        {**SAFE, "cargo_status": "CRITICAL"},
        {**SAFE, "evidence": {"yard_occupancy_percent": 42}},
        {"finding": "incomplete"},
    ],
)
def test_unsafe_gemini_falls_back_without_mutating_facts(response):
    baseline = CargoInvestigationAgent().investigate(request("2025-01-01T00:00:00Z"))
    result = CargoInvestigationAgent(llm_provider=MockProvider(response)).investigate(
        request("2025-01-01T00:00:00Z")
    )
    assert result.reasoning_mode == "deterministic_fallback"
    assert result.cargo_status == baseline.cargo_status
    assert result.evidence == baseline.evidence
    assert result.findings == baseline.findings
    assert result.limitations == baseline.limitations


def test_provider_failure_uses_deterministic_fallback():
    result = CargoInvestigationAgent(
        llm_provider=MockProvider(error=RuntimeError("provider unavailable"))
    ).investigate(request("2025-01-01T00:00:00Z"))
    assert result.reasoning_mode == "deterministic_fallback"
