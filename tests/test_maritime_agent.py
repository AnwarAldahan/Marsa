"""Maritime adapter and narrative tests; fixture-only synthetic cases are labeled."""

from datetime import UTC, datetime

import pytest

from marsa.agents.maritime_agent import (
    MaritimeForecastContext,
    MaritimeInvestigationAgent,
    MaritimeInvestigationRequest,
)
from marsa.agents.tools.maritime_status import MaritimeStatusSource
from marsa.common.exceptions import DataNotReadyError
from marsa.provenance.types import DataProvenance


def request(hour, forecast=None):
    return MaritimeInvestigationRequest(
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
    "finding": "Current evidence is consistent with the supplied port operations assessment.",
    "evidence_fields": ["maritime_status", "waiting_ratio"],
    "possible_operational_contribution": "The available Maritime evidence may indicate current operational pressure.",
}


@pytest.mark.parametrize(
    ("hour", "expected"),
    [
        ("2025-01-02T03:00:00Z", "NORMAL"),
        ("2025-01-01T00:00:00Z", "ELEVATED"),
        ("2025-01-01T03:00:00Z", "CONGESTED"),
    ],
)
def test_real_csv_status_is_teammate_status(hour, expected):
    source = MaritimeStatusSource()
    core = source._load_core()
    original = core.get_status(hour)
    result = MaritimeInvestigationAgent(source=source).investigate(request(hour))
    assert original["port_status"] == result.maritime_status == expected
    assert original["waiting_ratio"] == result.evidence.waiting_ratio
    assert original["average_speed_knots"] == result.evidence.average_speed_value
    assert original["port_throughput_last_4h"] == result.evidence.port_throughput_value
    assert result.reasoning_mode == "deterministic"
    assert {item.provenance for item in result.evidence_provenance} == {DataProvenance.DERIVED}


def test_forecast_invariance():
    agent = MaritimeInvestigationAgent()
    for hour, risk, expected in (
        ("2025-01-02T03:00:00Z", "high", "NORMAL"),
        ("2025-01-01T03:00:00Z", "low", "CONGESTED"),
    ):
        baseline = agent.investigate(request(hour))
        forecasted = agent.investigate(
            request(hour, MaritimeForecastContext(risk_level=risk, horizon_hours=6))
        )
        assert baseline.maritime_status == forecasted.maritime_status == expected
        assert baseline.evidence == forecasted.evidence
        assert baseline.findings == forecasted.findings
        assert forecasted.evidence_provenance[-1].provenance == DataProvenance.PREDICTED


@pytest.mark.parametrize(
    "hour",
    ["2025-01-01T00:30:00Z", "2024-12-31T23:00:00Z", "2026-01-01T00:00:00Z"],
)
def test_invalid_hour_rejected(hour):
    with pytest.raises(ValueError):
        MaritimeInvestigationAgent().investigate(request(hour))


def test_naive_hour_rejected():
    with pytest.raises(ValueError):
        MaritimeInvestigationRequest(timestamp_utc=datetime(2025, 1, 1, 0))


def test_missing_real_hour_rejected():
    # This hour is a preserved source gap; no nearest-hour substitution is allowed.
    with pytest.raises(DataNotReadyError):
        MaritimeInvestigationAgent().investigate(request("2025-12-05T04:00:00Z"))


def test_timezone_offset_resolves_exact_utc_hour():
    result = MaritimeInvestigationAgent().investigate(
        MaritimeInvestigationRequest(timestamp_utc=datetime.fromisoformat("2024-12-31T16:00:00-08:00"))
    )
    assert result.timestamp_utc == datetime(2025, 1, 1, tzinfo=UTC)


class MismatchCore:
    def get_status(self, timestamp):
        return {
            "timestamp": "2025-01-01 01:00:00+00:00",
            "vessels": 1,
            "waiting_vessels": 0,
            "waiting_ratio": 0.0,
            "approaching_vessels": 0,
            "departing_vessels": 0,
            "average_speed_knots": 0.0,
            "port_throughput_last_4h": 0,
            "port_status": "NORMAL",
        }


def test_returned_timestamp_mismatch_rejected():
    with pytest.raises(DataNotReadyError, match="different hour"):
        MaritimeStatusSource(core=MismatchCore()).get_status(datetime(2025, 1, 1, tzinfo=UTC))


def test_exactly_one_teammate_lookup_for_valid_hour():
    core = MismatchCore()
    calls = []
    original = core.get_status

    def counted(timestamp):
        calls.append(timestamp)
        row = original(timestamp)
        row["timestamp"] = "2025-01-01 00:00:00+00:00"
        return row

    core.get_status = counted
    MaritimeStatusSource(core=core).get_status(datetime(2025, 1, 1, tzinfo=UTC))
    assert len(calls) == 1


def test_teammate_error_dictionary_is_lookup_failure():
    class ErrorCore:
        def get_status(self, timestamp):
            return {"error": "No port data available for this timestamp"}

    with pytest.raises(DataNotReadyError):
        MaritimeStatusSource(core=ErrorCore()).get_status(datetime(2025, 1, 1, tzinfo=UTC))


def test_teammate_pure_logic_with_explicit_unit_test_fixtures():
    # These values are test fixtures, not purported historical observations.
    core = MaritimeStatusSource()._load_core()
    assert core.calculate_waiting_ratio(0, 0) == 0.0
    assert core.determine_port_status(0.262999, 0) == "NORMAL"
    assert core.determine_port_status(0.263, 0) == "ELEVATED"
    assert core.determine_port_status(0.304, 0) == "CONGESTED"
    assert core.determine_port_status(0.304, 1) == "ELEVATED"


@pytest.mark.parametrize(
    "response",
    [
        {**SAFE, "finding": "42 waiting vessels are recorded.", "evidence_fields": ["waiting_vessels"]},
        {**SAFE, "maritime_status": "ELEVATED"},
        {**SAFE, "finding": "Average speed is in knots."},
        {**SAFE, "finding": "Throughput covers the last four hours."},
        {**SAFE, "finding": "Waiting to Moored transitions stopped."},
        {**SAFE, "finding": "Approaching means arriving ships."},
        {**SAFE, "finding": "A collision incident occurred."},
        {**SAFE, "finding": "There are no vessels waiting."},
        {**SAFE, "finding": "Waiting caused congestion."},
        {**SAFE, "finding": "The port will be congested."},
        {**SAFE, "finding": "Recommend diverting vessels."},
        {**SAFE, "finding": "A subset of units are remaining stationary."},
        {**SAFE, "finding": "Observed activity indicates a subset of units remaining stationary relative to the total active population, suggesting potential holding patterns."},
        {**SAFE, "finding": "The waiting ratio suggests holding patterns."},
        {**SAFE, "finding": "The vessels are anchoring."},
        {**SAFE, "finding": "The berth queue is growing."},
        {**SAFE, "finding": "The waiting ratio shows delays."},
        {**SAFE, "finding": "Vessels are maneuvering around the port."},
        {**SAFE, "finding": "The activity indicates traffic flow changes."},
        {**SAFE, "finding": "The speed value shows vessel movement."},
        {**SAFE, "finding": "Approaching vessels show current activity."},
        {**SAFE, "finding": "The waiting ratio is 42%."},
        {**SAFE, "finding": "The waiting ratio is one quarter."},
        {**SAFE, "finding": "Two vessels are waiting."},
        {**SAFE, "finding": "Official port status is congested."},
        {**SAFE, "finding": "The forecast is congested."},
        {**SAFE, "possible_operational_contribution": "May assist planners in evaluating current resource allocation efficiency without implying future bottlenecks."},
        {**SAFE, "possible_operational_contribution": "The current Maritime evidence may indicate present operational delays."},
        {**SAFE, "possible_operational_contribution": "The current Maritime evidence may indicate present bottlenecks."},
        {"finding": "incomplete"},
    ],
)
def test_unsafe_gemini_falls_back(response):
    result = MaritimeInvestigationAgent(llm_provider=MockProvider(response)).investigate(
        request("2025-01-01T03:00:00Z")
    )
    assert result.reasoning_mode == "deterministic_fallback"
    assert result.maritime_status == "CONGESTED"
    assert len(result.findings) == 2


def test_valid_gemini_only_adds_narrative():
    baseline = MaritimeInvestigationAgent().investigate(request("2025-01-01T03:00:00Z"))
    result = MaritimeInvestigationAgent(llm_provider=MockProvider(SAFE)).investigate(
        request("2025-01-01T03:00:00Z")
    )
    assert result.reasoning_mode == "llm_assisted"
    assert result.evidence == baseline.evidence
    assert result.maritime_status == baseline.maritime_status
    assert result.limitations == baseline.limitations
    assert result.findings[-1].provenance == DataProvenance.AGENT_GENERATED


def test_grounded_contribution_remains_llm_assisted():
    grounded = {
        **SAFE,
        "finding": "The supplied waiting ratio is consistent with the current team assessment.",
        "possible_operational_contribution": (
            "The current Maritime evidence may contribute context when assessing "
            "present port-side operational pressure."
        ),
    }
    result = MaritimeInvestigationAgent(llm_provider=MockProvider(grounded)).investigate(
        request("2025-01-02T03:00:00Z")
    )
    assert result.reasoning_mode == "llm_assisted"
    assert result.findings[-1].evidence_fields == ("maritime_status", "waiting_ratio")


@pytest.mark.parametrize(
    ("hour", "label"),
    [
        ("2025-01-02T03:00:00Z", "normal"),
        ("2025-01-01T00:00:00Z", "elevated"),
        ("2025-01-01T03:00:00Z", "congested"),
    ],
)
def test_matching_deterministic_status_word_is_accepted(hour, label):
    narrative = {**SAFE, "finding": f"The current deterministic status is {label}."}
    result = MaritimeInvestigationAgent(llm_provider=MockProvider(narrative)).investigate(
        request(hour)
    )
    assert result.reasoning_mode == "llm_assisted"
    assert result.maritime_status.value == label.upper()


@pytest.mark.parametrize("finding", ["The port is congested.", "The forecast is normal.", "Official port status is normal."])
def test_mismatched_or_misrepresented_status_is_rejected(finding):
    result = MaritimeInvestigationAgent(llm_provider=MockProvider({**SAFE, "finding": finding})).investigate(
        request("2025-01-02T03:00:00Z")
    )
    assert result.reasoning_mode == "deterministic_fallback"


@pytest.mark.parametrize("ratio_text", ["0.25", "25%", "one quarter"])
def test_grounded_waiting_ratio_formats_are_accepted(ratio_text):
    narrative = {**SAFE, "finding": f"The supplied waiting ratio is {ratio_text}."}
    result = MaritimeInvestigationAgent(llm_provider=MockProvider(narrative)).investigate(
        request("2025-01-02T03:00:00Z")
    )
    assert result.reasoning_mode == "llm_assisted"


def test_ordinary_one_does_not_trigger_numeric_guardrail():
    narrative = {**SAFE, "finding": "This is one current team assessment of waiting evidence."}
    result = MaritimeInvestigationAgent(llm_provider=MockProvider(narrative)).investigate(
        request("2025-01-02T03:00:00Z")
    )
    assert result.reasoning_mode == "llm_assisted"


def test_exact_live_diagnostic_narrative_is_accepted():
    narrative = {
        "finding": "The current status is normal, supported by a waiting proportion of one quarter based on the recorded count.",
        "evidence_fields": ["maritime_status", "waiting_ratio"],
        "possible_operational_contribution": "Current maritime evidence provides context for assessing present port-side operational pressure.",
    }
    result = MaritimeInvestigationAgent(llm_provider=MockProvider(narrative)).investigate(
        request("2025-01-02T03:00:00Z")
    )
    assert result.reasoning_mode == "llm_assisted"
    assert result.maritime_status == "NORMAL"
    assert result.evidence.waiting_ratio == 0.25


def test_latest_live_diagnostic_counts_are_accepted_and_cited():
    narrative = {
        "finding": (
            "The current team heuristic status is NORMAL, supported by a waiting ratio "
            "of 0.25 derived from 10 waiting vessels out of 40 total vessels."
        ),
        "evidence_fields": ["maritime_status", "waiting_ratio", "waiting_vessels", "vessels"],
        "possible_operational_contribution": (
            "Current Maritime evidence provides context for assessing present "
            "port-side operational pressure."
        ),
    }
    result = MaritimeInvestigationAgent(llm_provider=MockProvider(narrative)).investigate(
        request("2025-01-02T03:00:00Z")
    )
    assert result.reasoning_mode == "llm_assisted"
    assert result.findings[-1].evidence_fields == tuple(narrative["evidence_fields"])
    assert result.evidence.vessels == 40
    assert result.evidence.waiting_vessels == 10


@pytest.mark.parametrize(
    ("finding", "fields"),
    [
        ("10 waiting vessels are recorded.", ["maritime_status", "waiting_ratio"]),
        ("40 total vessels are recorded.", ["maritime_status", "waiting_ratio"]),
        ("20 waiting vessels are recorded.", ["waiting_vessels"]),
        ("41 total vessels are recorded.", ["vessels"]),
        ("40 waiting vessels are recorded.", ["vessels"]),
        ("40 vessels are waiting.", ["vessels"]),
        ("10 total vessels are recorded.", ["waiting_vessels"]),
        ("The waiting ratio is 0.25.", ["maritime_status"]),
        ("The waiting ratio is 25%.", ["maritime_status"]),
        ("The waiting ratio is 30%.", ["waiting_ratio"]),
        ("The current status is NORMAL.", ["waiting_ratio"]),
        ("Capacity is 40 vessels.", ["vessels"]),
        ("0.25 vessels are recorded.", ["waiting_ratio", "vessels"]),
    ],
)
def test_numeric_or_status_claim_requires_exact_cited_evidence(finding, fields):
    narrative = {**SAFE, "finding": finding, "evidence_fields": fields}
    result = MaritimeInvestigationAgent(llm_provider=MockProvider(narrative)).investigate(
        request("2025-01-02T03:00:00Z")
    )
    assert result.reasoning_mode == "deterministic_fallback"


@pytest.mark.parametrize(
    ("finding", "fields"),
    [
        ("10 waiting vessels are recorded.", ["waiting_vessels"]),
        ("40 total vessels are recorded.", ["vessels"]),
        ("The waiting ratio is 0.25.", ["waiting_ratio"]),
    ],
)
def test_exact_cited_counts_and_ratio_are_accepted(finding, fields):
    narrative = {**SAFE, "finding": finding, "evidence_fields": fields}
    result = MaritimeInvestigationAgent(llm_provider=MockProvider(narrative)).investigate(
        request("2025-01-02T03:00:00Z")
    )
    assert result.reasoning_mode == "llm_assisted"
    assert result.findings[-1].evidence_fields == tuple(fields)


@pytest.mark.parametrize("provider", [MockProvider(error=RuntimeError("down")), MockProvider("bad json")])
def test_provider_failure_falls_back(provider):
    result = MaritimeInvestigationAgent(llm_provider=provider).investigate(
        request("2025-01-01T00:00:00Z")
    )
    assert result.reasoning_mode == "deterministic_fallback"
