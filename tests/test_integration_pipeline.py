from copy import deepcopy
from pathlib import Path

import pytest
from fastapi.testclient import TestClient

from marsa.agents.strategy_agent import SUPPORTED_ACTIONS, generate_candidates, synthesize
from marsa.api.main import app
from marsa.data.load import load_dataset, snapshot
from marsa.pipeline import run


def test_exact_hour_pipeline_and_provenance():
    result = run("2025-01-01T00:00:00Z")
    assert result["authoritative_timestamp"] == "2025-01-01T00:00:00+00:00"
    assert result["architecture"]["agents"] == ["maritime", "cargo", "events_weather", "strategy"]
    assert result["architecture"]["orchestrator_agent"] is False
    assert result["ml_forecast"]["status"] == "ml_unavailable"
    assert result["digital_twin"]["provenance"] == "SIMULATED"
    assert all(item["provenance"] == "SIMULATED" for item in result["digital_twin"]["results"])
    assert all(candidate["provenance"] == "AGENT_GENERATED" for candidate in result["strategy_candidates"])
    assert all(
        action["type"] in SUPPORTED_ACTIONS
        for candidate in result["strategy_candidates"]
        for action in candidate["actions"]
    )
    assert result["human_approval_required"] is True
    assert result["decision_support"]["autonomous_execution"] is False
    assert result["decision_support"]["reasoning_mode"] == "deterministic"


@pytest.mark.parametrize("timestamp", [
    "2025-01-01T00:00:00", "2025-01-01T00:30:00Z",
    "2024-12-31T23:00:00Z", "2025-01-01T03:00:00+03:00",
])
def test_invalid_starting_timestamp_rejected(timestamp):
    with pytest.raises(ValueError):
        run(timestamp)


def test_missing_authoritative_hour_rejected():
    with pytest.raises(KeyError):
        run("2025-12-05T04:00:00Z")


def test_snapshot_no_longer_floors():
    with pytest.raises(ValueError):
        snapshot(load_dataset(), "2025-01-01T00:10:00Z")


def test_strategy_accepts_conflicting_signals_without_mutation():
    forecast = {"status": "supplied_test_context", "classification": "HIGH", "test_only": True}
    maritime = {"maritime_status": "NORMAL", "evidence": {"waiting_vessels": 3}}
    cargo = {"cargo_status": "CRITICAL", "evidence": {"yard_occupancy_percent": 95, "gate_throughput_last_1h": 2}}
    weather = {"contextual_pressure": {"level": "low"}}
    before = deepcopy((forecast, maritime, cargo, weather))
    candidates = generate_candidates(forecast, maritime, cargo, weather)
    synthesis = synthesize(forecast, maritime, cargo, weather)
    assert (forecast, maritime, cargo, weather) == before
    assert synthesis["signals"] == {
        "ml": {
            "status": "supplied_test_context",
            "classification": "HIGH",
            "congestion_score": None,
            "is_real_model_prediction": False,
        },
        "maritime": "NORMAL",
        "cargo": "CRITICAL",
        "events_weather": "low",
    }
    assert {action["type"] for candidate in candidates for action in candidate["actions"]} <= SUPPORTED_ACTIONS
    assert any(candidate["candidate_id"] == "gate_extension" for candidate in candidates)


def test_fixed_seed_scoring_is_deterministic():
    first = run("2025-01-01T00:00:00Z")["deterministic_evaluation"]
    second = run("2025-01-01T00:00:00Z")["deterministic_evaluation"]
    assert first == second


def test_real_ml_signal_reaches_strategy_without_overwriting_domain_agents():
    result = run("2025-01-02T08:00:00Z")
    forecast = result["ml_forecast"]
    assert forecast["status"] == "success"
    assert forecast["congestion_score"] == pytest.approx(0.008864540606737137)
    assert forecast["classification"] == "LOW"
    assert forecast["provenance"] == "PREDICTED"
    assert forecast["is_real_model_prediction"] is True
    assert "probability" not in forecast
    assert result["strategy_synthesis"]["signals"]["ml"] == {
        "status": "success",
        "classification": "LOW",
        "congestion_score": pytest.approx(0.008864540606737137),
        "is_real_model_prediction": True,
    }
    assert set(result["domain_agents"]) == {"maritime", "cargo", "events_weather"}
    assert result["digital_twin"]["provenance"] == "SIMULATED"
    assert result["human_approval_required"] is True
    assert result["decision_support"]["autonomous_execution"] is False


def test_no_active_orchestrator_module():
    assert not Path("src/marsa/agents/orchestrator.py").exists()


def test_integrated_api_success_and_timestamp_errors():
    with TestClient(app) as client:
        ok = client.post("/api/decision-support/analyze", json={"timestamp_utc": "2025-01-01T00:00:00Z"})
        assert ok.status_code == 200
        assert ok.json()["human_approval_required"] is True
        missing = client.post("/api/decision-support/analyze", json={"timestamp_utc": "2025-12-05T04:00:00Z"})
        assert missing.status_code == 404
        naive = client.post("/api/decision-support/analyze", json={"timestamp_utc": "2025-01-01T00:00:00"})
        assert naive.status_code == 422
        non_hour = client.post("/api/decision-support/analyze", json={"timestamp_utc": "2025-01-01T00:15:00Z"})
        assert non_hour.status_code == 422
        outside = client.post("/api/decision-support/analyze", json={"timestamp_utc": "2024-01-01T00:00:00Z"})
        assert outside.status_code == 422
        offset = client.post("/api/decision-support/analyze", json={"timestamp_utc": "2025-01-01T03:00:00+03:00"})
        assert offset.status_code == 422
