"""Unified website comparisons use one frozen Hybrid context and unchanged physics."""
from copy import deepcopy

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marsa.api.main import app
from marsa.api.routes import unified_twin as route
from marsa.config import load_config
from marsa.twin import unified
from marsa.twin.coupled import CoupledScenario, simulate_coupled

TS = "2025-07-10T19:00:00+00:00"


def scenario(slots=1, inventory=10000, gate=0):
    return CoupledScenario(timestamp_utc=TS, horizon_hours=4, cohort_id="two-fixed-vessels",
        resource_pool_id="abstract-test-pool", parameters={
            "berth_slots": slots, "yard_capacity": 10000, "initial_inventory": inventory,
            "unloading_units_per_hour": 100, "gate_units_per_hour": gate},
        vessels=[{"simulation_id": key, "arrival_timestamp": TS, "initial_import_workload": 100,
                  "estimated_workload_for_priority": 100} for key in ("A", "B")])


def test_full_cargo_to_vessel_chain_and_fair_native_candidates():
    base = scenario()
    before = base.model_dump(mode="json")
    blocked = simulate_coupled(base)
    opened = simulate_coupled(unified.changed_scenario(base, [{"type": "gate_rate", "value": 100}]))
    assert (blocked["kpis"]["cargo_unloaded"], blocked["kpis"]["completed_vessels"],
            blocked["kpis"][unified.OBJECTIVE]) == (0, 0, 4)
    assert blocked["vessels"][0]["remaining_import_workload"] == 100
    assert blocked["vessels"][1]["berth_start_hour"] is None
    assert [v["completion_hour"] for v in opened["vessels"]] == [1, 2]
    assert opened["vessels"][1]["berth_start_hour"] == 1
    assert (opened["kpis"]["cargo_unloaded"], opened["kpis"]["completed_vessels"],
            opened["kpis"][unified.OBJECTIVE]) == (200, 2, 1)
    assert opened["kpis"]["system_conservation_residual"] == 0
    assert base.model_dump(mode="json") == before


def test_all_candidates_implemented_share_work_and_hold_other_assumptions():
    base = scenario(slots=2, inventory=0, gate=100)
    rows, excluded = unified.compare_candidates(base, unified.UnifiedSettings())
    assert not excluded
    assert {r["candidate_id"] for r in rows} == {"baseline", "estimated_shortest", "slot_plus_one", "gate_modest", "gate_large"}
    assert rows[0]["candidate_id"] == "baseline"  # all queue exposures tie
    assert len({r["workload_signature"] for r in rows}) == 1
    for row in rows:
        s = row["scenario"]
        assert s["vessels"] == base.model_dump(mode="json")["vessels"]
        assert (s["timestamp_utc"], s["horizon_hours"]) == (base.model_dump(mode="json")["timestamp_utc"], 4)
        for key in ["initial_inventory", "yard_capacity", "unloading_units_per_hour", "release_delay_hours"]:
            assert s["parameters"][key] == base.parameters.model_dump()[key]
        assert row["ranking_kpi"] == unified.OBJECTIVE
        unified.validate_result(row, CoupledScenario.model_validate(s))


def test_invalid_candidate_is_excluded_without_approximation(monkeypatch):
    original = unified.candidate_definitions
    monkeypatch.setattr(unified, "candidate_definitions", lambda b, s: original(b, s) + [
        {"candidate_id": "invented_crane", "title": "unsupported", "actions": [{"type": "crane", "value": 1}]}])
    rows, excluded = unified.compare_candidates(scenario(), unified.UnifiedSettings())
    assert "invented_crane" not in {r["candidate_id"] for r in rows}
    assert any(r["candidate_id"] == "invented_crane" for r in excluded)
    assert {r["candidate_id"] for r in excluded} >= {"gate_modest", "gate_large"}  # zero-rate duplicates


@pytest.fixture
def frozen(monkeypatch):
    calls = []
    domains = {"maritime": {"maritime_status": "ELEVATED"}, "cargo": {"cargo_status": "ELEVATED"},
               "events_weather": {"contextual_pressure": {"level": "low"}}}
    monkeypatch.setattr(unified, "collect_evidence", lambda *a, **k: (
        pd.Timestamp(TS), {}, {"status": "ml_unavailable"}, deepcopy(domains), {"mode": "deterministic_fallback"}))

    def build(*a, **k):
        calls.append(1)
        return scenario(), {"workload_estimation": {"settings": {"method": "constant_per_call_v1", "full_import_units": 100,
            "berthed_remaining_fraction": .5, "remaining_fraction_sensitivity": [.25, .5, .75]}}, "data_quality_warnings": []}

    monkeypatch.setattr(unified, "build_hybrid", build)
    return load_config(), unified.ContextStore(), calls


def test_frozen_context_reused_without_reloading_evidence_or_work(frozen, monkeypatch):
    cfg, store, calls = frozen
    first = unified.unified_analysis(TS, cfg, store=store)
    cfg["unified_hybrid"]["parameters"]["initial_inventory"] = 0
    monkeypatch.setattr(unified, "collect_evidence", lambda *a, **k: pytest.fail("evidence reloaded"))
    second = unified.unified_analysis(TS, cfg, context_id=first["context_id"],
        user_actions=[{"type": "gate_rate", "value": 100}], store=store)
    assert len(calls) == 1
    assert first["context_signature"] == second["context_signature"]
    def baseline(out):
        return next(r for r in out["digital_twin"]["results"] if r["candidate_id"] == "baseline")
    assert baseline(first)["kpis"] == baseline(second)["kpis"]
    user = next(r for r in second["digital_twin"]["results"] if r["candidate_id"] == "user_decision")
    assert user["kpis"]["completed_vessels"] == 2
    assert second["decision_support"]["highest_ranked_candidate"] != "user_decision"
    with pytest.raises(ValueError, match="exact frozen"):
        unified.unified_analysis("2025-07-11T19:00:00+00:00", cfg, context_id=first["context_id"], store=store)


def test_queue_relief_is_not_described_as_completion_recovery(frozen):
    cfg, store, _ = frozen
    out = unified.unified_analysis(TS, cfg, store=store)
    assert out["decision_support"]["highest_ranked_candidate"] == "slot_plus_one"
    row = next(r for r in out["strategy_advisory"]["comparisons"] if r["candidate_id"] == "slot_plus_one")
    assert row["assessment"] == "QUEUE_RELIEF_ONLY"
    assert "لم يتحسن الإكمال أو التفريغ" in out["decision_support"]["summary"]
    assert out["strategy_advisory"]["robustness"] == "NOT_EVALUATED"


@pytest.mark.parametrize("valid", [True, False])
def test_gemini_narrative_guard_and_provider_mutation_isolation(frozen, valid):
    cfg, store, _ = frozen

    class Provider:
        def generate_structured(self, **kwargs):
            payload = kwargs["payload"]
            payload["ranking"][0]["kpis"]["cargo_unloaded"] = 999999
            return {"summary": payload["allowed_narratives"][0]["summary"] if valid else "Choose gate_max; observed improvement."}

    out = unified.unified_analysis(TS, cfg, store=store, llm=Provider())
    assert out["decision_support"]["reasoning_mode"] == ("llm_assisted" if valid else "deterministic_fallback")
    assert all(r["kpis"]["cargo_unloaded"] != 999999 for r in out["digital_twin"]["results"])
    assert "observed improvement" not in out["decision_support"]["summary"]


def test_provider_failure_uses_deterministic_summary(frozen):
    cfg, store, _ = frozen

    class Failure:
        def generate_structured(self, **kwargs):
            raise RuntimeError("provider offline")

    out = unified.unified_analysis(TS, cfg, store=store, llm=Failure())
    assert out["decision_support"]["reasoning_mode"] == "deterministic_fallback"


def test_explicit_sensitivity_uses_frozen_population(frozen, monkeypatch):
    cfg, store, calls = frozen
    out = unified.unified_analysis(TS, cfg, store=store)
    monkeypatch.setattr(unified, "build_hybrid", lambda *a, **k: pytest.fail("AIS rebuilt"))
    report = unified.unified_sensitivity(out["context_id"], store=store)
    assert report["context_signature"] == out["context_signature"]
    assert report["run_count"] > 0 and len(calls) == 1
    for context in report["contexts"]:
        assert all(r["kpis"]["system_conservation_residual"] == 0 for r in context["comparisons"])


def test_malformed_internal_result_cannot_reach_advisory(frozen, monkeypatch):
    cfg, store, _ = frozen
    simulate = unified.simulate_coupled

    def corrupt(s):
        result = simulate(s)
        result["kpis"].pop("system_conservation_residual")
        return result

    monkeypatch.setattr(unified, "simulate_coupled", corrupt)
    with pytest.raises(ArithmeticError, match="conservation"):
        unified.unified_analysis(TS, cfg, store=store)


@pytest.mark.parametrize("action", [
    {"type": "gate_capacity", "value": "max_recorded"}, {"type": "queue_policy", "value": "shortest_service_first"},
    {"type": "set_slots", "value": True}, {"type": "set_slots", "value": 1.5},
    {"type": "gate_rate", "value": -1}, {"type": "gate_rate", "value": "100"},
])
def test_invalid_or_legacy_actions_rejected(action):
    with pytest.raises(ValueError):
        unified.changed_scenario(scenario(), [action])


def test_expired_context_has_no_synthetic_or_legacy_substitution():
    store = unified.ContextStore(ttl_seconds=-1)
    key = store.put({"scenario": "test"})
    with pytest.raises(unified.ContextUnavailable):
        store.get(key)


@pytest.mark.parametrize("timestamp", ["2025-03-04T10:00:00Z", "2025-07-10T19:00:00Z", "2025-11-05T12:00:00Z"])
def test_historical_unified_api_never_calls_legacy_simulation(timestamp, monkeypatch):
    monkeypatch.setattr(route, "get_strategy_provider", lambda: None)
    monkeypatch.setattr("marsa.pipeline.simulate_candidate", lambda *a, **k: pytest.fail("legacy simulated"))
    with TestClient(app) as client:
        response = client.post("/api/decision-support/unified", json={"timestamp_utc": timestamp})
    assert response.status_code == 200, response.text
    out = response.json()
    assert out["digital_twin"]["mode"] == "unified_hybrid"
    assert out["simulation_metadata"]["hindsight_assisted"]
    assert out["simulation_metadata"]["initial_inventory"]["value"] == 0
    assert len({r["workload_signature"] for r in out["digital_twin"]["results"]}) == 1
    assert out["decision_support"]["highest_ranked_candidate"] == "baseline"
    assert out["strategy_advisory"]["context_signature"] == out["context_signature"]


def test_api_errors_feature_gate_context_and_source(monkeypatch):
    cfg = load_config()
    monkeypatch.setattr(route, "load_config", lambda: cfg)
    monkeypatch.setattr(route, "get_strategy_provider", lambda: None)
    with TestClient(app) as client:
        assert client.post("/api/decision-support/unified", json={"timestamp_utc": TS, "context_id": "f" * 32}).status_code == 410
        assert client.post("/api/decision-support/unified", json={"timestamp_utc": "2026-01-01T00:00:00Z"}).status_code == 422
        cfg["unified_hybrid"]["enabled"] = False
        assert client.post("/api/decision-support/unified", json={"timestamp_utc": TS}).status_code == 403
        cfg["unified_hybrid"]["enabled"] = True
        monkeypatch.setattr(unified, "build_hybrid", lambda *a, **k: (_ for _ in ()).throw(FileNotFoundError()))
        response = client.post("/api/decision-support/unified", json={"timestamp_utc": TS})
        assert response.status_code == 503 and response.json()["detail"]["code"] == "SOURCE_UNAVAILABLE"


def test_initial_berth_occupancy_cannot_be_dropped(monkeypatch):
    monkeypatch.setattr(route, "get_strategy_provider", lambda: None)
    params = scenario().parameters.model_dump()
    with TestClient(app) as client:
        response = client.post("/api/decision-support/unified", json={"timestamp_utc": TS, "parameters": params})
    assert response.status_code == 422
    assert "berth_slots" in response.json()["detail"]


def test_canonical_website_is_only_the_selected_root_file():
    from marsa.config import ROOT

    expected = (ROOT / "Marsa updated.html").read_bytes()
    with TestClient(app) as client:
        for path in ["/", "/marsa.html", "/Marsa%20updated.html"]:
            response = client.get(path)
            assert response.status_code == 200 and response.content == expected
            assert response.headers["cache-control"] == "no-store"
        assert client.get("/.env").status_code == 404
        assert client.get("/config/port_config.yaml").status_code == 404


def test_context_cache_is_bounded_and_cannot_be_mutated_by_readers():
    store = unified.ContextStore(max_entries=1)
    old = store.put({"value": [1]})
    snapshot = store.get(old)
    snapshot["value"].append(999)
    assert store.get(old) == {"value": [1]}
    store.put({"value": [2]})
    with pytest.raises(unified.ContextUnavailable):
        store.get(old)
