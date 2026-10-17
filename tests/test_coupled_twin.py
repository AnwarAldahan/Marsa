"""Conservation, congestion feedback and isolation from the legacy decision flow."""
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from marsa.api.main import app
from marsa.api.routes import coupled_twin
from marsa.config import load_config
from marsa.pipeline import run
from marsa.twin.coupled import (
    CoupledParameters,
    CoupledScenario,
    CoupledSettings,
    simulate_coupled,
)

START = datetime(2025, 7, 10, 19, tzinfo=UTC)


def test_local_features_enabled_with_valid_contracts_and_safe_schema_defaults():
    from marsa.agents.candidate_selection import AdaptiveCandidateSettings
    from marsa.agents.investigation import InvestigationSettings
    from marsa.agents.strategy_advisory import AdvisorySettings
    from marsa.twin.hybrid import HybridSettings
    from marsa.twin.sensitivity import SensitivitySettings

    cfg = load_config()
    contracts = {
        "strategy_investigation": InvestigationSettings,
        "adaptive_candidates": AdaptiveCandidateSettings,
        "coupled_normalized": CoupledSettings,
        "hybrid_normalized": HybridSettings,
        "sensitivity_analysis": SensitivitySettings,
        "strategy_decision_support": AdvisorySettings,
    }
    for key, contract in contracts.items():
        assert cfg[key]["enabled"] is True
        assert contract.model_validate(cfg[key]).enabled
        omitted_flag = {name: value for name, value in cfg[key].items() if name != "enabled"}
        assert not contract.model_validate(omitted_flag).enabled
    assert cfg["strategy_investigation"]["use_gemini"] is True


def scenario(work=(5, 5), arrivals=None, horizon=4, **parameters):
    base = dict(yard_capacity=10, initial_inventory=10, berth_slots=1,
                unloading_units_per_hour=5, gate_units_per_hour=0)
    base.update(parameters)
    arrivals = [0] * len(work) if arrivals is None else arrivals
    return CoupledScenario(
        timestamp_utc=START, horizon_hours=horizon, cohort_id="explicit-test-cohort",
        resource_pool_id="abstract-test-pool", parameters=base,
        vessels=[dict(simulation_id=f"V{i}", call_id=f"TEST{i}",
                      arrival_timestamp=START + timedelta(hours=t), initial_import_workload=w)
                 for i, (w, t) in enumerate(zip(work, arrivals, strict=True))],
    )


def test_full_yard_vs_clearance_acceptance():
    a = scenario(work=(100, 100), yard_capacity=10000, initial_inventory=10000,
                 unloading_units_per_hour=100)
    b = a.model_copy(update={"parameters": a.parameters.model_copy(update={"gate_units_per_hour": 100.0})})
    blocked, clear = simulate_coupled(a), simulate_coupled(b)
    assert blocked["kpis"]["cargo_unloaded"] == 0
    assert blocked["kpis"]["completed_vessels"] == 0
    assert blocked["kpis"]["queue_exposure_vessel_hours"] == 4
    assert blocked["kpis"]["blocked_service_vessel_hours"] == 4
    assert clear["kpis"]["cargo_unloaded"] == 200
    assert clear["kpis"]["completed_vessels"] == 2
    assert clear["kpis"]["queue_exposure_vessel_hours"] == 1
    assert [v["completion_hour"] for v in clear["vessels"]] == [1, 2]
    assert [v["berth_start_hour"] for v in clear["vessels"]] == [0, 1]
    assert clear["kpis"]["cargo_cleared"] == 400
    assert clear["kpis"]["yard_inventory_end"] == 9800
    assert clear["kpis"]["system_conservation_residual"] == 0


def test_zero_productivity_no_gate_benefit():
    for gate in [0, 10]:
        r = simulate_coupled(scenario(unloading_units_per_hour=0, gate_units_per_hour=gate))
        assert r["kpis"]["cargo_unloaded"] == 0
        assert r["kpis"]["completed_vessels"] == 0
        assert r["kpis"]["queue_exposure_vessel_hours"] == 4


def test_nearly_full_yard_retains_blocked_work():
    r = simulate_coupled(scenario(work=(5,), initial_inventory=9, horizon=1))
    assert r["flows"][0]["unloaded"] == 1
    assert r["kpis"]["yard_blocked_unloading"] == 4
    assert r["vessels"][0]["remaining_import_workload"] == 4
    assert r["kpis"]["yard_inventory_end"] == 10


def test_larger_work_takes_longer_and_queue_is_distinct_from_completion():
    a = simulate_coupled(scenario(work=(5,), initial_inventory=0))
    b = simulate_coupled(scenario(work=(10,), initial_inventory=0))
    assert a["vessels"][0]["completion_hour"] == 1
    assert b["vessels"][0]["completion_hour"] == 2
    r = simulate_coupled(scenario(work=(10, 5), initial_inventory=0, gate_units_per_hour=5))
    assert [v["berth_start_hour"] for v in r["vessels"]] == [0, 2]
    assert [v["completion_hour"] for v in r["vessels"]] == [2, 3]
    assert r["kpis"]["queue_exposure_vessel_hours"] == 2
    assert r["kpis"]["occupied_berth_hours"] == 3
    assert r["kpis"]["berth_utilization"] == .75
    assert r["kpis"]["avg_completed_queue_wait_hours"] == 1


def test_fair_shared_productivity_redistributes_small_work():
    r = simulate_coupled(scenario(work=(1, 20, 20), initial_inventory=0,
                                 berth_slots=3, horizon=1, unloading_units_per_hour=9))
    assert [v["remaining_import_workload"] for v in r["vessels"]] == [0, 16, 16]
    assert r["flows"][0]["unloaded"] == 9
    assert r["kpis"]["completed_vessels"] == 1


@pytest.mark.parametrize("capacity", [9090.91, 10000, 10909.09])
def test_storage_sensitivity_keeps_units_and_conservation(capacity):
    r = simulate_coupled(scenario(work=(500, 800), yard_capacity=capacity,
                                 initial_inventory=capacity - 10,
                                 unloading_units_per_hour=100, gate_units_per_hour=25))
    assert all(0 <= point["yard_inventory"] <= capacity + 1e-9 for point in r["timeline"])
    assert all(f["unloaded"] <= 100 for f in r["flows"])
    assert r["kpis"]["system_conservation_residual"] == pytest.approx(0)
    assert r["units"] == "normalized_units"


def test_every_transfer_and_initial_stock_accounted_once():
    s = scenario(work=(.7, 2.3, 8), initial_inventory=4.1, berth_slots=2,
                 unloading_units_per_hour=1.3, gate_units_per_hour=.8, horizon=20)
    before = s.model_dump()
    r = simulate_coupled(s)
    assert r == simulate_coupled(s)
    assert s.model_dump() == before
    for f in r["flows"]:
        assert f["closing_inventory"] == pytest.approx(f["opening_inventory"] + f["unloaded"] - f["gate_cleared"])
        assert f["yard_conservation_residual"] == pytest.approx(0, abs=1e-10)
        assert f["vessel_conservation_residual"] == pytest.approx(0, abs=1e-10)
    assert r["kpis"]["cargo_unloaded"] == pytest.approx(11)
    assert r["kpis"]["cargo_cleared"] + r["kpis"]["yard_inventory_end"] == pytest.approx(15.1)
    assert sum(lot["quantity"] for lot in r["yard_lots"]) == pytest.approx(r["kpis"]["yard_inventory_end"])


def test_dwell_only_controls_eligibility_not_removal():
    r = simulate_coupled(scenario(work=(5,), initial_inventory=0, gate_units_per_hour=5,
                                 release_delay_hours=2, horizon=4))
    assert [f["gate_cleared"] for f in r["flows"]] == [0, 0, 0, 5]
    assert r["kpis"]["cargo_unloaded"] == r["kpis"]["cargo_cleared"] == 5
    assert r["kpis"]["yard_inventory_end"] == 0


def test_ineligible_initial_inventory_blocks_even_with_gate_capacity():
    r = simulate_coupled(scenario(initial_inventory_eligible_in_hours=10, gate_units_per_hour=100))
    assert r["kpis"]["cargo_cleared"] == r["kpis"]["cargo_unloaded"] == 0


def test_extra_gate_cannot_improve_service_when_yard_is_not_binding():
    a = simulate_coupled(scenario(work=(2, 3), initial_inventory=0))
    b = simulate_coupled(scenario(work=(2, 3), initial_inventory=0, gate_units_per_hour=100))
    assert [v["completion_hour"] for v in a["vessels"]] == [v["completion_hour"] for v in b["vessels"]]
    assert a["kpis"]["queue_exposure_vessel_hours"] == b["kpis"]["queue_exposure_vessel_hours"]
    assert b["kpis"]["cargo_cleared"] > a["kpis"]["cargo_cleared"]


def test_varied_resource_matrix_preserves_individual_work_and_stock():
    for slots in [0, 1, 3]:
        for rate in [0, .3, 7]:
            for gate in [0, .7, 100]:
                s = scenario(work=(.2, 3.1, 20), initial_inventory=9.8,
                             berth_slots=slots, unloading_units_per_hour=rate,
                             gate_units_per_hour=gate, horizon=12)
                r = simulate_coupled(s)
                assert all(0 <= v["remaining_import_workload"] <= v["initial_import_workload"] for v in r["vessels"])
                assert all(f["unloaded"] <= rate + 1e-9 and f["gate_cleared"] <= gate + 1e-9 for f in r["flows"])
                assert r["kpis"]["system_conservation_residual"] == pytest.approx(0, abs=1e-9)


def test_arrival_at_horizon_joins_queue_without_service_and_future_stays_future():
    r = simulate_coupled(scenario(work=(5, 5, 5), arrivals=[0, 4, 5], initial_inventory=0))
    assert [v["state"] for v in r["vessels"]] == ["completed", "waiting", "future"]
    assert r["vessels"][1]["berth_start_hour"] is None
    assert r["kpis"]["waiting_vessels"] == r["kpis"]["unfinished_vessels"] == 1
    assert r["kpis"]["future_vessels"] == 1
    assert r["kpis"]["cargo_remaining_aboard"] == 10
    assert r["kpis"]["unfinished_queue_waits"] == [{"simulation_id": "V1", "wait_hours": 0}]
    assert r["kpis"]["queue_exposure_vessel_hours"] == 0


def test_unfinished_queue_waits_are_not_reported_as_completed_waits():
    r = simulate_coupled(scenario())
    assert r["kpis"]["completed_queue_wait_count"] == 1  # V0 admitted, not service-completed
    assert r["kpis"]["avg_completed_queue_wait_hours"] == 0
    assert r["kpis"]["unfinished_queue_waits"] == [{"simulation_id": "V1", "wait_hours": 4}]
    assert r["kpis"]["unfinished_vessels"] == 2
    assert r["kpis"]["completed_vessels"] == 0


def test_zero_berths_and_empty_population_are_defined():
    r = simulate_coupled(scenario(berth_slots=0))
    assert r["kpis"]["berth_utilization"] is None
    assert r["kpis"]["avg_completed_queue_wait_hours"] is None
    assert r["kpis"]["queue_exposure_vessel_hours"] == 8
    empty = simulate_coupled(scenario(work=(), initial_inventory=0))
    assert empty["kpis"]["system_conservation_residual"] == 0


def test_shortest_policy_uses_declared_estimate_not_hidden_actual_work():
    raw = scenario(work=(1, 10), initial_inventory=0).model_dump()
    raw["queue_policy"] = "shortest_estimated_workload_first"
    for v, estimate in zip(raw["vessels"], [20, 2], strict=True):
        v["estimated_workload_for_priority"] = estimate
    r = simulate_coupled(CoupledScenario.model_validate(raw))
    assert r["vessels"][1]["berth_start_hour"] == 0
    assert r["vessels"][0]["berth_start_hour"] == 2


@pytest.mark.parametrize("change", [
    {"yard_capacity": 0}, {"initial_inventory": 11}, {"gate_units_per_hour": -1},
    {"unloading_units_per_hour": float("nan")}, {"gate_units_per_hour": float("inf")},
    {"berth_slots": 1.5}, {"berth_slots": True}, {"units": "TEU"},
    {"release_delay_hours": -1}, {"unknown": 123},
])
def test_bad_parameters_rejected(change):
    raw = scenario().parameters.model_dump()
    raw.update(change)
    with pytest.raises(ValidationError):
        CoupledParameters.model_validate(raw)


@pytest.mark.parametrize("problem", ["duplicate_id", "duplicate_call", "naive", "fractional", "past", "shortest_missing"])
def test_bad_vessel_scenarios_rejected(problem):
    raw = scenario().model_dump()
    if problem == "duplicate_id":
        raw["vessels"][1]["simulation_id"] = "V0"
    elif problem == "duplicate_call":
        raw["vessels"][1]["call_id"] = "TEST0"
    elif problem == "naive":
        raw["vessels"][0]["arrival_timestamp"] = START.replace(tzinfo=None)
    elif problem == "fractional":
        raw["vessels"][0]["arrival_timestamp"] += timedelta(minutes=30)
    elif problem == "past":
        raw["vessels"][0]["arrival_timestamp"] -= timedelta(hours=1)
    else:
        raw["queue_policy"] = "shortest_estimated_workload_first"
    with pytest.raises(ValidationError):
        CoupledScenario.model_validate(raw)


def test_endpoint_explicitly_disabled_and_old_configuration_compatible(monkeypatch):
    cfg = deepcopy(load_config())
    cfg["coupled_normalized"]["enabled"] = False
    assert not CoupledSettings.model_validate(cfg["coupled_normalized"]).enabled
    monkeypatch.setattr(coupled_twin, "load_config", lambda: cfg)
    body = scenario().model_dump(mode="json")
    with TestClient(app) as client:
        assert client.post("/api/twin/simulate-coupled", json=body).status_code == 403
        cfg.pop("coupled_normalized")
        monkeypatch.setattr(coupled_twin, "load_config", lambda: cfg)
        assert client.post("/api/twin/simulate-coupled", json=body).status_code == 403


def test_enabled_endpoint_explicit_mode_and_parameters_or_server_defaults(monkeypatch):
    cfg = deepcopy(load_config())
    cfg["coupled_normalized"]["enabled"] = True
    monkeypatch.setattr(coupled_twin, "load_config", lambda: cfg)
    body = scenario(gate_units_per_hour=5).model_dump(mode="json")
    with TestClient(app) as client:
        response = client.post("/api/twin/simulate-coupled", json=body)
        assert response.status_code == 200
        assert response.json() == simulate_coupled(CoupledScenario.model_validate(body))
        body.pop("parameters")
        defaults = client.post("/api/twin/simulate-coupled", json=body).json()
        assert defaults["scenario"]["parameters"] == CoupledSettings.model_validate(cfg["coupled_normalized"]).parameters.model_dump()
        body["mode"] = "historical_replay"
        assert client.post("/api/twin/simulate-coupled", json=body).status_code == 422
        legacy = client.post("/api/twin/simulate", json={"timestamp_utc": START.isoformat()})
        assert legacy.status_code == 200
        assert "mode" not in legacy.json()


def test_invalid_enabled_server_settings_and_overflow_fail_cleanly(monkeypatch):
    cfg = deepcopy(load_config())
    cfg["coupled_normalized"]["enabled"] = True
    monkeypatch.setattr(coupled_twin, "load_config", lambda: cfg)
    body = scenario().model_dump(mode="json")
    with TestClient(app) as client:
        body["parameters"]["unloading_units_per_hour"] = 1e308
        assert client.post("/api/twin/simulate-coupled", json=body).status_code == 422
        cfg["coupled_normalized"]["parameters"]["units"] = "TEU"
        assert client.post("/api/twin/simulate-coupled", json=body).status_code == 503


def test_feature_flag_never_changes_four_agent_pipeline_or_recommendations():
    cfg = deepcopy(load_config())
    cfg["strategy_investigation"]["use_gemini"] = False
    cfg["coupled_normalized"]["enabled"] = False
    before = run(START.isoformat(), cfg=cfg, save=False)
    cfg["coupled_normalized"]["enabled"] = True
    after = run(START.isoformat(), cfg=cfg, save=False)
    assert before == after
    assert len(after["architecture"]["agents"]) == 4
    assert "candidate_selection" in after
    assert "strategy_investigation" in after
