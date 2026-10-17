"""Analytical constraint fixtures, paired experiments and finite-range robustness."""
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from marsa.api.main import app
from marsa.api.routes import coupled_twin
from marsa.config import load_config
from marsa.twin.coupled import CoupledScenario, simulate_coupled
from marsa.twin.hybrid import HybridRequest, build_hybrid
from marsa.twin.sensitivity import (
    SensitivityPlan,
    analyze_sensitivity,
    diagnose_bottlenecks,
    summarize_robustness,
)

START = datetime(2025, 7, 10, 19, tzinfo=UTC)


def scenario(work=(100,), **updates):
    params = dict(yard_capacity=1000, initial_inventory=0, berth_slots=1,
                  unloading_units_per_hour=1, gate_units_per_hour=100)
    params.update(updates)
    return CoupledScenario(timestamp_utc=START, horizon_hours=4, cohort_id="explicit-fixture",
                           resource_pool_id="abstract-fixture", parameters=params,
                           vessels=[dict(simulation_id=f"V{i}", arrival_timestamp=START,
                                         initial_import_workload=w) for i, w in enumerate(work)])


def request(work=1000, **updates):
    raw = dict(timestamp_utc=START, horizon_hours=4, time_information_mode="as_of",
               cohort_id="snapshot-fixture", resource_pool_id="abstract-fixture",
               as_of_observations=[dict(call_id="B", vessel_id="IMO1", mmsi=101,
                                        available_at=START, state="berthed", arrival=START - timedelta(hours=2),
                                        berth_start=START - timedelta(hours=2), arrival_basis="berth_start_fallback"),
                                   dict(call_id="W", vessel_id="IMO2", mmsi=102,
                                        available_at=START, state="waiting", arrival=START - timedelta(hours=1),
                                        arrival_basis="matched_wait_start")],
               workload=dict(full_import_units=work),
               parameters=dict(yard_capacity=10000, initial_inventory=10000, berth_slots=1,
                               unloading_units_per_hour=1000, gate_units_per_hour=100))
    raw.update(updates)
    return HybridRequest.model_validate(raw)


@pytest.mark.parametrize("expected,work,parameters", [
    ("unloading_productivity_limitation", (100,), {}),
    ("yard_capacity_blocking", (100,), dict(yard_capacity=10, initial_inventory=10,
                                           initial_inventory_eligible_in_hours=10, gate_units_per_hour=100)),
    ("gate_clearance_limitation", (), dict(initial_inventory=100, gate_units_per_hour=1)),
    ("berth_slot_limitation", (1, 1, 1), dict(unloading_units_per_hour=100)),
    ("insufficient_demand", (), {}),
    ("insufficient_demand", (1,), dict(unloading_units_per_hour=100)),
])
def test_analytically_known_single_constraints(expected, work, parameters):
    result = simulate_coupled(scenario(work=work, **parameters))
    assert diagnose_bottlenecks(result)["classification"] == expected


def test_joint_zero_capacity_constraints_and_full_yard_are_not_zero_demand():
    result = simulate_coupled(scenario(yard_capacity=10, initial_inventory=10,
                                       gate_units_per_hour=0, unloading_units_per_hour=0))
    d = diagnose_bottlenecks(result)
    assert d["classification"] == "mixed"
    assert set(d["active_constraints"]) == {"yard_capacity_blocking", "gate_clearance_limitation", "unloading_productivity_limitation"}
    no_slots = simulate_coupled(scenario(berth_slots=0, unloading_units_per_hour=0))
    assert set(diagnose_bottlenecks(no_slots)["active_constraints"]) == {"berth_slot_limitation", "unloading_productivity_limitation"}


def test_yard_unloading_and_gate_are_distinguished_and_names_are_ignored():
    full = simulate_coupled(scenario(yard_capacity=10, initial_inventory=10, gate_units_per_hour=0))
    d = diagnose_bottlenecks(full)
    assert "yard_capacity_blocking" in d["active_constraints"]
    assert "unloading_productivity_limitation" not in d["active_constraints"]
    assert d["berth_queue_pressure_hours"] == 0
    assert d == diagnose_bottlenecks({**full, "intervention": "unloading_modest"})
    draining = simulate_coupled(scenario(yard_capacity=10, initial_inventory=10, gate_units_per_hour=100))
    assert diagnose_bottlenecks(draining)["classification"] == "unloading_productivity_limitation"


def test_eligible_stock_ledger_uses_only_released_cargo():
    result = simulate_coupled(scenario(work=(3,), unloading_units_per_hour=1,
                                       release_delay_hours=2, gate_units_per_hour=1))
    d = diagnose_bottlenecks(result)
    assert [e["eligible_yard_stock_open"] for e in d["hourly_evidence"]] == [0, 0, 0, 1]
    assert d["constraint_hours"]["gate_clearance_limitation"] == 0


def test_full_berths_are_not_automatically_an_independent_slot_bottleneck():
    result = simulate_coupled(scenario(work=(100, 100), initial_inventory=1000, gate_units_per_hour=0))
    d = diagnose_bottlenecks(result)
    assert d["berth_queue_pressure_hours"] == 4
    assert d["constraint_hours"]["berth_slot_limitation"] == 0


@pytest.mark.parametrize("change", [
    dict(occupancy_low=-.1), dict(remaining_high=1.1), dict(occupancy_low=.9, occupancy_high=.5),
    dict(storage_factors=(.9, 2, 3)), dict(rate_workload_factors=(0, 1, 2)),
    dict(modest_increase=.5), dict(large_gate_factor=float("nan")), dict(unknown=1),
])
def test_experiment_bounds_reject_invalid_assumptions(change):
    with pytest.raises(ValidationError):
        SensitivityPlan(**change)


def test_reproducible_bounded_oat_plan_pairing_and_conservation():
    req = request()
    before = req.model_dump()
    report = analyze_sensitivity(req)
    assert report == analyze_sensitivity(req)
    assert req.model_dump() == before
    assert 1 < len(report["contexts"]) <= 17
    assert report["run_count"] == 5 * len(report["contexts"])
    assert len({c["runtime_signature"] for c in report["contexts"]}) == len(report["contexts"])
    assert report["excluded_contexts"] == []
    assert report["ranges_low_base_high"]["yard_capacity"] == pytest.approx([9090.91, 10000, 10909.09])
    assert report["ranges_low_base_high"]["berth_slots"][0] == 1  # cannot reduce initial occupancy
    assert report["strategy_integration"] is False
    assert report["units"] == "normalized_units"
    for context in report["contexts"]:
        signatures = {r["workload_signature"] for r in context["comparisons"]}
        assert signatures == {context["workload_signature"]}
        rows = context["comparisons"]
        for row in rows:
            p, k = row["parameters"], row["kpis"]
            assert 0 <= p["initial_inventory"] <= p["yard_capacity"]
            assert abs(k["system_conservation_residual"]) < 1e-8
            assert k["yard_peak_occupancy_pct"] <= 100 + 1e-9
            assert p["units"] == "normalized_units"
        assert rows[1]["parameters"]["gate_units_per_hour"] == rows[0]["parameters"]["gate_units_per_hour"] * 1.2
        for row in rows[1:]:
            changed = {k for k, value in row["parameters"].items() if value != rows[0]["parameters"][k]}
            assert len(changed) <= 1


def test_base_matches_hybrid_and_storage_holds_occupancy_not_absolute_inventory():
    req = request()
    report = analyze_sensitivity(req)
    base, _ = build_hybrid(req)
    assert report["contexts"][0]["comparisons"][0]["kpis"] == simulate_coupled(base)["kpis"]
    for context in report["contexts"]:
        if context["axis"] == "yard_capacity":
            p = context["comparisons"][0]["parameters"]
            assert p["initial_inventory"] / p["yard_capacity"] == 1


def test_workload_and_remaining_fraction_changes_keep_calls_fixed():
    report = analyze_sensitivity(request())
    by_id = {c["context"]: c for c in report["contexts"]}
    assert by_id["full_import_units_low"]["workload_signature"] != by_id["base"]["workload_signature"]
    assert by_id["berthed_remaining_fraction_low"]["workload_signature"] != by_id["base"]["workload_signature"]
    assert by_id["unloading_units_per_hour_low"]["workload_signature"] == by_id["base"]["workload_signature"]
    assert [v["call_id"] for v in report["population"]] == ["B", "W"]
    assert by_id["full_import_units_low"]["comparisons"][0]["kpis"]["cargo_remaining_aboard"] != by_id["full_import_units_high"]["comparisons"][0]["kpis"]["cargo_remaining_aboard"]


def test_no_forced_gate_benefit_when_productivity_is_zero():
    raw = request().model_dump()
    raw["parameters"]["unloading_units_per_hour"] = 0
    report = analyze_sensitivity(HybridRequest.model_validate(raw))
    assert report["robustness"]["gate_modest"]["classification"] == "ineffective_under_tested_conditions"
    rows = report["contexts"][0]["comparisons"]
    assert rows[0]["kpis"]["queue_exposure_vessel_hours"] == rows[1]["kpis"]["queue_exposure_vessel_hours"]
    assert rows[1]["vs_baseline"]["cargo_unloaded_delta"] == 0


def test_as_of_analysis_inherits_finalized_source_isolation(monkeypatch):
    def forbidden():
        pytest.fail("analysis must preserve the hybrid as-of information boundary")
    monkeypatch.setattr("marsa.twin.hybrid.load_calls", forbidden)
    monkeypatch.setattr("marsa.twin.hybrid.load_cargo", forbidden)
    report = analyze_sensitivity(request())
    assert report["metadata"]["time_information_mode"] == "as_of"
    assert not report["metadata"]["hindsight_assisted"]
    assert report["metadata"]["source"]["future_arrivals"] == "UNSPECIFIED_NO_ARRIVAL_FEED"


def test_zero_vessel_demand_never_manufactures_a_delay_benefit():
    req = request(as_of_observations=[])
    report = analyze_sensitivity(req)
    for context in report["contexts"]:
        for row in context["comparisons"]:
            assert row["kpis"]["completed_vessels"] == 0
            assert row["kpis"]["queue_exposure_vessel_hours"] == 0
            if "vs_baseline" in row:
                assert row["vs_baseline"]["status"] == "ineffective"


@pytest.mark.parametrize("statuses,expected", [
    (["beneficial", "beneficial"], "robust_across_tested_assumptions"),
    (["beneficial", "ineffective"], "sensitive_to_specific_parameters"),
    (["ineffective", "ineffective"], "ineffective_under_tested_conditions"),
    (["beneficial", "tradeoff"], "inconclusive"),
    (["beneficial"], "inconclusive"),
])
def test_robustness_definitions_are_not_probabilities(statuses, expected):
    contexts = [dict(context=f"case{i}", axis="test", comparisons=[
        dict(intervention=key, vs_baseline=dict(status=status))
        for key in ["gate_modest", "gate_large", "unloading_modest", "berth_plus_one"]])
        for i, status in enumerate(statuses)]
    result = summarize_robustness(contexts, [])
    assert all(r["classification"] == expected for r in result.values())
    excluded = summarize_robustness(contexts, [dict(context="bad", reason="invalid")])
    assert all(r["classification"] == "inconclusive" for r in excluded.values())


def test_explicitly_disabled_independent_endpoint_and_existing_routes(monkeypatch):
    cfg = deepcopy(load_config())
    cfg["sensitivity_analysis"]["enabled"] = False
    cfg["hybrid_normalized"]["enabled"] = False
    monkeypatch.setattr(coupled_twin, "load_config", lambda: cfg)
    body = dict(scenario=request().model_dump(mode="json"), plan=dict(include_interactions=False))
    with TestClient(app) as client:
        assert client.post("/api/twin/analyze-sensitivity", json=body).status_code == 403
        cfg["sensitivity_analysis"]["enabled"] = True
        response = client.post("/api/twin/analyze-sensitivity", json=body)
        assert response.status_code == 200
        assert response.json()["strategy_integration"] is False
        # Existing endpoints remain independently disabled, not enabled by analysis.
        assert client.post("/api/twin/simulate-hybrid", json=body["scenario"]).status_code == 403
        assert client.post("/api/twin/simulate", json={"timestamp_utc": START.isoformat()}).status_code == 200
        cfg["sensitivity_analysis"]["plan"]["modest_increase"] = 10
        assert client.post("/api/twin/analyze-sensitivity", json=body).status_code == 503
        cfg.pop("sensitivity_analysis")
        assert client.post("/api/twin/analyze-sensitivity", json=body).status_code == 403


def test_two_historical_experiments_without_forced_delay_improvements():
    from scripts.demo_sensitivity_twin import historical_sensitivity_demo

    cases = historical_sensitivity_demo()
    assert [c["initial_counts"]["eligible_calls"] for c in cases.values()] == [20, 23]
    assert [c["run_count"] for c in cases.values()] == [80, 70]
    assert cases["B"]["skipped_duplicate_contexts"] == [dict(context="high_occupancy_low_gate", duplicate_of="gate_units_per_hour_low", axis="interaction")]
    for case in cases.values():
        rows = {r["intervention"]: r for r in case["base_comparisons"]}
        assert rows["baseline"]["kpis"]["queue_exposure_vessel_hours"] == rows["gate_modest"]["kpis"]["queue_exposure_vessel_hours"]
        assert rows["gate_modest"]["vs_baseline"]["status"] == "ineffective"
        assert rows["gate_modest"]["vs_baseline"]["cargo_progress_status"] == "improved"
        assert rows["unloading_modest"]["vs_baseline"]["status"] == "ineffective"
        assert all(abs(r["kpis"]["system_conservation_residual"]) < 1e-8 for r in rows.values())
    assert cases["A"]["base_comparisons"][2]["vs_baseline"]["status"] == "beneficial"
    assert cases["B"]["base_comparisons"][2]["vs_baseline"]["status"] == "ineffective"
