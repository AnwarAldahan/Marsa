"""Hybrid population/state initialization, information boundaries and regression."""
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pandas as pd
import pytest
from fastapi.testclient import TestClient
from pydantic import ValidationError

from marsa.api.main import app
from marsa.api.routes import coupled_twin
from marsa.common.exceptions import DataNotReadyError
from marsa.config import load_config
from marsa.twin.coupled import CoupledScenario, simulate_coupled
from marsa.twin.hybrid import HybridRequest, build_hybrid, simulate_hybrid, validate_calls

START = datetime(2025, 7, 10, 19, tzinfo=UTC)


def at(hour):
    return START + timedelta(hours=hour)


def table():
    records = []
    for call, vessel, arrival, berth, end, waited in [
        ("OLD", "IMO0", -20, -20, -1, False),
        ("B", "IMO1", -5, -5, 100, False),
        ("W", "IMO2", -2, 8, 20, True),
        ("F", "IMO3", 1, 1, 30, False),
        ("H", "IMO4", 4, 5, 25, True),
        ("AFTER", "IMO5", 5, 5, 30, False),
    ]:
        records.append(dict(call_id=call, vessel_id=vessel, mmsi=100 + len(records), imo=vessel,
                            length_m=200.0, arrival=at(arrival), waited=waited,
                            wait_start=at(arrival) if waited else pd.NaT,
                            wait_end=at(berth) if waited else pd.NaT,
                            berth_start=at(berth), berth_end=at(end),
                            wait_hours=berth - arrival, berth_hours=end - berth))
    return pd.DataFrame(records)


def request(**changes):
    raw = dict(timestamp_utc=START, horizon_hours=4, time_information_mode="retrospective",
               cohort_id="fixture-cargo-stays", resource_pool_id="abstract-fixture-pool",
               parameters=dict(yard_capacity=10000, initial_inventory=0, berth_slots=1,
                               unloading_units_per_hour=1000, gate_units_per_hour=100),
               intervention_gate_units_per_hour=1000)
    raw.update(changes)
    return HybridRequest.model_validate(raw)


def observations():
    return [dict(call_id="B", vessel_id="IMO1", mmsi=101, available_at=START,
                 state="berthed", arrival=at(-5), arrival_basis="berth_start_fallback", berth_start=at(-5)),
            dict(call_id="W", vessel_id="IMO2", mmsi=102, available_at=START,
                 state="waiting", arrival=at(-2), arrival_basis="matched_wait_start")]


def test_population_and_initial_remaining_work_conserved_without_duplicate_cargo():
    frame = table()
    before = frame.copy(deep=True)
    sc, meta = build_hybrid(request(), calls=frame)
    pd.testing.assert_frame_equal(frame, before)
    assert [v.call_id for v in sc.vessels] == ["B", "W", "F", "H"]
    assert meta["initial_counts"] == dict(eligible_calls=4, berthed=1, waiting=1, future=2)
    assert len({v.call_id for v in sc.vessels}) == 4
    assert [v.initial_import_workload for v in sc.vessels] == [500, 1000, 1000, 1000]
    assert meta["workload_estimation"]["initial_total"] == 3500
    r = simulate_coupled(sc)
    assert r["vessels"][0]["berth_start_hour"] == -5
    assert r["vessels"][0]["completion_hour"] == 1
    assert r["vessels"][1]["berth_start_hour"] == 1  # not actual historical berth hour 8
    assert r["kpis"]["cargo_unloaded"] == 2500
    assert r["kpis"]["cargo_remaining_aboard"] == 1000  # H arrives exactly at horizon
    assert r["kpis"]["system_conservation_residual"] == 0
    assert r["kpis"]["queue_exposure_vessel_hours"] == 2  # historical pre-t0 wait excluded
    assert r["kpis"]["avg_completed_queue_wait_hours"] == 1
    assert r["kpis"]["vessels_berthed"] == 2  # excludes initially berthed B


def test_unmatched_waiting_remains_unknown_and_metadata_survives():
    sc, meta = build_hybrid(request(), calls=table())
    assert meta["waiting_coverage"]["unmatched_calls"] == 2
    assert meta["waiting_coverage"]["matched_calls"] == 2
    assert meta["call_quality"][0]["waiting_history"] == "UNKNOWN"
    assert sc.vessels[0].arrival_basis == "berth_start_fallback"
    assert sc.vessels[0].source_vessel_id == "IMO1"
    assert all(v.workload_provenance == "SYNTHETIC_ESTIMATED" for v in sc.vessels)
    assert all(v.workload_units == "normalized_units" for v in sc.vessels)
    assert meta["hindsight_assisted"]


def test_possible_physical_duplicates_flagged_and_not_dropped():
    frame = table()
    frame.loc[frame.call_id == "F", "vessel_id"] = "IMO1"
    sc, meta = build_hybrid(request(), calls=frame)
    assert len(sc.vessels) == 4
    assert any(item["call_ids"] == ["B", "F"] for item in meta["identity_overlaps"])
    assert any("retained" in warning for warning in meta["data_quality_warnings"])


def test_duplicate_mmsi_with_conflicting_vessel_id_also_flagged():
    frame = table()
    frame.loc[frame.call_id == "F", "mmsi"] = 101
    _, meta = build_hybrid(request(), calls=frame)
    assert any(item["identity_fields"] == ["mmsi"] for item in meta["identity_overlaps"])


@pytest.mark.parametrize("column,value", [
    ("call_id", "B"), ("berth_end", at(-6)), ("wait_hours", -1),
    ("arrival", pd.NaT), ("length_m", float("inf")), ("mmsi", 1.5),
])
def test_invalid_dataset_rejected(column, value):
    frame = table()
    if column == "call_id":
        frame.loc[2, column] = value
    elif column == "mmsi":
        frame[column] = frame[column].astype(float)
        frame.loc[1, column] = value
    else:
        frame.loc[1, column] = value
    with pytest.raises((ValueError, ValidationError)):
        validate_calls(frame)


def test_missing_schema_column_and_invalid_wait_link_rejected():
    with pytest.raises(ValueError, match="missing call columns"):
        validate_calls(table().drop(columns="imo"))
    frame = table()
    frame.loc[2, "wait_end"] = at(12)
    with pytest.raises(ValueError, match="matched waiting"):
        validate_calls(frame)


def test_workload_does_not_depend_on_length_realized_duration_or_gate_rate():
    sc, meta = build_hybrid(request(), calls=table())
    frame = table()
    frame["length_m"] *= 2
    frame["berth_end"] += timedelta(hours=300)
    frame["berth_hours"] += 300
    # OLD would become active after this edit: explicitly keep the same cohort.
    ids = [v.call_id for v in sc.vessels]
    other, changed = build_hybrid(request(call_ids=ids), calls=frame)
    assert [v.initial_import_workload for v in sc.vessels] == [v.initial_import_workload for v in other.vessels]
    assert meta["workload_estimation"] == changed["workload_estimation"]


def test_remaining_fraction_is_an_explicit_sensitivity_not_historical_progress():
    sc, meta = build_hybrid(request(workload=dict(full_import_units=800, berthed_remaining_fraction=.25)), calls=table())
    assert sc.vessels[0].initial_import_workload == 200
    assert meta["workload_estimation"]["assignments"][0]["estimated_original_import_workload"] == 800


def test_as_of_never_reads_finalized_sources_or_future_attributes(monkeypatch):
    def forbidden():
        pytest.fail("as-of attempted to access a finalized source")
    monkeypatch.setattr("marsa.twin.hybrid.load_calls", forbidden)
    monkeypatch.setattr("marsa.twin.hybrid.load_cargo", forbidden)
    req = request(time_information_mode="as_of", as_of_observations=observations())
    original = simulate_hybrid(req, calls=table())
    future = table()
    future["berth_end"] = at(999)
    future["berth_hours"] = -999  # intentionally invalid: as-of must not even validate final history
    future["length_m"] = 9999
    future.loc[len(future)] = future.iloc[-1]
    assert original == simulate_hybrid(req, calls=future)
    assert original == simulate_hybrid(req)
    assert original["initial_counts"] == dict(eligible_calls=2, berthed=1, waiting=1, future=0)
    assert original["source"]["future_arrivals"] == "UNSPECIFIED_NO_ARRIVAL_FEED"
    assert not original["hindsight_assisted"]
    assert original["call_quality"][0]["available_at"] == START.isoformat()


@pytest.mark.parametrize("change", ["future_available", "future_arrival", "future_berth", "unmatched_wait", "duplicate", "future_end"])
def test_invalid_as_of_observations_rejected(change):
    obs = observations()
    if change == "future_available":
        obs[0]["available_at"] = at(1)
    elif change == "future_arrival":
        obs[1]["arrival"] = at(1)
    elif change == "future_berth":
        obs[0]["berth_start"] = at(1)
    elif change == "unmatched_wait":
        obs[1]["arrival_basis"] = "berth_start_fallback"
    elif change == "duplicate":
        obs[1]["call_id"] = "B"
    else:
        obs[0]["berth_end"] = at(100)
    with pytest.raises((ValueError, ValidationError)):
        build_hybrid(request(time_information_mode="as_of", as_of_observations=obs))


def test_as_of_requires_a_snapshot_and_forbids_finalized_cargo():
    with pytest.raises(ValidationError, match="independently supplied"):
        request(time_information_mode="as_of")
    with pytest.raises(ValidationError, match="Cargo CSV"):
        request(time_information_mode="as_of", as_of_observations=observations(), inventory_source="cargo_snapshot")


def test_snapshot_identity_conflicts_retained():
    obs = observations()
    obs[1]["vessel_id"] = "IMO1"
    sc, meta = build_hybrid(request(time_information_mode="as_of", as_of_observations=obs))
    assert len(sc.vessels) == 2
    assert len(meta["identity_overlaps"]) == 1


def test_cargo_snapshot_inventory_used_once_and_flows_ignored():
    cargo = pd.DataFrame(dict(timestamp=[START], containers_in_yard=[123.0],
                              container_arrivals=[1e9], container_departures=[1e9]))
    r = simulate_hybrid(request(inventory_source="cargo_snapshot"), calls=table(), cargo=cargo)
    assert r["baseline"]["scenario"]["parameters"]["initial_inventory"] == 123
    assert r["baseline"]["kpis"]["system_conservation_residual"] == pytest.approx(0)
    cargo["container_arrivals"] = -1e9
    cargo["container_departures"] = -1e9
    assert r == simulate_hybrid(request(inventory_source="cargo_snapshot"), calls=table(), cargo=cargo)
    cargo["containers_in_yard"] = 0
    sc, _ = build_hybrid(request(inventory_source="cargo_snapshot"), calls=table(), cargo=cargo)
    assert sc.parameters.initial_inventory == 0


def test_inventory_outside_capacity_or_missing_hour_rejected():
    cargo = pd.DataFrame(dict(timestamp=[START], containers_in_yard=[10001]))
    with pytest.raises(ValueError):
        build_hybrid(request(inventory_source="cargo_snapshot"), calls=table(), cargo=cargo)
    cargo["timestamp"] = at(-1)
    with pytest.raises(ValueError, match="exact-hour"):
        build_hybrid(request(inventory_source="cargo_snapshot"), calls=table(), cargo=cargo)


def test_initial_occupancy_cannot_be_silently_lost_or_slots_invented():
    raw = request().model_dump()
    raw["parameters"]["berth_slots"] = 0
    with pytest.raises(ValidationError, match="initial berth occupancy"):
        build_hybrid(HybridRequest.model_validate(raw), calls=table())


def test_call_subset_unknown_duplicate_and_boundary_censoring():
    with pytest.raises(ValidationError, match="unique"):
        request(call_ids=["B", "B"])
    with pytest.raises(ValueError, match="unknown requested"):
        build_hybrid(request(call_ids=["missing"]), calls=table())
    sc, meta = build_hybrid(request(call_ids=["B"]), calls=table())
    assert len(sc.vessels) == 1
    assert meta["call_quality"][0]["boundary_censoring_possible"]


def test_paired_runs_share_inputs_are_reproducible_and_show_gate_feedback():
    raw = request().model_dump()
    raw["parameters"].update(initial_inventory=10000, gate_units_per_hour=0)
    req = HybridRequest.model_validate(raw)
    r = simulate_hybrid(req, calls=table())
    assert r == simulate_hybrid(req, calls=table())
    assert r["baseline"]["scenario"]["vessels"] == r["intervention"]["scenario"]["vessels"]
    assert r["baseline"]["kpis"]["completed_vessels"] == 0
    assert r["intervention"]["kpis"]["completed_vessels"] == 3
    assert r["intervention"]["kpis"]["queue_exposure_vessel_hours"] < r["baseline"]["kpis"]["queue_exposure_vessel_hours"]
    for mode in ["baseline", "intervention"]:
        assert abs(r[mode]["kpis"]["system_conservation_residual"]) < 1e-8
    assert r["baseline"]["vessels"][-1]["state"] == "waiting"


def test_optional_intervention_and_no_forced_gate_benefit():
    raw = request(intervention_gate_units_per_hour=0).model_dump()
    raw["parameters"]["unloading_units_per_hour"] = 0
    r = simulate_hybrid(HybridRequest.model_validate(raw), calls=table())
    assert r["baseline"]["kpis"] == r["intervention"]["kpis"]
    r = simulate_hybrid(request(intervention_gate_units_per_hour=None), calls=table())
    assert r["intervention"] is None


def test_explicitly_disabled_endpoint_and_enabled_as_of_without_legacy_changes(monkeypatch):
    cfg = deepcopy(load_config())
    cfg["hybrid_normalized"]["enabled"] = False
    monkeypatch.setattr(coupled_twin, "load_config", lambda: cfg)
    body = request(time_information_mode="as_of", as_of_observations=observations()).model_dump(mode="json")
    with TestClient(app) as client:
        assert client.post("/api/twin/simulate-hybrid", json=body).status_code == 403
        cfg["hybrid_normalized"]["enabled"] = True
        response = client.post("/api/twin/simulate-hybrid", json=body)
        assert response.status_code == 200
        assert response.json()["initialization"] == "hybrid_vessel_calls"
        assert response.json()["baseline"]["kpis"]["system_conservation_residual"] == 0
        body["as_of_observations"][0]["available_at"] = at(1).isoformat()
        assert client.post("/api/twin/simulate-hybrid", json=body).status_code == 422
        cfg["hybrid_normalized"]["workload"]["full_import_units"] = -1
        assert client.post("/api/twin/simulate-hybrid", json=body).status_code == 503
        cfg.pop("hybrid_normalized")
        assert client.post("/api/twin/simulate-hybrid", json=body).status_code == 403


def test_two_actual_2025_demonstrations_and_saved_call_schema():
    from scripts.demo_hybrid_twin import historical_demo

    from marsa.twin.calls import load_calls

    frame = load_calls()
    before = frame.copy(deep=True)
    assert len(validate_calls(frame)) == frame.call_id.nunique() == 4704
    result = historical_demo()
    assert [r["initial_counts"]["eligible_calls"] for r in result.values()] == [20, 23]
    for case in result.values():
        assert case["hindsight_assisted"]
        assert case["intervention"]["completed_vessels"] > case["baseline"]["completed_vessels"]
        assert abs(case["baseline"]["system_conservation_residual"]) < 1e-8
        assert abs(case["intervention"]["system_conservation_residual"]) < 1e-8
    pd.testing.assert_frame_equal(frame, before)


def test_phase_one_initial_states_validate_and_do_not_count_historical_wait():
    sc, _ = build_hybrid(request(), calls=table())
    raw = sc.model_dump()
    raw["vessels"][0]["initial_berth_start_timestamp"] = at(1)
    with pytest.raises(ValidationError, match="berth start"):
        CoupledScenario.model_validate(raw)


@pytest.mark.parametrize("failure", [FileNotFoundError, PermissionError, DataNotReadyError])
@pytest.mark.parametrize("endpoint", ["hybrid", "sensitivity", "advisory"])
def test_unavailable_ais_has_structured_503_without_results(monkeypatch, failure, endpoint):
    import marsa.api.routes.decision_support as decision_route

    cfg = deepcopy(load_config())
    for flag in ["hybrid_normalized", "sensitivity_analysis", "strategy_decision_support"]:
        cfg[flag]["enabled"] = True
    monkeypatch.setattr(coupled_twin, "load_config", lambda: cfg)
    monkeypatch.setattr(decision_route, "load_config", lambda: cfg)
    def unavailable():
        raise failure("private/source/path must not appear in the public error")
    monkeypatch.setattr("marsa.twin.hybrid.load_calls", unavailable)
    scenario = request().model_dump(mode="json")
    if endpoint == "hybrid":
        url, body = "/api/twin/simulate-hybrid", scenario
    elif endpoint == "sensitivity":
        url, body = "/api/twin/analyze-sensitivity", {"scenario": scenario}
    else:
        url = "/api/decision-support/advisory"
        body = {"timestamp_utc": START.isoformat(), "experiment": {"scenario": scenario}}
    with TestClient(app) as client:
        response = client.post(url, json=body)
    assert response.status_code == 503
    assert response.json() == {"detail": {"code": "SOURCE_UNAVAILABLE",
                                         "message": "Required simulation source is unavailable."}}
