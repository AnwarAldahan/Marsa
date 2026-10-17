"""Per-call provenance, causal integration and continuous congestion measurements."""
from copy import deepcopy
from pathlib import Path

import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marsa.api.main import app
from marsa.common.exceptions import DataNotReadyError
from marsa.config import load_config
from marsa.twin.adapter import build_scenario
from marsa.twin.calls import CALLS_PATH, load_calls
from marsa.twin.engine import Scenario, Vessel, simulate_candidate
from marsa.twin.workloads import (
    WORKLOADS_PATH,
    derive_workloads,
    load_workloads,
    reference_workload,
    sha256,
    write_workloads,
)

TS = pd.Timestamp("2025-07-10T19:00:00Z")


def test_all_call_ids_unique_traceable_and_deterministic(tmp_path):
    source_hash = sha256(CALLS_PATH)
    derived, manifest = load_workloads()
    original = load_calls().set_index("call_id")
    assert len(derived) == len(original) == 4704
    assert derived.index.is_unique and set(derived.index) == set(original.index)
    assert not original.vessel_id.is_unique  # A vessel can have many separately keyed calls.
    assert (derived.mmsi == original.loc[derived.index].mmsi).all()
    assert (derived.estimated_workload == original.loc[derived.index].berth_hours).all()
    assert (derived.available_at_utc == original.loc[derived.index].berth_end).all()
    for name in ("one.csv", "two.csv"):
        write_workloads(CALLS_PATH, tmp_path / name)
        assert sha256(tmp_path / name) == manifest["dataset_sha256"]
        assert (tmp_path / name).read_bytes().replace(b"\r\n", b"\n") == WORKLOADS_PATH.read_bytes().replace(b"\r\n", b"\n")
    assert sha256(CALLS_PATH) == source_hash


def test_script_reproducible_with_custom_output(tmp_path):
    import subprocess
    import sys

    output = tmp_path / "derived.csv"
    script = Path(__file__).resolve().parents[1] / "scripts/build_vessel_cargo_workloads.py"
    result = subprocess.run([sys.executable, str(script), "--output", str(output)],
                            capture_output=True, text=True, check=False)
    assert result.returncode == 0, result.stderr
    assert output.read_bytes().replace(b"\r\n", b"\n") == WORKLOADS_PATH.read_bytes().replace(b"\r\n", b"\n")


@pytest.mark.parametrize("bad", ["duplicate", "negative", "nan", "timestamps", "missing_field"])
def test_bad_source_rejected(bad):
    calls = load_calls().head(3).copy()
    if bad == "duplicate":
        calls.loc[calls.index[1], "call_id"] = calls.iloc[0].call_id
    elif bad == "negative":
        calls.loc[calls.index[0], "berth_hours"] = -1
    elif bad == "nan":
        calls.loc[calls.index[0], "berth_hours"] = float("nan")
    elif bad == "timestamps":
        calls.loc[calls.index[0], "berth_end"] += pd.Timedelta(hours=1)
    else:
        calls = calls.drop(columns="mmsi")
    with pytest.raises(ValueError):
        derive_workloads(calls)


def test_original_source_cannot_be_overwritten():
    with pytest.raises(ValueError, match="overwrite"):
        write_workloads(CALLS_PATH, CALLS_PATH)


def test_retrospective_reference_cannot_leak_into_an_asof_forecast():
    row = load_workloads()[0].loc["LA00000"]
    with pytest.raises(ValueError, match="not available"):
        reference_workload(row, row.arrival)
    assert reference_workload(row, row.arrival, retrospective=True) == row.estimated_workload
    assert reference_workload(row, row.available_at_utc) == row.estimated_workload
    with pytest.raises(ValueError, match="UTC"):
        reference_workload(row, row.arrival.tz_localize(None))


def test_missing_and_corrupt_derived_artifacts_rejected(tmp_path):
    with pytest.raises(DataNotReadyError):
        load_workloads(tmp_path / "missing.csv")
    output = tmp_path / "corrupt.csv"
    write_workloads(CALLS_PATH, output)
    output.write_text(output.read_text().replace("LA00000", "LA00001", 1))
    with pytest.raises(ValueError, match="provenance"):
        load_workloads(output)


@pytest.mark.parametrize("source_ending", [b"\n", b"\r\n"], ids=["source-LF", "source-CRLF"])
@pytest.mark.parametrize("derived_ending", [b"\n", b"\r\n"], ids=["derived-LF", "derived-CRLF"])
def test_provenance_accepts_lf_and_crlf_checkouts_without_rewriting(tmp_path, source_ending, derived_ending):
    source = tmp_path / CALLS_PATH.name
    output = tmp_path / WORKLOADS_PATH.name
    source_bytes = CALLS_PATH.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", source_ending)
    derived_bytes = WORKLOADS_PATH.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", derived_ending)
    source.write_bytes(source_bytes)
    output.write_bytes(derived_bytes)
    output.with_suffix(".manifest.json").write_bytes(WORKLOADS_PATH.with_suffix(".manifest.json").read_bytes())
    loaded, manifest = load_workloads(output, source)
    expected, _ = load_workloads()
    pd.testing.assert_frame_equal(loaded, expected)
    assert len(loaded) == 4704 and loaded.index.is_unique
    assert sha256(source) == manifest["source_sha256"]
    assert sha256(output) == manifest["dataset_sha256"]
    assert source.read_bytes() == source_bytes
    assert output.read_bytes() == derived_bytes


@pytest.mark.parametrize("ending", [b"\n", b"\r\n"], ids=["LF", "CRLF"])
@pytest.mark.parametrize("changed", ["source", "derived"])
def test_line_ending_normalization_still_rejects_modified_content(tmp_path, ending, changed):
    source = tmp_path / CALLS_PATH.name
    output = tmp_path / WORKLOADS_PATH.name
    source.write_bytes(CALLS_PATH.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", ending))
    output.write_bytes(WORKLOADS_PATH.read_bytes().replace(b"\r\n", b"\n").replace(b"\n", ending))
    output.with_suffix(".manifest.json").write_bytes(WORKLOADS_PATH.with_suffix(".manifest.json").read_bytes())
    target = source if changed == "source" else output
    content = target.read_bytes()
    altered = content.replace(b"LA00000", b"LA99999", 1)
    assert altered != content
    target.write_bytes(altered)
    with pytest.raises(ValueError, match="provenance"):
        load_workloads(output, source)


@pytest.mark.parametrize("timestamp", ["2025-03-04T10:00:00Z", "2025-07-10T19:00:00Z", "2025-11-05T12:00:00Z"])
def test_scenario_maps_every_call_and_added_vessel_without_changing_reference_results(timestamp):
    sc, sources = build_scenario(pd.Timestamp(timestamp), load_config(), added=[{"service_hours": 2}])
    assert any(s.get("source") == WORKLOADS_PATH.name for s in sources)
    assert set(sc.workloads) == {v.call_id for v in sc.berthed + sc.vessels}
    assert sc.workloads["ADD1"]["initial_workload"] == 2
    for v in sc.berthed:
        assert sc.workloads[v.call_id]["initial_workload"] == pytest.approx(v.service_h + v.berth_h)
    mapped = simulate_candidate(sc, {"id": "baseline"})
    sc.workloads = None  # The previous validated implementation's inline reference formula.
    previous = simulate_candidate(sc, {"id": "baseline"})
    assert mapped["kpis"] == previous["kpis"]
    assert mapped["timeline"] == previous["timeline"]


def test_loaded_workload_drives_completion_and_berth_release_instead_of_service_field():
    sc = Scenario(berths=1, horizon_hours=4, berthed=[],
                  vessels=[Vessel("A", 1, 0, 0, 1), Vessel("B", 2, 0, 0, 1)],
                  cargo={"capacity": 10, "initial_inventory": 0, "gate_rate": 2},
                  workloads={"A": {"initial_workload": 2}, "B": {"initial_workload": 1}})
    before = deepcopy(sc)
    out = simulate_candidate(sc, {"id": "baseline"})
    assert sc == before
    assert out["vessels"][0]["completion_h"] == 2
    assert out["vessels"][1]["berth_h"] == 2
    assert out["kpis"]["total_wait_vessel_hours"] == 2
    assert out["kpis"]["completed_vessels"] == 2
    sc.workloads.pop("A")
    with pytest.raises(ValueError, match="cohort"):
        simulate_candidate(sc, {"id": "baseline"})


def test_loaded_dataset_perturbation_propagates_through_adapter_and_constrained_twin(monkeypatch):
    """Change only decoded workload A; keep AIS durations and all resources fixed."""
    import marsa.twin.workloads as module

    original_loader = module.load_workloads
    table, _ = original_loader()
    calls = table[table.estimated_workload == 1].head(2)
    first, second = calls.index
    scenario = Scenario(berths=1, horizon_hours=4, berthed=[],
                        vessels=[Vessel(key, int(row.mmsi), 0, 0, 1) for key, row in calls.iterrows()],
                        cargo={"capacity": 1., "initial_inventory": 0., "gate_rate": .5})
    original = deepcopy(scenario)
    results = []
    for workload in (1., 3.):
        def controlled_loader(workload=workload):
            loaded, manifest = original_loader()
            loaded = loaded.copy(deep=True)
            loaded.loc[first, "estimated_workload"] = workload
            return loaded, manifest

        monkeypatch.setattr(module, "load_workloads", controlled_loader)
        current = deepcopy(scenario)
        module.attach_workloads(current, TS)
        assert current.workloads[first]["initial_workload"] == workload
        assert current.workloads[second]["initial_workload"] == 1
        assert [v.service_h for v in current.vessels] == [1, 1]
        results.append(simulate_candidate(current, {"id": "baseline"}))
    low, high = results
    assert scenario == original
    assert low["kpis"]["completed_vessels"] == 2
    assert high["kpis"]["completed_vessels"] == 1
    assert low["kpis"]["cargo_unloaded"] == pytest.approx(2)
    assert high["kpis"]["cargo_unloaded"] == pytest.approx(3)
    assert low["kpis"]["total_wait_vessel_hours"] == 1
    assert high["kpis"]["total_wait_vessel_hours"] == 4
    assert low["kpis"]["yard_full_hours"] == 0
    assert high["kpis"]["yard_full_hours"] == pytest.approx(2)
    assert low["vessels"][1]["berth_h"] == 1
    assert high["vessels"][1]["berth_h"] is None
    for result in results:
        assert abs(result["kpis"]["yard_conservation_residual"]) < 1e-8
        assert abs(result["kpis"]["work_conservation_residual"]) < 1e-8


def test_congestion_durations_include_fractional_crossings_and_recovery():
    sc = Scenario(berths=1, horizon_hours=4, berthed=[], vessels=[Vessel("A", 1, 0, 0, 2)],
                  cargo={"capacity": 1., "initial_inventory": 0., "gate_rate": 0., "near_capacity_pct": 90.})
    out = simulate_candidate(sc, {"id": "baseline"})
    assert out["kpis"]["yard_near_capacity_hours"] == pytest.approx(3.1)
    assert out["kpis"]["yard_full_hours"] == pytest.approx(3)
    assert out["kpis"]["cargo_unloaded"] == pytest.approx(1)
    assert out["kpis"]["blocked_unloading"] == pytest.approx(3)
    assert out["kpis"]["remaining_vessel_workload"] == pytest.approx(1)
    assert out["timeline"][0]["cargo_congestion_status"] == "BELOW_THRESHOLD"
    assert out["timeline"][-1]["cargo_congestion_status"] == "FULL"
    assert sum(f["yard_full_hours"] for f in out["flows"]) == out["kpis"]["yard_full_hours"]
    sc.cargo["gate_rate"] = 1
    recovered = simulate_candidate(sc, {"id": "gate"})
    assert recovered["kpis"]["yard_full_hours"] == 0
    assert recovered["kpis"]["completed_vessels"] == 1
    assert recovered["kpis"]["remaining_vessel_workload"] == 0
    for result in (out, recovered):
        for p in result["timeline"]:
            assert p["yard_inventory"] == pytest.approx(p["cargo_unloaded"] - p["cargo_cleared"])
            assert p["remaining_vessel_workload"] + p["cargo_unloaded"] == pytest.approx(2)
        assert abs(result["kpis"]["work_conservation_residual"]) < 1e-8


@pytest.mark.parametrize("value", [0, -1, 101, float("nan")])
def test_invalid_congestion_threshold_rejected(value):
    sc, _ = build_scenario(TS, load_config())
    sc.cargo["near_capacity_pct"] = value
    with pytest.raises(ValueError, match="threshold"):
        simulate_candidate(sc, {"id": "baseline"})


def test_missing_workload_and_missing_call_return_controlled_api_errors(monkeypatch):
    import marsa.twin.workloads as module

    original = module.load_workloads
    for problem, status in [("unavailable", 404), ("mapping", 422)]:
        def fail(problem=problem):
            if problem == "unavailable":
                raise DataNotReadyError("Vessel cargo workload source unavailable")
            table, manifest = original()
            return table.iloc[:0], manifest

        monkeypatch.setattr(module, "load_workloads", fail)
        with TestClient(app) as client:
            for route in ("/api/twin/simulate", "/api/decision-support/analyze"):
                response = client.post(route, json={"timestamp_utc": TS.isoformat()})
                assert response.status_code == status
                assert "detail" in response.json()


def test_strategy_and_twin_share_workloads_congestion_and_waiting_objective():
    with TestClient(app) as client:
        decision = client.post("/api/decision-support/analyze", json={"timestamp_utc": TS.isoformat()})
        assert decision.status_code == 200
        data = decision.json()
        assert data["decision_support"]["highest_ranked_candidate"] == "baseline"
        assert "لم يصاحب زيادة التفريغ انخفاض" in data["decision_support"]["summary"]
        for candidate in data["digital_twin"]["results"]:
            result = client.post("/api/twin/simulate", json={"timestamp_utc": TS.isoformat(), "actions": candidate["actions"]})
            assert result.status_code == 200
            assert result.json()["scenario"] == data["digital_twin"]["scenario"]
            assert result.json()["result"]["kpis"] == candidate["kpis"]
            assert result.json()["result"]["timeline"] == candidate["timeline"]
            if candidate["candidate_id"] == "gate_max":
                assert candidate["vs_baseline"]["cargo_unloaded"] > 0
                assert candidate["vs_baseline"]["total_wait_vessel_hours"] == 0
                assert candidate["vs_baseline"]["yard_full_hours"] < 0


@pytest.mark.parametrize("manifest_text", ["[]", "{}", "{invalid"])
def test_malformed_manifest_fails_with_controlled_validation(tmp_path, manifest_text):
    output = tmp_path / "derived.csv"
    write_workloads(CALLS_PATH, output)
    output.with_suffix(".manifest.json").write_text(manifest_text)
    with pytest.raises(ValueError):
        load_workloads(output)


def test_full_yard_duration_ends_at_fractional_downward_threshold_crossing():
    sc = Scenario(berths=1, horizon_hours=2, berthed=[], vessels=[],
                  cargo={"capacity": 1., "initial_inventory": 1., "gate_rate": .5, "near_capacity_pct": 90.})
    out = simulate_candidate(sc, {"id": "baseline"})
    assert out["kpis"]["yard_near_capacity_hours"] == pytest.approx(.2)
    assert out["kpis"]["yard_full_hours"] == 0  # Touching full at t0 has no positive duration.
    assert out["timeline"][-1]["cargo_congestion_status"] == "BELOW_THRESHOLD"
