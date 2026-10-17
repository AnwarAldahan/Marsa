"""Digital Twin tests: replay of real AIS calls, stress scenarios, actions and ranking."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient

from marsa.api.main import app
from marsa.config import load_config
from marsa.twin.adapter import build_scenario, validate_stress
from marsa.twin.calls import load_calls
from marsa.twin.estimates import estimate_stay
from marsa.twin.engine import Scenario, Vessel, apply_actions, simulate_candidate
from marsa.twin.scoring import score_results


# Small hand-made scenario: 2 berths, both busy for 5 h, three vessels ready at hour 1.
def tiny(**kw):
    berthed = [Vessel("B1", 1, -10, -10, 15, berth_h=-10), Vessel("B2", 2, -10, -10, 15, berth_h=-10)]
    vessels = [Vessel("A", 3, 1, 1, 10), Vessel("B", 4, 1.5, 1.5, 2), Vessel("C", 5, 2, 2, 3)]
    base = dict(berths=2, horizon_hours=24, berthed=berthed, vessels=vessels)
    base.update(kw)
    return Scenario(**base)


# Without stress the twin berths every vessel exactly when AIS recorded it.
def test_replay_matches_ais_without_stress():
    cfg = load_config()
    mismatches = checked = 0
    for ts in pd.date_range("2025-01-03", "2025-12-28", freq="61h", tz="UTC"):
        sc, _ = build_scenario(ts, cfg)
        result = simulate_candidate(sc, {"id": "baseline"})
        assert result["kpis"]["capacity_wait_vessel_hours"] == 0
        for v in result["vessels"]:
            checked += 1
            mismatches += v["berth_h"] != v["scheduled_berth_h"]
    assert checked > 100 and mismatches == 0


# Vessels at berth in the scenario are the vessels AIS shows at an LA berth at that hour.
def test_state_counts_come_from_calls():
    calls = load_calls()
    ts = pd.Timestamp("2025-07-10T19:00:00Z")
    sc, sources = build_scenario(ts, load_config())
    expected = ((calls["berth_start"] <= ts) & (calls["berth_end"] > ts)).sum()
    assert len(sc.berthed) == expected
    assert any(s["provenance"] == "OBSERVED (AIS)" for s in sources)
    assert all(s["provenance"] != "USER INPUT" for s in sources)


def test_closing_berths_creates_capacity_wait():
    base = simulate_candidate(tiny(), {"id": "baseline"})
    assert base["kpis"]["capacity_wait_vessel_hours"] > 0
    assert simulate_candidate(tiny(berths_closed=0, berths=5), {"id": "x"})["kpis"][
        "capacity_wait_vessel_hours"] == 0


def test_extra_berth_never_increases_waiting():
    a = simulate_candidate(tiny(), {"id": "a"})
    b = simulate_candidate(tiny(), {"id": "b", "actions": [{"type": "add_berths", "value": 1}]})
    assert b["kpis"]["total_wait_vessel_hours"] <= a["kpis"]["total_wait_vessel_hours"]


def test_shortest_first_serves_short_vessel_before_long_one():
    fifo = simulate_candidate(tiny(), {"id": "f"})
    sf = simulate_candidate(tiny(), {"id": "s", "actions": [
        {"type": "queue_policy", "value": "shortest_service_first"}]})
    first = {r["call_id"]: r["berth_h"] for r in sf["vessels"]}
    assert first["B"] < first["A"]
    assert sf["kpis"]["total_wait_vessel_hours"] <= fifo["kpis"]["total_wait_vessel_hours"]


def test_simulation_is_deterministic():
    assert simulate_candidate(tiny(), {"id": "a"}) == simulate_candidate(tiny(), {"id": "a"})


def test_ranking_uses_total_waiting_and_keeps_baseline_on_tie():
    results = [simulate_candidate(tiny(), {"id": "baseline"}),
               simulate_candidate(tiny(), {"id": "same", "actions": [
                   {"type": "queue_policy", "value": "fcfs"}]}),
               simulate_candidate(tiny(), {"id": "more", "actions": [
                   {"type": "add_berths", "value": 2}]})]
    ranked = score_results(results)
    assert [r["candidate_id"] for r in ranked][:2] == ["more", "baseline"]
    assert ranked[0]["ranking_kpi"] == "total_wait_vessel_hours"


def test_unknown_action_and_policy_rejected():
    with pytest.raises(ValueError):
        apply_actions(tiny(), [{"type": "gate_boost", "value": 1.3}])
    with pytest.raises(ValueError):
        apply_actions(tiny(), [{"type": "queue_policy", "value": "random"}])


@pytest.mark.parametrize("stress", [
    {"berths_closed": -1}, {"berths_closed": 24}, {"arrival_multiplier": 0},
    {"arrival_multiplier": -1}, {"arrival_surge_pct": 100}, {"wind": 50},
])
def test_invalid_stress_rejected(stress):
    with pytest.raises(ValueError):
        validate_stress(stress, 23)


def test_arrival_surge_brings_more_real_vessels_into_horizon():
    cfg = load_config()
    ts = pd.Timestamp("2025-03-04T10:00:00Z")
    normal, _ = build_scenario(ts, cfg)
    surged, sources = build_scenario(ts, cfg, {"arrival_multiplier": 2})
    assert len(surged.vessels) >= len(normal.vessels)
    assert any(s["provenance"] == "USER INPUT" for s in sources)


# A large multiplier is allowed: the user is free to test any scenario.
def test_large_multiplier_is_accepted():
    assert validate_stress({"arrival_multiplier": 10}, 23)["arrival_multiplier"] == 10


# Hours missing from the AIS file inside the horizon are reported, never filled.
def test_missing_ais_hours_are_reported():
    _, sources = build_scenario(pd.Timestamp("2025-06-30T20:00:00Z"), load_config())
    gaps = [s for s in sources if s.get("kind") == "data_gap"]
    assert gaps and gaps[0]["value"][0][0].startswith("2025-07-01T00:00")
    _, sources = build_scenario(pd.Timestamp("2025-07-10T19:00:00Z"), load_config())
    assert not [s for s in sources if s.get("kind") == "data_gap"]


# --- /api/twin endpoints and added vessels ---------------------------------------------------
TS = "2025-07-10T19:00:00Z"


def test_simulate_returns_baseline_and_user_decision():
    with TestClient(app) as client:
        r = client.post("/api/twin/simulate", json={
            "timestamp_utc": TS, "stress": {"berths_closed": 6, "arrival_multiplier": 2.5},
            "actions": [{"type": "add_berths", "value": 1}]})
    assert r.status_code == 200
    body = r.json()
    assert body["baseline"]["candidate_id"] == "baseline"
    assert body["result"]["candidate_id"] == "user_decision"
    assert body["result"]["kpis"]["open_berths"] == body["baseline"]["kpis"]["open_berths"] + 1
    assert body["marker_hours"] == [6]
    assert "total_wait_vessel_hours" in body["result"]["vs_baseline"]


def test_simulate_gate_uses_recorded_maximum():
    with TestClient(app) as client:
        body = client.post("/api/twin/simulate", json={
            "timestamp_utc": TS, "actions": [{"type": "gate_capacity", "value": "max_recorded"}]}).json()
    assert body["result"]["kpis"]["gate_per_hour"] == body["scenario"]["gate_max_recorded"]
    assert body["result"]["kpis"]["yard_peak_occupancy_pct"] <= \
        body["baseline"]["kpis"]["yard_peak_occupancy_pct"]


def test_simulate_rejects_unknown_action_and_fields():
    with TestClient(app) as client:
        bad_action = client.post("/api/twin/simulate", json={
            "timestamp_utc": TS, "actions": [{"type": "fly", "value": 1}]})
        bad_vessel = client.post("/api/twin/simulate", json={
            "timestamp_utc": TS, "added": [{"arrival_in_hours": 2, "tonnage": 5}]})
        no_info = client.post("/api/twin/simulate", json={
            "timestamp_utc": TS, "added": [{"arrival_in_hours": 2}]})
    assert bad_action.status_code == 422 and bad_vessel.status_code == 422
    assert no_info.status_code == 422


def test_added_vessel_by_length_is_marked_and_sourced():
    with TestClient(app) as client:
        body = client.post("/api/twin/simulate", json={
            "timestamp_utc": TS, "added": [{"arrival_in_hours": 3, "length_m": 300}]}).json()
    added = [v for v in body["baseline"]["vessels"] if v["source"] == "added"]
    assert len(added) == 1 and added[0]["estimate"]["method"] == "similar_length"
    assert any(s.get("kind") == "added" for s in body["sources"])
    assert body["scenario"]["vessels_added"] == 1


def test_known_vessel_uses_own_history_before_the_hour():
    ts = pd.Timestamp(TS)
    with TestClient(app) as client:
        listed = client.get("/api/twin/vessels", params={"timestamp_utc": TS}).json()["vessels"]
    vid = listed[0]["vessel_id"]
    est = estimate_stay(ts, 5, vessel_id=vid)
    assert est["method"] == "own_history"
    assert est["hours"] == pytest.approx(listed[0]["median_stay_hours"])


def test_typed_service_hours_wins():
    assert estimate_stay(pd.Timestamp(TS), 5, length_m=300, service_hours=12)["method"] == "user_input"


def test_yard_stress_gate_closed_and_inflow():
    ts = pd.Timestamp(TS)
    cfg = load_config()
    base_sc, _ = build_scenario(ts, cfg)
    closed_sc, _ = build_scenario(ts, cfg, {"gate_closed_hours": 6})
    more_sc, _ = build_scenario(ts, cfg, {"container_inflow_multiplier": 1.5})
    base = simulate_candidate(base_sc, {"id": "b", "actions": []})["kpis"]
    closed = simulate_candidate(closed_sc, {"id": "b", "actions": []})["kpis"]
    more = simulate_candidate(more_sc, {"id": "b", "actions": []})["kpis"]
    assert closed["yard_peak_occupancy_pct"] > base["yard_peak_occupancy_pct"]
    assert more["yard_end_occupancy_pct"] > base["yard_end_occupancy_pct"]
    assert closed["total_wait_vessel_hours"] == base["total_wait_vessel_hours"]


def test_yard_stress_validation():
    with pytest.raises(ValueError):
        validate_stress({"gate_closed_hours": 30}, 23, 24)
    with pytest.raises(ValueError):
        validate_stress({"container_inflow_multiplier": 0}, 23, 24)


def test_hours_by_condition():
    with TestClient(app) as client:
        rain = client.get("/api/twin/hours", params={"condition": "rain"}).json()
        wind = client.get("/api/twin/hours", params={"condition": "wind", "limit": 3}).json()
        bad = client.get("/api/twin/hours", params={"condition": "snow"})
    assert rain["hours_found"] > 0
    assert all(61 <= h["weather_code"] <= 65 for h in rain["hours"])
    speeds = [h["wind_speed_10m"] for h in wind["hours"]]
    assert speeds == sorted(speeds, reverse=True) and len(speeds) == 3
    assert bad.status_code == 422
