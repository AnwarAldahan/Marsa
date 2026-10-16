"""Digital Twin tests: replay of real AIS calls, stress scenarios, actions and ranking."""
import pandas as pd
import pytest

from marsa.config import load_config
from marsa.twin.adapter import build_scenario, validate_stress
from marsa.twin.calls import load_calls
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
    {"berths_closed": -1}, {"berths_closed": 24}, {"arrival_surge_pct": -5},
    {"arrival_surge_pct": 500}, {"wind": 50},
])
def test_invalid_stress_rejected(stress):
    with pytest.raises(ValueError):
        validate_stress(stress, 23)


def test_arrival_surge_brings_more_real_vessels_into_horizon():
    cfg = load_config()
    ts = pd.Timestamp("2025-03-04T10:00:00Z")
    normal, _ = build_scenario(ts, cfg)
    surged, sources = build_scenario(ts, cfg, {"arrival_surge_pct": 100})
    assert len(surged.vessels) >= len(normal.vessels)
    assert any(s["provenance"] == "USER INPUT" for s in sources)
