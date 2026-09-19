import sys, random
from pathlib import Path
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marsa.twin.engine import Scenario, build_vessels, apply_actions, run_once, simulate_candidate
from marsa.twin.scoring import score_results

def sc(**kw):
    base = dict(berths=5, berthed_now=4, waiting_now=3, yard_inventory=6000, yard_capacity=10000,
                discharge_units_per_hour=14, gate_units_per_hour=100, arrival_rate_per_hour=0.5,
                mean_service_hours=12, horizon_hours=24)
    base.update(kw); return Scenario(**base)

def test_zero_berths_serves_nobody():
    r = run_once(sc(berths=0, berthed_now=0), build_vessels(random.Random(1), sc(berths=0, berthed_now=0)), "fcfs")
    assert r.vessels_served == 0 and r.max_wait_hours > 0

def test_more_berths_never_hurts_wait():
    a = simulate_candidate(sc(), {"id": "a", "actions": []}, runs=10)
    b = simulate_candidate(sc(), {"id": "b", "actions": [{"type": "add_berths", "value": 3}]}, runs=10)
    assert b["kpis"]["avg_wait_hours"]["mean"] <= a["kpis"]["avg_wait_hours"]["mean"]

def test_gate_boost_lowers_yard_peak():
    a = simulate_candidate(sc(), {"id": "a", "actions": []}, runs=5)
    b = simulate_candidate(sc(), {"id": "b", "actions": [{"type": "gate_boost", "value": 1.5}]}, runs=5)
    assert b["kpis"]["yard_peak_occupancy"]["mean"] <= a["kpis"]["yard_peak_occupancy"]["mean"]

def test_scoring_ranks_and_is_deterministic():
    res = [simulate_candidate(sc(), {"id": i, "actions": a}, runs=5) for i, a in
           [("baseline", []), ("x", [{"type": "add_berths", "value": 2}])]]
    ranked = score_results(res, {"avg_wait_hours": 0.5, "max_wait_hours": 0.2, "delayed_vessels": 0.2, "yard_peak_occupancy": 0.1})
    assert [r["rank"] for r in ranked] == [1, 2]
    assert simulate_candidate(sc(), {"id": "a", "actions": []}, runs=5) == simulate_candidate(sc(), {"id": "a", "actions": []}, runs=5)

def test_unknown_action_rejected():
    import pytest
    with pytest.raises(ValueError):
        apply_actions(sc(), [], [{"type": "teleport", "value": 1}])
