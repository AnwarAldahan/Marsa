"""Causal cargo progression in Main's existing Twin, without physical cargo claims."""
from copy import deepcopy

import pandas as pd
import pytest

from marsa.config import load_config
from marsa.twin.adapter import build_scenario
from marsa.twin.engine import Scenario, Vessel, simulate_candidate
from marsa.twin.scoring import score_results


def small(inventory=1., gate=0., capacity=1.):
    return Scenario(berths=1, horizon_hours=4, berthed=[],
                    vessels=[Vessel("A", 1, 0, 0, 1), Vessel("B", 2, 0, 0, 1)],
                    yard={"max_recorded_departures": 1.}, cargo={
                        "capacity": capacity, "initial_inventory": inventory,
                        "gate_rate": gate, "source_capacity": capacity})


def run(sc, actions=None):
    return simulate_candidate(sc, {"id": "baseline", "actions": actions or []})


def test_full_yard_blocks_work_release_and_waiting_until_gate_clears():
    sc = small()
    before = deepcopy(sc)
    blocked = run(sc)
    recovered = run(sc, [{"type": "gate_capacity", "value": 1}])
    assert sc == before
    assert blocked["kpis"]["cargo_unloaded"] == blocked["kpis"]["completed_vessels"] == 0
    assert blocked["kpis"]["total_wait_vessel_hours"] == 4
    assert blocked["vessels"][1]["berth_h"] is None
    assert recovered["kpis"]["cargo_unloaded"] == recovered["kpis"]["completed_vessels"] == 2
    assert recovered["kpis"]["total_wait_vessel_hours"] == 1
    assert [v["completion_h"] for v in recovered["vessels"]] == [1, 2]
    assert recovered["vessels"][1]["berth_h"] == 1


def test_unloading_fills_yard_then_slows_when_clearance_is_insufficient():
    sc = small(inventory=0, gate=0, capacity=.5)
    out = run(sc)
    assert out["kpis"]["cargo_unloaded"] == .5
    assert out["timeline"][-1]["yard_occupancy_pct"] == 100
    assert out["kpis"]["completed_vessels"] == 0
    slow = run(small(gate=.5))
    assert slow["vessels"][0]["completion_h"] == 2
    assert slow["vessels"][1]["berth_h"] == 2


def test_fractional_arrival_waits_until_readiness_without_losing_queue_exposure():
    sc = small(inventory=0, gate=2, capacity=10)
    sc.vessels = [Vessel("F", 1, .25, .75, .25)]
    out = run(sc)
    assert out["kpis"]["total_wait_vessel_hours"] == .5
    assert out["kpis"]["capacity_wait_vessel_hours"] == 0
    assert out["vessels"][0]["completion_h"] == 1
    assert out["kpis"]["peak_berths_used"] == 1


def test_subhour_yard_peak_is_not_lost_between_hourly_frames():
    sc = small(inventory=0, gate=.5, capacity=.1)
    sc.vessels = [Vessel("F", 1, 0, 0, .2)]
    out = run(sc)
    assert out["kpis"]["yard_peak_occupancy_pct"] == pytest.approx(100)
    assert out["kpis"]["yard_end_occupancy_pct"] == pytest.approx(0)


def test_berth_already_occupied_cannot_release_at_recorded_departure_if_blocked():
    sc = small()
    sc.berthed = [Vessel("I", 3, -1, -1, 2, berth_h=-1)]
    out = run(sc)
    assert out["vessels"][0]["completion_h"] is None
    assert out["vessels"][0]["remaining_workload"] == 1
    assert all(p["at_berth"] == 1 for p in out["timeline"])


@pytest.mark.parametrize("gate", [0., .5, 1., 10.])
def test_conservation_capacity_and_determinism(gate):
    sc = small(gate=gate)
    out = run(sc)
    assert run(sc) == out
    for flow in out["flows"]:
        assert abs(flow["yard_conservation_residual"]) < 1e-8
        assert abs(flow["work_conservation_residual"]) < 1e-8
        assert 0 <= flow["closing_inventory"] <= 1
    assert all(v["completion_h"] is None or v["remaining_workload"] == 0 for v in out["vessels"])


def test_existing_queue_and_extra_berth_actions_are_compared_on_same_input():
    sc = small(gate=2)
    before = deepcopy(sc)
    out = [simulate_candidate(sc, {"id": name, "actions": actions}) for name, actions in [
        ("baseline", []), ("same", [{"type": "queue_policy", "value": "fcfs"}]),
        ("extra_berth", [{"type": "add_berths", "value": 1}])]]
    assert sc == before
    assert score_results(out)[0]["candidate_id"] == "extra_berth"
    assert [r["candidate_id"] for r in score_results(out)][1:] == ["baseline", "same"]
    assert all(r["cargo_assumptions"] == sc.cargo for r in out)


def test_july_uses_only_opening_cargo_ratios_and_fixed_resource_scale():
    sc, _ = build_scenario(pd.Timestamp("2025-07-10T19:00:00Z"), load_config())
    assert sc.cargo["capacity"] == 23 * 24
    assert sc.cargo["initial_inventory"] / sc.cargo["capacity"] == pytest.approx(.948077)
    first = run(sc)
    # Later Cargo CSV rows cannot double-count inflow or leak into the coupled gate rate.
    sc.yard["arrivals"][1:] = [1e9] * (len(sc.yard["arrivals"]) - 1)
    sc.yard["departures"][1:] = [1e9] * (len(sc.yard["departures"]) - 1)
    assert run(sc) == first
    assert first["kpis"]["cargo_unloaded"] > 0
    assert first["kpis"]["completed_vessels"] > 0
    assert first["kpis"]["yard_peak_occupancy_pct"] <= 100 + 1e-8


def test_invalid_inventory_rejected_and_horizon_arrival_not_served():
    sc = small(inventory=2)
    with pytest.raises(ValueError):
        run(sc)
    sc = small(gate=1)
    sc.vessels.append(Vessel("H", 4, 4, 4, 1))
    out = run(sc)
    assert out["vessels"][-1]["berth_h"] is None
    assert out["kpis"]["waiting_end"] == 1


@pytest.mark.parametrize("mode", ["valid", "contradictory", "failed"])
def test_coupled_narrative_cannot_override_verified_outcomes(mode):
    from marsa.agents.strategy_agent import recommend

    class Provider:
        def generate_structured(self, **kwargs):
            if mode == "failed":
                raise RuntimeError("provider unavailable")
            return {"summary": kwargs["payload"]["deterministic_summary"] if mode == "valid" else "The winner is extra_berth; observed cargo improved."}

    row = dict(run(small()), rank=1, score=4, title="Baseline")
    out = recommend([row], {"status": "ml_unavailable"}, {
        "maritime": {"maritime_status": "NORMAL"}, "cargo": {"cargo_status": "NORMAL"},
        "events_weather": {"contextual_pressure": {"level": "low"}}}, llm=Provider())
    assert out["highest_ranked_candidate"] == "baseline"
    assert out["reasoning_mode"] == ("llm_assisted" if mode == "valid" else "deterministic_fallback")
    assert "extra_berth" not in out["summary"]


def test_pipeline_disclosures_distinguish_coupled_counterfactual_from_exact_replay():
    from marsa.pipeline import run

    cfg = load_config()
    for enabled in (True, False):
        cfg["twin"]["cargo_coupling"] = enabled
        out = run("2025-07-10T19:00:00Z", cfg=cfg)
        limitations = " ".join(out["limitations"])
        if enabled:
            assert "completion and berth release depend on simulated unloading" in limitations
            assert "not exact AIS replay" in limitations
            assert "Without a stress scenario the Digital Twin replays" not in limitations
        else:
            assert "Without a stress scenario the Digital Twin replays" in limitations
