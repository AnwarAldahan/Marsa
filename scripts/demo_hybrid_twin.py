"""Two retrospective 2025 paired experiments; no file writes, credentials or HTTP.

Run from repository root: python -B -m scripts.demo_hybrid_twin
All rates, workloads and slot counts below are explicit synthetic assumptions.
"""
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def historical_demo() -> dict:
    from marsa.twin.hybrid import HybridRequest, simulate_hybrid

    result = {}
    for label, timestamp, inventory_source, inventory, slots in [
        ("A_recorded_synthetic_inventory", "2025-03-04T10:00:00Z", "cargo_snapshot", 0, 13),
        ("B_controlled_full_yard", "2025-07-10T19:00:00Z", "explicit", 10000, 17),
    ]:
        output = simulate_hybrid(HybridRequest(
            timestamp_utc=timestamp, horizon_hours=24, time_information_mode="retrospective",
            cohort_id=f"all-eligible-heuristic-cargo-stays-{timestamp}", resource_pool_id="abstract-demo-yard",
            inventory_source=inventory_source, parameters={
                "yard_capacity": 10000, "initial_inventory": inventory, "berth_slots": slots,
                "unloading_units_per_hour": 1000, "gate_units_per_hour": 100,
            }, intervention_gate_units_per_hour=1000,
        ))
        result[label] = {
            "timestamp_utc": timestamp, "time_information_mode": output["time_information_mode"],
            "hindsight_assisted": output["hindsight_assisted"], "initial_counts": output["initial_counts"],
            "waiting_coverage": output["waiting_coverage"], "initial_inventory": output["initial_inventory"],
            "estimated_remaining_workload_total": output["workload_estimation"]["initial_total"],
            "resource_assumptions": output["baseline"]["scenario"]["parameters"],
            "baseline": output["baseline"]["kpis"], "intervention": output["intervention"]["kpis"],
            "data_quality_warnings": output["data_quality_warnings"],
        }
    return result


if __name__ == "__main__":
    print(json.dumps(historical_demo(), indent=2))
