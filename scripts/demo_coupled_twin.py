"""Run the deterministic yard-blocking acceptance fixture without enabling the API.

From the repository root: python -B -m scripts.demo_coupled_twin
No files, API keys, AIS data or network calls are used.
"""
import json
import sys
from datetime import UTC, datetime
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def acceptance_demo() -> dict:
    from marsa.twin.coupled import CoupledScenario, simulate_coupled

    start = datetime(2025, 1, 1, tzinfo=UTC)
    common = {
        "timestamp_utc": start, "horizon_hours": 4,
        "cohort_id": "explicit-two-vessel-demo", "resource_pool_id": "abstract-demo-yard",
        "vessels": [{"simulation_id": name, "arrival_timestamp": start,
                     "initial_import_workload": 100} for name in ("A", "B")],
    }
    results = {}
    for label, gate in [("A_no_clearance", 0), ("B_clearance", 100)]:
        result = simulate_coupled(CoupledScenario(**common, parameters={
            "yard_capacity": 10000, "initial_inventory": 10000, "berth_slots": 1,
            "unloading_units_per_hour": 100, "gate_units_per_hour": gate,
        }))
        results[label] = {
            "kpis": result["kpis"],
            "vessels": [{key: v[key] for key in (
                "simulation_id", "state", "berth_start_hour", "completion_hour",
                "remaining_import_workload",
            )} for v in result["vessels"]],
        }
    return results


if __name__ == "__main__":
    print(json.dumps(acceptance_demo(), indent=2))
