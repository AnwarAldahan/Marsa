"""Read-only Phase 3 historical experiments. Run: python -B -m scripts.demo_sensitivity_twin.

Use --full for all OAT/interactions and hourly diagnostic evidence. Default prints
base-context comparisons, ranges, and robustness including parameter dependence.
No output files, external services or operational decisions are created.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def historical_sensitivity_demo(full: bool = False) -> dict:
    from marsa.twin.hybrid import HybridRequest
    from marsa.twin.sensitivity import analyze_sensitivity

    results = {}
    for label, timestamp, source, inventory, slots in [
        ("A", "2025-03-04T10:00:00Z", "cargo_snapshot", 0, 13),
        ("B", "2025-07-10T19:00:00Z", "explicit", 10000, 17),
    ]:
        report = analyze_sensitivity(HybridRequest(
            timestamp_utc=timestamp, horizon_hours=24, time_information_mode="retrospective",
            cohort_id=f"heuristic-cargo-stays-{timestamp}", resource_pool_id="abstract-sensitivity-pool",
            inventory_source=source, parameters={
                "yard_capacity": 10000, "initial_inventory": inventory, "berth_slots": slots,
                "unloading_units_per_hour": 1000, "gate_units_per_hour": 100,
            },
        ))
        if full:
            results[label] = report
            continue
        rows = []
        for row in report["contexts"][0]["comparisons"]:
            rows.append({"intervention": row["intervention"], "kpis": row["kpis"],
                         "bottleneck": row["bottleneck"]["classification"],
                         "active_constraints": row["bottleneck"]["active_constraints"],
                         "constraint_hours": row["bottleneck"]["constraint_hours"],
                         "bottleneck_changed": row.get("bottleneck_changed"),
                         "vs_baseline": row.get("vs_baseline")})
        results[label] = {
            "timestamp_utc": timestamp, "initial_counts": report["metadata"]["initial_counts"],
            "initial_inventory": report["metadata"]["initial_inventory"],
            "ranges": report["ranges_low_base_high"], "context_count": len(report["contexts"]),
            "run_count": report["run_count"], "base_comparisons": rows,
            "robustness": report["robustness"], "excluded_contexts": report["excluded_contexts"],
            "skipped_duplicate_contexts": report["skipped_duplicate_contexts"],
        }
    return results


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--full", action="store_true")
    print(json.dumps(historical_sensitivity_demo(parser.parse_args().full), indent=2))
