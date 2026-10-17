"""Phase 4 paired historical examples. Run: python -B -m scripts.demo_strategy_advisory.

No files are written; no Gemini calls or operational actions are made.
"""
import json
import sys
from copy import deepcopy
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))


def historical_advisory_demo() -> dict:
    from marsa.agents.strategy_advisory import run_advisory_experiment
    from marsa.config import load_config
    from marsa.pipeline import run
    from marsa.twin.sensitivity import SensitivityRequest

    cfg = deepcopy(load_config())
    cfg["strategy_investigation"]["use_gemini"] = False
    cfg["strategy_decision_support"]["enabled"] = False
    output = {}
    for label, timestamp, source, inventory, slots in [
        ("A", "2025-03-04T10:00:00Z", "cargo_snapshot", 0, 13),
        ("B", "2025-07-10T19:00:00Z", "explicit", 10000, 17),
    ]:
        cfg["strategy_decision_support"]["enabled"] = False
        legacy = run(timestamp, cfg=cfg, llm=None, save=False)
        cfg["strategy_decision_support"]["enabled"] = True
        experiment = SensitivityRequest(scenario={
            "timestamp_utc": timestamp, "horizon_hours": 24, "time_information_mode": "retrospective",
            "cohort_id": f"heuristic-cargo-stays-{timestamp}", "resource_pool_id": "abstract-advisory-pool",
            "inventory_source": source, "parameters": {
                "yard_capacity": 10000, "initial_inventory": inventory, "berth_slots": slots,
                "unloading_units_per_hour": 1000, "gate_units_per_hour": 100,
            },
        })
        evidence = run_advisory_experiment(experiment, cfg)
        enabled = run(timestamp, cfg=cfg, llm=None, save=False, advisory_evidence=evidence)
        advisory = enabled.pop("strategy_advisory")
        if enabled != legacy:
            raise AssertionError("Advisory changed legacy outputs")
        output[label] = {
            "timestamp_utc": timestamp, "legacy_winner": legacy["decision_support"]["highest_ranked_candidate"],
            "winner_after_advisory": enabled["decision_support"]["highest_ranked_candidate"],
            "legacy_response_unchanged": True, "advisory": advisory,
        }
    return output


if __name__ == "__main__":
    print(json.dumps(historical_advisory_demo(), indent=2))
