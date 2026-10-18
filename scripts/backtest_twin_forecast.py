"""How accurate is the twin's forecast mode? Compare it with what really happened in 2025.

For every sampled 2025 hour (default: every 12 hours) the twin runs with no action twice:
  - replay:   recorded AIS times  -> this reproduces what actually happened (the truth)
  - forecast: only what was known at that hour (stays and remaining waits estimated from the past)
At +6, +12 and +24 hours we compare the number of vessels at LA berths and waiting, and the total
vessel-hours waiting over the horizon. A naive baseline is also scored: "it stays as it is now"
(the count at hour 0 carried forward). Cargo coupling is switched off here: the test is about the
vessel side, which is what AIS can verify.

Output: data/processed/twin_forecast_backtest.json
Usage:  python scripts/backtest_twin_forecast.py [--step-hours 12]
"""
import argparse
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from marsa.config import load_config  # noqa: E402
from marsa.twin.adapter import build_scenario  # noqa: E402
from marsa.twin.calls import ais_hours  # noqa: E402
from marsa.twin.engine import simulate_candidate  # noqa: E402

OUT = R / "data" / "processed" / "twin_forecast_backtest.json"
CHECK_HOURS = (6, 12, 24)


# Run the no-action twin at ts in one mode and return its hourly timeline and total waiting.
def run(ts, cfg, mode):
    sc, _ = build_scenario(ts, cfg, mode=mode)
    out = simulate_candidate(sc, {"id": "baseline", "actions": []})
    return out["timeline"], out["kpis"]["total_wait_vessel_hours"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--step-hours", type=int, default=12)
    args = parser.parse_args()
    cfg = load_config()
    cfg["twin"]["cargo_coupling"] = False
    horizon = int(cfg["twin"]["horizon_hours"])
    hours = ais_hours()
    hours = hours[(hours.hour % args.step_hours == 0)
                  & (hours <= hours.max() - pd.Timedelta(hours=horizon))]
    err = {f"{k}_{h}": {"forecast": [], "naive": []} for k in ("at_berth", "waiting") for h in CHECK_HOURS}
    wait_err, wait_true, used = [], [], 0
    for ts in hours:
        try:
            truth, w_true = run(ts, cfg, "replay")
            fc, w_fc = run(ts, cfg, "forecast")
        except (KeyError, ValueError):
            continue
        used += 1
        for key in ("at_berth", "waiting"):
            for h in CHECK_HOURS:
                err[f"{key}_{h}"]["forecast"].append(abs(fc[h][key] - truth[h][key]))
                err[f"{key}_{h}"]["naive"].append(abs(truth[0][key] - truth[h][key]))
        wait_err.append(abs(w_fc - w_true))
        wait_true.append(w_true)
    result = {
        "hours_tested": used, "step_hours": args.step_hours, "horizon_hours": horizon,
        "mean_absolute_error": {
            name: {"twin_forecast": round(float(np.mean(v["forecast"])), 2),
                   "naive_no_change": round(float(np.mean(v["naive"])), 2)}
            for name, v in err.items()},
        "total_wait_vessel_hours": {"mean_abs_error": round(float(np.mean(wait_err)), 2),
                                    "mean_actual": round(float(np.mean(wait_true)), 2)},
        "note": "Truth = replay of recorded AIS (it reproduces 2025 exactly). Forecast uses only data before each hour.",
    }
    OUT.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(json.dumps(result, indent=2))


if __name__ == "__main__":
    main()
