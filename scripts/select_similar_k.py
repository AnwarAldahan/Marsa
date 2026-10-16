"""Choose k (how many similar-length vessels to compare with) from the data, not by hand.

Test: for every 2025 LA stay, estimate its length of stay from the k vessels closest in length,
using only stays completed before that vessel arrived and never the vessel's own stays (it is
treated as a new vessel). The k with the smallest median absolute error wins; a tie goes to the
smaller mean absolute error. Candidates are the k values listed in CANDIDATE_K.

Output: data/processed/similar_k_selection.json (errors for every k); copy the chosen k into
config/port_config.yaml -> twin.similar_vessels_k.
Usage: python scripts/select_similar_k.py
"""
import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from marsa.twin.calls import load_calls  # noqa: E402
from marsa.twin.estimates import similar_length_hours  # noqa: E402

CANDIDATE_K = [5, 10, 20, 50, 100, 200]
OUT = R / "data" / "processed" / "similar_k_selection.json"


# Absolute errors of the similar-length estimate for every 2025 stay, for one k.
def errors_for(calls: pd.DataFrame, k: int) -> np.ndarray:
    test = calls[calls["arrival"].dt.year == 2025]
    errs = []
    for row in test.itertuples():
        history = calls[calls["berth_end"] <= row.arrival]
        guess, _ = similar_length_hours(history, row.length_m, k, exclude_vessel=row.vessel_id)
        errs.append(abs(row.berth_hours - guess))
    return np.array(errs)


# Run the test for every candidate k and write the results.
def main() -> None:
    calls = load_calls()
    results = []
    for k in CANDIDATE_K:
        e = errors_for(calls, k)
        results.append({"k": k, "median_abs_error_h": round(float(np.median(e)), 2),
                        "mean_abs_error_h": round(float(e.mean()), 2), "stays_tested": int(e.size)})
        print(results[-1], flush=True)
    best = min(results, key=lambda r: (r["median_abs_error_h"], r["mean_abs_error_h"]))
    OUT.write_text(json.dumps({"chosen_k": best["k"], "results": results}, indent=2), encoding="utf-8")
    print("chosen k:", best["k"], "->", OUT.relative_to(R))


if __name__ == "__main__":
    main()
