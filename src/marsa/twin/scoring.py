"""Rank Digital Twin results by one measured quantity: total vessel-hours spent waiting.

There are no weights. Lower total waiting is better; on a tie the earlier candidate in the list
keeps its place, so "no change" (always first) wins whenever an action brings no improvement.
"""
from __future__ import annotations

RANKING_KPI = "total_wait_vessel_hours"


# Sort candidates by total waiting and attach the difference to the no-change baseline.
def score_results(results: list[dict]) -> list[dict]:
    base = next((r for r in results if r["candidate_id"] == "baseline"), None)
    ranked = sorted(results, key=lambda r: r["kpis"][RANKING_KPI])
    out = []
    for rank, item in enumerate(ranked, 1):
        item = dict(item, rank=rank, ranking_kpi=RANKING_KPI,
                    score=item["kpis"][RANKING_KPI])
        if base is not None and item["candidate_id"] != "baseline":
            item["vs_baseline"] = {
                k: round(item["kpis"][k] - base["kpis"][k], 2)
                for k in ("total_wait_vessel_hours", "capacity_wait_vessel_hours",
                          "vessels_waited", "max_wait_hours", "waiting_end",
                          "yard_peak_occupancy_pct", "yard_end_occupancy_pct")
                if item["kpis"].get(k) is not None and base["kpis"].get(k) is not None
            }
        out.append(item)
    return out
