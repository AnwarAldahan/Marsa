"""Composite score for twin results. Deterministic and documented: lower is better.

score = sum_k w_k * normalised_k, where each KPI is min-max normalised across the candidates
of the same run (so scores are comparable within one decision, not across days).
"""
from __future__ import annotations


def score_results(results: list[dict], weights: dict) -> list[dict]:
    keys = list(weights)
    lo = {k: min(r["kpis"][k]["mean"] for r in results) for k in keys}
    hi = {k: max(r["kpis"][k]["mean"] for r in results) for k in keys}
    base = next((r for r in results if r["candidate_id"] == "baseline"), None)
    out = []
    for r in results:
        parts = {}
        for k in keys:
            span = hi[k] - lo[k]
            parts[k] = 0.0 if span == 0 else (r["kpis"][k]["mean"] - lo[k]) / span
        s = sum(weights[k] * parts[k] for k in keys)
        item = dict(r)
        item["score"] = round(s, 4)
        item["score_components"] = {k: round(v, 3) for k, v in parts.items()}
        if base is not None and base is not r:
            item["vs_baseline"] = {k: round(r["kpis"][k]["mean"] - base["kpis"][k]["mean"], 3) for k in keys}
        out.append(item)
    out.sort(key=lambda x: x["score"])
    for rank, item in enumerate(out, 1):
        item["rank"] = rank
    return out
