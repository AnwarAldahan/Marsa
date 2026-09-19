"""Strategy orchestrator: reads forecast + 3 agents, proposes candidates, sends them to the twin,
ranks the twin's results with the documented score, and writes the recommendation.

Candidate generation is rule-based (transparent). An LLM, if provided, only writes the narrative.
"""
from __future__ import annotations
from marsa.agents.llm import NoLLM
from marsa.twin.engine import simulate_candidate
from marsa.twin.scoring import score_results


def generate_candidates(forecast: dict, maritime: dict, cargo: dict, context: dict) -> list[dict]:
    cands = [{"id": "baseline", "label": "No intervention (FCFS)", "actions": []}]
    bottlenecks = {maritime.get("possible_bottleneck"), cargo.get("possible_bottleneck")}
    # queue rule always worth testing when vessels are waiting
    if maritime["waiting_vessels"] > 0:
        cands.append({"id": "shortest_first", "label": "Prioritise shortest service time",
                      "actions": [{"type": "queue_policy", "value": "shortest_service_first"}]})
    if "berth_capacity" in bottlenecks or "vessel_queue" in bottlenecks:
        cands.append({"id": "extra_berth", "label": "Open one additional berth (lay berth)",
                      "actions": [{"type": "add_berths", "value": 1}]})
        cands.append({"id": "virtual_arrival", "label": "Ask approaching vessels to slow (virtual arrival, +4h)",
                      "actions": [{"type": "delay_arrivals", "value": 4}]})
    if "yard_capacity" in bottlenecks or "yard_clearance" in bottlenecks or "gate_capacity" in bottlenecks:
        cands.append({"id": "gate_extension", "label": "Extend gate hours / add gate lanes (+30% clearance)",
                      "actions": [{"type": "gate_boost", "value": 1.3}]})
    if context["twin_multipliers"]["crane_multiplier"] < 1.0 or "berth_capacity" in bottlenecks:
        cands.append({"id": "crane_boost", "label": "Add crane gang on busiest vessels (+25% discharge)",
                      "actions": [{"type": "crane_boost", "value": 1.25}]})
    cands.append({"id": "combined", "label": "Shortest-first + gate extension",
                  "actions": [{"type": "queue_policy", "value": "shortest_service_first"},
                              {"type": "gate_boost", "value": 1.3}]})
    return cands


def evaluate(scenario, candidates: list[dict], cfg: dict) -> list[dict]:
    tw = cfg["twin"]
    results = [simulate_candidate(scenario, c, runs=tw["monte_carlo_runs"], seed=tw["random_seed"]) for c in candidates]
    labels = {c["id"]: c["label"] for c in candidates}
    ranked = score_results(results, cfg["scoring"]["weights"])
    for r in ranked:
        r["label"] = labels[r["candidate_id"]]
    return ranked


def recommend(ranked: list[dict], forecast: dict, agents: dict, llm=None) -> dict:
    base = next(r for r in ranked if r["candidate_id"] == "baseline")
    best = ranked[0]
    k = lambda r, m: r["kpis"][m]["mean"]
    dw, dd, dy = (k(best, "avg_wait_hours") - k(base, "avg_wait_hours"),
                  k(best, "delayed_vessels") - k(base, "delayed_vessels"),
                  k(best, "yard_peak_occupancy") - k(base, "yard_peak_occupancy"))
    MIN_GAIN = 0.05                      # score gap below this = not worth an intervention
    if best is not base and base["score"] - best["score"] < MIN_GAIN:
        best, dw, dd, dy = base, 0.0, 0.0, 0.0
    if best is base:
        text = (f"No intervention recommended: the twin finds no candidate that improves waiting, delays or yard "
                f"pressure meaningfully over the next {len(ranked)} tested options. Baseline estimate: average berth "
                f"wait {k(base,'avg_wait_hours'):.1f}h, {k(base,'delayed_vessels'):.1f} delayed vessels, "
                f"yard peak {100*k(base,'yard_peak_occupancy'):.0f}%.")
    else:
        text = (f"Recommended: {best['label']}. Over the next horizon the twin estimates average berth wait "
                f"{k(best,'avg_wait_hours'):.1f}h (baseline {k(base,'avg_wait_hours'):.1f}h, {dw:+.1f}h), "
                f"{k(best,'delayed_vessels'):.1f} delayed vessels ({dd:+.1f}) and yard peak "
                f"{100*k(best,'yard_peak_occupancy'):.0f}% ({100*dy:+.0f} pts). "
                f"Values are simulated estimates, not guaranteed outcomes.")
    llm = llm or NoLLM()
    narrative = llm.complete(f"Explain this port recommendation for an operations manager:\n{text}\n"
                             f"Evidence: {agents}") or text
    return {"recommended_candidate": best["candidate_id"], "label": best["label"],
            "expected_change_vs_baseline": {"avg_wait_hours": round(dw, 2), "delayed_vessels": round(dd, 2), "yard_peak_occupancy": round(dy, 3)},
            "narrative": narrative, "ranking": [{"rank": r["rank"], "id": r["candidate_id"], "score": r["score"]} for r in ranked]}
