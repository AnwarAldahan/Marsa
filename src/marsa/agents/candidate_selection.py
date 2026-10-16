"""Optional deterministic experiment priority, separate from simulated ranking.

This first policy retains the legacy pool (at most four candidates). Regional
trends cannot establish POLA intervention feasibility or justify suppressing a
previously eligible experiment. Complete evidence can change execution order.
"""
from __future__ import annotations

from pydantic import BaseModel, ConfigDict

from marsa.agents.investigation import _hypotheses
from marsa.agents.tools.diagnostics import DiagnosticResult
from marsa.data.load import parse_exact_utc_hour

CANDIDATE_BUDGET = 4
REQUIRED_TOOLS = {
    "baseline": set(),
    "shortest_first": {"maritime_history"},
    "gate_max": {"cargo_balance", "cargo_delay_profile"},
    "extra_berth": {"maritime_history"},
    "combined_queue_gate": {"cargo_balance", "cargo_delay_profile", "maritime_history"},
}
METRICS = {
    "cargo_balance": {"inventory_change", "net_cargo_flow", "yard_occupancy_change"},
    "cargo_delay_profile": {"truck_waiting_time_change"},
    "maritime_history": {"waiting_vessel_count_change", "waiting_ratio_change"},
}


class AdaptiveCandidateSettings(BaseModel):
    model_config = ConfigDict(strict=True, extra="forbid")
    enabled: bool = False


class EvidenceUnavailable(ValueError):
    pass


def _validated_hypotheses(investigation, required, timestamp, domains):
    if not investigation or investigation.get("mode") not in {"gemini", "deterministic_fallback"}:
        raise EvidenceUnavailable("investigation_unavailable")
    results = {}
    ts = parse_exact_utc_hour(timestamp)
    for raw in investigation["diagnostic_results"]:
        result = DiagnosticResult.model_validate(raw)
        if result.tool_name in results:
            raise EvidenceUnavailable("duplicate_diagnostic_evidence")
        results[result.tool_name] = result
    selected_results = []
    for tool in sorted(required):
        result = results.get(tool)
        if result is None:
            raise EvidenceUnavailable("required_diagnostic_missing")
        if (result.status != "complete" or result.coverage.expected_hours != 7
                or result.coverage.observed_hours != 7 or result.coverage.missing_timestamps
                or result.coverage.missing_values):
            raise EvidenceUnavailable("required_diagnostic_incomplete")
        if (parse_exact_utc_hour(result.timestamp_utc) != ts or result.window_hours != 6
                or result.evidence_id != tool):
            raise EvidenceUnavailable("diagnostic_time_or_identity_mismatch")
        maritime = tool == "maritime_history"
        expected_scope = ("REGIONAL_LA_LB_UNVERIFIED_MEMBERSHIP" if maritime
                          else "MIXED_SCOPE_SYNTHETIC")
        if (result.source.geographic_scope != expected_scope
                or not result.source.limitations or not result.source.identifier
                or (not maritime and "SYNTHETIC" not in result.source.provenance)):
            raise EvidenceUnavailable("diagnostic_provenance_invalid")
        if any(result.metrics.get(field) is None for field in METRICS[tool]):
            raise EvidenceUnavailable("required_diagnostic_metric_missing")
        evidence = domains["maritime" if maritime else "cargo"]["evidence"]
        for field, value in result.current_evidence.items():
            if (value is not None and field in evidence and evidence[field] is not None
                    and abs(float(evidence[field]) - value) > 1e-9):
                raise EvidenceUnavailable("diagnostic_endpoint_mismatch")
        selected_results.append(result)
    # Reuse the original deterministic hypothesis calculation without changing it.
    # Never trust an LLM preference or an unsupported hypothesis state in isolation.
    derived = _hypotheses(selected_results, domains)
    supplied = {item["hypothesis_id"]: item for item in investigation["hypotheses"]}
    for hypothesis in derived:
        original = supplied.get(hypothesis["hypothesis_id"], {})
        for key in ("state", "diagnostic_evidence_ids", "supporting_evidence_ids",
                    "contradicting_evidence_ids", "insufficient_evidence_ids"):
            if original.get(key) != hypothesis[key]:
                raise EvidenceUnavailable("hypothesis_evidence_mismatch")
    return {item["hypothesis_id"]: item for item in derived}


def _combined_state(states):
    if all(state == "supported" for state in states):
        return "strongly_supported"
    if all(state == "unsupported" for state in states):
        return "unsupported"
    return "mixed"


def _priority(candidate, hypotheses):
    key = candidate["candidate_id"]
    if key == "baseline":
        return {"candidate_id": key, "support": "reference", "priority_band": 0,
                "hypothesis_ids": [], "evidence_references": ["authoritative current port state"],
                "selection_rationale": "Required paired simulation reference; always evaluated first."}
    queue_id = "maritime_waiting_pressure_rising"
    gate_ids = ["cargo_accumulating", "cargo_delays_worsening"]
    ids = ([queue_id] if key in {"shortest_first", "extra_berth"} else gate_ids
           if key == "gate_max" else [queue_id, *gate_ids])
    support = _combined_state([hypotheses[name]["state"] for name in ids])
    reason = {
        "strongly_supported": "Concordant rising historical pressure prioritizes this existing comparative experiment.",
        "mixed": "Historical signals are mixed; retain the existing experiment conditionally.",
        "unsupported": "Recent trends do not support worsening pressure; retain the current-status experiment conditionally.",
    }[support]
    band = {"strongly_supported": 1, "mixed": 2, "unsupported": 3}[support]
    if key == "extra_berth":
        # Regional queue history is not evidence of POLA berth availability/need.
        support = "mixed" if support == "strongly_supported" else support
        band = 3
        reason = "Legacy CONGESTED status retains the berth experiment conditionally; regional waiting trends cannot verify POLA berth necessity or availability."
    refs = list(candidate["evidence_references"])
    for name in ids:
        item = hypotheses[name]
        refs.extend(item["supporting_evidence_ids"] + item["contradicting_evidence_ids"])
    return {"candidate_id": key, "support": support, "priority_band": band,
            "hypothesis_ids": ids, "evidence_references": list(dict.fromkeys(refs)),
            "selection_rationale": reason}


def _report(candidates, priorities, mode, reason=None):
    available = [item["candidate_id"] for item in candidates]
    selected = [item["candidate_id"] for item in priorities]
    return {
        "mode": mode, "policy_version": "historical_priority_v1",
        "available_candidate_ids": available, "selected_candidate_ids": selected,
        "candidate_priorities": [dict(item, evaluation_position=index, selected_for_simulation=True)
                                 for index, item in enumerate(priorities, 1)],
        "conditionally_retained_candidates": [item["candidate_id"] for item in priorities
                                              if item["support"] in {"mixed", "unsupported", "insufficient"}],
        "excluded_candidates": [], "not_selected_due_to_budget": [],
        "budget": {"limit": CANDIDATE_BUDGET, "available": len(available),
                   "selected": len(selected), "at_limit": len(selected) == CANDIDATE_BUDGET},
        "evaluation_order_changed": available != selected,
        "simulation_alternatives_changed": False,
        "fallback_reason": reason,
        "selection_rationale": (
            "Prioritize complete evidence of worsening pressure; retain every legacy-eligible experiment within the existing four-candidate pool."
            if mode == "adaptive" else
            "Preserve legacy selection and order because required adaptive evidence or configuration is unavailable or invalid."
        ),
        "limitations": [
            "Priority describes experiment relevance, not intervention benefit, confidence or the simulated ranking.",
            "Cargo is synthetic/mixed scope; Maritime is regional LA/LB, not verified POLA-only.",
            "No new feasibility exclusions: diagnostics cannot establish terminal-level feasibility.",
            "The legacy generator caps its pool at four; this policy does not add alternatives or expand that budget.",
            "Forecast eligibility and event/weather rules remain legacy rules; missing weather remains unresolved.",
        ],
        "provenance": "AGENT_GENERATED",
    }


def select_candidates(candidates, investigation, timestamp, domains, cfg):
    """Return (ordered original candidates, optional report); fail to legacy order.

    The policy reorders references, never edits candidate/action dictionaries.
    The independent feature switch leaves disabled responses exactly unchanged.
    """
    raw = cfg.get("adaptive_candidates", {})
    if isinstance(raw, dict) and raw.get("enabled", False) is False:
        return candidates, None
    reason = None
    try:
        AdaptiveCandidateSettings.model_validate(raw)
        if (not candidates or candidates[0]["candidate_id"] != "baseline"
                or len(candidates) > CANDIDATE_BUDGET):
            raise EvidenceUnavailable("invalid_legacy_pool")
        required = set().union(*(REQUIRED_TOOLS[item["candidate_id"]] for item in candidates))
        hypotheses = _validated_hypotheses(investigation, required, timestamp, domains)
        priorities = [_priority(candidate, hypotheses) for candidate in candidates]
        priorities.sort(key=lambda item: item["priority_band"])  # Stable legacy order within a band.
        by_id = {item["candidate_id"]: item for item in candidates}
        return [by_id[item["candidate_id"]] for item in priorities], _report(candidates, priorities, "adaptive")
    except EvidenceUnavailable as error:
        reason = str(error)  # Only application-owned fixed reason codes.
    except Exception:
        reason = "invalid_configuration_or_evidence"
    priorities = [{"candidate_id": item["candidate_id"],
                   "support": "reference" if item["candidate_id"] == "baseline" else "insufficient",
                   "priority_band": 0 if item["candidate_id"] == "baseline" else 4,
                   "hypothesis_ids": [], "evidence_references": item["evidence_references"],
                   "selection_rationale": "Legacy selection/order preserved because adaptive evidence is unavailable or invalid."}
                  for item in candidates]
    return candidates, _report(candidates, priorities, "legacy_fallback", reason)
