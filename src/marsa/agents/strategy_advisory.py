"""Optional deterministic Strategy evidence attachment; never ranks or calls an LLM.

Public HTTP inputs are experiment requests, not externally supplied result blobs.
The pipeline can consume an internal evidence envelope, validated for provenance,
pairing, accounting, numerical comparisons and diagnostic/robustness consistency.
"""
from __future__ import annotations

from collections import Counter
from copy import deepcopy
from datetime import datetime, timedelta
from math import isfinite
from typing import Literal

import pandas as pd
from pydantic import BaseModel, ConfigDict, Field

from marsa.data.load import parse_exact_utc_hour
from marsa.twin.coupled import Contract, CoupledParameters, utc
from marsa.twin.hybrid import HybridRequest, HybridSettings, WorkloadSettings
from marsa.twin.sensitivity import (
    SensitivityPlan,
    SensitivityRequest,
    SensitivitySettings,
    _benefit,
    _signature,
    analyze_sensitivity,
    summarize_robustness,
)

INTERVENTIONS = ("baseline", "gate_modest", "gate_large", "unloading_modest", "berth_plus_one")
CONSTRAINTS = ("yard_capacity_blocking", "gate_clearance_limitation",
               "unloading_productivity_limitation", "berth_slot_limitation")


class AdvisorySettings(Contract):
    enabled: bool = False


class ExcludedContext(Contract):
    context: str
    reason: str


class DuplicateContext(Contract):
    context: str
    duplicate_of: str
    axis: str


class ReportStructure(BaseModel):
    """Validate every report field later consumed by response construction."""

    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    analysis: Literal["sensitivity_bottleneck_v1"]
    mode: Literal["coupled_normalized"]
    units: Literal["normalized_units"]
    provenance: Literal["SIMULATED"]
    metadata: dict
    population: list[dict]
    population_signature: str
    plan: SensitivityPlan
    contexts: list[dict] = Field(min_length=1, max_length=17)
    excluded_contexts: list[ExcludedContext]
    skipped_duplicate_contexts: list[DuplicateContext]
    robustness: dict
    ranges_low_base_high: dict[str, tuple[float, float, float]]
    run_count: int = Field(strict=True, ge=0)


def _event_hour(value: str) -> datetime:
    """Vessel events may precede 2025; the decision-year contract is separate."""
    event = utc(pd.Timestamp(value))
    _check(not (event.minute or event.second or event.microsecond or event.nanosecond), "EVENT_NOT_HOURLY")
    return event


class Outcome(BaseModel):
    model_config = ConfigDict(extra="ignore", allow_inf_nan=False)
    queue_exposure_vessel_hours: float = Field(ge=0)
    completed_vessels: int = Field(ge=0, strict=True)
    unfinished_vessels: int = Field(ge=0, strict=True)
    cargo_unloaded: float = Field(ge=0)
    cargo_remaining_aboard: float = Field(ge=0)
    cargo_cleared: float = Field(ge=0)
    yard_inventory_end: float = Field(ge=-1e-9)
    yard_peak_occupancy_pct: float = Field(ge=-1e-9, le=100.000000001)
    yard_end_occupancy_pct: float = Field(ge=-1e-9, le=100.000000001)
    yard_blocked_unloading: float = Field(ge=-1e-9)
    blocked_service_vessel_hours: float = Field(ge=0)
    vessel_conservation_residual: float
    yard_conservation_residual: float
    system_conservation_residual: float


def run_advisory_experiment(experiment: SensitivityRequest, cfg: dict) -> dict:
    """Expensive work only on explicit request; no modification of existing calculators."""
    settings = SensitivitySettings.model_validate(cfg.get("sensitivity_analysis", {}))
    defaults = HybridSettings.model_validate(cfg.get("hybrid_normalized", {})).workload
    report = analyze_sensitivity(experiment.scenario, plan=experiment.plan or settings.plan, defaults=defaults)
    return {"decision_timestamp": experiment.scenario.timestamp_utc.isoformat(),
            "request": experiment.scenario.model_dump(mode="json"), "report": report}


def _check(condition, code):
    if not condition:
        raise ValueError(code)


def _diagnostic(raw: dict, horizon: int) -> dict:
    _check(raw["method"] == "ledger_binding_constraints_v1", "UNSUPPORTED_DIAGNOSTIC")
    evidence = raw["hourly_evidence"]
    _check([e["hour"] for e in evidence] == list(range(horizon)), "INCOMPLETE_DIAGNOSTIC")
    counts = Counter()
    for row in evidence:
        _check(set(row["active_constraints"]) <= set(CONSTRAINTS), "UNKNOWN_CONSTRAINT")
        _check(len(row["active_constraints"]) == len(set(row["active_constraints"])), "DUPLICATE_CONSTRAINT")
        for field in ("berth_work_remaining_open", "waiting_work_open", "eligible_yard_stock_open", "headroom_after_gate"):
            _check(isfinite(row[field]) and row[field] >= -1e-7, "INVALID_DIAGNOSTIC_NUMBER")
        counts.update(row["active_constraints"])
    hours = {name: counts[name] for name in CONSTRAINTS}
    active = [name for name in CONSTRAINTS if counts[name]]
    classification = "mixed" if len(active) > 1 else active[0] if active else "inconclusive" if any(e["berth_queue_pressure"] for e in evidence) else "insufficient_demand"
    _check(raw["constraint_hours"] == hours and raw["active_constraints"] == active
           and raw["classification"] == classification, "DIAGNOSTIC_INCONSISTENCY")
    return {"method": raw["method"], "classification": classification,
            "active_constraints": active, "constraint_hours": hours,
            "berth_queue_pressure_hours": sum(bool(e["berth_queue_pressure"]) for e in evidence),
            "interpretation": "Binding constraints within this synthetic simulation; mixed may cover different hours, not proven real-world causes."}


def _validate_evidence(envelope: dict, timestamp) -> tuple[dict, list[dict], dict, list[dict]]:
    request = HybridRequest.model_validate(envelope["request"])
    _check(parse_exact_utc_hour(envelope["decision_timestamp"]) == timestamp == request.timestamp_utc, "DECISION_TIMESTAMP_MISMATCH")
    ReportStructure.model_validate(envelope["report"])
    report, meta = envelope["report"], envelope["report"]["metadata"]
    _check(report["analysis"] == "sensitivity_bottleneck_v1" and report["mode"] == "coupled_normalized"
           and report["units"] == "normalized_units" and report["provenance"] == "SIMULATED", "INVALID_PROVENANCE_OR_UNITS")
    _check(meta["time_information_mode"] == request.time_information_mode
           and meta["hindsight_assisted"] == (request.time_information_mode == "retrospective"), "TEMPORAL_PROVENANCE_MISMATCH")
    _check(meta["cohort_id"] == request.cohort_id and meta["resource_pool_id"] == request.resource_pool_id
           and meta["geographic_scope"] == "ABSTRACT_POLA_INSPIRED_UNVERIFIED_MEMBERSHIP", "COHORT_MISMATCH")
    population = report["population"]
    ids = [v["call_id"] for v in population]
    _check(len(ids) == len(set(ids)) == meta["initial_counts"]["eligible_calls"]
           and report["population_signature"] == _signature(population), "POPULATION_PAIRING_FAILURE")
    _check(all(v["simulation_id"] == v["call_id"] for v in population), "CALL_IDENTITY_MISMATCH")
    for state in ("berthed", "waiting", "future"):
        _check(meta["initial_counts"][state] == sum(v["initial_state"] == state for v in population), "INITIAL_STATE_COUNT_MISMATCH")
    for vessel in population:
        arrival = _event_hour(vessel["arrival_timestamp"])
        state = vessel["initial_state"]
        _check(state in {"berthed", "waiting", "future"}, "INVALID_INITIAL_STATE")
        _check(arrival <= timestamp + timedelta(hours=request.horizon_hours), "POPULATION_OUTSIDE_HORIZON")
        _check(arrival > timestamp if state == "future" else arrival <= timestamp, "INITIAL_STATE_TIME_MISMATCH")
        berth = vessel["initial_berth_start_timestamp"]
        _check(berth is None if state != "berthed" else berth is not None
               and arrival <= _event_hour(berth) <= timestamp, "INITIAL_BERTH_TIME_MISMATCH")
    if request.time_information_mode == "as_of":
        _check(all(v["initial_state"] != "future" for v in population)
               and meta["source"]["future_arrivals"] == "UNSPECIFIED_NO_ARRIVAL_FEED", "AS_OF_FUTURE_INFORMATION")
    plan = SensitivityPlan.model_validate(report["plan"])
    workload = WorkloadSettings.model_validate(meta["workload_estimation"]["settings"])
    if request.workload is not None:
        _check(workload == request.workload, "WORKLOAD_SETTINGS_MISMATCH")
    contexts = report["contexts"]
    _check(0 < len(contexts) <= 17 and contexts[0]["context"] == "base", "MISSING_BASE_CONTEXT")
    _check(len({c["context"] for c in contexts}) == len(contexts), "DUPLICATE_CONTEXT")
    _check(report["run_count"] == 5 * len(contexts), "INCOMPLETE_COMPARISONS")
    for values in report["ranges_low_base_high"].values():
        _check(len(values) == 3 and all(isfinite(v) and v >= 0 for v in values)
               and values[0] <= values[1] <= values[2], "INVALID_SENSITIVITY_RANGE")
    expected_base_parameters = {**request.parameters.model_dump(), "initial_inventory": meta["initial_inventory"]["value"]}
    _check(contexts[0]["comparisons"][0]["parameters"] == expected_base_parameters, "BASE_PARAMETER_MISMATCH")
    if request.inventory_source == "explicit":
        _check(meta["initial_inventory"]["value"] == request.parameters.initial_inventory, "INITIAL_INVENTORY_MISMATCH")
    validated, issues = [], []
    for context in contexts:
        rows = context["comparisons"]
        _check(tuple(r["intervention"] for r in rows) == INTERVENTIONS, "INTERVENTION_CATALOG_MISMATCH")
        _check(all(r["workload_signature"] == context["workload_signature"] for r in rows), "WORKLOAD_PAIRING_FAILURE")
        work_context = WorkloadSettings.model_validate({**workload.model_dump(), **{
            key: value for key, value in context["knobs"].items() if key in workload.model_dump()}})
        full, fraction = work_context.full_import_units, work_context.berthed_remaining_fraction
        total_work = full * sum(fraction if v["initial_state"] == "berthed" else 1 for v in population)
        _check(isfinite(total_work), "NONFINITE_WORKLOAD_TOTAL")
        reference = None
        parsed = []
        for row in rows:
            params = CoupledParameters.model_validate(row["parameters"])
            outcome = Outcome.model_validate(row["kpis"]).model_dump()
            tolerance = 1e-9 * max(1.0, total_work + params.initial_inventory)
            residuals = {
                "vessel_conservation_residual": total_work - outcome["cargo_remaining_aboard"] - outcome["cargo_unloaded"],
                "yard_conservation_residual": params.initial_inventory + outcome["cargo_unloaded"] - outcome["cargo_cleared"] - outcome["yard_inventory_end"],
                "system_conservation_residual": total_work + params.initial_inventory - outcome["cargo_remaining_aboard"] - outcome["yard_inventory_end"] - outcome["cargo_cleared"],
            }
            ok = all(abs(value) <= tolerance and abs(outcome[key]) <= tolerance
                     and abs(outcome[key] - value) <= tolerance for key, value in residuals.items())
            if not ok:
                issues.append({"code": "CONSERVATION_FAILURE", "context": context["context"], "intervention": row["intervention"],
                               "reported_residuals": {k: outcome[k] for k in residuals}, "recomputed_residuals": residuals})
                return meta, [], {}, issues
            _check(outcome["completed_vessels"] + outcome["unfinished_vessels"] == len(population), "VESSEL_ACCOUNTING_FAILURE")
            _check(abs(outcome["yard_end_occupancy_pct"] - outcome["yard_inventory_end"] / params.yard_capacity * 100) < 1e-7
                   and outcome["yard_peak_occupancy_pct"] >= outcome["yard_end_occupancy_pct"] - 1e-7, "YARD_OCCUPANCY_INCONSISTENCY")
            diagnostic = _diagnostic(row["bottleneck"], request.horizon_hours)
            parsed_row = {"intervention": row["intervention"], "parameters": params.model_dump(), "kpis": outcome,
                          "bottleneck": diagnostic, "conservation": {"status": "PASS" if ok else "FAIL", "tolerance": tolerance, "residuals": residuals}}
            if reference is None:
                reference = parsed_row
            else:
                expected = dict(reference["parameters"])
                change = {
                    "gate_modest": ("gate_units_per_hour", reference["parameters"]["gate_units_per_hour"] * (1 + plan.modest_increase)),
                    "gate_large": ("gate_units_per_hour", reference["parameters"]["gate_units_per_hour"] * plan.large_gate_factor),
                    "unloading_modest": ("unloading_units_per_hour", reference["parameters"]["unloading_units_per_hour"] * (1 + plan.modest_increase)),
                    "berth_plus_one": ("berth_slots", reference["parameters"]["berth_slots"] + 1),
                }[row["intervention"]]
                expected[change[0]] = change[1]
                _check(expected == params.model_dump(), "INTERVENTION_PARAMETER_MISMATCH")
                benefit = _benefit(reference["kpis"], outcome)
                _check(benefit == row["vs_baseline"], "NUMERICAL_COMPARISON_MISMATCH")
                parsed_row["vs_baseline"] = benefit
                parsed_row["bottleneck_changed"] = reference["bottleneck"]["active_constraints"] != diagnostic["active_constraints"]
                parsed_row["parameter_change"] = {"parameter": change[0], "baseline": reference["parameters"][change[0]], "intervention": change[1]}
            parsed.append(parsed_row)
        validated.append({"context": context["context"], "axis": context["axis"], "knobs": deepcopy(context["knobs"]), "comparisons": parsed})
    robustness = summarize_robustness(validated, report["excluded_contexts"])
    _check(robustness == report["robustness"], "ROBUSTNESS_INCONSISTENCY")
    return meta, validated, robustness, issues


def _mapping(candidates: list[dict], available: bool) -> list[dict]:
    mappings = []
    for candidate in candidates:
        key, actions = candidate["candidate_id"], candidate["actions"]
        experiment, status, reason = None, "NOT_EVALUATED", "No equivalent sensitivity intervention was evaluated."
        if available and key == "baseline" and actions == []:
            experiment, status = "baseline", "REFERENCE_ONLY"
            reason = "No-change reference in a different workload/resource model; does not validate the legacy winner."
        elif (available and key == "extra_berth" and actions == [{"type": "add_berths", "value": 1}]
              and type(actions[0]["value"]) is int):
            experiment, status = "berth_plus_one", "CONDITIONAL_SEMANTIC_MATCH"
            reason = "One additional slot matches the action increment only; abstract cohort/resources differ from legacy replay."
        elif key in {"gate_max", "gate_extension"}:
            reason = "Recorded maximum clearance or gate-hour extension is not a normalized rate multiplier; gate experiments remain separate."
        elif key == "shortest_first":
            reason = "Sensitivity did not test queue-policy changes; recorded shortest service is not an estimated-workload test."
        mappings.append({"candidate_id": key, "original_actions": deepcopy(actions), "status": status,
                         "experiment_id": experiment, "reason": reason})
    return mappings


def build_advisory(timestamp, candidates: list[dict], recommendation: dict, cfg: dict,
                   evidence: dict | None = None) -> dict | None:
    raw = cfg.get("strategy_decision_support", {})
    if not isinstance(raw, dict) or raw.get("enabled", False) is not True:
        return None
    winner = recommendation["highest_ranked_candidate"]
    result = {
        "mode": "simulation_advisory_v1", "status": "UNAVAILABLE",
        "decision_timestamp": parse_exact_utc_hour(str(timestamp)).isoformat(),
        "reasoning_mode": "deterministic", "provenance": "SIMULATED",
        "original_winner": winner, "recommendation_preserved": True,
        "operator_review_required": True, "autonomous_execution": False,
        "candidate_experiment_mapping": _mapping(candidates, False), "interventions": [],
        "tradeoffs": [], "issues": [],
        "limitations": [
            "All intervention outcomes are simulated, not observed operational improvements or verified predictions.",
            "Workloads and remaining fractions are synthetic normalized units, not TEU or measured per-call cargo.",
            "One abstract yard; AIS broad-cargo cohort and container-terminal membership remain unverified.",
            "Finite sensitivity coverage is not real-world confidence or a probability.",
            "Legacy replay and advisory workloads/resources differ; evidence does not replace ranking or establish its winner.",
            "An as-of advisory does not relabel the existing historical-replay recommendation as an as-of prediction.",
        ],
    }
    try:
        AdvisorySettings.model_validate(raw)
        if evidence is None:
            result["issues"] = [{"code": "EVIDENCE_NOT_REQUESTED_OR_UNAVAILABLE"}]
            return result
        meta, contexts, robustness, issues = _validate_evidence(evidence, parse_exact_utc_hour(str(timestamp)))
    except (KeyError, TypeError, ValueError, OverflowError, AttributeError, IndexError):
        result.update(status="INSUFFICIENT_EVIDENCE", issues=[{"code": "INVALID_OR_INCOMPLETE_EVIDENCE"}])
        return result
    result.update(status="INSUFFICIENT_EVIDENCE" if issues else "AVAILABLE", issues=issues,
                  time_information_mode=meta["time_information_mode"], hindsight_assisted=meta["hindsight_assisted"],
                  scenario_provenance=deepcopy(meta), units="normalized_units")
    if issues:
        return result
    result["candidate_experiment_mapping"] = _mapping(candidates, True)
    rows = contexts[0]["comparisons"]
    result["baseline"] = deepcopy(rows[0])
    result["bottleneck_evidence"] = deepcopy(rows[0]["bottleneck"])
    descriptions = {
        "yard_capacity_blocking": "In this simulation, limited yard headroom restricts unloading and can extend berth occupation.",
        "gate_clearance_limitation": "Eligible yard cargo exceeds saturated gate service; extra clearance need not improve vessel completion if another constraint binds.",
        "unloading_productivity_limitation": "Pending work consumes the shared unloading budget, or zero productivity prevents service progression.",
        "berth_slot_limitation": "Queued calls cannot use available unloading headroom because effective slots are exhausted under hourly berth reuse.",
    }
    result["binding_constraint_explanations"] = [descriptions[key] for key in rows[0]["bottleneck"]["active_constraints"]]
    report = evidence["report"]
    result["sensitivity_scope"] = {
        "plan": deepcopy(report["plan"]), "ranges_low_base_high": deepcopy(report["ranges_low_base_high"]),
        "tested_contexts": [{"context": c["context"], "axis": c["axis"], "knobs": c["knobs"]} for c in contexts],
        "run_count": report["run_count"], "excluded_contexts": deepcopy(report["excluded_contexts"]),
        "skipped_duplicates": deepcopy(report["skipped_duplicate_contexts"]),
    }
    for row in rows[1:]:
        explanation = "No queue/completion benefit was demonstrated in this base context."
        assessment = "NO_DEMONSTRATED_DELAY_OR_COMPLETION_BENEFIT"
        change = row["vs_baseline"]
        if change["status"] == "tradeoff":
            assessment = "TRADEOFF"
            explanation = "Queue and completion outcomes conflict; no independent winner is declared."
        elif change["status"] == "beneficial":
            assessment = "SIMULATED_COMPLETION_OR_CARGO_SUPPORTED_IMPROVEMENT"
            explanation = "Queue/completion benefit under this context; robustness across other assumptions must also be reviewed."
            if change["vessel_metric_deltas"]["completed_vessels"] == 0 and change["cargo_progress_status"] == "unchanged":
                assessment = "QUEUE_RELIEF_ONLY"
                explanation = "Queue exposure decreases without more completions or unloading; admission relief is not demonstrated throughput recovery."
        elif change["cargo_progress_status"] == "improved":
            assessment = "CARGO_PROGRESS_ONLY"
            explanation = "Cargo progress improves without a queue/completion benefit within this horizon."
        result["interventions"].append({**deepcopy(row), "baseline_kpis": deepcopy(rows[0]["kpis"]),
                                         "robustness": robustness[row["intervention"]], "explanation": explanation,
                                         "advisory_assessment": assessment,
                                         "robustness_interpretation": "Phase 3 queue/completion coverage can include queue-only relief; it does not prove robust throughput gains or real-world confidence.",
                                         "provenance": "SIMULATED"})
    for context in contexts:
        for row in context["comparisons"][1:]:
            if row["vs_baseline"]["status"] in {"tradeoff", "harmful"}:
                result["tradeoffs"].append({"context": context["context"], "assumptions": context["knobs"],
                                           "intervention": row["intervention"], "baseline_kpis": context["comparisons"][0]["kpis"],
                                           "intervention_kpis": row["kpis"], "comparison": row["vs_baseline"]})
    mapping = next((m for m in result["candidate_experiment_mapping"] if m["candidate_id"] == winner), None)
    result["original_recommendation_assessment"] = {
        "candidate_id": winner, "status": "QUALIFIED_ABSTRACT_ANALOGUE" if mapping and mapping["status"] == "CONDITIONAL_SEMANTIC_MATCH" else "CANNOT_VALIDATE_ORIGINAL_WINNER",
        "explanation": "Original recommendation is retained. Separate experiments can qualify assumptions or reveal unsupported effects; no agreement with legacy ranking is inferred.",
    }
    result["summary"] = "Simulation-only advisory: review binding constraints, paired outcomes and assumption sensitivity. The existing Strategy winner is preserved; no operational action is executed."
    return result
