"""One frozen Hybrid population for the website's normalized candidate comparisons.

Legacy replay routes and Coupled physics are untouched. Resource interventions vary
only declared parameters; no Cargo/TEU/gate-transaction conversion is performed.
"""
from __future__ import annotations

import json
import time
from collections import OrderedDict
from copy import deepcopy
from dataclasses import dataclass
from threading import Lock
from typing import Literal
from uuid import uuid4

from pydantic import Field, field_validator

from marsa.agents.strategy_agent import StrategyNarrative, _normalized_narrative, synthesize
from marsa.pipeline import collect_evidence
from marsa.twin.coupled import Contract, CoupledParameters, CoupledScenario, simulate_coupled
from marsa.twin.hybrid import HybridRequest, HybridSettings, build_hybrid
from marsa.twin.sensitivity import (
    SensitivityPlan,
    _benefit,
    _signature,
    analyze_frozen_sensitivity,
    diagnose_bottlenecks,
)

OBJECTIVE = "queue_exposure_vessel_hours"
SEMANTICS = "Hybrid queue exposure; replaces AIS berth-window replay waiting. Lower is better; stable baseline-first ties."


class UnifiedSettings(Contract):
    enabled: bool = False
    horizon_hours: int = Field(default=24, gt=0, le=720, strict=True)
    parameters: CoupledParameters | None = None
    gate_modest_factor: float = Field(default=1.2, gt=1, le=2)
    gate_large_factor: float = Field(default=3.0, gt=2, le=10)


class UnifiedAction(Contract):
    type: Literal["queue_policy", "add_slots", "set_slots", "gate_rate"]
    value: str | int | float

    @field_validator("value", mode="before")
    @classmethod
    def no_boolean(cls, value):
        if isinstance(value, bool):
            raise ValueError("boolean actions are invalid")
        return value


class ContextUnavailable(ValueError):
    pass


@dataclass(frozen=True)
class FrozenContext:
    """Serialized immutable snapshot, including evidence, arrivals and workloads."""

    payload: str
    created_at: float


class ContextStore:
    """Bounded local-demo cache. Expired/other-worker contexts fail explicitly."""

    def __init__(self, max_entries=32, ttl_seconds=1800):
        self.entries = OrderedDict()
        self.lock = Lock()
        self.max_entries = max_entries
        self.ttl_seconds = ttl_seconds

    def put(self, payload):
        key = uuid4().hex
        frozen = FrozenContext(json.dumps(payload, allow_nan=False), time.monotonic())
        with self.lock:
            self.entries[key] = frozen
            while len(self.entries) > self.max_entries:
                self.entries.popitem(last=False)
        return key

    def get(self, key):
        with self.lock:
            value = self.entries.get(key)
            if value is None or time.monotonic() - value.created_at > self.ttl_seconds:
                self.entries.pop(key, None)
                raise ContextUnavailable("Frozen scenario expired or unavailable; reload the analysis.")
            return json.loads(value.payload)


STORE = ContextStore()


def changed_scenario(base: CoupledScenario, actions: list[dict]) -> CoupledScenario:
    parameters = base.parameters.model_dump()
    policy = base.queue_policy
    seen = set()
    for raw in actions:
        action = UnifiedAction.model_validate(raw)
        key = "slots" if action.type in {"add_slots", "set_slots"} else action.type
        if key in seen:
            raise ValueError("duplicate/conflicting unified actions")
        seen.add(key)
        value = action.value
        if action.type == "queue_policy":
            if value not in {"fcfs", "shortest_estimated_workload_first"}:
                raise ValueError("unsupported estimated-workload queue policy")
            policy = value
        elif action.type in {"add_slots", "set_slots"}:
            if type(value) is not int or value < 0:
                raise ValueError("slot actions require nonnegative integers")
            parameters["berth_slots"] = parameters["berth_slots"] + value if action.type == "add_slots" else value
        else:
            if type(value) not in {int, float} or value < 0:
                raise ValueError("gate_rate requires nonnegative normalized units/hour")
            parameters["gate_units_per_hour"] = value
    return CoupledScenario.model_validate({**base.model_dump(), "parameters": parameters, "queue_policy": policy})


def candidate_definitions(base, settings):
    return [
        {"candidate_id": "baseline", "title": "بدون تدخل", "actions": []},
        {"candidate_id": "estimated_shortest", "title": "أولوية أقل حمولة مقدّرة", "actions": [{"type": "queue_policy", "value": "shortest_estimated_workload_first"}]},
        {"candidate_id": "slot_plus_one", "title": "إضافة خانة افتراضية", "actions": [{"type": "add_slots", "value": 1}]},
        {"candidate_id": "gate_modest", "title": f"زيادة التخليص المطبّع {(settings.gate_modest_factor - 1) * 100:g}%", "actions": [{"type": "gate_rate", "value": base.parameters.gate_units_per_hour * settings.gate_modest_factor}]},
        {"candidate_id": "gate_large", "title": f"التخليص المطبّع ×{settings.gate_large_factor:g}", "actions": [{"type": "gate_rate", "value": base.parameters.gate_units_per_hour * settings.gate_large_factor}]},
    ]


def validate_result(result, scenario):
    """Evidence guard before ranking or advisory; physics already enforces each step."""
    if (result.get("provenance") != "SIMULATED" or result.get("units") != "normalized_units"
            or result.get("scenario") != scenario.model_dump(mode="json")):
        raise ArithmeticError("unified simulation provenance/scenario mismatch")
    tolerance = 1e-9 * max(1.0, scenario.parameters.initial_inventory + sum(v.initial_import_workload for v in scenario.vessels))
    for key in ("vessel_conservation_residual", "yard_conservation_residual", "system_conservation_residual"):
        value = result["kpis"].get(key)
        if not isinstance(value, (int, float)) or not abs(value) <= tolerance:
            raise ArithmeticError("unified conservation evidence invalid")
    if len(result["timeline"]) != scenario.horizon_hours + 1 or len(result["flows"]) != scenario.horizon_hours:
        raise ArithmeticError("unified ledger length mismatch")
    for index, point in enumerate(result["timeline"]):
        if (point["hour"] != index or not 0 <= point["yard_inventory"] <= scenario.parameters.yard_capacity + tolerance
                or not 0 <= point["at_berth"] <= scenario.parameters.berth_slots):
            raise ArithmeticError("unified capacity evidence invalid")
    for flow in result["flows"]:
        if any(not abs(flow[key]) <= tolerance for key in ("vessel_conservation_residual", "yard_conservation_residual")):
            raise ArithmeticError("unified hourly conservation invalid")


def compare_candidates(base, settings, user_actions=None):
    """Preflight every variant, then simulate identical frozen work in the same engine."""
    definitions = candidate_definitions(base, settings)
    if user_actions is not None:
        # Invalid custom requests are errors, not silently omitted or approximated.
        changed_scenario(base, user_actions)
        definitions.append({"candidate_id": "user_decision", "title": "قرار المستخدم", "actions": user_actions})
    rows, excluded, seen = [], [], set()
    workload_signature = _signature([v.model_dump(mode="json") for v in base.vessels])
    for candidate in definitions:
        try:
            scenario = changed_scenario(base, candidate["actions"])
        except ValueError as error:
            if candidate["candidate_id"] in {"baseline", "user_decision"}:
                raise
            excluded.append({"candidate_id": candidate["candidate_id"], "reason": str(error)})
            continue
        signature = _signature(scenario.model_dump(mode="json"))
        if signature in seen and candidate["candidate_id"] != "user_decision":
            excluded.append({"candidate_id": candidate["candidate_id"], "reason": "Duplicate scenario; no distinct implemented intervention."})
            continue
        seen.add(signature)
        result = simulate_coupled(scenario)
        validate_result(result, scenario)
        if _signature([v.model_dump(mode="json") for v in scenario.vessels]) != workload_signature:
            raise ArithmeticError("candidate population/workload changed")
        rows.append({**result, **candidate, "workload_signature": workload_signature,
                     "bottleneck": diagnose_bottlenecks(result)})
    baseline = next(r for r in rows if r["candidate_id"] == "baseline")
    for row in rows:
        row["comparison"] = _benefit(baseline["kpis"], row["kpis"])
    # Stable generation order is the tie breaker, never execution priority or Gemini.
    ranked = sorted(rows, key=lambda r: r["kpis"][OBJECTIVE])
    for rank, row in enumerate(ranked, 1):
        row.update(rank=rank, ranking_kpi=OBJECTIVE, score=row["kpis"][OBJECTIVE])
    return ranked, excluded


def recommend_unified(rows, evidence, llm=None):
    best = next(r for r in rows if r["candidate_id"] != "user_decision")
    base = next(r for r in rows if r["candidate_id"] == "baseline")
    k = best["kpis"]
    summary = (f"الخيار الأعلى ترتيبًا في المحاكاة هو «{best['title']}»؛ تعرض الانتظار {k[OBJECTIVE]:g} ساعة–سفينة، "
               f"واكتملت {k['completed_vessels']} سفينة وفُرّغت {k['cargo_unloaded']:g} وحدة مطبّعة. "
               "الترتيب حسب انتظار الطابور المحاكى، لا انتظار نوافذ AIS القديمة؛ التعادل يحفظ الأساس. "
               "هذه محاكاة بأحمال اصطناعية ومعرفة لاحقة، وليست فعالية تشغيلية مثبتة.")
    progress = any(r["kpis"]["completed_vessels"] > base["kpis"]["completed_vessels"] or
                   r["kpis"]["cargo_unloaded"] > base["kpis"]["cargo_unloaded"] for r in rows)
    if progress and best["candidate_id"] == "baseline":
        summary += " بعض البدائل تحسن الإكمال أو التفريغ دون خفض الانتظار؛ لا تتغير أفضلية الأساس عند تعادل الانتظار."
    if (best["comparison"]["vessel_metric_deltas"][OBJECTIVE] < 0
            and best["comparison"]["vessel_metric_deltas"]["completed_vessels"] == 0
            and best["comparison"]["cargo_progress_status"] == "unchanged"):
        summary += " انخفاض الطابور هنا تخفيف انتظار فقط؛ لم يتحسن الإكمال أو التفريغ."
    mode = "deterministic"
    if llm is not None:
        try:
            raw = llm.generate_structured(
                system_prompt="Select the supplied allowed summary verbatim. Never change winner or add benefit claims.",
                payload=deepcopy({"authoritative_winner": best["candidate_id"], "allowed_narratives": [{"summary": summary}],
                                  "ranking": [{"candidate_id": r["candidate_id"], "kpis": r["kpis"]} for r in rows]}),
                response_model=StrategyNarrative)
            value = raw.model_dump() if isinstance(raw, StrategyNarrative) else raw
            parsed = StrategyNarrative.model_validate_json(value) if isinstance(value, str) else StrategyNarrative.model_validate(value)
            if _normalized_narrative(parsed.summary) != _normalized_narrative(summary):
                raise ValueError("unverified unified narrative")
            mode = "llm_assisted"
        except Exception:
            mode = "deterministic_fallback"
    user = next((r for r in rows if r["candidate_id"] == "user_decision"), None)
    return {"highest_ranked_candidate": best["candidate_id"], "summary": summary, "reasoning_mode": mode,
            "ranking_kpi": OBJECTIVE, "ranking_semantics": SEMANTICS,
            "recommendation_scope": "Generated candidates only; manual user decisions are compared and ranked separately, as in Main.",
            "ranking": [{"candidate_id": r["candidate_id"], "rank": r["rank"], "score": r["score"]} for r in rows],
            "user_decision_assessment": None if user is None else {"candidate_id": "user_decision", "comparison": user["comparison"]},
            "evidence_summary": synthesize(evidence["ml_forecast"], **evidence["domain_agents"]),
            "human_approval_required": True, "autonomous_execution": False}


def unified_analysis(timestamp, cfg, *, context_id=None, parameters=None, horizon_hours=None,
                     user_actions=None, llm=None, store=STORE):
    settings = UnifiedSettings.model_validate(cfg["unified_hybrid"])
    if context_id:
        frozen = store.get(context_id)
        if parameters is not None or horizon_hours is not None or timestamp != frozen["evidence"]["authoritative_timestamp"]:
            raise ValueError("custom decisions must reuse the exact frozen timestamp and assumptions")
    else:
        ts, snap, forecast, domains, investigation = collect_evidence(timestamp, cfg=cfg, llm=llm)
        params = parameters or settings.parameters
        if params is None:
            raise ValueError("explicit normalized scenario parameters required")
        request = HybridRequest(timestamp_utc=ts.to_pydatetime(), horizon_hours=horizon_hours or settings.horizon_hours,
                                time_information_mode="retrospective", cohort_id="unified-eligible-heuristic-cargo-calls",
                                resource_pool_id="unified-abstract-demo-pool", parameters=params, inventory_source="explicit")
        defaults = HybridSettings.model_validate(cfg["hybrid_normalized"]).workload
        scenario, metadata = build_hybrid(request, defaults=defaults)
        frozen = {"scenario": scenario.model_dump(mode="json"), "metadata": metadata,
                  "settings": settings.model_dump(mode="json"),
                  "evidence": {"authoritative_timestamp": ts.isoformat(), "port_state": snap, "ml_forecast": forecast,
                               "domain_agents": domains, "strategy_investigation": investigation}}
        # No non-JSON scalars escape into the immutable context store.
        frozen = json.loads(json.dumps(frozen, default=str))
        context_id = store.put(frozen)
    base = CoupledScenario.model_validate(frozen["scenario"])
    settings = UnifiedSettings.model_validate(frozen["settings"])
    rows, excluded = compare_candidates(base, settings, user_actions)
    evidence = deepcopy(frozen["evidence"])
    decision = recommend_unified(rows, evidence, llm=llm if user_actions is None else None)
    signature = _signature(frozen["scenario"])
    out = {**evidence, "context_id": context_id, "context_signature": signature,
           "simulation_metadata": deepcopy(frozen["metadata"]), "decision_support": decision,
           "strategy_candidates": [{k: r[k] for k in ("candidate_id", "title", "actions")} for r in rows],
           "excluded_candidates": excluded + [
               {"candidate_id": "gate_max", "reason": "Recorded Cargo clearance is not convertible to a normalized gate rate."},
               {"candidate_id": "shortest_first", "reason": "Recorded service duration is replaced by explicitly estimated-workload ordering."}],
           "digital_twin": {"mode": "unified_hybrid", "provenance": "SIMULATED", "units": "normalized_units",
                            "simulation_horizon_hours": base.horizon_hours, "scenario": base.model_dump(mode="json"),
                            "results": rows, "ranking_semantics": SEMANTICS},
           "human_approval_required": True,
           "limitations": frozen["metadata"]["data_quality_warnings"] + [
               f"{base.parameters.berth_slots} slots are an abstract scenario assumption, not verified available operational berths.",
               "Initial inventory is an explicit normalized scenario assumption; no Cargo/TEU/gate conversion.",
               "Agents and XGBoost provide historical context, not forecasts of this counterfactual scenario.",
               SEMANTICS]}
    if cfg.get("strategy_decision_support", {}).get("enabled", False):
        out["strategy_advisory"] = {"status": "AVAILABLE", "provenance": "SIMULATED", "context_signature": signature,
            "summary": "Validated native paired outcomes; no legacy candidate equivalence or operational effectiveness inferred.",
            "original_winner": decision["highest_ranked_candidate"], "robustness": "NOT_EVALUATED",
            "comparisons": [{"candidate_id": r["candidate_id"], "comparison": r["comparison"],
                             "assessment": ("TRADEOFF" if r["comparison"]["status"] == "tradeoff" else
                                            "QUEUE_RELIEF_ONLY" if r["comparison"]["vessel_metric_deltas"][OBJECTIVE] < 0
                                            and r["comparison"]["vessel_metric_deltas"]["completed_vessels"] == 0
                                            and r["comparison"]["cargo_progress_status"] == "unchanged" else
                                            "CARGO_PROGRESS_ONLY" if r["comparison"]["status"] == "ineffective"
                                            and r["comparison"]["cargo_progress_status"] == "improved" else r["comparison"]["status"]),
                             "bottleneck": r["bottleneck"], "workload_signature": r["workload_signature"]} for r in rows],
            "human_approval_required": True, "autonomous_execution": False}
    return out


def unified_sensitivity(context_id, *, plan=None, store=STORE):
    frozen = store.get(context_id)
    base = CoupledScenario.model_validate(frozen["scenario"])
    report = analyze_frozen_sensitivity(base, frozen["metadata"], plan=plan or SensitivityPlan())
    return {**report, "context_id": context_id, "context_signature": _signature(frozen["scenario"])}
