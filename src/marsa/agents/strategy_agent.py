"""Final Strategy Agent: synthesize evidence, generate candidates, and explain ranking."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from marsa.agents.llm import StructuredLLMProvider
from marsa.provenance.types import DataProvenance
from marsa.twin.engine import SUPPORTED_ACTIONS as TWIN_ACTIONS
from marsa.twin.engine import simulate_candidate
from marsa.twin.scoring import score_results

# Actions the Digital Twin can test (berths and queue from AIS, gate from the Cargo agent file).
SUPPORTED_ACTIONS = set(TWIN_ACTIONS)

ARABIC_CANDIDATE_TITLES = {
    "baseline": "عدم إجراء تغيير تشغيلي",
    "shortest_first": "إعطاء الأولوية للسفن ذات مدة الخدمة الأقصر",
    "extra_berth": "فتح رصيف إضافي مؤقت",
    "gate_max": "تشغيل البوابة بأعلى طاقة سجّلها ملف البضائع",
}


class StrategyNarrative(BaseModel):
    model_config = ConfigDict(extra="forbid", frozen=True)
    summary: str


def _candidate(candidate_id: str, title: str, rationale: str,
               evidence_references: list[str], actions: list[dict]) -> dict:
    if any(action.get("type") not in SUPPORTED_ACTIONS for action in actions):
        raise ValueError("Strategy candidate contains an unsupported Digital Twin action")
    return {
        "candidate_id": candidate_id,
        "title": title,
        "rationale": rationale,
        "evidence_references": evidence_references,
        "actions": actions,
        "provenance": DataProvenance.AGENT_GENERATED.value,
    }


# Which signals ask for which twin test. Each candidate lists the signals that triggered it.
# Maritime pressure or a berth shortage in the twin -> berth tests; Cargo pressure or a real HIGH
# six-hour model alert -> gate test; any active external event -> every test (the twin shows which
# one helps). Weather gives context only: the 2025 data showed no clear effect on berth stays.
def _signals(forecast: dict, maritime: dict, cargo: dict, events_weather: dict,
             baseline: dict | None) -> dict[str, list[str]]:
    berth, gate = [], []
    if maritime["maritime_status"] in {"ELEVATED", "CONGESTED"} and \
            maritime["evidence"].get("waiting_vessels", 0) > 0:
        berth.append("maritime.maritime_status")
    if baseline is not None and baseline["kpis"]["capacity_wait_vessel_hours"] > 0:
        berth.append("digital_twin.baseline.kpis.capacity_wait_vessel_hours")
    if cargo["cargo_status"] in {"ELEVATED", "CRITICAL"}:
        gate.append("cargo.cargo_status")
    if (forecast.get("status") == "success" and forecast.get("classification") == "HIGH"
            and forecast.get("alert_active") is True and forecast.get("is_real_model_prediction")):
        gate.append("ml_forecast.alert_active")
    if (events_weather.get("external_event_assessment") or {}).get("active"):
        berth.append("events_weather.external_event_assessment.active")
        gate.append("events_weather.external_event_assessment.active")
    return {"berth": berth, "gate": gate}


# Generate the tests to run in the twin from the three agents, the model and the no-change run.
def generate_candidates(forecast: dict, maritime: dict, cargo: dict,
                        events_weather: dict, baseline: dict | None = None,
                        gate_available: bool = True) -> list[dict]:
    candidates = [_candidate(
        "baseline", "No operational change",
        "Replays the recorded data; required for the what-if comparison.",
        ["digital_twin.scenario"], [],
    )]
    signals = _signals(forecast, maritime, cargo, events_weather, baseline)
    if signals["berth"]:
        candidates.append(_candidate(
            "shortest_first", "Prioritize shorter vessel services",
            "Triggered by: " + ", ".join(signals["berth"]) + ". The twin tests giving the next "
            "free berth to the vessel with the shortest service time.",
            signals["berth"], [{"type": "queue_policy", "value": "shortest_service_first"}],
        ))
        candidates.append(_candidate(
            "extra_berth", "Temporarily open one additional berth",
            "Triggered by: " + ", ".join(signals["berth"]) + ". The twin tests one more open berth.",
            signals["berth"], [{"type": "add_berths", "value": 1}],
        ))
    if signals["gate"] and gate_available:
        candidates.append(_candidate(
            "gate_max", "Run the gate at its highest recorded hourly clearance",
            "Triggered by: " + ", ".join(signals["gate"]) + ". The twin tests clearing, every hour, "
            "the highest number of containers the Cargo agent file recorded leaving in one hour.",
            signals["gate"], [{"type": "gate_capacity", "value": "max_recorded"}],
        ))
    return candidates


def synthesize(forecast: dict, maritime: dict, cargo: dict, events_weather: dict) -> dict:
    """Preserve conflicting signals while identifying supported pressure areas."""
    signals = {
        "ml": {
            "status": forecast["status"],
            "classification": forecast.get("classification"),
            "congestion_score": forecast.get("congestion_score"),
            "is_real_model_prediction": forecast.get("is_real_model_prediction", False),
        },
        "maritime": maritime["maritime_status"],
        "cargo": cargo["cargo_status"],
        "events_weather": events_weather["contextual_pressure"]["level"],
    }
    pressure_areas = []
    if signals["maritime"] in {"ELEVATED", "CONGESTED"}:
        pressure_areas.append("maritime")
    if signals["cargo"] in {"ELEVATED", "CRITICAL"}:
        pressure_areas.append("cargo")
    if signals["events_weather"] in {"moderate", "elevated"}:
        pressure_areas.append("events_weather")
    return {
        "signals": signals,
        "supported_pressure_areas": pressure_areas,
        "interpretation": (
            "Signals are retained independently. Elevated evidence may contribute context "
            "for candidate evaluation but does not establish causality."
        ),
        "provenance": DataProvenance.AGENT_GENERATED.value,
    }


# Run every candidate in the twin and rank by total vessel-hours waiting (no weights).
def evaluate(scenario: object, candidates: list[dict], cfg: dict | None = None,
             baseline: dict | None = None) -> list[dict]:
    raw_results = []
    for candidate in candidates:
        if candidate["candidate_id"] == "baseline" and baseline is not None:
            result = dict(baseline)
        else:
            twin_candidate = {"id": candidate["candidate_id"], "actions": candidate["actions"]}
            result = simulate_candidate(scenario, twin_candidate)
        result["provenance"] = DataProvenance.SIMULATED.value
        raw_results.append(result)
    ranked = score_results(raw_results)
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    for result in ranked:
        candidate = by_id[result["candidate_id"]]
        result.update({
            "title": candidate["title"],
            "rationale": candidate["rationale"],
            "evidence_references": candidate["evidence_references"],
        })
    return ranked


# Arabic summary built only from the twin results (vessel and yard measures reported separately).
def _deterministic_summary(ranked: list[dict]) -> str:
    if len(ranked) == 1:
        return (
            "لا توجد إشارة من الوكلاء أو نموذج التنبؤ تستدعي اختبار قرار، ولا يُظهر التوأم الرقمي "
            "نقصًا في الأرصفة خلال أفق المحاكاة؛ وأي انتظار حالي هو انتظار لموعد الرسو المسجّل في "
            "بيانات AIS، لذلك لا حاجة لتدخل تشغيلي. لا تضمن المحاكاة النتيجة في التشغيل الفعلي."
        )
    base = next(r for r in ranked if r["candidate_id"] == "baseline")
    def title(r):
        return ARABIC_CANDIDATE_TITLES.get(r["candidate_id"], r["title"])
    vessel_best = min(ranked, key=lambda r: r["kpis"]["total_wait_vessel_hours"])
    parts = []
    if vessel_best["kpis"]["total_wait_vessel_hours"] < base["kpis"]["total_wait_vessel_hours"]:
        parts.append(f"خفّض خيار «{title(vessel_best)}» إجمالي ساعات انتظار السفن أكثر من غيره")
    yard_runs = [r for r in ranked if r["kpis"].get("yard_peak_occupancy_pct") is not None]
    if yard_runs and base["kpis"].get("yard_peak_occupancy_pct") is not None:
        yard_best = min(yard_runs, key=lambda r: r["kpis"]["yard_peak_occupancy_pct"])
        if yard_best["kpis"]["yard_peak_occupancy_pct"] < base["kpis"]["yard_peak_occupancy_pct"]:
            parts.append(f"وخفّض خيار «{title(yard_best)}» أعلى امتلاء للساحة أكثر من غيره")
    if not parts:
        return (
            "اختبر التوأم الرقمي الخيارات التي اقترحها الوكيل، ولم يحسّن أيٌّ منها انتظار السفن أو "
            "امتلاء الساحة مقارنة بعدم التغيير. لا تضمن المحاكاة النتيجة في التشغيل الفعلي."
        )
    return (
        "، ".join(parts) + ". تُعرض نتائج السفن والساحة منفصلة، والقرار النهائي لمشغّل الميناء. "
        "تعتمد المحاكاة على بيانات AIS وملف البضائع ولا تضمن النتيجة في التشغيل الفعلي."
    )


def recommend(ranked: list[dict], forecast: dict, agents: dict,
              llm: StructuredLLMProvider | None = None) -> dict:
    synthesis = synthesize(forecast, agents["maritime"], agents["cargo"], agents["events_weather"])
    summary = _deterministic_summary(ranked)
    reasoning_mode = "deterministic"
    if llm is not None:
        try:
            raw: Any = llm.generate_structured(
                system_prompt=(
                    "Write one concise decision-support sentence. Do not use numbers, claim "
                    "causality, alter facts, or describe simulated outcomes as observed. "
                    "Respond in Arabic only (Modern Standard Arabic, professional tone for a port "
                    "operations manager); keep the recommended option name faithful to the summary and translate it to arabaic between parentheses."
                ),
                payload={
                    "deterministic_summary": summary,
                    "domain_statuses": {
                        "maritime": agents["maritime"]["maritime_status"],
                        "cargo": agents["cargo"]["cargo_status"],
                        "events_weather": agents["events_weather"]["contextual_pressure"]["level"],
                    },
                },
                response_model=StrategyNarrative,
            )
            value = raw.model_dump() if isinstance(raw, BaseModel) else raw
            narrative = (StrategyNarrative.model_validate_json(value) if isinstance(value, str)
                         else StrategyNarrative.model_validate(value))
            if re.search(r"\d|\b(?:caus\w*|guarantee\w*|observed outcome|execute|must|should)\b", narrative.summary, re.I) \
                    or re.search(r"يضمن|بالتأكيد|نفّذ فورًا|السبب هو|يجب", narrative.summary):
                raise ValueError("unsafe Strategy narrative")
            summary = narrative.summary
            reasoning_mode = "llm_assisted"
        except Exception:
            reasoning_mode = "deterministic_fallback"

    return {
        "purpose": "Simulation-based candidate evaluation for human review.",
        "highest_ranked_candidate": ranked[0]["candidate_id"],
        "summary": summary,
        "evidence_summary": synthesis,
        "reasoning_mode": reasoning_mode,
        "provenance": DataProvenance.AGENT_GENERATED.value,
        "ranking": [
            {"rank": item["rank"], "candidate_id": item["candidate_id"], "score": item["score"]}
            for item in ranked
        ],
        "human_approval_required": True,
        "autonomous_execution": False,
    }
