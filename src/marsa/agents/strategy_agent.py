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

# The Digital Twin only simulates sea-side berth use from AIS, so only these two actions exist.
SUPPORTED_ACTIONS = set(TWIN_ACTIONS)

ARABIC_CANDIDATE_TITLES = {
    "baseline": "عدم إجراء تغيير تشغيلي",
    "shortest_first": "إعطاء الأولوية للسفن ذات مدة الخدمة الأقصر",
    "extra_berth": "فتح رصيف إضافي مؤقت",
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


# Candidates come from the twin's own no-change run: actions are only worth testing when the
# twin shows vessels waiting for a berth (capacity wait). Otherwise "no change" is the answer.
def generate_candidates(forecast: dict, maritime: dict, cargo: dict,
                        events_weather: dict, baseline: dict | None = None) -> list[dict]:
    candidates = [_candidate(
        "baseline", "No operational change",
        "Replays the real AIS schedule; required for the what-if comparison.",
        ["digital_twin.scenario"], [],
    )]
    if baseline is None or baseline["kpis"]["capacity_wait_vessel_hours"] <= 0:
        return candidates
    evidence = [
        "digital_twin.baseline.kpis.capacity_wait_vessel_hours",
        "digital_twin.baseline.kpis.waiting_end",
        "maritime.maritime_status",
    ]
    candidates.append(_candidate(
        "shortest_first", "Prioritize shorter vessel services",
        "The no-change run shows vessels waiting for a free berth; the twin tests giving the next "
        "free berth to the vessel with the shortest real service time.",
        evidence, [{"type": "queue_policy", "value": "shortest_service_first"}],
    ))
    candidates.append(_candidate(
        "extra_berth", "Temporarily open one additional berth",
        "The no-change run shows vessels waiting for a free berth; the twin tests one more "
        "open berth.",
        evidence, [{"type": "add_berths", "value": 1}],
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


# Arabic summary built only from the ranking; no numbers from an LLM.
def _deterministic_summary(ranked: list[dict]) -> str:
    if len(ranked) == 1:
        return (
            "لا يُظهر التوأم الرقمي نقصًا في الأرصفة خلال الساعات الأربع والعشرين القادمة؛ "
            "وأي انتظار حالي هو انتظار لموعد الرسو المسجّل في بيانات AIS وليس بسبب امتلاء "
            "الأرصفة، لذلك لا حاجة لتدخل تشغيلي. لا تضمن المحاكاة النتيجة في التشغيل الفعلي."
        )
    best = ranked[0]
    if best["candidate_id"] == "baseline":
        return (
            "يُظهر التوأم الرقمي انتظارًا بسبب نقص الأرصفة في سيناريو الضغط المحدد، لكن لم "
            "يحسّن أي من الخيارات المختبرة إجمالي ساعات انتظار السفن، لذلك بقي خيار «عدم إجراء "
            "تغيير تشغيلي» في المرتبة الأولى. لا تضمن المحاكاة النتيجة في التشغيل الفعلي."
        )
    candidate_title = ARABIC_CANDIDATE_TITLES.get(best["candidate_id"], best["title"])
    option_count = {
        2: "خيارين",
        3: "ثلاثة خيارات",
    }.get(len(ranked), f"{len(ranked)} خيارات")
    return (
        f"صنّف التوأم الرقمي خيار «{candidate_title}» في المرتبة الأولى من بين {option_count} "
        "وفق إجمالي ساعات انتظار السفن. تعتمد المحاكاة على جدول السفن الفعلي من بيانات AIS "
        "وعلى سيناريو الضغط المحدد، ولا تضمن تفوق الخيار في التشغيل الفعلي."
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
