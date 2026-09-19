"""Final Strategy Agent: synthesize evidence, generate candidates, and explain ranking."""
from __future__ import annotations

import re
from typing import Any

from pydantic import BaseModel, ConfigDict

from marsa.agents.llm import StructuredLLMProvider
from marsa.provenance.types import DataProvenance
from marsa.twin.engine import simulate_candidate
from marsa.twin.scoring import score_results

SUPPORTED_ACTIONS = {
    "queue_policy", "add_berths", "crane_boost", "gate_boost",
    "delay_arrivals", "prioritise_vessel",
}

ARABIC_CANDIDATE_TITLES = {
    "baseline": "عدم إجراء تغيير تشغيلي",
    "shortest_first": "إعطاء الأولوية للسفن ذات مدة الخدمة الأقصر",
    "gate_extension": "زيادة قدرة التخليص عبر البوابات",
    "extra_berth": "إضافة رصيف مؤقت",
    "combined_queue_gate": "الجمع بين تعديل أولوية الانتظار وزيادة قدرة البوابات",
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


def _has_synthetic_arrival_pressure(events_weather: dict, event_impact: dict) -> bool:
    event = events_weather.get("external_event_assessment", {})
    if not event.get("active") or event.get("provenance") != DataProvenance.SYNTHETIC.value:
        return False
    mapping = event_impact.get(event.get("type"), {}).get(event.get("severity"), {})
    return float(mapping.get("arrivals", 1.0)) > 1.0


def _supports_preventive_cargo_test(forecast: dict, cargo_status: str) -> bool:
    return (
        cargo_status == "MODERATE"
        and forecast.get("status") == "success"
        and forecast.get("target") == "operational_cargo_congestion_proxy"
        and forecast.get("classification") == "HIGH"
        and forecast.get("alert_active") is True
        and forecast.get("is_real_model_prediction") is True
    )


def generate_candidates(forecast: dict, maritime: dict, cargo: dict,
                        events_weather: dict, event_impact: dict | None = None) -> list[dict]:
    """Generate a small explainable set without forcing cross-domain consensus."""
    candidates = [_candidate(
        "baseline", "No operational change",
        "Provides the baseline required for paired what-if comparison.",
        ["authoritative current port state"], [],
    )]
    maritime_status = maritime["maritime_status"]
    cargo_status = cargo["cargo_status"]
    waiting = maritime["evidence"]["waiting_vessels"]

    if maritime_status in {"ELEVATED", "CONGESTED"} and waiting > 0:
        candidates.append(_candidate(
            "shortest_first", "Prioritize shorter vessel services",
            "Current Maritime status and waiting-vessel evidence support testing a queue-policy alternative without changing capacity.",
            [
                "maritime.maritime_status",
                "maritime.evidence.waiting_vessels",
                "maritime.evidence.waiting_ratio",
                "maritime.evidence.vessels",
            ],
            [{"type": "queue_policy", "value": "shortest_service_first"}],
        ))
    preventive_cargo_test = _supports_preventive_cargo_test(forecast, cargo_status)
    if cargo_status in {"ELEVATED", "CRITICAL"} or preventive_cargo_test:
        rationale = (
            "Current MODERATE synthetic Cargo evidence identifies the gate domain, and an "
            "active HIGH real six-hour cargo-proxy alert supports preventively testing the "
            "existing gate-capacity adjustment in simulation without implying it is necessary."
            if preventive_cargo_test else
            "Synthetic Cargo state is consistent with elevated cargo-side pressure; the "
            "simulation tests a 30% gate-capacity adjustment."
        )
        evidence_references = [
            "cargo.cargo_status",
            "cargo.evidence.yard_occupancy_percent",
            "cargo.evidence.truck_waiting_time_minutes",
            "cargo.evidence.gate_throughput_last_1h",
            "cargo.evidence.cargo_flow_ratio",
        ]
        if preventive_cargo_test:
            evidence_references.extend([
                "ml_forecast.target",
                "ml_forecast.classification",
                "ml_forecast.congestion_score",
                "ml_forecast.alert_active",
                "ml_forecast.is_real_model_prediction",
            ])
        candidates.append(_candidate(
            "gate_extension", "Increase gate clearance capacity",
            rationale,
            evidence_references,
            [{"type": "gate_boost", "value": 1.3}],
        ))
    if maritime_status == "CONGESTED" and waiting > 0:
        candidates.append(_candidate(
            "extra_berth", "Temporarily add one berth",
            "Current Maritime status is CONGESTED; the simulation tests one additional modeled berth.",
            [
                "maritime.maritime_status",
                "maritime.evidence.waiting_vessels",
                "maritime.evidence.waiting_ratio",
            ],
            [{"type": "add_berths", "value": 1}],
        ))
    candidate_ids = {candidate["candidate_id"] for candidate in candidates}
    synthetic_arrival_pressure = _has_synthetic_arrival_pressure(
        events_weather, event_impact or {},
    )
    if (
        len(candidates) < 4
        and {"shortest_first", "gate_extension"} <= candidate_ids
        and synthetic_arrival_pressure
    ):
        event = events_weather["external_event_assessment"]
        event_mapping = f"config.event_impact.{event['type']}.{event['severity']}.arrivals"
        candidates.append(_candidate(
            "combined_queue_gate", "Combine queue and gate adjustments",
            "Both component candidates are independently supported, and configured synthetic arrival-pressure context supports testing their combined modeled response without attributing current pressure to the event.",
            [
                "maritime.maritime_status",
                "maritime.evidence.waiting_vessels",
                "cargo.cargo_status",
                "events_weather.external_event_assessment.active",
                "events_weather.external_event_assessment.type",
                "events_weather.external_event_assessment.severity",
                event_mapping,
            ],
            [
                {"type": "queue_policy", "value": "shortest_service_first"},
                {"type": "gate_boost", "value": 1.3},
            ],
        ))
    return candidates[:4]


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


def evaluate(scenario: object, candidates: list[dict], cfg: dict) -> list[dict]:
    twin_cfg = cfg["twin"]
    raw_results = []
    for candidate in candidates:
        twin_candidate = {"id": candidate["candidate_id"], "actions": candidate["actions"]}
        result = simulate_candidate(
            scenario, twin_candidate,
            runs=twin_cfg["monte_carlo_runs"], seed=twin_cfg["random_seed"],
        )
        result["provenance"] = DataProvenance.SIMULATED.value
        raw_results.append(result)
    ranked = score_results(raw_results, cfg["scoring"]["weights"])
    by_id = {candidate["candidate_id"]: candidate for candidate in candidates}
    for result in ranked:
        candidate = by_id[result["candidate_id"]]
        result.update({
            "title": candidate["title"],
            "rationale": candidate["rationale"],
            "evidence_references": candidate["evidence_references"],
        })
    return ranked


def _deterministic_summary(ranked: list[dict]) -> str:
    best = ranked[0]
    candidate_title = ARABIC_CANDIDATE_TITLES.get(best["candidate_id"], best["title"])
    option_count = {
        1: "خيار واحد",
        2: "خيارين",
        3: "ثلاثة خيارات",
        4: "أربعة خيارات",
    }.get(len(ranked), f"{len(ranked)} خيارات")
    return (
        f"صنّف التقييم الحتمي لنتائج المحاكاة خيار «{candidate_title}» في المرتبة الأولى "
        f"من بين {option_count}. تقارن هذه النتيجة المخرجات المحاكاة وفق "
        "افتراضات موثقة، ولا تضمن تفوق الخيار في التشغيل الفعلي."
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
