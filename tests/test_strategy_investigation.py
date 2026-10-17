from __future__ import annotations

import hashlib
import json
from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient
from pydantic import SecretStr

from marsa.agents.gemini import GeminiStructuredProvider
from marsa.agents.investigation import InvestigationPlan, run_investigation
from marsa.agents.tools.diagnostics import DiagnosticArguments, DiagnosticTools
from marsa.api.main import app
from marsa.config import load_config
from marsa.pipeline import run

TS = "2025-07-10T19:00:00Z"
DOMAINS = {
    "cargo": {"cargo_status": "ELEVATED", "evidence": {}},
    "maritime": {"maritime_status": "ELEVATED", "evidence": {}},
    "events_weather": {"contextual_pressure": {"level": "unknown"}},
}
FORECAST = {"status": "success", "classification": "LOW"}
CONFIG = {"strategy_investigation": {"enabled": True}}
TOOLS = {"cargo_progression": "cargo_balance", "cargo_delays": "cargo_delay_profile",
         "maritime_progression": "maritime_history"}


def _legacy_config():
    cfg = load_config()
    cfg["strategy_investigation"]["enabled"] = False
    cfg["adaptive_candidates"]["enabled"] = False
    return cfg


def selection(question):
    return {"question_id": question, "selected_tool": TOOLS[question],
            "arguments": {"window_hours": 6}}


def plan(*questions, decision="CONTINUE"):
    return {"decision": decision, "selections": [selection(question) for question in questions]}


class Provider:
    def __init__(self, *answers):
        self.answers = list(answers)
        self.payloads = []

    def generate_structured(self, *, system_prompt, payload, response_model):
        assert response_model is InvestigationPlan
        self.payloads.append(deepcopy(payload))
        answer = self.answers.pop(0)
        if isinstance(answer, Exception):
            raise answer
        return answer


class Tools:
    def __init__(self, results):
        self.results = deepcopy(results)
        self.calls = []

    def execute(self, tool, timestamp, arguments):
        self.calls.append((tool, timestamp, arguments.window_hours))
        return self.results[tool]


@pytest.fixture(scope="module")
def diagnostic_results():
    tools = DiagnosticTools()
    return {tool: tools.execute(tool, TS, DiagnosticArguments(window_hours=6))
            for tool in TOOLS.values()}


@pytest.fixture
def tools(diagnostic_results):
    return Tools(diagnostic_results)


def investigate(tools, provider=None, cfg=None, domains=None):
    return run_investigation(TS, FORECAST, domains or deepcopy(DOMAINS),
                             cfg or deepcopy(CONFIG), llm=provider, tools=tools)


def test_adaptive_second_round_receives_verified_evidence(tools):
    provider = Provider(json.dumps(plan("cargo_progression")),
                        InvestigationPlan.model_validate(plan("maritime_progression", "cargo_delays")))
    before = deepcopy(DOMAINS)
    result = investigate(tools, provider)
    assert before == DOMAINS
    assert result["mode"] == "gemini"
    assert result["rounds_used"] == 2
    assert result["tool_calls_used"] == 3
    assert len(provider.payloads[0]["diagnostic_results"]) == 0
    second = provider.payloads[1]
    assert second["diagnostic_results"][0]["metrics"]["inventory_change"] == pytest.approx(-28.01)
    assert second["hypotheses"]
    assert second["remaining_budget"] == {"rounds": 1, "tool_calls": 2}
    assert "cargo_progression" not in {q["question_id"] for q in second["questions"]}
    assert all(item["origin"] == "gemini" for item in result["tool_selections"])
    assert result["unresolved_questions"] == [{"question_id": "weather_context",
                                               "reason": "no_weather_diagnostic_in_catalog"}]


def test_followup_can_depend_on_first_result_with_initial_inputs_held_constant(diagnostic_results):
    """Prove loop feedback, not Gemini reasoning, using labelled counterfactuals."""
    class ConditionalPlanner:
        def __init__(self):
            self.payloads = []

        def generate_structured(self, *, payload, response_model, **kwargs):
            assert response_model is InvestigationPlan
            self.payloads.append(deepcopy(payload))
            if not payload["diagnostic_results"]:
                return plan("cargo_progression")
            first = payload["diagnostic_results"][0]
            assert first["status"] == "complete"
            return plan("cargo_delays" if first["metrics"]["inventory_change"] > 0
                        else "maritime_progression")

    runs = []
    for delta in (10.0, -10.0):
        controlled = Tools(diagnostic_results)
        raw = controlled.results["cargo_balance"].model_dump()
        raw["metrics"].update(inventory_change=delta, net_cargo_flow=delta,
                              yard_occupancy_change=delta / 100)
        raw["direction"] = "rising" if delta > 0 else "falling"
        raw["source"]["limitations"].append("Counterfactual test fixture; not historical measurements.")
        controlled.results["cargo_balance"] = type(controlled.results["cargo_balance"]).model_validate(raw)
        planner = ConditionalPlanner()
        result = investigate(controlled, planner)
        assert result["mode"] == "gemini"
        assert result["rounds_used"] == result["tool_calls_used"] == 2
        assert len(planner.payloads) == 2
        assert planner.payloads[1]["diagnostic_results"][0]["metrics"]["inventory_change"] == delta
        assert planner.payloads[1]["hypotheses"][0]["state"] == (
            "supported" if delta > 0 else "unsupported")
        runs.append((planner.payloads, controlled.calls))
    assert runs[0][0][0] == runs[1][0][0]
    for key in runs[0][0][1]:
        if key not in {"diagnostic_results", "hypotheses"}:
            assert runs[0][0][1][key] == runs[1][0][1][key]
    assert [call[0] for call in runs[0][1]] == ["cargo_balance", "cargo_delay_profile"]
    assert [call[0] for call in runs[1][1]] == ["cargo_balance", "maritime_history"]


def test_no_provider_deterministic_router_and_hypotheses(tools):
    result = investigate(tools)
    assert result["mode"] == "deterministic_fallback"
    assert [call[0] for call in tools.calls] == list(TOOLS.values())
    hypotheses = {item["hypothesis_id"]: item for item in result["hypotheses"]}
    assert hypotheses["cargo_high_but_recently_draining"]["state"] == "supported"
    assert hypotheses["cargo_accumulating"]["state"] == "unsupported"
    assert hypotheses["cargo_delays_worsening"]["state"] == "unsupported"
    assert hypotheses["maritime_waiting_pressure_rising"]["state"] == "supported"
    assert hypotheses["cargo_accumulating"]["contradicting_evidence_ids"]
    for hypothesis in hypotheses.values():
        assert hypothesis["diagnostic_evidence_ids"]
        assert hypothesis["supporting_evidence_ids"] or hypothesis["contradicting_evidence_ids"]


@pytest.mark.parametrize("bad", [
    "not json", {}, {"decision": "OVERRIDE", "selections": []},
    {"decision": "CONTINUE", "selections": []},
    {"decision": "NO_TOOL_NEEDED", "selections": [selection("cargo_progression")]},
    {"decision": "CONTINUE", "selections": [{**selection("cargo_progression"), "selected_tool": "shell"}]},
    {"decision": "CONTINUE", "selections": [{**selection("cargo_progression"), "question_id": "invented"}]},
    {"decision": "CONTINUE", "selections": [{**selection("cargo_progression"), "selected_tool": "maritime_history"}]},
    {"decision": "CONTINUE", "selections": [{**selection("cargo_progression"), "arguments": {"window_hours": 7}}]},
    {"decision": "CONTINUE", "selections": [{**selection("cargo_progression"), "arguments": {"window_hours": "6"}}]},
    {"decision": "CONTINUE", "selections": [{**selection("cargo_progression"), "arguments": {"window_hours": 6, "path": "secret"}}]},
    {"decision": "CONTINUE", "selections": [{**selection("cargo_progression"), "timestamp": "future"}]},
    {**plan("cargo_progression"), "cargo_status": "NORMAL"},
    plan("cargo_progression", "cargo_progression"),
    plan("cargo_progression", "cargo_delays", "maritime_progression", "cargo_progression"),
])
def test_invalid_plan_falls_back_without_partial_or_duplicate_execution(tools, bad):
    result = investigate(tools, Provider(bad))
    assert result["mode"] == "deterministic_fallback"
    assert result["planner_errors"] == [{"round": 1, "code": "provider_or_plan_invalid"}]
    assert len(tools.calls) == len(set(tools.calls)) == 3
    assert all(item["origin"] == "deterministic_fallback" for item in result["tool_selections"])
    assert "secret" not in json.dumps(result)


def test_duplicate_across_rounds_rejected_and_router_selects_remaining(tools):
    provider = Provider(plan("cargo_progression"), plan("cargo_progression"))
    result = investigate(tools, provider)
    assert len(tools.calls) == len(set(tools.calls)) == 3
    assert result["mode"] == "deterministic_fallback"
    assert result["planner_errors"][0]["round"] == 2


@pytest.mark.parametrize("failure", [RuntimeError("sensitive provider response"),
                                      httpx.ReadTimeout("sensitive timeout")])
def test_provider_failure_is_sanitized_and_falls_back(tools, failure):
    result = investigate(tools, Provider(failure))
    assert result["mode"] == "deterministic_fallback"
    assert len(tools.calls) == 3
    assert "sensitive" not in json.dumps(result)


@pytest.mark.parametrize("decision", ["NO_TOOL_NEEDED", "INSUFFICIENT_EVIDENCE"])
def test_explicit_terminal_decision(tools, decision):
    result = investigate(tools, Provider(plan(decision=decision)))
    assert not tools.calls
    assert result["stopping_reason"] == decision
    assert result["mode"] == "gemini"
    assert len(result["unresolved_questions"]) == 4


@pytest.mark.parametrize("rounds,calls,expected", [(0, 3, 0), (2, 0, 0), (1, 3, 2), (2, 1, 1), (2, 2, 2)])
def test_budgets_are_enforced(tools, rounds, calls, expected):
    cfg = {"strategy_investigation": {"enabled": True, "max_rounds": rounds, "max_tool_calls": calls}}
    result = investigate(tools, cfg=cfg)
    assert result["tool_calls_used"] == len(tools.calls) == expected
    assert result["rounds_used"] <= rounds
    assert result["unresolved_questions"]


def test_provider_batch_exceeding_remaining_budget_falls_back(tools):
    cfg = {"strategy_investigation": {"enabled": True, "max_tool_calls": 1}}
    result = investigate(tools, Provider(plan("cargo_progression", "cargo_delays")), cfg=cfg)
    assert result["mode"] == "deterministic_fallback"
    assert len(tools.calls) == 1


@pytest.mark.parametrize("setting", [{"max_rounds": 3}, {"max_tool_calls": 4}, {"window_hours": 12},
                                      {"max_tool_calls": True}, {"enabled": "yes"}, {"extra": 1}])
def test_config_cannot_raise_hard_caps_or_coerce_types(tools, setting):
    result = investigate(tools, cfg={"strategy_investigation": {"enabled": True, **setting}})
    assert result["mode"] == "skipped"
    assert result["stopping_reason"] == "invalid_investigation_configuration"
    assert not tools.calls


def test_feature_disabled_or_missing_does_not_call_provider_or_tools(tools):
    for cfg in ({}, {"strategy_investigation": {"enabled": False}}):
        assert run_investigation(TS, FORECAST, DOMAINS, cfg, Provider(), tools) is None
    assert not tools.calls


def test_gemini_switch_and_no_question_skip(tools):
    cfg = {"strategy_investigation": {"enabled": True, "use_gemini": False}}
    assert investigate(tools, Provider(), cfg=cfg)["mode"] == "deterministic_fallback"
    quiet = deepcopy(DOMAINS)
    quiet["cargo"]["cargo_status"] = quiet["maritime"]["maritime_status"] = "NORMAL"
    tools.calls.clear()
    assert investigate(tools, Provider(), domains=quiet)["mode"] == "skipped"
    assert not tools.calls


def test_normal_current_cargo_with_high_forecast_is_investigated(tools):
    domains = deepcopy(DOMAINS)
    domains["cargo"]["cargo_status"] = "NORMAL"
    result = run_investigation(TS, {"status": "success", "classification": "HIGH"},
                               domains, CONFIG, tools=tools)
    assert "cargo_progression" in [q["question_id"] for q in result["questions_identified"]]
    assert domains["cargo"]["cargo_status"] == "NORMAL"


@pytest.mark.parametrize("corruption", ["throw", "wrong_timestamp", "wrong_tool", "endpoint_mismatch"])
def test_tool_failures_and_source_mismatch_remain_nonfatal(tools, corruption):
    domains = deepcopy(DOMAINS)
    if corruption == "throw":
        def execute(*args):
            raise RuntimeError("sensitive tool failure")
        tools.execute = execute
    elif corruption == "endpoint_mismatch":
        domains["cargo"]["evidence"]["containers_in_yard"] = 1
    else:
        field, value = (("timestamp_utc", "2025-07-10T20:00:00+00:00") if corruption == "wrong_timestamp"
                        else ("tool_name", "maritime_history"))
        tools.results["cargo_balance"] = tools.results["cargo_balance"].model_copy(update={field: value})
    result = investigate(tools, domains=domains)
    assert result["diagnostic_results"][0]["status"] == "error"
    assert result["hypotheses"][0]["state"] == "insufficient"
    assert result["tool_calls_used"] == 3
    assert "sensitive" not in json.dumps(result)


def test_incomplete_and_mixed_hypotheses_have_evidence_references(tools):
    original = tools.results["cargo_balance"]
    tools.results["cargo_balance"] = original.model_copy(update={
        "metrics": {**original.metrics, "inventory_change": 10, "net_cargo_flow": -2}})
    result = investigate(tools)
    assert result["hypotheses"][0]["state"] == "mixed"
    assert result["hypotheses"][0]["supporting_evidence_ids"]
    assert result["hypotheses"][0]["contradicting_evidence_ids"]
    tools.results["cargo_balance"] = original.model_copy(update={"status": "incomplete"})
    result = investigate(tools)
    assert result["hypotheses"][0]["state"] == "insufficient"
    assert result["hypotheses"][0]["insufficient_evidence_ids"] == ["cargo_balance.coverage"]
    assert {"question_id": "cargo_progression", "reason": "diagnostic_incomplete"} in result["unresolved_questions"]


def test_provider_cannot_mutate_original_assessments(tools):
    class MutatingProvider:
        def generate_structured(self, *, payload, **kwargs):
            payload["domain_assessments"]["cargo"]["cargo_status"] = "NORMAL"
            return plan(decision="NO_TOOL_NEEDED")
    domains = deepcopy(DOMAINS)
    investigate(tools, MutatingProvider(), domains=domains)
    assert domains == DOMAINS


def test_existing_gemini_transport_accepts_flattened_plan_schema(tools):
    requests = []

    def handler(request):
        body = json.loads(request.content)
        requests.append(body)
        schema = body["generationConfig"]["responseSchema"]
        assert "$ref" not in json.dumps(schema)
        assert "$defs" not in schema
        assert "additionalProperties" not in json.dumps(schema)
        answer = plan("cargo_progression") if len(requests) == 1 else plan(decision="NO_TOOL_NEEDED")
        return httpx.Response(200, json={"candidates": [{"content": {"parts": [{"text": json.dumps(answer)}]}}]})

    with httpx.Client(transport=httpx.MockTransport(handler)) as client:
        provider = GeminiStructuredProvider(SecretStr("mock-key"), client=client)
        result = investigate(tools, provider)
    assert result["mode"] == "gemini"
    assert result["tool_calls_used"] == 1
    assert len(requests) == 2


# Output fingerprints with investigation disabled, regenerated after the AIS-driven twin merge.
BASELINES = {
    "2025-07-10T19:00:00Z": "ccb43b7597282d4dcd40c68291ef42f30ec78b05ac0a0732b8d06fa6326d842c",
    "2025-01-12T02:00:00Z": "7d9053298426aaac67c7bf641eda5852c421178f3f0242ad53f1c17ffab1418d",
    "2025-11-02T08:00:00Z": "9f4effd73594f1fe622e65bf5b97335b9040c35c341f6757f0d8d7cc37a0d8cd",
}


@pytest.mark.parametrize("timestamp", BASELINES)
def test_three_historical_runs_preserve_all_legacy_outputs(timestamp):
    cfg = _legacy_config()
    legacy = run(timestamp, cfg=cfg)
    digest = hashlib.sha256(json.dumps(legacy, sort_keys=True, ensure_ascii=True).encode()).hexdigest()
    assert digest == BASELINES[timestamp]
    assert "strategy_investigation" not in legacy
    cfg["strategy_investigation"].update(enabled=True, use_gemini=False)
    enhanced = run(timestamp, cfg=cfg)
    investigation = enhanced.pop("strategy_investigation")
    assert investigation["mode"] == "deterministic_fallback"
    assert all(result["status"] == "complete" for result in investigation["diagnostic_results"])
    assert investigation["tool_calls_used"] <= 3
    for candidate in enhanced["strategy_candidates"]:
        justification = candidate.pop("diagnostic_justification")
        assert justification["selection_policy"] == "legacy_preserved"
    assert enhanced == legacy


def test_enabled_api_retains_contract_and_exposes_optional_section(monkeypatch):
    import marsa.api.routes.decision_support as route
    import marsa.pipeline as pipeline

    cfg = load_config()
    cfg["strategy_investigation"].update(enabled=True, use_gemini=False)
    monkeypatch.setattr(pipeline, "load_config", lambda: cfg)
    monkeypatch.setattr(route, "get_strategy_provider", lambda: None)
    with TestClient(app) as client:
        response = client.post("/api/decision-support/analyze", json={"timestamp_utc": TS})
    assert response.status_code == 200
    result = response.json()
    assert result["architecture"]["agents"] == ["maritime", "cargo", "events_weather", "strategy"]
    assert result["strategy_investigation"]["tool_calls_used"] == 3
    assert result["domain_agents"]["cargo"]["cargo_status"] == "ELEVATED"
    assert result["human_approval_required"] is True
    assert result["decision_support"]["autonomous_execution"] is False


def test_mocked_gemini_full_pipeline_keeps_actions_and_simulation():
    class PipelineProvider:
        def __init__(self):
            self.planning_calls = 0
            self.narrative_calls = 0

        def generate_structured(self, *, payload, response_model, **kwargs):
            if response_model is InvestigationPlan:
                self.planning_calls += 1
                if self.planning_calls == 1:
                    return plan("cargo_progression")
                assert payload["diagnostic_results"][0]["tool_name"] == "cargo_balance"
                return plan("cargo_delays", "maritime_progression")
            self.narrative_calls += 1
            return {"summary": "تقييم محاكاة للمراجعة البشرية."}

    legacy = run(TS, cfg=_legacy_config())
    cfg = _legacy_config()
    cfg["strategy_investigation"]["enabled"] = True
    provider = PipelineProvider()
    enhanced = run(TS, cfg=cfg, llm=provider)
    investigation = enhanced.pop("strategy_investigation")
    assert investigation["mode"] == "gemini"
    assert investigation["tool_calls_used"] == 3
    assert provider.planning_calls == 2
    assert provider.narrative_calls == 1
    for candidate in enhanced["strategy_candidates"]:
        candidate.pop("diagnostic_justification")
    for key in ("summary", "reasoning_mode"):
        enhanced["decision_support"].pop(key)
        legacy["decision_support"].pop(key)
    assert enhanced == legacy


def test_full_pipeline_provider_failure_preserves_legacy_deterministic_result():
    class FailingProvider:
        def generate_structured(self, **kwargs):
            raise httpx.ReadTimeout("private provider details")

    legacy = run(TS, cfg=_legacy_config())
    cfg = _legacy_config()
    cfg["strategy_investigation"]["enabled"] = True
    enhanced = run(TS, cfg=cfg, llm=FailingProvider())
    investigation = enhanced.pop("strategy_investigation")
    assert investigation["mode"] == "deterministic_fallback"
    assert investigation["tool_calls_used"] == 3
    assert enhanced["decision_support"]["reasoning_mode"] == "deterministic_fallback"
    enhanced["decision_support"]["reasoning_mode"] = "deterministic"
    for candidate in enhanced["strategy_candidates"]:
        candidate.pop("diagnostic_justification")
    assert enhanced == legacy


def test_unexpected_optional_error_is_contained(tools, monkeypatch):
    import marsa.agents.investigation as module

    def fail(*args):
        raise RuntimeError("private failure")

    monkeypatch.setattr(module, "_investigate", fail)
    result = investigate(tools)
    assert result["mode"] == "skipped"
    assert result["stopping_reason"] == "investigation_unavailable"
    assert "private" not in json.dumps(result)
