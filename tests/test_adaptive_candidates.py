from __future__ import annotations

from copy import deepcopy

import httpx
import pytest
from fastapi.testclient import TestClient

from marsa.agents import strategy_agent
from marsa.agents.candidate_selection import select_candidates
from marsa.agents.investigation import _hypotheses
from marsa.agents.tools.diagnostics import DiagnosticResult
from marsa.api.main import app
from marsa.config import load_config
from marsa.pipeline import run

TIMESTAMPS = ["2025-07-10T19:00:00Z", "2025-01-12T02:00:00Z", "2025-11-02T08:00:00Z",
              "2025-07-18T10:00:00Z", "2025-08-10T07:00:00Z"]
ENABLED = {"adaptive_candidates": {"enabled": True}}


def _legacy_config():
    cfg = load_config()
    cfg["strategy_investigation"]["enabled"] = False
    cfg["adaptive_candidates"]["enabled"] = False
    return cfg


@pytest.fixture(scope="module")
def historical():
    cfg = _legacy_config()
    cfg["strategy_investigation"].update(enabled=True, use_gemini=False)
    return {ts: run(ts, cfg=deepcopy(cfg), llm=None) for ts in TIMESTAMPS}


@pytest.fixture
def context(historical):
    return deepcopy(historical[TIMESTAMPS[0]])


def select(context, cfg=None):
    return select_candidates(context["strategy_candidates"], context["strategy_investigation"],
                             context["authoritative_timestamp"], context["domain_agents"],
                             cfg if cfg is not None else ENABLED)


def change_metrics(context, tool, **metrics):
    investigation = context["strategy_investigation"]
    for result in investigation["diagnostic_results"]:
        if result["tool_name"] == tool:
            result["metrics"].update(metrics)
    investigation["hypotheses"] = _hypotheses(
        [DiagnosticResult.model_validate(raw) for raw in investigation["diagnostic_results"]],
        context["domain_agents"],
    )


@pytest.mark.parametrize("cfg", [{}, {"adaptive_candidates": {"enabled": False}}])
def test_disabled_is_exact_legacy_with_investigation_enabled(context, cfg):
    before = deepcopy(context)
    candidates, report = select(context, cfg)
    assert candidates is context["strategy_candidates"]
    assert context == before
    assert report is None


def test_elevated_but_draining_gate_is_retained_and_queue_prioritized(context):
    before = deepcopy(context)
    candidates, report = select(context)
    assert report["mode"] == "adaptive"
    priorities = {item["candidate_id"]: item for item in report["candidate_priorities"]}
    assert priorities["shortest_first"]["support"] == "strongly_supported"
    assert priorities["gate_max"]["support"] == "unsupported"
    assert report["conditionally_retained_candidates"] == ["extra_berth", "gate_max"]
    assert report["selected_candidate_ids"] == ["baseline", "shortest_first", "extra_berth", "gate_max"]
    assert not report["excluded_candidates"]
    assert candidates[3]["actions"] == [{"type": "gate_capacity", "value": "max_recorded"}]
    assert context == before


def test_mixed_inventory_and_delay_signals_do_not_suppress_gate(context):
    change_metrics(context, "cargo_balance", inventory_change=10, net_cargo_flow=-2)
    candidates, report = select(context)
    gate = next(item for item in report["candidate_priorities"] if item["candidate_id"] == "gate_max")
    assert gate["support"] == "mixed"
    assert "gate_max" in report["conditionally_retained_candidates"]
    assert "cargo_balance.metrics.net_cargo_flow" in gate["evidence_references"]
    assert "gate_max" in [item["candidate_id"] for item in candidates]


def test_rising_inventory_but_improving_delay_is_mixed(context):
    change_metrics(context, "cargo_balance", inventory_change=10, net_cargo_flow=10)
    _, report = select(context)
    gate = next(item for item in report["candidate_priorities"] if item["candidate_id"] == "gate_max")
    assert gate["support"] == "mixed"


@pytest.mark.parametrize("case", ["absent", "skipped", "missing_tool", "incomplete", "coverage_gap",
                                 "missing_metric", "wrong_scope", "missing_limitations", "future",
                                 "wrong_window", "endpoint", "tampered_hypothesis", "duplicate"])
def test_missing_invalid_or_incomplete_evidence_falls_back_exactly(context, case):
    inv = context["strategy_investigation"]
    if case == "absent":
        context["strategy_investigation"] = None
    elif case == "skipped":
        inv["mode"] = "skipped"
    elif case == "missing_tool":
        inv["diagnostic_results"] = inv["diagnostic_results"][1:]
    elif case == "incomplete":
        inv["diagnostic_results"][0]["status"] = "incomplete"
    elif case == "coverage_gap":
        inv["diagnostic_results"][0]["coverage"]["observed_hours"] = 6
    elif case == "missing_metric":
        inv["diagnostic_results"][0]["metrics"]["net_cargo_flow"] = None
    elif case == "wrong_scope":
        inv["diagnostic_results"][0]["source"]["geographic_scope"] = "REGIONAL_LA_LB_UNVERIFIED_MEMBERSHIP"
    elif case == "missing_limitations":
        inv["diagnostic_results"][0]["source"]["limitations"] = []
    elif case == "future":
        inv["diagnostic_results"][0]["timestamp_utc"] = "2025-07-10T20:00:00+00:00"
    elif case == "wrong_window":
        inv["diagnostic_results"][0]["window_hours"] = 12
    elif case == "endpoint":
        inv["diagnostic_results"][0]["current_evidence"]["containers_in_yard"] = 1
    elif case == "tampered_hypothesis":
        inv["hypotheses"][0]["state"] = "supported"
    else:
        inv["diagnostic_results"].append(deepcopy(inv["diagnostic_results"][0]))
    original = deepcopy(context["strategy_candidates"])
    candidates, report = select(context)
    assert candidates == original
    assert report["mode"] == "legacy_fallback"
    assert report["fallback_reason"]
    assert all(item["support"] in {"reference", "insufficient"} for item in report["candidate_priorities"])


@pytest.mark.parametrize("settings", [{"enabled": "yes"}, {"enabled": True, "max_candidates": 5}, None])
def test_invalid_config_is_contained(context, settings):
    candidates, report = select(context, {"adaptive_candidates": settings})
    assert candidates == context["strategy_candidates"]
    assert report["mode"] == "legacy_fallback"


def test_no_additional_history_or_llm_access_in_priority_policy(context, monkeypatch):
    def fail(*args, **kwargs):
        raise AssertionError("Policy must use already-validated evidence only")

    monkeypatch.setattr("pandas.read_csv", fail)
    context["strategy_investigation"]["llm_preference"] = {"candidate": "extra_berth", "confidence": 100}
    _, report = select(context)
    assert report["mode"] == "adaptive"
    assert report["selected_candidate_ids"] == ["baseline", "shortest_first", "extra_berth", "gate_max"]


def test_extra_berth_never_promoted_by_regional_queue_trend_alone(historical):
    context = deepcopy(historical[TIMESTAMPS[4]])
    change_metrics(context, "maritime_history", waiting_vessel_count_change=2, waiting_ratio_change=.02)
    _, report = select(context)
    berth = next(item for item in report["candidate_priorities"] if item["candidate_id"] == "extra_berth")
    assert berth["support"] == "mixed"
    assert berth["priority_band"] == 3
    assert "regional" in berth["selection_rationale"]
    assert "extra_berth" in report["conditionally_retained_candidates"]
    assert report["budget"] == {"limit": 4, "available": 4, "selected": 4, "at_limit": True}


# The combined queue+gate candidate was removed with the old twin (it needed config.event_impact and
# gate_boost); an active event now tests every twin action, which still fits the four-candidate budget.
def test_active_event_pool_fits_budget(context):
    event = {"contextual_pressure": {"level": "moderate"},
             "external_event_assessment": {"active": True, "type": "demand_surge",
                                            "severity": "minor", "provenance": "SYNTHETIC"}}
    domains = context["domain_agents"]
    context["strategy_candidates"] = strategy_agent.generate_candidates(
        context["ml_forecast"], domains["maritime"], domains["cargo"], event,
        investigation=context["strategy_investigation"],
    )
    candidates, report = select(context)
    assert report["available_candidate_ids"] == ["baseline", "shortest_first", "extra_berth", "gate_max"]
    assert candidates[0]["candidate_id"] == "baseline"
    assert report["budget"]["at_limit"]
    assert report["not_selected_due_to_budget"] == report["excluded_candidates"] == []


@pytest.mark.parametrize("timestamp", TIMESTAMPS)
def test_three_modes_preserve_physics_actions_forecast_and_ranking(historical, timestamp, monkeypatch):
    legacy = run(timestamp, cfg=_legacy_config())
    investigated = historical[timestamp]
    cfg = _legacy_config()
    cfg["strategy_investigation"].update(enabled=True, use_gemini=False)
    cfg["adaptive_candidates"]["enabled"] = True
    executions = []
    simulate = strategy_agent.simulate_candidate

    def spy(scenario, candidate, **kwargs):
        executions.append(deepcopy(candidate))
        return simulate(scenario, candidate, **kwargs)

    monkeypatch.setattr(strategy_agent, "simulate_candidate", spy)
    adaptive = run(timestamp, cfg=cfg, llm=None)
    selection = adaptive.pop("candidate_selection")
    assert [item["id"] for item in executions] == selection["selected_candidate_ids"]
    original_by_id = {item["candidate_id"]: item for item in investigated["strategy_candidates"]}
    assert all(item == original_by_id[item["candidate_id"]] for item in adaptive["strategy_candidates"])
    assert all(item["actions"] == original_by_id[item["id"]]["actions"] for item in executions)
    assert adaptive["domain_agents"] == legacy["domain_agents"]
    assert adaptive["ml_forecast"] == legacy["ml_forecast"]
    assert adaptive["digital_twin"] == legacy["digital_twin"]
    assert adaptive["deterministic_evaluation"] == legacy["deterministic_evaluation"]
    assert adaptive["decision_support"] == legacy["decision_support"]
    assert adaptive["strategy_investigation"] == investigated["strategy_investigation"]
    adaptive["strategy_candidates"].sort(key=lambda item: selection["available_candidate_ids"].index(item["candidate_id"]))
    assert adaptive == investigated
    assert selection["evaluation_order_changed"] == (timestamp in TIMESTAMPS[3:])
    assert set(selection["selected_candidate_ids"]) == set(selection["available_candidate_ids"])


def test_reordering_does_not_change_equal_score_tie_break(context, monkeypatch):
    candidates = context["strategy_candidates"]
    original_order = [item["candidate_id"] for item in candidates]
    prioritized = [candidates[0], candidates[2], candidates[1], *candidates[3:]]
    cfg = load_config()

    def fake_simulation(scenario, candidate, **kwargs):
        value = 10 if candidate["id"] == "baseline" else 1
        return {"candidate_id": candidate["id"], "actions": candidate["actions"],
                "kpis": {"total_wait_vessel_hours": value}}

    monkeypatch.setattr(strategy_agent, "simulate_candidate", fake_simulation)
    expected = strategy_agent.evaluate(None, candidates, cfg)
    actual = strategy_agent.evaluate(None, prioritized, cfg, comparison_order=original_order)
    assert actual == expected
    assert actual[0]["candidate_id"] == "shortest_first"


def test_investigation_disabled_and_gemini_failure_paths():
    cfg = _legacy_config()
    cfg["adaptive_candidates"]["enabled"] = True
    legacy = run(TIMESTAMPS[0], cfg=_legacy_config())
    fallback = run(TIMESTAMPS[0], cfg=cfg, llm=None)
    assert fallback.pop("candidate_selection")["fallback_reason"] == "investigation_unavailable"
    assert fallback == legacy

    class FailedProvider:
        def generate_structured(self, **kwargs):
            raise httpx.ReadTimeout("Controlled provider failure")

    cfg["strategy_investigation"]["enabled"] = True
    result = run(TIMESTAMPS[3], cfg=cfg, llm=FailedProvider())
    assert result["strategy_investigation"]["mode"] == "deterministic_fallback"
    assert result["candidate_selection"]["mode"] == "adaptive"
    assert result["candidate_selection"]["evaluation_order_changed"]
    assert result["decision_support"]["autonomous_execution"] is False


def test_enabled_api_preserves_existing_fields(historical, monkeypatch):
    import marsa.api.routes.decision_support as route
    import marsa.pipeline as pipeline

    cfg = load_config()
    cfg["strategy_investigation"].update(enabled=True, use_gemini=False)
    cfg["adaptive_candidates"]["enabled"] = True
    monkeypatch.setattr(pipeline, "load_config", lambda: cfg)
    monkeypatch.setattr(route, "get_strategy_provider", lambda: None)
    with TestClient(app) as client:
        response = client.post("/api/decision-support/analyze", json={"timestamp_utc": TIMESTAMPS[3]})
    assert response.status_code == 200
    result = response.json()
    assert result["candidate_selection"]["evaluation_order_changed"]
    assert result["domain_agents"] == historical[TIMESTAMPS[3]]["domain_agents"]
    assert result["deterministic_evaluation"] == historical[TIMESTAMPS[3]]["deterministic_evaluation"]
    assert result["decision_support"]["highest_ranked_candidate"] == \
        historical[TIMESTAMPS[3]]["decision_support"]["highest_ranked_candidate"]
