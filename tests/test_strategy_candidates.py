"""Strategy Agent candidate rules: actions are tested only when the twin shows a berth shortage."""
from marsa.agents.strategy_agent import _deterministic_summary, generate_candidates

FORECAST = {"status": "ml_unavailable"}
MARITIME = {"maritime_status": "ELEVATED", "evidence": {"waiting_vessels": 12}}
CARGO = {"cargo_status": "CRITICAL", "evidence": {}}
WEATHER = {"contextual_pressure": {"level": "elevated"},
           "external_event_assessment": {"active": True, "type": "demand_surge",
                                         "severity": "severe", "provenance": "SYNTHETIC"}}


def baseline(capacity_wait):
    return {"kpis": {"capacity_wait_vessel_hours": capacity_wait, "waiting_end": 1}}


def ids(base):
    return [c["candidate_id"] for c in generate_candidates(FORECAST, MARITIME, CARGO, WEATHER, base)]


# Domain pressure alone (agents, events, ML) never creates a twin action.
def test_no_berth_shortage_means_baseline_only():
    assert ids(baseline(0.0)) == ["baseline"]
    assert ids(None) == ["baseline"]


def test_berth_shortage_tests_queue_policy_and_extra_berth():
    candidates = generate_candidates(FORECAST, MARITIME, CARGO, WEATHER, baseline(5.5))
    assert [c["candidate_id"] for c in candidates] == ["baseline", "shortest_first", "extra_berth"]
    assert candidates[1]["actions"] == [{"type": "queue_policy", "value": "shortest_service_first"}]
    assert candidates[2]["actions"] == [{"type": "add_berths", "value": 1}]
    assert all(c["provenance"] == "AGENT_GENERATED" for c in candidates)


def test_summary_without_shortage_says_no_intervention():
    summary = _deterministic_summary([{"candidate_id": "baseline", "title": "No operational change"}])
    assert "لا حاجة لتدخل" in summary
    assert "No operational change" not in summary


def test_summary_when_no_option_improves():
    summary = _deterministic_summary([
        {"candidate_id": "baseline", "title": "x"}, {"candidate_id": "extra_berth", "title": "y"}])
    assert "لم يحسّن" in summary


def test_summary_names_best_option_in_arabic():
    summary = _deterministic_summary([
        {"candidate_id": "extra_berth", "title": "Temporarily open one additional berth"},
        {"candidate_id": "baseline", "title": "No operational change"},
        {"candidate_id": "shortest_first", "title": "Prioritize shorter vessel services"},
    ])
    assert "فتح رصيف إضافي مؤقت" in summary
    assert "ثلاثة خيارات" in summary
    assert "Temporarily" not in summary
