"""Strategy Agent candidate rules: tests come from the three agents, the model and the twin."""
from marsa.agents.strategy_agent import _deterministic_summary, generate_candidates

NO_ML = {"status": "ml_unavailable"}
HIGH_ML = {"status": "success", "classification": "HIGH", "alert_active": True,
           "is_real_model_prediction": True}


def maritime(status="NORMAL", waiting=0):
    return {"maritime_status": status, "evidence": {"waiting_vessels": waiting}}


def cargo(status="NORMAL"):
    return {"cargo_status": status, "evidence": {}}


def events(active=False):
    return {"contextual_pressure": {"level": "low"},
            "external_event_assessment": {"active": active, "provenance": "SYNTHETIC"}}


def baseline(capacity_wait=0.0):
    return {"kpis": {"capacity_wait_vessel_hours": capacity_wait}}


def ids(*args, **kw):
    return [c["candidate_id"] for c in generate_candidates(*args, **kw)]


def test_no_signal_means_baseline_only():
    assert ids(NO_ML, maritime(), cargo(), events(), baseline()) == ["baseline"]


def test_maritime_pressure_tests_berth_actions():
    assert ids(NO_ML, maritime("ELEVATED", 3), cargo(), events(), baseline()) == \
        ["baseline", "shortest_first", "extra_berth"]


def test_twin_berth_shortage_tests_berth_actions():
    assert ids(NO_ML, maritime(), cargo(), events(), baseline(5.0)) == \
        ["baseline", "shortest_first", "extra_berth"]


def test_cargo_pressure_tests_gate_at_recorded_max():
    candidates = generate_candidates(NO_ML, maritime(), cargo("CRITICAL"), events(), baseline())
    assert [c["candidate_id"] for c in candidates] == ["baseline", "gate_max"]
    assert candidates[1]["actions"] == [{"type": "gate_capacity", "value": "max_recorded"}]


def test_high_model_alert_tests_gate_preventively():
    candidates = generate_candidates(HIGH_ML, maritime(), cargo(), events(), baseline())
    assert [c["candidate_id"] for c in candidates] == ["baseline", "gate_max"]
    assert "ml_forecast.alert_active" in candidates[1]["evidence_references"]


def test_high_label_without_alert_does_not_trigger():
    assert ids({**HIGH_ML, "alert_active": False}, maritime(), cargo(), events(), baseline()) == \
        ["baseline"]


def test_active_event_tests_every_action():
    assert ids(NO_ML, maritime(), cargo(), events(True), baseline()) == \
        ["baseline", "shortest_first", "extra_berth", "gate_max"]


def test_gate_skipped_when_no_yard_data():
    assert ids(NO_ML, maritime(), cargo("CRITICAL"), events(), baseline(),
               gate_available=False) == ["baseline"]


def test_summary_without_tests_says_no_intervention():
    summary = _deterministic_summary([{"candidate_id": "baseline", "title": "x", "kpis": {}}])
    assert "لا حاجة لتدخل" in summary


def run(cid, wait, yard):
    return {"candidate_id": cid, "title": cid,
            "kpis": {"total_wait_vessel_hours": wait, "yard_peak_occupancy_pct": yard}}


def test_summary_reports_vessel_and_yard_best_separately():
    summary = _deterministic_summary([run("extra_berth", 50, 90), run("baseline", 80, 90),
                                      run("gate_max", 80, 70)])
    assert "فتح رصيف إضافي مؤقت" in summary and "تشغيل البوابة" in summary


def test_summary_when_nothing_improves():
    summary = _deterministic_summary([run("baseline", 80, 90), run("gate_max", 80, 90)])
    assert "لم يحسّن" in summary
