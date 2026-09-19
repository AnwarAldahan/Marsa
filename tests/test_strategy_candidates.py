from marsa.agents.strategy_agent import generate_candidates

FORECAST = {"status": "ml_unavailable"}
HIGH_REAL_FORECAST = {
    "status": "success",
    "target": "operational_cargo_congestion_proxy",
    "classification": "HIGH",
    "congestion_score": 0.95,
    "alert_active": True,
    "is_real_model_prediction": True,
}
EVENT_IMPACT = {
    "demand_surge": {
        "minor": {"arrivals": 1.1},
    },
}


def maritime(status="NORMAL", waiting=0):
    return {
        "maritime_status": status,
        "evidence": {
            "vessels": 20,
            "waiting_vessels": waiting,
            "waiting_ratio": waiting / 20,
        },
    }


def cargo(status="NORMAL"):
    return {
        "cargo_status": status,
        "evidence": {
            "yard_occupancy_percent": 70.0,
            "truck_waiting_time_minutes": 20.0,
            "gate_throughput_last_1h": 100,
            "cargo_flow_ratio": 1.0,
        },
    }


def weather(*, active=False, event_type=None, severity=None, provenance="SYNTHETIC"):
    return {
        "contextual_pressure": {"level": "moderate" if active else "low"},
        "external_event_assessment": {
            "active": active,
            "type": event_type,
            "severity": severity,
            "provenance": provenance,
        },
    }


def candidate_ids(maritime_result, cargo_result, weather_result, forecast=FORECAST):
    candidates = generate_candidates(
        forecast,
        maritime_result,
        cargo_result,
        weather_result,
        EVENT_IMPACT,
    )
    return candidates, [candidate["candidate_id"] for candidate in candidates]


def test_normal_maritime_with_waiting_does_not_generate_shortest_first():
    _, ids = candidate_ids(maritime("NORMAL", 3), cargo(), weather())
    assert ids == ["baseline"]


def test_elevated_maritime_with_waiting_generates_shortest_first():
    _, ids = candidate_ids(maritime("ELEVATED", 3), cargo(), weather())
    assert ids == ["baseline", "shortest_first"]


def test_elevated_cargo_generates_gate_extension():
    _, ids = candidate_ids(maritime(), cargo("ELEVATED"), weather())
    assert ids == ["baseline", "gate_extension"]


def test_high_real_ml_and_moderate_cargo_generate_preventive_gate_test():
    candidates, ids = candidate_ids(
        maritime(), cargo("MODERATE"), weather(), HIGH_REAL_FORECAST,
    )
    assert ids == ["baseline", "gate_extension"]
    gate = candidates[-1]
    assert gate["actions"] == [{"type": "gate_boost", "value": 1.3}]
    assert "preventively testing" in gate["rationale"]
    assert "without implying it is necessary" in gate["rationale"]
    assert "cargo.cargo_status" in gate["evidence_references"]
    assert "ml_forecast.congestion_score" in gate["evidence_references"]
    assert "ml_forecast.alert_active" in gate["evidence_references"]


def test_high_real_ml_and_normal_cargo_remain_baseline_only():
    _, ids = candidate_ids(maritime(), cargo("NORMAL"), weather(), HIGH_REAL_FORECAST)
    assert ids == ["baseline"]


def test_low_ml_and_moderate_cargo_preserve_baseline_only():
    low = {**HIGH_REAL_FORECAST, "classification": "LOW", "alert_active": False}
    _, ids = candidate_ids(maritime(), cargo("MODERATE"), weather(), low)
    assert ids == ["baseline"]


def test_high_label_without_active_alert_does_not_trigger_preventive_candidate():
    inactive = {**HIGH_REAL_FORECAST, "alert_active": False}
    _, ids = candidate_ids(maritime(), cargo("MODERATE"), weather(), inactive)
    assert ids == ["baseline"]


def test_ml_unavailable_and_moderate_cargo_preserve_baseline_only():
    _, ids = candidate_ids(maritime(), cargo("MODERATE"), weather())
    assert ids == ["baseline"]


def test_domain_pressure_without_active_arrival_event_has_no_combined_candidate():
    _, ids = candidate_ids(maritime("ELEVATED", 3), cargo("ELEVATED"), weather())
    assert ids == ["baseline", "shortest_first", "gate_extension"]


def test_domain_pressure_and_synthetic_arrival_event_generate_combined_candidate():
    candidates, ids = candidate_ids(
        maritime("ELEVATED", 3),
        cargo("ELEVATED"),
        weather(active=True, event_type="demand_surge", severity="minor"),
    )
    assert ids == ["baseline", "shortest_first", "gate_extension", "combined_queue_gate"]
    combined = candidates[-1]
    assert combined["actions"] == [
        {"type": "queue_policy", "value": "shortest_service_first"},
        {"type": "gate_boost", "value": 1.3},
    ]
    assert "config.event_impact.demand_surge.minor.arrivals" in combined["evidence_references"]


def test_synthetic_demand_surge_alone_does_not_generate_intervention():
    _, ids = candidate_ids(
        maritime(),
        cargo(),
        weather(active=True, event_type="demand_surge", severity="minor"),
    )
    assert ids == ["baseline"]


def test_candidate_list_length_does_not_trigger_combined_candidate():
    _, ids = candidate_ids(maritime("NORMAL", 3), cargo("ELEVATED"), weather())
    assert ids == ["baseline", "gate_extension"]


def test_candidate_count_never_exceeds_four():
    candidates, ids = candidate_ids(
        maritime("CONGESTED", 3),
        cargo("CRITICAL"),
        weather(active=True, event_type="demand_surge", severity="minor"),
    )
    assert ids == ["baseline", "shortest_first", "gate_extension", "extra_berth"]
    assert len(candidates) == 4


def test_non_synthetic_event_cannot_trigger_combined_candidate():
    _, ids = candidate_ids(
        maritime("ELEVATED", 3),
        cargo("ELEVATED"),
        weather(
            active=True,
            event_type="demand_surge",
            severity="minor",
            provenance="OBSERVED",
        ),
    )
    assert ids == ["baseline", "shortest_first", "gate_extension"]
