"""Evidence integrity and additive advisory integration; all tests stay offline."""
from copy import deepcopy
from datetime import UTC, datetime, timedelta

import pytest
from fastapi.testclient import TestClient

from marsa.agents.strategy_advisory import build_advisory, run_advisory_experiment
from marsa.api.main import app
from marsa.config import load_config
from marsa.pipeline import run
from marsa.twin.sensitivity import SensitivityRequest

START = datetime(2025, 7, 10, 19, tzinfo=UTC)
CANDIDATES = [
    dict(candidate_id="baseline", actions=[]),
    dict(candidate_id="shortest_first", actions=[dict(type="queue_policy", value="shortest_service_first")]),
    dict(candidate_id="extra_berth", actions=[dict(type="add_berths", value=1)]),
    dict(candidate_id="gate_max", actions=[dict(type="gate_capacity", value="max_recorded")]),
]
RECOMMENDATION = dict(highest_ranked_candidate="baseline", summary="Original recommendation", ranking=[])


def config(enabled=True):
    cfg = deepcopy(load_config())
    cfg["strategy_decision_support"] = dict(enabled=enabled)
    cfg["strategy_investigation"]["use_gemini"] = False
    return cfg


def experiment():
    return SensitivityRequest(scenario={
        "timestamp_utc": START, "horizon_hours": 4, "time_information_mode": "as_of",
        "cohort_id": "explicit-snapshot-test", "resource_pool_id": "abstract-test",
        "parameters": dict(yard_capacity=10000, initial_inventory=10000, berth_slots=1,
                           unloading_units_per_hour=1000, gate_units_per_hour=100),
        "as_of_observations": [dict(call_id="B", vessel_id="IMO1", mmsi=101,
                                    available_at=START, arrival=START - timedelta(hours=2),
                                    state="berthed", arrival_basis="berth_start_fallback",
                                    berth_start=START - timedelta(hours=2)),
                               dict(call_id="W", vessel_id="IMO2", mmsi=102,
                                    available_at=START, arrival=START - timedelta(hours=1),
                                    state="waiting", arrival_basis="matched_wait_start")],
    })


@pytest.fixture(scope="module")
def evidence():
    return run_advisory_experiment(experiment(), config())


def advisory(evidence, candidates=None, recommendation=None):
    return build_advisory(START, candidates or CANDIDATES, recommendation or RECOMMENDATION, config(), evidence)


def test_disabled_ignores_even_invalid_evidence_and_missing_config():
    assert build_advisory(START, CANDIDATES, RECOMMENDATION, config(False), evidence={"invalid": True}) is None
    cfg = config()
    cfg.pop("strategy_decision_support")
    assert build_advisory(START, CANDIDATES, RECOMMENDATION, cfg) is None


def test_valid_evidence_exact_numerical_copy_and_no_mutation(evidence):
    before = deepcopy(evidence)
    candidates, recommendation = deepcopy(CANDIDATES), deepcopy(RECOMMENDATION)
    result = advisory(evidence, candidates, recommendation)
    assert result["status"] == "AVAILABLE"
    assert result["original_winner"] == "baseline"
    assert result["operator_review_required"] and not result["autonomous_execution"]
    rows = evidence["report"]["contexts"][0]["comparisons"]
    for emitted, source in zip(result["interventions"], rows[1:], strict=True):
        for key, value in emitted["kpis"].items():
            assert value == source["kpis"][key]
        assert emitted["baseline_kpis"] == result["baseline"]["kpis"]
        assert emitted["vs_baseline"] == source["vs_baseline"]
        assert emitted["robustness"] == evidence["report"]["robustness"][source["intervention"]]
    assert evidence == before
    assert candidates == CANDIDATES and recommendation == RECOMMENDATION


def test_catalog_mapping_preserves_parameters_and_separates_gate_experiments(evidence):
    result = advisory(evidence)
    mappings = {r["candidate_id"]: r for r in result["candidate_experiment_mapping"]}
    assert mappings["baseline"]["status"] == "REFERENCE_ONLY"
    assert mappings["extra_berth"]["experiment_id"] == "berth_plus_one"
    assert mappings["extra_berth"]["status"] == "CONDITIONAL_SEMANTIC_MATCH"
    assert mappings["shortest_first"]["status"] == mappings["gate_max"]["status"] == "NOT_EVALUATED"
    for candidate in CANDIDATES:
        assert mappings[candidate["candidate_id"]]["original_actions"] == candidate["actions"]
    for key in ["gate_extension", "unknown"]:
        assert advisory(evidence, [dict(candidate_id=key, actions=[])])["candidate_experiment_mapping"][0]["status"] == "NOT_EVALUATED"
    changed = [dict(candidate_id="extra_berth", actions=[dict(type="add_berths", value=2)])]
    assert advisory(evidence, changed)["candidate_experiment_mapping"][0]["status"] == "NOT_EVALUATED"
    changed[0]["actions"][0]["value"] = True
    assert advisory(evidence, changed)["candidate_experiment_mapping"][0]["status"] == "NOT_EVALUATED"


def test_missing_evidence_returns_unavailable_without_claims():
    result = advisory(None)
    assert result["status"] == "UNAVAILABLE" and not result["interventions"]
    assert all(m["status"] == "NOT_EVALUATED" for m in result["candidate_experiment_mapping"])


@pytest.mark.parametrize("corruption", [
    "missing_metric", "timestamp", "units", "pairing", "population", "parameter", "robustness",
    "comparison", "diagnostic", "temporal", "nan", "missing_context", "base_parameters",
])
def test_incomplete_inconsistent_evidence_is_not_presented_as_valid(evidence, corruption):
    bad = deepcopy(evidence)
    report = bad["report"]
    context = report["contexts"][0]
    row = context["comparisons"][1]
    if corruption == "missing_metric":
        row["kpis"].pop("cargo_unloaded")
    elif corruption == "timestamp":
        bad["decision_timestamp"] = (START + timedelta(hours=1)).isoformat()
    elif corruption == "units":
        report["units"] = "TEU"
    elif corruption == "pairing":
        row["workload_signature"] = "unpaired"
    elif corruption == "population":
        report["population"][0]["call_id"] = "different"
    elif corruption == "parameter":
        row["parameters"]["gate_units_per_hour"] += 1
    elif corruption == "robustness":
        report["robustness"]["gate_modest"]["classification"] = "robust_across_tested_assumptions"
        report["robustness"]["gate_modest"]["tested_contexts"] = -1
    elif corruption == "comparison":
        row["vs_baseline"]["vessel_metric_deltas"]["completed_vessels"] = 999
    elif corruption == "diagnostic":
        row["bottleneck"]["classification"] = "insufficient_demand"
    elif corruption == "temporal":
        report["metadata"]["hindsight_assisted"] = True
    elif corruption == "nan":
        row["kpis"]["cargo_unloaded"] = float("nan")
    elif corruption == "base_parameters":
        context["comparisons"][0]["parameters"]["berth_slots"] += 1
    else:
        report["contexts"] = []
    result = advisory(bad)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert not result["interventions"] and result["issues"]


def test_conservation_failure_flagged_instead_of_reporting_benefits(evidence):
    bad = deepcopy(evidence)
    bad["report"]["contexts"][0]["comparisons"][1]["kpis"]["system_conservation_residual"] = 100
    result = advisory(bad)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["issues"][0]["code"] == "CONSERVATION_FAILURE"
    assert not result["interventions"]


def test_temporal_labels_units_and_explicit_synthetic_limitations(evidence):
    result = advisory(evidence)
    assert result["time_information_mode"] == "as_of" and not result["hindsight_assisted"]
    assert result["units"] == "normalized_units"
    assert result["scenario_provenance"]["source"]["call_source"] == "CALLER_SUPPLIED_AS_OF_OBSERVATIONS"
    assert "synthetic" in " ".join(result["limitations"]).lower()
    assert "not TEU" in " ".join(result["limitations"])


def test_enabling_feature_does_not_compute_sensitivity_or_change_pipeline(monkeypatch):
    def forbidden(*args, **kwargs):
        pytest.fail("ordinary Strategy requests must not trigger expensive experiments")
    monkeypatch.setattr("marsa.agents.strategy_advisory.analyze_sensitivity", forbidden)
    cfg = config(False)
    legacy = run(START.isoformat(), cfg=cfg, save=False)
    cfg["strategy_decision_support"]["enabled"] = True
    enabled = run(START.isoformat(), cfg=cfg, save=False)
    assert enabled.pop("strategy_advisory")["status"] == "UNAVAILABLE"
    assert enabled == legacy
    assert enabled["ml_forecast"] == legacy["ml_forecast"]


def test_pipeline_uses_existing_evidence_without_more_gemini_calls(evidence):
    class UnavailableGemini:
        def __init__(self):
            self.calls = 0

        def generate_structured(self, **kwargs):
            self.calls += 1
            raise RuntimeError("controlled unavailable Gemini")
    cfg = config(False)
    cfg["strategy_investigation"]["use_gemini"] = True
    old_provider, new_provider = UnavailableGemini(), UnavailableGemini()
    legacy = run(START.isoformat(), cfg=cfg, llm=old_provider, save=False)
    cfg["strategy_decision_support"]["enabled"] = True
    enabled = run(START.isoformat(), cfg=cfg, llm=new_provider, save=False, advisory_evidence=evidence)
    result = enabled.pop("strategy_advisory")
    assert result["status"] == "AVAILABLE" and result["reasoning_mode"] == "deterministic"
    assert enabled == legacy
    assert new_provider.calls == old_provider.calls


def test_api_is_opt_in_additive_and_never_calls_gemini(monkeypatch):
    import marsa.api.routes.decision_support as route

    cfg = config(False)
    monkeypatch.setattr(route, "load_config", lambda: cfg)
    def forbidden():
        pytest.fail("complementary advisory endpoint must not request Gemini")
    monkeypatch.setattr(route, "get_strategy_provider", forbidden)
    body = dict(timestamp_utc=START.isoformat(), experiment=experiment().model_dump(mode="json"))
    with TestClient(app) as client:
        assert client.post("/api/decision-support/advisory", json=body).status_code == 403
        cfg["strategy_decision_support"]["enabled"] = True
        response = client.post("/api/decision-support/advisory", json=body)
        assert response.status_code == 200
        result = response.json()
        assert result["strategy_advisory"]["status"] == "AVAILABLE"
        assert result["decision_support"]["highest_ranked_candidate"] == result["strategy_advisory"]["original_winner"]
        body.pop("experiment")
        assert client.post("/api/decision-support/advisory", json=body).json()["strategy_advisory"]["status"] == "UNAVAILABLE"
        body["experiment"] = experiment().model_dump(mode="json")
        body["timestamp_utc"] = (START + timedelta(hours=1)).isoformat()
        assert client.post("/api/decision-support/advisory", json=body).status_code == 422
        cfg.pop("strategy_decision_support")
        assert client.post("/api/decision-support/advisory", json=body).status_code == 403


def test_existing_analyze_api_contract_unchanged_when_disabled(monkeypatch):
    import marsa.api.routes.decision_support as route
    import marsa.pipeline as pipeline

    cfg = config(False)
    monkeypatch.setattr(route, "get_strategy_provider", lambda: None)
    monkeypatch.setattr(pipeline, "load_config", lambda: cfg)
    expected = run(START.isoformat(), cfg=cfg, save=False)
    with TestClient(app) as client:
        result = client.post("/api/decision-support/analyze", json=dict(timestamp_utc=START.isoformat())).json()
    assert result == expected
    assert "strategy_advisory" not in result


def test_two_historical_examples_preserve_winners_and_expose_tradeoffs():
    from scripts.demo_strategy_advisory import historical_advisory_demo

    cases = historical_advisory_demo()
    assert len(cases) == 2
    for case in cases.values():
        result = case["advisory"]
        assert case["legacy_response_unchanged"]
        assert case["legacy_winner"] == case["winner_after_advisory"] == "baseline"
        assert result["status"] == "AVAILABLE"
        assert result["time_information_mode"] == "retrospective" and result["hindsight_assisted"]
        rows = {r["intervention"]: r for r in result["interventions"]}
        assert rows["gate_modest"]["advisory_assessment"] == "CARGO_PROGRESS_ONLY"
        assert rows["berth_plus_one"]["advisory_assessment"] == "QUEUE_RELIEF_ONLY"
    a, b = cases["A"]["advisory"], cases["B"]["advisory"]
    assert a["tradeoffs"][0]["comparison"]["status"] == "tradeoff"
    assert a["tradeoffs"][0]["intervention_kpis"]["completed_vessels"] < a["tradeoffs"][0]["baseline_kpis"]["completed_vessels"]
    rows = {r["intervention"]: r for r in b["interventions"]}
    assert rows["gate_modest"]["robustness"]["classification"] == "ineffective_under_tested_conditions"
    assert rows["gate_large"]["robustness"]["classification"] == "sensitive_to_specific_parameters"
    assert rows["berth_plus_one"]["robustness"]["classification"] == "robust_across_tested_assumptions"


@pytest.mark.parametrize("path", [
    ("report",), ("request",), ("decision_timestamp",),
    ("report", "skipped_duplicate_contexts"), ("report", "excluded_contexts"),
    ("report", "plan"), ("report", "ranges_low_base_high"),
    ("report", "contexts", 0, "axis"), ("report", "contexts", 0, "knobs"),
    ("report", "metadata", "source"),
    ("report", "metadata", "initial_inventory", "value"),
    ("report", "contexts", 0, "comparisons", 0, "bottleneck", "hourly_evidence"),
])
def test_missing_nested_envelope_fields_fail_closed(evidence, path):
    bad = deepcopy(evidence)
    target = bad
    for key in path[:-1]:
        target = target[key]
    del target[path[-1]]
    result = advisory(bad)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert result["issues"] and not result["interventions"]


@pytest.mark.parametrize("field,value", [
    ("skipped_duplicate_contexts", None), ("skipped_duplicate_contexts", [{}]),
    ("excluded_contexts", [{}]), ("contexts", [None]),
    ("contexts", [{"comparisons": []}]), ("metadata", []),
    ("ranges_low_base_high", []), ("run_count", "20"),
    ("robustness", None), ("population", [None]),
])
def test_malformed_envelope_types_fail_closed(evidence, field, value):
    bad = deepcopy(evidence)
    bad["report"][field] = value
    result = advisory(bad)
    assert result["status"] == "INSUFFICIENT_EVIDENCE"
    assert not result["interventions"]


def test_december_arrivals_remain_valid_for_january_decision():
    january = datetime(2025, 1, 2, tzinfo=UTC)
    req = SensitivityRequest(scenario={
        "timestamp_utc": january, "horizon_hours": 24, "time_information_mode": "retrospective",
        "cohort_id": "year-boundary-regression", "resource_pool_id": "abstract-test",
        "parameters": dict(yard_capacity=10000, initial_inventory=0, berth_slots=23,
                           unloading_units_per_hour=1000, gate_units_per_hour=100),
    })
    envelope = run_advisory_experiment(req, config())
    assert any(v["arrival_timestamp"].startswith("2024-") for v in envelope["report"]["population"])
    result = build_advisory(january, CANDIDATES, RECOMMENDATION, config(), envelope)
    assert result["status"] == "AVAILABLE" and result["hindsight_assisted"]
    assert result["baseline"]["conservation"]["status"] == "PASS"
    # Decision years and dataset coverage remain restricted independently.
    with pytest.raises(ValueError, match="2025"):
        build_advisory(january.replace(year=2024), CANDIDATES, RECOMMENDATION, config(), envelope)
    outside = req.scenario.model_copy(update={"timestamp_utc": datetime(2022, 1, 1, tzinfo=UTC)})
    with pytest.raises(ValueError, match="coverage"):
        run_advisory_experiment(SensitivityRequest(scenario=outside), config())


@pytest.mark.parametrize("arrival", [
    "2025-07-10T19:00:00", "2025-07-10T19:00:00+03:00",
    "2025-07-10T19:30:00Z", "2025-07-10T19:00:00.000000001Z",
    "2025-07-11T19:00:00Z",
])
def test_event_validation_retains_utc_hour_and_horizon_rules(evidence, arrival):
    from marsa.twin.sensitivity import _signature

    bad = deepcopy(evidence)
    bad["report"]["population"][0]["arrival_timestamp"] = arrival
    bad["report"]["population_signature"] = _signature(bad["report"]["population"])
    assert advisory(bad)["status"] == "INSUFFICIENT_EVIDENCE"
