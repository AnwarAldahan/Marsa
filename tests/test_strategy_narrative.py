"""Recommendation language may select supported facts, never change the decision."""
from copy import deepcopy

import pytest

from marsa.agents.strategy_agent import recommend
from marsa.twin.scoring import score_results


def inputs(improved=False):
    rows = [dict(candidate_id=key, title=key, kpis={
        "total_wait_vessel_hours": wait, "yard_peak_occupancy_pct": yard,
    }) for key, wait, yard in [
        ("baseline", 24, 98), ("extra_berth", 12 if improved else 24, 98),
        ("gate_max", 24, 90),
    ]]
    domains = {"maritime": {"maritime_status": "ELEVATED"},
               "cargo": {"cargo_status": "ELEVATED"},
               "events_weather": {"contextual_pressure": {"level": "low"}}}
    return score_results(rows), {"status": "ml_unavailable"}, domains


class Provider:
    def __init__(self, summary=None, choose=None):
        self.summary, self.choose, self.calls = summary, choose, 0

    def generate_structured(self, **kwargs):
        self.calls += 1
        self.payload = kwargs["payload"]
        return {"summary": self.choose(self.payload) if self.choose else self.summary}


@pytest.mark.parametrize("text", [
    "The recommended option is extra_berth.",
    "الخيار الموصى به هو فتح رصيف إضافي مؤقت.",
    "baseline is highest ranked, but choose extra_berth instead.",
    "extra_berth reduced vessel waiting in the simulation.",
    "تشغيل البوابة خفّض انتظار السفن في المحاكاة.",
    "baseline reduced vessel waiting compared to baseline.",
    "The best choice will improve operations.",
    "قد يكون الخيار الآخر أفضل.",
    "", "   ", "No operational change is definitely the safest action.",
])
def test_contradictory_unsupported_or_ambiguous_text_falls_back(text):
    args = inputs()
    before = deepcopy(args)
    expected = recommend(*args)
    provider = Provider(text)
    actual = recommend(*args, llm=provider)
    assert actual == {**expected, "reasoning_mode": "deterministic_fallback"}
    assert args == before and provider.calls == 1


@pytest.mark.parametrize("improved", [False, True])
def test_valid_grounded_choices_preserved_with_authoritative_ranking(improved):
    args = inputs(improved)
    original = deepcopy(args)
    provider = Provider(choose=lambda payload: payload["allowed_narratives"][1]["summary"])
    result = recommend(*args, llm=provider)
    assert result["reasoning_mode"] == "llm_assisted"
    assert result["summary"] == provider.payload["allowed_narratives"][1]["summary"]
    assert result["highest_ranked_candidate"] == args[0][0]["candidate_id"]
    assert provider.payload["authoritative_winner"] == result["highest_ranked_candidate"]
    assert [r["candidate_id"] for r in provider.payload["authoritative_ranking"]] == [r["candidate_id"] for r in args[0]]
    assert args == original
    # Every allowed language choice preserves the same public response structure.
    for option in provider.payload["allowed_narratives"]:
        accepted = recommend(*args, llm=Provider(option["summary"]))
        assert accepted["reasoning_mode"] == "llm_assisted"
        assert accepted["ranking"] == result["ranking"]
        assert accepted.keys() == recommend(*args).keys()


def test_only_simulator_supported_improvement_claims_are_offered():
    provider = Provider(choose=lambda p: p["allowed_narratives"][1]["summary"])
    recommend(*inputs(False), llm=provider)
    claims = provider.payload["allowed_narratives"][2:]
    assert len(claims) == 1  # Yard progress exists; vessel waiting improvement does not.
    assert "gate_max.kpis.yard_peak_occupancy_pct" in claims[0]["evidence_references"]
    assert "هذا المقياس لا يدخل" in claims[0]["summary"]
    recommend(*inputs(True), llm=provider)
    assert any("extra_berth.kpis.total_wait_vessel_hours" in c["evidence_references"]
               for c in provider.payload["allowed_narratives"])


def test_modified_winner_improvement_claim_cannot_reuse_another_context():
    args = inputs(True)
    provider = Provider(choose=lambda p: p["allowed_narratives"][1]["summary"])
    recommend(*args, llm=provider)
    winner_text = provider.payload["allowed_narratives"][1]["summary"]
    result = recommend(*inputs(False), llm=Provider(winner_text))
    assert result["reasoning_mode"] == "deterministic_fallback"
    assert result["highest_ranked_candidate"] == "baseline"


@pytest.mark.parametrize("suffix", [" Therefore open another berth.", "\u200b", " خفّض الانتظار فعليًا."])
def test_appended_claim_or_invisible_text_is_rejected(suffix):
    provider = Provider(choose=lambda p: p["allowed_narratives"][1]["summary"] + suffix)
    assert recommend(*inputs(), llm=provider)["reasoning_mode"] == "deterministic_fallback"


def test_whitespace_variation_renders_canonical_valid_text():
    provider = Provider(choose=lambda p: "\n " + p["allowed_narratives"][1]["summary"].replace(" ", "  ") + "\n")
    result = recommend(*inputs(), llm=provider)
    assert result["reasoning_mode"] == "llm_assisted"
    assert result["summary"] == provider.payload["allowed_narratives"][1]["summary"]


def test_provider_failure_and_invalid_schema_preserve_safe_fallback():
    class Failing:
        def generate_structured(self, **kwargs):
            raise RuntimeError("provider unavailable")
    class Invalid:
        def generate_structured(self, **kwargs):
            return {"summary": "", "winner": "extra_berth"}
    expected = recommend(*inputs())
    for provider in [Failing(), Invalid()]:
        assert recommend(*inputs(), llm=provider) == {**expected, "reasoning_mode": "deterministic_fallback"}


@pytest.mark.parametrize("valid", [False, True])
def test_main_user_decision_comparison_keeps_agent_winner_and_narrative_guard(valid):
    ranked, forecast, domains = inputs()
    user = deepcopy(ranked[0])
    user.update(candidate_id="user_decision", title="User decision", actions=[])
    user["kpis"]["total_wait_vessel_hours"] = 0
    ranked = score_results([user, *ranked])
    provider = (Provider(choose=lambda p: p["allowed_narratives"][0]["summary"]) if valid
                else Provider("The recommended option is user_decision."))
    actual = recommend(ranked, forecast, domains, llm=provider)
    assert actual["highest_ranked_candidate"] == "baseline"
    assert actual["ranking"][0]["candidate_id"] == "user_decision"
    assert actual["user_decision_assessment"] is not None
    assert provider.payload["authoritative_winner"] == "baseline"
    assert actual["reasoning_mode"] == ("llm_assisted" if valid else "deterministic_fallback")
