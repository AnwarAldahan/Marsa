"""Final four-agent Marsa decision-support pipeline."""
from __future__ import annotations

import json

import pandas as pd

from marsa.agents import strategy_agent
from marsa.agents.cargo_agent import (
    CargoForecastContext,
    CargoInvestigationAgent,
    CargoInvestigationRequest,
)
from marsa.agents.events_weather_agent import (
    EventsWeatherAgent,
    EventsWeatherInvestigationRequest,
    ForecastContext,
)
from marsa.agents.maritime_agent import (
    MaritimeForecastContext,
    MaritimeInvestigationAgent,
    MaritimeInvestigationRequest,
)
from marsa.config import OUTPUTS_DIR, load_config
from marsa.data.load import load_dataset, parse_exact_utc_hour, snapshot
from marsa.model.predict import forecast as ml_forecast
from marsa.model.predict import validate_test_forecast
from marsa.twin.adapter import build_scenario, scenario_summary
from marsa.twin.engine import simulate_candidate


def _forecast_risk(forecast: dict) -> str | None:
    return None if forecast["status"] == "ml_unavailable" else forecast.get("classification")


# Run evidence collection, the Digital Twin, strategy generation and ranking for one hour.
# stress: optional user scenario {"berths_closed": int, "arrival_multiplier": float}.
# added: optional vessels added by the user [{"arrival_in_hours", "vessel_id"|"length_m"|"service_hours"}].
# user_actions: optional decision typed by the user; the Strategy agent evaluates it next to its own.
def run(timestamp: str, cfg=None, df=None, llm=None, save=False,
        agents=None, forecast_override: dict | None = None, stress: dict | None = None,
        added: list[dict] | None = None, user_actions: list[dict] | None = None) -> dict:
    cfg = cfg or load_config()
    frame = df if df is not None else load_dataset()
    ts = parse_exact_utc_hour(timestamp)
    snap = snapshot(frame, ts)
    forecast = (validate_test_forecast(forecast_override, ts) if forecast_override is not None
                else ml_forecast(frame, ts, cfg))
    risk = _forecast_risk(forecast)

    implementations = {
        "maritime": MaritimeInvestigationAgent(),
        "cargo": CargoInvestigationAgent(),
        "events_weather": EventsWeatherAgent(),
    }
    implementations.update(agents or {})
    maritime_context = MaritimeForecastContext(risk_level=risk, horizon_hours=6) if risk else None
    cargo_context = CargoForecastContext(risk_level=risk, horizon_hours=6) if risk else None
    events_context = ForecastContext(risk_level=risk, horizon_hours=6) if risk else None
    maritime = implementations["maritime"].investigate(MaritimeInvestigationRequest(
        timestamp_utc=ts.to_pydatetime(), forecast_context=maritime_context,
    )).model_dump(mode="json")
    cargo = implementations["cargo"].investigate(CargoInvestigationRequest(
        timestamp_utc=ts.to_pydatetime(), forecast_context=cargo_context,
    )).model_dump(mode="json")
    events_weather = implementations["events_weather"].investigate(EventsWeatherInvestigationRequest(
        timestamp_utc=ts.to_pydatetime(), forecast_context=events_context,
    )).model_dump(mode="json")

    domain_results = {"maritime": maritime, "cargo": cargo, "events_weather": events_weather}
    investigation = strategy_agent.run_investigation(ts, forecast, domain_results, cfg, llm=llm)
    scenario, sources = build_scenario(ts, cfg, stress, added)
    baseline = simulate_candidate(scenario, {"id": "baseline", "actions": []})
    candidates = strategy_agent.generate_candidates(
        forecast, maritime, cargo, events_weather, baseline,
        gate_available=scenario.yard is not None, investigation=investigation,
    )
    legacy_order = [candidate["candidate_id"] for candidate in candidates]
    candidates, candidate_selection = strategy_agent.select_candidates(
        candidates, investigation, ts, domain_results, cfg,
    )
    if user_actions is not None:
        # added after the agent's own selection, so it never changes what the agent chose to test
        candidates = candidates + [strategy_agent.user_candidate(user_actions)]
        legacy_order = legacy_order + [strategy_agent.USER_DECISION_ID]
    synthesis = strategy_agent.synthesize(forecast, maritime, cargo, events_weather)
    ranked = strategy_agent.evaluate(
        scenario, candidates, cfg, baseline=baseline,
        comparison_order=None if candidate_selection is None else legacy_order,
    )
    decision_support = strategy_agent.recommend(ranked, forecast, domain_results, llm=llm)
    out = {
        "architecture": {
            "agents": ["maritime", "cargo", "events_weather", "strategy"],
            "orchestrator_agent": False,
            "ml_is_agent": False,
            "digital_twin_is_agent": False,
        },
        "authoritative_timestamp": ts.isoformat(),
        "port_state": snap,
        "ml_forecast": forecast,
        "domain_agents": domain_results,
        "strategy_candidates": candidates,
        "strategy_synthesis": synthesis,
        "digital_twin": {
            "port": "Port of Los Angeles - cargo vessels (AIS types 70-79) and yard",
            "marker_hours": [int(cfg["ml_contract"]["horizon_hours"])],
            "simulation_horizon_hours": scenario.horizon_hours,
            "scenario": scenario_summary(scenario),
            "sources": sources,
            "results": ranked,
            "provenance": "SIMULATED",
        },
        "deterministic_evaluation": ranked,
        "decision_support": decision_support,
        "limitations": [
            "The ML risk horizon is six hours; the Digital Twin simulation horizon is configured independently.",
            "The ML score is uncalibrated and targets a retrospective operational cargo-congestion proxy.",
            "The finalized Cargo inputs and weather-pressure baseline have retrospective construction limitations.",
            "Cargo state is synthetic/calibrated synthetic, not observed terminal state.",
            ("Connected cargo uses a normalized service-effort yard proxy, not TEU or measured unloading. "
             "Only t0 Cargo ratios initialize stock/clearance; recorded service durations and later AIS arrivals are hindsight. "
             "CSV cargo arrivals are not replayed in addition to vessel unloading."
             if scenario.cargo is not None else
             "The Digital Twin simulates berths and waiting vessels from AIS and the yard from the synthetic Cargo agent file; the yard does not change vessel waiting (no data on that effect)."),
            ("AIS service durations supply normalized reference effort; completion and berth release depend on simulated unloading. "
             "Vessels added by the user retain earlier-history service estimates or explicit service input."
             if scenario.cargo is not None else
             "Stays of vessels in AIS are the recorded ones (replay); vessels added by the user get a stay estimated from earlier stays."),
            ("Connected mode retains historical arrivals and berth-readiness floors; it is a retrospective counterfactual, not exact AIS replay."
             if scenario.cargo is not None else
             "Without a stress scenario the Digital Twin replays the real AIS schedule; stress values are chosen by the user."),
            "No operational action is executed automatically.",
        ],
        "human_approval_required": True,
    }
    if investigation is not None:
        out["strategy_investigation"] = investigation
    if candidate_selection is not None:
        out["candidate_selection"] = candidate_selection
    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        path = OUTPUTS_DIR / f"run_{pd.Timestamp(ts).strftime('%Y%m%dT%H')}.json"
        path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        out["saved_to"] = str(path)
    return out
