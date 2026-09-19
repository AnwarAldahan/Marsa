"""Final four-agent Marsa decision-support pipeline."""
from __future__ import annotations

import json
from dataclasses import asdict

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
from marsa.twin.adapter import build_scenario


def _forecast_risk(forecast: dict) -> str | None:
    return None if forecast["status"] == "ml_unavailable" else forecast.get("classification")


def run(timestamp: str, cfg=None, df=None, llm=None, save=False,
        agents=None, forecast_override: dict | None = None) -> dict:
    """Run evidence collection, strategy generation, simulation, and evaluation."""
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
    scenario, assumptions = build_scenario(snap, cfg, events_weather)
    candidates = strategy_agent.generate_candidates(
        forecast, maritime, cargo, events_weather, cfg["event_impact"],
    )
    synthesis = strategy_agent.synthesize(forecast, maritime, cargo, events_weather)
    ranked = strategy_agent.evaluate(scenario, candidates, cfg)
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
            "simulation_horizon_hours": scenario.horizon_hours,
            "scenario": asdict(scenario),
            "assumptions": assumptions,
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
            "Digital Twin outcomes are simulated estimates under documented assumptions.",
            "No operational action is executed automatically.",
        ],
        "human_approval_required": True,
    }
    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        path = OUTPUTS_DIR / f"run_{pd.Timestamp(ts).strftime('%Y%m%dT%H')}.json"
        path.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        out["saved_to"] = str(path)
    return out
