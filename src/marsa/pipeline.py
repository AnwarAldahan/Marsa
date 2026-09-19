"""End-to-end run for one timestamp:
data snapshot -> ML forecast -> 3 agents -> strategy candidates -> twin -> ranking -> recommendation.
"""
from __future__ import annotations
import json
import pandas as pd
from marsa.config import load_config, OUTPUTS_DIR
from marsa.data.load import load_dataset, snapshot
from marsa.model.predict import forecast as ml_forecast
from marsa.agents import maritime_agent, cargo_agent, context_agent, strategy_agent
from marsa.twin.state import scenario_from_snapshot
from dataclasses import asdict


def run(timestamp: str, cfg=None, df=None, feature_set="ais_only", llm=None, save=True, agents=None) -> dict:
    """agents: optional overrides {"maritime"|"cargo"|"context": callable(snap, cfg) -> dict}, e.g. RemoteAgent(...).assess"""
    cfg = cfg or load_config()
    df = df if df is not None else load_dataset()
    snap = snapshot(df, timestamp)
    fc = ml_forecast(df, timestamp, cfg, feature_set=feature_set)
    ag = {"maritime": maritime_agent.assess, "cargo": cargo_agent.assess, "context": context_agent.assess}
    ag.update(agents or {})
    m, c, x = ag["maritime"](snap, cfg), ag["cargo"](snap, cfg), ag["context"](snap, cfg)
    sc = scenario_from_snapshot(snap, cfg, context=x["twin_multipliers"])
    cands = strategy_agent.generate_candidates(fc, m, c, x)
    ranked = strategy_agent.evaluate(sc, cands, cfg)
    rec = strategy_agent.recommend(ranked, fc, {"maritime": m, "cargo": c, "context": x}, llm=llm)
    out = {"timestamp_utc": snap["hour_key"], "port_state": snap, "forecast": fc,
           "agents": {"maritime": m, "cargo": c, "context": x},
           "twin_scenario": asdict(sc), "twin_results": ranked, "recommendation": rec}
    if save:
        OUTPUTS_DIR.mkdir(exist_ok=True)
        p = OUTPUTS_DIR / f"run_{pd.Timestamp(snap['hour_key']).strftime('%Y%m%dT%H')}.json"
        p.write_text(json.dumps(out, indent=2, default=str), encoding="utf-8")
        out["saved_to"] = str(p)
    return out
