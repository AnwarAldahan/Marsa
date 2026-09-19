"""
=====================================================================
 HERE: AGENT 3 - CONTEXT (WEATHER + EVENTS)
=====================================================================
Owner : <name>
Status: TODO - replace this stub with the real implementation.

Contract (do not change the function signature or the JSON keys,
the pipeline and the digital twin depend on them):

assess(snap: dict, cfg: dict) -> dict
  input : one hourly row (wind_speed_10m, wave_height, weather_code, event_active, event_type, event_severity, ...)
  output: {"agent": "context", "weather_level": low|moderate|elevated|severe|unknown,
           "event": {"type", "severity", "impact", "provenance": "SYNTHETIC"} | None,
           "twin_multipliers": {"crane_multiplier": float, "gate_multiplier": float, "arrival_multiplier": float},
           "findings": [...], "limitations": [...], "provenance": ["OBSERVED", "SYNTHETIC"]}
  note  : twin_multipliers is REQUIRED (1.0 = no effect). cfg["event_impact"] and cfg["weather"] hold the tables.
  API   : if the agent runs as FastAPI, wrap it with marsa.agents.remote.RemoteAgent instead of editing this file.
=====================================================================
"""
from __future__ import annotations


def assess(snap: dict, cfg: dict) -> dict:
    raise NotImplementedError("Agent 3 (context) goes here")
