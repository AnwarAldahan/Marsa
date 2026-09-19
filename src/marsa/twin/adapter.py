"""Deterministic adapter from authoritative state and context to the teammate twin."""
from __future__ import annotations

from dataclasses import asdict

from marsa.twin.state import scenario_from_snapshot


def build_scenario(snap: dict, cfg: dict, events_weather: dict) -> tuple[object, list[dict]]:
    multipliers = {
        "crane_multiplier": 1.0,
        "gate_multiplier": 1.0,
        "arrival_multiplier": 1.0,
    }
    assumptions = [{
        "parameter": "base_scenario",
        "source": "config/port_config.yaml and authoritative current snapshot",
        "statement": "Capacity, service, flow, and LA-share values are modeling assumptions.",
    }]

    wind = snap.get("wind_speed_10m")
    if wind is not None:
        weather_cfg = cfg["weather"]
        if wind >= weather_cfg["crane_stop_wind_kmh"]:
            multipliers["crane_multiplier"] = 0.0
            assumptions.append({
                "parameter": "crane_multiplier", "value": 0.0,
                "source": "config/port_config.yaml crane_stop_wind_kmh",
                "statement": "Configured wind stop threshold applied as a simulation assumption.",
            })
        elif wind >= weather_cfg["crane_slowdown_wind_kmh"]:
            multipliers["crane_multiplier"] = 0.8
            assumptions.append({
                "parameter": "crane_multiplier", "value": 0.8,
                "source": "config/port_config.yaml crane_slowdown_wind_kmh",
                "statement": "Configured 20% wind slowdown applied as a simulation assumption.",
            })

    event = events_weather["external_event_assessment"]
    if event["active"]:
        event_type, severity = event["type"], event["severity"]
        mapping = cfg["event_impact"].get(event_type, {}).get(severity, {})
        targets = {"cranes": "crane_multiplier", "gate": "gate_multiplier", "arrivals": "arrival_multiplier"}
        for key, value in mapping.items():
            multipliers[targets[key]] *= float(value)
        assumptions.append({
            "parameter": "event_multipliers", "value": mapping,
            "source": f"config/port_config.yaml event_impact.{event_type}.{severity}",
            "statement": "Synthetic event context mapped deterministically for what-if simulation only.",
        })

    scenario = scenario_from_snapshot(snap, cfg, context=multipliers)
    assumptions.append({
        "parameter": "resolved_scenario", "value": asdict(scenario),
        "source": "src/marsa/twin/state.py",
        "statement": "Resolved Digital Twin starting state; not an observed measurement bundle.",
    })
    return scenario, assumptions
