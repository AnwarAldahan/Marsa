"""Dependency factories for the active four-agent architecture."""
from __future__ import annotations

from functools import lru_cache

from marsa.agents.cargo_agent import CargoInvestigationAgent
from marsa.agents.events_weather_agent import EventsWeatherAgent
from marsa.agents.gemini import GeminiStructuredProvider
from marsa.agents.maritime_agent import MaritimeInvestigationAgent


def _provider():
    try:
        return GeminiStructuredProvider.from_env()
    except ValueError:
        return None


@lru_cache
def get_maritime_agent() -> MaritimeInvestigationAgent:
    return MaritimeInvestigationAgent(llm_provider=_provider())


@lru_cache
def get_cargo_agent() -> CargoInvestigationAgent:
    return CargoInvestigationAgent(llm_provider=_provider())


@lru_cache
def get_events_weather_agent() -> EventsWeatherAgent:
    return EventsWeatherAgent(llm_provider=_provider())
