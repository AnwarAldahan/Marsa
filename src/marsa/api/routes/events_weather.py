"""Thin HTTP transport for the existing Events & Weather Agent."""

from __future__ import annotations

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from marsa.agents.events_weather_agent import EventsWeatherAgent, EventsWeatherResult
from marsa.api.dependencies import get_events_weather_agent
from marsa.api.schemas.agents import EventsWeatherApiRequest
from marsa.common.exceptions import DataNotReadyError

router = APIRouter(prefix="/api/agents/events-weather", tags=["events-weather-agent"])


@router.post("/investigate", response_model=EventsWeatherResult)
def investigate_events_weather(
    request: EventsWeatherApiRequest,
    agent: EventsWeatherAgent = Depends(get_events_weather_agent),
) -> EventsWeatherResult:
    try:
        return agent.investigate(request.to_agent_request())
    except DataNotReadyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValidationError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(
            status_code=500,
            detail="Events & Weather investigation failed.",
        ) from error
