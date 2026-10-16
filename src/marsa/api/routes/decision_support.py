"""Integrated decision-support endpoint."""
from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, Field, field_validator

from marsa.api.dependencies import get_strategy_provider
from marsa.common.exceptions import DataNotReadyError
from marsa.pipeline import run

router = APIRouter(prefix="/api/decision-support", tags=["decision-support"])


# Optional stress scenario chosen by the user; empty means replay the real AIS schedule.
class StressScenario(BaseModel):
    model_config = ConfigDict(extra="forbid")
    berths_closed: int = Field(default=0, ge=0)
    arrival_surge_pct: float = Field(default=0.0, ge=0.0)


class DecisionSupportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp_utc: datetime
    stress: StressScenario | None = None

    @field_validator("timestamp_utc")
    @classmethod
    def require_explicit_utc(cls, value: datetime) -> datetime:
        if value.tzinfo is None:
            raise ValueError("timestamp_utc must be timezone-aware and explicitly UTC")
        if value.utcoffset() != timedelta(0):
            raise ValueError("timestamp_utc must use UTC (offset +00:00 or Z)")
        return value


@router.post("/analyze")
def analyze(request: DecisionSupportRequest) -> dict:
    try:
        stress = request.stress.model_dump() if request.stress else None
        return run(request.timestamp_utc.isoformat(), llm=get_strategy_provider(), stress=stress)
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except DataNotReadyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Decision-support analysis failed.") from error
