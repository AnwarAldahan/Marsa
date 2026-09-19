"""Integrated decision-support endpoint."""
from __future__ import annotations

from datetime import datetime, timedelta

from fastapi import APIRouter, HTTPException
from pydantic import BaseModel, ConfigDict, field_validator

from marsa.common.exceptions import DataNotReadyError
from marsa.api.dependencies import get_strategy_provider
from marsa.pipeline import run

router = APIRouter(prefix="/api/decision-support", tags=["decision-support"])


class DecisionSupportRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    timestamp_utc: datetime

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
        return run(request.timestamp_utc.isoformat(), llm=get_strategy_provider())
    except KeyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except DataNotReadyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except ValueError as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Decision-support analysis failed.") from error
