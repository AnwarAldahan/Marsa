"""HTTP transport for Maritime investigation."""

from fastapi import APIRouter, Depends, HTTPException
from pydantic import ValidationError

from marsa.agents.maritime_agent import MaritimeInvestigationAgent, MaritimeResult
from marsa.api.dependencies import get_maritime_agent
from marsa.api.schemas.agents import MaritimeApiRequest
from marsa.common.exceptions import DataNotReadyError

router = APIRouter(prefix="/api/agents/maritime", tags=["maritime-agent"])


@router.post("/investigate", response_model=MaritimeResult)
def investigate_maritime(
    request: MaritimeApiRequest,
    agent: MaritimeInvestigationAgent = Depends(get_maritime_agent),
) -> MaritimeResult:
    try:
        return agent.investigate(request.to_agent_request())
    except DataNotReadyError as error:
        raise HTTPException(status_code=404, detail=str(error)) from error
    except (ValidationError, ValueError) as error:
        raise HTTPException(status_code=422, detail=str(error)) from error
    except Exception as error:
        raise HTTPException(status_code=500, detail="Maritime investigation failed.") from error
