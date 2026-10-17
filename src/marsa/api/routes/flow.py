"""GET /api/flow/norms: the usual range of each section-١ number, computed from the data files.

The dashboard compares the hour on screen with these percentiles. See marsa.data.flow_norms.
"""
from fastapi import APIRouter

from marsa.data.flow_norms import flow_norms

router = APIRouter(prefix="/api/flow", tags=["dashboard"])


@router.get("/norms")
def get_flow_norms() -> dict:
    return flow_norms()
