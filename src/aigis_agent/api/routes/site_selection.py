"""Simple site-selection endpoints."""

from fastapi import APIRouter

from aigis_agent.schemas.site_selection import SiteSelectionRequest, SiteSelectionResponse
from aigis_agent.services.site_selection_service import SiteSelectionService

router = APIRouter(prefix="/site-selection", tags=["site-selection"])
_service = SiteSelectionService()


@router.post("/score", response_model=SiteSelectionResponse)
def score_sites(payload: SiteSelectionRequest) -> SiteSelectionResponse:
    """Score candidate sites with first-wave lightweight logic."""
    ranked = [
        _service.score(c.id, c.location, payload.demand_points)
        for c in payload.candidates
    ]
    ranked.sort(key=lambda x: x.score, reverse=True)
    return SiteSelectionResponse(ranked_candidates=ranked)
