"""Schemas for simple site-selection scoring."""

from pydantic import BaseModel, Field

from aigis_agent.schemas.routing import Coordinate


class CandidatePoint(BaseModel):
    """Candidate point used in quick location analysis."""

    id: str
    location: Coordinate


class SiteSelectionRequest(BaseModel):
    """Input for weighted site scoring."""

    candidates: list[CandidatePoint]
    demand_points: list[Coordinate] = Field(default_factory=list)


class CandidateScore(BaseModel):
    """Simple scored output for each candidate."""

    id: str
    score: float
    reason: str


class SiteSelectionResponse(BaseModel):
    """Sorted candidate ranking."""

    ranked_candidates: list[CandidateScore]
    note: str = "TODO: add real GIS indicators and multi-factor weighting"
