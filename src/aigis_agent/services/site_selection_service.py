"""Simple site-selection scoring service."""

from __future__ import annotations

from aigis_agent.schemas.routing import Coordinate
from aigis_agent.schemas.site_selection import CandidateScore
from aigis_agent.utils.geo import haversine_distance_m


class SiteSelectionService:
    """Rank candidate sites using simple distance-based score."""

    def score(self, candidate_id: str, candidate: Coordinate, demand_points: list[Coordinate]) -> CandidateScore:
        """Return a single candidate score for fast MVP analysis."""
        if not demand_points:
            return CandidateScore(id=candidate_id, score=0.0, reason="No demand points provided")

        distances = [
            haversine_distance_m(candidate.lat, candidate.lon, d.lat, d.lon)
            for d in demand_points
        ]

        avg_dist = sum(distances) / len(distances)
        score = 1 / (1 + avg_dist)
        return CandidateScore(
            id=candidate_id,
            score=score,
            reason="Distance-only placeholder score; lower average distance is better",
        )
