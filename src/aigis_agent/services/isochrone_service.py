"""Service-area (isochrone) computations for first iteration."""

from __future__ import annotations

from aigis_agent.schemas.routing import Coordinate
from aigis_agent.utils.geo import approximate_circle_polygon


class IsochroneService:
    """Compute travel-time coverage area with a simple placeholder model."""

    def calculate(self, center: Coordinate, minutes: int, profile: str = "driving") -> dict:
        """Return a rough isochrone polygon for early prototyping."""
        # TODO: replace with graph traversal from OSRM/OSMnx travel times.
        speed_kmh = 35 if profile == "driving" else 5
        radius_m = minutes * speed_kmh * 1000 / 60
        return approximate_circle_polygon(center.lat, center.lon, radius_m)
