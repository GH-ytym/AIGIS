"""OSRM routing service wrapper via routingpy."""

from __future__ import annotations

from routingpy.routers import OSRM

from aigis_agent.core.config import settings
from aigis_agent.schemas.routing import Coordinate


class OSRMService:
    """Query OSRM for route analysis between two coordinates."""

    def __init__(self) -> None:
        self._client = OSRM(base_url=settings.osrm_base_url)

    def route(self, origin: Coordinate, destination: Coordinate, profile: str = "driving") -> dict:
        """Return route summary and GeoJSON geometry if available."""
        # TODO: handle network failures and fallback to local graph shortest path.
        result = self._client.directions(
            locations=[
                [origin.lon, origin.lat],
                [destination.lon, destination.lat],
            ],
            profile=profile,
            geometry="geojson",
        )

        route = result.routes[0] if result.routes else None
        if route is None:
            return {
                "distance_m": None,
                "duration_s": None,
                "geometry_geojson": None,
            }

        return {
            "distance_m": float(route.distance),
            "duration_s": float(route.duration),
            "geometry_geojson": route.geometry,
        }
