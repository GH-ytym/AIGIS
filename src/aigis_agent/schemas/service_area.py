"""Schemas for service-area (isochrone) analysis."""

from pydantic import BaseModel, Field

from aigis_agent.schemas.routing import Coordinate


class IsochroneRequest(BaseModel):
    """Input for service range analysis by travel time."""

    center: Coordinate
    minutes: int = Field(default=10, ge=1, le=120)
    profile: str = Field(default="driving")


class IsochroneResponse(BaseModel):
    """GeoJSON polygon response for isochrone boundary."""

    center: Coordinate
    minutes: int
    polygon_geojson: dict
    note: str = "TODO: compute real polygon from graph traversal"
