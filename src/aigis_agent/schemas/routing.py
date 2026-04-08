"""Schemas for simple route analysis."""

from pydantic import BaseModel, Field


class Coordinate(BaseModel):
    """WGS84 coordinate pair."""

    lat: float = Field(..., ge=-90, le=90)
    lon: float = Field(..., ge=-180, le=180)


class RouteRequest(BaseModel):
    """Route query input between two points."""

    origin: Coordinate
    destination: Coordinate
    profile: str = Field(default="driving", description="driving/walking/cycling")


class RouteResponse(BaseModel):
    """Route summary response from OSRM placeholder service."""

    distance_m: float | None = None
    duration_s: float | None = None
    geometry_geojson: dict | None = None
    note: str = "TODO: replace placeholder with OSRM real route"
