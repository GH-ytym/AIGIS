"""LangChain tools that bridge to first-wave GIS service modules."""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel, Field, ValidationError

from aigis_agent.core.config import settings
from aigis_agent.schemas.routing import Coordinate
from aigis_agent.services.isochrone_service import IsochroneService
from aigis_agent.services.osrm_service import OSRMService
from aigis_agent.services.poi_search_service import POISearchService
from aigis_agent.services.site_selection_service import SiteSelectionService

_osrm_service = OSRMService()
_isochrone_service = IsochroneService()
_poi_search_service = POISearchService()
_site_selection_service = SiteSelectionService()


@dataclass(frozen=True)
class ExecutableToolSpec:
    """Execution spec for tools allowed in direct agent execution path."""

    tool_obj: Any
    args_model: type[BaseModel]
    enabled: bool = True


class SearchPOIArgs(BaseModel):
    """Validated arguments for production POI search tool."""

    query: str = Field(default="", max_length=settings.poi_ai_query_max_length)
    city_hint: str = ""
    limit: int = Field(default=10, ge=1, le=50)
    allow_city_switch: bool = False


def _not_enabled_tool_result(tool_name: str, message: str) -> dict[str, Any]:
    """Return standardized payload for tools intentionally disabled in execution path."""
    return {
        "status": "not_enabled",
        "tool": tool_name,
        "message": message,
        "count": 0,
        "items": [],
    }


def _tool_validation_error(tool_name: str, exc: ValidationError) -> dict[str, Any]:
    """Return normalized validation error payload for tool args."""
    return {
        "status": "error",
        "tool": tool_name,
        "error": f"Invalid args for {tool_name}",
        "validation_errors": exc.errors(),
        "count": 0,
        "items": [],
    }


@tool
def resolve_place(query: str, city_hint: str = "") -> dict:
    """Resolve place text to canonical candidates (not enabled in execution path)."""
    _ = (query, city_hint)
    return _not_enabled_tool_result("resolve_place", "Tool not enabled yet.")


@tool
def search_keyword_poi(query: str, city_hint: str = "", limit: int = 10) -> dict:
    """Search POIs by keyword (not enabled in execution path)."""
    _ = (query, city_hint, limit)
    return _not_enabled_tool_result("search_keyword_poi", "Tool not enabled yet.")


@tool
def search_nearby_poi(
    center_lat: float,
    center_lon: float,
    keywords: str,
    radius_m: int = 2000,
    limit: int = 20,
) -> dict:
    """Search nearby POIs around a center point (not enabled in execution path)."""
    _ = (center_lat, center_lon, keywords, radius_m, limit)
    return _not_enabled_tool_result("search_nearby_poi", "Tool not enabled yet.")


@tool
def get_poi_detail(poi_id: str) -> dict:
    """Fetch POI details by POI ID (not enabled in execution path)."""
    _ = poi_id
    return _not_enabled_tool_result("get_poi_detail", "Tool not enabled yet.")


@tool
def reverse_geocode_point(lat: float, lon: float) -> dict:
    """Convert coordinates to readable address (not enabled in execution path)."""
    _ = (lat, lon)
    return _not_enabled_tool_result("reverse_geocode_point", "Tool not enabled yet.")


@tool
def plan_route(
    origin: str,
    destination: str,
    mode: str = "driving",
) -> dict:
    """Plan route between origin and destination (not enabled in execution path)."""
    _ = (origin, destination, mode)
    return _not_enabled_tool_result("plan_route", "Tool not enabled yet.")


@tool
def resolve_admin_area(keywords: str, subdistrict: int = 1) -> dict:
    """Resolve administrative area hierarchy (not enabled in execution path)."""
    _ = (keywords, subdistrict)
    return _not_enabled_tool_result("resolve_admin_area", "Tool not enabled yet.")


@tool
def convert_coordinate(
    coords_text: str,
    from_system: str = "gps",
    to_system: str = "amap",
) -> dict:
    """Convert coordinate system for map interoperability (not enabled in execution path)."""
    _ = (coords_text, from_system, to_system)
    return _not_enabled_tool_result("convert_coordinate", "Tool not enabled yet.")


@tool
def search_poi(
    query: str,
    city_hint: str = "",
    limit: int = 10,
    allow_city_switch: bool = False,
) -> dict:
    """Search POI/address by natural language and return normalized item list."""
    try:
        parsed = SearchPOIArgs.model_validate(
            {
                "query": query,
                "city_hint": city_hint,
                "limit": limit,
                "allow_city_switch": allow_city_switch,
            }
        )
    except ValidationError as exc:
        return _tool_validation_error("search_poi", exc)

    city = parsed.city_hint.strip() or None
    safe_limit = max(1, min(parsed.limit, 50))

    try:
        result = _poi_search_service.search_with_debug(
            query=parsed.query,
            city_hint=city,
            limit=safe_limit,
            allow_city_switch=parsed.allow_city_switch,
        )
        return {
            "status": "ok",
            "query": parsed.query,
            "count": len(result.items),
            "items": [item.model_dump() for item in result.items],
            "debug": result.debug.model_dump(),
        }
    except Exception as exc:
        return {
            "status": "error",
            "error": f"POI search failed: {exc}",
            "count": 0,
            "items": [],
            "debug": {
                "status": "error",
                "message": str(exc),
            },
        }


@tool
def analyze_route(
    origin_lat: float,
    origin_lon: float,
    destination_lat: float,
    destination_lon: float,
    profile: str = "driving",
) -> dict:
    """Analyze route summary between two coordinates using OSRM."""
    try:
        origin = Coordinate(lat=origin_lat, lon=origin_lon)
        destination = Coordinate(lat=destination_lat, lon=destination_lon)
        route = _osrm_service.route(origin=origin, destination=destination, profile=profile)
        return route
    except Exception as exc:
        return {
            "error": f"Route analysis failed: {exc}",
            "distance_m": None,
            "duration_s": None,
            "geometry_geojson": None,
        }


@tool
def build_service_area(
    center_lat: float,
    center_lon: float,
    minutes: int = 10,
    profile: str = "driving",
) -> dict:
    """Build service-area polygon (current version uses placeholder isochrone model)."""
    try:
        center = Coordinate(lat=center_lat, lon=center_lon)
        safe_minutes = max(1, min(minutes, 120))
        polygon = _isochrone_service.calculate(center=center, minutes=safe_minutes, profile=profile)
        return {
            "center": center.model_dump(),
            "minutes": safe_minutes,
            "polygon_geojson": polygon,
        }
    except Exception as exc:
        return {
            "error": f"Service-area analysis failed: {exc}",
            "center": {"lat": center_lat, "lon": center_lon},
            "minutes": minutes,
            "polygon_geojson": None,
        }


@tool
def score_sites(candidates_json: str, demand_points_json: str) -> dict:
    """Score candidate sites from JSON arrays."""
    try:
        raw_candidates = json.loads(candidates_json)
        raw_demands = json.loads(demand_points_json)

        demand_points = [Coordinate(lat=float(d["lat"]), lon=float(d["lon"])) for d in raw_demands]

        ranked = []
        for candidate in raw_candidates:
            score = _site_selection_service.score(
                candidate_id=str(candidate["id"]),
                candidate=Coordinate(lat=float(candidate["lat"]), lon=float(candidate["lon"])),
                demand_points=demand_points,
            )
            ranked.append(score.model_dump())

        ranked.sort(key=lambda x: x["score"], reverse=True)
        return {
            "ranked_candidates": ranked,
            "count": len(ranked),
        }
    except Exception as exc:
        return {
            "error": f"Site-selection scoring failed: {exc}",
            "ranked_candidates": [],
            "count": 0,
        }


def get_gis_tools() -> list:
    """Return all GIS tools exposed to the LangChain agent."""
    return [
        resolve_place,
        search_keyword_poi,
        search_nearby_poi,
        get_poi_detail,
        reverse_geocode_point,
        plan_route,
        resolve_admin_area,
        convert_coordinate,
        search_poi,
        analyze_route,
        build_service_area,
        score_sites,
    ]


_EXECUTABLE_TOOL_REGISTRY: dict[str, ExecutableToolSpec] = {
    "search_poi": ExecutableToolSpec(tool_obj=search_poi, args_model=SearchPOIArgs, enabled=True),
}


def get_execution_tool_names() -> list[str]:
    """Return enabled tool names ready for direct execution."""
    return [
        tool_name
        for tool_name, spec in _EXECUTABLE_TOOL_REGISTRY.items()
        if spec.enabled
    ]


def invoke_execution_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Invoke an executable tool by name with validated args and normalized output."""
    spec = _EXECUTABLE_TOOL_REGISTRY.get(tool_name)
    if spec is None:
        raise ValueError(f"Tool not executable: {tool_name}")

    if not spec.enabled:
        return _not_enabled_tool_result(tool_name, "Tool is disabled by execution policy.")

    try:
        validated_args = spec.args_model.model_validate(args or {})
    except ValidationError as exc:
        return _tool_validation_error(tool_name, exc)

    result = spec.tool_obj.invoke(validated_args.model_dump())
    if isinstance(result, dict):
        return result

    return {
        "status": "ok",
        "result": result,
    }
