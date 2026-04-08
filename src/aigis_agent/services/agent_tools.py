"""LangChain tools that bridge to first-wave GIS service modules."""

from __future__ import annotations

import json
from typing import Any

from langchain_core.tools import tool
from pydantic import BaseModel

from aigis_agent.schemas.amap_tools import (
    CoordinateConvertResult,
    DistrictQueryResult,
    KeywordSearchResult,
    NearbySearchResult,
    POIDetailResult,
    ResolvePlaceResult,
    ReverseGeocodeResult,
    RoutePlanResult,
)
from aigis_agent.schemas.routing import Coordinate
from aigis_agent.services.amap_service import AMapService
from aigis_agent.services.isochrone_service import IsochroneService
from aigis_agent.services.osrm_service import OSRMService
from aigis_agent.services.poi_search_service import POISearchService
from aigis_agent.services.site_selection_service import SiteSelectionService

_amap_service = AMapService()
_osrm_service = OSRMService()
_isochrone_service = IsochroneService()
_poi_search_service = POISearchService()
_site_selection_service = SiteSelectionService()


def _todo_tool_result(tool_name: str, api_list: list[str], schema: BaseModel) -> dict:
    """Return placeholder payload for not-yet-implemented tools."""
    return {
        "status": "todo",
        "tool": tool_name,
        "api_candidates": api_list,
        "schema": schema.model_dump(),
        "message": "Tool scaffold created. Implementation pending.",
    }


@tool
def resolve_place(query: str, city_hint: str = "") -> dict:
    """Resolve place text to canonical candidates (placeholder).

    Suggested AMap APIs: Input Tips, Keyword Search, Geocode, ID Query.
    """
    _ = (query, city_hint)
    return _todo_tool_result(
        tool_name="resolve_place",
        api_list=["input_tips", "keyword_search", "geocode", "id_query"],
        schema=ResolvePlaceResult(),
    )


@tool
def search_keyword_poi(query: str, city_hint: str = "", limit: int = 10) -> dict:
    """Search POIs by keyword (placeholder).

    Suggested AMap API: Keyword Search.
    """
    _ = (query, city_hint, limit)
    return _todo_tool_result(
        tool_name="search_keyword_poi",
        api_list=["keyword_search"],
        schema=KeywordSearchResult(),
    )


@tool
def search_nearby_poi(
    center_lat: float,
    center_lon: float,
    keywords: str,
    radius_m: int = 2000,
    limit: int = 20,
) -> dict:
    """Search nearby POIs around a center point (placeholder).

    Suggested AMap API: Nearby Search.
    """
    _ = (center_lat, center_lon, keywords, radius_m, limit)
    return _todo_tool_result(
        tool_name="search_nearby_poi",
        api_list=["nearby_search"],
        schema=NearbySearchResult(),
    )


@tool
def get_poi_detail(poi_id: str) -> dict:
    """Fetch POI details by POI ID (placeholder).

    Suggested AMap API: ID Query.
    """
    _ = poi_id
    return _todo_tool_result(
        tool_name="get_poi_detail",
        api_list=["id_query"],
        schema=POIDetailResult(),
    )


@tool
def reverse_geocode_point(lat: float, lon: float) -> dict:
    """Convert coordinates to readable address (placeholder).

    Suggested AMap API: Reverse Geocode.
    """
    _ = (lat, lon)
    return _todo_tool_result(
        tool_name="reverse_geocode_point",
        api_list=["reverse_geocode"],
        schema=ReverseGeocodeResult(),
    )


@tool
def plan_route(
    origin: str,
    destination: str,
    mode: str = "driving",
) -> dict:
    """Plan route between origin and destination (placeholder).

    Suggested AMap API: Route Planning.
    """
    _ = (origin, destination, mode)
    return _todo_tool_result(
        tool_name="plan_route",
        api_list=["route_planning"],
        schema=RoutePlanResult(),
    )


@tool
def resolve_admin_area(keywords: str, subdistrict: int = 1) -> dict:
    """Resolve administrative area hierarchy (placeholder).

    Suggested AMap API: District Query.
    """
    _ = (keywords, subdistrict)
    return _todo_tool_result(
        tool_name="resolve_admin_area",
        api_list=["district_query"],
        schema=DistrictQueryResult(),
    )


@tool
def convert_coordinate(
    coords_text: str,
    from_system: str = "gps",
    to_system: str = "amap",
) -> dict:
    """Convert coordinate system for map interoperability (placeholder).

    Suggested AMap API: Coordinate Convert.
    """
    _ = (coords_text, from_system, to_system)
    return _todo_tool_result(
        tool_name="convert_coordinate",
        api_list=["coordinate_convert"],
        schema=CoordinateConvertResult(),
    )


@tool
def search_poi(
    query: str,
    city_hint: str = "",
    limit: int = 5,
    allow_city_switch: bool = False,
) -> dict:
    """Search POI/address by natural language and return normalized item list."""
    city = city_hint or None
    safe_limit = max(1, min(limit, 20))
    try:
        result = _poi_search_service.search_with_debug(
            query=query,
            city_hint=city,
            limit=safe_limit,
            allow_city_switch=allow_city_switch,
        )
        return {
            "query": query,
            "count": len(result.items),
            "items": [item.model_dump() for item in result.items],
            "debug": result.debug.model_dump(),
        }
    except Exception as exc:
        return {
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
    """Score candidate sites from JSON arrays.

    candidates_json example:
    [{"id":"A","lat":31.23,"lon":121.47},{"id":"B","lat":31.20,"lon":121.50}]

    demand_points_json example:
    [{"lat":31.24,"lon":121.48},{"lat":31.22,"lon":121.46}]
    """
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


_EXECUTABLE_TOOL_REGISTRY: dict[str, Any] = {
    "search_poi": search_poi,
}


def get_execution_tool_names() -> list[str]:
    """Return tool names that are production-ready for direct execution."""
    return list(_EXECUTABLE_TOOL_REGISTRY.keys())


def invoke_execution_tool(tool_name: str, args: dict[str, Any]) -> dict[str, Any]:
    """Invoke an executable tool by name with normalized dict output."""
    tool_obj = _EXECUTABLE_TOOL_REGISTRY.get(tool_name)
    if tool_obj is None:
        raise ValueError(f"Tool not executable: {tool_name}")

    result = tool_obj.invoke(args)
    if isinstance(result, dict):
        return result

    return {
        "result": result,
    }
