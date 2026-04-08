"""Route analysis endpoints based on OSRM."""

from fastapi import APIRouter

from aigis_agent.schemas.routing import RouteRequest, RouteResponse
from aigis_agent.services.osrm_service import OSRMService

router = APIRouter(prefix="/routing", tags=["routing"])
_service = OSRMService()


@router.post("/route", response_model=RouteResponse)
def route_analysis(payload: RouteRequest) -> RouteResponse:
    """Compute simple route summary between origin and destination."""
    try:
        data = _service.route(payload.origin, payload.destination, payload.profile)
        return RouteResponse(**data, note="OSRM route result")
    except Exception:
        return RouteResponse(note="TODO: route failed, add robust fallback and diagnostics")
