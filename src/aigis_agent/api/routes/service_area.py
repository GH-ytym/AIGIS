"""Service-area analysis endpoints."""

from fastapi import APIRouter

from aigis_agent.schemas.service_area import IsochroneRequest, IsochroneResponse
from aigis_agent.services.isochrone_service import IsochroneService

router = APIRouter(prefix="/service-area", tags=["service-area"])
_service = IsochroneService()


@router.post("/isochrone", response_model=IsochroneResponse)
def service_area(payload: IsochroneRequest) -> IsochroneResponse:
    """Return first-phase isochrone polygon."""
    polygon = _service.calculate(payload.center, payload.minutes, payload.profile)
    return IsochroneResponse(
        center=payload.center,
        minutes=payload.minutes,
        polygon_geojson=polygon,
        note="Placeholder radius model; replace with network-time isochrone",
    )
