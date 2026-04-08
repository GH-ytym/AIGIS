"""POI and address search endpoints."""

from fastapi import APIRouter, HTTPException, Query, status

from aigis_agent.schemas.poi import (
    DistrictQueryResponse,
    POISearchRequest,
    POISearchResponse,
)
from aigis_agent.services.amap_service import AMapProviderError, AMapService
from aigis_agent.services.poi_search_service import POISearchService

router = APIRouter(prefix="/poi", tags=["poi"])
_poi_search_service = POISearchService()
_district_service = AMapService()


@router.post("/search", response_model=POISearchResponse)
def search_poi(payload: POISearchRequest) -> POISearchResponse:
    """Search POI/address from natural language and return normalized items."""
    try:
        result = _poi_search_service.search_with_debug(
            payload.query,
            payload.city_hint,
            payload.limit,
            payload.allow_city_switch,
        )
    except AMapProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    return POISearchResponse(query=payload.query, items=result.items, debug=result.debug)


@router.get("/districts", response_model=DistrictQueryResponse)
def query_districts(
    keywords: str = Query(default="中国", description="行政区关键词（名称/citycode/adcode）"),
    subdistrict: int = Query(default=1, ge=0, le=3, description="下级行政区层级深度"),
    page: int = Query(default=1, ge=1, description="分页页码"),
    offset: int = Query(default=50, ge=1, le=50, description="返回条数"),
    filter_adcode: str | None = Query(default=None, alias="filter", description="按 adcode 过滤"),
) -> DistrictQueryResponse:
    """Query administrative districts for province/city/county cascading selector."""
    try:
        items = _district_service.query_districts(
            keywords=keywords,
            subdistrict=subdistrict,
            page=page,
            offset=offset,
            filter_adcode=filter_adcode,
        )
    except AMapProviderError as exc:
        raise HTTPException(
            status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
            detail=str(exc),
        ) from exc

    query_term = filter_adcode or keywords
    return DistrictQueryResponse(query=query_term, items=items)
