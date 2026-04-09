"""Schemas for geocoding and fuzzy POI search."""

from pydantic import BaseModel, Field


class POISearchRequest(BaseModel):
    """Natural-language query payload for POI or address lookup."""

    query: str = Field(..., description="Natural language POI/address query")
    city_hint: str | None = Field(default=None, description="Optional city hint to improve recall")
    limit: int = Field(default=10, ge=1, le=50)
    allow_city_switch: bool = Field(
        default=True,
        description="Whether AI/strategy can switch city hint for landmark conflicts",
    )


class POIItem(BaseModel):
    """Normalized POI result item."""

    name: str
    address: str
    lat: float
    lon: float
    source: str = "amap"
    province: str | None = None
    city: str | None = None
    district: str | None = None


class POISearchDebug(BaseModel):
    """Debug payload for POI query normalization and AI refinement."""

    ai_refine_triggered: bool = False
    ai_refine_applied: bool = False
    parallel_recall_used: bool = False
    parallel_local_result_count: int = 0
    parallel_global_result_count: int = 0
    parallel_local_score: float = 0.0
    parallel_best_city: str | None = None
    parallel_best_city_score: float = 0.0
    parallel_switch_reason: str | None = None
    city_switch_suggested: bool = False
    city_switch_applied: bool = False
    original_query: str
    normalized_query: str
    final_query: str
    original_city_hint: str | None = None
    final_city_hint: str | None = None
    city_switch_from: str | None = None
    city_switch_to: str | None = None
    trigger_reason: str | None = None
    corrected_query: str | None = None
    corrected_city_hint: str | None = None
    corrected_result_count: int = 0
    corrected_items: list[POIItem] = Field(default_factory=list)
    ai_confidence: float | None = None
    fallback_reason: str | None = None


class POISearchResponse(BaseModel):
    """Response payload for fuzzy search/geocode result set."""

    query: str
    items: list[POIItem]
    debug: POISearchDebug | None = None


class DistrictItem(BaseModel):
    """Administrative district item from AMap district query."""

    name: str
    adcode: str
    level: str
    center_lat: float | None = None
    center_lon: float | None = None


class DistrictQueryResponse(BaseModel):
    """Response payload for administrative district list."""

    query: str
    items: list[DistrictItem]
