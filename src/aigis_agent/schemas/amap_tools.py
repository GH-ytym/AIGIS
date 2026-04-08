"""Placeholder schemas for AMap-oriented LangChain tools."""

from pydantic import BaseModel


class ResolvePlaceResult(BaseModel):
    """TODO: fill fields for place resolution output."""

    pass


class KeywordSearchResult(BaseModel):
    """TODO: fill fields for keyword search output."""

    pass


class NearbySearchResult(BaseModel):
    """TODO: fill fields for nearby search output."""

    pass


class POIDetailResult(BaseModel):
    """TODO: fill fields for POI detail output."""

    pass


class ReverseGeocodeResult(BaseModel):
    """TODO: fill fields for reverse geocode output."""

    pass


class RoutePlanResult(BaseModel):
    """TODO: fill fields for route planning output."""

    pass


class DistrictQueryResult(BaseModel):
    """TODO: fill fields for district query output."""

    pass


class CoordinateConvertResult(BaseModel):
    """TODO: fill fields for coordinate conversion output."""

    pass