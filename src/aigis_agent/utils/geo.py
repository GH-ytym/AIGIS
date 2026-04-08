"""Geospatial helper functions used by placeholder analysis flows."""

from __future__ import annotations

from math import asin, cos, pi, radians, sin, sqrt


def haversine_distance_m(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
    """Compute approximate distance in meters on Earth surface."""
    r = 6371000.0
    d_lat = radians(lat2 - lat1)
    d_lon = radians(lon2 - lon1)
    a = (
        sin(d_lat / 2) ** 2
        + cos(radians(lat1)) * cos(radians(lat2)) * sin(d_lon / 2) ** 2
    )
    c = 2 * asin(sqrt(a))
    return r * c


def approximate_circle_polygon(lat: float, lon: float, radius_m: float, steps: int = 36) -> dict:
    """Generate a rough GeoJSON circle polygon in WGS84 for demo use."""
    if steps < 8:
        steps = 8

    lat_deg_per_m = 1 / 111000
    lon_deg_per_m = 1 / (111000 * max(cos(radians(lat)), 0.01))

    coords: list[list[float]] = []
    for i in range(steps):
        angle = 2 * pi * i / steps
        y = lat + sin(angle) * radius_m * lat_deg_per_m
        x = lon + cos(angle) * radius_m * lon_deg_per_m
        coords.append([x, y])
    coords.append(coords[0])

    return {
        "type": "Feature",
        "geometry": {
            "type": "Polygon",
            "coordinates": [coords],
        },
        "properties": {},
    }
