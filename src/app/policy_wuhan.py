from __future__ import annotations

WUHAN_CENTER_LAT = 30.5928
WUHAN_CENTER_LON = 114.3055

# Approximate Wuhan city bounding box for future strict geo filtering.
WUHAN_BBOX = {
	"min_lat": 29.97,
	"max_lat": 31.36,
	"min_lon": 113.69,
	"max_lon": 115.08,
}


def in_wuhan_bbox(lat: float, lon: float) -> bool:
	"""Check whether a coordinate falls inside an approximate Wuhan bbox."""
	return (
		WUHAN_BBOX["min_lat"] <= lat <= WUHAN_BBOX["max_lat"]
		and WUHAN_BBOX["min_lon"] <= lon <= WUHAN_BBOX["max_lon"]
	)
