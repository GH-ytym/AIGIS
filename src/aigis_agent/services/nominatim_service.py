"""Nominatim-based geocoding and fuzzy POI search service."""

from __future__ import annotations

import time
from urllib.parse import urlparse

from geopy.exc import GeocoderServiceError, GeocoderTimedOut, GeocoderUnavailable
from geopy.geocoders import Nominatim

from aigis_agent.core.config import settings
from aigis_agent.schemas.poi import POIItem


class NominatimProviderError(RuntimeError):
    """Raised when upstream Nominatim request fails."""


class NominatimService:
    """Wrapper around Nominatim for POI/address search."""

    def __init__(self, user_agent: str | None = None) -> None:
        base_url = settings.nominatim_base_url.strip()
        parsed = urlparse(base_url if "://" in base_url else f"https://{base_url}")
        domain = parsed.netloc or parsed.path or "nominatim.openstreetmap.org"
        scheme = parsed.scheme or "https"

        self._geocoder = Nominatim(
            user_agent=user_agent or settings.nominatim_user_agent,
            domain=domain,
            scheme=scheme,
            timeout=settings.nominatim_timeout_s,
        )
        self._retry_count = max(0, settings.nominatim_retry_count)

    def search(self, query: str, city_hint: str | None = None, limit: int = 10) -> list[POIItem]:
        """Perform fuzzy search and normalize outputs to POI items."""
        full_query = f"{query}, {city_hint}" if city_hint else query

        locations = None
        for attempt in range(self._retry_count + 1):
            try:
                locations = self._geocoder.geocode(
                    full_query,
                    exactly_one=False,
                    limit=limit,
                    addressdetails=True,
                )
                break
            except (GeocoderTimedOut, GeocoderUnavailable) as exc:
                if attempt >= self._retry_count:
                    raise NominatimProviderError(
                        "Nominatim 请求超时或不可用，请稍后重试或切换可用节点。"
                    ) from exc
                time.sleep(min(0.5 * (attempt + 1), 2.0))
            except GeocoderServiceError as exc:
                raise NominatimProviderError(f"Nominatim 服务错误: {exc}") from exc
            except Exception as exc:
                raise NominatimProviderError(
                    f"Nominatim 调用失败: {exc.__class__.__name__}"
                ) from exc

        if not locations:
            return []

        items: list[POIItem] = []
        for loc in locations:
            items.append(
                POIItem(
                    name=(loc.raw.get("name") or query),
                    address=loc.address,
                    lat=loc.latitude,
                    lon=loc.longitude,
                )
            )
        return items
