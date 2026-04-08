"""AMap-based POI search service for mainland China scenarios."""

from __future__ import annotations

import time

import httpx

from aigis_agent.core.config import settings
from aigis_agent.schemas.poi import DistrictItem, POIItem


class AMapProviderError(RuntimeError):
    """Raised when upstream AMap place API fails."""


class AMapService:
    """Wrapper around AMap Place Text API for POI/address lookup."""

    def __init__(self) -> None:
        self._api_key = settings.amap_api_key
        self._base_url = settings.amap_base_url.rstrip("/")
        self._timeout_s = settings.amap_timeout_s
        self._retry_count = max(0, settings.amap_retry_count)

    def search(self, query: str, city_hint: str | None = None, limit: int = 10) -> list[POIItem]:
        """Search POI by keyword and optional city constraint."""
        if not self._api_key:
            raise AMapProviderError("未配置 AIGIS_AMAP_API_KEY，无法调用高德 POI 服务。")

        safe_limit = max(1, min(limit, 50))
        params = {
            "key": self._api_key,
            "keywords": query,
            "offset": safe_limit,
            "page": 1,
            "extensions": "base",
            "citylimit": "true" if city_hint else "false",
        }
        if city_hint:
            params["city"] = city_hint

        data: dict | None = None
        endpoint = f"{self._base_url}/v3/place/text"
        for attempt in range(self._retry_count + 1):
            try:
                response = httpx.get(endpoint, params=params, timeout=self._timeout_s)
                response.raise_for_status()
                data = response.json()
                break
            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as exc:
                if attempt >= self._retry_count:
                    raise AMapProviderError(
                        "高德 POI 请求超时或连接失败，请稍后重试。"
                    ) from exc
                time.sleep(min(0.5 * (attempt + 1), 2.0))
            except httpx.HTTPStatusError as exc:
                raise AMapProviderError(f"高德服务 HTTP 错误: {exc.response.status_code}") from exc
            except ValueError as exc:
                raise AMapProviderError("高德返回数据无法解析为 JSON。") from exc
            except httpx.RequestError as exc:
                raise AMapProviderError(f"高德请求失败: {exc.__class__.__name__}") from exc

        if data is None:
            return []

        if str(data.get("status", "0")) != "1":
            info = data.get("info") or "未知错误"
            infocode = data.get("infocode") or "-"
            raise AMapProviderError(f"高德服务错误: {info} (infocode={infocode})")

        pois = data.get("pois") or []
        items: list[POIItem] = []
        for poi in pois:
            location = str(poi.get("location") or "")
            if not location or "," not in location:
                continue

            lon_str, lat_str = location.split(",", 1)
            try:
                lon = float(lon_str)
                lat = float(lat_str)
            except ValueError:
                continue

            address_parts = [
                str(poi.get("pname") or "").strip(),
                str(poi.get("cityname") or "").strip(),
                str(poi.get("adname") or "").strip(),
                str(poi.get("address") or "").strip(),
            ]
            merged_address = " ".join(part for part in address_parts if part)
            if not merged_address:
                merged_address = str(poi.get("name") or query)

            province = str(poi.get("pname") or "").strip() or None
            city = str(poi.get("cityname") or "").strip() or None
            district = str(poi.get("adname") or "").strip() or None

            items.append(
                POIItem(
                    name=str(poi.get("name") or query),
                    address=merged_address,
                    lat=lat,
                    lon=lon,
                    source="amap",
                    province=province,
                    city=city,
                    district=district,
                )
            )

        return items

    def query_districts(
        self,
        keywords: str = "中国",
        subdistrict: int = 1,
        page: int = 1,
        offset: int = 50,
        filter_adcode: str | None = None,
    ) -> list[DistrictItem]:
        """Query administrative district list from AMap district API."""
        if not self._api_key:
            raise AMapProviderError("未配置 AIGIS_AMAP_API_KEY，无法调用高德行政区划服务。")

        safe_subdistrict = max(0, min(subdistrict, 3))
        safe_page = max(1, page)
        safe_offset = max(1, min(offset, 50))

        params: dict[str, str | int] = {
            "key": self._api_key,
            "keywords": keywords or "中国",
            "subdistrict": safe_subdistrict,
            "page": safe_page,
            "offset": safe_offset,
            "extensions": "base",
        }
        if filter_adcode:
            params["filter"] = filter_adcode

        data: dict | None = None
        endpoint = f"{self._base_url}/v3/config/district"
        for attempt in range(self._retry_count + 1):
            try:
                response = httpx.get(endpoint, params=params, timeout=self._timeout_s)
                response.raise_for_status()
                data = response.json()
                break
            except (httpx.ConnectTimeout, httpx.ReadTimeout, httpx.ConnectError) as exc:
                if attempt >= self._retry_count:
                    raise AMapProviderError(
                        "高德行政区划请求超时或连接失败，请稍后重试。"
                    ) from exc
                time.sleep(min(0.5 * (attempt + 1), 2.0))
            except httpx.HTTPStatusError as exc:
                raise AMapProviderError(f"高德服务 HTTP 错误: {exc.response.status_code}") from exc
            except ValueError as exc:
                raise AMapProviderError("高德返回数据无法解析为 JSON。") from exc
            except httpx.RequestError as exc:
                raise AMapProviderError(f"高德请求失败: {exc.__class__.__name__}") from exc

        if data is None:
            return []

        if str(data.get("status", "0")) != "1":
            info = data.get("info") or "未知错误"
            infocode = data.get("infocode") or "-"
            raise AMapProviderError(f"高德服务错误: {info} (infocode={infocode})")

        roots = data.get("districts") or []
        if not roots:
            return []

        parent = roots[0] if isinstance(roots[0], dict) else {}
        children = parent.get("districts") or []
        if not children:
            children = [parent]

        items: list[DistrictItem] = []
        for raw_item in children:
            if not isinstance(raw_item, dict):
                continue

            name = str(raw_item.get("name") or "").strip()
            adcode = str(raw_item.get("adcode") or "").strip()
            level = str(raw_item.get("level") or "unknown").strip()
            center = str(raw_item.get("center") or "").strip()

            center_lon: float | None = None
            center_lat: float | None = None
            if center and "," in center:
                lon_str, lat_str = center.split(",", 1)
                try:
                    center_lon = float(lon_str)
                    center_lat = float(lat_str)
                except ValueError:
                    center_lon = None
                    center_lat = None

            if not name:
                continue

            items.append(
                DistrictItem(
                    name=name,
                    adcode=adcode,
                    level=level,
                    center_lat=center_lat,
                    center_lon=center_lon,
                )
            )

        return items
