"""Regression tests for rebuilt POI fuzzy search orchestration."""

from __future__ import annotations

from aigis_agent.schemas.poi import POIItem
from aigis_agent.core.config import settings
from aigis_agent.services.poi_search_service import (
    POISearchService,
    QueryRefineDecision,
    StrongLandmarkDecision,
)


def _poi(name: str, city: str, address: str = "") -> POIItem:
    return POIItem(
        name=name,
        address=address or f"{city}{name}",
        lat=30.0,
        lon=114.0,
        city=city,
        province="测试省",
        district="测试区",
    )


class FakeAMapService:
    """In-memory deterministic search backend for unit tests."""

    def search(self, query: str, city_hint: str | None = None, limit: int = 10):
        if city_hint == "上海市":
            return [_poi("东方明珠", "上海市")][:limit]
        if city_hint == "武汉市":
            return [_poi("武汉广场", "武汉市")][:limit]
        return [_poi("默认地点", city_hint or "未知市")][:limit]


class FakeRefiner:
    """Deterministic refiner used to emulate AI outputs."""

    def __init__(
        self,
        decision: QueryRefineDecision | None,
        enabled: bool = True,
        strong_decision: StrongLandmarkDecision | None = None,
    ):
        self._decision = decision
        self.enabled = enabled
        self._strong_decision = strong_decision

    def refine(self, **_kwargs):
        return self._decision

    def detect_strong_landmark(self, **_kwargs):
        return self._strong_decision


class FakeAMapStrongLandmarkService:
    """Service where local city has misleading full-match landmark names."""

    def search(self, query: str, city_hint: str | None = None, limit: int = 10):
        if city_hint == "武汉市":
            return [_poi("东方明珠3期", "武汉市", "武汉市某小区")][:limit]
        if city_hint == "上海市":
            return [_poi("东方明珠", "上海市", "上海市浦东新区")][:limit]
        return [_poi("默认地点", city_hint or "未知市")][:limit]


class FakeAMapCrossCityLeakService:
    """Service that leaks cross-city result despite city_hint constraint."""

    def search(self, query: str, city_hint: str | None = None, limit: int = 10):
        _ = query
        if city_hint == "上海市":
            return [_poi("北京大学", "北京市", "北京市海淀区")][:limit]
        return [_poi("默认地点", city_hint or "未知市")][:limit]


def test_search_with_debug_low_confidence_fallback_reason() -> None:
    decision = QueryRefineDecision(
        cleaned_query="东方明珠",
        city_hint_override="上海市",
        confidence=0.1,
        reason="low",
    )
    service = POISearchService(amap_service=FakeAMapService(), query_refiner=FakeRefiner(decision))

    result = service.search_with_debug(
        query="东方明珠",
        city_hint="武汉市",
        limit=10,
        allow_city_switch=False,
    )

    assert result.debug.ai_refine_triggered is True
    assert result.debug.ai_refine_applied is False
    assert result.debug.fallback_reason == "ai_low_confidence"
    assert result.debug.final_city_hint == "武汉市"


def test_search_with_debug_city_switch_suggested_when_disallowed() -> None:
    decision = QueryRefineDecision(
        cleaned_query="东方明珠",
        city_hint_override="上海市",
        confidence=0.95,
        reason="city mismatch",
    )
    service = POISearchService(amap_service=FakeAMapService(), query_refiner=FakeRefiner(decision))

    result = service.search_with_debug(
        query="东方明珠",
        city_hint="武汉市",
        limit=10,
        allow_city_switch=False,
    )

    assert result.debug.city_switch_suggested is True
    assert result.debug.city_switch_to == "上海市"
    assert result.debug.city_switch_applied is False
    assert result.debug.corrected_result_count == 1
    assert result.debug.final_city_hint == "武汉市"


def test_search_with_debug_city_switch_applied_when_allowed() -> None:
    decision = QueryRefineDecision(
        cleaned_query="东方明珠",
        city_hint_override="上海市",
        confidence=0.95,
        reason="city mismatch",
    )
    service = POISearchService(amap_service=FakeAMapService(), query_refiner=FakeRefiner(decision))

    result = service.search_with_debug(
        query="东方明珠",
        city_hint="武汉市",
        limit=10,
        allow_city_switch=True,
    )

    assert result.debug.city_switch_suggested is True
    assert result.debug.city_switch_applied is True
    assert result.debug.ai_refine_applied is True
    assert result.debug.final_city_hint == "上海市"
    assert result.items[0].name == "东方明珠"


def test_normalize_city_generalization() -> None:
    assert POISearchService._normalize_city("上海城区") == "上海市"
    assert POISearchService._normalize_city("杭州") == "杭州市"
    assert POISearchService._normalize_city("上海市（直辖）") == "上海市"
    assert POISearchService._normalize_city("延边朝鲜族自治州") == "延边朝鲜族自治州"
    assert POISearchService._normalize_city("吉林省") == "吉林省"


def test_city_scope_guard_suggests_switch_and_blocks_cross_city_when_disallowed() -> None:
    service = POISearchService(
        amap_service=FakeAMapCrossCityLeakService(),
        query_refiner=FakeRefiner(decision=None, enabled=False),
    )

    result = service.search_with_debug(
        query="北京大学",
        city_hint="上海市",
        limit=10,
        allow_city_switch=False,
    )

    assert result.debug.city_switch_suggested is True
    assert result.debug.city_switch_from == "上海市"
    assert result.debug.city_switch_to == "北京市"
    assert result.debug.city_switch_applied is False
    assert result.debug.trigger_reason == "top_result_city_mismatch"
    assert result.debug.fallback_reason == "city_scope_no_local_result"
    assert result.items == []


def test_strong_landmark_guard_suggests_city_switch_when_retry_not_triggered() -> None:
    strong = StrongLandmarkDecision(
        is_strong_landmark=True,
        city_hint_override="上海市",
        cleaned_query="东方明珠",
        confidence=0.95,
        reason="iconic landmark",
    )
    service = POISearchService(
        amap_service=FakeAMapStrongLandmarkService(),
        query_refiner=FakeRefiner(decision=None, strong_decision=strong),
    )

    result = service.search_with_debug(
        query="东方明珠",
        city_hint="武汉市",
        limit=10,
        allow_city_switch=False,
    )

    assert result.debug.ai_refine_triggered is True
    assert result.debug.city_switch_suggested is True
    assert result.debug.city_switch_to == "上海市"
    assert result.debug.city_switch_applied is False
    assert result.debug.trigger_reason == "strong_landmark_guard"


def test_strong_landmark_guard_can_apply_when_city_switch_allowed() -> None:
    strong = StrongLandmarkDecision(
        is_strong_landmark=True,
        city_hint_override="上海市",
        cleaned_query="东方明珠",
        confidence=0.95,
        reason="iconic landmark",
    )
    service = POISearchService(
        amap_service=FakeAMapStrongLandmarkService(),
        query_refiner=FakeRefiner(decision=None, strong_decision=strong),
    )

    result = service.search_with_debug(
        query="东方明珠",
        city_hint="武汉市",
        limit=10,
        allow_city_switch=True,
    )

    assert result.debug.ai_refine_triggered is True
    assert result.debug.city_switch_suggested is True
    assert result.debug.city_switch_applied is True
    assert result.debug.final_city_hint == "上海市"
    assert result.items[0].name == "东方明珠"


def test_strong_landmark_guard_suggests_but_not_apply_when_confidence_between_thresholds() -> None:
    old_suggest = settings.poi_ai_strong_landmark_suggest_confidence_threshold
    old_apply = settings.poi_ai_strong_landmark_confidence_threshold
    settings.poi_ai_strong_landmark_suggest_confidence_threshold = 0.60
    settings.poi_ai_strong_landmark_confidence_threshold = 0.80

    try:
        strong = StrongLandmarkDecision(
            is_strong_landmark=True,
            city_hint_override="上海市",
            cleaned_query="东方明珠",
            confidence=0.70,
            reason="likely landmark",
        )
        service = POISearchService(
            amap_service=FakeAMapStrongLandmarkService(),
            query_refiner=FakeRefiner(decision=None, strong_decision=strong),
        )

        result = service.search_with_debug(
            query="东方明珠",
            city_hint="武汉市",
            limit=10,
            allow_city_switch=True,
        )

        assert result.debug.city_switch_suggested is True
        assert result.debug.city_switch_applied is False
        assert result.debug.city_switch_to == "上海市"
        assert result.debug.fallback_reason == "strong_landmark_suggested_not_applied"
        assert result.debug.final_city_hint == "武汉市"
    finally:
        settings.poi_ai_strong_landmark_suggest_confidence_threshold = old_suggest
        settings.poi_ai_strong_landmark_confidence_threshold = old_apply
