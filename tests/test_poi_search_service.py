"""Regression tests for rebuilt POI fuzzy search orchestration."""

from __future__ import annotations

import random

import pytest

from aigis_agent.schemas.poi import POIItem
from aigis_agent.core.config import settings
from aigis_agent.services.poi_search_service import (
    POISearchService,
    QueryRefineDecision,
    StrongLandmarkDecision,
)


def _build_random_scope_cases(count: int = 10) -> list[tuple[str, str, str]]:
    """Build deterministic random city/district pairs for switch regression checks."""
    city_districts = {
        "上海市": ["浦东新区", "黄浦区", "徐汇区", "静安区"],
        "武汉市": ["洪山区", "江夏区", "武昌区", "青山区"],
        "北京市": ["海淀区", "朝阳区", "东城区", "丰台区"],
        "杭州市": ["西湖区", "拱墅区", "上城区", "滨江区"],
        "成都市": ["武侯区", "锦江区", "青羊区", "金牛区"],
    }

    rng = random.Random(20260409)
    unique_cases: set[tuple[str, str, str]] = set()
    cities = list(city_districts.keys())

    while len(unique_cases) < count:
        city = rng.choice(cities)
        source_district, target_district = rng.sample(city_districts[city], 2)
        unique_cases.add((city, source_district, target_district))

    return sorted(unique_cases)


RANDOM_SCOPE_CASES = _build_random_scope_cases()


def _poi(name: str, city: str, address: str = "", district: str = "测试区") -> POIItem:
    return POIItem(
        name=name,
        address=address or f"{city}{name}",
        lat=30.0,
        lon=114.0,
        city=city,
        province="测试省",
        district=district,
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


class FakeAMapSpecificPoiFuzzyService:
    """Service to emulate noisy-query recall and refined specific-POI hit."""

    def __init__(self, *, city_for_search: str, target_name: str, target_district: str, cross_district: str):
        self.city_for_search = city_for_search
        self.target_name = target_name
        self.target_district = target_district
        self.cross_district = cross_district

    def search(self, query: str, city_hint: str | None = None, limit: int = 10):
        assert city_hint == self.city_for_search

        if query == self.target_name:
            return [
                _poi(
                    self.target_name,
                    self.city_for_search,
                    f"{self.city_for_search}{self.cross_district}测试路",
                    district=self.cross_district,
                ),
                _poi(
                    self.target_name,
                    self.city_for_search,
                    f"{self.city_for_search}{self.target_district}示例街",
                    district=self.target_district,
                ),
            ][:limit]

        return [
            _poi(
                "模糊召回点",
                self.city_for_search,
                f"{self.city_for_search}{self.target_district}",
                district=self.target_district,
            )
        ][:limit]


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

    assert result.debug.city_switch_suggested is False
    assert result.debug.city_switch_to is None
    assert result.debug.city_switch_applied is False
    assert result.debug.corrected_result_count == 0
    assert result.debug.final_city_hint == "武汉市"
    assert result.debug.fallback_reason == "ai_no_effect"


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

    assert result.debug.city_switch_suggested is False
    assert result.debug.city_switch_applied is False
    assert result.debug.ai_refine_applied is False
    assert result.debug.final_city_hint == "武汉市"
    assert result.items[0].name == "武汉广场"


def test_normalize_city_generalization() -> None:
    assert POISearchService._normalize_city("上海城区") == "上海市"
    assert POISearchService._normalize_city("杭州") == "杭州市"
    assert POISearchService._normalize_city("上海市（直辖）") == "上海市"
    assert POISearchService._normalize_city("北京市东城区") == "北京市东城区"
    assert POISearchService._normalize_city("杭州市上城区") == "杭州市上城区"
    assert POISearchService._normalize_city("武汉洪山区") == "武汉市洪山区"
    assert POISearchService._normalize_city("武汉市洪山区") == "武汉市洪山区"
    assert POISearchService._normalize_city("延边朝鲜族自治州") == "延边朝鲜族自治州"
    assert POISearchService._normalize_city("吉林省") == "吉林省"


@pytest.mark.parametrize(
    ("raw_hint", "expected"),
    [
        ("武汉洪山区", "武汉市洪山区"),
        ("武汉市洪山区", "武汉市洪山区"),
        ("上海浦东新区", "上海市浦东新区"),
        ("北京市海淀区", "北京市海淀区"),
        ("杭州西湖区", "杭州市西湖区"),
    ],
)
def test_normalize_city_district_examples(raw_hint: str, expected: str) -> None:
    assert POISearchService._normalize_city(raw_hint) == expected


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

    assert result.debug.city_switch_suggested is False
    assert result.debug.city_switch_from is None
    assert result.debug.city_switch_to is None
    assert result.debug.city_switch_applied is False
    assert result.debug.trigger_reason is None
    assert result.debug.fallback_reason == "city_scope_no_local_result"
    assert result.items == []


def test_city_scope_guard_does_not_mismatch_when_city_hint_contains_district() -> None:
    class FakeAMapDistrictMatchService:
        def search(self, query: str, city_hint: str | None = None, limit: int = 10):
            _ = query
            assert city_hint == "武汉市"
            return [
                _poi("洪山饭店", "武汉市", "武汉市洪山区珞喻路", district="洪山区"),
                _poi("武昌饭店", "武汉市", "武汉市武昌区", district="武昌区"),
            ][:limit]

    service = POISearchService(
        amap_service=FakeAMapDistrictMatchService(),
        query_refiner=FakeRefiner(decision=None, enabled=False),
    )

    result = service.search_with_debug(
        query="饭店",
        city_hint="武汉市洪山区",
        limit=10,
        allow_city_switch=False,
    )

    assert result.debug.city_switch_suggested is False
    assert result.debug.city_switch_from is None
    assert result.debug.city_switch_to is None
    assert result.debug.final_city_hint == "武汉市洪山区"
    assert len(result.items) == 1
    assert result.items[0].district == "洪山区"


def test_city_scope_guard_filters_cross_district_when_city_hint_contains_district() -> None:
    class FakeAMapDistrictLeakService:
        def search(self, query: str, city_hint: str | None = None, limit: int = 10):
            _ = query
            assert city_hint == "武汉市"
            return [
                _poi("武昌饭店", "武汉市", "武汉市武昌区", district="武昌区"),
                _poi("洪山饭店", "武汉市", "武汉市洪山区珞喻路", district="洪山区"),
            ][:limit]

    service = POISearchService(
        amap_service=FakeAMapDistrictLeakService(),
        query_refiner=FakeRefiner(decision=None, enabled=False),
    )

    result = service.search_with_debug(
        query="饭店",
        city_hint="武汉市洪山区",
        limit=10,
        allow_city_switch=False,
    )

    assert result.debug.city_switch_suggested is False
    assert result.debug.city_switch_from is None
    assert result.debug.city_switch_to is None
    assert result.debug.city_switch_applied is False
    assert result.debug.fallback_reason is None
    assert len(result.items) == 1
    assert result.items[0].district == "洪山区"


@pytest.mark.parametrize(
    ("city_hint", "pass_city_hint", "target_district", "cross_district"),
    [
        ("武汉市洪山区", "武汉市", "洪山区", "武昌区"),
        ("上海市浦东新区", "上海市", "浦东新区", "黄浦区"),
        ("北京市海淀区", "北京市", "海淀区", "朝阳区"),
        ("杭州市西湖区", "杭州市", "西湖区", "拱墅区"),
    ],
)
def test_city_scope_guard_filters_cross_district_examples(
    city_hint: str,
    pass_city_hint: str,
    target_district: str,
    cross_district: str,
) -> None:
    class FakeAMapMultiDistrictService:
        def search(self, query: str, city_hint: str | None = None, limit: int = 10):
            _ = query
            assert city_hint == pass_city_hint
            city_name = pass_city_hint
            return [
                _poi("跨区饭店", city_name, f"{city_name}{cross_district}", district=cross_district),
                _poi("目标饭店", city_name, f"{city_name}{target_district}", district=target_district),
            ][:limit]

    service = POISearchService(
        amap_service=FakeAMapMultiDistrictService(),
        query_refiner=FakeRefiner(decision=None, enabled=False),
    )

    result = service.search_with_debug(
        query="饭店",
        city_hint=city_hint,
        limit=10,
        allow_city_switch=False,
    )

    assert result.debug.city_switch_suggested is False
    assert result.debug.city_switch_from is None
    assert result.debug.city_switch_applied is False
    assert result.debug.fallback_reason is None
    assert len(result.items) == 1
    assert result.items[0].district == target_district


@pytest.mark.parametrize(
    (
        "query",
        "city_hint",
        "refined_poi_name",
        "search_city",
        "target_district",
        "cross_district",
    ),
    [
        (
            "找一找武汉洪山区咖啡店",
            "武汉市洪山区",
            "星巴克武汉光谷店",
            "武汉市",
            "洪山区",
            "武昌区",
        ),
        (
            "麻烦帮我看哈武汉洪山去星巴氪咖非店呗",
            "武汉市洪山区",
            "星巴克武汉光谷店",
            "武汉市",
            "洪山区",
            "武昌区",
        ),
        (
            "帮我在上海浦东新区找下瑞幸咔啡",
            "上海市浦东新区",
            "瑞幸咖啡上海陆家嘴店",
            "上海市",
            "浦东新区",
            "黄浦区",
        ),
    ],
)
def test_fuzzy_nl_query_hits_specific_poi_with_district_precision(
    query: str,
    city_hint: str,
    refined_poi_name: str,
    search_city: str,
    target_district: str,
    cross_district: str,
) -> None:
    decision = QueryRefineDecision(
        cleaned_query=refined_poi_name,
        city_hint_override=None,
        confidence=0.95,
        reason="nl typo normalize",
    )
    service = POISearchService(
        amap_service=FakeAMapSpecificPoiFuzzyService(
            city_for_search=search_city,
            target_name=refined_poi_name,
            target_district=target_district,
            cross_district=cross_district,
        ),
        query_refiner=FakeRefiner(decision=decision),
    )

    result = service.search_with_debug(
        query=query,
        city_hint=city_hint,
        limit=10,
        allow_city_switch=False,
    )

    assert result.debug.ai_refine_triggered is True
    assert result.debug.ai_refine_applied is True
    assert result.debug.final_query == refined_poi_name
    assert result.debug.final_city_hint == city_hint
    assert len(result.items) == 1
    assert result.items[0].name == refined_poi_name
    assert result.items[0].district == target_district


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

    assert result.debug.ai_refine_triggered is False
    assert result.debug.city_switch_suggested is False
    assert result.debug.city_switch_to is None
    assert result.debug.city_switch_applied is False
    assert result.debug.trigger_reason is None


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

    assert result.debug.ai_refine_triggered is False
    assert result.debug.city_switch_suggested is False
    assert result.debug.city_switch_applied is False
    assert result.debug.final_city_hint == "武汉市"
    assert result.items[0].name == "东方明珠3期"


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

        assert result.debug.city_switch_suggested is False
        assert result.debug.city_switch_applied is False
        assert result.debug.city_switch_to is None
        assert result.debug.fallback_reason is None
        assert result.debug.final_city_hint == "武汉市"
    finally:
        settings.poi_ai_strong_landmark_suggest_confidence_threshold = old_suggest
        settings.poi_ai_strong_landmark_confidence_threshold = old_apply


@pytest.mark.parametrize(
    ("query", "stale_city_hint", "expected_search_city", "expected_district", "target_name"),
    [
        (
            "帮我在上海浦东新区找下瑞幸咔啡",
            "云南省昆明市",
            "上海市",
            "浦东新区",
            "瑞幸咖啡上海浦东店",
        ),
        (
            "找一找武汉洪山区咖啡店",
            "云南省昆明市",
            "武汉市",
            "洪山区",
            "光谷咖啡馆",
        ),
    ],
)
def test_query_explicit_scope_overrides_stale_city_hint(
    query: str,
    stale_city_hint: str,
    expected_search_city: str,
    expected_district: str,
    target_name: str,
) -> None:
    class FakeAMapExplicitScopeService:
        def search(self, query: str, city_hint: str | None = None, limit: int = 10):
            assert city_hint == expected_search_city
            _ = query
            return [
                _poi("云南干扰点", "昆明市", "昆明市盘龙区", district="盘龙区"),
                _poi(target_name, expected_search_city, f"{expected_search_city}{expected_district}", district=expected_district),
            ][:limit]

    service = POISearchService(
        amap_service=FakeAMapExplicitScopeService(),
        query_refiner=FakeRefiner(decision=None, enabled=False),
    )

    result = service.search_with_debug(
        query=query,
        city_hint=stale_city_hint,
        limit=10,
        allow_city_switch=False,
    )

    assert result.debug.final_city_hint == f"{expected_search_city}{expected_district}"
    assert result.debug.city_switch_suggested is False
    assert len(result.items) == 1
    assert result.items[0].name == target_name
    assert result.items[0].district == expected_district


def test_district_locked_query_does_not_switch_to_other_district() -> None:
    class FakeAMapOtherDistrictOnlyService:
        def search(self, query: str, city_hint: str | None = None, limit: int = 10):
            assert city_hint == "武汉市"
            _ = query
            return [
                _poi("江夏咖啡店A", "武汉市", "武汉市江夏区", district="江夏区"),
                _poi("江夏咖啡店B", "武汉市", "武汉市江夏区", district="江夏区"),
            ][:limit]

    service = POISearchService(
        amap_service=FakeAMapOtherDistrictOnlyService(),
        query_refiner=FakeRefiner(decision=None, enabled=False),
    )

    result = service.search_with_debug(
        query="找一找武汉洪山区咖啡店",
        city_hint="武汉市洪山区",
        limit=10,
        allow_city_switch=True,
    )

    assert result.items == []
    assert result.debug.final_city_hint == "武汉市洪山区"
    assert result.debug.city_switch_suggested is False
    assert result.debug.city_switch_applied is False
    assert result.debug.fallback_reason == "district_locked_no_result"


@pytest.mark.parametrize(("city", "source_district", "target_district"), RANDOM_SCOPE_CASES)
def test_random_district_switch_disallowed_returns_empty(
    city: str,
    source_district: str,
    target_district: str,
) -> None:
    class FakeAMapSwitchTargetOnlyService:
        def search(self, query: str, city_hint: str | None = None, limit: int = 10):
            assert city_hint == city
            _ = query
            return [
                _poi("目标咖啡店A", city, f"{city}{target_district}1号", district=target_district),
                _poi("目标咖啡店B", city, f"{city}{target_district}2号", district=target_district),
            ][:limit]

    service = POISearchService(
        amap_service=FakeAMapSwitchTargetOnlyService(),
        query_refiner=FakeRefiner(decision=None, enabled=False),
    )

    result = service.search_with_debug(
        query="咖啡店",
        city_hint=f"{city}{source_district}",
        limit=10,
        allow_city_switch=False,
    )

    assert result.items == []
    assert result.debug.city_switch_suggested is True
    assert result.debug.city_switch_applied is False
    assert result.debug.city_switch_from == f"{city}{source_district}"
    assert result.debug.city_switch_to == f"{city}{target_district}"
    assert result.debug.fallback_reason == "district_scope_no_result"


@pytest.mark.parametrize(("city", "source_district", "target_district"), RANDOM_SCOPE_CASES)
def test_random_district_switch_allowed_applies_target_scope(
    city: str,
    source_district: str,
    target_district: str,
) -> None:
    class FakeAMapSwitchTargetOnlyService:
        def search(self, query: str, city_hint: str | None = None, limit: int = 10):
            assert city_hint == city
            _ = query
            return [
                _poi("目标咖啡店A", city, f"{city}{target_district}1号", district=target_district),
                _poi("目标咖啡店B", city, f"{city}{target_district}2号", district=target_district),
            ][:limit]

    service = POISearchService(
        amap_service=FakeAMapSwitchTargetOnlyService(),
        query_refiner=FakeRefiner(decision=None, enabled=False),
    )

    result = service.search_with_debug(
        query="咖啡店",
        city_hint=f"{city}{source_district}",
        limit=10,
        allow_city_switch=True,
    )

    assert len(result.items) == 2
    assert all(item.district == target_district for item in result.items)
    assert result.debug.city_switch_suggested is True
    assert result.debug.city_switch_applied is True
    assert result.debug.city_switch_from == f"{city}{source_district}"
    assert result.debug.city_switch_to == f"{city}{target_district}"
    assert result.debug.final_city_hint == f"{city}{target_district}"
