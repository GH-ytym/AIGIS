"""POI search orchestration with AI-enhanced fuzzy retrieval."""

from __future__ import annotations

import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from aigis_agent.core.config import settings
from aigis_agent.schemas.poi import POIItem, POISearchDebug
from aigis_agent.services.amap_service import AMapProviderError, AMapService


class QueryRefineDecision(BaseModel):
    """Structured output for AI query refinement."""

    cleaned_query: str
    city_hint_override: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class StrongLandmarkDecision(BaseModel):
    """Structured output for strong landmark arbitration."""

    is_strong_landmark: bool = False
    city_hint_override: str | None = None
    cleaned_query: str = ""
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    reason: str = ""


class RefineCandidate(BaseModel):
    """Execution-ready retry candidate with optional city suggestion."""

    query: str
    city: str | None = None
    suggested_city: str | None = None


class POISearchExecution(BaseModel):
    """POI execution result with debug details for UI inspection."""

    items: list[POIItem] = Field(default_factory=list)
    debug: POISearchDebug


class POIQueryRefiner:
    """Use LLM to refine typo/pinyin/noisy query input for POI search."""

    def __init__(self) -> None:
        self._llm: ChatOpenAI | None = None
        if not settings.poi_ai_refine_enabled or not settings.openai_api_key:
            return

        self._llm = ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url or None,
            temperature=0.0,
            timeout=settings.poi_ai_timeout_s,
        )

    @property
    def enabled(self) -> bool:
        """Whether AI refinement is available in current runtime."""
        return self._llm is not None

    def refine(
        self,
        query: str,
        city_hint: str | None,
        trigger: str,
        result_count: int,
        top_items: list[POIItem],
    ) -> QueryRefineDecision | None:
        """Refine query with strict JSON output and safe fallback."""
        if self._llm is None:
            return None

        system_prompt = (
            "You are a Chinese GIS query refiner for POI fuzzy search. "
            "Only do: typo correction, pinyin-to-Chinese normalization, and optional city-level hint inference. "
            "Do not fabricate detailed addresses and do not output district/street suggestions. "
            "Infer only city-level hint when current retrieval looks off-topic or sparse. "
            "Return JSON only with keys: cleaned_query, city_hint_override, confidence, reason. "
            "If no refinement is needed, keep cleaned_query equal to input and city_hint_override as null."
        )
        payload = {
            "query": query,
            "city_hint": city_hint,
            "trigger": trigger,
            "result_count": result_count,
            "top_results": [
                {
                    "name": item.name,
                    "address": item.address,
                    "city": item.city,
                    "district": item.district,
                }
                for item in top_items[:3]
            ],
        }

        try:
            message = self._llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ]
            )
            text = self._extract_text(message)
            data = self._parse_json(text)
            if data is None:
                return None

            decision = QueryRefineDecision.model_validate(data)
            cleaned_query = str(decision.cleaned_query or "").strip()
            decision.cleaned_query = cleaned_query or query

            if decision.city_hint_override is not None:
                cleaned_city = str(decision.city_hint_override).strip()
                decision.city_hint_override = cleaned_city or None

            decision.reason = str(decision.reason or "").strip()
            return decision
        except Exception:
            return None

    def detect_strong_landmark(
        self,
        query: str,
        city_hint: str,
        local_items: list[POIItem],
    ) -> StrongLandmarkDecision | None:
        """Judge whether query is strong landmark and suggest city-level override."""
        if self._llm is None:
            return None

        system_prompt = (
            "You are a GIS landmark disambiguation judge. "
            "Decide whether query is a strong landmark that is city-anchored. "
            "Prefer recall over precision for city-switch suggestion: when query likely refers to a famous landmark, return is_strong_landmark=true. "
            "Even if current-city hits contain compound names (for example residential phase/parking/commercial aliases), you may still suggest switch. "
            "If yes, provide city-level hint only in city_hint_override. "
            "Never output full street address. "
            "Return JSON only with keys: is_strong_landmark, city_hint_override, cleaned_query, confidence, reason."
        )
        payload = {
            "query": query,
            "current_city": city_hint,
            "local_top_results": [
                {
                    "name": item.name,
                    "address": item.address,
                    "city": item.city,
                    "district": item.district,
                }
                for item in local_items[:5]
            ],
        }

        try:
            message = self._llm.invoke(
                [
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ]
            )
            text = self._extract_text(message)
            data = self._parse_json(text)
            if data is None:
                return None

            decision = StrongLandmarkDecision.model_validate(data)
            decision.cleaned_query = str(decision.cleaned_query or "").strip() or query
            if decision.city_hint_override is not None:
                cleaned_city = str(decision.city_hint_override).strip()
                decision.city_hint_override = cleaned_city or None
            decision.reason = str(decision.reason or "").strip()
            decision.is_strong_landmark = bool(decision.is_strong_landmark)
            return decision
        except Exception:
            return None

    @staticmethod
    def _extract_text(message: Any) -> str:
        """Extract plain text from langchain message content."""
        content = getattr(message, "content", "")
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
                elif isinstance(item, str):
                    chunks.append(item)
            return "\n".join(chunks)

        return str(content)

    @staticmethod
    def _parse_json(text: str) -> dict[str, Any] | None:
        """Parse JSON object from raw model output text."""
        stripped = str(text or "").strip()
        if not stripped:
            return None

        fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", stripped, flags=re.IGNORECASE)
        candidate = fenced.group(1) if fenced else stripped

        try:
            data = json.loads(candidate)
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            pass

        start = candidate.find("{")
        end = candidate.rfind("}")
        if start < 0 or end < 0 or end <= start:
            return None

        try:
            data = json.loads(candidate[start : end + 1])
            return data if isinstance(data, dict) else None
        except json.JSONDecodeError:
            return None


class POISearchService:
    """Orchestrate deterministic cleanup, AMap search and optional AI retry."""

    def __init__(
        self,
        amap_service: AMapService | None = None,
        query_refiner: POIQueryRefiner | None = None,
    ) -> None:
        self._amap_service = amap_service or AMapService()
        self._query_refiner = query_refiner or POIQueryRefiner()

    def search(
        self,
        query: str,
        city_hint: str | None = None,
        limit: int = 10,
        allow_city_switch: bool = True,
    ) -> list[POIItem]:
        """Search POI with optional AI-assisted input correction."""
        return self.search_with_debug(
            query=query,
            city_hint=city_hint,
            limit=limit,
            allow_city_switch=allow_city_switch,
        ).items

    def search_with_debug(
        self,
        query: str,
        city_hint: str | None = None,
        limit: int = 10,
        allow_city_switch: bool = True,
    ) -> POISearchExecution:
        """Search POI and return debug details for refinement decisions."""
        raw_query = str(query or "")
        normalized_query = self._normalize_query(raw_query)
        normalized_city = self._normalize_city(city_hint)
        safe_limit = max(1, min(int(limit), 50))

        debug = POISearchDebug(
            original_query=raw_query,
            normalized_query=normalized_query,
            final_query=normalized_query,
            original_city_hint=city_hint,
            final_city_hint=normalized_city,
        )

        if not normalized_query:
            return POISearchExecution(items=[], debug=debug)

        base_items = self._amap_service.search(
            query=normalized_query,
            city_hint=normalized_city,
            limit=safe_limit,
        )

        base_items, debug, should_return = self._apply_city_scope_guard(
            city_hint=normalized_city,
            items=base_items,
            debug=debug,
            allow_city_switch=allow_city_switch,
        )

        final_items = list(base_items)
        debug.final_query = normalized_query
        debug.final_city_hint = normalized_city

        if should_return:
            return POISearchExecution(items=final_items, debug=debug)

        should_retry, retry_trigger = self._should_retry_with_ai(
            normalized_query,
            normalized_city,
            base_items,
        )
        if not should_retry or retry_trigger is None:
            final_items, debug = self._apply_strong_landmark_guard(
                query=normalized_query,
                city_hint=normalized_city,
                base_items=base_items,
                current_items=final_items,
                debug=debug,
                limit=safe_limit,
                allow_city_switch=allow_city_switch,
            )
            return POISearchExecution(items=final_items, debug=debug)

        debug.ai_refine_triggered = True
        debug.trigger_reason = retry_trigger

        candidate, ai_confidence, candidate_reason = self._get_ai_candidate(
            query=normalized_query,
            city_hint=normalized_city,
            trigger=retry_trigger,
            base_items=base_items,
        )
        debug.ai_confidence = ai_confidence

        if candidate is None:
            debug.fallback_reason = candidate_reason or "ai_no_candidate"
            final_items, debug = self._apply_strong_landmark_guard(
                query=normalized_query,
                city_hint=normalized_city,
                base_items=base_items,
                current_items=final_items,
                debug=debug,
                limit=safe_limit,
                allow_city_switch=allow_city_switch,
            )
            return POISearchExecution(items=final_items, debug=debug)

        retry_query = candidate.query
        retry_city = candidate.city
        suggested_city = candidate.suggested_city

        if suggested_city and normalized_city and self._city_key(suggested_city) != self._city_key(normalized_city):
            debug.city_switch_suggested = True
            debug.city_switch_from = normalized_city
            debug.city_switch_to = suggested_city

        try:
            retry_items = self._amap_service.search(
                query=retry_query,
                city_hint=retry_city,
                limit=safe_limit,
            )
        except AMapProviderError:
            debug.corrected_query = retry_query
            debug.corrected_city_hint = suggested_city or retry_city
            debug.fallback_reason = "ai_retry_provider_failed"
            final_items, debug = self._apply_strong_landmark_guard(
                query=normalized_query,
                city_hint=normalized_city,
                base_items=base_items,
                current_items=final_items,
                debug=debug,
                limit=safe_limit,
                allow_city_switch=allow_city_switch,
            )
            return POISearchExecution(items=final_items, debug=debug)

        debug.corrected_query = retry_query
        debug.corrected_city_hint = suggested_city or retry_city
        debug.corrected_items = list(retry_items)
        debug.corrected_result_count = len(retry_items)

        city_switched = bool(
            normalized_city
            and retry_city
            and self._city_key(retry_city) != self._city_key(normalized_city)
        )
        if city_switched:
            debug.city_switch_suggested = True
            debug.city_switch_from = normalized_city
            debug.city_switch_to = retry_city

        can_apply = allow_city_switch or not city_switched
        if can_apply and self._prefer_retry(base_items, retry_items, normalized_query, retry_query):
            final_items = retry_items
            debug.ai_refine_applied = True
            debug.final_query = retry_query
            debug.final_city_hint = retry_city
            if city_switched and allow_city_switch:
                debug.city_switch_applied = True
        else:
            debug.fallback_reason = "ai_applied_no_gain"

        final_items, debug = self._apply_strong_landmark_guard(
            query=normalized_query,
            city_hint=normalized_city,
            base_items=base_items,
            current_items=final_items,
            debug=debug,
            limit=safe_limit,
            allow_city_switch=allow_city_switch,
        )

        return POISearchExecution(items=final_items, debug=debug)

    def _apply_strong_landmark_guard(
        self,
        query: str,
        city_hint: str | None,
        base_items: list[POIItem],
        current_items: list[POIItem],
        debug: POISearchDebug,
        limit: int,
        allow_city_switch: bool,
    ) -> tuple[list[POIItem], POISearchDebug]:
        """Second defense: AI strong-landmark arbitration for city-switch suggestion."""
        if not city_hint:
            return current_items, debug
        if not settings.poi_ai_strong_landmark_enabled or not self._query_refiner.enabled:
            return current_items, debug
        if debug.city_switch_suggested or debug.city_switch_applied:
            return current_items, debug

        query_key = self._text_key(query)
        min_len = max(1, int(settings.poi_ai_strong_landmark_min_query_len))
        max_len = max(min_len, int(settings.poi_ai_strong_landmark_max_query_len))
        if len(query_key) < min_len or len(query_key) > max_len:
            return current_items, debug

        decision = self._query_refiner.detect_strong_landmark(
            query=query,
            city_hint=city_hint,
            local_items=base_items,
        )
        if decision is None:
            return current_items, debug

        debug.ai_refine_triggered = True

        debug.ai_confidence = max(float(debug.ai_confidence or 0.0), float(decision.confidence))

        suggest_threshold = max(0.0, min(1.0, settings.poi_ai_strong_landmark_suggest_confidence_threshold))
        apply_threshold = max(
            suggest_threshold,
            max(0.0, min(1.0, settings.poi_ai_strong_landmark_confidence_threshold)),
        )

        if decision.confidence < suggest_threshold:
            return current_items, debug
        if not decision.is_strong_landmark:
            return current_items, debug

        suggested_city = self._normalize_city(decision.city_hint_override)
        if not suggested_city:
            return current_items, debug
        if self._city_key(suggested_city) == self._city_key(city_hint):
            return current_items, debug

        refined_query = self._normalize_query(decision.cleaned_query) or query

        debug.city_switch_suggested = True
        debug.city_switch_from = city_hint
        debug.city_switch_to = suggested_city
        debug.corrected_query = refined_query
        debug.corrected_city_hint = suggested_city
        if debug.trigger_reason:
            debug.trigger_reason = f"{debug.trigger_reason};strong_landmark_guard"
        else:
            debug.trigger_reason = "strong_landmark_guard"

        if not allow_city_switch:
            if not debug.fallback_reason:
                debug.fallback_reason = "strong_landmark_suggested"
            return current_items, debug

        if decision.confidence < apply_threshold:
            if not debug.fallback_reason:
                debug.fallback_reason = "strong_landmark_suggested_not_applied"
            return current_items, debug

        try:
            switched_items = self._amap_service.search(
                query=refined_query,
                city_hint=suggested_city,
                limit=limit,
            )
        except AMapProviderError:
            debug.fallback_reason = "strong_landmark_retry_provider_failed"
            return current_items, debug

        debug.corrected_items = list(switched_items)
        debug.corrected_result_count = len(switched_items)

        has_target_full_match = self._has_full_query_name_match(refined_query, switched_items)
        if switched_items and (has_target_full_match or self._prefer_retry(base_items, switched_items, query, refined_query)):
            debug.ai_refine_applied = True
            debug.city_switch_applied = True
            debug.final_query = refined_query
            debug.final_city_hint = suggested_city
            return switched_items, debug

        if not debug.fallback_reason:
            debug.fallback_reason = "strong_landmark_no_gain"
        return current_items, debug

    def _get_ai_candidate(
        self,
        query: str,
        city_hint: str | None,
        trigger: str,
        base_items: list[POIItem],
    ) -> tuple[RefineCandidate | None, float | None, str | None]:
        """Return refined query/city and confidence if candidate is worth retrying."""
        decision = self._query_refiner.refine(
            query=query,
            city_hint=city_hint,
            trigger=trigger,
            result_count=len(base_items),
            top_items=base_items,
        )
        if decision is None:
            return None, None, "ai_unavailable_or_parse_failed"

        confidence = float(decision.confidence)
        if confidence < settings.poi_ai_confidence_threshold:
            return None, confidence, "ai_low_confidence"

        refined_query = self._normalize_query(decision.cleaned_query)
        if not refined_query:
            return None, confidence, "ai_empty_query"

        suggested_city = self._normalize_city(decision.city_hint_override)
        if suggested_city and city_hint and self._city_key(suggested_city) == self._city_key(city_hint):
            suggested_city = None

        refined_city = suggested_city or city_hint

        if refined_query == query and refined_city == city_hint and suggested_city is None:
            return None, confidence, "ai_no_effect"

        return (
            RefineCandidate(
                query=refined_query,
                city=refined_city,
                suggested_city=suggested_city,
            ),
            confidence,
            None,
        )

    def _should_retry_with_ai(
        self,
        query: str,
        city_hint: str | None,
        items: list[POIItem],
    ) -> tuple[bool, str | None]:
        """Decide whether second-pass AI correction is worth trying."""
        if not settings.poi_ai_refine_enabled or not self._query_refiner.enabled:
            return False, None

        if city_hint and not self._has_full_query_name_match(query, items):
            return True, "name_miss_query_substring"

        if settings.poi_ai_retry_zero_result and not items:
            return True, "zero_result"

        if self._is_alpha_only(query) and len(items) < settings.poi_ai_alpha_result_threshold:
            return True, "alpha_input_low_result"

        if len(items) < settings.poi_ai_sparse_result_threshold and not self._has_full_query_name_match(query, items):
            return True, "sparse_result_no_full_match"

        return False, None

    def _prefer_retry(
        self,
        base_items: list[POIItem],
        retry_items: list[POIItem],
        base_query: str,
        retry_query: str,
    ) -> bool:
        """Use retry only when quality score is meaningfully better."""
        if not retry_items:
            return False
        if not base_items:
            return True

        base_has_full_match = self._has_full_query_name_match(base_query, base_items)
        retry_has_full_match = self._has_full_query_name_match(retry_query, retry_items)
        if retry_has_full_match and not base_has_full_match:
            return True

        base_score = self._result_score(base_items, base_query)
        retry_score = self._result_score(retry_items, retry_query)
        return retry_score > base_score + settings.poi_ai_result_gain_threshold

    def _result_score(self, items: list[POIItem], query: str) -> float:
        """Simple score combining top-1 relevance and recall count."""
        if not items:
            return 0.0

        top1 = items[0]
        top_text = f"{top1.name} {top1.address}"
        relevance = self._similarity(query, top_text)
        recall = min(len(items), 5) / 5.0
        return relevance * 0.7 + recall * 0.3

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Apply deterministic cleanup before any provider call."""
        text = unicodedata.normalize("NFKC", str(query or ""))
        text = text.strip()
        text = re.sub(r"\s+", "", text)
        text = text[: settings.poi_ai_query_max_length]
        text = text.strip("()[]{}<>【】（）'\"“”‘’`")
        text = re.sub(r"[!！?？,，。．;；:：]+$", "", text)
        return text

    @staticmethod
    def _normalize_city(city_hint: str | None) -> str | None:
        """Normalize city hint while preserving user-intended city label."""
        if city_hint is None:
            return None

        city = unicodedata.normalize("NFKC", str(city_hint)).strip()
        city = re.sub(r"\s+", "", city)
        city = re.sub(r"(\(直辖\)|（直辖）)", "", city)
        if not city:
            return None

        city = re.sub(r"(?:市)?城区$", "", city)
        if not city:
            return None

        if city in {"北京", "上海", "天津", "重庆"}:
            return f"{city}市"

        if re.search(r"(市|自治州|地区|盟|特别行政区|自治区|省|县|区|旗)$", city):
            return city

        return f"{city}市"

    def _apply_city_scope_guard(
        self,
        city_hint: str | None,
        items: list[POIItem],
        debug: POISearchDebug,
        allow_city_switch: bool,
    ) -> tuple[list[POIItem], POISearchDebug, bool]:
        """Guard against provider cross-city leakage when city hint is provided."""
        if not city_hint or not items:
            return items, debug, False

        top_city = self._normalize_city(items[0].city or items[0].province)
        if not top_city:
            return items, debug, False
        if self._city_key(top_city) == self._city_key(city_hint):
            return items, debug, False

        debug.city_switch_suggested = True
        debug.city_switch_from = city_hint
        debug.city_switch_to = top_city
        if not debug.trigger_reason:
            debug.trigger_reason = "top_result_city_mismatch"

        if allow_city_switch:
            return items, debug, False

        local_items = self._filter_items_by_city(items, city_hint)
        if local_items:
            if not debug.fallback_reason:
                debug.fallback_reason = "city_scope_filtered_cross_city_results"
            return local_items, debug, False

        if not debug.fallback_reason:
            debug.fallback_reason = "city_scope_no_local_result"
        return [], debug, True

    def _filter_items_by_city(self, items: list[POIItem], city_hint: str) -> list[POIItem]:
        """Keep only items that belong to current city key."""
        target_key = self._city_key(city_hint)
        local_items: list[POIItem] = []
        for item in items:
            item_city = self._normalize_city(item.city or item.province)
            if not item_city:
                continue
            if self._city_key(item_city) == target_key:
                local_items.append(item)
        return local_items

    @staticmethod
    def _city_key(city: str) -> str:
        """Canonical city key for conflict comparison."""
        norm = unicodedata.normalize("NFKC", city).strip()
        norm = re.sub(r"\s+", "", norm)
        norm = re.sub(r"(?:市)?城区$", "", norm)
        return re.sub(r"(省|市|自治区|特别行政区|地区|盟|州)$", "", norm)

    @staticmethod
    def _is_alpha_only(query: str) -> bool:
        """Check if query is pure Latin letters (common pinyin input case)."""
        lowered = query.lower()
        return bool(re.fullmatch(r"[a-z]+", lowered))

    @staticmethod
    def _similarity(left: str, right: str) -> float:
        """Character-level similarity used by low-cost relevance checks."""
        left_key = POISearchService._text_key(left)
        right_key = POISearchService._text_key(right)
        if not left_key or not right_key:
            return 0.0
        return SequenceMatcher(None, left_key, right_key).ratio()

    @staticmethod
    def _text_key(text: str) -> str:
        """Normalize text for rough similarity scoring."""
        norm = unicodedata.normalize("NFKC", str(text or "")).lower()
        return re.sub(r"[^0-9a-z\u4e00-\u9fff]+", "", norm)

    @staticmethod
    def _query_in_name(query: str, name: str) -> bool:
        """Whether query appears as full normalized substring in name."""
        query_key = POISearchService._text_key(query)
        name_key = POISearchService._text_key(name)
        return bool(query_key and query_key in name_key)

    @staticmethod
    def _has_full_query_name_match(query: str, items: list[POIItem]) -> bool:
        """Whether any result name contains the full normalized query substring."""
        for item in items:
            if POISearchService._query_in_name(query, item.name):
                return True

        return False
