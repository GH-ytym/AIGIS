"""POI search orchestration with AI-enhanced fuzzy retrieval."""

from __future__ import annotations

import json
import re
import unicodedata
from dataclasses import dataclass
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


@dataclass(frozen=True)
class AdminScope:
    """Normalized administrative scope split into city and district."""

    city: str | None = None
    district: str | None = None

    @property
    def has_value(self) -> bool:
        """Whether any scope constraint exists."""
        return bool(self.city or self.district)

    @property
    def search_hint(self) -> str | None:
        """Provider hint used in upstream POI API request."""
        return self.city or self.district

    @property
    def label(self) -> str | None:
        """Human-readable scope label used by debug payload."""
        if self.city and self.district:
            return f"{self.city}{self.district}"
        return self.city or self.district


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
        hint_scope = self._parse_admin_scope(city_hint)
        query_scope = self._extract_scope_from_query(raw_query)
        scope = query_scope if query_scope.has_value else hint_scope
        prepared_query = self._prepare_query_text(raw_query, scope)
        normalized_query = self._normalize_query(prepared_query)
        normalized_city = scope.label
        district_locked = self._is_district_locked(raw_query, scope)
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
            city_hint=scope.search_hint,
            limit=safe_limit,
        )

        base_items, debug, should_return = self._apply_city_scope_guard(
            scope=scope,
            items=base_items,
            debug=debug,
            allow_city_switch=allow_city_switch,
            district_locked=district_locked,
        )

        final_items = list(base_items)
        debug.final_query = normalized_query
        if not debug.city_switch_applied:
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
                district_locked=district_locked,
            )
            return POISearchExecution(items=final_items, debug=debug)

        debug.ai_refine_triggered = True
        debug.trigger_reason = retry_trigger

        candidate, ai_confidence, candidate_reason = self._get_ai_candidate(
            query=normalized_query,
            city_hint=normalized_city,
            trigger=retry_trigger,
            base_items=base_items,
            district_locked=district_locked,
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
                district_locked=district_locked,
            )
            return POISearchExecution(items=final_items, debug=debug)

        retry_query = candidate.query
        retry_city = candidate.city
        suggested_city = candidate.suggested_city
        retry_scope = self._parse_admin_scope(retry_city)
        retry_scope_label = retry_scope.label or retry_city
        retry_search_hint = retry_scope.search_hint

        if suggested_city and normalized_city and self._is_scope_changed(normalized_city, suggested_city):
            debug.city_switch_suggested = True
            debug.city_switch_from = normalized_city
            debug.city_switch_to = suggested_city

        try:
            retry_items = self._amap_service.search(
                query=retry_query,
                city_hint=retry_search_hint,
                limit=safe_limit,
            )
        except AMapProviderError:
            debug.corrected_query = retry_query
            debug.corrected_city_hint = suggested_city or retry_scope_label
            debug.fallback_reason = "ai_retry_provider_failed"
            final_items, debug = self._apply_strong_landmark_guard(
                query=normalized_query,
                city_hint=normalized_city,
                base_items=base_items,
                current_items=final_items,
                debug=debug,
                limit=safe_limit,
                allow_city_switch=allow_city_switch,
                district_locked=district_locked,
            )
            return POISearchExecution(items=final_items, debug=debug)

        if retry_scope.has_value:
            retry_items = self._filter_items_by_scope(retry_items, retry_scope)

        debug.corrected_query = retry_query
        debug.corrected_city_hint = suggested_city or retry_scope_label
        debug.corrected_items = list(retry_items)
        debug.corrected_result_count = len(retry_items)

        city_switched = bool(
            normalized_city
            and retry_scope_label
            and self._is_scope_changed(normalized_city, retry_scope_label)
        )
        if city_switched:
            debug.city_switch_suggested = True
            debug.city_switch_from = normalized_city
            debug.city_switch_to = retry_scope_label

        can_apply = allow_city_switch or not city_switched
        if can_apply and self._prefer_retry(base_items, retry_items, normalized_query, retry_query):
            final_items = retry_items
            debug.ai_refine_applied = True
            debug.final_query = retry_query
            debug.final_city_hint = retry_scope_label
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
            district_locked=district_locked,
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
        district_locked: bool,
    ) -> tuple[list[POIItem], POISearchDebug]:
        """Second defense: AI strong-landmark arbitration for city-switch suggestion."""
        if not city_hint:
            return current_items, debug
        if district_locked:
            return current_items, debug
        source_scope = self._parse_admin_scope(city_hint)
        if source_scope.city:
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
        if not self._is_scope_changed(city_hint, suggested_city):
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
            switched_scope = self._parse_admin_scope(suggested_city)
            switched_items = self._amap_service.search(
                query=refined_query,
                city_hint=switched_scope.search_hint,
                limit=limit,
            )
        except AMapProviderError:
            debug.fallback_reason = "strong_landmark_retry_provider_failed"
            return current_items, debug

        if switched_scope.has_value:
            switched_items = self._filter_items_by_scope(switched_items, switched_scope)

        debug.corrected_items = list(switched_items)
        debug.corrected_result_count = len(switched_items)

        has_target_full_match = self._has_full_query_name_match(refined_query, switched_items)
        if switched_items and (has_target_full_match or self._prefer_retry(base_items, switched_items, query, refined_query)):
            debug.ai_refine_applied = True
            debug.city_switch_applied = True
            debug.final_query = refined_query
            debug.final_city_hint = switched_scope.label or suggested_city
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
        district_locked: bool,
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
        if district_locked:
            suggested_city = None
        elif suggested_city and city_hint and self._is_cross_city_switch(city_hint, suggested_city):
            suggested_city = None
        elif suggested_city and city_hint and not self._is_scope_changed(city_hint, suggested_city):
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
        """Normalize input hint into canonical city/district label."""
        return POISearchService._parse_admin_scope(city_hint).label

    @staticmethod
    def _normalize_admin_text(text: str | None) -> str:
        """Shared cleanup for administrative text."""
        norm = unicodedata.normalize("NFKC", str(text or "")).strip()
        norm = re.sub(r"\s+", "", norm)
        norm = re.sub(r"(\(直辖\)|（直辖）)", "", norm)
        return norm

    @staticmethod
    def _normalize_city_name(city_hint: str | None) -> str | None:
        """Normalize city/province-level text to canonical label."""
        city = POISearchService._normalize_admin_text(city_hint)
        if not city:
            return None

        city = re.sub(r"(?:市)?城区$", "", city)
        if not city:
            return None

        if city in {"北京", "上海", "天津", "重庆"}:
            return f"{city}市"

        if re.search(r"(市|自治州|地区|盟|特别行政区|自治区|省)$", city):
            return city

        return f"{city}市"

    @staticmethod
    def _normalize_district(district_hint: str | None) -> str | None:
        """Normalize district/county-level text to canonical label."""
        district = POISearchService._normalize_admin_text(district_hint)
        return district or None

    @staticmethod
    def _parse_admin_scope(city_hint: str | None) -> AdminScope:
        """Split mixed hint into city and district parts."""
        text = POISearchService._normalize_admin_text(city_hint)
        if not text:
            return AdminScope()

        if re.fullmatch(r"(?:北京|上海|天津|重庆)(?:市)?城区", text):
            return AdminScope(city=POISearchService._normalize_city_name(text))

        explicit = re.match(
            r"^(?P<city>.+?(?:市|自治州|地区|盟|特别行政区|自治区|省))(?P<district>.+?(?:自治县|自治旗|特区|林区|区|县|旗))$",
            text,
        )
        if explicit:
            return AdminScope(
                city=POISearchService._normalize_city_name(explicit.group("city")),
                district=POISearchService._normalize_district(explicit.group("district")),
            )

        loose = re.match(
            r"^(?P<city>[\u4e00-\u9fff]{2,8}?)(?P<district>[\u4e00-\u9fff]{2,12}(?:自治县|自治旗|特区|林区|区|县|旗))$",
            text,
        )
        if loose:
            city_part = loose.group("city")
            district_part = loose.group("district")
            if len(city_part) >= 2 and len(district_part) >= 2:
                return AdminScope(
                    city=POISearchService._normalize_city_name(city_part),
                    district=POISearchService._normalize_district(district_part),
                )

        if re.search(r"(自治县|自治旗|特区|林区|区|县|旗)$", text):
            return AdminScope(district=POISearchService._normalize_district(text))

        return AdminScope(city=POISearchService._normalize_city_name(text))

    def _apply_city_scope_guard(
        self,
        scope: AdminScope,
        items: list[POIItem],
        debug: POISearchDebug,
        allow_city_switch: bool,
        district_locked: bool,
    ) -> tuple[list[POIItem], POISearchDebug, bool]:
        """Guard against provider scope leakage with district-first strategy."""
        if not scope.has_value or not items:
            return items, debug, False

        in_scope_items = self._filter_items_by_scope(items, scope)
        if in_scope_items:
            return in_scope_items, debug, False

        if district_locked:
            if not debug.fallback_reason:
                debug.fallback_reason = "district_locked_no_result"
            return [], debug, True

        if scope.city:
            same_city_items = self._filter_items_by_scope(items, AdminScope(city=scope.city))
            if same_city_items:
                best_same_city_scope = self._scope_from_item(same_city_items[0])
                target_label = best_same_city_scope.label or scope.city

                debug.city_switch_suggested = True
                debug.city_switch_from = scope.label
                debug.city_switch_to = target_label
                if not debug.trigger_reason:
                    debug.trigger_reason = "top_result_district_mismatch"

                if allow_city_switch and scope.district and best_same_city_scope.district:
                    debug.city_switch_applied = True
                    debug.final_city_hint = target_label
                    return same_city_items, debug, False

                if not debug.fallback_reason:
                    debug.fallback_reason = "district_scope_no_result"
                return [], debug, True

            if not debug.fallback_reason:
                debug.fallback_reason = "city_scope_no_local_result"
            return [], debug, True

        top_scope = self._scope_from_item(items[0])
        if not top_scope.has_value:
            return items, debug, False
        if self._scope_matches(scope, top_scope):
            return items, debug, False

        debug.city_switch_suggested = True
        debug.city_switch_from = scope.label
        debug.city_switch_to = top_scope.label
        if not debug.trigger_reason:
            debug.trigger_reason = "top_result_city_mismatch"

        if allow_city_switch:
            return items, debug, False

        if not debug.fallback_reason:
            debug.fallback_reason = "city_scope_no_local_result"
        return [], debug, True

    @staticmethod
    def _extract_scope_from_query(query: str) -> AdminScope:
        """Extract explicit city/district scope from natural-language query."""
        text = POISearchService._normalize_admin_text(query)
        if not text:
            return AdminScope()

        text = re.sub(
            r"^(?:帮我|麻烦帮我|麻烦|请|请帮我|给我|找一找|找一下|找下|查一下|查下|搜一下|搜下|看下|看看|帮忙|帮我在|在)+",
            "",
            text,
        )
        if not text:
            return AdminScope()

        city_district_prefix = re.match(
            r"^(?P<scope>[\u4e00-\u9fff]{2,14}(?:自治县|自治旗|特区|林区|区|县|旗))",
            text,
        )
        if city_district_prefix:
            candidate = POISearchService._parse_admin_scope(city_district_prefix.group("scope"))
            if candidate.has_value:
                return candidate

        city_prefix = re.match(
            r"^(?P<scope>[\u4e00-\u9fff]{2,12}(?:市|自治州|地区|盟|特别行政区|自治区|省))",
            text,
        )
        if city_prefix:
            candidate = POISearchService._parse_admin_scope(city_prefix.group("scope"))
            if candidate.has_value:
                return candidate

        return AdminScope()

    @staticmethod
    def _prepare_query_text(query: str, scope: AdminScope) -> str:
        """Remove politeness/location wrappers and keep POI keyword core."""
        text = unicodedata.normalize("NFKC", str(query or "")).strip()
        if not text:
            return ""

        text = re.sub(r"\s+", "", text)
        text = re.sub(
            r"^(?:帮我|麻烦帮我|麻烦|请|请帮我|给我|找一找|找一下|找下|查一下|查下|搜一下|搜下|看下|看看|帮忙|帮我在|在)+",
            "",
            text,
        )

        if scope.label:
            text = text.replace(scope.label, "")
        if scope.city:
            text = text.replace(scope.city, "")
            city_alias = re.sub(r"(市|自治州|地区|盟|特别行政区|自治区|省)$", "", scope.city)
            if city_alias:
                text = text.replace(city_alias, "")
        if scope.district:
            text = text.replace(scope.district, "")
            district_alias = re.sub(r"(自治县|自治旗|特区|林区|区|县|旗)$", "", scope.district)
            if district_alias:
                text = text.replace(district_alias, "")

        text = re.sub(r"^(?:找|查|搜|看|去|到)+", "", text)
        text = re.sub(r"(?:一下|一哈|一下下|哈|呗|呀|吧|嘛|呢)+$", "", text)
        return text or query

    @staticmethod
    def _is_district_locked(query: str, scope: AdminScope) -> bool:
        """Whether query explicitly anchors target district and should avoid auto-switch."""
        if not scope.district:
            return False
        query_key = POISearchService._text_key(query)
        district_key = POISearchService._district_key(scope.district)
        return bool(query_key and district_key and district_key in query_key)

    @staticmethod
    def _is_cross_city_switch(source_hint: str | None, target_hint: str | None) -> bool:
        """Whether scope change crosses city boundary."""
        source = POISearchService._parse_admin_scope(source_hint)
        target = POISearchService._parse_admin_scope(target_hint)
        source_city = POISearchService._city_key(source.city or "") if source.city else ""
        target_city = POISearchService._city_key(target.city or "") if target.city else ""
        return bool(source_city and target_city and source_city != target_city)

    def _filter_items_by_scope(self, items: list[POIItem], scope: AdminScope) -> list[POIItem]:
        """Keep only items that belong to current city/district scope."""
        local_items: list[POIItem] = []
        for item in items:
            if self._scope_matches(scope, self._scope_from_item(item)):
                local_items.append(item)
        return local_items

    def _scope_from_item(self, item: POIItem) -> AdminScope:
        """Build normalized scope from provider item fields."""
        return AdminScope(
            city=self._normalize_city_name(item.city or item.province),
            district=self._normalize_district(item.district),
        )

    def _scope_matches(self, expected: AdminScope, actual: AdminScope) -> bool:
        """Whether provider item scope satisfies expected city/district hint."""
        if not expected.has_value:
            return True

        expected_city_key = self._city_key(expected.city or "") if expected.city else ""
        actual_city_key = self._city_key(actual.city or "") if actual.city else ""
        if expected_city_key:
            if not actual_city_key or expected_city_key != actual_city_key:
                return False

        expected_district_key = self._district_key(expected.district)
        if expected_district_key:
            actual_district_key = self._district_key(actual.district)
            if not actual_district_key or expected_district_key != actual_district_key:
                return False

        if not expected_city_key and expected_district_key:
            actual_district_key = self._district_key(actual.district)
            return bool(actual_district_key and actual_district_key == expected_district_key)

        return bool(expected_city_key or expected_district_key)

    @staticmethod
    def _is_scope_changed(source_hint: str | None, target_hint: str | None) -> bool:
        """Check whether two hints point to different city/district scopes."""
        source = POISearchService._parse_admin_scope(source_hint)
        target = POISearchService._parse_admin_scope(target_hint)

        source_city = POISearchService._city_key(source.city or "") if source.city else ""
        target_city = POISearchService._city_key(target.city or "") if target.city else ""
        if source_city and target_city and source_city != target_city:
            return True

        source_district = POISearchService._district_key(source.district)
        target_district = POISearchService._district_key(target.district)
        if source_district and target_district and source_district != target_district:
            return True

        if not source_city and not target_city and source_district and target_district:
            return source_district != target_district

        return False

    @staticmethod
    def _city_key(city: str) -> str:
        """Canonical city key for conflict comparison."""
        scope = POISearchService._parse_admin_scope(city)
        base = scope.city or city
        norm = unicodedata.normalize("NFKC", base).strip()
        norm = re.sub(r"\s+", "", norm)
        norm = re.sub(r"(?:市)?城区$", "", norm)
        return re.sub(r"(省|市|自治区|特别行政区|地区|盟|自治州)$", "", norm)

    @staticmethod
    def _district_key(district: str | None) -> str:
        """Canonical district key for district-level scope matching."""
        norm = POISearchService._normalize_district(district) or ""
        if not norm:
            return ""
        norm = re.sub(r"\s+", "", norm)
        return re.sub(r"(自治县|自治旗|特区|林区|区|县|旗)$", "", norm)

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
