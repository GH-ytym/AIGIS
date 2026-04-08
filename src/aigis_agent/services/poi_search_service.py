"""POI search orchestration with optional AI query refinement."""

from __future__ import annotations

from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
import json
import re
import unicodedata
from difflib import SequenceMatcher
from typing import Any

from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from aigis_agent.core.config import settings
from aigis_agent.schemas.poi import POIItem, POISearchDebug
from aigis_agent.services.amap_service import AMapService

MUNICIPALITY_BASES: set[str] = {"北京", "上海", "天津", "重庆"}

TYPO_RISK_TOKENS: tuple[str, ...] = (
    "咖非",
    "星巴客",
    "医阮",
    "地铁占",
    "附进",
    "俯近",
)


class QueryRefineDecision(BaseModel):
    """Structured output for query refinement."""

    cleaned_query: str
    city_hint_override: str | None = None
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    is_strong_landmark: bool = False
    force_city_switch: bool = False
    reason: str = ""


class RefineCandidate(BaseModel):
    """Execution-ready retry candidate and optional city-switch suggestion."""

    query: str
    city: str | None = None
    suggested_city: str | None = None


class POISearchExecution(BaseModel):
    """POI execution result with debug details for UI inspection."""

    items: list[POIItem] = Field(default_factory=list)
    debug: POISearchDebug


class POIQueryRefiner:
    """Use LLM to refine typo/pinyin/city-conflict query inputs."""

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
            "You are a GIS keyword normalizer for AMap search. "
            "Only handle these cases: typo correction, pinyin-to-Chinese conversion, "
            "and city hint correction when current-city results are sparse. "
            "When trigger indicates name mismatch in current city, "
            "you may infer and return the most likely city-level hint for well-known landmarks/scenic spots. "
            "Do not return detailed addresses, only city-level hint in city_hint_override. "
            "Do not invent specific addresses. "
            "Return JSON only with keys: cleaned_query, city_hint_override, confidence, reason. "
            "If no change is needed, keep cleaned_query same as input and city_hint_override as null."
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

            return decision
        except Exception:
            return None

    def detect_strong_landmark(
        self,
        query: str,
        city_hint: str,
        local_items: list[POIItem],
        candidate_city: str,
        candidate_score: float,
        candidate_items: list[POIItem],
    ) -> QueryRefineDecision | None:
        """Decide whether query is a strong landmark that should force city switch."""
        if self._llm is None:
            return None

        system_prompt = (
            "You are a GIS landmark disambiguation engine. "
            "Determine whether the query is a strong landmark that should force city switch. "
            "Strong landmark means iconic and city-anchored POI (example: 东方明珠 -> 上海). "
            "If yes, set is_strong_landmark=true, force_city_switch=true, and return city_hint_override. "
            "Use only city-level hint, never detailed addresses. "
            "Return JSON only with keys: cleaned_query, city_hint_override, confidence, "
            "is_strong_landmark, force_city_switch, reason."
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
            "candidate_city": candidate_city,
            "candidate_city_score": candidate_score,
            "candidate_city_top_results": [
                {
                    "name": item.name,
                    "address": item.address,
                    "city": item.city,
                    "district": item.district,
                }
                for item in candidate_items[:5]
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

            decision.is_strong_landmark = bool(decision.is_strong_landmark)
            decision.force_city_switch = bool(decision.force_city_switch)
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
        stripped = text.strip()
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
        normalized_query = self._normalize_query(query)
        normalized_city = self._normalize_city(city_hint)
        debug = POISearchDebug(
            original_query=raw_query,
            normalized_query=normalized_query,
            final_query=normalized_query,
            original_city_hint=city_hint,
            final_city_hint=normalized_city,
        )

        if not normalized_query:
            return POISearchExecution(items=[], debug=debug)

        active_query = normalized_query
        active_city = normalized_city

        if normalized_city and settings.poi_parallel_recall_enabled:
            local_items, global_items = self._search_parallel_local_global(
                query=active_query,
                city_hint=active_city,
                limit=limit,
            )
            base_items = local_items
            final_items = local_items

            local_score = self._city_relevance_score(active_query, local_items)
            best_city, best_city_items, best_city_score = self._select_best_global_city(
                query=active_query,
                local_city=active_city,
                global_items=global_items,
            )

            debug.parallel_recall_used = True
            debug.parallel_local_result_count = len(local_items)
            debug.parallel_global_result_count = len(global_items)
            debug.parallel_local_score = round(local_score, 4)
            debug.parallel_best_city = best_city
            debug.parallel_best_city_score = round(best_city_score, 4)

            switch_reason = self._switch_reason_by_relevance(
                query=active_query,
                local_city=active_city,
                local_items=local_items,
                local_score=local_score,
                candidate_city=best_city,
                candidate_items=best_city_items,
                candidate_score=best_city_score,
            )
            if switch_reason:
                debug.parallel_switch_reason = switch_reason
                debug.city_switch_suggested = True
                debug.city_switch_from = active_city
                debug.city_switch_to = best_city
                debug.trigger_reason = f"parallel_recall_compare:{switch_reason}"
                debug.corrected_query = active_query
                debug.corrected_city_hint = best_city
                debug.corrected_items = list(best_city_items)
                debug.corrected_result_count = len(best_city_items)

                if allow_city_switch:
                    final_items = best_city_items
                    debug.city_switch_applied = True
                    debug.final_query = active_query
                    debug.final_city_hint = best_city
                    return POISearchExecution(items=final_items, debug=debug)

                debug.final_query = active_query
                debug.final_city_hint = active_city
                return POISearchExecution(items=final_items, debug=debug)

            strong_landmark_candidate: RefineCandidate | None = None
            if self._should_run_strong_landmark_check(
                query=active_query,
                local_items=local_items,
                best_city=best_city,
                local_score=local_score,
                best_city_score=best_city_score,
            ):
                strong_landmark_candidate = self._get_strong_landmark_candidate(
                    query=active_query,
                    city_hint=active_city,
                    local_items=local_items,
                    best_city=best_city,
                    best_city_score=best_city_score,
                    best_city_items=best_city_items,
                    allow_city_switch=allow_city_switch,
                )

            if strong_landmark_candidate is not None and strong_landmark_candidate.suggested_city:
                forced_city = strong_landmark_candidate.suggested_city
                forced_query = strong_landmark_candidate.query
                forced_items = (
                    list(best_city_items)
                    if self._same_city(best_city, forced_city)
                    else []
                )

                debug.ai_refine_triggered = True
                debug.parallel_switch_reason = "strong_landmark_ai_force_switch"
                debug.city_switch_suggested = True
                debug.city_switch_from = active_city
                debug.city_switch_to = forced_city
                if debug.trigger_reason:
                    debug.trigger_reason = f"{debug.trigger_reason};strong_landmark_ai_force_switch"
                else:
                    debug.trigger_reason = "strong_landmark_ai_force_switch"
                debug.corrected_query = forced_query
                debug.corrected_city_hint = forced_city

                if allow_city_switch and not forced_items:
                    try:
                        forced_items = self._amap_service.search(
                            query=forced_query,
                            city_hint=forced_city,
                            limit=limit,
                        )
                    except Exception:
                        forced_items = []

                debug.corrected_items = list(forced_items)
                debug.corrected_result_count = len(forced_items)

                if allow_city_switch and forced_items:
                    debug.ai_refine_applied = True
                    debug.city_switch_applied = True
                    debug.final_query = forced_query
                    debug.final_city_hint = forced_city
                    return POISearchExecution(items=forced_items, debug=debug)

                debug.final_query = active_query
                debug.final_city_hint = active_city
                return POISearchExecution(items=final_items, debug=debug)
        else:
            base_items = self._amap_service.search(
                query=active_query,
                city_hint=active_city,
                limit=limit,
            )
            final_items = base_items

        debug.final_query = active_query
        debug.final_city_hint = active_city

        if not self._should_retry_with_ai(normalized_query, normalized_city, base_items):
            return POISearchExecution(items=final_items, debug=debug)

        retry_trigger = self._build_retry_trigger(normalized_query, normalized_city, base_items)
        debug.ai_refine_triggered = True
        if debug.trigger_reason:
            debug.trigger_reason = f"{debug.trigger_reason};{retry_trigger}"
        else:
            debug.trigger_reason = retry_trigger

        retry_candidate = self._get_ai_candidate(
            query=normalized_query,
            city_hint=normalized_city,
            trigger=retry_trigger,
            base_items=base_items,
            allow_city_switch=allow_city_switch,
        )
        if retry_candidate is None:
            return POISearchExecution(items=final_items, debug=debug)

        retry_query = retry_candidate.query
        retry_city = retry_candidate.city

        suggested_city = retry_candidate.suggested_city
        if suggested_city and normalized_city and self._city_key(suggested_city) != self._city_key(normalized_city):
            debug.city_switch_suggested = True
            debug.city_switch_from = normalized_city
            debug.city_switch_to = suggested_city

        if retry_query == active_query and retry_city == active_city:
            if debug.city_switch_suggested:
                debug.trigger_reason = retry_trigger
                debug.ai_refine_triggered = True
                debug.corrected_query = retry_query
                debug.corrected_city_hint = suggested_city
            return POISearchExecution(items=final_items, debug=debug)

        retry_items = self._amap_service.search(
            query=retry_query,
            city_hint=retry_city,
            limit=limit,
        )
        debug.corrected_query = retry_query
        debug.corrected_city_hint = suggested_city or retry_city
        city_switched = bool(
            normalized_city and retry_city and self._city_key(retry_city) != self._city_key(normalized_city)
        )
        if city_switched:
            debug.city_switch_suggested = True
            debug.city_switch_from = normalized_city
            debug.city_switch_to = retry_city
        debug.corrected_items = list(retry_items)
        debug.corrected_result_count = len(retry_items)

        if self._prefer_retry(base_items, retry_items, active_query, retry_query):
            final_items = retry_items
            debug.ai_refine_applied = True
            debug.final_query = retry_query
            debug.final_city_hint = retry_city
            if city_switched and allow_city_switch:
                debug.city_switch_applied = True

        return POISearchExecution(items=final_items, debug=debug)

    def _get_ai_candidate(
        self,
        query: str,
        city_hint: str | None,
        trigger: str,
        base_items: list[POIItem],
        allow_city_switch: bool,
    ) -> RefineCandidate | None:
        """Return refined query/city only when confidence and diff checks pass."""
        decision = self._query_refiner.refine(
            query=query,
            city_hint=city_hint,
            trigger=trigger,
            result_count=len(base_items),
            top_items=base_items,
        )
        if decision is None:
            return None

        if decision.confidence < settings.poi_ai_confidence_threshold:
            return None

        refined_query = self._normalize_query(decision.cleaned_query)
        if not refined_query:
            return None

        suggested_city = self._normalize_city(decision.city_hint_override)
        if suggested_city and city_hint and self._city_key(suggested_city) == self._city_key(city_hint):
            suggested_city = None

        if allow_city_switch:
            refined_city = suggested_city or city_hint
        else:
            refined_city = city_hint

        if refined_query == query and refined_city == city_hint and suggested_city is None:
            return None

        return RefineCandidate(query=refined_query, city=refined_city, suggested_city=suggested_city)

    def _should_retry_with_ai(self, query: str, city_hint: str | None, items: list[POIItem]) -> bool:
        """Decide whether second-pass AI correction is worth trying."""
        if not settings.poi_ai_refine_enabled or not self._query_refiner.enabled:
            return False

        if city_hint and not self._has_full_query_name_match(query, items):
            return True

        if settings.poi_ai_retry_zero_result and not items:
            return True

        if self._is_alpha_only(query) and len(items) < settings.poi_ai_alpha_result_threshold:
            return True

        if self._is_typo_risk(query):
            top1_name = items[0].name if items else ""
            if self._similarity(query, top1_name) < 0.5:
                return True

        return False

    def _build_retry_trigger(self, query: str, city_hint: str | None, items: list[POIItem]) -> str:
        """Build concise trigger reason for the AI refiner."""
        if city_hint and not self._has_full_query_name_match(query, items):
            return "name_miss_query_substring"
        if not items:
            return "zero_result"
        if self._is_alpha_only(query):
            return "alpha_input_low_result"
        return "typo_risk_low_similarity"

    def _search_parallel_local_global(
        self,
        query: str,
        city_hint: str,
        limit: int,
    ) -> tuple[list[POIItem], list[POIItem]]:
        """Run local-city and nationwide recall in parallel for comparison."""
        with ThreadPoolExecutor(max_workers=2) as pool:
            local_future = pool.submit(self._amap_service.search, query, city_hint, limit)
            global_future = pool.submit(self._amap_service.search, query, None, limit)
            local_items = local_future.result()
            try:
                global_items = global_future.result()
            except Exception:
                # Keep local result path available if global recall is rate-limited/unavailable.
                global_items = []

        return local_items, global_items

    def _select_best_global_city(
        self,
        query: str,
        local_city: str,
        global_items: list[POIItem],
    ) -> tuple[str | None, list[POIItem], float]:
        """Pick the best candidate city from nationwide results by relevance score."""
        local_key = self._city_key(local_city)
        grouped: dict[str, list[POIItem]] = defaultdict(list)
        for item in global_items:
            city = self._normalize_city(item.city)
            if city is None:
                continue
            grouped[city].append(item)

        best_city: str | None = None
        best_items: list[POIItem] = []
        best_score = 0.0
        for city, items in grouped.items():
            if self._city_key(city) == local_key:
                continue

            ranked_items = sorted(
                items,
                key=lambda item: self._item_relevance_score(query, item),
                reverse=True,
            )
            score = self._city_relevance_score(query, ranked_items)
            if score > best_score:
                best_city = city
                best_items = ranked_items
                best_score = score

        return best_city, best_items, best_score

    def _switch_reason_by_relevance(
        self,
        query: str,
        local_city: str,
        local_items: list[POIItem],
        local_score: float,
        candidate_city: str | None,
        candidate_items: list[POIItem],
        candidate_score: float,
    ) -> str | None:
        """Return switch reason when candidate city is significantly more relevant."""
        if candidate_city is None or not candidate_items:
            return None
        if self._city_key(candidate_city) == self._city_key(local_city):
            return None

        margin = max(0.0, settings.poi_parallel_city_switch_margin)
        min_score = max(0.0, settings.poi_parallel_city_switch_min_score)
        score_delta = candidate_score - local_score

        local_full_match = self._has_full_query_name_match(query, local_items)
        candidate_full_match = self._has_full_query_name_match(query, candidate_items)
        local_ordered_match = self._has_ordered_query_name_match(query, local_items)
        candidate_ordered_match = self._has_ordered_query_name_match(query, candidate_items)

        if candidate_full_match and not local_full_match:
            return "full_name_match_gain"

        if candidate_ordered_match and not local_ordered_match and score_delta >= margin * 0.25:
            return "ordered_name_match_gain"

        if candidate_score >= min_score and score_delta >= margin * 1.5:
            return "high_relevance_gain"

        return None

    def _should_run_strong_landmark_check(
        self,
        query: str,
        local_items: list[POIItem],
        best_city: str | None,
        local_score: float,
        best_city_score: float,
    ) -> bool:
        """Decide whether to run AI strong-landmark arbitration."""
        if not settings.poi_ai_strong_landmark_enabled or not self._query_refiner.enabled:
            return False
        if best_city is None or not local_items:
            return False

        query_key = self._text_key(query)
        if len(query_key) < 2 or len(query_key) > 12:
            return False

        # Only arbitrate when local looks relevant enough to cause ambiguity.
        if not self._has_full_query_name_match(query, local_items):
            return False

        # Candidate city should be competitive enough to be meaningful.
        if best_city_score < settings.poi_parallel_city_switch_min_score * 0.8:
            return False
        if best_city_score + 0.03 < local_score:
            return False

        return True

    def _get_strong_landmark_candidate(
        self,
        query: str,
        city_hint: str,
        local_items: list[POIItem],
        best_city: str,
        best_city_score: float,
        best_city_items: list[POIItem],
        allow_city_switch: bool,
    ) -> RefineCandidate | None:
        """Get AI forced city-switch candidate for strong landmark queries."""
        decision = self._query_refiner.detect_strong_landmark(
            query=query,
            city_hint=city_hint,
            local_items=local_items,
            candidate_city=best_city,
            candidate_score=best_city_score,
            candidate_items=best_city_items,
        )
        if decision is None:
            return None

        if decision.confidence < settings.poi_ai_strong_landmark_confidence_threshold:
            return None
        if not (decision.is_strong_landmark or decision.force_city_switch):
            return None

        refined_query = self._normalize_query(decision.cleaned_query)
        if not refined_query:
            refined_query = query

        suggested_city = self._normalize_city(decision.city_hint_override) or best_city
        if suggested_city is None:
            return None
        if self._city_key(suggested_city) == self._city_key(city_hint):
            return None

        refined_city = suggested_city if allow_city_switch else city_hint
        return RefineCandidate(query=refined_query, city=refined_city, suggested_city=suggested_city)

    def _city_relevance_score(self, query: str, items: list[POIItem]) -> float:
        """Compute city-level relevance score from ranked POI items."""
        if not items:
            return 0.0

        top_items = items[:8]
        weighted_sum = 0.0
        weight_total = 0.0
        full_match_count = 0
        ordered_match_count = 0

        for idx, item in enumerate(top_items):
            weight = 1.0 / (idx + 1)
            weighted_sum += self._item_relevance_score(query, item) * weight
            weight_total += weight
            if self._query_in_name(query, item.name):
                full_match_count += 1
            if self._query_in_order(query, item.name):
                ordered_match_count += 1

        rank_score = weighted_sum / weight_total if weight_total else 0.0
        full_match_ratio = full_match_count / len(top_items)
        ordered_match_ratio = ordered_match_count / len(top_items)
        recall_score = min(len(items), 8) / 8.0
        return min(1.0, rank_score * 0.6 + full_match_ratio * 0.2 + ordered_match_ratio * 0.15 + recall_score * 0.05)

    def _item_relevance_score(self, query: str, item: POIItem) -> float:
        """Compute item-level relevance score for sorting and comparison."""
        query_key = self._text_key(query)
        name_key = self._text_key(item.name)

        full_match_bonus = 0.25 if query_key and query_key in name_key else 0.0
        ordered_match_bonus = 0.18 if self._query_in_order(query, item.name) else 0.0
        prefix_bonus = 0.10 if query_key and name_key.startswith(query_key) else 0.0
        name_similarity = self._similarity(query, item.name)
        address_similarity = self._similarity(query, item.address)

        return min(
            1.0,
            name_similarity * 0.45 + address_similarity * 0.08 + full_match_bonus + ordered_match_bonus + prefix_bonus,
        )

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
        return retry_score > base_score + 0.12

    def _result_score(self, items: list[POIItem], query: str) -> float:
        """Simple score combining top-1 relevance and recall count."""
        if not items:
            return 0.0

        top1 = items[0]
        top_text = f"{top1.name} {top1.address}"
        relevance = self._similarity(query, top_text)
        recall = min(len(items), 5) / 5.0
        return relevance * 0.65 + recall * 0.35

    @staticmethod
    def _normalize_query(query: str) -> str:
        """Apply deterministic cleanup before any provider call."""
        text = unicodedata.normalize("NFKC", str(query or ""))
        text = text.strip()
        text = re.sub(r"\s+", "", text)
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
        if not city:
            return None

        municipality = re.fullmatch(r"(北京|上海|天津|重庆)(?:市)?城区", city)
        if municipality:
            return f"{municipality.group(1)}市"

        if city.endswith("城区") and len(city) <= 4:
            city = city[:-2]

        if city in MUNICIPALITY_BASES:
            return f"{city}市"

        return city or None

    @staticmethod
    def _city_key(city: str) -> str:
        """Canonical city key for conflict comparison."""
        norm = unicodedata.normalize("NFKC", city).strip()
        norm = re.sub(r"\s+", "", norm)
        norm = re.sub(r"(?:市)?城区$", "", norm)
        return re.sub(r"(省|市|自治区|特别行政区)$", "", norm)

    def _same_city(self, left: str | None, right: str | None) -> bool:
        """Whether two city labels refer to the same canonical city."""
        if not left or not right:
            return False
        return self._city_key(left) == self._city_key(right)

    @staticmethod
    def _is_alpha_only(query: str) -> bool:
        """Check if query is pure Latin letters (common pinyin input case)."""
        lowered = query.lower()
        return bool(re.fullmatch(r"[a-z]+", lowered))

    @staticmethod
    def _is_typo_risk(query: str) -> bool:
        """Detect typo-prone tokens that often degrade first-pass precision."""
        return any(token in query for token in TYPO_RISK_TOKENS)

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
    def _query_in_order(query: str, name: str) -> bool:
        """Whether query chars appear in order within normalized name."""
        query_key = POISearchService._text_key(query)
        name_key = POISearchService._text_key(name)
        if not query_key or not name_key:
            return False

        i = 0
        for ch in name_key:
            if ch == query_key[i]:
                i += 1
                if i == len(query_key):
                    return True

        return False

    @staticmethod
    def _has_ordered_query_name_match(query: str, items: list[POIItem]) -> bool:
        """Whether any result name includes ordered query chars."""
        for item in items:
            if POISearchService._query_in_order(query, item.name):
                return True

        return False

    @staticmethod
    def _has_full_query_name_match(query: str, items: list[POIItem]) -> bool:
        """Whether any result name contains the full normalized query substring."""
        for item in items:
            if POISearchService._query_in_name(query, item.name):
                return True

        return False
