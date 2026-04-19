from __future__ import annotations

import json
import os
import re
from typing import Any, Callable

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.mcp_gateway import TencentMCPGateway
from app.schemas import ChatMessageResponse


WUHAN_FIXED_ROUTE_ORIGIN = "武汉大学正门（牌坊门）"
WUHAN_FIXED_ROUTE_ORIGIN_COORD = "30.5362,114.3644"

SEARCH_TOOL_PRIORITY = (
	"placeSuggestion",
	"placeSearchNearby",
	"geocoder",
	"placeDetail",
)

ROUTE_TOOL_BY_MODE = {
	"driving": "directionDriving",
	"walking": "directionWalking",
	"bicycling": "directionBicycling",
	"transit": "directionTransit",
}

ROUTE_TOOL_FALLBACK = (
	"directionDriving",
	"directionTransit",
	"directionWalking",
	"directionBicycling",
)


class AppOrchestrator:
	"""Orchestrator for NL intake, tool planning, and MCP execution."""

	def __init__(self) -> None:
		self._gateway = TencentMCPGateway()
		self._minimax_model = os.getenv("MINIMAX_MODEL", "MiniMax-M2.7")
		self._minimax_base_url = os.getenv("MINIMAX_BASE_URL", "https://api.minimaxi.com/v1")
		self._minimax_api_key = os.getenv("MINIMAX_API_KEY", "").strip()
		self._minimax_llm: ChatOpenAI | None = None

		if self._minimax_api_key:
			self._minimax_llm = ChatOpenAI(
				api_key=self._minimax_api_key,
				base_url=self._minimax_base_url,
				model=self._minimax_model,
				temperature=0.2,
				timeout=60,
			)

	async def handle_user_message(self, message: str) -> ChatMessageResponse:
		"""Run NL -> intent extraction -> search + route MCP execution."""
		safe_message = str(message or "").strip()
		if not safe_message:
			return ChatMessageResponse(
				ok=False,
				detail="请输入自然语言查询内容。",
				error="empty_message",
			)

		available_tools, mcp_error = await self._gateway.get_available_tool_names()
		search_tool = self._pick_search_tool(available_tools)

		destination_query = ""
		travel_mode = "driving"
		planner_warning: str | None = None

		if self._minimax_llm is None:
			planner_warning = "未配置 MINIMAX_API_KEY，已降级到规则解析。"
			destination_query, travel_mode = self._fallback_intent(safe_message)
		else:
			destination_query, travel_mode, planner_warning = await self._extract_intent_with_llm(safe_message)
			if not destination_query:
				destination_query, travel_mode = self._fallback_intent(safe_message)
				if planner_warning is None:
					planner_warning = "LLM 解析结果不完整，已降级到规则解析。"

		route_tool = self._pick_route_tool(available_tools, travel_mode)

		llm_tasks = [
			"解析用户意图，提取目的地与出行方式",
			f"调用 {search_tool or '搜索工具'} 搜索目的地候选",
			f"固定起点为 {WUHAN_FIXED_ROUTE_ORIGIN}，调用 {route_tool or '路径工具'} 进行路径规划",
			"汇总搜索与路径结果并返回前端",
		]

		missing_tools: list[str] = []
		if search_tool is None:
			missing_tools.append("缺少搜索工具（placeSuggestion/placeSearchNearby/geocoder/placeDetail）")
		if route_tool is None:
			missing_tools.append("缺少路径规划工具（directionDriving/directionTransit/directionWalking/directionBicycling）")

		search_result: Any = None
		route_result: Any = None
		search_error: str | None = None
		route_error: str | None = None

		if not mcp_error and not missing_tools and search_tool and route_tool:
			search_payloads = self._build_search_payload_candidates(search_tool, destination_query)
			search_result, search_error = await self._invoke_tool_with_fallback_payloads(search_tool, search_payloads)

			resolved_destination, destination_coord = self._resolve_destination(search_result, destination_query)
			route_payloads = self._build_route_payload_candidates(resolved_destination, destination_coord)
			route_result, route_error = await self._invoke_tool_with_fallback_payloads(
				route_tool,
				route_payloads,
				result_ok_checker=self._route_result_has_path,
			)

		errors = [item for item in [mcp_error, *missing_tools, search_error, route_error] if item]
		ok = len(errors) == 0

		if ok:
			detail = (
				f"已执行 MCP 工具：{search_tool} -> {route_tool}，"
				f"路径规划起点固定为 {WUHAN_FIXED_ROUTE_ORIGIN}。"
			)
		else:
			detail = "已尝试执行搜索和路径规划，但部分步骤失败。"

		if planner_warning:
			detail = f"{detail} 提示：{planner_warning}"

		return ChatMessageResponse(
			ok=ok,
			detail=detail,
			llm_task_order=llm_tasks,
			mcp_tool_order=[item for item in [search_tool, route_tool] if item],
			available_mcp_tools=available_tools,
			search_tool=search_tool,
			route_tool=route_tool,
			fixed_route_origin=WUHAN_FIXED_ROUTE_ORIGIN,
			search_result_preview=self._preview_tool_output(search_result),
			route_result_preview=self._preview_tool_output(route_result),
			error=" | ".join(errors) if errors else None,
		)

	async def _extract_intent_with_llm(self, message: str) -> tuple[str, str, str | None]:
		"""Use LLM to extract destination query and travel mode."""
		if self._minimax_llm is None:
			destination_query, travel_mode = self._fallback_intent(message)
			return destination_query, travel_mode, "未配置 MINIMAX_API_KEY，无法执行 LLM 解析。"

		prompt = (
			"你是武汉 GIS 助手。请从用户输入中提取路径规划意图。"
			"仅输出 JSON 对象，不要输出 markdown。"
			'{"destination_query":"地点关键词","travel_mode":"driving|walking|bicycling|transit"}。'
			"destination_query 必须是可用于地图搜索的短文本。"
		)
		payload = {
			"user_message": message,
			"city_scope": "武汉市",
			"fixed_route_origin": WUHAN_FIXED_ROUTE_ORIGIN,
		}

		try:
			result = await self._minimax_llm.ainvoke(
				[
					SystemMessage(content=prompt),
					HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
				]
			)
			raw_text = self._extract_langchain_text(result)
			cleaned_text = self._strip_thinking_block(raw_text) or raw_text
			parsed = self._parse_json_object(cleaned_text)
			if parsed is None:
				destination_query, travel_mode = self._fallback_intent(message)
				return destination_query, travel_mode, "LLM 未返回可解析 JSON，已降级到规则解析。"

			destination_query = str(parsed.get("destination_query") or "").strip()
			travel_mode = self._normalize_travel_mode(parsed.get("travel_mode"))
			if not destination_query:
				destination_query, fallback_mode = self._fallback_intent(message)
				if not travel_mode:
					travel_mode = fallback_mode
				return destination_query, travel_mode, "LLM 未返回有效 destination_query，已降级到规则解析。"

			return destination_query, travel_mode or "driving", None
		except Exception as exc:
			destination_query, travel_mode = self._fallback_intent(message)
			return destination_query, travel_mode, f"LLM 解析失败: {exc}"

	@staticmethod
	def _normalize_travel_mode(mode: Any) -> str:
		value = str(mode or "").strip().lower()
		if value in ROUTE_TOOL_BY_MODE:
			return value
		return ""

	@staticmethod
	def _fallback_intent(message: str) -> tuple[str, str]:
		text = str(message or "").strip()
		travel_mode = "driving"

		if AppOrchestrator._contains_any(text, ["步行", "走路", "步行去"]):
			travel_mode = "walking"
		elif AppOrchestrator._contains_any(text, ["骑行", "自行车", "骑车"]):
			travel_mode = "bicycling"
		elif AppOrchestrator._contains_any(text, ["公交", "地铁", "换乘", "公共交通"]):
			travel_mode = "transit"

		destination = ""
		patterns = [
			r"到(?P<dest>[^，。,；;？?]+?)(怎么走|怎么去|路线|导航|$)",
			r"去(?P<dest>[^，。,；;？?]+?)(怎么走|怎么去|路线|导航|$)",
			r"前往(?P<dest>[^，。,；;？?]+?)(怎么走|怎么去|路线|导航|$)",
		]

		for pattern in patterns:
			match = re.search(pattern, text)
			if match:
				destination = str(match.group("dest") or "").strip()
				break

		if not destination:
			cleaned = re.sub(r"(从|到|去|前往|导航|怎么走|怎么去|路线规划|路径规划)", " ", text)
			cleaned = re.sub(r"\s+", " ", cleaned).strip(" ，。,；;？?")
			destination = cleaned or text

		return destination, travel_mode

	@staticmethod
	def _pick_search_tool(available_tools: list[str]) -> str | None:
		tool_set = set(available_tools)
		for candidate in SEARCH_TOOL_PRIORITY:
			if candidate in tool_set:
				return candidate
		return None

	@staticmethod
	def _pick_route_tool(available_tools: list[str], travel_mode: str) -> str | None:
		tool_set = set(available_tools)

		mode_tool = ROUTE_TOOL_BY_MODE.get(travel_mode)
		if mode_tool and mode_tool in tool_set:
			return mode_tool

		for candidate in ROUTE_TOOL_FALLBACK:
			if candidate in tool_set:
				return candidate

		return None

	@staticmethod
	def _build_search_payload_candidates(search_tool: str, destination_query: str) -> list[dict[str, Any]]:
		query = str(destination_query or "").strip() or "武汉市"
		candidates: list[dict[str, Any]] = []

		if search_tool == "placeSuggestion":
			candidates = [
				{"keyword": query, "region": "武汉市"},
				{"keyword": query},
				{"query": query},
			]
		elif search_tool == "placeSearchNearby":
			candidates = [
				{"keyword": query, "location": WUHAN_FIXED_ROUTE_ORIGIN_COORD, "radius": 5000},
				{"keyword": query, "location": WUHAN_FIXED_ROUTE_ORIGIN_COORD},
				{"keyword": query},
			]
		elif search_tool == "geocoder":
			candidates = [
				{"address": f"武汉市{query}"},
				{"address": query},
			]
		elif search_tool == "placeDetail":
			candidates = [
				{"keyword": query},
				{"id": query},
			]
		else:
			candidates = [{"query": query}, {"keyword": query}]

		return AppOrchestrator._deduplicate_payloads(candidates)

	@staticmethod
	def _build_route_payload_candidates(destination_text: str, destination_coord: str | None) -> list[dict[str, Any]]:
		dest_text = str(destination_text or "").strip() or "武汉市中心"
		candidates: list[dict[str, Any]] = []

		if destination_coord:
			candidates.extend(
				[
					{"from": WUHAN_FIXED_ROUTE_ORIGIN_COORD, "to": destination_coord},
					{"from": WUHAN_FIXED_ROUTE_ORIGIN_COORD, "to": dest_text},
					{"from": WUHAN_FIXED_ROUTE_ORIGIN, "to": destination_coord},
					{"from": WUHAN_FIXED_ROUTE_ORIGIN, "to": dest_text},
					{"origin": WUHAN_FIXED_ROUTE_ORIGIN_COORD, "destination": destination_coord},
					{"origin": WUHAN_FIXED_ROUTE_ORIGIN, "destination": dest_text},
				]
			)
		else:
			candidates.extend(
				[
					{"from": WUHAN_FIXED_ROUTE_ORIGIN, "to": dest_text},
					{"from": WUHAN_FIXED_ROUTE_ORIGIN_COORD, "to": dest_text},
					{"origin": WUHAN_FIXED_ROUTE_ORIGIN, "destination": dest_text},
				]
			)

		return AppOrchestrator._deduplicate_payloads(candidates)

	@staticmethod
	def _deduplicate_payloads(items: list[dict[str, Any]]) -> list[dict[str, Any]]:
		seen: set[str] = set()
		unique_items: list[dict[str, Any]] = []
		for payload in items:
			if not isinstance(payload, dict):
				continue
			key = json.dumps(payload, ensure_ascii=False, sort_keys=True)
			if key in seen:
				continue
			seen.add(key)
			unique_items.append(payload)
		return unique_items

	async def _invoke_tool_with_fallback_payloads(
		self,
		tool_name: str,
		payload_candidates: list[dict[str, Any]],
		result_ok_checker: Callable[[Any], bool] | None = None,
	) -> tuple[Any, str | None]:
		tool = str(tool_name or "").strip()
		if not tool:
			return None, "工具名为空。"

		last_error: str | None = None
		for payload in payload_candidates:
			result, error = await self._gateway.invoke_tool(tool, payload)
			if error is None:
				if result_ok_checker is None or result_ok_checker(result):
					return result, None
				last_error = f"{tool} 返回结果无效; payload={self._preview_tool_output(payload)}; result={self._preview_tool_output(result)}"
				continue
			last_error = f"{error}; payload={self._preview_tool_output(payload)}"

		return None, last_error or f"{tool} 调用失败。"

	@staticmethod
	def _resolve_destination(search_result: Any, fallback_query: str) -> tuple[str, str | None]:
		name = AppOrchestrator._extract_text_candidate(search_result)
		destination_text = name or str(fallback_query or "").strip() or "武汉市中心"
		coord = AppOrchestrator._extract_coordinate_string(search_result)
		return destination_text, coord

	@staticmethod
	def _extract_text_candidate(value: Any) -> str:
		if isinstance(value, dict):
			for key in ("title", "name", "address", "poi_name", "poiName", "formatted_address"):
				candidate = value.get(key)
				if isinstance(candidate, str):
					text = candidate.strip()
					if text:
						return text

			for nested in value.values():
				text = AppOrchestrator._extract_text_candidate(nested)
				if text:
					return text
			return ""

		if isinstance(value, list):
			for item in value:
				text = AppOrchestrator._extract_text_candidate(item)
				if text:
					return text
			return ""

		if isinstance(value, str):
			raw = value.strip()
			if not raw:
				return ""

			maybe_json = AppOrchestrator._parse_json_object(raw)
			if maybe_json is not None:
				return AppOrchestrator._extract_text_candidate(maybe_json)

			# placeSuggestion text usually starts with lines like: (1)光谷广场[地铁站]
			poi_match = re.search(r"\(\d+\)\s*([^\[\]\n\r]+)", raw)
			if poi_match:
				candidate = str(poi_match.group(1) or "").strip()
				if candidate:
					return candidate

			lines = [line.strip() for line in raw.splitlines() if line.strip()]
			if lines:
				return lines[0][:120]
			return raw[:120]

		return ""

	@staticmethod
	def _extract_coordinate_string(value: Any) -> str | None:
		coord = AppOrchestrator._extract_coordinate_tuple(value)
		if coord is None:
			return None
		return f"{coord[0]},{coord[1]}"

	@staticmethod
	def _extract_coordinate_tuple(value: Any) -> tuple[float, float] | None:
		if isinstance(value, dict):
			lat = AppOrchestrator._coerce_float(
				value.get("lat")
				or value.get("latitude")
				or value.get("y")
			)
			lng = AppOrchestrator._coerce_float(
				value.get("lng")
				or value.get("lon")
				or value.get("longitude")
				or value.get("x")
			)
			if lat is not None and lng is not None:
				return lat, lng

			location = value.get("location")
			if location is not None:
				parsed = AppOrchestrator._extract_coordinate_tuple(location)
				if parsed is not None:
					return parsed

			for nested in value.values():
				parsed = AppOrchestrator._extract_coordinate_tuple(nested)
				if parsed is not None:
					return parsed
			return None

		if isinstance(value, list):
			for item in value:
				parsed = AppOrchestrator._extract_coordinate_tuple(item)
				if parsed is not None:
					return parsed
			return None

		if isinstance(value, str):
			raw = value.strip()
			if not raw:
				return None

			# Match Chinese label style, e.g. 纬度：30.506150\n经度：114.399646
			zh_match = re.search(
				r"纬度[:：]\s*(-?\d{1,3}\.\d+)[\s\S]*?经度[:：]\s*(-?\d{1,3}\.\d+)",
				raw,
				flags=re.IGNORECASE,
			)
			if zh_match:
				lat = AppOrchestrator._coerce_float(zh_match.group(1))
				lng = AppOrchestrator._coerce_float(zh_match.group(2))
				if lat is not None and lng is not None:
					return lat, lng

			# Match "lat,lng" style pairs inside plain text.
			match = re.search(r"(-?\d{1,3}\.\d+)\s*[,，]\s*(-?\d{1,3}\.\d+)", raw)
			if match:
				lat = AppOrchestrator._coerce_float(match.group(1))
				lng = AppOrchestrator._coerce_float(match.group(2))
				if lat is not None and lng is not None:
					return lat, lng

			maybe_json = AppOrchestrator._parse_json_object(raw)
			if maybe_json is not None:
				return AppOrchestrator._extract_coordinate_tuple(maybe_json)
			return None

		return None

	@staticmethod
	def _coerce_float(value: Any) -> float | None:
		try:
			if value is None:
				return None
			return float(value)
		except (TypeError, ValueError):
			return None

	@staticmethod
	def _preview_tool_output(value: Any, max_chars: int = 260) -> str | None:
		if value is None:
			return None

		if isinstance(value, (dict, list)):
			text = json.dumps(value, ensure_ascii=False)
		else:
			text = str(value)

		text = text.strip()
		if not text:
			return None

		if len(text) > max_chars:
			return f"{text[:max_chars]}..."
		return text

	@staticmethod
	def _route_result_has_path(value: Any) -> bool:
		if isinstance(value, dict):
			routes = value.get("routes")
			if isinstance(routes, list) and len(routes) > 0:
				return True

			for nested in value.values():
				if AppOrchestrator._route_result_has_path(nested):
					return True
			return False

		if isinstance(value, list):
			for item in value:
				if AppOrchestrator._route_result_has_path(item):
					return True
			return False

		if isinstance(value, str):
			raw = value.strip()
			if not raw:
				return False

			negative_markers = ("无路线信息", "无结果", "未找到", "失败", "error")
			if any(marker in raw for marker in negative_markers):
				return False

			positive_markers = ("路线总距离", "预估用时", "乘坐方案", "途经道路", "公里", "分钟", "distance")
			if any(marker in raw for marker in positive_markers):
				return True

			return len(raw) >= 12

		return False

	@staticmethod
	def _contains_any(text: str, needles: list[str]) -> bool:
		value = str(text or "")
		for needle in needles:
			if needle and needle in value:
				return True
		return False

	@staticmethod
	def _parse_json_object(text: str) -> dict[str, Any] | None:
		"""Parse a JSON object from raw LLM text with fenced/inline fallback."""
		raw = str(text or "").strip()
		if not raw:
			return None

		fenced = re.search(r"```(?:json)?\s*(\{[\s\S]*\})\s*```", raw, flags=re.IGNORECASE)
		candidate = fenced.group(1).strip() if fenced else raw

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

	@staticmethod
	def _extract_langchain_text(result: Any) -> str:
		content = getattr(result, "content", "")
		if isinstance(content, str):
			return content.strip()
		if isinstance(content, list):
			parts: list[str] = []
			for item in content:
				if isinstance(item, dict):
					text = item.get("text")
					if isinstance(text, str):
						parts.append(text)
				elif isinstance(item, str):
					parts.append(item)
			return "\n".join(part.strip() for part in parts if part.strip())
		return str(content).strip()

	@staticmethod
	def _strip_thinking_block(text: str) -> str:
		"""Remove embedded <think>...</think> blocks for stable downstream parsing."""
		cleaned = re.sub(r"<think>[\s\S]*?</think>", "", text, flags=re.IGNORECASE)
		return cleaned.strip()
