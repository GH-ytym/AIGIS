from __future__ import annotations

import json
import os
import re
from typing import Any

from langchain_core.messages import HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI

from app.mcp_gateway import TencentMCPGateway
from app.schemas import ChatMessageResponse


class AppOrchestrator:
	"""Orchestrator for NL intake and planning."""

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
		"""Run NL -> LLM task split -> MCP tool-order planning (no tool execution)."""
		safe_message = str(message or "").strip()
		if not safe_message:
			return ChatMessageResponse(
				ok=False,
				detail="请输入自然语言查询内容。",
				error="empty_message",
			)

		available_tools, mcp_error = await self._gateway.get_available_tool_names()

		tasks: list[str] = []
		tool_order: list[str] = []
		planner_error: str | None = None

		if self._minimax_llm is None:
			planner_error = "未配置 MINIMAX_API_KEY，已降级到规则规划。"
			tasks, tool_order = self._fallback_plan(safe_message, available_tools)
		else:
			tasks, tool_order, planner_error = await self._plan_with_llm(safe_message, available_tools)
			if not tasks or (available_tools and not tool_order):
				fallback_tasks, fallback_tools = self._fallback_plan(safe_message, available_tools)
				if not tasks:
					tasks = fallback_tasks
				if available_tools and not tool_order:
					tool_order = fallback_tools
				if planner_error is None:
					planner_error = "LLM 规划结果不完整，已降级到规则规划。"

		errors = [item for item in [mcp_error, planner_error] if item]
		detail = "已完成闭环规划，返回了任务顺序和MCP工具调用顺序（未执行具体工具）。"
		if errors:
			detail = "已完成闭环规划（含降级），返回了任务顺序和MCP工具调用顺序（未执行具体工具）。"

		return ChatMessageResponse(
			ok=not errors,
			detail=detail,
			llm_task_order=tasks,
			mcp_tool_order=tool_order,
			available_mcp_tools=available_tools,
			error=" | ".join(errors) if errors else None,
		)

	async def _plan_with_llm(
		self,
		message: str,
		available_tools: list[str],
	) -> tuple[list[str], list[str], str | None]:
		"""Use LLM to output task split and tool order in strict JSON."""
		if self._minimax_llm is None:
			return [], [], "未配置 MINIMAX_API_KEY，无法执行 LLM 规划。"

		planner_prompt = (
			"你是武汉GIS助手的任务编排器。"
			"请基于用户输入拆解任务步骤，并从 available_mcp_tools 中选择工具调用顺序。"
			"当前阶段只做规划，不执行任何工具。"
			"你必须只输出 JSON 对象，禁止输出 markdown。"
			"JSON 结构: "
			'{"tasks":["步骤1","步骤2"],"tool_order":["工具名1","工具名2"]}'
			"其中 tool_order 里的值必须是 available_mcp_tools 中的精确工具名。"
		)
		payload = {
			"user_message": message,
			"available_mcp_tools": available_tools,
			"city_scope": "武汉市",
			"execute_tools": False,
		}

		try:
			result = await self._minimax_llm.ainvoke(
				[
					SystemMessage(content=planner_prompt),
					HumanMessage(content=json.dumps(payload, ensure_ascii=False)),
				]
			)
			raw_text = self._extract_langchain_text(result)
			cleaned_text = self._strip_thinking_block(raw_text) or raw_text
			parsed = self._parse_json_object(cleaned_text)

			if parsed is None:
				return [], [], "LLM 未返回可解析的 JSON 规划结果。"

			tasks = self._normalize_tasks(parsed.get("tasks"))
			tool_order = self._normalize_tool_order(parsed.get("tool_order"), available_tools)

			if not tasks:
				return [], tool_order, "LLM 未返回有效任务拆分。"
			if available_tools and not tool_order:
				return tasks, [], "LLM 未返回有效 MCP 工具顺序。"

			return tasks, tool_order, None
		except Exception as exc:
			return [], [], f"LLM 规划失败: {exc}"

	def _fallback_plan(self, message: str, available_tools: list[str]) -> tuple[list[str], list[str]]:
		"""Fallback planner when LLM is unavailable or output is invalid."""
		tasks = [
			"解析用户意图与地理约束（武汉范围）",
			"拆分子任务并确定每步需要的地理能力",
			"按顺序组织 MCP 工具调用计划",
			"汇总执行说明并返回前端",
		]

		ordered_candidates: list[str]
		if self._contains_any(message, ["天气", "气温", "降雨"]):
			ordered_candidates = ["weather", "geocoder", "placeSuggestion"]
		elif self._contains_any(message, ["路线", "导航", "怎么去", "驾车", "步行", "公交", "骑行"]):
			ordered_candidates = ["geocoder", "directionDriving", "directionTransit", "directionWalking", "directionBicycling"]
		elif self._contains_any(message, ["附近", "周边", "附近有什么", "找", "搜索"]):
			ordered_candidates = ["geocoder", "placeSearchNearby", "placeSuggestion", "placeDetail"]
		else:
			ordered_candidates = ["geocoder", "placeSuggestion", "placeDetail", "directionDriving"]

		tool_order: list[str] = []
		for name in ordered_candidates:
			if name in available_tools and name not in tool_order:
				tool_order.append(name)

		if not tool_order and available_tools:
			tool_order = available_tools[: min(3, len(available_tools))]

		return tasks, tool_order

	@staticmethod
	def _contains_any(text: str, needles: list[str]) -> bool:
		value = str(text or "")
		for needle in needles:
			if needle and needle in value:
				return True
		return False

	@staticmethod
	def _normalize_tasks(raw_tasks: Any) -> list[str]:
		if not isinstance(raw_tasks, list):
			return []

		normalized: list[str] = []
		for item in raw_tasks:
			if isinstance(item, str):
				text = item.strip()
			elif isinstance(item, dict):
				text = str(
					item.get("task")
					or item.get("name")
					or item.get("step")
					or item.get("description")
					or ""
				).strip()
			else:
				text = str(item).strip()

			if text:
				normalized.append(text)
			if len(normalized) >= 8:
				break

		return normalized

	@staticmethod
	def _normalize_tool_order(raw_order: Any, available_tools: list[str]) -> list[str]:
		if not isinstance(raw_order, list):
			return []

		tool_set = set(available_tools)
		normalized: list[str] = []
		for item in raw_order:
			if isinstance(item, str):
				name = item.strip()
			elif isinstance(item, dict):
				name = str(item.get("tool") or item.get("tool_name") or item.get("name") or "").strip()
			else:
				name = str(item).strip()

			if name in tool_set and name not in normalized:
				normalized.append(name)

		return normalized

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
