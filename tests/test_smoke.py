from __future__ import annotations

import asyncio
import sys
from pathlib import Path
from typing import Any


ROOT_DIR = Path(__file__).resolve().parents[1]
SRC_DIR = ROOT_DIR / "src"
if str(SRC_DIR) not in sys.path:
	sys.path.insert(0, str(SRC_DIR))


from app.orchestrator import (
	AppOrchestrator,
	WUHAN_FIXED_ROUTE_ORIGIN,
	WUHAN_FIXED_ROUTE_ORIGIN_COORD,
)


class FakeGateway:
	def __init__(self, available_tools: list[str]) -> None:
		self.available_tools = available_tools
		self.calls: list[tuple[str, dict[str, Any]]] = []

	async def get_available_tool_names(self) -> tuple[list[str], str | None]:
		return self.available_tools, None

	async def invoke_tool(self, tool_name: str, tool_input: dict[str, Any]) -> tuple[Any, str | None]:
		self.calls.append((tool_name, dict(tool_input)))

		if tool_name in {"placeSuggestion", "placeSearchNearby"}:
			return {
				"data": [
					{
						"title": "光谷广场",
						"location": {"lat": 30.5122, "lng": 114.4138},
					}
				]
			}, None

		if tool_name in {"directionDriving", "directionWalking"}:
			return {
				"routes": [
					{"distance": 12000, "duration": 1800},
				]
			}, None

		return None, f"unexpected tool: {tool_name}"


class FakeLLMResult:
	def __init__(self, content: str) -> None:
		self.content = content


class FakeLLM:
	def __init__(self, content: str) -> None:
		self._content = content

	async def ainvoke(self, _messages: list[Any]) -> FakeLLMResult:
		return FakeLLMResult(self._content)


def test_execute_search_and_route_with_llm_intent() -> None:
	gateway = FakeGateway(["placeSuggestion", "directionDriving"])
	orchestrator = AppOrchestrator()
	orchestrator._gateway = gateway
	orchestrator._minimax_llm = FakeLLM('{"destination_query":"光谷广场","travel_mode":"driving"}')

	response = asyncio.run(orchestrator.handle_user_message("从武大去光谷广场怎么走"))

	assert response.ok is True
	assert response.search_tool == "placeSuggestion"
	assert response.route_tool == "directionDriving"
	assert response.fixed_route_origin == WUHAN_FIXED_ROUTE_ORIGIN
	assert response.mcp_tool_order == ["placeSuggestion", "directionDriving"]
	assert response.search_result_preview is not None
	assert response.route_result_preview is not None

	assert len(gateway.calls) == 2
	assert gateway.calls[0][0] == "placeSuggestion"
	assert gateway.calls[1][0] == "directionDriving"
	assert gateway.calls[1][1]["from"] == WUHAN_FIXED_ROUTE_ORIGIN_COORD
	assert gateway.calls[1][1]["to"] == "30.5122,114.4138"


def test_execute_search_and_route_with_rule_fallback_mode() -> None:
	gateway = FakeGateway(["placeSuggestion", "directionWalking"])
	orchestrator = AppOrchestrator()
	orchestrator._gateway = gateway
	orchestrator._minimax_llm = None

	response = asyncio.run(orchestrator.handle_user_message("从武汉大学到光谷广场步行怎么走"))

	assert response.ok is True
	assert response.route_tool == "directionWalking"
	assert "已降级到规则解析" in response.detail
	assert len(gateway.calls) == 2
	assert gateway.calls[1][0] == "directionWalking"
	assert gateway.calls[1][1]["from"] == WUHAN_FIXED_ROUTE_ORIGIN_COORD
