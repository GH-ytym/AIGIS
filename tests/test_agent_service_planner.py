"""Regression tests for planner parsing and fallback behavior."""

from __future__ import annotations

from types import SimpleNamespace

from aigis_agent.services.agent_service import AgentService


class FakeChatModel:
    """Minimal fake model that returns a fixed text payload."""

    def __init__(self, text: str):
        self._text = text

    def invoke(self, _messages):
        return SimpleNamespace(content=self._text)


def test_parse_json_object_with_fenced_json() -> None:
    text = """```json\n{"tasks":[{"step_id":1,"tool_name":"search_poi","reason":"ok","args":{"query":"咖啡"}}],"summary":"ok"}\n```"""
    parsed = AgentService._parse_json_object(text)
    assert isinstance(parsed, dict)
    assert parsed.get("summary") == "ok"


def test_plan_assist_tasks_succeeds_with_plain_json() -> None:
    service = AgentService()
    service._chat_model = FakeChatModel(
        '{"tasks":[{"step_id":1,"tool_name":"search_poi","reason":"plan","args":{"query":"武汉咖啡","limit":5}}],"summary":"planner-ok"}'
    )

    plan, fallback = service._plan_assist_tasks(
        query="武汉咖啡",
        city_hint="武汉市",
        limit=20,
        allow_city_switch=False,
    )

    assert fallback is False
    assert plan.summary == "planner-ok"
    assert len(plan.tasks) == 1
    assert plan.tasks[0].tool_name == "search_poi"


def test_plan_assist_tasks_fallback_when_invalid_json() -> None:
    service = AgentService()
    service._chat_model = FakeChatModel("not a json payload")

    plan, fallback = service._plan_assist_tasks(
        query="武汉咖啡",
        city_hint="武汉市",
        limit=20,
        allow_city_switch=False,
    )

    assert fallback is True
    assert plan.summary == "fallback-poi"
    assert len(plan.tasks) == 1
    assert plan.tasks[0].tool_name == "search_poi"
