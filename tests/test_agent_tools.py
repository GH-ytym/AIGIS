"""Regression tests for executable tool validation and execution policy."""

from __future__ import annotations

import pytest

from aigis_agent.services.agent_tools import (
    get_execution_tool_names,
    invoke_execution_tool,
)


def test_execution_tool_registry_contains_search_poi() -> None:
    assert "search_poi" in get_execution_tool_names()


def test_invoke_execution_tool_rejects_invalid_limit_type() -> None:
    result = invoke_execution_tool("search_poi", {"query": "咖啡", "limit": "bad"})
    assert result.get("status") == "error"
    assert isinstance(result.get("validation_errors"), list)


def test_invoke_execution_tool_raises_for_unknown_tool() -> None:
    with pytest.raises(ValueError):
        invoke_execution_tool("unknown_tool", {"query": "咖啡"})
