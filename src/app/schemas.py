from __future__ import annotations

from pydantic import BaseModel, Field


class ChatMessageRequest(BaseModel):
	"""Frontend natural-language message payload."""

	message: str = Field(default="", max_length=2000)


class ChatMessageResponse(BaseModel):
	"""Ack payload for message endpoint."""

	ok: bool
	detail: str
	llm_task_order: list[str] = Field(default_factory=list)
	mcp_tool_order: list[str] = Field(default_factory=list)
	available_mcp_tools: list[str] = Field(default_factory=list)
	error: str | None = None
