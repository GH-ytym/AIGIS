"""Schemas for LangChain-powered agent requests."""

from typing import Any, Literal

from pydantic import BaseModel, Field


class AgentQueryRequest(BaseModel):
    """Natural language command for GIS workflow routing."""

    query: str


class AgentQueryResponse(BaseModel):
    """Agent response placeholder with inferred intent."""

    intent: str
    message: str
    suggested_endpoint: str


class AgentTaskStep(BaseModel):
    """Single planned task produced by the assistant planner."""

    step_id: int
    tool_name: str
    reason: str = ""
    args: dict[str, Any] = Field(default_factory=dict)


class AgentAssistRequest(BaseModel):
    """Assistant request for NL planning and tool execution."""

    query: str
    city_hint: str | None = None
    limit: int = Field(default=20, ge=1, le=50)
    allow_city_switch: bool = False


class AgentAssistResponse(BaseModel):
    """Assistant response containing plan and executed tool output."""

    message: str
    selected_tool: str
    tasks: list[AgentTaskStep] = Field(default_factory=list)
    tool_result: dict[str, Any] = Field(default_factory=dict)
    provider: str
    model: str
    fallback_used: bool = False


class AgentChatMessage(BaseModel):
    """Single chat message used for optional short history context."""

    role: Literal["user", "assistant"]
    content: str


class AgentChatRequest(BaseModel):
    """Chat request payload for DeepSeek/OpenAI-compatible model."""

    message: str
    history: list[AgentChatMessage] = Field(default_factory=list)


class AgentChatResponse(BaseModel):
    """Chat response returned to web client."""

    reply: str
    provider: str
    model: str
