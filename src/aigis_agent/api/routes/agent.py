"""Agent endpoint that routes NL requests to GIS capability modules."""

import json
from typing import Iterator

from fastapi import APIRouter
from fastapi.responses import StreamingResponse

from aigis_agent.schemas.agent import (
    AgentAssistRequest,
    AgentAssistResponse,
    AgentChatRequest,
    AgentChatResponse,
    AgentQueryRequest,
    AgentQueryResponse,
)
from aigis_agent.services.agent_service import AgentService

router = APIRouter(prefix="/agent", tags=["agent"])
_service = AgentService()


@router.post("/query", response_model=AgentQueryResponse)
def query_agent(payload: AgentQueryRequest) -> AgentQueryResponse:
    """Infer user intent and return recommended API entrypoint."""
    decision = _service.analyze_query(payload.query)
    return AgentQueryResponse(
        intent=decision.intent,
        message=decision.message,
        suggested_endpoint=decision.suggested_endpoint,
    )


@router.post("/assist", response_model=AgentAssistResponse)
def assist_agent(payload: AgentAssistRequest) -> AgentAssistResponse:
    """Plan from natural language and execute the selected concrete tool."""
    outcome = _service.assist(
        query=payload.query,
        city_hint=payload.city_hint,
        limit=payload.limit,
        allow_city_switch=payload.allow_city_switch,
    )
    return AgentAssistResponse(
        message=outcome.message,
        selected_tool=outcome.selected_tool,
        tasks=outcome.tasks,
        tool_result=outcome.tool_result,
        provider=_service.get_provider_name(),
        model=_service.get_model_name(),
        fallback_used=outcome.fallback_used,
    )


@router.post("/assist/stream")
def assist_agent_stream(payload: AgentAssistRequest) -> StreamingResponse:
    """Stream planning and tool execution events for frontend chat bubble."""

    def generate() -> Iterator[str]:
        for event in _service.assist_stream(
            query=payload.query,
            city_hint=payload.city_hint,
            limit=payload.limit,
            allow_city_switch=payload.allow_city_switch,
        ):
            yield json.dumps(event, ensure_ascii=False) + "\n"

    return StreamingResponse(generate(), media_type="application/x-ndjson")


@router.post("/chat", response_model=AgentChatResponse)
def chat_agent(payload: AgentChatRequest) -> AgentChatResponse:
    """Chat with configured LLM provider (DeepSeek/OpenAI-compatible)."""
    history = [
        {"role": item.role, "content": item.content}
        for item in payload.history
    ]
    reply = _service.chat(payload.message, history=history)
    return AgentChatResponse(
        reply=reply,
        provider=_service.get_provider_name(),
        model=_service.get_model_name(),
    )
