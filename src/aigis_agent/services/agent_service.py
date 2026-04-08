"""LangChain-based GIS agent service with robust fallback behavior."""

from __future__ import annotations

import json
from typing import Any, Iterator

from langchain.agents import create_agent
from langchain_core.messages import AIMessage, HumanMessage, SystemMessage
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field

from aigis_agent.core.config import settings
from aigis_agent.schemas.agent import AgentTaskStep
from aigis_agent.services.agent_tools import (
    get_execution_tool_names,
    get_gis_tools,
    invoke_execution_tool,
)


class AgentDecision(BaseModel):
    """Structured result returned by the NL GIS agent."""

    intent: str
    message: str
    suggested_endpoint: str


class AssistPlan(BaseModel):
    """Planner output before tool execution."""

    tasks: list[AgentTaskStep] = Field(default_factory=list)
    summary: str = ""


class AgentAssistOutcome(BaseModel):
    """Final outcome returned by assistant endpoint."""

    message: str
    selected_tool: str
    tasks: list[AgentTaskStep] = Field(default_factory=list)
    tool_result: dict[str, Any] = Field(default_factory=dict)
    fallback_used: bool = False


class AgentService:
    """LangChain-backed intent router for first-wave GIS capabilities."""

    def __init__(self) -> None:
        self._chat_model = self._build_chat_model()
        self._agent = self._build_langchain_agent()
        self._execution_tool_names = get_execution_tool_names()

    def _build_chat_model(self) -> ChatOpenAI | None:
        """Create chat model using OpenAI-compatible endpoint (e.g. DeepSeek)."""
        if not settings.openai_api_key:
            return None

        return ChatOpenAI(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            base_url=settings.openai_base_url or None,
            temperature=settings.llm_temperature,
            timeout=settings.llm_timeout_s,
        )

    def _build_langchain_agent(self) -> Any | None:
        """Create a real LangChain agent if LLM credentials are configured."""
        llm = self._chat_model
        if llm is None:
            return None

        system_prompt = (
            "You are an expert GIS assistant. "
            "Use tools when needed to validate your decision. "
            "Classify user intent into one of: "
            "poi_search, route_analysis, service_area, site_selection, unknown. "
            "Then return a concise Chinese message and matching endpoint. "
            "Endpoint mapping: "
            "poi_search->/v1/poi/search, "
            "route_analysis->/v1/routing/route, "
            "service_area->/v1/service-area/isochrone, "
            "site_selection->/v1/site-selection/score, "
            "unknown->/v1/agent/query."
        )

        return create_agent(
            model=llm,
            tools=get_gis_tools(),
            system_prompt=system_prompt,
            response_format=AgentDecision,
            name="aigis-gis-agent",
        )

    def chat(self, message: str, history: list[dict[str, str]] | None = None) -> str:
        """Chat with configured OpenAI-compatible model for map assistant UI."""
        question = str(message or "").strip()
        if not question:
            return "请输入你想聊的内容。"

        if self._chat_model is None:
            return "未检测到 AIGIS_OPENAI_API_KEY，当前无法连接 DeepSeek。"

        chat_messages: list[Any] = [
            SystemMessage(
                content=(
                    "你是 AIGIS 地图助手。"
                    "请使用简洁中文回答，优先给出可执行建议，长度尽量控制在 120 字以内。"
                )
            )
        ]

        for item in (history or [])[-8:]:
            role = str(item.get("role") or "").strip().lower()
            content = str(item.get("content") or "").strip()
            if not content:
                continue
            if role == "assistant":
                chat_messages.append(AIMessage(content=content))
                continue
            chat_messages.append(HumanMessage(content=content))

        chat_messages.append(HumanMessage(content=question))

        try:
            result = self._chat_model.invoke(chat_messages)
        except Exception as exc:
            return f"对话模型调用失败 ({exc.__class__.__name__})，请稍后重试。"

        text = self._extract_message_text(result)
        if text:
            return text
        return "我暂时没有生成有效回复，换个问法试试。"

    @staticmethod
    def get_provider_name() -> str:
        """Infer provider name from configured model/base URL."""
        marker = f"{settings.openai_base_url or ''} {settings.openai_model}".lower()
        return "deepseek" if "deepseek" in marker else "openai-compatible"

    @staticmethod
    def get_model_name() -> str:
        """Return configured model name."""
        return settings.openai_model

    def analyze_query(self, query: str) -> AgentDecision:
        """Infer GIS intent with LangChain and fallback safely if unavailable."""
        if self._agent is None:
            return self._fallback_decision(
                query,
                "未检测到 AIGIS_OPENAI_API_KEY，已使用本地规则路由。",
            )

        try:
            result = self._agent.invoke(
                {"messages": [{"role": "user", "content": query}]}
            )
            decision = self._parse_agent_result(result)
            if decision is not None:
                return decision
            return self._fallback_decision(
                query,
                "LangChain 未返回可解析结构，已自动降级为规则路由。",
            )
        except Exception as exc:
            return self._fallback_decision(
                query,
                f"LangChain 调用失败 ({exc.__class__.__name__})，已自动降级为规则路由。",
            )

    def assist(
        self,
        query: str,
        city_hint: str | None = None,
        limit: int = 20,
        allow_city_switch: bool = False,
    ) -> AgentAssistOutcome:
        """Plan tool steps from NL input and execute first concrete tool."""
        normalized_query = str(query or "").strip()
        if not normalized_query:
            return AgentAssistOutcome(
                message="请输入你想查询的内容。",
                selected_tool="search_poi",
                tasks=[
                    AgentTaskStep(
                        step_id=1,
                        tool_name="search_poi",
                        reason="空输入默认走 POI 查询。",
                        args={
                            "query": "",
                            "city_hint": city_hint or "",
                            "limit": max(1, min(limit, 50)),
                            "allow_city_switch": allow_city_switch,
                        },
                    )
                ],
                tool_result={
                    "error": "Empty query",
                    "count": 0,
                    "items": [],
                },
                fallback_used=True,
            )

        plan, planner_fallback = self._plan_assist_tasks(
            query=normalized_query,
            city_hint=city_hint,
            limit=limit,
            allow_city_switch=allow_city_switch,
        )
        tasks = self._normalize_tasks(
            plan.tasks,
            query=normalized_query,
            city_hint=city_hint,
            limit=limit,
            allow_city_switch=allow_city_switch,
        )

        if not tasks:
            tasks = [
                self._default_poi_task(
                    query=normalized_query,
                    city_hint=city_hint,
                    limit=limit,
                    allow_city_switch=allow_city_switch,
                    step_id=1,
                    reason="未产出有效任务，已降级到默认 POI 工具。",
                )
            ]
            planner_fallback = True

        selected_tool = tasks[0].tool_name
        fallback_used = planner_fallback

        task_to_run = tasks[0]
        if selected_tool not in self._execution_tool_names:
            fallback_used = True
            task_to_run = self._default_poi_task(
                query=normalized_query,
                city_hint=city_hint,
                limit=limit,
                allow_city_switch=allow_city_switch,
                step_id=task_to_run.step_id,
                reason=f"工具 {selected_tool} 暂未开放执行，自动降级到 search_poi。",
            )
            tasks[0] = task_to_run
            selected_tool = task_to_run.tool_name

        tool_result = self._execute_tool_task(task_to_run)
        message = self._build_assist_message(selected_tool, tool_result, fallback_used)

        return AgentAssistOutcome(
            message=message,
            selected_tool=selected_tool,
            tasks=tasks,
            tool_result=tool_result,
            fallback_used=fallback_used,
        )

    def assist_stream(
        self,
        query: str,
        city_hint: str | None = None,
        limit: int = 20,
        allow_city_switch: bool = False,
    ) -> Iterator[dict[str, Any]]:
        """Stream planner and tool execution events for frontend bubble rendering."""
        normalized_query = str(query or "").strip()
        safe_limit = max(1, min(limit, 50))

        yield {
            "event": "status",
            "message": "已接收请求，开始解析用户输入。",
        }

        if not normalized_query:
            outcome = AgentAssistOutcome(
                message="请输入你想查询的内容。",
                selected_tool="search_poi",
                tasks=[
                    AgentTaskStep(
                        step_id=1,
                        tool_name="search_poi",
                        reason="空输入默认走 POI 查询。",
                        args={
                            "query": "",
                            "city_hint": city_hint or "",
                            "limit": safe_limit,
                            "allow_city_switch": allow_city_switch,
                        },
                    )
                ],
                tool_result={
                    "error": "Empty query",
                    "count": 0,
                    "items": [],
                },
                fallback_used=True,
            )
            yield {
                "event": "planner_result",
                "summary": "empty-query",
                "tasks": [task.model_dump() for task in outcome.tasks],
                "fallback_used": True,
            }
            yield {
                "event": "done",
                "outcome": outcome.model_dump(),
            }
            return

        yield {
            "event": "planner_start",
            "message": "正在调用 DeepSeek 生成任务列表。",
        }
        plan, planner_fallback = self._plan_assist_tasks(
            query=normalized_query,
            city_hint=city_hint,
            limit=safe_limit,
            allow_city_switch=allow_city_switch,
        )
        tasks = self._normalize_tasks(
            plan.tasks,
            query=normalized_query,
            city_hint=city_hint,
            limit=safe_limit,
            allow_city_switch=allow_city_switch,
        )

        if not tasks:
            tasks = [
                self._default_poi_task(
                    query=normalized_query,
                    city_hint=city_hint,
                    limit=safe_limit,
                    allow_city_switch=allow_city_switch,
                    step_id=1,
                    reason="未产出有效任务，已降级到默认 POI 工具。",
                )
            ]
            planner_fallback = True

        yield {
            "event": "planner_result",
            "summary": plan.summary,
            "tasks": [task.model_dump() for task in tasks],
            "fallback_used": planner_fallback,
        }

        selected_tool = tasks[0].tool_name
        fallback_used = planner_fallback
        task_to_run = tasks[0]

        if selected_tool not in self._execution_tool_names:
            fallback_used = True
            yield {
                "event": "status",
                "message": f"工具 {selected_tool} 暂未开放执行，自动降级到 search_poi。",
            }
            task_to_run = self._default_poi_task(
                query=normalized_query,
                city_hint=city_hint,
                limit=safe_limit,
                allow_city_switch=allow_city_switch,
                step_id=task_to_run.step_id,
                reason=f"工具 {selected_tool} 暂未开放执行，自动降级到 search_poi。",
            )
            tasks[0] = task_to_run
            selected_tool = task_to_run.tool_name

        yield {
            "event": "tool_start",
            "tool_name": task_to_run.tool_name,
            "args": task_to_run.args,
        }
        tool_result = self._execute_tool_task(task_to_run)
        yield {
            "event": "tool_result",
            "tool_name": task_to_run.tool_name,
            "result": self._summarize_tool_result(task_to_run.tool_name, tool_result),
        }

        message = self._build_assist_message(selected_tool, tool_result, fallback_used)
        outcome = AgentAssistOutcome(
            message=message,
            selected_tool=selected_tool,
            tasks=tasks,
            tool_result=tool_result,
            fallback_used=fallback_used,
        )
        yield {
            "event": "done",
            "outcome": outcome.model_dump(),
        }

    def _parse_agent_result(self, result: dict[str, Any]) -> AgentDecision | None:
        """Parse structured output first; fallback to JSON text in last message."""
        structured = result.get("structured_response")
        if structured is not None:
            try:
                return AgentDecision.model_validate(structured)
            except Exception:
                pass

        text = self._extract_last_text(result.get("messages", []))
        if not text:
            return None

        try:
            data = json.loads(text)
            return AgentDecision.model_validate(data)
        except Exception:
            return None

    @staticmethod
    def _extract_last_text(messages: list[Any]) -> str | None:
        """Get text content from the last LangChain message."""
        if not messages:
            return None

        last = messages[-1]
        content = getattr(last, "content", None)
        if isinstance(content, str):
            return content

        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"])
            if chunks:
                return "\n".join(chunks)

        return None

    @staticmethod
    def _extract_message_text(message: Any) -> str | None:
        """Get normalized text content from a single model response message."""
        content = getattr(message, "content", None)
        if isinstance(content, str):
            text = content.strip()
            return text or None

        if isinstance(content, list):
            chunks: list[str] = []
            for item in content:
                if isinstance(item, dict) and isinstance(item.get("text"), str):
                    chunks.append(item["text"].strip())
            joined = "\n".join(chunk for chunk in chunks if chunk)
            return joined or None

        return None

    def _fallback_decision(self, query: str, message: str) -> AgentDecision:
        """Fallback to deterministic keyword-based intent routing."""
        intent, endpoint = self._infer_intent_keyword(query)
        return AgentDecision(
            intent=intent,
            message=message,
            suggested_endpoint=endpoint,
        )

    def _plan_assist_tasks(
        self,
        query: str,
        city_hint: str | None,
        limit: int,
        allow_city_switch: bool,
    ) -> tuple[AssistPlan, bool]:
        """Use LLM to generate task plan; fallback to deterministic POI task."""
        fallback_plan = AssistPlan(
            tasks=[
                self._default_poi_task(
                    query=query,
                    city_hint=city_hint,
                    limit=limit,
                    allow_city_switch=allow_city_switch,
                    step_id=1,
                    reason="默认使用 POI 检索作为首个可执行工具。",
                )
            ],
            summary="fallback-poi",
        )

        if self._chat_model is None:
            return fallback_plan, True

        planner_prompt = (
            "你是 GIS Agent 规划器。"
            "请把用户输入拆成任务步骤并返回结构化 JSON。"
            "tool_name 可选：search_poi, analyze_route, build_service_area, score_sites。"
            "当前仅 search_poi 是可执行工具，其他会被自动降级。"
            "如果信息不足，优先先用 search_poi 解析地点。"
        )
        payload = {
            "query": query,
            "city_hint": city_hint,
            "limit": max(1, min(limit, 50)),
            "allow_city_switch": allow_city_switch,
        }

        try:
            planner = self._chat_model.with_structured_output(AssistPlan)
            plan = planner.invoke(
                [
                    {"role": "system", "content": planner_prompt},
                    {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
                ]
            )
            if isinstance(plan, AssistPlan):
                return plan, False

            validated = AssistPlan.model_validate(plan)
            return validated, False
        except Exception:
            return fallback_plan, True

    def _normalize_tasks(
        self,
        tasks: list[AgentTaskStep],
        query: str,
        city_hint: str | None,
        limit: int,
        allow_city_switch: bool,
    ) -> list[AgentTaskStep]:
        """Normalize planner tasks and enforce required defaults."""
        normalized: list[AgentTaskStep] = []
        for idx, task in enumerate(tasks, start=1):
            tool_name = str(task.tool_name or "").strip() or "search_poi"
            args = dict(task.args or {})
            reason = str(task.reason or "").strip()

            if tool_name == "search_poi":
                args["query"] = str(args.get("query") or query)
                args["city_hint"] = str(args.get("city_hint") or city_hint or "")
                args["limit"] = max(1, min(int(args.get("limit") or limit), 50))
                args["allow_city_switch"] = bool(args.get("allow_city_switch", allow_city_switch))

            normalized.append(
                AgentTaskStep(
                    step_id=idx,
                    tool_name=tool_name,
                    reason=reason,
                    args=args,
                )
            )

        return normalized

    @staticmethod
    def _default_poi_task(
        query: str,
        city_hint: str | None,
        limit: int,
        allow_city_switch: bool,
        step_id: int,
        reason: str,
    ) -> AgentTaskStep:
        """Create default search_poi task with normalized args."""
        return AgentTaskStep(
            step_id=step_id,
            tool_name="search_poi",
            reason=reason,
            args={
                "query": query,
                "city_hint": city_hint or "",
                "limit": max(1, min(limit, 50)),
                "allow_city_switch": allow_city_switch,
            },
        )

    @staticmethod
    def _execute_tool_task(task: AgentTaskStep) -> dict[str, Any]:
        """Execute selected task and return normalized dict payload."""
        try:
            return invoke_execution_tool(task.tool_name, task.args)
        except Exception as exc:
            return {
                "error": f"Tool execution failed ({task.tool_name}): {exc}",
                "count": 0,
                "items": [],
            }

    @staticmethod
    def _build_assist_message(tool_name: str, tool_result: dict[str, Any], fallback_used: bool) -> str:
        """Generate concise assistant message from tool execution result."""
        prefix = "已按你的输入执行助手任务。"
        if fallback_used:
            prefix = "已执行助手任务（含降级策略）。"

        if tool_result.get("error"):
            return f"{prefix} 执行失败：{tool_result['error']}"

        if tool_name == "search_poi":
            count = int(tool_result.get("count") or 0)
            if count > 0:
                return f"{prefix} 已找到 {count} 个地点候选。"
            return f"{prefix} 暂未找到匹配地点，建议补充更具体关键词。"

        return f"{prefix} 已完成 {tool_name} 调用。"

    @staticmethod
    def _summarize_tool_result(tool_name: str, tool_result: dict[str, Any]) -> dict[str, Any]:
        """Summarize tool result for streaming display to avoid oversized payload."""
        if tool_result.get("error"):
            return {
                "status": "error",
                "error": str(tool_result.get("error")),
            }

        if tool_name == "search_poi":
            items = tool_result.get("items") if isinstance(tool_result.get("items"), list) else []
            top_items = []
            for item in items[:3]:
                if not isinstance(item, dict):
                    continue
                top_items.append(
                    {
                        "name": item.get("name"),
                        "address": item.get("address"),
                    }
                )

            return {
                "status": "ok",
                "count": int(tool_result.get("count") or 0),
                "top_items": top_items,
            }

        return {
            "status": "ok",
        }

    @staticmethod
    def _infer_intent_keyword(text: str) -> tuple[str, str]:
        """Infer intent and endpoint from keywords when LLM is unavailable."""
        q = text.lower()
        if any(k in q for k in ["poi", "地址", "附近", "search", "geocode"]):
            return "poi_search", "/v1/poi/search"
        if any(k in q for k in ["路线", "route", "路径", "导航"]):
            return "route_analysis", "/v1/routing/route"
        if any(k in q for k in ["服务范围", "isochrone", "可达", "覆盖"]):
            return "service_area", "/v1/service-area/isochrone"
        if any(k in q for k in ["选址", "site", "location analysis"]):
            return "site_selection", "/v1/site-selection/score"
        return "unknown", "/v1/agent/query"
