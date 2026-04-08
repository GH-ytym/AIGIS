"""Startup greeting provider for the center search input."""

from __future__ import annotations

import random
from typing import Any

from langchain_openai import ChatOpenAI

from aigis_agent.core.config import settings

COMMON_GREETINGS = [
    "想去哪里？",
    "在忙什么？",
    "附近有什么好去处？",
    "想找点好吃的还是好玩的？",
    "下一站准备去哪儿？",
    "要不要看看周边热点？",
    "今天想怎么安排？",
]

_startup_greeting = random.choice(COMMON_GREETINGS)


def initialize_startup_greeting() -> str:
    """Generate one greeting on server startup for the current process."""
    global _startup_greeting

    seed_greeting = random.choice(COMMON_GREETINGS)
    _startup_greeting = _generate_greeting_with_llm(seed_greeting) or seed_greeting
    return _startup_greeting


def get_startup_greeting() -> str:
    """Return the greeting selected/generated at startup."""
    return _startup_greeting


def _generate_greeting_with_llm(seed_greeting: str) -> str | None:
    """Use OpenAI-compatible endpoint (e.g. DeepSeek) to generate one greeting."""
    if not settings.openai_api_key:
        return None

    llm = ChatOpenAI(
        api_key=settings.openai_api_key,
        model=settings.openai_model,
        base_url=settings.openai_base_url or None,
        temperature=0.7,
        timeout=settings.llm_timeout_s,
    )

    prompt = (
        "你是地图应用文案助手。"
        "请生成 1 句中文招呼语，用于搜索框 placeholder。"
        f"参考示例：{seed_greeting}。"
        "要求：8-16 字，口语自然，不要引号，不要换行，只输出这 1 句话。"
    )

    try:
        result = llm.invoke(prompt)
    except Exception:
        return None

    return _normalize_greeting(_extract_text(result))


def _extract_text(result: Any) -> str:
    content = getattr(result, "content", "")
    if isinstance(content, str):
        return content

    if isinstance(content, list):
        chunks: list[str] = []
        for item in content:
            if isinstance(item, dict) and isinstance(item.get("text"), str):
                chunks.append(item["text"])
        return "".join(chunks)

    return ""


def _normalize_greeting(text: str) -> str | None:
    cleaned = str(text or "").replace("\n", "").replace("\r", "").strip()
    cleaned = cleaned.strip(" \"'“”")

    if not cleaned:
        return None

    if len(cleaned) > 24:
        cleaned = cleaned[:24].rstrip("，。！？!? ")

    return cleaned or None
