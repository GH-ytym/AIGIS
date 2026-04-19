from __future__ import annotations

import os
from typing import Any

try:
	from langchain_mcp_adapters.client import MultiServerMCPClient

	HAS_MCP_ADAPTER = True
except Exception:
	HAS_MCP_ADAPTER = False


DEFAULT_TENCENT_MCP_KEY = ""


class TencentMCPGateway:
	"""Tencent MCP gateway wrapper for tool discovery."""

	def __init__(self) -> None:
		self.key = os.getenv("TENCENT_MCP_KEY", DEFAULT_TENCENT_MCP_KEY).strip()
		self.format = os.getenv("TENCENT_MCP_FORMAT", "1").strip() or "1"
		self.transport = os.getenv("TENCENT_MCP_TRANSPORT", "streamable_http").strip() or "streamable_http"
		self.url = f"https://mcp.map.qq.com/mcp?key={self.key}&format={self.format}"

	def _build_client(self) -> Any | None:
		if not HAS_MCP_ADAPTER:
			return None

		return MultiServerMCPClient(
			{
				"qq_map": {
					"transport": self.transport,
					"url": self.url,
				}
			}
		)

	async def _load_tools(self) -> tuple[dict[str, Any], str | None]:
		"""Load MCP tools and return a name->tool mapping."""
		if not HAS_MCP_ADAPTER:
			return {}, "langchain_mcp_adapters 未安装或导入失败。"

		if not self.key:
			return {}, "未配置 TENCENT_MCP_KEY。"

		client = self._build_client()
		if client is None:
			return {}, "MCP 客户端初始化失败。"

		try:
			tools = await client.get_tools(server_name="qq_map")
			tool_map: dict[str, Any] = {}
			for tool in tools:
				name = str(getattr(tool, "name", "")).strip()
				if name:
					tool_map[name] = tool
			return tool_map, None
		except Exception as exc:
			return {}, f"获取 MCP 工具列表失败: {exc}"

	async def get_available_tool_names(self) -> tuple[list[str], str | None]:
		"""Discover available MCP tool names without executing any tool."""
		tool_map, error = await self._load_tools()
		if error is not None:
			return [], error

		tool_names = sorted(tool_map.keys())
		return tool_names, None

	async def invoke_tool(self, tool_name: str, tool_input: dict[str, Any]) -> tuple[Any, str | None]:
		"""Invoke one MCP tool by name with given input payload."""
		name = str(tool_name or "").strip()
		if not name:
			return None, "工具名为空，无法调用。"

		tool_map, error = await self._load_tools()
		if error is not None:
			return None, error

		tool = tool_map.get(name)
		if tool is None:
			return None, f"工具不存在或当前不可用: {name}"

		payload = tool_input if isinstance(tool_input, dict) else {}
		try:
			if hasattr(tool, "ainvoke"):
				result = await tool.ainvoke(payload)
			else:
				result = tool.invoke(payload)
			return result, None
		except Exception as exc:
			return None, f"调用 MCP 工具失败({name}): {exc}"
