from __future__ import annotations

import os

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

	async def get_available_tool_names(self) -> tuple[list[str], str | None]:
		"""Discover available MCP tool names without executing any tool."""
		if not HAS_MCP_ADAPTER:
			return [], "langchain_mcp_adapters 未安装或导入失败。"

		if not self.key:
			return [], "未配置 TENCENT_MCP_KEY。"

		try:
			client = MultiServerMCPClient(
				{
					"qq_map": {
						"transport": self.transport,
						"url": self.url,
					}
				}
			)
			tools = await client.get_tools(server_name="qq_map")
			tool_names: list[str] = []
			for tool in tools:
				name = str(getattr(tool, "name", "")).strip()
				if name:
					tool_names.append(name)
			tool_names = sorted(set(tool_names))

			return tool_names, None
		except Exception as exc:
			return [], f"获取 MCP 工具列表失败: {exc}"
