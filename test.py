import asyncio
import os
from typing import Any

from langchain_mcp_adapters.client import MultiServerMCPClient


async def _probe_geocoder_if_exists(tools: list[Any]) -> None:
    """Call geocoder once when available to verify end-to-end tool invocation."""
    if not tools:
        print("未发现可用工具，跳过工具调用验证。")
        return

    geocoder_tool = next((tool for tool in tools if tool.name == "geocoder"), None)
    if geocoder_tool is None:
        print("未找到 geocoder 工具，跳过 geocoder 调用验证。")
        return

    test_address = os.getenv("TENCENT_MCP_TEST_ADDRESS", "深圳市南山区腾讯滨海大厦")
    print(f"尝试调用 geocoder，测试地址: {test_address}")

    # LangChain tool may expose sync invoke or async ainvoke depending on adapter version.
    if hasattr(geocoder_tool, "ainvoke"):
        result = await geocoder_tool.ainvoke({"address": test_address})
    else:
        result = geocoder_tool.invoke({"address": test_address})

    preview = str(result)
    if len(preview) > 500:
        preview = preview[:500] + " ..."
    print("geocoder 调用成功，返回预览:")
    print(preview)


async def test_map_connection() -> None:
    default_key = "YGVBZ-WVIK3-UDZ3R-RZFJC-YV7HT-PFB4I"
    tencent_key = os.getenv("TENCENT_MAP_KEY", default_key).strip()

    if not tencent_key:
        print("未提供腾讯地图 Key，请设置环境变量 TENCENT_MAP_KEY。")
        return

    candidates = [
        (
            "streamable_http",
            f"https://mcp.map.qq.com/mcp?key={tencent_key}&format=1",
        ),
        (
            "sse",
            f"https://mcp.map.qq.com/sse?key={tencent_key}&format=1",
        ),
    ]

    errors: list[str] = []
    for transport, mcp_url in candidates:
        print(f"\n正在尝试连接: {mcp_url}")
        try:
            connections = {
                "qq_map": {
                    "transport": transport,
                    "url": mcp_url,
                }
            }

            client = MultiServerMCPClient(connections)
            tools = await client.get_tools(server_name="qq_map")

            print("--- 成功连接腾讯地图 MCP ---")
            print(f"已加载工具数量: {len(tools)}")
            for tool in tools:
                print(f" - {tool.name}")

            await _probe_geocoder_if_exists(tools)
            print("\n结论: 腾讯地图 MCP 连接正常。")
            return
        except Exception as exc:
            message = f"{mcp_url} -> {exc}"
            errors.append(message)
            print(f"连接失败: {exc}")

    print("\n结论: 腾讯地图 MCP 未连接成功。")
    print("失败详情:")
    for item in errors:
        print(f" - {item}")
    print("提示：请检查网络、Key 权限配额和依赖版本（mcp/langchain-mcp-adapters）。")


if __name__ == "__main__":
    asyncio.run(test_map_connection())