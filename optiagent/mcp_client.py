from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass, field
import json
from pathlib import Path
import sys
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parent.parent


@dataclass(frozen=True)
class MCPToolLoadResult:
    """MCP 工具发现结果，支持部分服务降级。"""

    tools: list[Any] = field(default_factory=list)
    connected_servers: list[str] = field(default_factory=list)
    errors: dict[str, str] = field(default_factory=dict)

    @property
    def status(self) -> str:
        if not self.connected_servers and self.errors:
            details = "；".join(f"{name}: {message}" for name, message in self.errors.items())
            return f"MCP 服务全部不可用：{details}"
        connected = "、".join(self.connected_servers) or "无"
        if self.errors:
            failed = "、".join(self.errors)
            return f"MCP 已连接：{connected}；降级：{failed}。"
        return f"MCP 已连接：{connected}，共发现 {len(self.tools)} 个工具。"


def builtin_mcp_config() -> dict[str, dict[str, Any]]:
    """生成与当前 Python 环境匹配的三个内置 stdio 服务。"""

    base = {
        "transport": "stdio",
        "command": sys.executable,
        "cwd": str(PROJECT_ROOT),
    }
    return {
        "document": {**base, "args": ["-m", "optiagent.mcp_servers.document_server"]},
        "data": {**base, "args": ["-m", "optiagent.mcp_servers.data_server"]},
        "solver": {**base, "args": ["-m", "optiagent.mcp_servers.solver_server"]},
    }


def parse_mcp_config(config_json: str = "") -> dict[str, dict[str, Any]]:
    """解析新旧两种配置形式，默认启用项目内置 MCP。"""

    if not config_json.strip():
        return builtin_mcp_config()
    try:
        payload = json.loads(config_json)
    except json.JSONDecodeError as exc:
        raise ValueError(f"MCP JSON 解析失败：{exc}") from exc
    if not isinstance(payload, dict):
        raise ValueError("MCP 配置顶层必须是 JSON 对象。")

    if "servers" in payload:
        servers = payload.get("servers")
        if not isinstance(servers, dict):
            raise ValueError("MCP servers 必须是 JSON 对象。")
        config = builtin_mcp_config() if payload.get("include_builtin", True) else {}
        config.update(servers)
    else:
        # 兼容项目原有的 MultiServerMCPClient 裸配置。
        config = {**builtin_mcp_config(), **payload}
    _validate_connections(config)
    return config


def load_mcp_tools_sync(config_json: str = "") -> MCPToolLoadResult:
    """在普通函数和已运行事件循环中都能安全执行 MCP 发现。"""

    try:
        config = parse_mcp_config(config_json)
    except ValueError as exc:
        return MCPToolLoadResult(errors={"config": str(exc)})

    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(_load_mcp_tools(config))

    # FastAPI 的事件循环不允许嵌套 asyncio.run，因此在独立线程完成发现。
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="optiagent-mcp") as executor:
        return executor.submit(asyncio.run, _load_mcp_tools(config)).result()


async def _load_mcp_tools(config: dict[str, dict[str, Any]]) -> MCPToolLoadResult:
    from langchain_mcp_adapters.client import MultiServerMCPClient

    async def load_one(name: str, connection: dict[str, Any]):
        client = MultiServerMCPClient({name: connection}, tool_name_prefix=True, handle_tool_errors=True)
        return await client.get_tools(server_name=name)

    names = list(config)
    loaded = await asyncio.gather(
        *(load_one(name, config[name]) for name in names),
        return_exceptions=True,
    )
    tools: list[Any] = []
    connected: list[str] = []
    errors: dict[str, str] = {}
    for name, result in zip(names, loaded, strict=True):
        if isinstance(result, BaseException):
            errors[name] = f"{type(result).__name__}: {result}"
            continue
        connected.append(name)
        tools.extend(result)
    return MCPToolLoadResult(tools=tools, connected_servers=connected, errors=errors)


def _validate_connections(config: dict[str, Any]) -> None:
    if not config:
        raise ValueError("MCP 配置中没有服务。")
    supported = {"stdio", "sse", "streamable_http", "websocket"}
    for name, connection in config.items():
        if not isinstance(connection, dict):
            raise ValueError(f"MCP 服务 {name} 的配置必须是对象。")
        transport = str(connection.get("transport", ""))
        if transport not in supported:
            raise ValueError(f"MCP 服务 {name} 的 transport 无效：{transport}")
        if transport == "stdio" and not connection.get("command"):
            raise ValueError(f"stdio MCP 服务 {name} 缺少 command。")
        if transport != "stdio" and not connection.get("url"):
            raise ValueError(f"网络 MCP 服务 {name} 缺少 url。")
