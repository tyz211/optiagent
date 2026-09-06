from __future__ import annotations

import os
import math
from pathlib import Path
from typing import Any


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def allowed_roots() -> list[Path]:
    """返回 MCP 可读取的目录，默认仅限当前项目。"""

    configured = os.getenv("OPTIAGENT_MCP_ALLOWED_ROOTS", "").strip()
    if not configured:
        return [PROJECT_ROOT]
    roots = [Path(item).expanduser().resolve() for item in configured.split(os.pathsep) if item.strip()]
    return roots or [PROJECT_ROOT]


def resolve_readable_path(path: str) -> Path:
    """解析并校验读取路径，防止 MCP 越界读取任意本地文件。"""

    candidate = Path(path).expanduser()
    if not candidate.is_absolute():
        candidate = PROJECT_ROOT / candidate
    candidate = candidate.resolve()
    if not any(candidate == root or root in candidate.parents for root in allowed_roots()):
        roots = "、".join(str(root) for root in allowed_roots())
        raise ValueError(f"文件不在 MCP 允许的目录中：{roots}")
    if not candidate.exists() or not candidate.is_file():
        raise FileNotFoundError(f"文件不存在：{candidate}")
    return candidate


def json_safe(value: Any) -> Any:
    """将 pandas/numpy 标量等值转换为 JSON 可序列化类型。"""

    if isinstance(value, float) and not math.isfinite(value):
        return None
    if value is None or isinstance(value, (str, int, float, bool)):
        return value
    if isinstance(value, dict):
        return {str(key): json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple, set)):
        return [json_safe(item) for item in value]
    if hasattr(value, "item"):
        try:
            return value.item()
        except (TypeError, ValueError):
            pass
    return str(value)


def run_server(server: Any, default_port: int) -> None:
    """按环境变量启动 stdio 或 Streamable HTTP 服务。"""

    transport = os.getenv("OPTIAGENT_MCP_TRANSPORT", "stdio").strip().lower()
    if transport == "stdio":
        try:
            server.run(transport="stdio")
        except KeyboardInterrupt:
            # 交互式停止属于正常关闭，不向用户输出异常堆栈。
            pass
        return
    if transport != "streamable-http":
        raise ValueError("OPTIAGENT_MCP_TRANSPORT 仅支持 stdio 或 streamable-http。")
    # MCP 1.x 通过 FastMCP settings 控制 HTTP 参数，这样也便于不同 1.x 小版兼容。
    server.settings.host = os.getenv("OPTIAGENT_MCP_HOST", "127.0.0.1")
    server.settings.port = int(os.getenv("OPTIAGENT_MCP_PORT", str(default_port)))
    server.settings.streamable_http_path = "/mcp"
    server.settings.json_response = True
    server.settings.stateless_http = True
    try:
        server.run(transport="streamable-http")
    except KeyboardInterrupt:
        # Streamable HTTP 服务也应在 Ctrl+C 时干净退出。
        pass
