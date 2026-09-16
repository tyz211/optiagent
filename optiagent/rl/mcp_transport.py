from __future__ import annotations

import asyncio
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
import sys
from time import perf_counter
from typing import Literal
from uuid import uuid4

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client
from pydantic import BaseModel, Field


PROJECT_ROOT = Path(__file__).resolve().parents[2]
TransportFaultKind = Literal["normal", "timeout", "disconnect", "invalid_return"]
TRANSPORT_FAULT_KINDS: tuple[TransportFaultKind, ...] = (
    "normal",
    "timeout",
    "disconnect",
    "invalid_return",
)


class MCPTransportTrace(BaseModel):
    """一次真实 MCP stdio 调用的安全、可持久化观测。"""

    schema_version: str = "1.0"
    trace_id: str
    fault: TransportFaultKind
    success: bool
    response_schema_valid: bool
    elapsed_ms: float = Field(ge=0.0)
    error_type: str | None = None


def collect_mcp_transport_traces(
    *,
    timeout_ms: int = 80,
    delay_ms: int = 240,
    nonce_prefix: str = "transport",
) -> list[MCPTransportTrace]:
    """在同步或已有事件循环的调用点安全采集四类 MCP transport 轨迹。"""

    coroutine = _collect_all(timeout_ms=timeout_ms, delay_ms=delay_ms, nonce_prefix=nonce_prefix)
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coroutine)
    with ThreadPoolExecutor(max_workers=1, thread_name_prefix="optiagent-mcp-fault") as executor:
        return executor.submit(asyncio.run, coroutine).result()


async def _collect_all(
    *,
    timeout_ms: int,
    delay_ms: int,
    nonce_prefix: str,
) -> list[MCPTransportTrace]:
    traces = []
    for fault in TRANSPORT_FAULT_KINDS:
        traces.append(
            await _collect_one(
                fault,
                timeout_ms=timeout_ms,
                delay_ms=delay_ms,
                nonce=f"{nonce_prefix}-{fault}-{uuid4().hex[:8]}",
            )
        )
    return traces


async def _collect_one(
    fault: TransportFaultKind,
    *,
    timeout_ms: int,
    delay_ms: int,
    nonce: str,
) -> MCPTransportTrace:
    server = StdioServerParameters(
        command=sys.executable,
        args=["-m", "optiagent.mcp_servers.fault_probe_server"],
        cwd=str(PROJECT_ROOT),
    )
    started = perf_counter()
    response_schema_valid = False
    success = False
    error_type: str | None = None
    try:
        async with stdio_client(server) as (read_stream, write_stream):
            async with ClientSession(read_stream, write_stream) as session:
                await session.initialize()
                call = session.call_tool(
                    "mcp_fault_probe",
                    {
                        "mode": fault,
                        "delay_ms": delay_ms if fault == "timeout" else 0,
                        "nonce": nonce,
                    },
                )
                result = await asyncio.wait_for(call, timeout=max(timeout_ms, 1) / 1000.0)
                structured = result.structuredContent
                response_schema_valid = bool(
                    isinstance(structured, dict)
                    and structured.get("schema_version") == "1.0"
                    and structured.get("ok") is True
                )
                success = not result.isError and response_schema_valid
                if not success:
                    error_type = "InvalidStructuredContent"
    except TimeoutError:
        error_type = "TimeoutError"
    except BaseException as exc:
        # ExceptionGroup 或 BrokenResourceError 都按断连观测记录，不保存原始错误文本。
        error_type = type(exc).__name__
    return MCPTransportTrace(
        trace_id=f"mcp-{fault}-{uuid4().hex[:12]}",
        fault=fault,
        success=success,
        response_schema_valid=response_schema_valid,
        elapsed_ms=round((perf_counter() - started) * 1000.0, 6),
        error_type=error_type,
    )
