from __future__ import annotations

import os
import time
from typing import Any, Literal

from mcp.server.fastmcp import FastMCP

from optiagent.mcp_servers.common import run_server


FaultProbeMode = Literal["normal", "timeout", "disconnect", "invalid_return"]


mcp = FastMCP(
    "OptiAgent MCP Fault Probe",
    instructions="仅用于可重复的 MCP transport 故障实验，不参与生产优化求解。",
    json_response=True,
)


@mcp.tool(title="制造可控 MCP 故障")
def mcp_fault_probe(
    mode: FaultProbeMode,
    delay_ms: int = 0,
    nonce: str = "probe",
) -> dict[str, Any]:
    """通过真实 stdio MCP 会话产生正常、超时、断连或非法返回。"""

    if mode == "timeout":
        time.sleep(max(0, delay_ms) / 1000.0)
    elif mode == "disconnect":
        # 仅终止由测试启动的独立子进程，用于产生真实 EOF/断连。
        os._exit(23)
    elif mode == "invalid_return":
        return {"unexpected_payload": [nonce, 7]}
    return {
        "schema_version": "1.0",
        "ok": True,
        "nonce": nonce,
    }


if __name__ == "__main__":
    run_server(mcp, default_port=8199)
