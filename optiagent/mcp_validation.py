"""Data MCP 与 Solver MCP 共用的注册式数据校验入口。"""

from __future__ import annotations

from typing import Any

from optiagent.mcp_contracts import ValidationReport
from optiagent.template_extensions import get_template_extension


def validate_problem_data(template_id: str, data: dict[str, Any]) -> ValidationReport:
    """按完整扩展分派校验，未注册模板不会进入求解。"""
    extension = get_template_extension(template_id)
    if extension is None:
        return ValidationReport(valid=False, errors=[f"不支持的模板：{template_id}"])
    return extension.validate_data(data)
