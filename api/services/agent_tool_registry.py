"""Agent 工具的元数据、参数校验、前置条件和执行分派统一入口。"""

from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
from threading import RLock
from typing import Callable, TYPE_CHECKING

from pydantic import BaseModel

if TYPE_CHECKING:
    from api.services.agent_decision import ControllerDecision
    from api.services.agent_tools import AgentToolContext


ToolPrecondition = Callable[[dict, Counter], bool]
ToolHandler = Callable[["AgentToolContext", "ControllerDecision"], dict]


@dataclass(frozen=True)
class AgentTool:
    """终止行动保留给主控生命周期处理，其他工具必须提供执行函数。"""

    name: str
    label: str
    description: str
    parameters: type[BaseModel]
    precondition: ToolPrecondition
    handler: ToolHandler | None = None
    terminal: bool = False


_TOOLS: dict[str, AgentTool] = {}
_LOCK = RLock()
_BUILTINS_LOADED = False


def _check_tool(tool: AgentTool) -> None:
    """工具定义完整且与共享行动合同一致时才允许发布。"""
    from api.services.agent_decision import ControllerDecision
    if not tool.name.strip() or not tool.label.strip() or not tool.description.strip():
        raise ValueError("工具名称、标签和描述不能为空")
    if not callable(tool.precondition):
        raise ValueError("工具必须提供前置条件")
    if not isinstance(tool.parameters, type) or not issubclass(tool.parameters, BaseModel):
        raise ValueError("工具参数必须使用 Pydantic 模型")
    parameter_fields = set(ControllerDecision.model_fields) - {"action", "reason"}
    if not set(tool.parameters.model_fields).issubset(parameter_fields):
        raise ValueError("工具参数必须来自共享行动合同字段")
    if tool.terminal:
        if tool.name not in {"finish", "clarify"} or tool.handler is not None:
            raise ValueError("终止行动只能由主控处理 finish 或 clarify")
    elif tool.name in {"finish", "clarify"} or not callable(tool.handler):
        raise ValueError("普通工具必须提供处理函数且不能占用终止行动")


def _ensure_builtins_loaded() -> None:
    """一次性初始化内置工具，避免导入副作用和部分注册。"""
    global _BUILTINS_LOADED
    with _LOCK:
        if _BUILTINS_LOADED:
            return
        from api.services.agent_tool_catalog import builtin_tools
        staged: dict[str, AgentTool] = {}
        for tool in builtin_tools():
            _check_tool(tool)
            if tool.name in staged:
                raise ValueError(f"内置工具重复：{tool.name}")
            staged[tool.name] = tool
        _TOOLS.update(staged)
        _BUILTINS_LOADED = True


def register_agent_tool(tool: AgentTool) -> None:
    """应用启动时扩展工具，重复名称拒绝覆盖主控保护条件。"""
    _check_tool(tool)
    _ensure_builtins_loaded()
    with _LOCK:
        if tool.name in _TOOLS:
            raise ValueError(f"工具已注册：{tool.name}")
        _TOOLS[tool.name] = tool


def list_agent_tools() -> list[AgentTool]:
    """返回只读定义快照，Schema 和可用行动由此生成。"""
    _ensure_builtins_loaded()
    with _LOCK:
        return list(_TOOLS.values())


def get_agent_tool(name: str) -> AgentTool:
    """未知工具明确拒绝，不能通过任意字符串执行。"""
    _ensure_builtins_loaded()
    with _LOCK:
        if name not in _TOOLS:
            raise ValueError(f"未注册工具：{name}")
        return _TOOLS[name]


def available_actions(state: dict, counts: Counter) -> dict[str, str]:
    """工具分别判断是否可用，调用顺序由模型根据反馈决定。"""
    return {tool.name: tool.description for tool in list_agent_tools()
            if tool.precondition(state, counts)}


def execute_agent_tool(context: AgentToolContext, decision: ControllerDecision, counts: Counter) -> dict:
    """执行前重新检查条件和参数，入口调用者也无法绕过工具约束。"""
    tool = get_agent_tool(decision.action)
    if tool.terminal:
        raise ValueError("终止行动由主控循环处理")
    if not tool.precondition(context.state, counts):
        raise ValueError(f"行动 {tool.name} 不满足前置条件")
    parameters = tool.parameters.model_validate({name: getattr(decision, name) for name in tool.parameters.model_fields})
    # 使用参数模型核对和规范化后的值，避免扩展校验器的变换被忽略。
    validated = decision.model_copy(update={name: getattr(parameters, name) for name in tool.parameters.model_fields})
    return tool.handler(context, validated)
