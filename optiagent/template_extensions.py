"""模板扩展的唯一运行时注册表：说明、校验、求解与独立验算一起注册。"""

from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import Any, Callable

from optiagent.mcp_contracts import ProblemEnvelope, SolveEnvelope, ValidationReport
from optiagent.solution_verifier import VerificationState
from optiagent.solver_registry import GenericSolverAdapter
from optiagent.templates.capabilities import TemplateCapability
from optiagent.templates.definitions import OptimizationTemplate


DataValidator = Callable[[dict[str, Any]], ValidationReport]
DecisionVerifier = Callable[[dict[str, Any], SolveEnvelope, VerificationState], None]
EnvelopeSolver = Callable[[ProblemEnvelope, ValidationReport, int | None], SolveEnvelope]


@dataclass(frozen=True)
class TemplateExtension:
    """通用求解适配器和专有合同适配器必须且只能提供一个。"""

    template: OptimizationTemplate
    capability: TemplateCapability
    validate_data: DataValidator
    verify_decisions: DecisionVerifier
    generic_solver: GenericSolverAdapter | None = None
    envelope_solver: EnvelopeSolver | None = None
    envelope_solver_name: str = ""

    @property
    def solver_name(self) -> str:
        """能力发现与实际执行使用同一份求解器名称。"""
        return self.generic_solver.solver_name if self.generic_solver else self.envelope_solver_name


_EXTENSIONS: dict[str, TemplateExtension] = {}
_LOCK = RLock()
_BUILTINS_LOADED = False


def _check_extension(extension: TemplateExtension) -> None:
    """在发布前拒绝缺少求解、校验、验算或标识不一致的扩展。"""
    if not isinstance(extension.template, OptimizationTemplate) or not isinstance(extension.capability, TemplateCapability):
        raise ValueError("模板必须提供完整说明与能力描述")
    template_id = extension.template.template_id
    if not template_id or not template_id.strip():
        raise ValueError("模板标识不能为空")
    if not callable(extension.template.builder) or not callable(extension.validate_data) or not callable(extension.verify_decisions):
        raise ValueError("模板必须提供建模、数据校验和独立验算回调")
    if (extension.generic_solver is None) == (extension.envelope_solver is None):
        raise ValueError("模板必须且只能提供一个求解适配器")
    if extension.generic_solver is not None:
        if extension.generic_solver.template_id != template_id:
            raise ValueError("求解适配器与模板标识不一致")
        if not callable(extension.generic_solver.solve) or not callable(extension.generic_solver.extract_from_question):
            raise ValueError("通用适配器必须提供求解和文本提取回调")
    elif not callable(extension.envelope_solver):
        raise ValueError("合同求解适配器必须可调用")
    if not extension.solver_name:
        raise ValueError("求解器名称不能为空")


def _ensure_builtins_loaded() -> None:
    """延迟构造后一次性发布；失败不留下部分注册，初始化不受扩展数量影响。"""
    global _BUILTINS_LOADED
    with _LOCK:
        if _BUILTINS_LOADED:
            return
        from optiagent.templates.builtins import builtin_extensions
        builtins = builtin_extensions()
        staged: dict[str, TemplateExtension] = {}
        for extension in builtins:
            _check_extension(extension)
            template_id = extension.template.template_id
            if template_id in staged:
                raise ValueError(f"内置模板重复：{template_id}")
            staged[template_id] = extension
        _EXTENSIONS.update(staged)
        _BUILTINS_LOADED = True


def register_template_extension(extension: TemplateExtension, *, replace: bool = False) -> None:
    """启动时注册完整扩展，覆盖已有实现必须显式声明。"""
    _check_extension(extension)
    _ensure_builtins_loaded()
    with _LOCK:
        template_id = extension.template.template_id
        if template_id in _EXTENSIONS and not replace:
            raise ValueError(f"模板已注册：{template_id}")
        _EXTENSIONS[template_id] = extension


def get_template_extension(template_id: str) -> TemplateExtension | None:
    """读取单个完整扩展，未知标识返回空值。"""
    _ensure_builtins_loaded()
    with _LOCK:
        return _EXTENSIONS.get(template_id)


def list_template_extensions() -> list[TemplateExtension]:
    """返回注册快照，调用者无法修改内部容器。"""
    _ensure_builtins_loaded()
    with _LOCK:
        return list(_EXTENSIONS.values())
