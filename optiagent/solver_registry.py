from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Protocol

from optiagent.problem_spec import ProblemSpec


class GenericSolverFn(Protocol):
    def __call__(
        self,
        data: dict[str, Any],
        data_source: str = "用户数据",
        warnings: list[str] | None = None,
        time_limit: int = 20,
    ):
        ...


DataExtractorFn = Callable[[str], tuple[dict[str, Any], str, list[str]]]


@dataclass(frozen=True)
class GenericSolverAdapter:
    template_id: str
    display_name: str
    solver_name: str
    solve: GenericSolverFn
    extract_from_question: DataExtractorFn


def register_generic_solver(adapter: GenericSolverAdapter) -> None:
    """兼容旧接口：仅显式替换已注册模板的通用求解适配器。"""
    from dataclasses import replace
    from optiagent.template_extensions import get_template_extension, register_template_extension
    extension = get_template_extension(adapter.template_id)
    if extension is None:
        raise ValueError("新模板必须通过 register_template_extension 同时注册校验和验算")
    register_template_extension(replace(extension, generic_solver=adapter, envelope_solver=None), replace=True)


def get_generic_solver(template_id: str) -> GenericSolverAdapter | None:
    """从统一扩展读取适配器，不依赖导入模块的副作用。"""
    from optiagent.template_extensions import get_template_extension
    extension = get_template_extension(template_id)
    return extension.generic_solver if extension else None


def list_generic_solvers() -> list[GenericSolverAdapter]:
    """返回所有通用适配器，仓库选址使用专有合同适配器。"""
    from optiagent.template_extensions import list_template_extensions
    return [extension.generic_solver for extension in list_template_extensions()
            if extension.generic_solver is not None]


def solve_with_registered_solver(question: str, spec: ProblemSpec):
    """保留旧的文本提取调用接口。"""
    adapter = get_generic_solver(spec.template_id)
    if adapter is None:
        return None
    data, source, warnings = adapter.extract_from_question(question)
    return adapter.solve(data, data_source=source, warnings=warnings)
