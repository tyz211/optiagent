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


_GENERIC_SOLVERS: dict[str, GenericSolverAdapter] = {}


def _ensure_builtin_solvers_loaded() -> None:
    if _GENERIC_SOLVERS:
        return
    import optiagent.generic_solvers  # noqa: F401


def register_generic_solver(adapter: GenericSolverAdapter) -> None:
    _GENERIC_SOLVERS[adapter.template_id] = adapter


def get_generic_solver(template_id: str) -> GenericSolverAdapter | None:
    _ensure_builtin_solvers_loaded()
    return _GENERIC_SOLVERS.get(template_id)


def list_generic_solvers() -> list[GenericSolverAdapter]:
    _ensure_builtin_solvers_loaded()
    return list(_GENERIC_SOLVERS.values())


def solve_with_registered_solver(question: str, spec: ProblemSpec):
    _ensure_builtin_solvers_loaded()
    adapter = get_generic_solver(spec.template_id)
    if adapter is None:
        return None
    data, source, warnings = adapter.extract_from_question(question)
    return adapter.solve(data, data_source=source, warnings=warnings)
