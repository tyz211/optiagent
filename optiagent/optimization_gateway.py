from __future__ import annotations

import json
from typing import Any

import pandas as pd

from optiagent.data import SupplyChainData, normalize_data
from optiagent.generic_solvers import GenericSolveResult
from optiagent.mcp_contracts import (
    ProblemEnvelope,
    ProblemSpecModel,
    SolveEnvelope,
    SourceReference,
    ValidationReport,
)
from optiagent.mcp_servers.common import json_safe
from optiagent.mcp_validation import validate_problem_data
from optiagent.problem_spec import ProblemSpec
from optiagent.solver import SolveResult, solve_facility_location
from optiagent.solver_registry import get_generic_solver
from optiagent.templates.registry import get_template


IN_PROCESS_MCP_STATUS = "已通过 MCP Gateway（进程内传输）执行。"


def build_problem_envelope(
    template_id: str,
    data: dict[str, Any],
    *,
    question: str = "",
    sources: list[SourceReference] | None = None,
    warnings: list[str] | None = None,
    metadata: dict[str, Any] | None = None,
) -> ProblemEnvelope:
    """为本地调用和 MCP 调用构建完全相同的问题合同。"""

    template = get_template(template_id)
    spec = ProblemSpecModel.from_problem_spec(template.build_spec(question, None))
    spec.confidence = 1.0
    report = validate_problem_data(template_id, data)
    report.warnings = [*(warnings or []), *report.warnings]
    if report.valid:
        spec.missing_data = []
    return ProblemEnvelope(
        problem_spec=spec,
        data=json_safe(data),
        sources=list(sources or []),
        validation=report,
        metadata=json_safe(metadata or {}),
    )


def validate_problem_envelope(problem: ProblemEnvelope) -> ValidationReport:
    """在统一求解边界重新验证版本、数据和上游警告。"""

    report = validate_problem_data(problem.problem_spec.template_id, problem.data)
    if problem.schema_version != "1.0":
        report.valid = False
        report.errors.insert(0, f"不支持的 ProblemEnvelope 版本：{problem.schema_version}")
    report.warnings = [*problem.validation.warnings, *report.warnings]
    return report


def solve_problem_envelope(problem: ProblemEnvelope, time_limit: int | None = None) -> SolveEnvelope:
    """唯一求解入口：校验 ProblemEnvelope，路由求解器，返回 SolveEnvelope。"""

    report = validate_problem_envelope(problem)
    template_id = problem.problem_spec.template_id
    if not report.valid:
        return SolveEnvelope(
            template_id=template_id,
            status="INVALID_DATA",
            summary="数据校验未通过，未启动求解器。",
            warnings=[*report.errors, *report.warnings],
            validation=report,
            provenance=problem.sources,
        )

    try:
        if template_id == "facility_location":
            return _solve_facility_envelope(problem, report, time_limit)
        adapter = get_generic_solver(template_id)
        if adapter is None:
            return SolveEnvelope(
                template_id=template_id,
                status="UNSUPPORTED",
                summary=f"未注册求解器：{template_id}",
                validation=report,
                provenance=problem.sources,
            )
        result = adapter.solve(
            problem.data,
            data_source=_source_label(problem),
            warnings=report.warnings,
            time_limit=time_limit,
        )
        return SolveEnvelope(
            template_id=result.template_id,
            status=result.status,
            objective_value=result.objective_value,
            objective_label=result.objective_label,
            solver_name=result.solver_name,
            summary=result.summary,
            decisions=json_safe(result.decisions),
            metrics=json_safe(result.metrics),
            warnings=result.warnings,
            validation=report,
            provenance=problem.sources,
        )
    except Exception as exc:
        return SolveEnvelope(
            template_id=template_id,
            status="SOLVER_ERROR",
            summary=f"求解器执行失败：{type(exc).__name__}: {exc}",
            validation=report,
            provenance=problem.sources,
        )


def solve_generic_via_gateway(
    template_id: str,
    data: dict[str, Any],
    *,
    data_source: str = "用户数据",
    question: str = "",
    warnings: list[str] | None = None,
    time_limit: int | None = None,
) -> GenericSolveResult:
    """通过 MCP 合同求解，再适配为现有响应层需要的结果对象。"""

    source = SourceReference(kind="inline", name=data_source)
    problem = build_problem_envelope(
        template_id,
        data,
        question=question,
        sources=[source],
        warnings=warnings,
        metadata={"gateway_transport": "in_process"},
    )
    solved = solve_problem_envelope(problem, time_limit=time_limit)
    return GenericSolveResult(
        template_id=solved.template_id,
        display_name=problem.problem_spec.display_name,
        status=solved.status,
        objective_value=solved.objective_value,
        objective_label=solved.objective_label,
        solver_name=solved.solver_name,
        summary=solved.summary,
        decisions=solved.decisions,
        metrics=solved.metrics,
        warnings=solved.warnings,
        data_source=data_source,
    )


def solve_question_via_gateway(question: str, spec: ProblemSpec) -> GenericSolveResult | None:
    """保留自然语言 JSON 入口，但所有求解都经过统一 Gateway。"""

    adapter = get_generic_solver(spec.template_id)
    if adapter is None:
        return None
    data, data_source, warnings = adapter.extract_from_question(question)
    return solve_generic_via_gateway(
        spec.template_id,
        data,
        data_source=data_source,
        question=question,
        warnings=warnings,
    )


def solve_facility_via_gateway(
    data: SupplyChainData,
    *,
    data_source: str = "结构化仓库选址数据",
    question: str = "",
    warnings: list[str] | None = None,
    time_limit: int | None = None,
) -> SolveResult:
    """将仓库选址也迁移到与通用模板相同的 MCP Gateway。"""

    normalized = normalize_data(data)
    payload = {
        "warehouses": _frame_records(normalized.warehouses),
        "customers": _frame_records(normalized.customers),
        "costs": _frame_records(normalized.costs),
    }
    problem = build_problem_envelope(
        "facility_location",
        payload,
        question=question,
        sources=[SourceReference(kind="inline", name=data_source)],
        warnings=warnings,
        metadata={"gateway_transport": "in_process"},
    )
    solved = solve_problem_envelope(problem, time_limit=time_limit)
    return _facility_result_from_envelope(solved)


def _solve_facility_envelope(
    problem: ProblemEnvelope,
    report: ValidationReport,
    time_limit: int | None,
) -> SolveEnvelope:
    data = normalize_data(
        SupplyChainData(
            warehouses=pd.DataFrame(problem.data["warehouses"]),
            customers=pd.DataFrame(problem.data["customers"]),
            costs=pd.DataFrame(problem.data["costs"]),
        )
    )
    result = solve_facility_location(data, time_limit=time_limit)
    decisions = {
        "allocations": _frame_records(result.allocations),
        "warehouses": _frame_records(result.warehouse_summary),
        "customers": _frame_records(result.customer_summary),
    }
    return SolveEnvelope(
        template_id="facility_location",
        status=result.status,
        objective_value=result.objective_value,
        objective_label="最小总成本",
        solver_name=result.solver_name,
        summary=result.message,
        decisions=[decisions],
        metrics={
            "transport_cost": result.transport_cost,
            "fixed_cost": result.fixed_cost,
            "mip_gap": result.mip_gap,
            "optimality_proven": result.optimality_proven,
            "model_type": result.model_type,
            **report.checks,
        },
        warnings=report.warnings,
        validation=report,
        provenance=problem.sources,
    )


def _facility_result_from_envelope(result: SolveEnvelope) -> SolveResult:
    decisions = result.decisions[0] if result.decisions else {}
    metrics = result.metrics or {}
    return SolveResult(
        status=result.status,
        objective_value=result.objective_value,
        transport_cost=_optional_float(metrics.get("transport_cost")),
        fixed_cost=_optional_float(metrics.get("fixed_cost")),
        allocations=pd.DataFrame(decisions.get("allocations", [])),
        warehouse_summary=pd.DataFrame(decisions.get("warehouses", [])),
        customer_summary=pd.DataFrame(decisions.get("customers", [])),
        message=result.summary,
        solver_name=result.solver_name,
        model_type=str(metrics.get("model_type") or "MILP"),
        mip_gap=_optional_float(metrics.get("mip_gap")),
        optimality_proven=bool(metrics.get("optimality_proven", False)),
    )


def _frame_records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    if frame.empty:
        return []
    # pandas 的 JSON 转换可以稳定处理 NaN、numpy 数值与时间对象。
    return json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))


def _source_label(problem: ProblemEnvelope) -> str:
    names = [source.name for source in problem.sources]
    return "MCP：" + "、".join(names) if names else "MCP 结构化输入"


def _optional_float(value: Any) -> float | None:
    try:
        return float(value) if value is not None else None
    except (TypeError, ValueError):
        return None
