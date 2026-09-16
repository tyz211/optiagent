from __future__ import annotations

from collections.abc import Iterable
from copy import deepcopy
from statistics import mean
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from optiagent.mcp_contracts import SolveEnvelope, SourceReference
from optiagent.optimization_gateway import build_problem_envelope, solve_problem_envelope
from optiagent.solution_verifier import verify_solution


FaultKind = Literal["clean", "missing_schema", "tampered_objective"]
SUCCESS_STATUSES = {"OPTIMAL", "NEAR_OPTIMAL", "FEASIBLE"}


class EndToEndBenchmarkInstance(BaseModel):
    """通过真实 Gateway/Solver/Verifier 执行的小规模优化实例。"""

    schema_version: str = "1.0"
    task_id: str
    template_id: str
    question: str
    data: dict[str, Any] = Field(default_factory=dict)
    split: Literal["smoke", "train", "validation", "test"] = "smoke"


class EndToEndCaseResult(BaseModel):
    """单个问题与故障组合的可审计测试结果。"""

    schema_version: str = "1.0"
    case_id: str
    task_id: str
    template_id: str
    fault: FaultKind
    passed: bool
    solver_status: str
    validation_valid: bool
    solution_verifiable: bool
    solution_verified: bool
    fault_detected: bool
    objective_value: float | None = None
    elapsed_ms: float
    details: list[str] = Field(default_factory=list)


def build_smoke_instances(seed: int = 42) -> list[EndToEndBenchmarkInstance]:
    """生成六类问题的真实小规模烟雾测试集。"""

    offset = abs(int(seed)) % 7
    return [
        EndToEndBenchmarkInstance(
            task_id=f"knapsack-smoke-s{seed}",
            template_id="knapsack",
            question="在容量限制下求解 0-1 背包问题。",
            data={
                "capacity": 7 + offset,
                "items": [
                    {"item": "A", "value": 8 + offset, "weight": 3},
                    {"item": "B", "value": 6, "weight": 2},
                    {"item": "C", "value": 9, "weight": 5},
                ],
            },
        ),
        EndToEndBenchmarkInstance(
            task_id=f"assignment-smoke-s{seed}",
            template_id="assignment",
            question="最小化两个资源完成两个任务的指派成本。",
            data={
                "resources": ["R1", "R2"],
                "tasks": ["T1", "T2"],
                "costs": [
                    {"resource": "R1", "task": "T1", "cost": 2 + offset},
                    {"resource": "R1", "task": "T2", "cost": 8},
                    {"resource": "R2", "task": "T1", "cost": 7},
                    {"resource": "R2", "task": "T2", "cost": 3},
                ],
            },
        ),
        EndToEndBenchmarkInstance(
            task_id=f"tsp-smoke-s{seed}",
            template_id="tsp",
            question="求访问所有城市并回到起点的最短 TSP 回路。",
            data={
                "nodes": ["A", "B", "C", "D"],
                "distance_matrix": [
                    [0, 2, 7 + offset, 4],
                    [2, 0, 3, 6],
                    [7 + offset, 3, 0, 2],
                    [4, 6, 2, 0],
                ],
            },
        ),
        EndToEndBenchmarkInstance(
            task_id=f"job-shop-smoke-s{seed}",
            template_id="job_shop_scheduling",
            question="最小化两个作业在两台机器上的完工时间。",
            data={
                "tasks": [
                    {"job": "J1", "machine": "M1", "duration": 2 + offset % 2, "order": 1},
                    {"job": "J1", "machine": "M2", "duration": 2, "order": 2},
                    {"job": "J2", "machine": "M2", "duration": 1, "order": 1},
                    {"job": "J2", "machine": "M1", "duration": 3, "order": 2},
                ]
            },
        ),
        EndToEndBenchmarkInstance(
            task_id=f"production-mix-smoke-s{seed}",
            template_id="production_mix",
            question="在劳动力和设备容量下最大化产品组合利润。",
            data={
                "products": [
                    {"product": "P1", "profit": 7 + offset, "labor": 2, "machine": 1, "max_qty": 4},
                    {"product": "P2", "profit": 5, "labor": 1, "machine": 2, "max_qty": 5},
                ],
                "capacities": {"labor": 8, "machine": 8},
                "integer": True,
            },
        ),
        EndToEndBenchmarkInstance(
            task_id=f"facility-location-smoke-s{seed}",
            template_id="facility_location",
            question="最小化仓库固定成本与客户运输成本。",
            data={
                "warehouses": [
                    {"warehouse": "W1", "region": "华中", "capacity": 12, "fixed_cost": 5 + offset},
                    {"warehouse": "W2", "region": "华南", "capacity": 12, "fixed_cost": 7},
                ],
                "customers": [
                    {"customer": "C1", "demand": 5},
                    {"customer": "C2", "demand": 4},
                ],
                "costs": [
                    {"warehouse": "W1", "customer": "C1", "cost": 2},
                    {"warehouse": "W1", "customer": "C2", "cost": 4},
                    {"warehouse": "W2", "customer": "C1", "cost": 5},
                    {"warehouse": "W2", "customer": "C2", "cost": 1},
                ],
            },
        ),
    ]


def run_end_to_end_case(
    instance: EndToEndBenchmarkInstance,
    fault: FaultKind = "clean",
    *,
    time_limit: int = 10,
) -> EndToEndCaseResult:
    """对单个实例执行真实求解或故障检测。"""

    started = perf_counter()
    data = _inject_schema_fault(instance.template_id, instance.data) if fault == "missing_schema" else deepcopy(instance.data)
    problem = build_problem_envelope(
        instance.template_id,
        data,
        question=instance.question,
        sources=[SourceReference(kind="inline", name=f"benchmark:{instance.task_id}")],
        metadata={"benchmark": "e2e-v1", "fault": fault},
    )
    solved = solve_problem_envelope(problem, time_limit=time_limit)
    report = solved.solution_verification
    fault_detected = False
    details: list[str] = []

    if fault == "clean":
        passed = bool(solved.status in SUCCESS_STATUSES and report and report.passed)
        if not passed:
            details.extend(solved.warnings or [solved.summary])
    elif fault == "missing_schema":
        fault_detected = solved.status == "INVALID_DATA" and not solved.validation.valid
        passed = fault_detected
        details.extend(solved.validation.errors)
    elif fault == "tampered_objective":
        if solved.status in SUCCESS_STATUSES and solved.objective_value is not None:
            tampered = _tamper_objective(solved)
            report = verify_solution(problem, tampered)
            fault_detected = report.verifiable and not report.passed and report.objective_consistent is False
            details.extend(report.violations)
        else:
            details.append("原始求解未成功，无法执行结果篡改检测。")
        passed = fault_detected
    else:
        raise ValueError(f"未知故障类型：{fault}")

    return EndToEndCaseResult(
        case_id=f"{instance.task_id}:{fault}",
        task_id=instance.task_id,
        template_id=instance.template_id,
        fault=fault,
        passed=passed,
        solver_status=solved.status,
        validation_valid=solved.validation.valid,
        solution_verifiable=bool(report and report.verifiable),
        solution_verified=bool(report and report.passed),
        fault_detected=fault_detected,
        objective_value=solved.objective_value,
        elapsed_ms=round((perf_counter() - started) * 1000, 3),
        details=details,
    )


def run_end_to_end_benchmark(
    instances: list[EndToEndBenchmarkInstance] | None = None,
    *,
    faults: tuple[FaultKind, ...] = ("clean", "missing_schema", "tampered_objective"),
    seed: int = 42,
    time_limit: int = 10,
) -> dict[str, Any]:
    """运行完整测试矩阵并返回可持久化的指标。"""

    selected = build_smoke_instances(seed=seed) if instances is None else instances
    results = [
        run_end_to_end_case(instance, fault, time_limit=time_limit)
        for instance in selected
        for fault in faults
    ]
    clean = [item for item in results if item.fault == "clean"]
    faulted = [item for item in results if item.fault != "clean"]
    return {
        "schema_version": "1.0",
        "benchmark": "optimization-agent-e2e-smoke",
        "seed": seed,
        "case_count": len(results),
        "metrics": {
            "pass_rate": _rate(item.passed for item in results),
            "clean_success_rate": _rate(item.passed for item in clean),
            "fault_detection_rate": _rate(item.fault_detected for item in faulted),
            "average_latency_ms": round(mean(item.elapsed_ms for item in results), 3) if results else 0.0,
        },
        "cases": [item.model_dump(mode="json") for item in results],
    }


def _inject_schema_fault(template_id: str, source: dict[str, Any]) -> dict[str, Any]:
    """按模板删除一个必需结构，用于验证 Data/Gateway 拦截能力。"""

    data = deepcopy(source)
    if template_id == "knapsack":
        data.pop("capacity", None)
    elif template_id == "assignment":
        data.pop("resources", None)
    elif template_id == "tsp":
        data["distance_matrix"] = [[0, 1], [1]]
    elif template_id == "job_shop_scheduling":
        for task in data.get("tasks", []):
            task.pop("duration", None)
    elif template_id == "production_mix":
        data["capacities"] = {}
    elif template_id == "facility_location":
        data["costs"] = []
    else:
        raise ValueError(f"未知模板：{template_id}")
    return data


def _tamper_objective(solved: SolveEnvelope) -> SolveEnvelope:
    """仅篡改声称的目标值，保留决策供 Verifier 独立复算。"""

    return solved.model_copy(
        update={
            "objective_value": float(solved.objective_value or 0.0) + 12345.0,
            "solution_verification": None,
        }
    )


def _rate(values: Iterable[bool]) -> float:
    items = [float(value) for value in values]
    return round(mean(items), 6) if items else 0.0
