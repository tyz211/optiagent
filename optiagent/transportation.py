"""单商品运输分配：严格输入合同、线性编译和独立业务验算。"""

from __future__ import annotations

from collections import defaultdict
import math
from typing import Annotated

from pydantic import BaseModel, ConfigDict, Field, StringConstraints, model_validator

from optiagent.linear_model import LinearModel


# 数量和成本拒绝布尔值、字符串及非有限值，单位必须由用户明确声明。
Name = Annotated[str, StringConstraints(strict=True, strip_whitespace=True, min_length=1, max_length=80)]
Amount = Annotated[float, Field(strict=True, ge=0, allow_inf_nan=False)]


class TransportRecord(BaseModel):
    """拒绝未知字段，防止额外业务限制在反序列化时被丢弃。"""

    model_config = ConfigDict(extra="forbid", allow_inf_nan=False)


class Supplier(TransportRecord):
    name: Name
    supply: Amount


class Consumer(TransportRecord):
    name: Name
    demand: Amount


class Route(TransportRecord):
    source: Name
    target: Name


class CostedRoute(Route):
    cost: Amount


class TransportationData(TransportRecord):
    """供给允许剩余、需求必须恰好满足；缺失线路必须显式声明禁运。"""

    suppliers: list[Supplier] = Field(min_length=1, max_length=100)
    consumers: list[Consumer] = Field(min_length=1, max_length=100)
    routes: list[CostedRoute] = Field(max_length=200)
    forbidden_routes: list[Route] = Field(default_factory=list, max_length=200)
    quantity_unit: Name
    currency: Name

    @model_validator(mode="after")
    def check_network(self):
        sources = [item.name for item in self.suppliers]
        targets = [item.name for item in self.consumers]
        if len(set(sources)) != len(sources) or len(set(targets)) != len(targets):
            raise ValueError("同类节点名称必须唯一，不能重复声明供给或需求")
        if len(sources) * len(targets) > 200:
            raise ValueError("首版运输模板最多支持 200 个供需组合")
        allowed = [(row.source, row.target) for row in self.routes]
        blocked = [(row.source, row.target) for row in self.forbidden_routes]
        if len(set(allowed)) != len(allowed) or len(set(blocked)) != len(blocked):
            raise ValueError("线路不能重复声明")
        if set(allowed) & set(blocked):
            raise ValueError("同一线路不能同时提供运费并声明禁运")
        expected = {(source, target) for source in sources for target in targets}
        provided = set(allowed) | set(blocked)
        if provided - expected:
            raise ValueError("线路引用了未声明的供给点或需求点")
        missing = sorted(expected - provided)
        if missing:
            raise ValueError("线路缺少单位运费或显式禁运声明：" + "、".join(f"{s}→{t}" for s, t in missing[:8]))
        # 供不应求属于有效但不可行的模型，留给求解器报告 INFEASIBLE。
        return self


def compile_transportation(problem: TransportationData) -> LinearModel:
    """使用内部变量名编译，保留业务约束标签，不把节点名称当表达式执行。"""

    names = [f"flow_{index}" for index in range(len(problem.routes))]
    variables = [{"name": name, "kind": "continuous", "lower": 0} for name in names]
    # 全部线路禁运时仍建立固定为零的占位变量，以表达零流量可行或需求不可行。
    if not variables:
        variables = [{"name": "zero", "kind": "continuous", "lower": 0, "upper": 0}]
    constraints = []
    for row in problem.suppliers:
        constraints.append({
            "coefficients": {names[i]: 1 for i, route in enumerate(problem.routes) if route.source == row.name},
            "sense": "<=", "rhs": row.supply, "label": f"供给上限：{row.name}",
        })
    for row in problem.consumers:
        constraints.append({
            "coefficients": {names[i]: 1 for i, route in enumerate(problem.routes) if route.target == row.name},
            "sense": "=", "rhs": row.demand, "label": f"需求满足：{row.name}",
        })
    return LinearModel.model_validate({
        "variables": variables,
        "objective": {"sense": "min", "coefficients": {names[i]: row.cost for i, row in enumerate(problem.routes)}},
        "constraints": constraints,
    })


def diagnose_transportation_infeasibility(problem: TransportationData) -> dict:
    """用供需网络给出可复算的不可行原因，不宣称这是完整 IIS。"""

    total_supply = math.fsum(row.supply for row in problem.suppliers)
    total_demand = math.fsum(row.demand for row in problem.consumers)
    supply_by_target = {
        consumer.name: math.fsum(
            supplier.supply
            for supplier in problem.suppliers
            if any(route.source == supplier.name and route.target == consumer.name
                   for route in problem.routes)
        )
        for consumer in problem.consumers
    }
    demand_shortfalls = [
        {
            "consumer": consumer.name,
            "demand": consumer.demand,
            "reachable_supply": supply_by_target[consumer.name],
            "shortfall": consumer.demand - supply_by_target[consumer.name],
        }
        for consumer in problem.consumers
        if supply_by_target[consumer.name] + 1e-9 < consumer.demand
    ]
    active_targets = {consumer.name for consumer in problem.consumers if consumer.demand > 0}
    disconnected_suppliers = [
        supplier.name
        for supplier in problem.suppliers
        if supplier.supply > 0 and not any(
            route.source == supplier.name and route.target in active_targets
            for route in problem.routes
        )
    ]
    issues: list[dict] = []
    if total_supply + 1e-9 < total_demand:
        issues.append({
            "type": "global_supply_shortfall",
            "shortfall": total_demand - total_supply,
            "total_supply": total_supply,
            "total_demand": total_demand,
        })
    if demand_shortfalls:
        issues.append({"type": "consumer_reachable_supply_shortfall", "items": demand_shortfalls})
    if disconnected_suppliers:
        issues.append({"type": "suppliers_without_active_route", "suppliers": disconnected_suppliers})
    if not issues:
        issues.append({
            "type": "network_capacity_or_forbidden_route",
            "message": "总供给覆盖总需求，但线路连通性或禁运组合限制了可行分配；需要进一步检查网络割集。",
        })
    summary_parts = []
    if total_supply + 1e-9 < total_demand:
        summary_parts.append(f"总供给不足 {total_demand - total_supply:g}{problem.quantity_unit}")
    if demand_shortfalls:
        summary_parts.append("需求点可达供给不足：" + "、".join(item["consumer"] for item in demand_shortfalls))
    if disconnected_suppliers:
        summary_parts.append("无有效需求线路的供给点：" + "、".join(disconnected_suppliers))
    if not summary_parts:
        summary_parts.append("总量满足但网络线路或禁运限制导致不可行")
    return {
        "kind": "transportation_infeasibility",
        "summary": "；".join(summary_parts) + "。",
        "quantity_unit": problem.quantity_unit,
        "total_supply": total_supply,
        "total_demand": total_demand,
        "issues": issues,
    }


def extract_transportation_data(question: str) -> tuple[dict, str, list[str]]:
    """直接调用 Gateway 时也检查 JSON 周围的文字，不忽略额外限制。"""

    from optiagent.transportation_text import parse_transportation_request

    data, _ = parse_transportation_request(question)
    return data, "用户运输分配输入", []


def solve_transportation(data: dict, data_source="用户数据", warnings=None, time_limit=None):
    """复用线性求解器，但返回运输业务决策和原始求解状态。"""

    from optiagent.generic_solvers import GenericSolveResult
    from optiagent.linear_solver import solve_linear_model

    problem = TransportationData.model_validate(data)
    compiled = compile_transportation(problem)
    result = solve_linear_model(compiled.model_dump(), data_source, warnings, time_limit)
    values = {row["variable"]: row["value"] for row in result.decisions}
    decisions = []
    if result.objective_value is not None:
        decisions = [
            {"source": row.source, "target": row.target, "quantity": values[f"flow_{i}"],
             "unit_cost": row.cost, "total_cost": values[f"flow_{i}"] * row.cost}
            for i, row in enumerate(problem.routes)
        ]
    metrics = {
        "supplier_count": len(problem.suppliers), "consumer_count": len(problem.consumers),
        "total_supply": math.fsum(row.supply for row in problem.suppliers),
        "total_demand": math.fsum(row.demand for row in problem.consumers),
        "quantity_unit": problem.quantity_unit, "currency": problem.currency,
        "optimality_proven": result.status == "OPTIMAL",
    }
    diagnosis = None
    if result.status == "INFEASIBLE":
        diagnosis = diagnose_transportation_infeasibility(problem)
        metrics["infeasibility_diagnosis"] = diagnosis
    if result.objective_value is None:
        summary = result.summary
        if diagnosis:
            summary = f"{summary} 原因分析：{diagnosis['summary']}"
    else:
        label = "最优运输成本" if result.status == "OPTIMAL" else "当前可行运输成本"
        summary = f"{label}为 {result.objective_value:g} {problem.currency}，需求总量 {metrics['total_demand']:g} {problem.quantity_unit}。"
    return GenericSolveResult(
        template_id="transportation", display_name="运输分配", status=result.status,
        objective_value=result.objective_value, objective_label="运输总成本",
        solver_name=result.solver_name, summary=summary, decisions=decisions,
        metrics=metrics,
        warnings=[*(warnings or []), *( ["运输模型不可行：" + diagnosis["summary"]] if diagnosis else [])],
        data_source=data_source,
    )


def verify_transportation(data, solution, state):
    """直接从业务数据复算供给、需求、禁运及成本，不复用编译器以免同错同过。"""

    from optiagent.solution_verifier import ABS_TOLERANCE

    problem = TransportationData.model_validate(data)
    costs = {(row.source, row.target): row.cost for row in problem.routes}
    rows = solution.decisions
    pairs = [(row.get("source"), row.get("target")) for row in rows]
    complete = len(set(pairs)) == len(pairs) and set(pairs) == set(costs)
    state.check("routes_complete", complete, "运输线路必须完整且唯一，不能包含禁运或未知线路")
    if not complete:
        return
    quantities = [row.get("quantity") for row in rows]
    finite = all(type(q) in (float, int) and math.isfinite(q) for q in quantities)
    state.check("finite_quantities", finite, "运输量必须为有限数值")
    if not finite:
        return
    outgoing, incoming = defaultdict(float), defaultdict(float)
    for pair, value in zip(pairs, quantities, strict=True):
        state.check(f"nonnegative:{pair}", value >= -ABS_TOLERANCE, f"线路 {pair} 的运输量为负", max(0, -value))
        outgoing[pair[0]] += value
        incoming[pair[1]] += value
    for row in problem.suppliers:
        error = max(0, outgoing[row.name] - row.supply)
        state.check(f"supply:{row.name}", error <= ABS_TOLERANCE, f"{row.name} 超过供给上限", error)
    for row in problem.consumers:
        error = abs(incoming[row.name] - row.demand)
        state.check(f"demand:{row.name}", error <= ABS_TOLERANCE, f"{row.name} 未恰好满足需求", error)
    state.recomputed_objective = math.fsum(q * costs[pair] for pair, q in zip(pairs, quantities, strict=True))
