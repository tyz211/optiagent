from __future__ import annotations

from collections import Counter, defaultdict
from dataclasses import dataclass, field
import math
from typing import Any, Callable

import pandas as pd

from optiagent.data import SupplyChainData, normalize_data
from optiagent.mcp_contracts import ProblemEnvelope, SolutionVerificationReport, SolveEnvelope


ABS_TOLERANCE = 1e-6
REL_TOLERANCE = 1e-7
VERIFIABLE_STATUSES = {"OPTIMAL", "NEAR_OPTIMAL", "FEASIBLE"}


@dataclass
class _VerificationState:
    """收集模板验证过程中的检查项和最大违反量。"""

    checks: dict[str, Any] = field(default_factory=dict)
    violations: list[str] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)
    max_violation: float = 0.0
    recomputed_objective: float | None = None

    def check(self, name: str, passed: bool, message: str, violation: float = 0.0) -> None:
        self.checks[name] = bool(passed)
        if passed:
            return
        self.violations.append(message)
        self.max_violation = max(self.max_violation, max(float(violation), 0.0))


def verify_solution(problem: ProblemEnvelope, solution: SolveEnvelope) -> SolutionVerificationReport:
    """独立复算求解结果，不依赖求解器自行声明的指标。"""

    template_id = problem.problem_spec.template_id
    reported_objective = _optional_float(solution.objective_value)
    if solution.template_id != template_id:
        return SolutionVerificationReport(
            template_id=template_id,
            verifiable=True,
            reported_objective=reported_objective,
            violations=[f"问题模板 {template_id} 与求解结果模板 {solution.template_id} 不一致。"],
        )
    if solution.status not in VERIFIABLE_STATUSES:
        return SolutionVerificationReport(
            template_id=template_id,
            reported_objective=reported_objective,
            warnings=[f"求解状态 {solution.status} 不包含可验证决策，跳过数学验算。"],
        )

    verifier = _VERIFIERS.get(template_id)
    if verifier is None:
        return SolutionVerificationReport(
            template_id=template_id,
            reported_objective=reported_objective,
            warnings=[f"模板 {template_id} 尚未注册 SolutionVerifier。"],
        )

    state = _VerificationState()
    try:
        verifier(problem.data, solution, state)
    except Exception as exc:
        return SolutionVerificationReport(
            template_id=template_id,
            reported_objective=reported_objective,
            violations=[f"验证器执行失败：{type(exc).__name__}: {exc}"],
        )

    feasible = not state.violations
    max_constraint_violation = state.max_violation
    objective_consistent = _numbers_close(reported_objective, state.recomputed_objective)
    state.check(
        "objective_consistent",
        objective_consistent,
        f"目标值不一致：求解器返回 {reported_objective}，独立复算为 {state.recomputed_objective}。",
        _difference(reported_objective, state.recomputed_objective),
    )
    passed = feasible and objective_consistent
    if solution.status == "OPTIMAL":
        state.warnings.append("本验证器确认可行性与目标值一致性，不通过重新求解独立证明全局最优性。")
    return SolutionVerificationReport(
        template_id=template_id,
        verifiable=True,
        passed=passed,
        feasible=feasible,
        objective_consistent=objective_consistent,
        reported_objective=reported_objective,
        recomputed_objective=state.recomputed_objective,
        max_constraint_violation=max_constraint_violation,
        checks=state.checks,
        violations=state.violations,
        warnings=state.warnings,
    )


def _verify_knapsack(data: dict[str, Any], solution: SolveEnvelope, state: _VerificationState) -> None:
    items = {str(row["item"]): row for row in data.get("items", [])}
    decisions = solution.decisions
    decision_names = [str(row.get("item")) for row in decisions]
    state.check("unique_items", len(decision_names) == len(set(decision_names)), "背包决策包含重复物品。")
    state.check("known_items", set(decision_names) == set(items), "背包决策的物品集合与输入不一致。")

    used_weight = 0.0
    selected_value = 0.0
    binary_valid = True
    for row in decisions:
        item = str(row.get("item"))
        selected = _optional_float(row.get("selected"))
        if item not in items or selected is None or not (_numbers_close(selected, 0.0) or _numbers_close(selected, 1.0)):
            binary_valid = False
            continue
        used_weight += float(items[item]["weight"]) * selected
        selected_value += float(items[item]["value"]) * selected
    state.check("binary_decisions", binary_valid, "背包 selected 必须是 0 或 1。")
    capacity = float(data["capacity"])
    state.check(
        "capacity",
        used_weight <= capacity + ABS_TOLERANCE,
        f"背包容量违反：使用 {used_weight}，容量 {capacity}。",
        used_weight - capacity,
    )
    state.checks["used_weight"] = used_weight
    state.recomputed_objective = selected_value


def _verify_assignment(data: dict[str, Any], solution: SolveEnvelope, state: _VerificationState) -> None:
    resources = {str(item) for item in data.get("resources", [])}
    tasks = {str(item) for item in data.get("tasks", [])}
    cost_map = {
        (str(row["resource"]), str(row["task"])): float(row["cost"])
        for row in data.get("costs", [])
    }
    pairs = [(str(row.get("resource")), str(row.get("task"))) for row in solution.decisions]
    state.check("known_pairs", all(pair in cost_map for pair in pairs), "指派结果包含输入成本矩阵中不存在的组合。")
    state.check("unique_pairs", len(pairs) == len(set(pairs)), "指派结果包含重复组合。")
    resource_counts = Counter(resource for resource, _task in pairs)
    task_counts = Counter(task for _resource, task in pairs)
    state.check("resource_capacity", all(resource_counts[item] <= 1 for item in resources), "同一资源被分配给多个任务。")
    state.check("task_coverage", all(task_counts[item] == 1 for item in tasks), "每个任务必须且只能被分配一次。")
    state.recomputed_objective = sum(cost_map.get(pair, 0.0) for pair in pairs)


def _verify_tsp(data: dict[str, Any], solution: SolveEnvelope, state: _VerificationState) -> None:
    nodes, costs = _tsp_nodes_and_costs(data)
    route = [str(item) for item in (solution.metrics or {}).get("route", [])]
    if not route:
        route = _route_from_arcs(solution.decisions)
    cycle_shape = len(route) == len(nodes) + 1 and bool(route) and route[0] == route[-1]
    state.check("closed_cycle", cycle_shape, "TSP 路线必须首尾相同且包含 n+1 个节点。")
    visited = route[:-1] if route else []
    state.check("visit_once", len(visited) == len(set(visited)) and set(visited) == nodes, "TSP 路线必须恰好访问每个节点一次。")
    missing_edges = [(source, target) for source, target in zip(route, route[1:], strict=False) if (source, target) not in costs]
    state.check("known_edges", not missing_edges, f"TSP 路线包含未知边：{missing_edges[:3]}。")
    if solution.decisions:
        expected_arcs = list(zip(route, route[1:], strict=False))
        returned_arcs = [(str(row.get("from")), str(row.get("to"))) for row in solution.decisions]
        state.check("decision_arcs_match", returned_arcs == expected_arcs, "TSP 决策边与汇总路线不一致。")
    state.recomputed_objective = sum(costs.get((source, target), 0.0) for source, target in zip(route, route[1:], strict=False))


def _verify_job_shop(data: dict[str, Any], solution: SolveEnvelope, state: _VerificationState) -> None:
    operations = _job_shop_operations(data)
    decisions: dict[tuple[str, int], dict[str, Any]] = {}
    duplicate = False
    for row in solution.decisions:
        key = (str(row.get("job")), int(row.get("order", 0)))
        duplicate = duplicate or key in decisions
        decisions[key] = row
    state.check("unique_operations", not duplicate, "调度结果包含重复工序。")
    state.check("operation_coverage", set(decisions) == set(operations), "调度结果的工序集合与输入不一致。")

    interval_valid = True
    machine_match = True
    intervals_by_machine: dict[str, list[tuple[float, float, tuple[str, int]]]] = defaultdict(list)
    for key, row in decisions.items():
        operation = operations.get(key)
        if operation is None:
            continue
        start = _optional_float(row.get("start"))
        end = _optional_float(row.get("end"))
        duration = float(operation["duration"])
        machine = str(row.get("machine"))
        machine_match = machine_match and machine == str(operation["machine"])
        if start is None or end is None or start < -ABS_TOLERANCE or not _numbers_close(end - start, duration):
            interval_valid = False
            continue
        intervals_by_machine[machine].append((start, end, key))
    state.check("machine_match", machine_match, "调度结果中的机器与输入工序不一致。")
    state.check("duration_and_bounds", interval_valid, "调度开始时间、结束时间或工序时长不一致。")

    precedence_valid = True
    for job in {key[0] for key in operations}:
        ordered = sorted((key for key in operations if key[0] == job), key=lambda item: item[1])
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if previous not in decisions or current not in decisions:
                precedence_valid = False
                continue
            previous_end = _optional_float(decisions[previous].get("end"))
            current_start = _optional_float(decisions[current].get("start"))
            if previous_end is None or current_start is None or current_start + ABS_TOLERANCE < previous_end:
                precedence_valid = False
                if previous_end is not None and current_start is not None:
                    state.max_violation = max(state.max_violation, previous_end - current_start)
    state.check("job_precedence", precedence_valid, "同一作业的工序先后顺序被违反。")

    no_overlap = True
    for intervals in intervals_by_machine.values():
        ordered = sorted(intervals)
        for previous, current in zip(ordered, ordered[1:], strict=False):
            if current[0] + ABS_TOLERANCE < previous[1]:
                no_overlap = False
                state.max_violation = max(state.max_violation, previous[1] - current[0])
    state.check("machine_no_overlap", no_overlap, "同一机器上的工序发生时间重叠。")
    ends = [_optional_float(row.get("end")) for row in decisions.values()]
    state.recomputed_objective = max((value for value in ends if value is not None), default=0.0)


def _verify_production_mix(data: dict[str, Any], solution: SolveEnvelope, state: _VerificationState) -> None:
    products = {str(row["product"]): row for row in data.get("products", [])}
    capacities = data.get("capacities", {})
    if isinstance(capacities, list):
        capacities = {str(row["resource"]): float(row["capacity"]) for row in capacities}
    quantities: dict[str, float] = {}
    duplicate = False
    numeric_valid = True
    for row in solution.decisions:
        product = str(row.get("product"))
        duplicate = duplicate or product in quantities
        quantity = _optional_float(row.get("quantity"))
        if quantity is None:
            numeric_valid = False
            continue
        quantities[product] = quantity
    state.check("unique_products", not duplicate, "生产计划包含重复产品。")
    state.check("product_coverage", set(quantities) == set(products), "生产计划的产品集合与输入不一致。")
    state.check("numeric_quantities", numeric_valid, "产品数量必须是有限数值。")

    bounds_valid = True
    integer_valid = True
    for product, quantity in quantities.items():
        source = products.get(product)
        if source is None:
            bounds_valid = False
            continue
        lower = float(source.get("min_qty", 0) or 0)
        upper = _optional_float(source.get("max_qty"))
        if quantity < lower - ABS_TOLERANCE or (upper is not None and quantity > upper + ABS_TOLERANCE):
            bounds_valid = False
        if bool(data.get("integer", False)) and not _numbers_close(quantity, round(quantity)):
            integer_valid = False
    state.check("quantity_bounds", bounds_valid, "产品数量违反非负、最小量或最大量约束。")
    state.check("integer_quantities", integer_valid, "整数生产计划包含非整数数量。")

    resource_valid = True
    resource_usage: dict[str, float] = {}
    for resource, capacity in capacities.items():
        used = sum(float(products[product][str(resource)]) * quantity for product, quantity in quantities.items() if product in products)
        resource_usage[str(resource)] = used
        if used > float(capacity) + ABS_TOLERANCE:
            resource_valid = False
            state.max_violation = max(state.max_violation, used - float(capacity))
    state.check("resource_capacities", resource_valid, "生产计划超过资源容量。")
    state.checks["resource_usage"] = resource_usage
    state.recomputed_objective = sum(float(products[product]["profit"]) * quantity for product, quantity in quantities.items() if product in products)


def _verify_facility_location(data: dict[str, Any], solution: SolveEnvelope, state: _VerificationState) -> None:
    normalized = normalize_data(
        SupplyChainData(
            warehouses=pd.DataFrame(data.get("warehouses", [])),
            customers=pd.DataFrame(data.get("customers", [])),
            costs=pd.DataFrame(data.get("costs", [])),
        )
    )
    warehouses = normalized.warehouses.set_index("warehouse")
    customers = normalized.customers.set_index("customer")
    costs = normalized.costs.set_index(["warehouse", "customer"])
    payload = solution.decisions[0] if solution.decisions else {}
    warehouse_rows = payload.get("warehouses", [])
    warehouse_names = [str(row.get("warehouse")) for row in warehouse_rows]
    state.check("unique_warehouses", len(warehouse_names) == len(set(warehouse_names)), "仓库启用决策包含重复仓库。")
    raw_open_map = {str(row.get("warehouse")): _optional_float(row.get("is_open")) for row in warehouse_rows}
    open_map = {name: int(round(value or 0.0)) for name, value in raw_open_map.items()}
    state.check("warehouse_coverage", set(open_map) == set(warehouses.index), "仓库启用决策与输入仓库集合不一致。")
    state.check(
        "binary_open",
        all(value is not None and (_numbers_close(value, 0.0) or _numbers_close(value, 1.0)) for value in raw_open_map.values()),
        "仓库启用变量必须是 0 或 1。",
    )

    shipped_by_warehouse: dict[str, float] = defaultdict(float)
    received_by_customer: dict[str, float] = defaultdict(float)
    transport_cost = 0.0
    allocations_valid = True
    nonnegative = True
    for row in payload.get("allocations", []):
        warehouse = str(row.get("warehouse"))
        customer = str(row.get("customer"))
        quantity = _optional_float(row.get("quantity"))
        if quantity is None or (warehouse, customer) not in costs.index:
            allocations_valid = False
            continue
        nonnegative = nonnegative and quantity >= -ABS_TOLERANCE
        shipped_by_warehouse[warehouse] += quantity
        received_by_customer[customer] += quantity
        transport_cost += float(costs.loc[(warehouse, customer), "cost"]) * quantity
    state.check("known_allocations", allocations_valid, "仓库分配包含未知仓库、客户或非法数量。")
    state.check("nonnegative_shipments", nonnegative, "仓库分配数量不能为负数。")

    demand_valid = True
    for customer, row in customers.iterrows():
        difference = abs(received_by_customer[customer] - float(row["demand"]))
        if difference > ABS_TOLERANCE:
            demand_valid = False
            state.max_violation = max(state.max_violation, difference)
    state.check("customer_demand", demand_valid, "客户收到的总量与需求不一致。")

    capacity_valid = True
    min_open_valid = True
    force_valid = True
    for warehouse, row in warehouses.iterrows():
        is_open = open_map.get(warehouse, 0)
        shipped = shipped_by_warehouse[warehouse]
        capacity = float(row["capacity"])
        if shipped > capacity * is_open + ABS_TOLERANCE:
            capacity_valid = False
            state.max_violation = max(state.max_violation, shipped - capacity * is_open)
        minimum = capacity * float(row["min_open_ratio"]) * is_open
        if shipped + ABS_TOLERANCE < minimum:
            min_open_valid = False
            state.max_violation = max(state.max_violation, minimum - shipped)
        if int(row["force_open"]) == 1 and is_open != 1:
            force_valid = False
        if int(row["force_closed"]) == 1 and is_open != 0:
            force_valid = False
    state.check("warehouse_capacity", capacity_valid, "仓库发货量超过启用容量或关闭仓库仍在发货。")
    state.check("minimum_utilization", min_open_valid, "启用仓库未满足最低利用率。")
    state.check("forced_open_close", force_valid, "仓库强制启用或关闭约束被违反。")
    fixed_cost = sum(float(warehouses.loc[name, "fixed_cost"]) * value for name, value in open_map.items() if name in warehouses.index)
    state.checks["transport_cost"] = transport_cost
    state.checks["fixed_cost"] = fixed_cost
    state.recomputed_objective = transport_cost + fixed_cost


def _tsp_nodes_and_costs(data: dict[str, Any]) -> tuple[set[str], dict[tuple[str, str], float]]:
    if "distance_matrix" in data:
        matrix = data["distance_matrix"]
        node_list = [str(item) for item in data.get("nodes", list(range(len(matrix))))]
        costs = {
            (source, target): float(matrix[i][j])
            for i, source in enumerate(node_list)
            for j, target in enumerate(node_list)
            if i != j
        }
        return set(node_list), costs
    costs: dict[tuple[str, str], float] = {}
    nodes: set[str] = set()
    for row in data.get("distances", []):
        source = str(row["from"])
        target = str(row["to"])
        distance = float(row["distance"])
        nodes.update({source, target})
        costs[(source, target)] = distance
        costs.setdefault((target, source), distance)
    return nodes, costs


def _route_from_arcs(arcs: list[dict[str, Any]]) -> list[str]:
    successors = {str(row.get("from")): str(row.get("to")) for row in arcs}
    if not successors:
        return []
    start = str(arcs[0].get("from"))
    route = [start]
    for _ in range(len(arcs)):
        target = successors.get(route[-1])
        if target is None:
            break
        route.append(target)
    return route


def _job_shop_operations(data: dict[str, Any]) -> dict[tuple[str, int], dict[str, Any]]:
    counters: dict[str, int] = defaultdict(int)
    operations: dict[tuple[str, int], dict[str, Any]] = {}
    for row in data.get("tasks", []):
        job = str(row["job"])
        order = int(row.get("order", counters[job]))
        counters[job] += 1
        operations[(job, order)] = row
    return operations


def _optional_float(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return None
    return number if math.isfinite(number) else None


def _numbers_close(left: float | None, right: float | None) -> bool:
    if left is None or right is None:
        return False
    return math.isclose(left, right, rel_tol=REL_TOLERANCE, abs_tol=ABS_TOLERANCE)


def _difference(left: float | None, right: float | None) -> float:
    if left is None or right is None:
        return 0.0
    return abs(left - right)


_VERIFIERS: dict[str, Callable[[dict[str, Any], SolveEnvelope, _VerificationState], None]] = {
    "knapsack": _verify_knapsack,
    "assignment": _verify_assignment,
    "tsp": _verify_tsp,
    "job_shop_scheduling": _verify_job_shop,
    "production_mix": _verify_production_mix,
    "facility_location": _verify_facility_location,
}
