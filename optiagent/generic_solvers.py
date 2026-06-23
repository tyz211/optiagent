from __future__ import annotations

from dataclasses import asdict, dataclass, field
from itertools import permutations
import json
import re
from typing import Any

import pandas as pd

from optiagent.problem_spec import ProblemSpec
from optiagent.solver_registry import GenericSolverAdapter, register_generic_solver, solve_with_registered_solver


@dataclass(frozen=True)
class GenericSolveResult:
    template_id: str
    display_name: str
    status: str
    objective_value: float | None
    objective_label: str
    solver_name: str
    summary: str
    decisions: list[dict[str, Any]] = field(default_factory=list)
    metrics: dict[str, Any] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)
    data_source: str = "用户数据"

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def solve_by_problem_spec(question: str, spec: ProblemSpec) -> GenericSolveResult | None:
    return solve_with_registered_solver(question, spec)


def solve_knapsack(
    data: dict[str, Any],
    data_source: str = "用户数据",
    warnings: list[str] | None = None,
    time_limit: int = 20,
) -> GenericSolveResult:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:
        return _generic_error("knapsack", "0-1 背包选择问题", f"无法导入 Gurobi: {exc}", warnings)

    items = pd.DataFrame(data.get("items", []))
    capacity = float(data.get("capacity", 0))
    if items.empty or not {"item", "value", "weight"}.issubset(items.columns) or capacity <= 0:
        return _generic_error(
            "knapsack",
            "0-1 背包选择问题",
            "背包数据缺少 items[item,value,weight] 或 capacity。",
            warnings,
        )

    items["value"] = pd.to_numeric(items["value"], errors="raise")
    items["weight"] = pd.to_numeric(items["weight"], errors="raise")
    names = items["item"].astype(str).tolist()
    values = dict(zip(names, items["value"].astype(float), strict=True))
    weights = dict(zip(names, items["weight"].astype(float), strict=True))

    try:
        model = gp.Model("generic_knapsack")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit
        choose = model.addVars(names, vtype=GRB.BINARY, name="choose")
        model.setObjective(gp.quicksum(values[item] * choose[item] for item in names), GRB.MAXIMIZE)
        model.addConstr(gp.quicksum(weights[item] * choose[item] for item in names) <= capacity, name="capacity")
        model.optimize()
    except gp.GurobiError as exc:
        return _generic_error("knapsack", "0-1 背包选择问题", f"Gurobi 求解异常: {exc}", warnings)

    status = _gurobi_status_name(model.Status)
    if model.SolCount == 0:
        return _generic_error("knapsack", "0-1 背包选择问题", f"未找到可行解，状态为 {status}。", warnings)

    decisions = []
    selected_weight = 0.0
    selected_value = 0.0
    for item in names:
        selected = int(round(choose[item].X))
        row = {
            "item": item,
            "selected": selected,
            "value": values[item],
            "weight": weights[item],
        }
        if selected:
            selected_weight += weights[item]
            selected_value += values[item]
        decisions.append(row)

    chosen = [_display_item_name(row["item"]) for row in decisions if row["selected"] == 1]
    return GenericSolveResult(
        template_id="knapsack",
        display_name="0-1 背包选择问题",
        status=status,
        objective_value=float(model.ObjVal),
        objective_label="最大价值",
        solver_name=f"Gurobi {'.'.join(map(str, gp.gurobi.version()))}",
        summary=f"建议选择 {', '.join(chosen) if chosen else '暂无项目'}，总价值 {selected_value:,.2f}，容量使用 {selected_weight:,.2f}/{capacity:,.2f}。",
        decisions=decisions,
        metrics={
            "capacity": capacity,
            "used_weight": selected_weight,
            "remaining_capacity": capacity - selected_weight,
            "selected_value": selected_value,
            "selected_count": len(chosen),
        },
        warnings=warnings or [],
        data_source=data_source,
    )


def solve_assignment(
    data: dict[str, Any],
    data_source: str = "用户数据",
    warnings: list[str] | None = None,
    time_limit: int = 20,
) -> GenericSolveResult:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:
        return _generic_error("assignment", "指派匹配问题", f"无法导入 Gurobi: {exc}", warnings)

    resources = [str(item) for item in data.get("resources", [])]
    tasks = [str(item) for item in data.get("tasks", [])]
    costs = pd.DataFrame(data.get("costs", []))
    if not resources or not tasks or costs.empty or not {"resource", "task", "cost"}.issubset(costs.columns):
        return _generic_error(
            "assignment",
            "指派匹配问题",
            "指派数据缺少 resources、tasks 或 costs[resource,task,cost]。",
            warnings,
        )

    costs["resource"] = costs["resource"].astype(str)
    costs["task"] = costs["task"].astype(str)
    costs["cost"] = pd.to_numeric(costs["cost"], errors="raise")
    cost_map = {(row.resource, row.task): float(row.cost) for row in costs.itertuples(index=False)}
    missing_pairs = [(resource, task) for resource in resources for task in tasks if (resource, task) not in cost_map]
    if missing_pairs:
        preview = ", ".join(f"{resource}->{task}" for resource, task in missing_pairs[:6])
        return _generic_error("assignment", "指派匹配问题", f"成本矩阵缺少组合：{preview}", warnings)

    try:
        model = gp.Model("generic_assignment")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit
        assign = model.addVars(resources, tasks, vtype=GRB.BINARY, name="assign")
        model.setObjective(
            gp.quicksum(cost_map[(resource, task)] * assign[resource, task] for resource in resources for task in tasks),
            GRB.MINIMIZE,
        )
        for task in tasks:
            model.addConstr(gp.quicksum(assign[resource, task] for resource in resources) == 1, name=f"cover_{task}")
        for resource in resources:
            model.addConstr(gp.quicksum(assign[resource, task] for task in tasks) <= 1, name=f"capacity_{resource}")
        model.optimize()
    except gp.GurobiError as exc:
        return _generic_error("assignment", "指派匹配问题", f"Gurobi 求解异常: {exc}", warnings)

    status = _gurobi_status_name(model.Status)
    if model.SolCount == 0:
        return _generic_error("assignment", "指派匹配问题", f"未找到可行解，状态为 {status}。", warnings)

    decisions = []
    for resource in resources:
        for task in tasks:
            if assign[resource, task].X > 0.5:
                decisions.append(
                    {
                        "resource": resource,
                        "task": task,
                        "cost": cost_map[(resource, task)],
                    }
                )
    pair_text = "；".join(f"{row['resource']} -> {row['task']}" for row in decisions)
    return GenericSolveResult(
        template_id="assignment",
        display_name="指派匹配问题",
        status=status,
        objective_value=float(model.ObjVal),
        objective_label="最小匹配成本",
        solver_name=f"Gurobi {'.'.join(map(str, gp.gurobi.version()))}",
        summary=f"建议匹配：{pair_text}。总成本 {float(model.ObjVal):,.2f}。",
        decisions=decisions,
        metrics={
            "resource_count": len(resources),
            "task_count": len(tasks),
            "assigned_count": len(decisions),
        },
        warnings=warnings or [],
        data_source=data_source,
    )


def solve_tsp(
    data: dict[str, Any],
    data_source: str = "用户数据",
    warnings: list[str] | None = None,
    time_limit: int = 20,
) -> GenericSolveResult:
    matrix, nodes, matrix_warnings = _build_tsp_matrix(data)
    warnings = [*(warnings or []), *matrix_warnings]
    if not matrix or len(matrix) < 2:
        return _generic_error("tsp", "旅行商路径问题", "TSP 数据缺少有效距离矩阵。", warnings, solver_name="OR-Tools")
    order, solver_name, is_optimal = _solve_tsp_order(matrix)
    route = [nodes[idx] for idx in order]
    arcs, total_distance = _route_arcs(route, matrix, nodes)
    status = "OPTIMAL" if is_optimal else "FEASIBLE"
    objective_label = "最短总距离" if is_optimal else "启发式路线距离"
    optimality_note = "已证明全局最优" if is_optimal else "当前为启发式可行解，未证明全局最优"

    return GenericSolveResult(
        template_id="tsp",
        display_name="旅行商路径问题",
        status=status,
        objective_value=total_distance,
        objective_label=objective_label,
        solver_name=solver_name,
        summary=f"建议访问顺序：{' -> '.join(route)}，总距离 {total_distance:,.2f}（{optimality_note}）。",
        decisions=arcs,
        metrics={"node_count": len(nodes), "total_distance": total_distance, "route": route, "optimality_proven": is_optimal},
        warnings=warnings,
        data_source=data_source,
    )


def solve_job_shop_scheduling(
    data: dict[str, Any],
    data_source: str = "用户数据",
    warnings: list[str] | None = None,
    time_limit: int = 20,
) -> GenericSolveResult:
    tasks = pd.DataFrame(data.get("tasks", []))
    if tasks.empty or not {"job", "machine", "duration"}.issubset(tasks.columns):
        return _generic_error(
            "job_shop_scheduling",
            "作业车间调度问题",
            "调度数据缺少 tasks[job,machine,duration]。",
            warnings,
            solver_name="OR-Tools CP-SAT",
        )

    tasks = tasks.copy()
    tasks["job"] = tasks["job"].astype(str)
    tasks["machine"] = tasks["machine"].astype(str)
    tasks["duration"] = pd.to_numeric(tasks["duration"], errors="raise").astype(int)
    if "order" not in tasks.columns:
        tasks["order"] = tasks.groupby("job").cumcount()
    tasks["order"] = pd.to_numeric(tasks["order"], errors="raise").astype(int)
    tasks = tasks.sort_values(["job", "order"]).reset_index(drop=True)
    horizon = int(tasks["duration"].sum())

    decisions = _list_schedule(tasks)
    objective = float(max((item["end"] for item in decisions), default=0))
    return GenericSolveResult(
        template_id="job_shop_scheduling",
        display_name="作业车间调度问题",
        status="FEASIBLE",
        objective_value=objective,
        objective_label="最大完工时间",
        solver_name="List Scheduling Heuristic",
        summary=f"已生成作业车间调度方案，最大完工时间 {objective:,.0f}。",
        decisions=sorted(decisions, key=lambda item: (item["machine"], item["start"], item["job"])),
        metrics={"makespan": objective, "job_count": tasks["job"].nunique(), "machine_count": tasks["machine"].nunique()},
        warnings=[*(warnings or []), "当前环境使用启发式调度器；如 OR-Tools CP-SAT 可用，可升级为证明最优的精确求解。"],
        data_source=data_source,
    )


def solve_production_mix(
    data: dict[str, Any],
    data_source: str = "用户数据",
    warnings: list[str] | None = None,
    time_limit: int = 20,
) -> GenericSolveResult:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:
        return _generic_error("production_mix", "产品组合与生产计划问题", f"无法导入 Gurobi: {exc}", warnings)

    products = pd.DataFrame(data.get("products", []))
    capacities = data.get("capacities", {})
    if isinstance(capacities, list):
        capacities = {str(row["resource"]): float(row["capacity"]) for row in capacities if "resource" in row and "capacity" in row}
    if products.empty or "product" not in products.columns or "profit" not in products.columns or not capacities:
        return _generic_error("production_mix", "产品组合与生产计划问题", "产品组合数据缺少 products[product,profit,...resource usage] 或 capacities。", warnings)

    products = products.copy()
    products["product"] = products["product"].astype(str)
    products["profit"] = pd.to_numeric(products["profit"], errors="raise")
    resource_names = [str(name) for name in capacities.keys() if str(name) in products.columns]
    if not resource_names:
        return _generic_error("production_mix", "产品组合与生产计划问题", "products 中没有与 capacities 匹配的资源消耗列。", warnings)
    for resource in resource_names:
        products[resource] = pd.to_numeric(products[resource], errors="raise")

    try:
        model = gp.Model("production_mix")
        model.Params.OutputFlag = 0
        model.Params.TimeLimit = time_limit
        vtype = GRB.INTEGER if bool(data.get("integer", False)) else GRB.CONTINUOUS
        quantity = model.addVars(products["product"].tolist(), lb=0, vtype=vtype, name="quantity")
        for row in products.itertuples(index=False):
            lower = float(getattr(row, "min_qty", 0) or 0) if "min_qty" in products.columns else 0
            upper = getattr(row, "max_qty", None) if "max_qty" in products.columns else None
            if lower > 0:
                model.addConstr(quantity[row.product] >= lower, name=f"min_{row.product}")
            if upper is not None and pd.notna(upper):
                model.addConstr(quantity[row.product] <= float(upper), name=f"max_{row.product}")
        for resource in resource_names:
            model.addConstr(
                gp.quicksum(float(products.loc[idx, resource]) * quantity[products.loc[idx, "product"]] for idx in products.index)
                <= float(capacities[resource]),
                name=f"capacity_{resource}",
            )
        model.setObjective(
            gp.quicksum(float(products.loc[idx, "profit"]) * quantity[products.loc[idx, "product"]] for idx in products.index),
            GRB.MAXIMIZE,
        )
        model.optimize()
    except gp.GurobiError as exc:
        return _generic_error("production_mix", "产品组合与生产计划问题", f"Gurobi 求解异常: {exc}", warnings)

    status = _gurobi_status_name(model.Status)
    if model.SolCount == 0:
        return _generic_error("production_mix", "产品组合与生产计划问题", f"未找到可行解，状态为 {status}。", warnings)

    decisions = []
    resource_usage = {resource: 0.0 for resource in resource_names}
    for row in products.itertuples(index=False):
        qty = float(quantity[row.product].X)
        decisions.append({"product": row.product, "quantity": qty, "profit": float(row.profit), "total_profit": qty * float(row.profit)})
        for resource in resource_names:
            resource_usage[resource] += qty * float(getattr(row, resource))
    resource_summary = {
        resource: {
            "used": used,
            "capacity": float(capacities[resource]),
            "remaining": float(capacities[resource]) - used,
        }
        for resource, used in resource_usage.items()
    }
    objective = float(model.ObjVal)
    return GenericSolveResult(
        template_id="production_mix",
        display_name="产品组合与生产计划问题",
        status=status,
        objective_value=objective,
        objective_label="最大利润",
        solver_name=f"Gurobi {'.'.join(map(str, gp.gurobi.version()))}",
        summary=f"最优生产计划利润 {objective:,.2f}。",
        decisions=decisions,
        metrics={"resource_usage": resource_summary, "integer": bool(data.get("integer", False))},
        warnings=warnings or [],
        data_source=data_source,
    )


def _extract_knapsack_data(question: str) -> tuple[dict[str, Any], str, list[str]]:
    payload = _extract_json_payload(question)
    if payload:
        data = payload.get("knapsack", payload)
        if "items" in data and "capacity" in data:
            return data, "用户问题中的 JSON 数据", []
    return {"capacity": 0, "items": []}, "未提供数据", ["未在问题或上传文件中识别到背包数据，请提供 capacity 与 items[item,value,weight]。"]


def _extract_assignment_data(question: str) -> tuple[dict[str, Any], str, list[str]]:
    payload = _extract_json_payload(question)
    if payload:
        data = payload.get("assignment", payload)
        if "resources" in data and "tasks" in data and "costs" in data:
            return data, "用户问题中的 JSON 数据", []
    return {"resources": [], "tasks": [], "costs": []}, "未提供数据", ["未在问题或上传文件中识别到指派数据，请提供 resources、tasks 与 costs[resource,task,cost]。"]


def _extract_tsp_data(question: str) -> tuple[dict[str, Any], str, list[str]]:
    payload = _extract_json_payload(question)
    if payload:
        data = payload.get("tsp", payload)
        if "distances" in data or "distance_matrix" in data:
            return data, "用户问题中的 JSON 数据", []
    return {"distances": []}, "未提供数据", ["未在问题或上传文件中识别到 TSP 距离数据，请提供 distances[from,to,distance] 或 distance_matrix。"]


def _extract_job_shop_data(question: str) -> tuple[dict[str, Any], str, list[str]]:
    payload = _extract_json_payload(question)
    if payload:
        data = payload.get("job_shop", payload.get("scheduling", payload))
        if "tasks" in data:
            return data, "用户问题中的 JSON 数据", []
    return {"tasks": []}, "未提供数据", ["未在问题或上传文件中识别到调度数据，请提供 tasks[job,machine,duration,order]。"]


def _extract_production_mix_data(question: str) -> tuple[dict[str, Any], str, list[str]]:
    payload = _extract_json_payload(question)
    if payload:
        data = payload.get("production_mix", payload.get("production", payload))
        if "products" in data and "capacities" in data:
            return data, "用户问题中的 JSON 数据", []
    return {"products": [], "capacities": {}}, "未提供数据", ["未在问题或上传文件中识别到产品组合数据，请提供 products 与 capacities。"]


def _build_tsp_matrix(data: dict[str, Any]) -> tuple[list[list[float]], list[str], list[str]]:
    warnings: list[str] = []
    if "distance_matrix" in data:
        matrix = [[float(value) for value in row] for row in data["distance_matrix"]]
        nodes = [str(item) for item in data.get("nodes", list(range(len(matrix))))]
        return matrix, nodes, warnings

    distances = pd.DataFrame(data.get("distances", []))
    if distances.empty or not {"from", "to", "distance"}.issubset(distances.columns):
        return [], [], warnings
    distances["from"] = distances["from"].astype(str)
    distances["to"] = distances["to"].astype(str)
    distances["distance"] = pd.to_numeric(distances["distance"], errors="raise")
    nodes = sorted(set(distances["from"]) | set(distances["to"]))
    index = {node: idx for idx, node in enumerate(nodes)}
    big_m = float(distances["distance"].max()) * max(len(nodes), 1) * 1000 + 1
    matrix = [[0.0 if i == j else big_m for j in nodes] for i in nodes]
    for row in distances.to_dict(orient="records"):
        i = index[str(row["from"])]
        j = index[str(row["to"])]
        matrix[i][j] = float(row["distance"])
        if matrix[j][i] == big_m:
            matrix[j][i] = float(row["distance"])
    if any(value == big_m for row in matrix for value in row):
        warnings.append("距离矩阵不完整，缺失边已用大惩罚值处理。")
    return matrix, nodes, warnings


def _solve_tsp_order(matrix: list[list[float]]) -> tuple[list[int], str, bool]:
    node_count = len(matrix)
    if node_count <= 9:
        best_order: list[int] | None = None
        best_distance = float("inf")
        for middle in permutations(range(1, node_count)):
            order = [0, *middle, 0]
            distance = sum(matrix[order[idx]][order[idx + 1]] for idx in range(len(order) - 1))
            if distance < best_distance:
                best_distance = distance
                best_order = list(order)
        return best_order or [0, 0], "Exact TSP Enumeration", True

    if node_count <= 20:
        return _solve_tsp_held_karp(matrix), "Held-Karp Dynamic Programming", True

    unvisited = set(range(1, node_count))
    order = [0]
    current = 0
    while unvisited:
        nxt = min(unvisited, key=lambda node: matrix[current][node])
        order.append(nxt)
        unvisited.remove(nxt)
        current = nxt
    order.append(0)
    return order, "Nearest Neighbor TSP Heuristic", False


def _solve_tsp_held_karp(matrix: list[list[float]]) -> list[int]:
    node_count = len(matrix)
    dp: dict[tuple[int, int], float] = {}
    parent: dict[tuple[int, int], int] = {}
    for node in range(1, node_count):
        mask = 1 << (node - 1)
        dp[(mask, node)] = matrix[0][node]
        parent[(mask, node)] = 0

    full_mask = (1 << (node_count - 1)) - 1
    for size in range(2, node_count):
        for mask in range(1 << (node_count - 1)):
            if mask.bit_count() != size:
                continue
            for node in range(1, node_count):
                node_bit = 1 << (node - 1)
                if not mask & node_bit:
                    continue
                previous_mask = mask ^ node_bit
                best_cost = float("inf")
                best_previous = 0
                for previous in range(1, node_count):
                    if not previous_mask & (1 << (previous - 1)):
                        continue
                    cost = dp[(previous_mask, previous)] + matrix[previous][node]
                    if cost < best_cost:
                        best_cost = cost
                        best_previous = previous
                dp[(mask, node)] = best_cost
                parent[(mask, node)] = best_previous

    best_last = min(
        range(1, node_count),
        key=lambda node: dp[(full_mask, node)] + matrix[node][0],
    )
    mask = full_mask
    reverse_middle = []
    current = best_last
    while current:
        reverse_middle.append(current)
        previous = parent[(mask, current)]
        mask ^= 1 << (current - 1)
        current = previous
    return [0, *reversed(reverse_middle), 0]


def _route_arcs(route: list[str], matrix: list[list[float]], nodes: list[str]) -> tuple[list[dict[str, Any]], float]:
    node_index = {node: idx for idx, node in enumerate(nodes)}
    arcs = []
    total_distance = 0.0
    for source, target in zip(route, route[1:], strict=False):
        distance = float(matrix[node_index[source]][node_index[target]])
        total_distance += distance
        arcs.append({"from": source, "to": target, "distance": distance})
    return arcs, total_distance


def _list_schedule(tasks: pd.DataFrame) -> list[dict[str, Any]]:
    machine_available: dict[str, int] = {}
    job_available: dict[str, int] = {}
    decisions: list[dict[str, Any]] = []
    for row in tasks.itertuples(index=False):
        machine = str(row.machine)
        job = str(row.job)
        start = max(machine_available.get(machine, 0), job_available.get(job, 0))
        end = start + int(row.duration)
        machine_available[machine] = end
        job_available[job] = end
        decisions.append(
            {
                "job": job,
                "machine": machine,
                "order": int(row.order),
                "duration": int(row.duration),
                "start": start,
                "end": end,
            }
        )
    return decisions


def _extract_json_payload(text: str) -> dict[str, Any] | None:
    candidates = re.findall(r"```json\s*(\{.*?\})\s*```", text, flags=re.DOTALL)
    candidates.extend(re.findall(r"(\{.*\})", text, flags=re.DOTALL))
    for candidate in candidates:
        try:
            value = json.loads(candidate)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    return None


def _generic_error(
    template_id: str,
    display_name: str,
    message: str,
    warnings: list[str] | None = None,
    solver_name: str = "Gurobi",
) -> GenericSolveResult:
    return GenericSolveResult(
        template_id=template_id,
        display_name=display_name,
        status="ERROR",
        objective_value=None,
        objective_label="目标值",
        solver_name=solver_name,
        summary=message,
        warnings=warnings or [],
    )


def _display_item_name(item: str) -> str:
    text = str(item)
    return f"物品 {text}" if text.isdigit() else text


def _gurobi_status_name(status_code: int) -> str:
    statuses = {
        1: "LOADED",
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        8: "NODE_LIMIT",
        9: "TIME_LIMIT",
        13: "SUBOPTIMAL",
    }
    return statuses.get(status_code, f"UNKNOWN_{status_code}")


def _register_builtin_solvers() -> None:
    register_generic_solver(
        GenericSolverAdapter(
            template_id="knapsack",
            display_name="0-1 背包选择问题",
            solver_name="Gurobi",
            solve=solve_knapsack,
            extract_from_question=_extract_knapsack_data,
        )
    )
    register_generic_solver(
        GenericSolverAdapter(
            template_id="tsp",
            display_name="旅行商路径问题",
            solver_name="OR-Tools Routing",
            solve=solve_tsp,
            extract_from_question=_extract_tsp_data,
        )
    )
    register_generic_solver(
        GenericSolverAdapter(
            template_id="job_shop_scheduling",
            display_name="作业车间调度问题",
            solver_name="OR-Tools CP-SAT",
            solve=solve_job_shop_scheduling,
            extract_from_question=_extract_job_shop_data,
        )
    )
    register_generic_solver(
        GenericSolverAdapter(
            template_id="production_mix",
            display_name="产品组合与生产计划问题",
            solver_name="Gurobi",
            solve=solve_production_mix,
            extract_from_question=_extract_production_mix_data,
        )
    )
    register_generic_solver(
        GenericSolverAdapter(
            template_id="assignment",
            display_name="指派匹配问题",
            solver_name="Gurobi",
            solve=solve_assignment,
            extract_from_question=_extract_assignment_data,
        )
    )


_register_builtin_solvers()
