from __future__ import annotations

import math
import re

from optiagent.linear_model import LinearModel, TEMPLATE_ID, parse_linear_text
from optiagent.solver_config import configure_gurobi_model


def extract_linear_data(question: str) -> tuple[dict, str, list[str]]:
    """兼容已规范化 JSON 和直接数学文本，统一交给 Gateway 校验。"""
    from optiagent.generic_solvers import _extract_json_payload

    payload = _extract_json_payload(question)
    if payload is not None:
        return payload.get(TEMPLATE_ID, payload), "会话内数学模型", []
    parsed = parse_linear_text(question)
    if parsed is None:
        raise ValueError("请提供完整线性目标、约束和变量取值范围")
    return parsed, "数学文本 / LaTeX", []


def describe_linear_model(data: dict) -> tuple[str, list[str], list[str]]:
    """把真正执行的模型显示给用户，不沿用通用模板的空泛说明。"""
    model = LinearModel.model_validate(data)

    def expression(coefficients, constant=0):
        terms = [f"{value:g}*{name}" for name, value in coefficients.items() if value]
        if constant or not terms:
            terms.append(f"{constant:g}")
        return " + ".join(terms).replace("+ -", "- ")

    objective = f"{'最大化' if model.objective.sense == 'max' else '最小化'} " + expression(model.objective.coefficients, model.objective.constant)
    variables = []
    for variable in model.variables:
        if variable.values is not None:
            domain = "{" + ",".join(f"{value:g}" for value in variable.values) + "}"
        else:
            domain = {"binary": "{0,1}", "integer": "整数", "continuous": "实数"}[variable.kind]
            domain += f"，下界 {variable.lower if variable.lower is not None else '无'}，上界 {variable.upper if variable.upper is not None else '无'}"
        variables.append(f"{variable.name} ∈ {domain}")
    constraints = [f"{expression(row.coefficients, row.constant)} {row.sense} {row.rhs:g}" for row in model.constraints]
    return objective, variables, constraints


def solve_linear_model(data: dict, data_source: str = "数学文本", warnings=None, time_limit=None):
    """从受限合同构建 LP/MILP；非连续整数集合使用选择变量精确表达。"""
    import gurobipy as gp
    from gurobipy import GRB
    from optiagent.generic_solvers import GenericSolveResult

    problem = LinearModel.model_validate(data)
    with gp.Model("explicit_linear_model") as model:
        configure_gurobi_model(model)
        if time_limit is not None:
            model.Params.TimeLimit = time_limit
        # 显式数学模型要求完整求解证明；浮点容差比验算标准更严格。
        model.Params.MIPGap = 0
        model.Params.FeasibilityTol = 1e-9
        model.Params.IntFeasTol = 1e-9
        variables = {}
        for variable in problem.variables:
            kind = {"binary": GRB.BINARY, "integer": GRB.INTEGER, "continuous": GRB.CONTINUOUS}[variable.kind]
            lower = variable.lower if variable.lower is not None else (0 if variable.kind == "binary" else -GRB.INFINITY)
            upper = variable.upper if variable.upper is not None else (1 if variable.kind == "binary" else GRB.INFINITY)
            variables[variable.name] = model.addVar(lb=lower, ub=upper, vtype=kind, name=variable.name)
            if variable.values is not None:
                values = sorted(variable.values)
                model.addConstr(variables[variable.name] >= min(values))
                model.addConstr(variables[variable.name] <= max(values))
                if len(values) != max(values) - min(values) + 1:
                    # {0,2} 不能放宽成 0≤x≤2 的整数区间，否则会错误允许 1。
                    selectors = model.addVars(len(values), vtype=GRB.BINARY, name=f"domain_{variable.name}")
                    model.addConstr(gp.quicksum(selectors.values()) == 1)
                    model.addConstr(variables[variable.name] == gp.quicksum(v * selectors[i] for i, v in enumerate(values)))
        objective = gp.quicksum(coefficient * variables[name] for name, coefficient in problem.objective.coefficients.items()) + problem.objective.constant
        model.setObjective(objective, GRB.MAXIMIZE if problem.objective.sense == "max" else GRB.MINIMIZE)
        for index, constraint in enumerate(problem.constraints):
            left = gp.quicksum(value * variables[name] for name, value in constraint.coefficients.items()) + constraint.constant
            relation = {"<=": lambda: left <= constraint.rhs, ">=": lambda: left >= constraint.rhs, "=": lambda: left == constraint.rhs}
            model.addConstr(relation[constraint.sense](), name=f"constraint_{index + 1}")
        model.optimize()
        if model.Status == GRB.INF_OR_UNBD:
            # 关闭对偶约简后再区分不可行和无界，避免把两者混称为无解。
            model.Params.DualReductions = 0
            model.optimize()
        proven = model.Status == GRB.OPTIMAL
        status = "OPTIMAL" if proven else ("FEASIBLE" if model.SolCount else {
            GRB.INFEASIBLE: "INFEASIBLE", GRB.UNBOUNDED: "UNBOUNDED", GRB.TIME_LIMIT: "TIME_LIMIT",
        }.get(model.Status, "SOLVER_ERROR"))
        values = {name: (0.0 if variable.X == 0 else float(variable.X)) for name, variable in variables.items()} if model.SolCount else {}
        # 按自然编号展示 x1…x10，同时仅规范负零，不掩盖任何非零约束违反。
        ordered = sorted(problem.variables, key=lambda item: [int(part) if part.isdigit() else part for part in re.split(r'(\d+)', item.name)])
        decisions = [{"variable": variable.name, "value": values[variable.name], "kind": variable.kind}
                     for variable in ordered] if values else []
        value = float(model.ObjVal) if model.SolCount else None
        checks = []
        descriptions = describe_linear_model(data)[2]
        for index, constraint in enumerate(problem.constraints):
            if not values:
                break
            left = sum(coefficient * values[name] for name, coefficient in constraint.coefficients.items()) + constraint.constant
            slack = constraint.rhs - left if constraint.sense == "<=" else left - constraint.rhs
            checks.append({"constraint": index + 1, "expression": descriptions[index],
                           "lhs": left, "sense": constraint.sense, "rhs": constraint.rhs,
                           "slack": slack if constraint.sense != "=" else abs(slack)})
        labels = {"INFEASIBLE": "约束相互冲突，模型不可行。", "UNBOUNDED": "目标无界，请检查是否缺少必要边界。",
                  "TIME_LIMIT": "达到时间限制，尚未找到可行解。", "SOLVER_ERROR": "未取得可用求解结果。"}
        summary = f"已解析 {len(variables)} 个变量、{len(problem.constraints)} 条线性约束。"
        summary += (f"{'最优' if proven else '当前可行'}目标值为 {value:g}。" if value is not None else labels.get(status, status))
        result = GenericSolveResult(
            template_id=TEMPLATE_ID, display_name="文本线性规划 / 整数规划", status=status,
            objective_value=value, objective_label="最大目标值" if problem.objective.sense == "max" else "最小目标值",
            solver_name="Gurobi LP/MILP", summary=summary, decisions=decisions,
            metrics={"variable_count": len(variables), "constraint_count": len(problem.constraints),
                     "optimality_proven": proven, "constraint_activity": checks,
                     "parsed_model": problem.model_dump(mode="json")},
            warnings=list(warnings or []), data_source=data_source,
        )
        return result


def verify_linear_decisions(data, solution, state):
    """独立从原始合同复算变量域、所有约束和目标，不读取求解器汇总指标。"""
    from optiagent.solution_verifier import ABS_TOLERANCE

    problem = LinearModel.model_validate(data)
    rows = solution.decisions
    names = [row.get("variable") for row in rows]
    expected = {variable.name for variable in problem.variables}
    state.check("variables_complete", len(names) == len(set(names)) and set(names) == expected,
                "返回变量必须完整且唯一，不能包含未知变量")
    if not state.checks["variables_complete"]:
        return
    values = {row["variable"]: float(row["value"]) for row in rows}
    finite = all(math.isfinite(value) for value in values.values())
    state.check("finite_values", finite, "变量值必须为有限数值")
    if not finite:
        return
    for variable in problem.variables:
        value = values[variable.name]
        if variable.kind in {"binary", "integer"}:
            error = abs(value - round(value))
            state.check(f"integer:{variable.name}", error <= ABS_TOLERANCE, f"{variable.name} 不满足整数性", error)
        if variable.kind == "binary" or variable.values is not None:
            allowed = variable.values if variable.values is not None else [0, 1]
            error = min(abs(value - candidate) for candidate in allowed)
            state.check(f"domain:{variable.name}", error <= ABS_TOLERANCE, f"{variable.name} 不属于声明的取值集合", error)
        for label, error in (("lower", max(0, variable.lower - value) if variable.lower is not None else 0),
                             ("upper", max(0, value - variable.upper) if variable.upper is not None else 0)):
            state.check(f"{label}:{variable.name}", error <= ABS_TOLERANCE, f"{variable.name} 违反{label}边界", error)
    for index, constraint in enumerate(problem.constraints):
        left = sum(coefficient * values[name] for name, coefficient in constraint.coefficients.items()) + constraint.constant
        difference = left - constraint.rhs
        error = max(0, difference) if constraint.sense == "<=" else max(0, -difference) if constraint.sense == ">=" else abs(difference)
        state.check(f"constraint:{index + 1}", error <= ABS_TOLERANCE, f"第 {index + 1} 条约束不满足：{left:g} {constraint.sense} {constraint.rhs:g}", error)
    state.recomputed_objective = sum(coefficient * values[name] for name, coefficient in problem.objective.coefficients.items()) + problem.objective.constant
