from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from optiagent.data import SupplyChainData
from optiagent.solver_config import configure_gurobi_model, get_solver_config, quality_status


@dataclass(frozen=True)
class SolveResult:
    status: str
    objective_value: float | None
    transport_cost: float | None
    fixed_cost: float | None
    allocations: pd.DataFrame
    warehouse_summary: pd.DataFrame
    customer_summary: pd.DataFrame
    message: str
    solver_name: str
    model_type: str
    mip_gap: float | None = None
    optimality_proven: bool = False


def solve_facility_location(data: SupplyChainData, time_limit: int | None = None) -> SolveResult:
    try:
        import gurobipy as gp
        from gurobipy import GRB
    except Exception as exc:
        return _empty_result(
            "GUROBI_UNAVAILABLE",
            f"无法导入 Gurobi: {exc}",
            solver_name="Gurobi",
            model_type="MILP",
        )

    warehouses = data.warehouses.set_index("warehouse")
    customers = data.customers.set_index("customer")
    costs = data.costs.set_index(["warehouse", "customer"])

    warehouse_names = list(warehouses.index)
    customer_names = list(customers.index)

    try:
        model = gp.Model("warehouse_facility_location")
        config = get_solver_config()
        if time_limit is not None:
            config = type(config)(**{**config.__dict__, "time_limit": time_limit})
        configure_gurobi_model(model, config)

        open_var = model.addVars(warehouse_names, vtype=GRB.BINARY, name="open")
        ship = model.addVars(warehouse_names, customer_names, lb=0, vtype=GRB.CONTINUOUS, name="ship")

        transport_expr = gp.quicksum(
            float(costs.loc[(warehouse, customer), "cost"]) * ship[warehouse, customer]
            for warehouse in warehouse_names
            for customer in customer_names
        )
        fixed_expr = gp.quicksum(
            float(warehouses.loc[warehouse, "fixed_cost"]) * open_var[warehouse]
            for warehouse in warehouse_names
        )
        model.setObjective(transport_expr + fixed_expr, GRB.MINIMIZE)

        for customer in customer_names:
            model.addConstr(
                gp.quicksum(ship[warehouse, customer] for warehouse in warehouse_names)
                == float(customers.loc[customer, "demand"]),
                name=f"demand_{customer}",
            )

        for warehouse in warehouse_names:
            capacity = float(warehouses.loc[warehouse, "capacity"])
            min_open_ratio = float(warehouses.loc[warehouse, "min_open_ratio"])
            model.addConstr(
                gp.quicksum(ship[warehouse, customer] for customer in customer_names)
                <= capacity * open_var[warehouse],
                name=f"capacity_{warehouse}",
            )
            if min_open_ratio > 0:
                model.addConstr(
                    gp.quicksum(ship[warehouse, customer] for customer in customer_names)
                    >= capacity * min_open_ratio * open_var[warehouse],
                    name=f"min_open_{warehouse}",
                )
            if int(warehouses.loc[warehouse, "force_open"]) == 1:
                model.addConstr(open_var[warehouse] == 1, name=f"force_open_{warehouse}")
            if int(warehouses.loc[warehouse, "force_closed"]) == 1:
                model.addConstr(open_var[warehouse] == 0, name=f"force_closed_{warehouse}")

        model.optimize()
    except gp.GurobiError as exc:
        return _empty_result(
            "GUROBI_ERROR",
            f"Gurobi 求解异常: {exc}",
            solver_name="Gurobi",
            model_type="MILP",
        )

    status = quality_status(_gurobi_status_name(model.Status), _safe_mip_gap(model))
    usable_statuses = {
        GRB.OPTIMAL,
        GRB.TIME_LIMIT,
        GRB.SUBOPTIMAL,
        GRB.NODE_LIMIT,
        GRB.ITERATION_LIMIT,
        GRB.SOLUTION_LIMIT,
        GRB.INTERRUPTED,
    }
    if model.Status not in usable_statuses:
        return _empty_result(
            status,
            f"仓库选址 MILP 未得到可用解，状态为 {status}。",
            solver_name="Gurobi",
            model_type="MILP",
        )
    if model.SolCount == 0:
        return _empty_result(
            status,
            f"仓库选址 MILP 未返回可行解，状态为 {status}。",
            solver_name="Gurobi",
            model_type="MILP",
        )

    allocation_rows = []
    for warehouse in warehouse_names:
        for customer in customer_names:
            quantity = ship[warehouse, customer].X
            if quantity > 1e-6:
                unit_cost = float(costs.loc[(warehouse, customer), "cost"])
                allocation_rows.append(
                    {
                        "warehouse": warehouse,
                        "customer": customer,
                        "quantity": quantity,
                        "unit_cost": unit_cost,
                        "total_cost": quantity * unit_cost,
                    }
                )

    allocations = pd.DataFrame(allocation_rows)
    if allocations.empty:
        allocations = pd.DataFrame(
            columns=["warehouse", "customer", "quantity", "unit_cost", "total_cost"]
        )

    open_decisions = {
        warehouse: int(round(open_var[warehouse].X))
        for warehouse in warehouse_names
    }
    transport_cost = float(sum(row["total_cost"] for row in allocation_rows))
    fixed_cost = float(
        sum(
            float(warehouses.loc[warehouse, "fixed_cost"]) * open_decisions[warehouse]
            for warehouse in warehouse_names
        )
    )

    warehouse_summary = _warehouse_summary(data, allocations, open_decisions)
    customer_summary = _customer_summary(data, allocations)

    if status == "OPTIMAL":
        message = "Gurobi 已找到最优仓库选址方案。"
    elif status == "NEAR_OPTIMAL":
        message = f"Gurobi 已找到接近最优的仓库选址方案，当前 MIPGap 约为 {_safe_mip_gap(model):.2%}。"
    else:
        message = "Gurobi 已找到可行仓库选址方案。"
    return SolveResult(
        status=status,
        objective_value=float(model.ObjVal),
        transport_cost=transport_cost,
        fixed_cost=fixed_cost,
        allocations=allocations,
        warehouse_summary=warehouse_summary,
        customer_summary=customer_summary,
        message=message,
        solver_name=f"Gurobi {'.'.join(map(str, gp.gurobi.version()))}",
        model_type="MILP",
        mip_gap=_safe_mip_gap(model),
        optimality_proven=status == "OPTIMAL",
    )


def _warehouse_summary(
    data: SupplyChainData,
    allocations: pd.DataFrame,
    open_decisions: dict[str, int],
) -> pd.DataFrame:
    shipped = (
        allocations.groupby("warehouse", as_index=False)["quantity"].sum()
        if not allocations.empty
        else pd.DataFrame(columns=["warehouse", "quantity"])
    )
    summary = data.warehouses.merge(shipped, on="warehouse", how="left")
    summary["quantity"] = summary["quantity"].fillna(0)
    summary["is_open"] = summary["warehouse"].map(open_decisions).fillna(0).astype(int)
    summary["active_fixed_cost"] = summary["fixed_cost"] * summary["is_open"]
    summary["remaining_capacity"] = summary["capacity"] - summary["quantity"]
    summary["utilization"] = summary["quantity"] / summary["capacity"].replace(0, pd.NA)
    return summary.rename(columns={"quantity": "used_capacity"})


def _customer_summary(data: SupplyChainData, allocations: pd.DataFrame) -> pd.DataFrame:
    received = (
        allocations.groupby("customer", as_index=False)["quantity"].sum()
        if not allocations.empty
        else pd.DataFrame(columns=["customer", "quantity"])
    )
    summary = data.customers.merge(received, on="customer", how="left")
    summary["quantity"] = summary["quantity"].fillna(0)
    summary["gap"] = summary["quantity"] - summary["demand"]
    return summary.rename(columns={"quantity": "received"})


def _empty_result(
    status: str,
    message: str,
    solver_name: str = "Unknown",
    model_type: str = "Unknown",
) -> SolveResult:
    return SolveResult(
        status=status,
        objective_value=None,
        transport_cost=None,
        fixed_cost=None,
        allocations=pd.DataFrame(),
        warehouse_summary=pd.DataFrame(),
        customer_summary=pd.DataFrame(),
        message=message,
        solver_name=solver_name,
        model_type=model_type,
    )


def _safe_mip_gap(model) -> float | None:
    try:
        gap = float(model.MIPGap)
    except Exception:
        return None
    if pd.isna(gap):
        return None
    return gap


def _gurobi_status_name(status_code: int) -> str:
    statuses = {
        1: "LOADED",
        2: "OPTIMAL",
        3: "INFEASIBLE",
        4: "INF_OR_UNBD",
        5: "UNBOUNDED",
        6: "CUTOFF",
        7: "ITERATION_LIMIT",
        8: "NODE_LIMIT",
        9: "TIME_LIMIT",
        10: "SOLUTION_LIMIT",
        11: "INTERRUPTED",
        12: "NUMERIC",
        13: "SUBOPTIMAL",
        14: "INPROGRESS",
        15: "USER_OBJ_LIMIT",
    }
    return statuses.get(status_code, f"UNKNOWN_{status_code}")
