from __future__ import annotations

from typing import Any

import pandas as pd

from optiagent.data import SupplyChainData, normalize_data, validate_data
from optiagent.mcp_contracts import ValidationReport
from optiagent.mcp_servers.common import json_safe


def validate_transportation(data: dict[str, Any]) -> ValidationReport:
    """独立校验 transportation 的数据，保持原有业务边界。"""
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    # 数据完整性与数学可行性分开：供不应求仍交给 Solver 报告不可行。
    from optiagent.transportation import TransportationData
    try:
        model = TransportationData.model_validate(data)
        checks.update(supplier_count=len(model.suppliers), consumer_count=len(model.consumers),
                      route_count=len(model.routes), forbidden_count=len(model.forbidden_routes))
    except ValueError as exc:
        errors.append(f"运输分配数据不完整或不受支持：{str(exc)[:1800]}")
    return ValidationReport(valid=not errors, errors=errors, warnings=warnings, checks=json_safe(checks))


def validate_linear_program(data: dict[str, Any]) -> ValidationReport:
    """独立校验 linear_program 的数据，保持原有业务边界。"""
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    # 同一合同校验用于文本、JSON 和 MCP，禁止绕过变量引用及数值检查。
    from optiagent.linear_model import LinearModel
    try:
        model = LinearModel.model_validate(data)
        checks.update(variable_count=len(model.variables), constraint_count=len(model.constraints))
    except ValueError as exc:
        errors.append(f"线性模型不完整或格式错误：{str(exc)[:1800]}")
    return ValidationReport(valid=not errors, errors=errors, warnings=warnings, checks=json_safe(checks))


def validate_facility_location(data: dict[str, Any]) -> ValidationReport:
    """独立校验 facility_location 的数据，保持原有业务边界。"""
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    try:
        supply_data = normalize_data(
            SupplyChainData(
                warehouses=pd.DataFrame(data.get("warehouses", [])),
                customers=pd.DataFrame(data.get("customers", [])),
                costs=pd.DataFrame(data.get("costs", [])),
            )
        )
        errors.extend(validate_data(supply_data))
        checks.update(
            {
                "warehouse_count": len(supply_data.warehouses),
                "customer_count": len(supply_data.customers),
                "total_capacity": float(supply_data.warehouses["capacity"].sum()),
                "total_demand": float(supply_data.customers["demand"].sum()),
            }
        )
    except Exception as exc:
        errors.append(f"仓库选址数据无法标准化：{exc}")
    return ValidationReport(valid=not errors, errors=errors, warnings=warnings, checks=json_safe(checks))


def validate_knapsack(data: dict[str, Any]) -> ValidationReport:
    """独立校验 knapsack 的数据，保持原有业务边界。"""
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    items = pd.DataFrame(data.get("items", []))
    capacity = _to_float(data.get("capacity"))
    _require_columns(items, {"item", "value", "weight"}, "items", errors)
    if capacity is None or capacity <= 0:
        errors.append("capacity 必须是正数。")
    _validate_numeric_columns(items, ["value", "weight"], errors)
    if "weight" in items and (pd.to_numeric(items["weight"], errors="coerce") < 0).any():
        errors.append("weight 不能为负数。")
    checks.update({"item_count": len(items), "capacity": capacity})
    return ValidationReport(valid=not errors, errors=errors, warnings=warnings, checks=json_safe(checks))


def validate_assignment(data: dict[str, Any]) -> ValidationReport:
    """独立校验 assignment 的数据，保持原有业务边界。"""
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    costs = pd.DataFrame(data.get("costs", []))
    resources = [str(item) for item in data.get("resources", [])]
    tasks = [str(item) for item in data.get("tasks", [])]
    _require_columns(costs, {"resource", "task", "cost"}, "costs", errors)
    _validate_numeric_columns(costs, ["cost"], errors)
    if not resources or not tasks:
        errors.append("resources 和 tasks 不能为空。")
    if len(resources) < len(tasks):
        errors.append("资源数少于任务数，在每个资源最多承担一个任务的约束下不可行。")
    if not costs.empty and {"resource", "task"}.issubset(costs.columns):
        pairs = set(zip(costs["resource"].astype(str), costs["task"].astype(str), strict=False))
        missing = [(resource, task) for resource in resources for task in tasks if (resource, task) not in pairs]
        if missing:
            errors.append("成本矩阵缺少组合：" + "、".join(f"{a}->{b}" for a, b in missing[:8]))
    checks.update({"resource_count": len(resources), "task_count": len(tasks)})
    return ValidationReport(valid=not errors, errors=errors, warnings=warnings, checks=json_safe(checks))


def validate_tsp(data: dict[str, Any]) -> ValidationReport:
    """独立校验 tsp 的数据，保持原有业务边界。"""
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    distances = pd.DataFrame(data.get("distances", []))
    matrix = data.get("distance_matrix")
    if matrix is None:
        _require_columns(distances, {"from", "to", "distance"}, "distances", errors)
        _validate_numeric_columns(distances, ["distance"], errors)
        nodes = set(distances.get("from", [])) | set(distances.get("to", []))
        checks["node_count"] = len(nodes)
        if len(nodes) < 2:
            errors.append("TSP 至少需要两个节点。")
    else:
        size = len(matrix) if isinstance(matrix, list) else 0
        if not size or any(not isinstance(row, list) or len(row) != size for row in matrix):
            errors.append("distance_matrix 必须是非空方阵。")
        checks["node_count"] = size
    return ValidationReport(valid=not errors, errors=errors, warnings=warnings, checks=json_safe(checks))


def validate_job_shop_scheduling(data: dict[str, Any]) -> ValidationReport:
    """独立校验 job_shop_scheduling 的数据，保持原有业务边界。"""
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    tasks = pd.DataFrame(data.get("tasks", []))
    _require_columns(tasks, {"job", "machine", "duration"}, "tasks", errors)
    _validate_numeric_columns(tasks, ["duration"], errors)
    if not tasks.empty and "duration" in tasks:
        durations = pd.to_numeric(tasks["duration"], errors="coerce")
        if (durations <= 0).any():
            errors.append("duration 必须全部为正数。")
    checks["operation_count"] = len(tasks)
    return ValidationReport(valid=not errors, errors=errors, warnings=warnings, checks=json_safe(checks))


def validate_production_mix(data: dict[str, Any]) -> ValidationReport:
    """独立校验 production_mix 的数据，保持原有业务边界。"""
    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    products = pd.DataFrame(data.get("products", []))
    capacities = data.get("capacities", {})
    _require_columns(products, {"product", "profit"}, "products", errors)
    _validate_numeric_columns(products, ["profit"], errors)
    if not capacities:
        errors.append("capacities 不能为空。")
    capacity_names = set(capacities) if isinstance(capacities, dict) else {
        str(item.get("resource")) for item in capacities if isinstance(item, dict)
    }
    missing_resources = capacity_names - set(products.columns)
    if missing_resources:
        errors.append("products 缺少资源消耗列：" + "、".join(sorted(missing_resources)))
    _validate_numeric_columns(products, sorted(capacity_names & set(products.columns)), errors)
    checks.update({"product_count": len(products), "resource_count": len(capacity_names)})
    return ValidationReport(valid=not errors, errors=errors, warnings=warnings, checks=json_safe(checks))


def _require_columns(frame: pd.DataFrame, required: set[str], name: str, errors: list[str]) -> None:
    missing = required - set(frame.columns)
    if missing:
        errors.append(f"{name} 缺少字段：" + "、".join(sorted(missing)))


def _validate_numeric_columns(frame: pd.DataFrame, columns: list[str], errors: list[str]) -> None:
    for column in columns:
        if column in frame and pd.to_numeric(frame[column], errors="coerce").isna().any():
            errors.append(f"{column} 存在非数值。")


def _to_float(value: Any) -> float | None:
    try:
        return float(value)
    except (TypeError, ValueError):
        return None
