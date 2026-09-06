from __future__ import annotations

from typing import Any

import pandas as pd

from optiagent.data import SupplyChainData, normalize_data, validate_data
from optiagent.mcp_contracts import ValidationReport
from optiagent.mcp_servers.common import json_safe


def validate_problem_data(template_id: str, data: dict[str, Any]) -> ValidationReport:
    """在 Data MCP 与 Solver MCP 间共享的确定性数据校验。"""

    errors: list[str] = []
    warnings: list[str] = []
    checks: dict[str, Any] = {}
    if template_id == "facility_location":
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
    elif template_id == "knapsack":
        items = pd.DataFrame(data.get("items", []))
        capacity = _to_float(data.get("capacity"))
        _require_columns(items, {"item", "value", "weight"}, "items", errors)
        if capacity is None or capacity <= 0:
            errors.append("capacity 必须是正数。")
        _validate_numeric_columns(items, ["value", "weight"], errors)
        if "weight" in items and (pd.to_numeric(items["weight"], errors="coerce") < 0).any():
            errors.append("weight 不能为负数。")
        checks.update({"item_count": len(items), "capacity": capacity})
    elif template_id == "assignment":
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
    elif template_id == "tsp":
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
    elif template_id == "job_shop_scheduling":
        tasks = pd.DataFrame(data.get("tasks", []))
        _require_columns(tasks, {"job", "machine", "duration"}, "tasks", errors)
        _validate_numeric_columns(tasks, ["duration"], errors)
        if not tasks.empty and "duration" in tasks:
            durations = pd.to_numeric(tasks["duration"], errors="coerce")
            if (durations <= 0).any():
                errors.append("duration 必须全部为正数。")
        checks["operation_count"] = len(tasks)
    elif template_id == "production_mix":
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
    else:
        errors.append(f"不支持的模板：{template_id}")
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
