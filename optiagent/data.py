from __future__ import annotations

from dataclasses import dataclass

import pandas as pd


@dataclass(frozen=True)
class SupplyChainData:
    warehouses: pd.DataFrame
    customers: pd.DataFrame
    costs: pd.DataFrame

def normalize_data(data: SupplyChainData) -> SupplyChainData:
    warehouses = data.warehouses.copy()
    customers = data.customers.copy()
    costs = data.costs.copy()

    for frame in (warehouses, customers, costs):
        frame.columns = [str(column).strip().lower().lstrip("\ufeff") for column in frame.columns]

    warehouses = _rename_columns(
        warehouses,
        {
            "warehouse": {
                "warehouse",
                "warehouse_id",
                "warehouse_name",
                "facility",
                "facility_id",
                "facility_name",
                "site",
                "site_id",
                "site_name",
                "仓库",
                "仓库id",
                "仓库编号",
                "仓库名称",
                "候选仓库",
                "设施",
                "设施点",
                "网点",
            },
            "capacity": {"capacity", "cap", "supply", "产能", "容量", "仓库容量", "可用容量", "供应量"},
            "fixed_cost": {
                "fixed_cost",
                "fixed cost",
                "fixedcost",
                "open_cost",
                "opening_cost",
                "setup_cost",
                "启动成本",
                "启用成本",
                "开仓成本",
                "固定成本",
                "建设成本",
                "运营成本",
            },
            "region": {"region", "区域", "地区", "城市"},
            "min_open_ratio": {"min_open_ratio", "min utilization", "最低利用率", "最低使用率"},
            "force_open": {"force_open", "must_open", "强制启用", "必须启用"},
            "force_closed": {"force_closed", "must_close", "强制关闭", "必须关闭"},
        },
    )
    customers = _rename_columns(
        customers,
        {
            "customer": {
                "customer",
                "customer_id",
                "customer_name",
                "demand_point",
                "client",
                "客户",
                "客户id",
                "客户编号",
                "客户名称",
                "需求点",
                "门店",
                "门店编号",
                "城市",
            },
            "demand": {"demand", "qty", "quantity", "需求", "需求量", "订单量", "销量"},
        },
    )
    costs = _rename_columns(
        costs,
        {
            "warehouse": {
                "warehouse",
                "warehouse_id",
                "warehouse_name",
                "facility",
                "facility_id",
                "facility_name",
                "site",
                "仓库",
                "仓库id",
                "仓库编号",
                "仓库名称",
                "候选仓库",
                "设施",
                "设施点",
                "起点",
                "from",
                "source",
            },
            "customer": {
                "customer",
                "customer_id",
                "customer_name",
                "demand_point",
                "client",
                "客户",
                "客户id",
                "客户编号",
                "客户名称",
                "需求点",
                "门店",
                "终点",
                "to",
                "destination",
            },
            "cost": {
                "cost",
                "unit_cost",
                "transport_cost",
                "shipping_cost",
                "运输成本",
                "单位成本",
                "单位运输成本",
                "配送成本",
                "费用",
            },
        },
    )

    for column in ["warehouse"]:
        warehouses[column] = warehouses[column].astype(str).str.strip()
        costs[column] = costs[column].astype(str).str.strip()

    customers["customer"] = customers["customer"].astype(str).str.strip()
    costs["customer"] = costs["customer"].astype(str).str.strip()

    warehouses["capacity"] = pd.to_numeric(warehouses["capacity"], errors="raise")
    if "fixed_cost" not in warehouses.columns:
        warehouses["fixed_cost"] = 0.0
    if "min_open_ratio" not in warehouses.columns:
        warehouses["min_open_ratio"] = 0.0
    if "force_open" not in warehouses.columns:
        warehouses["force_open"] = 0
    if "force_closed" not in warehouses.columns:
        warehouses["force_closed"] = 0
    if "region" not in warehouses.columns:
        warehouses["region"] = "未分区"

    warehouses["fixed_cost"] = pd.to_numeric(warehouses["fixed_cost"], errors="raise")
    warehouses["min_open_ratio"] = pd.to_numeric(warehouses["min_open_ratio"], errors="raise")
    warehouses["force_open"] = pd.to_numeric(warehouses["force_open"], errors="raise").astype(int)
    warehouses["force_closed"] = pd.to_numeric(warehouses["force_closed"], errors="raise").astype(int)
    warehouses["region"] = warehouses["region"].astype(str).str.strip()

    customers["demand"] = pd.to_numeric(customers["demand"], errors="raise")
    costs["cost"] = pd.to_numeric(costs["cost"], errors="raise")

    return SupplyChainData(warehouses=warehouses, customers=customers, costs=costs)


def _rename_columns(frame: pd.DataFrame, aliases: dict[str, set[str]]) -> pd.DataFrame:
    rename: dict[str, str] = {}
    normalized_aliases = {
        target: {name.strip().lower() for name in names}
        for target, names in aliases.items()
    }
    for column in frame.columns:
        if column in aliases:
            continue
        for target, names in normalized_aliases.items():
            if column in names:
                rename[column] = target
                break
    return frame.rename(columns=rename)


def validate_data(data: SupplyChainData) -> list[str]:
    errors: list[str] = []
    required = {
        "warehouses": (data.warehouses, {"warehouse", "capacity"}),
        "customers": (data.customers, {"customer", "demand"}),
        "costs": (data.costs, {"warehouse", "customer", "cost"}),
    }

    for name, (frame, columns) in required.items():
        missing = columns - set(frame.columns)
        if missing:
            errors.append(f"{name}.csv 缺少列: {', '.join(sorted(missing))}")

    if errors:
        return errors

    if (data.warehouses["capacity"] < 0).any():
        errors.append("仓库 capacity 不能为负数。")
    if (data.warehouses["fixed_cost"] < 0).any():
        errors.append("仓库 fixed_cost 不能为负数。")
    if ((data.warehouses["min_open_ratio"] < 0) | (data.warehouses["min_open_ratio"] > 1)).any():
        errors.append("仓库 min_open_ratio 必须在 0 到 1 之间。")
    if (data.warehouses["force_open"] & data.warehouses["force_closed"]).any():
        errors.append("同一个仓库不能同时 force_open=1 且 force_closed=1。")
    if (data.customers["demand"] < 0).any():
        errors.append("客户 demand 不能为负数。")
    if (data.costs["cost"] < 0).any():
        errors.append("运输 cost 不能为负数。")

    warehouse_set = set(data.warehouses["warehouse"])
    customer_set = set(data.customers["customer"])
    cost_warehouse_set = set(data.costs["warehouse"])
    cost_customer_set = set(data.costs["customer"])

    unknown_warehouses = cost_warehouse_set - warehouse_set
    unknown_customers = cost_customer_set - customer_set
    if unknown_warehouses:
        errors.append(f"costs.csv 中存在未知仓库: {', '.join(sorted(unknown_warehouses))}")
    if unknown_customers:
        errors.append(f"costs.csv 中存在未知客户: {', '.join(sorted(unknown_customers))}")

    required_pairs = {(w, c) for w in warehouse_set for c in customer_set}
    provided_pairs = set(zip(data.costs["warehouse"], data.costs["customer"], strict=False))
    missing_pairs = required_pairs - provided_pairs
    if missing_pairs:
        preview = ", ".join(f"{w}->{c}" for w, c in sorted(missing_pairs)[:8])
        suffix = "..." if len(missing_pairs) > 8 else ""
        errors.append(f"costs.csv 缺少运输成本组合: {preview}{suffix}")

    if data.warehouses["capacity"].sum() < data.customers["demand"].sum():
        errors.append("总仓库容量小于总客户需求，模型大概率不可行。")

    return errors
