from __future__ import annotations

import re
from dataclasses import dataclass

import pandas as pd

from optiagent.data import SupplyChainData
from optiagent.solver import SolveResult


@dataclass(frozen=True)
class Scenario:
    name: str
    data: SupplyChainData
    changes: list[str]
    warnings: list[str]


def apply_what_if(text: str, data: SupplyChainData) -> Scenario:
    query = (text or "").strip()
    if not query:
        return Scenario("基准场景", data, ["未输入 what-if，使用原始数据。"], [])

    scenario = SupplyChainData(
        warehouses=data.warehouses.copy(),
        customers=data.customers.copy(),
        costs=data.costs.copy(),
    )
    changes: list[str] = []
    warnings: list[str] = []
    lowered = query.lower()
    closing_intent = any(
        keyword in lowered for keyword in ["停用", "关闭", "不可用", "down", "disable", "close"]
    )
    opening_intent = any(keyword in query for keyword in ["启用", "开启", "新增", "新开", "保留"]) or any(
        keyword in lowered for keyword in ["open", "enable", "add"]
    )

    demand_percent = _extract_percent_after_keywords(query, ["需求", "demand"])
    if demand_percent is not None:
        factor = 1 + demand_percent / 100
        scenario.customers["demand"] = scenario.customers["demand"] * factor
        changes.append(f"所有客户需求调整 {demand_percent:+.1f}%。")

    capacity_percent = _extract_percent_after_keywords(query, ["容量", "产能", "capacity"])
    warehouse_name = _match_name(
        query,
        scenario.warehouses["warehouse"].tolist(),
        keywords=["启用", "开启", "新增", "新开", "关闭", "停用", "容量", "固定成本", "启用成本", "租金"],
    )
    if capacity_percent is not None:
        factor = 1 + capacity_percent / 100
        if warehouse_name:
            mask = scenario.warehouses["warehouse"] == warehouse_name
            scenario.warehouses.loc[mask, "capacity"] *= factor
            changes.append(f"{warehouse_name} 仓库容量调整 {capacity_percent:+.1f}%。")
        else:
            scenario.warehouses["capacity"] *= factor
            changes.append(f"所有仓库容量调整 {capacity_percent:+.1f}%。")

    fixed_cost_percent = _extract_percent_after_keywords(query, ["固定成本", "启用成本", "租金", "fixed"])
    if fixed_cost_percent is not None:
        factor = 1 + fixed_cost_percent / 100
        if warehouse_name:
            mask = scenario.warehouses["warehouse"] == warehouse_name
            scenario.warehouses.loc[mask, "fixed_cost"] *= factor
            changes.append(f"{warehouse_name} 仓库固定启用成本调整 {fixed_cost_percent:+.1f}%。")
        else:
            scenario.warehouses["fixed_cost"] *= factor
            changes.append(f"所有仓库固定启用成本调整 {fixed_cost_percent:+.1f}%。")

    min_open_ratio = _extract_min_open_ratio(query)
    if min_open_ratio is not None:
        scenario.warehouses["min_open_ratio"] = min_open_ratio
        changes.append(f"每个启用仓库最低使用率设为 {min_open_ratio:.0%}。")

    if closing_intent:
        if warehouse_name:
            mask = scenario.warehouses["warehouse"] == warehouse_name
            scenario.warehouses.loc[mask, "capacity"] = 0
            scenario.warehouses.loc[mask, "force_closed"] = 1
            scenario.warehouses.loc[mask, "force_open"] = 0
            changes.append(f"{warehouse_name} 仓库已停用，容量设为 0。")
        else:
            warnings.append("检测到停用/关闭意图，但没有识别出仓库名称。")

    if opening_intent and not closing_intent:
        if warehouse_name:
            mask = scenario.warehouses["warehouse"] == warehouse_name
            scenario.warehouses.loc[mask, "force_open"] = 1
            scenario.warehouses.loc[mask, "force_closed"] = 0
            changes.append(f"{warehouse_name} 仓库被强制纳入候选启用方案。")
        else:
            warnings.append("检测到启用/新增意图，但没有识别出仓库名称。")

    if not changes:
        warnings.append("暂未识别出可执行的 what-if 修改，已按基准场景求解。")
        changes.append("无数据修改。")

    name = "；".join(changes[:2])
    return Scenario(name=name, data=scenario, changes=changes, warnings=warnings)


def explain_result(
    baseline: SolveResult,
    scenario: SolveResult,
    scenario_changes: list[str],
) -> list[str]:
    notes = []
    notes.extend(scenario_changes)

    if scenario.objective_value is None:
        notes.append("场景没有可行解，优先检查总需求是否超过总容量，或是否停用了关键仓库。")
        return notes

    if baseline.objective_value is not None:
        delta = scenario.objective_value - baseline.objective_value
        pct = delta / baseline.objective_value * 100 if baseline.objective_value else 0
        direction = "增加" if delta >= 0 else "降低"
        notes.append(f"相对基准，总成本{direction} {abs(delta):,.2f}，变化 {pct:+.2f}%。")

    if scenario.transport_cost is not None and scenario.fixed_cost is not None:
        notes.append(
            f"场景成本拆分为运输成本 {scenario.transport_cost:,.2f}，仓库固定启用成本 {scenario.fixed_cost:,.2f}。"
        )

    if not scenario.warehouse_summary.empty:
        utilization = pd.to_numeric(scenario.warehouse_summary["utilization"], errors="coerce").fillna(0)
        high_util = scenario.warehouse_summary[
            utilization >= 0.9
        ]
        if not high_util.empty:
            names = ", ".join(high_util["warehouse"].tolist())
            notes.append(f"容量利用率超过 90% 的仓库: {names}，它们可能是当前方案的瓶颈。")

        idle = scenario.warehouse_summary[scenario.warehouse_summary["used_capacity"] <= 1e-6]
        if not idle.empty:
            names = ", ".join(idle["warehouse"].tolist())
            notes.append(f"未被使用的仓库: {names}，可能是成本较高或容量被设为 0。")

        open_names = scenario.warehouse_summary.loc[
            scenario.warehouse_summary["is_open"] == 1,
            "warehouse",
        ].tolist()
        if open_names:
            notes.append(f"建议启用仓库: {', '.join(open_names)}。")

    top_routes = scenario.allocations.sort_values("total_cost", ascending=False).head(3)
    if not top_routes.empty:
        formatted = ", ".join(
            f"{row.warehouse}->{row.customer}({row.quantity:.0f})"
            for row in top_routes.itertuples(index=False)
        )
        notes.append(f"成本贡献最高的路线: {formatted}。")

    return notes


def _extract_percent_after_keywords(text: str, keywords: list[str]) -> float | None:
    lowered = text.lower()
    negative_words = {"降低", "下降", "减少", "下调", "decrease", "down", "reduce"}
    positive_words = {"上涨", "上升", "增加", "提高", "增长", "rise", "increase", "up"}
    direction_words = negative_words | positive_words

    for keyword in keywords:
        keyword_lower = keyword.lower()
        for match in re.finditer(re.escape(keyword_lower), lowered):
            window = lowered[max(0, match.start() - 12): match.end() + 18]
            percent_match = re.search(r"(\d+(?:\.\d+)?)\s*%", window)
            if not percent_match:
                continue
            direction = next((word for word in direction_words if word in window), None)
            if not direction:
                continue
            number = float(percent_match.group(1))
            return -number if direction in negative_words else number
    return None


def _match_name(text: str, names: list[str], keywords: list[str] | None = None) -> str | None:
    lowered = text.lower()
    aliases = {
        "上海": "Shanghai",
        "北京": "Beijing",
        "广州": "Guangzhou",
        "成都": "Chengdu",
        "南京": "Nanjing",
        "武汉": "Wuhan",
        "杭州": "Hangzhou",
        "深圳": "Shenzhen",
        "重庆": "Chongqing",
        "西安": "Xi'an",
        "青岛": "Qingdao",
        "天津": "Tianjin",
        "苏州": "Suzhou",
    }
    candidates: list[tuple[int, str]] = []
    keyword_positions = [
        match.start()
        for keyword in (keywords or [])
        for match in re.finditer(re.escape(keyword), text)
    ]
    for alias, canonical in aliases.items():
        if canonical not in names:
            continue
        for match in re.finditer(re.escape(alias), text):
            distance = min((abs(match.start() - pos) for pos in keyword_positions), default=10_000)
            candidates.append((distance, canonical))

    for name in names:
        for match in re.finditer(re.escape(name.lower()), lowered):
            distance = min((abs(match.start() - pos) for pos in keyword_positions), default=10_000)
            candidates.append((distance, name))

    if candidates and keyword_positions:
        candidates.sort(key=lambda item: item[0])
        if candidates[0][0] <= 8:
            return candidates[0][1]

    for alias, canonical in aliases.items():
        if alias in text and canonical in names:
            return canonical

    for name in names:
        if name.lower() in lowered:
            return name

    return None


def _extract_min_open_ratio(text: str) -> float | None:
    if not any(keyword in text for keyword in ["启用仓", "开启仓", "每个启用", "最低使用", "至少使用", "最少使用"]):
        return None
    match = re.search(r"(?:至少|最少|最低)?\s*(\d+(?:\.\d+)?)\s*%", text)
    if not match:
        return None
    ratio = float(match.group(1)) / 100
    if ratio < 0 or ratio > 1:
        return None
    return ratio
