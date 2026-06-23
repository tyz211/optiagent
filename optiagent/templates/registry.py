from __future__ import annotations

from dataclasses import dataclass
import re
from typing import Callable

from optiagent.data import SupplyChainData
from optiagent.problem_spec import DataRequirement, ProblemSpec


SpecBuilder = Callable[[str, SupplyChainData | None, float], ProblemSpec]


@dataclass(frozen=True)
class OptimizationTemplate:
    template_id: str
    display_name: str
    problem_type: str
    keywords: list[str]
    builder: SpecBuilder

    def score(self, question: str, data: SupplyChainData | None = None) -> float:
        lowered = question.lower()
        keyword_hits = sum(1 for keyword in self.keywords if keyword.lower() in lowered)
        score = keyword_hits / max(len(self.keywords), 1)
        if self.template_id == "facility_location" and data is not None:
            score += 0.28
        if self.template_id == "knapsack" and re.search(r"背包|knapsack", lowered):
            score += 0.55
        if self.template_id == "assignment" and re.search(r"指派|匹配|班次|assignment|matching", lowered):
            score += 0.45
        if self.problem_type in {"MILP", "IP"} and re.search(r"整数|0-1|binary|启用|选择|固定成本", lowered):
            score += 0.12
        if self.template_id == "tsp" and re.search(r"旅行商|tsp|巡回|最短回路|访问.*返回", lowered):
            score += 0.62
        if self.template_id == "job_shop_scheduling" and re.search(r"调度|排产|工序|机器|job.?shop|schedule|makespan", lowered):
            score += 0.58
        if self.template_id == "production_mix" and re.search(r"产品组合|生产计划|资源约束|原料|利润最大|产量|mixed|milp", lowered):
            score += 0.58
        return min(score, 1.0)

    def build_spec(self, question: str, data: SupplyChainData | None = None) -> ProblemSpec:
        return self.builder(question, data, max(self.score(question, data), 0.35))


def rank_templates(question: str, data: SupplyChainData | None = None) -> list[OptimizationTemplate]:
    return sorted(TEMPLATES, key=lambda item: item.score(question, data), reverse=True)


def get_template(template_id: str) -> OptimizationTemplate:
    for template in TEMPLATES:
        if template.template_id == template_id:
            return template
    raise KeyError(f"Unknown optimization template: {template_id}")


def list_templates() -> list[OptimizationTemplate]:
    return list(TEMPLATES)


def _facility_location_spec(question: str, data: SupplyChainData | None, confidence: float) -> ProblemSpec:
    missing = []
    assumptions = ["当前版本使用仓库、客户、运输成本三张表作为标准输入。"]
    if data is None:
        missing = ["warehouses.csv", "customers.csv", "costs.csv"]
    return ProblemSpec(
        problem_type="MILP",
        display_name="仓库选址与客户分配",
        objective="最小化运输成本与仓库固定启用成本之和",
        sets=["仓库集合 W", "客户集合 C"],
        parameters=["capacity[w]", "fixed_cost[w]", "demand[c]", "cost[w,c]", "min_open_ratio[w]"],
        decision_variables=["x[w,c]：仓库 w 向客户 c 的发货量", "y[w]：仓库 w 是否启用的 0-1 变量"],
        constraints=[
            "每个客户需求必须被完全满足",
            "仓库发货量不能超过启用后的容量",
            "未启用仓库不能发货",
            "可选：启用后最低运营比例、强制启用、强制关闭",
        ],
        recommended_solver="Gurobi",
        solver_reason="该问题含固定成本和 0-1 启用变量，属于典型 MILP，Gurobi 更适合稳定求解。",
        data_requirements=[
            DataRequirement("warehouses", ["warehouse", "capacity", "fixed_cost"], "候选仓库、容量和启用固定成本"),
            DataRequirement("customers", ["customer", "demand"], "客户需求"),
            DataRequirement("costs", ["warehouse", "customer", "cost"], "每个仓库到每个客户的单位运输成本"),
        ],
        output_schema=["总成本", "运输成本", "固定成本", "启用仓库", "客户分配", "容量利用率"],
        template_id="facility_location",
        confidence=confidence,
        assumptions=assumptions,
        missing_data=missing,
        notes=["这是当前项目已实现的可执行模板。"],
    )


def _knapsack_spec(question: str, data: SupplyChainData | None, confidence: float) -> ProblemSpec:
    return ProblemSpec(
        problem_type="IP",
        display_name="0-1 背包选择问题",
        objective="在容量或预算限制下最大化价值、收益或优先级",
        sets=["物品集合 I"],
        parameters=["value[i]", "weight[i]", "capacity"],
        decision_variables=["z[i]：是否选择物品 i 的 0-1 变量"],
        constraints=["选择物品的总重量或总预算不超过容量", "每个物品最多选择一次"],
        recommended_solver="Gurobi / OR-Tools CP-SAT",
        solver_reason="小中规模 0-1 整数优化可用 CP-SAT；含复杂线性业务约束时 Gurobi 更直接。",
        data_requirements=[DataRequirement("items", ["item", "value", "weight"], "可选择对象、价值和资源消耗")],
        output_schema=["最优价值", "选择清单", "容量使用量", "未选择原因"],
        template_id="knapsack",
        confidence=confidence,
        assumptions=["当前版本可直接从问题 JSON 或上传 CSV 中读取 item/value/weight 数据并求解。"],
    )


def _assignment_spec(question: str, data: SupplyChainData | None, confidence: float) -> ProblemSpec:
    return ProblemSpec(
        problem_type="MILP",
        display_name="指派匹配问题",
        objective="最小化匹配成本或最大化匹配收益",
        sets=["任务集合 T", "资源集合 R"],
        parameters=["cost[r,t]", "eligibility[r,t]"],
        decision_variables=["a[r,t]：资源 r 是否分配给任务 t 的 0-1 变量"],
        constraints=["每个任务被指定数量的资源覆盖", "每个资源最多承担限定数量任务", "技能或资格约束"],
        recommended_solver="Gurobi / OR-Tools CP-SAT",
        solver_reason="指派问题结构清晰，线性 0-1 模型和 CP-SAT 都适合；含成本矩阵时 Gurobi 易扩展。",
        data_requirements=[
            DataRequirement("resources", ["resource"], "候选资源"),
            DataRequirement("tasks", ["task"], "待分配任务"),
            DataRequirement("costs", ["resource", "task", "cost"], "资源到任务的匹配成本"),
        ],
        output_schema=["总成本", "资源-任务匹配表", "未覆盖任务", "资源利用情况"],
        template_id="assignment",
        confidence=confidence,
        assumptions=["当前版本可直接从问题 JSON 或上传 CSV 中读取 resource/task/cost 数据并求解。"],
    )


def _tsp_spec(question: str, data: SupplyChainData | None, confidence: float) -> ProblemSpec:
    return ProblemSpec(
        problem_type="Routing/CP",
        display_name="旅行商路径问题",
        objective="从起点出发访问每个节点一次并返回起点，最小化总距离或成本",
        sets=["节点集合 N", "弧集合 A"],
        parameters=["distance[i,j]"],
        decision_variables=["route[i,j]：是否从节点 i 前往节点 j"],
        constraints=["每个节点恰好进入一次", "每个节点恰好离开一次", "消除子回路", "路径回到起点"],
        recommended_solver="Exact enumeration / Nearest-neighbor heuristic",
        solver_reason="当前实现小规模 TSP 使用固定起点精确枚举；节点较多时使用最近邻启发式生成可行路径。",
        data_requirements=[
            DataRequirement("distances", ["from", "to", "distance"], "节点间距离、时间或成本矩阵"),
        ],
        output_schema=["访问顺序", "总距离", "每段路径成本"],
        template_id="tsp",
        confidence=confidence,
        assumptions=["若距离矩阵是无向图，系统会自动补齐反向边。"],
        notes=["这是当前项目已实现的可执行模板。"],
    )


def _job_shop_scheduling_spec(question: str, data: SupplyChainData | None, confidence: float) -> ProblemSpec:
    return ProblemSpec(
        problem_type="Scheduling",
        display_name="作业车间调度问题",
        objective="安排每个作业的工序开始时间，在机器互斥和工序顺序约束下最小化最大完工时间",
        sets=["作业集合 J", "机器集合 M", "工序集合 O"],
        parameters=["duration[o]", "machine[o]", "order[o]"],
        decision_variables=["start[o]：工序开始时间", "end[o]：工序结束时间", "interval[o]：工序占用机器区间"],
        constraints=["同一作业内工序按顺序执行", "同一机器同一时间最多加工一道工序", "工序开始和结束时间满足加工时长"],
        recommended_solver="List scheduling heuristic",
        solver_reason="当前实现使用稳定的列表调度启发式生成可行排产方案；结果标记为 FEASIBLE，不承诺全局最优。",
        data_requirements=[
            DataRequirement("tasks", ["job", "machine", "duration", "order"], "每道工序所属作业、机器、时长和顺序"),
        ],
        output_schema=["工序甘特表", "最大完工时间", "机器占用计划"],
        template_id="job_shop_scheduling",
        confidence=confidence,
        assumptions=["若未提供 order，会按上传顺序作为同一作业的工序顺序。"],
        notes=["这是当前项目已实现的可执行模板。"],
    )


def _production_mix_spec(question: str, data: SupplyChainData | None, confidence: float) -> ProblemSpec:
    return ProblemSpec(
        problem_type="MILP",
        display_name="产品组合与生产计划问题",
        objective="在原料、工时、预算等资源约束下决定各产品产量，使利润或收益最大",
        sets=["产品集合 P", "资源集合 R"],
        parameters=["profit[p]", "usage[p,r]", "capacity[r]", "demand_min[p]", "demand_max[p]"],
        decision_variables=["q[p]：产品 p 的生产数量", "y[p]：可选，产品 p 是否投产"],
        constraints=["每类资源用量不超过容量", "产量上下界约束", "可选整数或投产 0-1 约束"],
        recommended_solver="Gurobi",
        solver_reason="产品组合通常是线性规划或混合整数线性规划，Gurobi 适合处理连续、整数和 0-1 变量混合场景。",
        data_requirements=[
            DataRequirement("products", ["product", "profit", "resource columns"], "产品收益和每单位产品资源消耗"),
            DataRequirement("capacities", ["resource", "capacity"], "资源容量，可通过 JSON 或 CSV 提供"),
        ],
        output_schema=["最优产量", "最大利润", "资源使用量", "资源剩余量"],
        template_id="production_mix",
        confidence=confidence,
        assumptions=["若未声明 integer=true，默认产量为连续变量。"],
        notes=["这是当前项目已实现的可执行模板。"],
    )


TEMPLATES = [
    OptimizationTemplate(
        "facility_location",
        "仓库选址与客户分配",
        "MILP",
        ["仓库", "选址", "启用", "固定成本", "客户", "分配", "facility", "location"],
        _facility_location_spec,
    ),
    OptimizationTemplate(
        "knapsack",
        "0-1 背包选择问题",
        "IP",
        ["背包", "预算", "选择", "容量", "收益", "价值", "knapsack"],
        _knapsack_spec,
    ),
    OptimizationTemplate(
        "assignment",
        "指派匹配问题",
        "MILP",
        ["指派", "匹配", "人员", "任务", "班次", "assignment", "matching"],
        _assignment_spec,
    ),
    OptimizationTemplate(
        "tsp",
        "旅行商路径问题",
        "Routing/CP",
        ["旅行商", "tsp", "巡回", "访问", "回到起点", "最短回路", "tour"],
        _tsp_spec,
    ),
    OptimizationTemplate(
        "job_shop_scheduling",
        "作业车间调度问题",
        "Scheduling",
        ["调度", "排产", "工序", "机器", "最大完工时间", "job shop", "schedule", "makespan"],
        _job_shop_scheduling_spec,
    ),
    OptimizationTemplate(
        "production_mix",
        "产品组合与生产计划问题",
        "MILP",
        ["产品组合", "生产计划", "资源约束", "原料", "利润最大", "产量", "混合优化", "mixed", "milp"],
        _production_mix_spec,
    ),
]
