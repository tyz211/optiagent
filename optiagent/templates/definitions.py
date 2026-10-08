"""内置模板说明与建模规格；运行时发现统一由扩展注册表负责。"""

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
        if self.template_id == "linear_program":
            # 数学模型有独立入口，不将多约束整数模型误判成单容量背包。
            from optiagent.linear_model import looks_like_linear_model
            return 0.95 if looks_like_linear_model(question) or '"variables"' in question and '"objective"' in question else 0.0
        if self.template_id == "transportation":
            # 供需分配与仓库启用分开判断，不仅凭“仓库”“客户”就选址。
            return 0.98 if re.search(r"运输分配|运输问题|供给点|需求点|单位运费|transportation", lowered) else 0.0
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


def _transportation_spec(question: str, data: SupplyChainData | None, confidence: float) -> ProblemSpec:
    """只描述本版实际编译的单商品运输约束。"""
    return ProblemSpec(
        problem_type="LP", display_name="运输分配", objective="最小化运输总成本",
        sets=["供给点集合 S", "需求点集合 D", "允许运输线路 A"],
        parameters=["supply[s]", "demand[d]", "cost[s,d]", "quantity_unit", "currency"],
        decision_variables=["flow[s,d]：从供给点到需求点的非负连续运输量"],
        constraints=["各供给点发货量不超过供给上限", "各需求点收货量恰好等于需求", "禁运线路运输量为零"],
        recommended_solver="Gurobi LP", solver_reason="线性运输成本与供需约束可编译为 LP，并独立复算业务限制。",
        data_requirements=[DataRequirement("suppliers", ["name", "supply"], "供给上限"),
                           DataRequirement("consumers", ["name", "demand"], "需求量"),
                           DataRequirement("routes", ["source", "target", "cost"], "每单位运输成本；缺失组合须显式禁运"),
                           DataRequirement("units", ["quantity_unit", "currency"], "统一数量单位与币种")],
        output_schema=["运输量", "运输总成本", "供需验算"], template_id="transportation", confidence=confidence,
        assumptions=["单周期、单商品；允许剩余供给，禁止缺货；数量为连续值。"],
        notes=["最多 200 个供需组合；不支持车辆、时间窗、整数运输量或线路容量。"],
    )


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
        recommended_solver="Gurobi",
        solver_reason="当前背包适配器使用 Gurobi 求解单容量 0-1 选择模型。",
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
        objective="最小化一对一匹配总成本",
        sets=["任务集合 T", "资源集合 R"],
        parameters=["cost[r,t]"],
        decision_variables=["a[r,t]：资源 r 是否分配给任务 t 的 0-1 变量"],
        constraints=["每个任务恰好分配一个资源", "每个资源最多承担一个任务"],
        recommended_solver="Gurobi",
        solver_reason="当前适配器使用 Gurobi 求解完整成本矩阵上的一对一指派。",
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
        recommended_solver="Exact DP / Gurobi MILP / Multi-start 2-opt",
        solver_reason="小规模使用枚举或 Held-Karp；中等规模使用 Gurobi MILP；较大规模使用多起点最近邻与 2-opt。",
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
        recommended_solver="Gurobi MILP / List scheduling fallback",
        solver_reason="当前适配器优先使用 Gurobi MILP；规模较大或精确求解不可用时使用列表调度。",
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
        parameters=["profit[p]", "usage[p,r]", "capacity[r]", "min_qty[p]", "max_qty[p]", "integer"],
        decision_variables=["q[p]：产品 p 的生产数量"],
        constraints=["每类资源用量不超过容量", "min_qty/max_qty 产量上下界", "可选全体产量整数约束"],
        recommended_solver="Gurobi",
        solver_reason="当前适配器使用 Gurobi，按 integer 统一选择连续或整数产量。",
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


def _linear_program_spec(question: str, data: SupplyChainData | None, confidence: float) -> ProblemSpec:
    """数学文本直接形成受限的可执行线性模型合同。"""
    return ProblemSpec(
        problem_type="LP/MILP", display_name="文本线性规划 / 整数规划",
        objective="按照用户显式给出的线性目标最大化或最小化",
        sets=["用户声明的变量"], parameters=["目标系数", "约束系数", "变量取值范围"],
        decision_variables=["用户声明的连续、整数或二元变量"],
        constraints=["逐条保留输入中的线性等式、不等式及变量域"],
        recommended_solver="Gurobi", solver_reason="显式 LP/MILP 合同可直接构建线性模型并独立复算。",
        data_requirements=[DataRequirement("model", ["variables", "objective", "constraints"], "可直接粘贴数学文本或 LaTeX，无需上传表格")],
        output_schema=["目标值", "全部变量值", "逐条约束验算"], template_id="linear_program", confidence=confidence,
    )


BUILTIN_TEMPLATES = [
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
    OptimizationTemplate("linear_program", "文本线性规划 / 整数规划", "LP/MILP", [], _linear_program_spec),
    OptimizationTemplate("transportation", "运输分配", "LP", ["运输分配", "供给点", "需求点", "transportation"], _transportation_spec),
]
