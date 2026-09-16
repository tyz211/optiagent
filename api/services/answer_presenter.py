from __future__ import annotations

import pandas as pd

from optiagent.data import SupplyChainData
from optiagent.rag import load_knowledge_base, rag_context_pack, rag_summary


def facility_profile(data: SupplyChainData) -> dict:
    """生成仓库网络的数据画像、建议和风险。"""

    total_capacity = float(data.warehouses["capacity"].sum())
    total_demand = float(data.customers["demand"].sum())
    ratio = total_demand / total_capacity if total_capacity else 0.0
    fixed_cost_total = float(data.warehouses["fixed_cost"].sum()) if "fixed_cost" in data.warehouses else 0.0
    warehouse_count = int(len(data.warehouses))
    customer_count = int(len(data.customers))
    summary = (
        f"当前数据包含 {warehouse_count} 个候选仓库、{customer_count} 个客户点；"
        f"总容量 {total_capacity:,.0f}，总需求 {total_demand:,.0f}，需求/容量比 {ratio:.1%}。"
    )

    recommendations: list[str] = []
    risks: list[str] = []
    if ratio > 0.9:
        risks.append("总需求接近或超过总容量，建议优先关注扩容、应急仓或需求波动缓冲。")
    elif ratio < 0.45:
        recommendations.append("整体容量冗余较高，可重点评估高固定成本仓库是否需要保留。")
    else:
        recommendations.append("整体容量与需求比例处于可分析区间，建议进一步比较固定成本与运输成本权衡。")

    capacity_rank = data.warehouses.sort_values("capacity", ascending=False).head(3)
    if not capacity_rank.empty:
        names = ", ".join(capacity_rank["warehouse"].astype(str).tolist())
        recommendations.append(f"容量较大的候选仓库包括：{names}，可作为后续主力仓或区域覆盖分析重点。")

    if "fixed_cost" in data.warehouses and fixed_cost_total:
        high_fixed = data.warehouses.sort_values("fixed_cost", ascending=False).head(3)
        names = ", ".join(
            f"{row.warehouse}({float(row.fixed_cost):,.0f})"
            for row in high_fixed.itertuples(index=False)
        )
        risks.append(f"固定成本较高的仓库包括：{names}，若运输优势不明显，可能拉高总成本。")

    if not data.costs.empty:
        cost_series = pd.to_numeric(data.costs["cost"], errors="coerce").dropna()
        if not cost_series.empty:
            recommendations.append(
                f"运输成本范围为 {cost_series.min():,.0f} 到 {cost_series.max():,.0f}，建议关注高成本线路是否可替代。"
            )

    warehouse_summary = data.warehouses.copy()
    warehouse_summary["used_capacity"] = 0.0
    warehouse_summary["is_open"] = 0
    warehouse_summary["active_fixed_cost"] = 0.0
    warehouse_summary["remaining_capacity"] = warehouse_summary["capacity"]
    warehouse_summary["utilization"] = 0.0
    return {
        "summary": summary,
        "warehouse_count": warehouse_count,
        "customer_count": customer_count,
        "total_capacity": total_capacity,
        "total_demand": total_demand,
        "demand_capacity_ratio": ratio,
        "fixed_cost_total": fixed_cost_total,
        "recommendations": recommendations,
        "risks": risks,
        "warehouse_summary": warehouse_summary.to_dict(orient="records"),
    }


def tool_names_for_web_research(agent_plan: dict | None = None) -> list[str]:
    """返回 Web 研究分支的可审计工具列表。"""

    prefix = ["llm_problem_router"] if agent_plan else ["local_problem_router"]
    return [*prefix, "web_search_tool", "entity_guard", "data_gap_checker"]


def web_research_steps(
    agent_plan: dict | None,
    sources: list[dict],
    data_gaps: list[str],
    search_warning: str | None,
) -> list[dict[str, str]]:
    """将外部数据检索过程转换为前端可展示的步骤。"""

    if agent_plan:
        first = {
            "step": "LLM 识别与路由",
            "tool": "llm_problem_router",
            "output": f"{agent_plan.get('template_id')} / 置信度 {float(agent_plan.get('confidence', 0)):.0%}",
        }
    else:
        first = {
            "step": "本地兜底路由",
            "tool": "local_problem_router",
            "output": "识别为需要外部事实支撑的问题。",
        }
    search_output = search_warning or f"获得 {len(sources)} 条网页来源"
    return [
        first,
        {"step": "外部来源检索", "tool": "web_search_tool", "output": search_output},
        {"step": "实体约束检查", "tool": "entity_guard", "output": "未根据网页结果自动新增优化实体或参数。"},
        {"step": "数据缺口检查", "tool": "data_gap_checker", "output": "；".join(data_gaps[:3])},
    ]


def facility_answer_text(result) -> str:
    """生成仓库选址求解结果的简明文本。"""

    if result.objective_value is None:
        return result.message
    return "\n".join(
        [
            "结论：已基于当前对话上传的 warehouses、customers、costs 数据调用仓库选址 MILP 求解器。",
            f"求解状态：{result.status}。",
            f"总成本：{result.objective_value:,.2f}。",
            f"固定成本：{result.fixed_cost:,.2f}。",
            f"运输成本：{result.transport_cost:,.2f}。",
        ]
    )


def generic_answer_text(generic_result, problem_spec, rag_notes: list[str]) -> str:
    """按通用优化模板生成用户可读的答案摘要。"""

    evidence = "；".join(rag_notes[:2]) if rag_notes else "已检索本地运筹优化知识库。"
    if generic_result.template_id == "knapsack":
        selected = [row for row in generic_result.decisions if row.get("selected") == 1]
        selected_items = ", ".join(_display_item_name(row["item"]) for row in selected) or "无"
        metrics = generic_result.metrics
        return "\n".join(
            [
                "结论：已识别为 0-1 背包问题，并调用通用优化求解工具完成求解。",
                f"最优选择：{selected_items}。",
                f"最大总价值：{metrics.get('selected_value', generic_result.objective_value):,.2f}。",
                f"容量使用：{metrics.get('used_weight'):,.2f}/{metrics.get('capacity'):,.2f}。",
                f"求解器：{generic_result.solver_name}。",
                f"RAG 依据：{evidence}",
            ]
        )
    if generic_result.template_id == "tsp":
        route = generic_result.metrics.get("route", [])
        return "\n".join(
            [
                "结论：已识别为旅行商路径问题，并调用路径优化工具求解。",
                f"推荐路径：{' -> '.join(route)}。",
                f"最短总距离：{generic_result.objective_value:,.2f}。",
                f"RAG 依据：{evidence}",
            ]
        )
    if generic_result.template_id == "job_shop_scheduling":
        return "\n".join(
            [
                "结论：已识别为作业车间调度问题，并调用调度工具生成可行方案。",
                f"最小最大完工时间：{generic_result.objective_value:,.0f}。",
                f"工序数量：{len(generic_result.decisions)}。",
                f"RAG 依据：{evidence}",
            ]
        )
    if generic_result.template_id == "production_mix":
        return "\n".join(
            [
                "结论：已识别为产品组合与生产计划问题，并调用 MILP 求解器求解。",
                f"最大利润：{generic_result.objective_value:,.2f}。",
                generic_result.summary,
                f"RAG 依据：{evidence}",
            ]
        )
    return "\n".join(
        [
            f"结论：已识别为{problem_spec.display_name}，并调用通用优化求解工具完成求解。",
            generic_result.summary,
            f"求解器：{generic_result.solver_name}。",
            f"RAG 依据：{evidence}",
        ]
    )


def rag_summary_for_generic(question: str, template_id: str):
    """为通用模板筛选与问题类型相符的 RAG 证据。"""

    notes, docs = rag_summary(question, top_k=5)
    keyword_map = {
        "knapsack": ["背包", "0-1"],
        "assignment": ["指派", "匹配"],
        "tsp": ["旅行商", "路径", "TSP", "Routing"],
        "job_shop_scheduling": ["调度", "排产", "工序", "CP-SAT"],
        "production_mix": ["产品组合", "生产计划", "MILP", "资源"],
    }
    keywords = keyword_map.get(template_id)
    return _filter_rag(notes, docs, keywords) if keywords else (notes[:3], docs[:3])


def structured_answer(
    answer: str,
    result,
    baseline,
    changes: list[str],
    rag_notes: list[str],
    open_warehouses: list[str],
    problem_spec,
    generic_result,
) -> dict:
    """将求解结果转换为前端统一的结论、指标、建议和风险结构。"""

    if generic_result:
        metrics = generic_result.metrics or {}
        return {
            "conclusion": f"识别为{problem_spec.display_name}；已调用工具求解，状态为 {generic_result.status}。",
            "metrics": {
                "objective_label": generic_result.objective_label or "目标值",
                "total_cost": generic_result.objective_value,
                "transport_cost": None,
                "fixed_cost": None,
                "cost_delta": None,
                "cost_delta_pct": None,
                "mip_gap": metrics.get("mip_gap"),
                "optimality_proven": metrics.get("optimality_proven"),
                "quality_note": metrics.get("quality_note"),
                "extra": _generic_extra_metrics(generic_result),
            },
            "recommendations": [
                "工具调用链：problem_spec_tool -> mcp_gateway -> data_validate_problem -> solver_solve_problem",
                f"推荐求解器：{problem_spec.recommended_solver}",
                f"数据来源：{generic_result.data_source}",
                generic_result.summary,
            ],
            "risks": generic_result.warnings or ["当前通用模板未发现明显风险。"],
            "evidence": rag_notes[:3],
            "raw_answer": answer,
        }

    delta = None
    delta_pct = None
    if result.objective_value is not None and baseline.objective_value:
        delta = result.objective_value - baseline.objective_value
        delta_pct = delta / baseline.objective_value * 100
    risks = []
    if not result.warehouse_summary.empty:
        high_util = result.warehouse_summary[
            result.warehouse_summary["utilization"].fillna(0).astype(float) >= 0.9
        ]["warehouse"].tolist()
        if high_util:
            risks.append(f"高利用率仓库：{', '.join(high_util)}，建议关注容量缓冲。")
    return {
        "conclusion": f"识别为{problem_spec.display_name}；建议启用 {', '.join(open_warehouses) if open_warehouses else '暂无'}；当前方案状态为 {result.status}。",
        "metrics": {
            "total_cost": result.objective_value,
            "transport_cost": result.transport_cost,
            "fixed_cost": result.fixed_cost,
            "cost_delta": delta,
            "cost_delta_pct": delta_pct,
            "mip_gap": result.mip_gap,
            "optimality_proven": result.optimality_proven,
        },
        "recommendations": [
            f"推荐求解器：{problem_spec.recommended_solver}",
            f"启用仓库：{', '.join(open_warehouses)}" if open_warehouses else "当前未找到可启用仓库。",
            *changes,
        ],
        "risks": risks or ["未发现明显容量风险。"],
        "evidence": rag_notes[:3],
        "raw_answer": answer,
    }


def rag_context_preview(pack: dict[str, list[dict]]) -> dict[str, list[dict]]:
    """移除 RAG 正文，只向前端返回来源与得分摘要。"""

    return {
        category: [
            {"title": doc["title"], "score": doc["score"], "source": doc["source"]}
            for doc in docs
        ]
        for category, docs in pack.items()
    }


def rag_context_pack_for_template(question: str, template_id: str) -> dict[str, list[dict]]:
    """按优化模板组织建模、Schema、代码和求解策略证据。"""

    labels = {
        "modeling": "建模知识",
        "schema": "数据要求",
        "template": "代码模板",
        "solver": "求解策略",
    }
    specs = _template_doc_specs(template_id)
    if not specs:
        return rag_context_pack(question)

    docs = load_knowledge_base()
    return {
        labels[category]: [
            _doc_to_context_dict(doc)
            for doc in docs
            if any(pattern == doc.title for pattern in patterns)
        ][:2]
        for category, patterns in specs.items()
    }


def tool_names_for_generic(agent_plan: dict | None = None) -> list[str]:
    """返回通用优化分支的实际或默认工具链。"""

    tools = agent_plan.get("tool_chain") if agent_plan else None
    if isinstance(tools, list) and tools:
        return ["llm_problem_router", *[str(tool) for tool in tools]]
    return [
        "local_problem_router",
        "problem_spec_tool",
        "rag_context_pack_tool",
        "mcp_gateway",
        "data_validate_problem",
        "solver_solve_problem",
    ]


def generic_agent_steps(
    generic_result,
    agent_plan: dict | None = None,
    plan_warning: str | None = None,
) -> list[dict[str, str]]:
    """生成通用模板求解的可审计步骤。"""

    if agent_plan:
        steps = [
            {
                "step": "LLM 识别与路由",
                "tool": "llm_problem_router",
                "output": f"{agent_plan.get('template_id')} / 置信度 {float(agent_plan.get('confidence', 0)):.0%}",
            }
        ]
    else:
        steps = [
            {
                "step": "本地兜底路由",
                "tool": "local_problem_router",
                "output": plan_warning or "未配置 LLM 或 LLM 路由不可用。",
            }
        ]
    steps.extend(
        [
            {"step": "识别问题", "tool": "problem_spec_tool", "output": generic_result.display_name},
            {"step": "检索知识", "tool": "rag_context_pack_tool", "output": "读取建模模板、数据 Schema 与求解策略"},
            {"step": "构建合同", "tool": "mcp_gateway", "output": f"ProblemEnvelope v1.0 / {generic_result.data_source}"},
            {"step": "校验数据", "tool": "data_validate_problem", "output": "统一 Schema 与可行性前置校验"},
            {"step": "执行求解", "tool": "solver_solve_problem", "output": f"{generic_result.solver_name} / {generic_result.status}"},
            {"step": "最优性校验", "tool": "optimality_checker", "output": _optimality_check_text(generic_result)},
            {"step": "结构化结果", "tool": "result_formatter", "output": generic_result.summary},
        ]
    )
    return steps


def structured_agent_steps(problem_spec, generic_result) -> list[dict[str, str]]:
    """生成结构化数据集分支的可审计步骤。"""

    if generic_result:
        return generic_agent_steps(generic_result)
    return [
        {"step": "识别问题", "tool": "problem_spec_tool", "output": f"{problem_spec.display_name} / {problem_spec.problem_type}"},
        {"step": "检索知识", "tool": "rag_context_pack_tool", "output": "读取建模模板、数据 Schema 与求解策略"},
        {"step": "构建合同", "tool": "mcp_gateway", "output": "生成 ProblemEnvelope v1.0"},
        {"step": "校验数据", "tool": "data_validate_problem", "output": "校验仓库、客户、成本表与总体容量"},
        {"step": "执行求解", "tool": "solver_solve_problem", "output": "通过 SolveEnvelope 调用仓库选址 MILP 求解器"},
        {"step": "结构化结果", "tool": "result_formatter", "output": "生成成本、启用仓库、风险与依据"},
    ]


def facility_agent_steps(problem_spec, result, changes: list[str]) -> list[dict[str, str]]:
    """生成仓库选址分支的可审计步骤。"""

    return [
        {
            "step": "本地/LLM 路由",
            "tool": "problem_router",
            "output": "识别为仓库选址与客户分配问题，优先使用当前对话上传数据。",
        },
        {"step": "识别问题", "tool": "problem_spec_tool", "output": f"{problem_spec.display_name} / {problem_spec.problem_type}"},
        {"step": "检索知识", "tool": "rag_context_pack_tool", "output": "读取仓库选址建模模板、数据 Schema 与求解策略。"},
        {"step": "解析数据", "tool": "mcp_gateway", "output": "组装当前对话数据并生成 ProblemEnvelope v1.0。"},
        {"step": "应用业务约束", "tool": "scenario_parser", "output": "；".join(changes) if changes else "无额外场景约束。"},
        {"step": "执行求解", "tool": "solver_solve_problem", "output": f"{result.solver_name} / {result.status}"},
        {"step": "最优性校验", "tool": "optimality_checker", "output": _facility_optimality_check_text(result)},
        {"step": "结构化结果", "tool": "result_formatter", "output": "输出启用仓库、客户分配、总成本、固定成本、运输成本和利用率。"},
    ]


def _filter_rag(notes, docs, keywords: list[str]):
    """按关键词过滤 RAG 证据，无匹配时保留前三条。"""

    filtered = [
        (note, doc)
        for note, doc in zip(notes, docs, strict=False)
        if any(keyword.lower() in f"{doc.title}\n{doc.content}".lower() for keyword in keywords)
    ]
    if filtered:
        filtered_notes, filtered_docs = zip(*filtered, strict=False)
        return list(filtered_notes), list(filtered_docs)
    return notes[:3], docs[:3]


def _display_item_name(item) -> str:
    """为纯数字物品编号补充可读前缀。"""

    text = str(item)
    return f"物品 {text}" if text.isdigit() else text


def _template_doc_specs(template_id: str) -> dict[str, list[str]]:
    """声明每类优化模板需要展示的知识库文档。"""

    return {
        "knapsack": {
            "modeling": ["0-1 背包 IP 建模模板"],
            "schema": ["背包数据 Schema"],
            "template": ["Gurobi 代码模板策略", "OR-Tools 代码模板策略"],
            "solver": ["求解器选择经验"],
        },
        "assignment": {
            "modeling": ["指派匹配 MILP 建模模板"],
            "schema": ["指派数据 Schema"],
            "template": ["Gurobi 代码模板策略", "OR-Tools 代码模板策略"],
            "solver": ["求解器选择经验"],
        },
        "tsp": {
            "modeling": ["旅行商 TSP 建模模板"],
            "schema": ["TSP 数据 Schema"],
            "template": ["OR-Tools 代码模板策略"],
            "solver": ["TSP 求解策略", "求解器选择经验"],
        },
        "job_shop_scheduling": {
            "modeling": ["作业车间调度 CP-SAT 建模模板"],
            "schema": ["调度数据 Schema"],
            "template": ["OR-Tools 代码模板策略"],
            "solver": ["调度求解策略", "求解器选择经验"],
        },
        "production_mix": {
            "modeling": ["产品组合 MILP 建模模板"],
            "schema": ["产品组合数据 Schema"],
            "template": ["Gurobi 代码模板策略"],
            "solver": ["产品组合求解策略", "求解器选择经验"],
        },
    }.get(template_id, {})


def _doc_to_context_dict(doc) -> dict:
    """将知识库文档转换为 API 上下文字典。"""

    return {
        "title": doc.title,
        "score": round(doc.score, 4),
        "source": doc.source,
        "content": doc.content,
    }


def _generic_extra_metrics(generic_result) -> list[dict[str, object]]:
    """返回不同模板特有的补充指标。"""

    metrics = generic_result.metrics or {}
    quality = _quality_extra_metrics(generic_result.status, metrics)
    if generic_result.template_id == "knapsack":
        return [
            {"label": "容量使用", "value": metrics.get("used_weight"), "suffix": f"/{metrics.get('capacity')}"},
            {"label": "剩余容量", "value": metrics.get("remaining_capacity")},
            {"label": "选择数量", "value": metrics.get("selected_count")},
            *quality,
        ]
    if generic_result.template_id == "tsp":
        return [
            {"label": "节点数", "value": metrics.get("node_count")},
            {"label": "路线段数", "value": max(len(metrics.get("route", [])) - 1, 0)},
            *quality,
        ]
    if generic_result.template_id == "job_shop_scheduling":
        return [
            {"label": "作业数", "value": metrics.get("job_count")},
            {"label": "机器数", "value": metrics.get("machine_count")},
            *quality,
        ]
    if generic_result.template_id == "production_mix":
        usage = metrics.get("resource_usage", {})
        return [
            {"label": f"{resource} 使用", "value": summary.get("used"), "suffix": f"/{summary.get('capacity')}"}
            for resource, summary in usage.items()
        ] + quality
    return quality


def _quality_extra_metrics(status: str, metrics: dict) -> list[dict[str, object]]:
    """生成最优性证明与 MIP Gap 指标。"""

    items: list[dict[str, object]] = []
    if metrics.get("optimality_proven") is not None:
        items.append({"label": "最优性证明", "value": "已证明" if metrics.get("optimality_proven") else "未证明"})
    if metrics.get("mip_gap") is not None:
        items.append({"label": "MIP Gap", "value": float(metrics["mip_gap"]), "format": "percent"})
    if status == "NEAR_OPTIMAL" and not any(item["label"] == "最优性证明" for item in items):
        items.append({"label": "最优性证明", "value": "接近最优"})
    return items


def _facility_optimality_check_text(result) -> str:
    """说明仓库选址结果的最优性状态。"""

    if result.status == "OPTIMAL":
        return "Gurobi 返回 OPTIMAL，当前固定成本与运输成本之和已证明最小。"
    if result.status == "NEAR_OPTIMAL":
        return "Gurobi 返回接近最优解，当前 gap 已达到系统设定阈值，可作为大规模问题的高质量方案。"
    if result.status in {"TIME_LIMIT", "SUBOPTIMAL"} and result.objective_value is not None:
        return "已得到可行解，但当前状态未证明全局最优。"
    return result.message or f"求解状态：{result.status}。"


def _optimality_check_text(generic_result) -> str:
    """说明通用模板结果的最优性状态。"""

    if generic_result.status == "OPTIMAL":
        return "求解器返回 OPTIMAL，已按工具能力证明当前目标值最优。"
    if generic_result.status == "NEAR_OPTIMAL":
        return "求解器返回接近最优解，gap 已达到系统设定阈值，适合较大规模数据下使用。"
    if generic_result.template_id == "tsp" and generic_result.metrics.get("optimality_proven") is False:
        return "当前路线由启发式生成，未证明全局最优。"
    if generic_result.status == "FEASIBLE":
        return "已生成可行解，但未证明全局最优。"
    return f"求解状态：{generic_result.status}。"
