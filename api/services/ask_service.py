from __future__ import annotations

from io import StringIO
import json
import math
import re

import pandas as pd
from fastapi import HTTPException

from api.database import (
    get_active_dataset_id_or_none,
    get_active_llm_config,
    list_uploaded_files,
    load_dataset,
    save_run,
    touch_conversation,
)
from optiagent.generic_solvers import (
    solve_assignment,
    solve_by_problem_spec,
    solve_job_shop_scheduling,
    solve_knapsack,
    solve_production_mix,
    solve_tsp,
)
from optiagent.langchain_agents import run_configured_multi_agent
from optiagent.data import SupplyChainData, normalize_data, validate_data
from optiagent.llm import LLMConfig
from optiagent.problem_spec import infer_problem_spec, spec_summary
from optiagent.rag import load_knowledge_base, rag_context_pack, rag_summary
from optiagent.scenario import apply_what_if, explain_result
from optiagent.schema_mapping import (
    TableMapping,
    apply_table_mapping,
    assemble_facility_data,
    infer_facility_table,
    mapping_summary,
)
from optiagent.solver import solve_facility_location
from optiagent.web_research import web_search


EXECUTABLE_TEMPLATE_IDS = {
    "knapsack",
    "assignment",
    "tsp",
    "job_shop_scheduling",
    "production_mix",
}


def handle_ask(
    question: str,
    requested_dataset_id: int | None,
    mcp_config: str,
    user_id: int | None,
    conversation_id: int | None,
) -> dict:
    dataset_id = requested_dataset_id or get_active_dataset_id_or_none(user_id=user_id, conversation_id=conversation_id)
    llm_config = _active_llm_config(user_id)
    uploaded_context = list_uploaded_files(limit=30, user_id=user_id, conversation_id=conversation_id)

    agent_plan, plan_warning = _llm_agent_plan(question, uploaded_context, llm_config)
    preferred_template = agent_plan.get("template_id") if agent_plan else None
    solver_intent = _should_solve_optimization(question)

    uploaded_generic = _solve_uploaded_generic(question, uploaded_context, preferred_template) if solver_intent else None
    if uploaded_generic and requested_dataset_id is None:
        response = _build_uploaded_generic_response(question, uploaded_generic, agent_plan, plan_warning)
        response["conversation_id"] = conversation_id
        response["run_id"] = save_run(None, question, response["answer"], response, user_id=user_id, conversation_id=conversation_id)
        touch_conversation(conversation_id, question)
        return response

    uploaded_facility = _facility_data_from_uploaded_files(uploaded_context)
    if uploaded_facility and requested_dataset_id is None and _is_facility_question(question, preferred_template) and not solver_intent:
        response = _answer_from_facility_analysis(question, uploaded_facility, plan_warning)
        response["conversation_id"] = conversation_id
        response["run_id"] = save_run(None, question, response["answer"], response, user_id=user_id, conversation_id=conversation_id)
        touch_conversation(conversation_id, question)
        return response

    if uploaded_facility and requested_dataset_id is None and _is_facility_question(question, preferred_template):
        response = _answer_from_facility_data(question, uploaded_facility, llm_config, mcp_config, plan_warning)
        response["conversation_id"] = conversation_id
        response["run_id"] = save_run(None, question, response["answer"], response, user_id=user_id, conversation_id=conversation_id)
        touch_conversation(conversation_id, question)
        return response

    if solver_intent and requested_dataset_id is None and uploaded_context and _is_facility_question(question, preferred_template):
        llm_facility, llm_mapping_warning = _facility_data_from_llm_mapping(question, uploaded_context, llm_config)
        if llm_facility:
            warnings = "; ".join(filter(None, [plan_warning, llm_mapping_warning])) or None
            response = _answer_from_facility_data(question, llm_facility, llm_config, mcp_config, warnings)
            response["conversation_id"] = conversation_id
            response["run_id"] = save_run(None, question, response["answer"], response, user_id=user_id, conversation_id=conversation_id)
            touch_conversation(conversation_id, question)
            return response
        schema_warning = llm_mapping_warning or _facility_schema_diagnostics(uploaded_context)
        if schema_warning:
            response = _answer_schema_confirmation(question, uploaded_context, schema_warning)
            response["conversation_id"] = conversation_id
            response["run_id"] = save_run(None, question, response["answer"], response, user_id=user_id, conversation_id=conversation_id)
            touch_conversation(conversation_id, question)
            return response

    json_generic = _solve_json_generic(question, preferred_template) if solver_intent else None
    if json_generic and requested_dataset_id is None:
        response = _build_uploaded_generic_response(question, json_generic, agent_plan, plan_warning)
        response["conversation_id"] = conversation_id
        response["run_id"] = save_run(None, question, response["answer"], response, user_id=user_id, conversation_id=conversation_id)
        touch_conversation(conversation_id, question)
        return response

    if dataset_id is not None and _is_facility_question(question, preferred_template) and not solver_intent:
        response = _answer_from_facility_analysis(question, load_dataset(dataset_id), plan_warning)
        response["conversation_id"] = conversation_id
        response["run_id"] = save_run(dataset_id, question, response["answer"], response, user_id=user_id, conversation_id=conversation_id)
        touch_conversation(conversation_id, question)
        return response

    if dataset_id is not None and _is_facility_question(question, preferred_template):
        response = _answer_from_structured_dataset(question, dataset_id, llm_config, mcp_config)
        response["conversation_id"] = conversation_id
        response["run_id"] = save_run(dataset_id, question, response["answer"], response, user_id=user_id, conversation_id=conversation_id)
        touch_conversation(conversation_id, question)
        return response

    if _needs_external_data(question) and requested_dataset_id is None:
        response = _answer_from_web_research(question, uploaded_context, agent_plan, plan_warning)
        response["conversation_id"] = conversation_id
        response["run_id"] = save_run(None, question, response["answer"], response, user_id=user_id, conversation_id=conversation_id)
        touch_conversation(conversation_id, question)
        return response

    if uploaded_context and requested_dataset_id is None:
        response = _answer_from_uploaded_files(question, uploaded_context, llm_config)
        response["conversation_id"] = conversation_id
        response["run_id"] = save_run(None, question, response["answer"], response, user_id=user_id, conversation_id=conversation_id)
        touch_conversation(conversation_id, question)
        return response

    if dataset_id is None:
        raise HTTPException(status_code=400, detail="请先上传 CSV 文件。")

    response = _answer_from_structured_dataset(question, dataset_id, llm_config, mcp_config)
    response["conversation_id"] = conversation_id
    response["run_id"] = save_run(dataset_id, question, response["answer"], response, user_id=user_id, conversation_id=conversation_id)
    touch_conversation(conversation_id, question)
    return response


def _build_uploaded_generic_response(question: str, uploaded_generic, agent_plan: dict | None = None, plan_warning: str | None = None) -> dict:
    problem_spec = _problem_spec_for_template(question, uploaded_generic.template_id)
    rag_pack = _rag_context_pack_for_template(question, uploaded_generic.template_id)
    rag_notes, docs = _rag_summary_for_generic(question, uploaded_generic.template_id)
    answer = _generic_answer_text(uploaded_generic, problem_spec, rag_notes)
    warnings = list(uploaded_generic.warnings or [])
    if plan_warning:
        warnings.append(plan_warning)
    return {
        "answer": answer,
        "structured_answer": _structured_answer(answer, None, None, [], rag_notes, [], problem_spec, uploaded_generic),
        "problem_spec": problem_spec.to_dict(),
        "problem_summary": spec_summary(problem_spec),
        "rag_context": _rag_context_preview(rag_pack),
        "generic_result": uploaded_generic.to_dict(),
        "question": question,
        "status": uploaded_generic.status,
        "objective_value": uploaded_generic.objective_value,
        "transport_cost": None,
        "fixed_cost": None,
        "baseline_objective": None,
        "open_warehouses": [],
        "scenario_changes": [],
        "warnings": warnings,
        "explanation": [],
        "rag_notes": rag_notes,
        "rag_docs": [doc.title for doc in docs],
        "tool_names": _tool_names_for_generic(agent_plan),
        "agent_steps": _generic_agent_steps(uploaded_generic, agent_plan, plan_warning),
        "mcp_status": "未使用 MCP。",
        "warehouse_summary": [],
        "allocations": [],
    }


def _answer_from_structured_dataset(
    question: str,
    dataset_id: int,
    llm_config: LLMConfig | None,
    mcp_config: str,
) -> dict:
    data = load_dataset(dataset_id)
    problem_spec = infer_problem_spec(question, data)
    rag_pack = rag_context_pack(question)
    rag_notes, docs = rag_summary(question)
    generic_result = solve_by_problem_spec(question, problem_spec)

    baseline = solve_facility_location(data)
    scenario = apply_what_if(question, data)
    result = solve_facility_location(scenario.data) if generic_result is None else baseline
    agent_result = run_configured_multi_agent(question, data, llm_config, mcp_config)
    display_answer = generic_result.summary if generic_result else agent_result.answer

    open_warehouses = []
    warehouse_summary = []
    allocations = []
    if not result.warehouse_summary.empty:
        open_warehouses = result.warehouse_summary.loc[
            result.warehouse_summary["is_open"] == 1,
            "warehouse",
        ].tolist()
        warehouse_summary = result.warehouse_summary.to_dict(orient="records")
    if not result.allocations.empty:
        allocations = result.allocations.to_dict(orient="records")

    return {
        "answer": display_answer,
        "structured_answer": _structured_answer(display_answer, result, baseline, scenario.changes, rag_notes, open_warehouses, problem_spec, generic_result),
        "problem_spec": problem_spec.to_dict(),
        "problem_summary": spec_summary(problem_spec),
        "rag_context": _rag_context_preview(rag_pack),
        "generic_result": generic_result.to_dict() if generic_result else None,
        "question": question,
        "status": generic_result.status if generic_result else result.status,
        "objective_value": generic_result.objective_value if generic_result else result.objective_value,
        "transport_cost": None if generic_result else result.transport_cost,
        "fixed_cost": None if generic_result else result.fixed_cost,
        "baseline_objective": baseline.objective_value,
        "open_warehouses": open_warehouses,
        "scenario_changes": scenario.changes,
        "warnings": scenario.warnings,
        "explanation": explain_result(baseline, result, scenario.changes),
        "rag_notes": rag_notes,
        "rag_docs": [doc.title for doc in docs],
        "tool_names": agent_result.tool_names,
        "agent_steps": _structured_agent_steps(agent_result, problem_spec, generic_result),
        "mcp_status": agent_result.mcp_status,
        "warehouse_summary": warehouse_summary,
        "allocations": allocations,
    }


def _answer_from_facility_data(
    question: str,
    data: SupplyChainData,
    llm_config: LLMConfig | None,
    mcp_config: str,
    plan_warning: str | None = None,
) -> dict:
    problem_spec = infer_problem_spec(question, data)
    rag_pack = rag_context_pack(question)
    rag_notes, docs = rag_summary(question)
    baseline = solve_facility_location(data)
    scenario = apply_what_if(question, data)
    result = solve_facility_location(scenario.data)
    display_answer = _facility_answer_text(result)
    open_warehouses = []
    warehouse_summary = []
    allocations = []
    if not result.warehouse_summary.empty:
        open_warehouses = result.warehouse_summary.loc[
            result.warehouse_summary["is_open"] == 1,
            "warehouse",
        ].tolist()
        warehouse_summary = result.warehouse_summary.to_dict(orient="records")
    if not result.allocations.empty:
        allocations = result.allocations.to_dict(orient="records")
    warnings = list(scenario.warnings)
    if plan_warning:
        warnings.append(plan_warning)
    return {
        "answer": display_answer,
        "structured_answer": _structured_answer(display_answer, result, baseline, scenario.changes, rag_notes, open_warehouses, problem_spec, None),
        "problem_spec": problem_spec.to_dict(),
        "problem_summary": spec_summary(problem_spec),
        "rag_context": _rag_context_preview(rag_pack),
        "generic_result": None,
        "question": question,
        "status": result.status,
        "objective_value": result.objective_value,
        "transport_cost": result.transport_cost,
        "fixed_cost": result.fixed_cost,
        "baseline_objective": baseline.objective_value,
        "open_warehouses": open_warehouses,
        "scenario_changes": scenario.changes,
        "warnings": warnings,
        "explanation": explain_result(baseline, result, scenario.changes),
        "rag_notes": rag_notes,
        "rag_docs": [doc.title for doc in docs],
        "tool_names": ["local_problem_router", "problem_spec_tool", "rag_context_pack_tool", "data_parser", "gurobi_facility_location_tool"],
        "agent_steps": _facility_agent_steps(problem_spec, result, scenario.changes),
        "mcp_status": "未使用 MCP。",
        "warehouse_summary": warehouse_summary,
        "allocations": allocations,
    }


def _answer_from_facility_analysis(question: str, data: SupplyChainData, plan_warning: str | None = None) -> dict:
    problem_spec = infer_problem_spec(question, data)
    rag_pack = rag_context_pack(question)
    rag_notes, docs = rag_summary(question)
    profile = _facility_profile(data)
    answer = "\n".join(
        [
            "结论：已基于当前对话上传的数据完成仓库网络分析，当前问题未要求直接求最优解。",
            profile["summary"],
            *[f"- {item}" for item in profile["recommendations"]],
        ]
    )
    warnings = list(profile["risks"])
    if plan_warning:
        warnings.append(plan_warning)
    return {
        "answer": answer,
        "structured_answer": {
            "conclusion": "已完成仓库数据分析；当前回复不调用优化求解器。",
            "metrics": {
                "objective_label": "数据画像",
                "total_cost": None,
                "transport_cost": None,
                "fixed_cost": profile["fixed_cost_total"],
                "cost_delta": None,
                "cost_delta_pct": None,
                "extra": [
                    {"label": "总容量", "value": profile["total_capacity"]},
                    {"label": "总需求", "value": profile["total_demand"]},
                    {"label": "仓库数", "value": profile["warehouse_count"]},
                    {"label": "客户数", "value": profile["customer_count"]},
                ],
            },
            "recommendations": profile["recommendations"],
            "risks": profile["risks"] or ["当前数据画像未发现明显结构性风险。"],
            "evidence": rag_notes[:3],
            "raw_answer": answer,
        },
        "problem_spec": problem_spec.to_dict(),
        "problem_summary": spec_summary(problem_spec),
        "rag_context": _rag_context_preview(rag_pack),
        "generic_result": None,
        "question": question,
        "status": "DATA_ANALYSIS",
        "objective_value": None,
        "transport_cost": None,
        "fixed_cost": profile["fixed_cost_total"],
        "baseline_objective": None,
        "open_warehouses": [],
        "scenario_changes": [],
        "warnings": warnings,
        "explanation": profile["recommendations"],
        "rag_notes": rag_notes,
        "rag_docs": [doc.title for doc in docs],
        "tool_names": ["local_problem_router", "data_profile_tool", "rag_context_pack_tool", "risk_diagnostic_tool"],
        "agent_steps": [
            {"step": "识别意图", "tool": "local_problem_router", "output": "识别为数据分析/建议类问题，未触发优化求解。"},
            {"step": "读取数据", "tool": "data_profile_tool", "output": profile["summary"]},
            {"step": "检索知识", "tool": "rag_context_pack_tool", "output": "读取仓库选址业务规则与数据 Schema。"},
            {"step": "生成建议", "tool": "risk_diagnostic_tool", "output": "基于容量、需求、固定成本和成本矩阵生成诊断建议。"},
        ],
        "mcp_status": "未使用 MCP。",
        "warehouse_summary": profile["warehouse_summary"],
        "allocations": [],
    }


def _answer_from_uploaded_files(question: str, files: list[dict], llm_config: LLMConfig | None) -> dict:
    context = "\n\n".join(
        [
            f"文件：{item['filename']}\n识别角色：{item.get('role') or '通用数据'}\n列：{', '.join(json.loads(item['columns_json'])) if item.get('columns_json') else ''}\n样本：\n{item['preview_csv']}"
            for item in files
        ]
    )
    if llm_config:
        try:
            from optiagent.llm import call_openai_compatible_chat

            answer = call_openai_compatible_chat(
                llm_config,
                [
                    {
                        "role": "system",
                        "content": "你是运筹优化数据分析 Agent。请只基于用户上传的 CSV 样本和问题回答，必要时说明还缺少哪些字段才能求解。",
                    },
                    {"role": "user", "content": f"用户问题：{question}\n\n上传文件上下文：\n{context}"},
                ],
            ).strip()
        except Exception as exc:
            answer = f"模型服务暂时不可用，已切换为本地规则解析。\n\n{_local_uploaded_file_answer(files)}"
            warnings = [f"LLM 文件回答失败：{type(exc).__name__}"]
        else:
            warnings = []
    else:
        answer = _local_uploaded_file_answer(files)
        warnings = []

    file_names = [item["filename"] for item in files]
    return {
        "answer": answer,
        "structured_answer": {
            "conclusion": "已基于最近上传的 CSV 文件回答。若需要直接求解，请在问题中补充目标、约束或上传完整参数。",
            "metrics": {
                "total_cost": None,
                "transport_cost": None,
                "fixed_cost": None,
                "cost_delta": None,
                "cost_delta_pct": None,
            },
            "recommendations": [answer],
            "risks": ["当前回答基于上传文件样本；如果 CSV 很大，建议补充明确的目标函数和约束。"],
            "evidence": [f"使用文件：{', '.join(file_names)}"],
            "raw_answer": answer,
        },
        "problem_spec": None,
        "problem_summary": [],
        "rag_context": {},
        "generic_result": None,
        "question": question,
        "status": "FILE_ANSWER",
        "objective_value": None,
        "transport_cost": None,
        "fixed_cost": None,
        "baseline_objective": None,
        "open_warehouses": [],
        "scenario_changes": [],
        "warnings": warnings,
        "explanation": [],
        "rag_notes": [],
        "rag_docs": [],
        "tool_names": ["uploaded_file_context"],
        "agent_steps": [
            {"step": "解析上传文件", "tool": "uploaded_file_context", "output": f"读取 {len(files)} 个当前对话文件"},
            {"step": "数据缺口检查", "tool": "data_gap_checker", "output": "未发现可直接求解的完整优化参数"},
        ],
        "mcp_status": "未使用 MCP。",
        "warehouse_summary": [],
        "allocations": [],
    }


def _answer_from_web_research(
    question: str,
    files: list[dict],
    agent_plan: dict | None = None,
    plan_warning: str | None = None,
) -> dict:
    try:
        results = web_search(question, max_results=5)
        search_warning = None
    except Exception as exc:
        results = []
        search_warning = f"Web 搜索失败：{type(exc).__name__}。"

    data_gaps = _external_data_gaps(question, files)
    sources = [item.to_dict() for item in results]
    source_lines = [
        f"- {item.title}：{item.url}" + (f"\n  摘要：{item.snippet}" if item.snippet else "")
        for item in results
    ]
    answer_lines = [
        "结论：已识别为需要真实数据支撑的运筹优化问题，并完成外部来源检索。",
        "当前不会基于搜索结果自动新增候选仓库、客户、需求、容量或成本参数。",
    ]
    if source_lines:
        answer_lines.extend(["可用来源：", *source_lines])
    else:
        answer_lines.append("暂未获得可用网页来源。")
    answer_lines.extend(["仍需补充的数据：", *[f"- {gap}" for gap in data_gaps]])
    answer = "\n".join(answer_lines)
    warnings = [
        "Web 搜索仅提供事实证据，不能替代结构化优化输入。",
        "如需直接求解，请上传或确认候选点、需求点、容量、需求量和成本/距离矩阵。",
    ]
    if plan_warning:
        warnings.append(plan_warning)
    if search_warning:
        warnings.append(search_warning)

    return {
        "answer": answer,
        "structured_answer": {
            "conclusion": "已完成外部数据检索；由于缺少完整优化参数，当前不直接求解。",
            "metrics": {
                "total_cost": None,
                "transport_cost": None,
                "fixed_cost": None,
                "cost_delta": None,
                "cost_delta_pct": None,
            },
            "recommendations": [
                "工具调用链：llm_problem_router -> web_search_tool -> data_gap_checker",
                "使用检索来源补充事实背景；优化实体必须来自用户上传数据或用户确认的来源数据。",
                *data_gaps,
            ],
            "risks": warnings,
            "evidence": [f"{item.title}：{item.url}" for item in results],
            "raw_answer": answer,
        },
        "problem_spec": None,
        "problem_summary": [],
        "rag_context": {},
        "generic_result": None,
        "question": question,
        "status": "WEB_RESEARCH",
        "objective_value": None,
        "transport_cost": None,
        "fixed_cost": None,
        "baseline_objective": None,
        "open_warehouses": [],
        "scenario_changes": [],
        "warnings": warnings,
        "explanation": [],
        "rag_notes": [],
        "rag_docs": [],
        "tool_names": _tool_names_for_web_research(agent_plan),
        "agent_steps": _web_research_steps(agent_plan, sources, data_gaps, search_warning),
        "mcp_status": "未使用 MCP。",
        "warehouse_summary": [],
        "allocations": [],
        "web_sources": sources,
    }


def _answer_schema_confirmation(question: str, files: list[dict], message: str) -> dict:
    file_names = [item["filename"] for item in files]
    answer = "\n".join(
        [
            "结论：当前上传文件存在字段映射歧义，暂不直接求解。",
            message,
            "请补充说明每个文件对应的表类型，以及关键字段含义，例如：仓库名、容量、固定成本、客户名、需求量、单位运输成本。",
        ]
    )
    return {
        "answer": answer,
        "structured_answer": {
            "conclusion": "字段映射需要确认，未调用求解器。",
            "metrics": {
                "total_cost": None,
                "transport_cost": None,
                "fixed_cost": None,
                "cost_delta": None,
                "cost_delta_pct": None,
            },
            "recommendations": [
                "工具调用链：schema_mapping_tool -> data_gap_checker",
                message,
            ],
            "risks": ["为避免误解 CSV 字段含义，系统不会在低置信度映射下直接求解。"],
            "evidence": [f"使用文件：{', '.join(file_names)}"],
            "raw_answer": answer,
        },
        "problem_spec": None,
        "problem_summary": [],
        "rag_context": {},
        "generic_result": None,
        "question": question,
        "status": "NEEDS_SCHEMA_CONFIRMATION",
        "objective_value": None,
        "transport_cost": None,
        "fixed_cost": None,
        "baseline_objective": None,
        "open_warehouses": [],
        "scenario_changes": [],
        "warnings": [message],
        "explanation": [],
        "rag_notes": [],
        "rag_docs": [],
        "tool_names": ["schema_mapping_tool", "data_gap_checker"],
        "agent_steps": [
            {"step": "字段映射", "tool": "schema_mapping_tool", "output": message},
            {"step": "求解保护", "tool": "entity_guard", "output": "字段含义未确认，未调用优化求解器。"},
        ],
        "mcp_status": "未使用 MCP。",
        "warehouse_summary": [],
        "allocations": [],
    }


def _llm_agent_plan(question: str, files: list[dict], llm_config: LLMConfig | None) -> tuple[dict | None, str | None]:
    if not llm_config:
        return None, "未配置 LLM，已使用本地规则路由。"

    file_summaries = []
    for item in files:
        columns = json.loads(item["columns_json"]) if item.get("columns_json") else []
        file_summaries.append(
            {
                "filename": item["filename"],
                "role": item.get("role") or "generic",
                "columns": columns,
                "preview_csv": item.get("preview_csv", "")[:1200],
            }
        )
    prompt = {
        "question": question,
        "uploaded_files": file_summaries,
        "available_problem_templates": [
            "knapsack",
            "assignment",
            "tsp",
            "job_shop_scheduling",
            "production_mix",
            "facility_location",
            "file_answer",
        ],
        "available_tools": [
            "problem_spec_tool",
            "rag_context_pack_tool",
            "web_search_tool",
            "data_parser",
            "generic_optimizer_tool",
            "gurobi_facility_location_tool",
            "uploaded_file_context",
        ],
    }
    try:
        from optiagent.llm import call_openai_compatible_chat

        raw = call_openai_compatible_chat(
            llm_config,
            [
                {
                    "role": "system",
                    "content": (
                        "你是运筹优化 LLM Agent 的路由器。"
                        "请根据用户问题和上传文件，选择问题类型与工具调用计划。"
                        "用户上传数据优先；外部事实只能来自 web_search_tool 或已给定数据。"
                        "当问题明确要求真实/当前/公开数据，或城市选址、市场/物流事实缺少数据支撑时，tool_chain 必须包含 web_search_tool。"
                        "如果用户已经上传 warehouses/customers/costs 或其他可执行模板数据，不要因为仓库选址关键词调用 web_search_tool，应优先调用求解工具。"
                        "web_search_tool 只能提供来源证据，不能自动生成候选仓库、客户、需求、容量、成本或距离矩阵。"
                        "如缺少可求解参数，应选择 file_answer 或 needs_solver=false，并说明缺口。"
                        "当用户要求求解优化问题时，tool_chain 必须包含 generic_optimizer_tool 或 gurobi_facility_location_tool。"
                        "工具计划必须要求求解器返回可证明最优解；如果工具只能给启发式可行解，必须在结果中标记未证明最优。"
                        "只输出 JSON，不要输出 Markdown。"
                        "JSON 字段：template_id, confidence, objective, selected_file, tool_chain, reasoning, needs_solver, data_gaps。"
                        "template_id 只能是 knapsack、assignment、tsp、job_shop_scheduling、production_mix、facility_location、file_answer。"
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        )
        plan = _parse_json_object(raw)
    except Exception as exc:
        return None, f"LLM 路由暂时不可用，已回退到本地规则路由（{type(exc).__name__}）。"

    template_id = str(plan.get("template_id", "")).strip()
    if template_id not in {*EXECUTABLE_TEMPLATE_IDS, "facility_location", "file_answer"}:
        return None, "LLM 路由返回了未知问题类型，已回退到本地规则路由。"
    tool_chain = plan.get("tool_chain")
    if not isinstance(tool_chain, list) or not tool_chain:
        plan["tool_chain"] = _default_tool_chain(template_id)
    has_executable_upload = bool(_facility_data_from_uploaded_files(files) or _has_executable_generic_upload(files, template_id))
    if _needs_external_data(question) and not has_executable_upload and "web_search_tool" not in plan["tool_chain"]:
        insert_at = 1 if plan["tool_chain"] else 0
        plan["tool_chain"].insert(insert_at, "web_search_tool")
    plan["template_id"] = template_id
    plan["confidence"] = _safe_float(plan.get("confidence"), default=0.0)
    plan["llm_used"] = True
    return plan, None


def _parse_json_object(text: str) -> dict:
    cleaned = text.strip()
    cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned)
    cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError("LLM plan is not a JSON object")
    return value


def _safe_float(value, default: float = 0.0) -> float:
    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return max(0.0, min(number, 1.0))


def _default_tool_chain(template_id: str) -> list[str]:
    if template_id in EXECUTABLE_TEMPLATE_IDS:
        return ["problem_spec_tool", "rag_context_pack_tool", "data_parser", "generic_optimizer_tool", "result_formatter"]
    if template_id == "facility_location":
        return ["problem_spec_tool", "rag_context_pack_tool", "data_parser", "gurobi_facility_location_tool", "result_formatter"]
    return ["uploaded_file_context"]


def _should_solve_optimization(question: str) -> bool:
    text = (question or "").lower()
    strong_solve_keywords = [
        "求解",
        "最优",
        "最小化",
        "最大化",
        "最大",
        "最小",
        "minimize",
        "maximize",
        "optimal",
        "solve",
        "选择哪些",
        "选择哪",
        "启用哪些",
        "开启哪些",
        "分配方案",
        "客户分配",
        "访问顺序",
        "路径",
        "路线",
        "排程",
        "排产",
    ]
    optimization_context = ["目标函数", "约束", "方案", "模型", "决策变量", "成本最", "利润最"]
    analysis_keywords = [
        "分析",
        "建议",
        "情况",
        "现状",
        "怎么看",
        "说明",
        "解释",
        "总结",
        "概览",
        "文件",
        "数据里",
        "有什么",
    ]
    if any(keyword in text for keyword in strong_solve_keywords):
        return True
    if "优化" in text and any(keyword in text for keyword in optimization_context):
        return True
    if any(keyword in text for keyword in analysis_keywords):
        return False
    return False


def _needs_external_data(question: str) -> bool:
    lowered = question.lower()
    keywords = [
        "真实数据",
        "公开数据",
        "最新",
        "web",
        "网页",
        "搜索",
        "网上",
        "市场",
        "物流园",
        "仓储",
        "城市选址",
        "门店选址",
        "人口",
        "gdp",
        "经纬度",
        "地址",
    ]
    return any(keyword in lowered for keyword in keywords)


def _external_data_gaps(question: str, files: list[dict]) -> list[str]:
    has_files = bool(files)
    lowered = question.lower()
    if "仓库" in question or "选址" in question or "facility" in lowered:
        gaps = [
            "候选仓库或设施点清单，需要明确每个候选点是否可选。",
            "客户或需求点清单及需求量。",
            "候选点容量、固定成本或建设/运营成本。",
            "候选点到需求点的运输成本、距离、时效或可计算这些成本的坐标。",
        ]
    elif "tsp" in lowered or "旅行商" in question or "路径" in question:
        gaps = [
            "需要访问的节点清单。",
            "节点间距离/时间矩阵，或每个节点的经纬度坐标。",
        ]
    else:
        gaps = [
            "目标函数：最大化收益、最小化成本、最短距离或最小完工时间。",
            "决策对象清单，例如候选设施、任务、产品、路径节点或可选物品。",
            "关键参数表，例如需求、容量、成本、距离、资源消耗和时间。",
            "业务约束，例如预算、容量、服务范围、时间窗、班次或工序顺序。",
        ]
    if has_files:
        return [f"已读取当前对话上传文件，但仍需确认：{gap}" for gap in gaps]
    return gaps


def _facility_profile(data: SupplyChainData) -> dict:
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


def _tool_names_for_web_research(agent_plan: dict | None = None) -> list[str]:
    prefix = ["llm_problem_router"] if agent_plan else ["local_problem_router"]
    return [*prefix, "web_search_tool", "entity_guard", "data_gap_checker"]


def _web_research_steps(
    agent_plan: dict | None,
    sources: list[dict],
    data_gaps: list[str],
    search_warning: str | None,
) -> list[dict[str, str]]:
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


def _problem_spec_for_template(question: str, template_id: str):
    from optiagent.templates.registry import get_template

    if template_id:
        try:
            return get_template(template_id).build_spec(question, None)
        except KeyError:
            pass
    return infer_problem_spec(question, None)


def _solve_uploaded_generic(question: str, files: list[dict], preferred_template: str | None = None):
    lowered = question.lower()
    frames = [_uploaded_item_to_frame(item) for item in files]
    roles = {str(item.get("role") or "") for item in files}
    if _should_try_template(preferred_template, lowered, "knapsack", ["背包", "knapsack", "最大价值"], roles):
        for frame, filename in frames:
            normalized = _normalize_generic_frame(frame)
            columns = {str(column).strip().lower() for column in normalized.columns}
            if {"item", "value", "weight"}.issubset(columns):
                capacity = _extract_capacity(question) or float(normalized["weight"].sum())
                data = {"capacity": capacity, "items": normalized.to_dict(orient="records")}
                return solve_knapsack(data, data_source=f"上传文件：{filename}", warnings=[])
    if _should_try_template(preferred_template, lowered, "assignment", ["指派", "匹配", "assignment"], roles):
        for frame, filename in frames:
            normalized = _normalize_generic_frame(frame)
            columns = {str(column).strip().lower() for column in normalized.columns}
            if {"resource", "task", "cost"}.issubset(columns):
                data = {
                    "resources": sorted(normalized["resource"].astype(str).unique().tolist()),
                    "tasks": sorted(normalized["task"].astype(str).unique().tolist()),
                    "costs": normalized.to_dict(orient="records"),
                }
                return solve_assignment(data, data_source=f"上传文件：{filename}", warnings=[])
    if _should_try_template(preferred_template, lowered, "tsp", ["旅行商", "tsp", "巡回", "最短路径"], roles):
        for frame, filename in frames:
            normalized = _normalize_generic_frame(frame)
            columns = {str(column).strip().lower() for column in normalized.columns}
            if {"from", "to", "distance"}.issubset(columns):
                data = {"distances": normalized.to_dict(orient="records")}
                return solve_tsp(data, data_source=f"上传文件：{filename}", warnings=[])
            coordinate_data = _coordinates_to_tsp_data(normalized)
            if coordinate_data:
                return solve_tsp(coordinate_data, data_source=f"上传文件：{filename}", warnings=["上传文件为坐标表，已按欧氏距离构造 TSP 距离矩阵。"])
    if _should_try_template(preferred_template, lowered, "job_shop_scheduling", ["调度", "排产", "工序", "job", "schedule"], roles):
        for frame, filename in frames:
            normalized = _normalize_generic_frame(frame)
            columns = {str(column).strip().lower() for column in normalized.columns}
            if {"job", "machine", "duration"}.issubset(columns):
                data = {"tasks": normalized.to_dict(orient="records")}
                return solve_job_shop_scheduling(data, data_source=f"上传文件：{filename}", warnings=[])
    if _should_try_template(preferred_template, lowered, "production_mix", ["产品组合", "生产计划", "利润", "资源约束", "milp"], roles):
        production = _extract_production_mix_from_files(frames, question)
        if production:
            data, filename = production
            return solve_production_mix(data, data_source=f"上传文件：{filename}", warnings=[])
    return None


def _should_try_template(
    preferred_template: str | None,
    lowered_question: str,
    template_id: str,
    keywords: list[str],
    uploaded_roles: set[str] | None = None,
) -> bool:
    if preferred_template:
        return preferred_template == template_id
    if uploaded_roles and template_id in uploaded_roles:
        return True
    return any(keyword.lower() in lowered_question for keyword in keywords)


def _is_facility_question(question: str, preferred_template: str | None = None) -> bool:
    if preferred_template == "facility_location":
        return True
    lowered = question.lower()
    keywords = ["仓库", "仓", "选址", "facility", "warehouse", "固定成本", "运输成本", "客户分配", "利用率"]
    return any(keyword in lowered for keyword in keywords)


def _facility_data_from_uploaded_files(files: list[dict]) -> SupplyChainData | None:
    frames: list[tuple[pd.DataFrame, str, str | None]] = []
    for item in files:
        try:
            frame, filename = _uploaded_item_to_frame(item)
        except Exception:
            continue
        frames.append((frame, filename, item.get("role")))
    assembly = assemble_facility_data(frames)
    return assembly.data


def _facility_data_from_llm_mapping(
    question: str,
    files: list[dict],
    llm_config: LLMConfig | None,
) -> tuple[SupplyChainData | None, str | None]:
    if not llm_config:
        return None, None
    frames: dict[str, pd.DataFrame] = {}
    file_summaries = []
    for item in files:
        try:
            frame, filename = _uploaded_item_to_frame(item)
        except Exception:
            continue
        frames[filename] = frame
        file_summaries.append(
            {
                "filename": filename,
                "stored_role": item.get("role"),
                "columns": [str(column) for column in frame.columns],
                "preview_csv": frame.head(8).to_csv(index=False),
            }
        )
    if len(file_summaries) < 3:
        return None, None
    try:
        from optiagent.llm import call_openai_compatible_chat

        raw = call_openai_compatible_chat(
            llm_config,
            [
                {
                    "role": "system",
                    "content": (
                        "你是 schema_mapping_tool，只负责把用户上传 CSV 映射到仓库选址标准 Schema。"
                        "不要编造数据，不要新增实体。只允许选择上传文件已有列。"
                        "标准 Schema：warehouses(warehouse, capacity, fixed_cost 可选)、"
                        "customers(customer, demand)、costs(warehouse, customer, cost)。"
                        "若无法确定，confidence 低于 0.72 并在 warnings 中说明。"
                        "只输出 JSON，不要 Markdown。"
                        "JSON 字段：confidence, tables。tables 是数组，每项包含 filename, role, columns。"
                        "role 只能是 warehouses/customers/costs/ignore；columns 是 原列名 到 标准列名 的映射。"
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {"question": question, "uploaded_files": file_summaries},
                        ensure_ascii=False,
                    ),
                },
            ],
        )
        payload = _parse_json_object(raw)
    except Exception as exc:
        return None, f"LLM 字段映射失败：{type(exc).__name__}。"

    confidence = _safe_float(payload.get("confidence"), default=0.0)
    if confidence < 0.72:
        return None, f"LLM 字段映射置信度不足（{confidence:.0%}），请确认字段含义。"

    mapped_frames: list[tuple[pd.DataFrame, str, str | None]] = []
    mapping_lines: list[str] = []
    for table in payload.get("tables", []):
        if not isinstance(table, dict):
            continue
        filename = str(table.get("filename") or "")
        role = str(table.get("role") or "")
        if role not in {"warehouses", "customers", "costs"} or filename not in frames:
            continue
        columns = table.get("columns") if isinstance(table.get("columns"), dict) else {}
        mapping = TableMapping(
            role=role,
            confidence=confidence,
            column_mapping={str(source): str(target) for source, target in columns.items()},
            method="llm",
        )
        mapped_frames.append((apply_table_mapping(frames[filename], mapping), filename, role))
        mapping_lines.append(f"{filename}: {mapping_summary(mapping)}")

    assembly = assemble_facility_data(mapped_frames)
    if assembly.data is None:
        detail = "；".join(assembly.warnings[:3])
        return None, f"LLM 字段映射未能通过数据校验：{detail}"
    return assembly.data, "LLM 字段映射已通过校验：" + "；".join(mapping_lines)


def _facility_schema_diagnostics(files: list[dict]) -> str | None:
    frames: list[tuple[pd.DataFrame, str, str | None]] = []
    for item in files:
        try:
            frame, filename = _uploaded_item_to_frame(item)
        except Exception:
            continue
        frames.append((frame, filename, item.get("role")))
    if len(frames) < 3:
        return None
    assembly = assemble_facility_data(frames)
    if assembly.data is not None:
        return None
    local_mappings = []
    for frame, filename, _role in frames[:6]:
        mapping = infer_facility_table(frame, filename)
        if mapping.role or mapping.confidence >= 0.48:
            local_mappings.append(f"{filename}: {mapping_summary(mapping)}")
    if not local_mappings and not assembly.warnings:
        return None
    lines = [
        "本地 schema_mapping_tool 未能把当前文件校验为完整可求解数据。",
        *local_mappings,
    ]
    lines.extend(assembly.warnings[:4])
    return "\n".join(lines)


def _has_executable_generic_upload(files: list[dict], preferred_template: str | None = None) -> bool:
    roles = {str(item.get("role") or "") for item in files}
    if preferred_template in EXECUTABLE_TEMPLATE_IDS and preferred_template in roles:
        return True
    if roles & EXECUTABLE_TEMPLATE_IDS:
        return True
    for item in files:
        try:
            frame, _filename = _uploaded_item_to_frame(item)
        except Exception:
            continue
        normalized = _normalize_generic_frame(frame)
        columns = {str(column).strip().lower() for column in normalized.columns}
        if {"item", "value", "weight"}.issubset(columns):
            return True
        if {"resource", "task", "cost"}.issubset(columns):
            return True
        if {"from", "to", "distance"}.issubset(columns):
            return True
        if {"city", "x", "y"}.issubset(columns):
            return True
        if {"job", "machine", "duration"}.issubset(columns):
            return True
        if {"product", "profit"}.issubset(columns):
            return True
    return False


def _facility_answer_text(result) -> str:
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


def _coordinates_to_tsp_data(frame: pd.DataFrame) -> dict | None:
    columns = {str(column).strip().lower(): column for column in frame.columns}
    city_col = columns.get("city") or columns.get("城市") or columns.get("node") or columns.get("节点")
    x_col = columns.get("x") or columns.get("lng") or columns.get("lon") or columns.get("longitude") or columns.get("经度")
    y_col = columns.get("y") or columns.get("lat") or columns.get("latitude") or columns.get("纬度")
    if not city_col or not x_col or not y_col:
        return None
    points = frame[[city_col, x_col, y_col]].dropna().copy()
    if len(points) < 2:
        return None
    points[x_col] = pd.to_numeric(points[x_col], errors="raise")
    points[y_col] = pd.to_numeric(points[y_col], errors="raise")
    distances = []
    rows = list(points.itertuples(index=False, name=None))
    for i, source in enumerate(rows):
        for j, target in enumerate(rows):
            if i == j:
                continue
            distance = math.dist((float(source[1]), float(source[2])), (float(target[1]), float(target[2])))
            distances.append({"from": str(source[0]), "to": str(target[0]), "distance": distance})
    return {"distances": distances}


def _solve_json_generic(question: str, preferred_template: str | None = None):
    problem_spec = _problem_spec_for_template(question, preferred_template) if preferred_template in EXECUTABLE_TEMPLATE_IDS else infer_problem_spec(question, None)
    result = solve_by_problem_spec(question, problem_spec)
    if result and result.status != "ERROR":
        return result
    return None


def _uploaded_item_to_frame(item: dict) -> tuple[pd.DataFrame, str]:
    csv_text = item.get("content_csv") or item["preview_csv"]
    return pd.read_csv(StringIO(csv_text)), item["filename"]


def _extract_capacity(text: str) -> float | None:
    match = re.search(r"(?:容量|capacity|预算|限制)[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _normalize_generic_frame(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "item": {"item", "物品", "物品编号", "编号", "id", "name", "项目"},
        "value": {"value", "价值", "收益", "效益", "profit"},
        "weight": {"weight", "重量", "资源消耗", "消耗", "成本", "cost_weight"},
        "resource": {"resource", "资源", "员工", "人员", "worker"},
        "task": {"task", "任务", "班次", "岗位", "job"},
        "cost": {"cost", "成本", "匹配成本", "费用"},
        "from": {"from", "起点", "出发", "源点", "source"},
        "to": {"to", "终点", "到达", "目的地", "destination"},
        "distance": {"distance", "距离", "里程", "时间", "时长", "travel_time"},
        "job": {"job", "作业", "订单", "工件"},
        "machine": {"machine", "机器", "设备", "产线"},
        "duration": {"duration", "加工时长", "处理时间", "工时", "时长"},
        "order": {"order", "顺序", "工序顺序", "序号"},
        "product": {"product", "产品", "品类"},
        "profit": {"profit", "利润", "收益", "单位利润"},
        "capacity": {"capacity", "容量", "可用量", "资源上限"},
    }
    rename: dict[str, str] = {}
    for column in frame.columns:
        normalized = str(column).strip().lower().lstrip("\ufeff")
        for canonical, names in aliases.items():
            if normalized in {name.lower() for name in names}:
                rename[column] = canonical
                break
    return frame.rename(columns=rename)


def _extract_production_mix_from_files(frames: list[tuple[pd.DataFrame, str]], question: str):
    for frame, filename in frames:
        normalized = _normalize_generic_frame(frame)
        columns = {str(column).strip().lower() for column in normalized.columns}
        if {"product", "profit"}.issubset(columns):
            capacities = _extract_capacities(question, normalized)
            if capacities:
                return {"products": normalized.to_dict(orient="records"), "capacities": capacities}, filename
    return None


def _extract_capacities(question: str, products: pd.DataFrame) -> dict[str, float]:
    payload_match = re.search(r"capacities?\s*[:=]\s*(\{.*?\})", question, flags=re.IGNORECASE | re.DOTALL)
    if payload_match:
        try:
            value = json.loads(payload_match.group(1))
            if isinstance(value, dict):
                return {str(key): float(item) for key, item in value.items()}
        except json.JSONDecodeError:
            pass
    capacities: dict[str, float] = {}
    excluded = {"product", "profit", "min_qty", "max_qty"}
    for column in products.columns:
        column_name = str(column)
        if column_name in excluded:
            continue
        match = re.search(rf"{re.escape(column_name)}[^0-9]{{0,8}}([0-9]+(?:\.[0-9]+)?)", question, flags=re.IGNORECASE)
        if match:
            capacities[column_name] = float(match.group(1))
    return capacities


def _local_uploaded_file_answer(files: list[dict]) -> str:
    lines = ["我已读取最近上传的 CSV 文件："]
    for item in files:
        columns = json.loads(item["columns_json"])
        csv_text = item.get("content_csv") or item["preview_csv"]
        row_count = max(len(csv_text.splitlines()) - 1, 0)
        lines.append(f"- {item['filename']}：{row_count} 行，字段包括 {', '.join(columns)}。")
    lines.append("请在问题中说明要优化的目标、约束和容量/预算等参数，我可以继续帮你转成优化模型。")
    return "\n".join(lines)


def _generic_answer_text(generic_result, problem_spec, rag_notes: list[str]) -> str:
    if generic_result.template_id == "knapsack":
        selected = [row for row in generic_result.decisions if row.get("selected") == 1]
        selected_items = ", ".join(_display_item_name(row["item"]) for row in selected) or "无"
        used_weight = generic_result.metrics.get("used_weight")
        capacity = generic_result.metrics.get("capacity")
        selected_value = generic_result.metrics.get("selected_value", generic_result.objective_value)
        return "\n".join(
            [
                "结论：已识别为 0-1 背包问题，并调用通用优化求解工具完成求解。",
                f"最优选择：{selected_items}。",
                f"最大总价值：{selected_value:,.2f}。",
                f"容量使用：{used_weight:,.2f}/{capacity:,.2f}。",
                f"求解器：{generic_result.solver_name}。",
                f"RAG 依据：{'；'.join(rag_notes[:2]) if rag_notes else '已检索本地运筹优化知识库。'}",
            ]
        )
    if generic_result.template_id == "tsp":
        route = generic_result.metrics.get("route", [])
        return "\n".join(
            [
                "结论：已识别为旅行商路径问题，并调用路径优化工具求解。",
                f"推荐路径：{' -> '.join(route)}。",
                f"最短总距离：{generic_result.objective_value:,.2f}。",
                f"RAG 依据：{'；'.join(rag_notes[:2]) if rag_notes else '已检索本地运筹优化知识库。'}",
            ]
        )
    if generic_result.template_id == "job_shop_scheduling":
        return "\n".join(
            [
                "结论：已识别为作业车间调度问题，并调用调度工具生成可行方案。",
                f"最小最大完工时间：{generic_result.objective_value:,.0f}。",
                f"工序数量：{len(generic_result.decisions)}。",
                f"RAG 依据：{'；'.join(rag_notes[:2]) if rag_notes else '已检索本地运筹优化知识库。'}",
            ]
        )
    if generic_result.template_id == "production_mix":
        return "\n".join(
            [
                "结论：已识别为产品组合与生产计划问题，并调用 MILP 求解器求解。",
                f"最大利润：{generic_result.objective_value:,.2f}。",
                generic_result.summary,
                f"RAG 依据：{'；'.join(rag_notes[:2]) if rag_notes else '已检索本地运筹优化知识库。'}",
            ]
        )
    return "\n".join(
        [
            f"结论：已识别为{problem_spec.display_name}，并调用通用优化求解工具完成求解。",
            generic_result.summary,
            f"求解器：{generic_result.solver_name}。",
            f"RAG 依据：{'；'.join(rag_notes[:2]) if rag_notes else '已检索本地运筹优化知识库。'}",
        ]
    )


def _rag_summary_for_generic(question: str, template_id: str):
    notes, docs = rag_summary(question, top_k=5)
    if template_id == "knapsack":
        filtered = [
            (note, doc)
            for note, doc in zip(notes, docs, strict=False)
            if "背包" in doc.title or "背包" in doc.content or "0-1" in doc.title
        ]
        if filtered:
            filtered_notes, filtered_docs = zip(*filtered, strict=False)
            return list(filtered_notes), list(filtered_docs)
    if template_id == "assignment":
        filtered = [
            (note, doc)
            for note, doc in zip(notes, docs, strict=False)
            if "指派" in doc.title or "匹配" in doc.title or "指派" in doc.content
        ]
        if filtered:
            filtered_notes, filtered_docs = zip(*filtered, strict=False)
            return list(filtered_notes), list(filtered_docs)
    if template_id == "tsp":
        return _filter_rag(notes, docs, ["旅行商", "路径", "TSP", "Routing"])
    if template_id == "job_shop_scheduling":
        return _filter_rag(notes, docs, ["调度", "排产", "工序", "CP-SAT"])
    if template_id == "production_mix":
        return _filter_rag(notes, docs, ["产品组合", "生产计划", "MILP", "资源"])
    return notes[:3], docs[:3]


def _filter_rag(notes, docs, keywords: list[str]):
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
    text = str(item)
    return f"物品 {text}" if text.isdigit() else text


def _structured_answer(
    answer: str,
    result,
    baseline,
    changes: list[str],
    rag_notes: list[str],
    open_warehouses: list[str],
    problem_spec,
    generic_result,
) -> dict:
    if generic_result:
        metric_label = generic_result.objective_label or "目标值"
        extra_metrics = _generic_extra_metrics(generic_result)
        return {
            "conclusion": f"识别为{problem_spec.display_name}；已调用工具求解，状态为 {generic_result.status}。",
            "metrics": {
                "objective_label": metric_label,
                "total_cost": generic_result.objective_value,
                "transport_cost": None,
                "fixed_cost": None,
                "cost_delta": None,
                "cost_delta_pct": None,
                "extra": extra_metrics,
            },
            "recommendations": [
                "工具调用链：llm_problem_router -> problem_spec_tool -> rag_context_pack_tool -> data_parser -> generic_optimizer_tool",
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


def _rag_context_preview(pack: dict[str, list[dict]]) -> dict[str, list[dict]]:
    preview: dict[str, list[dict]] = {}
    for category, docs in pack.items():
        preview[category] = [
            {
                "title": doc["title"],
                "score": doc["score"],
                "source": doc["source"],
            }
            for doc in docs
        ]
    return preview


def _rag_context_pack_for_template(question: str, template_id: str) -> dict[str, list[dict]]:
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
    pack: dict[str, list[dict]] = {}
    for category, patterns in specs.items():
        matches = [
            _doc_to_context_dict(doc)
            for doc in docs
            if any(pattern == doc.title for pattern in patterns)
        ]
        pack[labels[category]] = matches[:2]
    return pack


def _template_keywords(template_id: str) -> list[str]:
    return {
        "knapsack": ["背包", "0-1", "容量", "物品"],
        "assignment": ["指派", "匹配", "资源", "任务"],
        "tsp": ["TSP", "旅行商", "路径", "Routing"],
        "job_shop_scheduling": ["调度", "排产", "工序", "机器", "CP-SAT"],
        "production_mix": ["产品组合", "生产计划", "资源", "MILP", "利润"],
    }.get(template_id, [])


def _template_doc_specs(template_id: str) -> dict[str, list[str]]:
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
    return {
        "title": doc.title,
        "score": round(doc.score, 4),
        "source": doc.source,
        "content": doc.content,
    }


def _doc_matches(doc: dict, keywords: list[str]) -> bool:
    text = f"{doc.get('title', '')}\n{doc.get('content', '')}".lower()
    return any(keyword.lower() in text for keyword in keywords)


def _generic_extra_metrics(generic_result) -> list[dict[str, object]]:
    metrics = generic_result.metrics or {}
    if generic_result.template_id == "knapsack":
        return [
            {"label": "容量使用", "value": metrics.get("used_weight"), "suffix": f"/{metrics.get('capacity')}"},
            {"label": "剩余容量", "value": metrics.get("remaining_capacity")},
            {"label": "选择数量", "value": metrics.get("selected_count")},
        ]
    if generic_result.template_id == "tsp":
        return [
            {"label": "节点数", "value": metrics.get("node_count")},
            {"label": "路线段数", "value": max(len(metrics.get("route", [])) - 1, 0)},
        ]
    if generic_result.template_id == "job_shop_scheduling":
        return [
            {"label": "作业数", "value": metrics.get("job_count")},
            {"label": "机器数", "value": metrics.get("machine_count")},
        ]
    if generic_result.template_id == "production_mix":
        usage = metrics.get("resource_usage", {})
        return [
            {"label": f"{resource} 使用", "value": summary.get("used"), "suffix": f"/{summary.get('capacity')}"}
            for resource, summary in usage.items()
        ]
    return []


def _tool_names_for_generic(agent_plan: dict | None = None) -> list[str]:
    tools = agent_plan.get("tool_chain") if agent_plan else None
    if isinstance(tools, list) and tools:
        return ["llm_problem_router", *[str(tool) for tool in tools]]
    return [
        "local_problem_router",
        "problem_spec_tool",
        "rag_context_pack_tool",
        "data_parser",
        "generic_optimizer_tool",
    ]


def _generic_agent_steps(generic_result, agent_plan: dict | None = None, plan_warning: str | None = None) -> list[dict[str, str]]:
    steps = []
    if agent_plan:
        steps.append(
            {
                "step": "LLM 识别与路由",
                "tool": "llm_problem_router",
                "output": f"{agent_plan.get('template_id')} / 置信度 {float(agent_plan.get('confidence', 0)):.0%}",
            }
        )
    else:
        steps.append(
            {
                "step": "本地兜底路由",
                "tool": "local_problem_router",
                "output": plan_warning or "未配置 LLM 或 LLM 路由不可用。",
            }
        )
    steps.extend([
        {"step": "识别问题", "tool": "problem_spec_tool", "output": f"{generic_result.display_name}"},
        {"step": "检索知识", "tool": "rag_context_pack_tool", "output": "读取建模模板、数据 Schema 与求解策略"},
        {"step": "解析数据", "tool": "data_parser", "output": generic_result.data_source},
        {"step": "执行求解", "tool": "generic_optimizer_tool", "output": f"{generic_result.solver_name} / {generic_result.status}"},
        {"step": "最优性校验", "tool": "optimality_checker", "output": _optimality_check_text(generic_result)},
        {"step": "结构化结果", "tool": "result_formatter", "output": generic_result.summary},
    ])
    return steps


def _structured_agent_steps(agent_result, problem_spec, generic_result) -> list[dict[str, str]]:
    if generic_result:
        return _generic_agent_steps(generic_result, None, None)
    return [
        {"step": "识别问题", "tool": "problem_spec_tool", "output": f"{problem_spec.display_name} / {problem_spec.problem_type}"},
        {"step": "检索知识", "tool": "rag_context_pack_tool", "output": "读取建模模板、数据 Schema 与求解策略"},
        {"step": "解析数据", "tool": "data_profile_tool", "output": "读取当前结构化供应链数据集"},
        {"step": "执行求解", "tool": "gurobi_facility_location_tool", "output": "调用仓库选址 MILP 求解器"},
        {"step": "结构化结果", "tool": "result_formatter", "output": "生成成本、启用仓库、风险与依据"},
    ]


def _facility_agent_steps(problem_spec, result, changes: list[str]) -> list[dict[str, str]]:
    return [
        {
            "step": "本地/LLM 路由",
            "tool": "problem_router",
            "output": "识别为仓库选址与客户分配问题，优先使用当前对话上传数据。",
        },
        {
            "step": "识别问题",
            "tool": "problem_spec_tool",
            "output": f"{problem_spec.display_name} / {problem_spec.problem_type}",
        },
        {
            "step": "检索知识",
            "tool": "rag_context_pack_tool",
            "output": "读取仓库选址建模模板、数据 Schema 与求解策略。",
        },
        {
            "step": "解析数据",
            "tool": "data_parser",
            "output": "组装当前对话上传的 warehouses、customers、costs 三类 CSV 数据。",
        },
        {
            "step": "应用业务约束",
            "tool": "scenario_parser",
            "output": "；".join(changes) if changes else "无额外场景约束。",
        },
        {
            "step": "执行求解",
            "tool": "gurobi_facility_location_tool",
            "output": f"{result.solver_name} / {result.status}",
        },
        {
            "step": "最优性校验",
            "tool": "optimality_checker",
            "output": _facility_optimality_check_text(result),
        },
        {
            "step": "结构化结果",
            "tool": "result_formatter",
            "output": "输出启用仓库、客户分配、总成本、固定成本、运输成本和利用率。",
        },
    ]


def _facility_optimality_check_text(result) -> str:
    if result.status == "OPTIMAL":
        return "Gurobi 返回 OPTIMAL，当前固定成本与运输成本之和已证明最小。"
    if result.status in {"TIME_LIMIT", "SUBOPTIMAL"} and result.objective_value is not None:
        return "已得到可行解，但当前状态未证明全局最优。"
    return result.message or f"求解状态：{result.status}。"


def _optimality_check_text(generic_result) -> str:
    if generic_result.status == "OPTIMAL":
        return "求解器返回 OPTIMAL，已按工具能力证明当前目标值最优。"
    if generic_result.template_id == "tsp" and generic_result.metrics.get("optimality_proven") is False:
        return "当前路线由启发式生成，未证明全局最优。"
    if generic_result.status == "FEASIBLE":
        return "已生成可行解，但未证明全局最优。"
    return f"求解状态：{generic_result.status}。"


def _active_llm_config(user_id: int | None) -> LLMConfig | None:
    config = get_active_llm_config(user_id)
    if not config:
        return None
    return LLMConfig(
        enabled=True,
        api_key=config["api_key"],
        base_url=config["base_url"],
        model=config["model"],
        temperature=float(config["temperature"]),
    )
