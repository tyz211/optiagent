from __future__ import annotations

from dataclasses import dataclass
import json

from fastapi import HTTPException

from api.database import (
    get_active_dataset_id_or_none,
    get_active_llm_config,
    list_uploaded_files,
    load_dataset,
    save_run,
    touch_conversation,
)
from api.services.answer_presenter import (
    facility_agent_steps as _facility_agent_steps,
    facility_answer_text as _facility_answer_text,
    facility_profile as _facility_profile,
    generic_agent_steps as _generic_agent_steps,
    generic_answer_text as _generic_answer_text,
    rag_context_pack_for_template as _rag_context_pack_for_template,
    rag_context_preview as _rag_context_preview,
    rag_summary_for_generic as _rag_summary_for_generic,
    structured_agent_steps as _structured_agent_steps,
    structured_answer as _structured_answer,
    tool_names_for_generic as _tool_names_for_generic,
    tool_names_for_web_research as _tool_names_for_web_research,
    web_research_steps as _web_research_steps,
)
from api.services.ask_routing import (
    build_agent_plan as _llm_agent_plan,
    external_data_gaps as _external_data_gaps,
    is_facility_question as _is_facility_question,
    needs_external_data as _needs_external_data,
    should_solve_optimization as _should_solve_optimization,
)
from api.services.uploaded_data_service import (
    facility_data_from_llm_mapping as _facility_data_from_llm_mapping,
    facility_data_from_uploaded_files as _facility_data_from_uploaded_files,
    facility_schema_diagnostics as _facility_schema_diagnostics,
    local_uploaded_file_answer as _local_uploaded_file_answer,
    problem_spec_for_template as _problem_spec_for_template,
    solve_json_generic as _solve_json_generic,
    solve_uploaded_generic as _solve_uploaded_generic,
)
from optiagent.langchain_agents import run_configured_multi_agent
from optiagent.data import SupplyChainData
from optiagent.llm import LLMConfig, llm_config_from_record
from optiagent.optimization_gateway import (
    IN_PROCESS_MCP_STATUS,
    solve_facility_via_gateway,
    solve_question_via_gateway,
)
from optiagent.problem_spec import infer_problem_spec, spec_summary
from optiagent.rag import rag_context_pack, rag_summary
from optiagent.scenario import apply_what_if, explain_result
from optiagent.web_research import web_search


@dataclass(frozen=True)
class AskExecutionContext:
    """Planner 节点准备的请求上下文，供后续状态图节点复用。"""

    dataset_id: int | None
    llm_config: LLMConfig | None
    uploaded_context: list[dict]
    agent_plan: dict | None
    plan_warning: str | None
    preferred_template: str | None
    solver_intent: bool


def prepare_ask_context(
    question: str,
    requested_dataset_id: int | None,
    user_id: int | None,
    conversation_id: int | None,
) -> AskExecutionContext:
    """集中完成上下文读取与问题路由，避免状态图执行阶段重复调用 LLM。"""

    dataset_id = requested_dataset_id or get_active_dataset_id_or_none(
        user_id=user_id,
        conversation_id=conversation_id,
    )
    llm_config = _active_llm_config(user_id)
    uploaded_context = list_uploaded_files(limit=30, user_id=user_id, conversation_id=conversation_id)
    agent_plan, plan_warning = _llm_agent_plan(question, uploaded_context, llm_config)
    preferred_template = agent_plan.get("template_id") if agent_plan else None
    return AskExecutionContext(
        dataset_id=dataset_id,
        llm_config=llm_config,
        uploaded_context=uploaded_context,
        agent_plan=agent_plan,
        plan_warning=plan_warning,
        preferred_template=preferred_template,
        solver_intent=_should_solve_optimization(question),
    )


def handle_ask(
    question: str,
    requested_dataset_id: int | None,
    mcp_config: str,
    user_id: int | None,
    conversation_id: int | None,
    prepared_context: AskExecutionContext | None = None,
) -> dict:
    """执行现有业务回答链路；状态图可传入已经准备好的 Planner 上下文。"""

    context = prepared_context or prepare_ask_context(
        question,
        requested_dataset_id,
        user_id,
        conversation_id,
    )
    dataset_id = context.dataset_id
    llm_config = context.llm_config
    uploaded_context = context.uploaded_context
    agent_plan = context.agent_plan
    plan_warning = context.plan_warning
    preferred_template = context.preferred_template
    solver_intent = context.solver_intent

    uploaded_generic = _solve_uploaded_generic(question, uploaded_context, preferred_template) if solver_intent else None
    if uploaded_generic and requested_dataset_id is None:
        response = _build_uploaded_generic_response(question, uploaded_generic, agent_plan, plan_warning)
        return _persist_response(response, None, question, user_id, conversation_id)

    uploaded_facility = _facility_data_from_uploaded_files(uploaded_context)
    if uploaded_facility and requested_dataset_id is None and _is_facility_question(question, preferred_template) and not solver_intent:
        response = _answer_from_facility_analysis(question, uploaded_facility, plan_warning)
        return _persist_response(response, None, question, user_id, conversation_id)

    if uploaded_facility and requested_dataset_id is None and _is_facility_question(question, preferred_template):
        response = _answer_from_facility_data(question, uploaded_facility, llm_config, mcp_config, plan_warning)
        return _persist_response(response, None, question, user_id, conversation_id)

    if solver_intent and requested_dataset_id is None and uploaded_context and _is_facility_question(question, preferred_template):
        llm_facility, llm_mapping_warning = _facility_data_from_llm_mapping(question, uploaded_context, llm_config)
        if llm_facility:
            warnings = "; ".join(filter(None, [plan_warning, llm_mapping_warning])) or None
            response = _answer_from_facility_data(question, llm_facility, llm_config, mcp_config, warnings)
            return _persist_response(response, None, question, user_id, conversation_id)
        schema_warning = llm_mapping_warning or _facility_schema_diagnostics(uploaded_context)
        if schema_warning:
            response = _answer_schema_confirmation(question, uploaded_context, schema_warning)
            return _persist_response(response, None, question, user_id, conversation_id)

    json_generic = _solve_json_generic(question, preferred_template) if solver_intent else None
    if json_generic and requested_dataset_id is None:
        response = _build_uploaded_generic_response(question, json_generic, agent_plan, plan_warning)
        return _persist_response(response, None, question, user_id, conversation_id)

    if dataset_id is not None and _is_facility_question(question, preferred_template) and not solver_intent:
        response = _answer_from_facility_analysis(question, load_dataset(dataset_id), plan_warning)
        return _persist_response(response, dataset_id, question, user_id, conversation_id)

    if dataset_id is not None and _is_facility_question(question, preferred_template):
        response = _answer_from_structured_dataset(question, dataset_id, llm_config, mcp_config)
        return _persist_response(response, dataset_id, question, user_id, conversation_id)

    if _needs_external_data(question) and requested_dataset_id is None:
        response = _answer_from_web_research(question, uploaded_context, agent_plan, plan_warning)
        return _persist_response(response, None, question, user_id, conversation_id)

    if uploaded_context and requested_dataset_id is None:
        response = _answer_from_uploaded_files(question, uploaded_context, llm_config)
        return _persist_response(response, None, question, user_id, conversation_id)

    if dataset_id is None:
        raise HTTPException(status_code=400, detail="请先上传 CSV 文件。")

    response = _answer_from_structured_dataset(question, dataset_id, llm_config, mcp_config)
    return _persist_response(response, dataset_id, question, user_id, conversation_id)


def _persist_response(
    response: dict,
    dataset_id: int | None,
    question: str,
    user_id: int | None,
    conversation_id: int | None,
) -> dict:
    """统一持久化各回答分支，避免新增路由时漏写运行记录。"""

    response["conversation_id"] = conversation_id
    response["run_id"] = save_run(
        dataset_id,
        question,
        response["answer"],
        response,
        user_id=user_id,
        conversation_id=conversation_id,
    )
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
        "solution_verification": uploaded_generic.solution_verification,
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
        "mcp_status": IN_PROCESS_MCP_STATUS,
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
    generic_result = solve_question_via_gateway(question, problem_spec)

    baseline = solve_facility_via_gateway(data, data_source=f"数据集 {dataset_id} 基线", question=question)
    scenario = apply_what_if(question, data)
    result = (
        solve_facility_via_gateway(
            scenario.data,
            data_source=f"数据集 {dataset_id} 场景",
            question=question,
            warnings=scenario.warnings,
        )
        if generic_result is None
        else baseline
    )
    if llm_config and llm_config.enabled:
        agent_result = run_configured_multi_agent(question, data, llm_config, mcp_config)
        display_answer = generic_result.summary if generic_result else agent_result.answer
        tool_names = agent_result.tool_names
        mcp_status = f"{IN_PROCESS_MCP_STATUS} {agent_result.mcp_status}"
    else:
        # 本地路径已经完成统一求解，不再启动 Agent 造成重复求解。
        display_answer = generic_result.summary if generic_result else _facility_answer_text(result)
        tool_names = (
            _tool_names_for_generic(None)
            if generic_result
            else ["local_problem_router", "problem_spec_tool", "mcp_gateway", "data_validate_problem", "solver_solve_problem"]
        )
        mcp_status = IN_PROCESS_MCP_STATUS

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
        "solution_verification": generic_result.solution_verification if generic_result else result.solution_verification,
        "question": question,
        "status": generic_result.status if generic_result else result.status,
        "objective_value": generic_result.objective_value if generic_result else result.objective_value,
        "transport_cost": None if generic_result else result.transport_cost,
        "fixed_cost": None if generic_result else result.fixed_cost,
        "baseline_objective": baseline.objective_value,
        "open_warehouses": open_warehouses,
        "scenario_changes": scenario.changes,
        "warnings": generic_result.warnings if generic_result else scenario.warnings,
        "explanation": explain_result(baseline, result, scenario.changes),
        "rag_notes": rag_notes,
        "rag_docs": [doc.title for doc in docs],
        "tool_names": tool_names,
        "agent_steps": _structured_agent_steps(problem_spec, generic_result),
        "mcp_status": mcp_status,
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
    baseline = solve_facility_via_gateway(data, data_source="当前对话基线数据", question=question)
    scenario = apply_what_if(question, data)
    result = solve_facility_via_gateway(
        scenario.data,
        data_source="当前对话场景数据",
        question=question,
        warnings=scenario.warnings,
    )
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
        "solution_verification": result.solution_verification,
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
        "tool_names": [
            "local_problem_router",
            "problem_spec_tool",
            "rag_context_pack_tool",
            "mcp_gateway",
            "data_validate_problem",
            "solver_solve_problem",
        ],
        "agent_steps": _facility_agent_steps(problem_spec, result, scenario.changes),
        "mcp_status": IN_PROCESS_MCP_STATUS,
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


def _active_llm_config(user_id: int | None) -> LLMConfig | None:
    return llm_config_from_record(get_active_llm_config(user_id))
