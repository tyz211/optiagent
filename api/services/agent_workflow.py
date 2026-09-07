from __future__ import annotations

from collections.abc import Callable
from operator import add
from time import perf_counter
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from api.database import update_run_result
from api.services.ask_service import AskExecutionContext, handle_ask, prepare_ask_context
from optiagent.problem_spec import infer_problem_spec
from optiagent.templates.registry import get_template


AgentEventCallback = Callable[[dict[str, Any]], None]


class AgentWorkflowState(TypedDict, total=False):
    """一次运筹 Agent 运行期间在各节点之间传递的共享状态。"""

    question: str
    requested_dataset_id: int | None
    mcp_config: str
    user_id: int | None
    conversation_id: int | None
    execution_context: AskExecutionContext
    data_context: dict[str, Any]
    problem_spec: dict[str, Any] | None
    result: dict[str, Any]
    verification: dict[str, Any]
    trace_nodes: Annotated[list[dict[str, Any]], add]


NODE_DEFINITIONS = [
    ("planner", "Planner", "识别意图与规划工具链"),
    ("data", "Data Agent", "定位并检查当前数据上下文"),
    ("modeler", "Modeler", "生成结构化问题定义"),
    ("solver", "Solver", "通过 MCP Gateway 执行求解"),
    ("verifier", "Verifier", "检查响应合同与求解状态"),
    ("explainer", "Explainer", "组织业务结论与可审计轨迹"),
]


def run_agent_workflow(
    *,
    question: str,
    requested_dataset_id: int | None,
    mcp_config: str,
    user_id: int | None,
    conversation_id: int | None,
    event_callback: AgentEventCallback | None = None,
) -> dict[str, Any]:
    """运行 LangGraph 状态图，并返回兼容现有前端的最终结果。"""

    initial_state: AgentWorkflowState = {
        "question": question,
        "requested_dataset_id": requested_dataset_id,
        "mcp_config": mcp_config,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "trace_nodes": [],
    }
    final_state = AGENT_GRAPH.invoke(
        initial_state,
        config={"configurable": {"event_callback": event_callback}},
    )
    result = dict(final_state["result"])
    result["workflow_verification"] = final_state.get("verification", {})
    result["agent_graph"] = {
        "version": "1.0",
        "engine": "LangGraph",
        "status": "completed",
        "nodes": final_state.get("trace_nodes", []),
    }
    run_id = result.get("run_id")
    if isinstance(run_id, int):
        update_run_result(run_id, result)
    return result


def agent_graph_manifest() -> dict[str, Any]:
    """返回前端可以预先绘制的节点清单。"""

    return {
        "version": "1.0",
        "engine": "LangGraph",
        "nodes": [
            {"id": node_id, "label": label, "description": description, "sequence": index}
            for index, (node_id, label, description) in enumerate(NODE_DEFINITIONS, start=1)
        ],
    }


def _planner_node(state: AgentWorkflowState) -> dict[str, Any]:
    context = prepare_ask_context(
        state["question"],
        state.get("requested_dataset_id"),
        state.get("user_id"),
        state.get("conversation_id"),
    )
    route_source = "LLM" if context.agent_plan else "本地规则"
    # 本地 Planner 也提前暴露模板推断结果，让执行轨迹可以解释后续路由。
    inferred_spec = infer_problem_spec(state["question"], None) if not context.preferred_template else None
    template_id = context.preferred_template or (inferred_spec.template_id if inferred_spec else "unknown")
    intent = "执行优化" if context.solver_intent else "分析数据"
    return {
        "execution_context": context,
        "_trace_detail": f"{route_source} · {template_id} · {intent}",
    }


def _data_node(state: AgentWorkflowState) -> dict[str, Any]:
    context = state["execution_context"]
    uploaded = context.uploaded_context
    if context.dataset_id is not None:
        source_type = "structured_dataset"
        summary = f"使用结构化数据集 #{context.dataset_id}"
    elif uploaded:
        roles = sorted({str(item.get("role") or "未分类") for item in uploaded})
        source_type = "uploaded_files"
        summary = f"发现 {len(uploaded)} 个上传文件 · {' / '.join(roles)}"
    elif "{" in state["question"] and "}" in state["question"]:
        source_type = "inline_json"
        summary = "检测到问题中的内联 JSON 数据"
    else:
        source_type = "conversation"
        summary = "当前请求未绑定结构化文件"
    return {
        "data_context": {
            "source_type": source_type,
            "dataset_id": context.dataset_id,
            "uploaded_file_count": len(uploaded),
        },
        "_trace_detail": summary,
    }


def _modeler_node(state: AgentWorkflowState) -> dict[str, Any]:
    context = state["execution_context"]
    template_id = context.preferred_template
    if template_id == "file_answer" and not context.solver_intent:
        return {"problem_spec": None, "_trace_detail": "分析型请求，无需生成求解模型"}
    try:
        spec = get_template(template_id).build_spec(state["question"], None) if template_id else infer_problem_spec(state["question"], None)
    except KeyError:
        spec = infer_problem_spec(state["question"], None)
    return {
        "problem_spec": spec.to_dict(),
        "_trace_detail": f"{spec.display_name} · 置信度 {spec.confidence:.0%}",
    }


def _solver_node(state: AgentWorkflowState) -> dict[str, Any]:
    result = handle_ask(
        question=state["question"],
        requested_dataset_id=state.get("requested_dataset_id"),
        mcp_config=state.get("mcp_config", ""),
        user_id=state.get("user_id"),
        conversation_id=state.get("conversation_id"),
        prepared_context=state["execution_context"],
    )
    status = str(result.get("status") or "COMPLETED")
    objective = result.get("objective_value")
    detail = f"{status} · 目标值 {objective:,.2f}" if isinstance(objective, (int, float)) else status
    return {"result": result, "_trace_detail": detail}


def _verifier_node(state: AgentWorkflowState) -> dict[str, Any]:
    result = state["result"]
    errors: list[str] = []
    checks = {
        "has_answer": bool(str(result.get("answer") or "").strip()),
        "has_status": bool(str(result.get("status") or "").strip()),
        "has_structured_answer": isinstance(result.get("structured_answer"), dict),
        "objective_present_when_required": True,
    }
    if result.get("status") in {"OPTIMAL", "NEAR_OPTIMAL", "FEASIBLE"}:
        checks["objective_present_when_required"] = result.get("objective_value") is not None
    for check_name, passed in checks.items():
        if not passed:
            errors.append(f"响应合同检查失败：{check_name}")
    verification = {
        "scope": "response_contract",
        "passed": not errors,
        "checks": checks,
        "errors": errors,
        "note": "第一阶段验证响应合同；数学可行性验算将在后续 SolutionVerifier 中实现。",
    }
    detail = f"响应合同 {sum(checks.values())}/{len(checks)} 项通过"
    return {"verification": verification, "_trace_detail": detail}


def _explainer_node(state: AgentWorkflowState) -> dict[str, Any]:
    result = dict(state["result"])
    result["workflow_verification"] = state.get("verification", {})
    return {"result": result, "_trace_detail": "结论、模型与执行轨迹已整理"}


def _traced_node(
    node_id: str,
    label: str,
    description: str,
    worker: Callable[[AgentWorkflowState], dict[str, Any]],
):
    """为每个节点统一生成运行事件和可持久化摘要。"""

    sequence = next(index for index, item in enumerate(NODE_DEFINITIONS, start=1) if item[0] == node_id)

    def run(state: AgentWorkflowState, config: RunnableConfig) -> dict[str, Any]:
        started = perf_counter()
        _emit_event(
            config,
            {
                "node_id": node_id,
                "label": label,
                "description": description,
                "sequence": sequence,
                "status": "running",
                "detail": description,
            },
        )
        try:
            update = worker(state)
        except Exception as exc:
            _emit_event(
                config,
                {
                    "node_id": node_id,
                    "label": label,
                    "description": description,
                    "sequence": sequence,
                    "status": "failed",
                    "detail": f"{type(exc).__name__}: {exc}",
                },
            )
            raise
        detail = str(update.pop("_trace_detail", description))
        trace = {
            "node_id": node_id,
            "label": label,
            "description": description,
            "sequence": sequence,
            "status": "completed",
            "detail": detail,
            "elapsed_ms": round((perf_counter() - started) * 1000, 2),
        }
        _emit_event(config, trace)
        update["trace_nodes"] = [trace]
        return update

    return run


def _emit_event(config: RunnableConfig, event: dict[str, Any]) -> None:
    callback = config.get("configurable", {}).get("event_callback")
    if callable(callback):
        callback(event)


def _build_agent_graph():
    """构建固定职责节点，后续可在同一状态合同上加入反馈回路。"""

    workers = {
        "planner": _planner_node,
        "data": _data_node,
        "modeler": _modeler_node,
        "solver": _solver_node,
        "verifier": _verifier_node,
        "explainer": _explainer_node,
    }
    graph = StateGraph(AgentWorkflowState)
    for node_id, label, description in NODE_DEFINITIONS:
        graph.add_node(node_id, _traced_node(node_id, label, description, workers[node_id]))
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "data")
    graph.add_edge("data", "modeler")
    graph.add_edge("modeler", "solver")
    graph.add_edge("solver", "verifier")
    graph.add_edge("verifier", "explainer")
    graph.add_edge("explainer", END)
    return graph.compile()


AGENT_GRAPH = _build_agent_graph()
