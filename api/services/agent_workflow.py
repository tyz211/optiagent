from __future__ import annotations

from collections.abc import Callable
from operator import add
from time import perf_counter
from typing import Annotated, Any, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, START, StateGraph

from api.database import (
    complete_agent_episode,
    create_agent_episode,
    finish_agent_step,
    set_agent_episode_template,
    start_agent_step,
    update_run_result,
)
from api.services.ask_service import AskExecutionContext, handle_ask, prepare_ask_context
from optiagent.agent_policy import decide_after_verification
from optiagent.mcp_servers.common import json_safe
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
    episode_id: str
    execution_context: AskExecutionContext
    data_context: dict[str, Any]
    problem_spec: dict[str, Any] | None
    result: dict[str, Any]
    verification: dict[str, Any]
    route_action: str
    recovery_feedback: list[str]
    model_attempt: int
    solver_attempt: int
    policy_decisions: Annotated[list[dict[str, Any]], add]
    trace_nodes: Annotated[list[dict[str, Any]], add]


NODE_DEFINITIONS = [
    ("planner", "Planner", "识别意图与规划工具链"),
    ("data", "Data Agent", "定位并检查当前数据上下文"),
    ("modeler", "Modeler", "生成结构化问题定义"),
    ("solver", "Solver", "通过 MCP Gateway 执行求解"),
    ("verifier", "Verifier", "检查响应合同与求解状态"),
    ("policy", "Policy", "根据验证反馈选择接受、恢复或终止"),
    ("explainer", "Explainer", "组织业务结论与可审计轨迹"),
]

NODE_ACTIONS = {
    "planner": ("plan", "选择问题模板与执行意图"),
    "data": ("inspect_data", "检查数据来源与可用上下文"),
    "modeler": ("build_problem_spec", "构建结构化优化问题"),
    "solver": ("call_solver", "调用 MCP Optimization Gateway"),
    "verifier": ("verify_solution", "复算约束、目标值与响应合同"),
    "policy": ("route", "在候选恢复动作中选择下一步"),
    "explainer": ("return_result", "组织并返回可审计结果"),
}


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

    episode_id = create_agent_episode(
        question,
        user_id,
        conversation_id,
        policy_name="langgraph_baseline",
        policy_version="1.0",
    )
    initial_state: AgentWorkflowState = {
        "question": question,
        "requested_dataset_id": requested_dataset_id,
        "mcp_config": mcp_config,
        "user_id": user_id,
        "conversation_id": conversation_id,
        "episode_id": episode_id,
        "model_attempt": 0,
        "solver_attempt": 0,
        "policy_decisions": [],
        "trace_nodes": [],
    }
    try:
        final_state = AGENT_GRAPH.invoke(
            initial_state,
            config={"configurable": {"event_callback": event_callback}},
        )
        result = dict(final_state["result"])
        result["workflow_verification"] = final_state.get("verification", {})
        result["agent_episode_id"] = episode_id
        result["agent_graph"] = {
            "version": "1.0",
            "engine": "LangGraph",
            "status": "completed",
            "nodes": final_state.get("trace_nodes", []),
        }
        result["agent_policy"] = {
            "name": "deterministic_recovery_baseline",
            "version": "1.0",
            "decisions": final_state.get("policy_decisions", []),
        }
        run_id = result.get("run_id")
        if isinstance(run_id, int):
            update_run_result(run_id, result)
        template_id = (final_state.get("problem_spec") or {}).get("template_id")
        reward = (final_state.get("verification") or {}).get("reward", {})
        complete_agent_episode(
            episode_id,
            status="completed",
            run_id=run_id if isinstance(run_id, int) else None,
            template_id=template_id,
            reward=reward,
            result=result,
        )
        return result
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
        complete_agent_episode(
            episode_id,
            status="failed",
            reward=_failure_reward(error),
            error=error,
        )
        raise


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
    problem_spec = spec.to_dict()
    feedback = state.get("recovery_feedback") or []
    if feedback:
        # 确定性模板暂不生成新代码，但会显式携带 Verifier 反馈供后续学习策略使用。
        problem_spec.setdefault("notes", []).append("恢复反馈：" + "；".join(feedback[:5]))
    detail = f"{spec.display_name} · 置信度 {spec.confidence:.0%}"
    if feedback:
        detail += f" · 第 {int(state.get('model_attempt', 0)) + 1} 次建模"
    return {"problem_spec": problem_spec, "_trace_detail": detail}


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
        "terminal_status_valid": True,
    }
    successful_statuses = {"OPTIMAL", "NEAR_OPTIMAL", "FEASIBLE"}
    if result.get("status") in successful_statuses:
        checks["objective_present_when_required"] = result.get("objective_value") is not None
    context = state.get("execution_context")
    if context and context.solver_intent:
        checks["terminal_status_valid"] = result.get("status") in successful_statuses
    for check_name, passed in checks.items():
        if not passed:
            errors.append(f"响应合同检查失败：{check_name}")
    contract_passed = not errors
    mathematical = result.get("solution_verification")
    requires_mathematical_check = result.get("status") in successful_statuses
    mathematical_passed = bool(mathematical and mathematical.get("verifiable") and mathematical.get("passed"))
    if requires_mathematical_check and not mathematical_passed:
        errors.append("数学 SolutionVerifier 未通过。")
    reward = _deterministic_reward(result, checks, mathematical)
    verification = {
        "scope": "response_contract_and_solution",
        "passed": contract_passed and (not requires_mathematical_check or mathematical_passed),
        "checks": checks,
        "errors": errors,
        "mathematical": mathematical,
        "reward": reward,
        "note": "数学验证器独立复算约束与目标值；全局最优性仍以求解器证明信息为准。",
    }
    if mathematical and mathematical.get("verifiable"):
        detail = "数学验算通过" if mathematical_passed else f"数学验算发现 {len(mathematical.get('violations', []))} 项异常"
    else:
        detail = f"响应合同 {sum(checks.values())}/{len(checks)} 项通过"
    return {"verification": verification, "_trace_detail": detail}


def _policy_node(state: AgentWorkflowState) -> dict[str, Any]:
    """运行高层恢复策略，并把候选动作、mask 和选择原因写入状态。"""

    decision = decide_after_verification(
        state.get("verification") or {},
        model_attempt=int(state.get("model_attempt", 0)),
        solver_attempt=int(state.get("solver_attempt", 0)),
    ).model_dump(mode="json")
    selected = decision["selected_action"]
    feedback = _verification_feedback(state.get("verification") or {}) if selected == "rebuild_model" else []
    return {
        "route_action": selected,
        "recovery_feedback": feedback,
        "policy_decisions": [decision],
        "_trace_detail": f"{selected} · {decision['reason']}",
    }


def _verification_feedback(verification: dict[str, Any]) -> list[str]:
    """把验证异常压缩成 Modeler 可消费的恢复上下文。"""

    feedback = [str(item) for item in verification.get("errors", [])]
    mathematical = verification.get("mathematical") or {}
    feedback.extend(str(item) for item in mathematical.get("violations", []))
    return feedback or ["Verifier 未通过，请重新检查模型和数据映射。"]


def _route_after_policy(state: AgentWorkflowState) -> str:
    """为 LangGraph 条件边返回经过 Schema 约束的高层动作。"""

    action = state.get("route_action", "terminate")
    if action not in {"accept_solution", "retry_solver", "rebuild_model", "terminate"}:
        return "terminate"
    return action


def _deterministic_reward(
    result: dict[str, Any],
    contract_checks: dict[str, bool],
    mathematical: dict[str, Any] | None,
) -> dict[str, Any]:
    """把确定性验证信号转换为第一版可审计终局奖励。"""

    contract_ratio = sum(bool(value) for value in contract_checks.values()) / max(len(contract_checks), 1)
    status = str(result.get("status") or "")
    feasible = mathematical.get("feasible") if mathematical else None
    objective_consistent = mathematical.get("objective_consistent") if mathematical else None
    terminal_success = (
        status in {"OPTIMAL", "NEAR_OPTIMAL", "FEASIBLE"}
        and feasible is True
        and objective_consistent is True
    )
    components = {
        "response_contract": round(0.1 * contract_ratio, 4),
        "terminal_success": 0.1 if terminal_success else 0.0,
        "solution_feasibility": 0.45 if feasible is True else (-0.55 if feasible is False else 0.0),
        "objective_consistency": 0.35 if objective_consistent is True else (-0.35 if objective_consistent is False else 0.0),
        "unsolved_penalty": -0.5 if mathematical is not None and status not in {"OPTIMAL", "NEAR_OPTIMAL", "FEASIBLE"} else 0.0,
    }
    total = max(-1.0, min(1.0, sum(components.values())))
    return {
        "version": "1.0",
        "total": round(total, 4),
        "components": components,
        "deterministic": True,
        "terminal": True,
    }


def _failure_reward(error: str) -> dict[str, Any]:
    """为执行异常生成稳定的负样本终局奖励。"""

    return {
        "version": "1.0",
        "total": -1.0,
        "components": {"execution_failure": -1.0},
        "deterministic": True,
        "terminal": True,
        "error": error,
    }


def _explainer_node(state: AgentWorkflowState) -> dict[str, Any]:
    result = dict(state["result"])
    result["workflow_verification"] = state.get("verification", {})
    result["policy_decisions"] = state.get("policy_decisions", [])
    return {"result": result, "_trace_detail": "结论、模型与执行轨迹已整理"}


def _traced_node(
    node_id: str,
    label: str,
    description: str,
    worker: Callable[[AgentWorkflowState], dict[str, Any]],
):
    """为每个节点统一生成运行事件和可持久化摘要。"""

    graph_sequence = next(index for index, item in enumerate(NODE_DEFINITIONS, start=1) if item[0] == node_id)

    def run(state: AgentWorkflowState, config: RunnableConfig) -> dict[str, Any]:
        started = perf_counter()
        episode_id = state["episode_id"]
        previous_traces = state.get("trace_nodes", [])
        attempt = 1 + sum(item.get("node_id") == node_id for item in previous_traces)
        transition_sequence = len(previous_traces) + 1
        start_agent_step(
            episode_id,
            transition_sequence,
            node_id,
            _trajectory_state(state, node_id),
            _trajectory_action(state, node_id),
            attempt=attempt,
        )
        _emit_event(
            config,
            {
                "episode_id": episode_id,
                "node_id": node_id,
                "label": label,
                "description": description,
                "sequence": graph_sequence,
                "transition_sequence": transition_sequence,
                "attempt": attempt,
                "status": "running",
                "detail": description,
            },
        )
        try:
            update = worker(state)
        except Exception as exc:
            elapsed_ms = round((perf_counter() - started) * 1000, 2)
            error = f"{type(exc).__name__}: {exc}"
            failure_reward = _failure_reward(error)
            finish_agent_step(
                episode_id,
                node_id,
                {"error_type": type(exc).__name__, "message": str(exc)},
                status="failed",
                elapsed_ms=elapsed_ms,
                reward=failure_reward,
                error=error,
                attempt=attempt,
            )
            _emit_event(
                config,
                {
                    "episode_id": episode_id,
                    "node_id": node_id,
                    "label": label,
                    "description": description,
                    "sequence": graph_sequence,
                    "transition_sequence": transition_sequence,
                    "attempt": attempt,
                    "status": "failed",
                    "detail": error,
                    "elapsed_ms": elapsed_ms,
                },
            )
            raise
        detail = str(update.pop("_trace_detail", description))
        elapsed_ms = round((perf_counter() - started) * 1000, 2)
        # Modeler 产生标准问题定义后立即写入标签，后续节点失败也可用于训练分组。
        if node_id == "modeler":
            problem_spec = update.get("problem_spec") or {}
            template_id = problem_spec.get("template_id") if isinstance(problem_spec, dict) else None
            if template_id:
                set_agent_episode_template(episode_id, str(template_id))
            update["model_attempt"] = attempt
        elif node_id == "solver":
            update["solver_attempt"] = attempt
        reward = (update.get("verification") or {}).get("reward") if isinstance(update.get("verification"), dict) else None
        finish_agent_step(
            episode_id,
            node_id,
            _trajectory_observation(update, detail),
            status="completed",
            elapsed_ms=elapsed_ms,
            reward=reward,
            attempt=attempt,
        )
        trace = {
            "episode_id": episode_id,
            "node_id": node_id,
            "label": label,
            "description": description,
            "sequence": graph_sequence,
            "transition_sequence": transition_sequence,
            "attempt": attempt,
            "status": "completed",
            "detail": detail,
            "elapsed_ms": elapsed_ms,
        }
        _emit_event(config, trace)
        update["trace_nodes"] = [trace]
        return update

    return run


def _emit_event(config: RunnableConfig, event: dict[str, Any]) -> None:
    callback = config.get("configurable", {}).get("event_callback")
    if callable(callback):
        callback(event)


def _trajectory_state(state: AgentWorkflowState, node_id: str) -> dict[str, Any]:
    """生成稳定且不包含 LLM 密钥的 policy state 快照。"""

    context = state.get("execution_context")
    context_snapshot = None
    if context:
        context_snapshot = {
            "dataset_id": context.dataset_id,
            "uploaded_file_count": len(context.uploaded_context),
            "preferred_template": context.preferred_template,
            "solver_intent": context.solver_intent,
            "route_source": "llm" if context.agent_plan else "local_rule",
        }
    result = state.get("result") or {}
    return json_safe(
        {
            "schema_version": "1.0",
            "episode_id": state["episode_id"],
            "current_node": node_id,
            "question": state["question"],
            "requested_dataset_id": state.get("requested_dataset_id"),
            "execution_context": context_snapshot,
            "data_context": state.get("data_context"),
            "problem_spec": state.get("problem_spec"),
            "result_summary": {
                "status": result.get("status"),
                "objective_value": result.get("objective_value"),
                "has_solution_verification": bool(result.get("solution_verification")),
            },
            "verification": state.get("verification"),
            "recovery_feedback": state.get("recovery_feedback", []),
            "model_attempt": state.get("model_attempt", 0),
            "solver_attempt": state.get("solver_attempt", 0),
            "policy_decisions": state.get("policy_decisions", []),
            "completed_nodes": [item.get("node_id") for item in state.get("trace_nodes", [])],
        }
    )


def _trajectory_action(state: AgentWorkflowState, node_id: str) -> dict[str, Any]:
    """记录基线 policy 选择的动作及其最小参数集合。"""

    if node_id == "policy":
        # Policy 节点的决策函数是纯函数，因此可在执行前完整记录实际动作。
        return decide_after_verification(
            state.get("verification") or {},
            model_attempt=int(state.get("model_attempt", 0)),
            solver_attempt=int(state.get("solver_attempt", 0)),
        ).model_dump(mode="json")
    action_type, description = NODE_ACTIONS[node_id]
    problem_spec = state.get("problem_spec") or {}
    context = state.get("execution_context")
    return {
        "schema_version": "1.0",
        "type": action_type,
        "description": description,
        "arguments": {
            "template_id": problem_spec.get("template_id"),
            "dataset_id": context.dataset_id if context else state.get("requested_dataset_id"),
        },
    }


def _trajectory_observation(update: dict[str, Any], detail: str) -> dict[str, Any]:
    """把节点输出转换为可回放 observation，并单独处理 Planner 上下文。"""

    observation: dict[str, Any] = {"schema_version": "1.0", "detail": detail}
    for key, value in update.items():
        if key == "execution_context" and isinstance(value, AskExecutionContext):
            observation[key] = {
                "dataset_id": value.dataset_id,
                "uploaded_file_count": len(value.uploaded_context),
                "preferred_template": value.preferred_template,
                "solver_intent": value.solver_intent,
                "agent_plan": value.agent_plan,
                "plan_warning": value.plan_warning,
            }
        elif key != "trace_nodes":
            observation[key] = json_safe(value)
    return observation


def _build_agent_graph(worker_overrides: dict[str, Callable[[AgentWorkflowState], dict[str, Any]]] | None = None):
    """构建固定职责节点，后续可在同一状态合同上加入反馈回路。"""

    workers = {
        "planner": _planner_node,
        "data": _data_node,
        "modeler": _modeler_node,
        "solver": _solver_node,
        "verifier": _verifier_node,
        "policy": _policy_node,
        "explainer": _explainer_node,
    }
    workers.update(worker_overrides or {})
    graph = StateGraph(AgentWorkflowState)
    for node_id, label, description in NODE_DEFINITIONS:
        graph.add_node(node_id, _traced_node(node_id, label, description, workers[node_id]))
    graph.add_edge(START, "planner")
    graph.add_edge("planner", "data")
    graph.add_edge("data", "modeler")
    graph.add_edge("modeler", "solver")
    graph.add_edge("solver", "verifier")
    graph.add_edge("verifier", "policy")
    graph.add_conditional_edges(
        "policy",
        _route_after_policy,
        {
            "accept_solution": "explainer",
            "retry_solver": "solver",
            "rebuild_model": "modeler",
            "terminate": "explainer",
        },
    )
    graph.add_edge("explainer", END)
    return graph.compile()


AGENT_GRAPH = _build_agent_graph()
