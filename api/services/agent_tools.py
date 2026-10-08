"""本地工具执行上下文与处理函数；生命周期、预算和审计由主控负责。"""

from __future__ import annotations

from dataclasses import dataclass
from typing import TYPE_CHECKING

from api.database import (
    compare_and_save_requirement, get_active_dataset_id_or_none, get_conversation_requirement,
    list_datasets, list_uploaded_files, set_agent_episode_template, update_run_result,
)
from api.services.agent_memory import search_memories
from api.services.ask_service import AskExecutionContext
from api.services.requirement_service import build_clarification_result
from api.services.result_evidence import explanation_request, read_result_evidence, render_result_explanation
from optiagent.agent_context import requirement_view
from optiagent.rag import retrieve
from optiagent.requirement_analysis import RequirementBrief
from optiagent.requirement_patch import build_patched_requirement, editable_targets

if TYPE_CHECKING:
    from api.services.agent_decision import ControllerDecision


@dataclass
class AgentToolContext:
    """本轮作用域与共享状态，完整数据始终留在本地执行层。"""

    state: dict
    question: str
    requested_dataset_id: int | None
    user_id: int | None
    conversation_id: int | None
    episode_id: int
    memories: list[dict]

    def prepare_context(self) -> None:
        """仅读取受作用域保护的数据，不启动第二个 LLM Agent 或重复路由。"""
        state = self.state
        requested_dataset_id = self.requested_dataset_id
        user_id = self.user_id
        conversation_id = self.conversation_id
        contract = (state.get("requirement_analysis") or {}).get("dialogue_contract") or {}
        dataset_id = requested_dataset_id or get_active_dataset_id_or_none(user_id=user_id, conversation_id=conversation_id)
        if contract.get("data"):
            dataset_id = None
            state["requested_dataset_id"] = None
        elif dataset_id is not None:
            visible_ids = {row["id"] for row in list_datasets(user_id=user_id, conversation_id=conversation_id)}
            if dataset_id not in visible_ids:
                raise PermissionError("数据集不属于当前用户或会话")
        state["execution_context"] = AskExecutionContext(
            dataset_id=dataset_id, llm_config=None,
            uploaded_context=list_uploaded_files(limit=30, user_id=user_id, conversation_id=conversation_id),
            agent_plan=None, plan_warning=None,
            preferred_template=(state.get("requirement_analysis") or {}).get("template_id"),
            solver_intent=(state.get("requirement_analysis") or {}).get("readiness") == "ready_to_solve")

    def check_result_reference(self) -> dict:
        """生成解释和最终返回前均复核源证据，失效时不接受已生成的解释记录。"""
        state = self.state
        question = self.question
        user_id = self.user_id
        conversation_id = self.conversation_id
        evidence = state["result_evidence"]
        _, explicit_run = explanation_request(question)
        current = get_conversation_requirement(conversation_id, user_id=user_id) or {}
        refreshed = read_result_evidence(user_id=user_id, conversation_id=conversation_id,
            brief=current, run_id=evidence["run_id"])
        if (refreshed["fingerprint"] != evidence["fingerprint"]
                or refreshed["matches_current_requirement"] != evidence["matches_current_requirement"]
                or (explicit_run is None and not refreshed["matches_current_requirement"])):
            raise ValueError("结果证据或当前版本已变化，请重新读取")
        state["requirement_analysis"] = current
        return {"summary": "源方案证据复核通过", "result_ref": evidence["run_id"]}

    def run_workflow_node(self, name: str) -> dict:
        """延迟解析既有节点，保留测试替换能力并避免主入口循环导入。"""
        from api.services import agent_workflow
        return getattr(agent_workflow, name)(self.state)


def read_requirement(context: AgentToolContext, decision: ControllerDecision) -> dict:
    """执行 read_requirement，沿用现有作用域和数学校验约束。"""
    state = context.state
    user_id = context.user_id
    conversation_id = context.conversation_id
    if not state.get("requirements_done") and conversation_id is not None:
        # 冲突反馈后的重新读取必须获取数据库最新状态，不能重复返回旧快照。
        current = get_conversation_requirement(conversation_id, user_id=user_id)
        if current is not None:
            state["requirement_analysis"] = current
    return {"summary": "已读取当前可编辑字段", "requirement_fields": editable_targets(state["requirement_analysis"], decision.query)}


def apply_requirement_patch(context: AgentToolContext, decision: ControllerDecision) -> dict:
    """执行 apply_requirement_patch，沿用现有作用域和数学校验约束。"""
    state = context.state
    question = context.question
    user_id = context.user_id
    conversation_id = context.conversation_id
    if decision.patch is None or conversation_id is None:
        raise ValueError("修改必须提供结构化参数并处于有效会话")
    previous = state["requirement_analysis"]
    candidate = build_patched_requirement(previous, question, decision.patch)
    compare_and_save_requirement(conversation_id, user_id=user_id, expected=previous, replacement=candidate)
    state.update(requirement_analysis=candidate, requirements_done=True,
                 question=candidate["resolved_request"], requirement_route="proceed")
    context.prepare_context()
    return {"summary": "修改已核对并原子保存", "requirements": requirement_view(candidate),
            "changes": candidate["dialogue_contract"]["changes"]}


def read_verified_result(context: AgentToolContext, decision: ControllerDecision) -> dict:
    """执行 read_verified_result，沿用现有作用域和数学校验约束。"""
    state = context.state
    question = context.question
    user_id = context.user_id
    conversation_id = context.conversation_id
    eligible, explicit_run = explanation_request(question)
    if not eligible or decision.run_id != explicit_run:
        raise ValueError("结果引用必须与用户本轮明确指令一致")
    current = get_conversation_requirement(conversation_id, user_id=user_id) or state["requirement_analysis"]
    # 解释不会改变需求版本，也不会复用旧解作为本轮求解候选。
    evidence = read_result_evidence(user_id=user_id, conversation_id=conversation_id,
        brief=current, run_id=explicit_run, query=decision.query)
    state.update(result_evidence=evidence, requirement_analysis=current, requirements_done=True, analysis_only=True)
    return {"summary": "已读取有验算证据的方案", "result_evidence": evidence}


def explain_result(context: AgentToolContext, decision: ControllerDecision) -> dict:
    """执行 explain_result，沿用现有作用域和数学校验约束。"""
    state = context.state
    question = context.question
    requested_dataset_id = context.requested_dataset_id
    user_id = context.user_id
    conversation_id = context.conversation_id
    evidence = state["result_evidence"]
    try:
        context.check_result_reference()
    except ValueError:
        state.pop("result_evidence", None)
        raise
    answer, citations = render_result_explanation(evidence, decision.fact_ids or [], question=question)
    result = build_clarification_result(question=question, brief=RequirementBrief.model_validate(state["requirement_analysis"]),
        requested_dataset_id=None, user_id=user_id, conversation_id=conversation_id,
        response_status="ANALYSIS", message=answer)
    result["result_reference"] = {name: evidence[name] for name in ("run_id", "requirement_revision", "matches_current_requirement")}
    result["fact_citations"] = citations
    result["structured_answer"]["evidence"] = [f"方案 #{item['run_id']} · {item['source_path']}" for item in citations]
    state.update(result=result, verification={"passed": True, "scope": "verified_result_reference", "mathematical": None})
    return {"summary": "解释已绑定源方案事实", "result_reference": result["result_reference"], "fact_citations": citations}


def analyze_requirements(context: AgentToolContext, decision: ControllerDecision) -> dict:
    """执行 analyze_requirements，沿用现有作用域和数学校验约束。"""
    state = context.state
    update = context.run_workflow_node("_requirements_node")
    state.update(update)
    state["requirements_done"] = True
    context.prepare_context()
    return {"summary": update.get("_trace_detail"), "requirements": requirement_view(state["requirement_analysis"])}


def inspect_data(context: AgentToolContext, decision: ControllerDecision) -> dict:
    """执行 inspect_data，沿用现有作用域和数学校验约束。"""
    state = context.state
    context.prepare_context()
    update = context.run_workflow_node("_data_node")
    state.update(update)
    return {"summary": update.get("_trace_detail"), "data": update["data_context"]}


def build_model(context: AgentToolContext, decision: ControllerDecision) -> dict:
    """执行 build_model，沿用现有作用域和数学校验约束。"""
    state = context.state
    episode_id = context.episode_id
    context.prepare_context()
    state["verification"] = {}
    state.pop("result", None)
    update = context.run_workflow_node("_modeler_node")
    state.update(update)
    state["model_attempt"] += 1
    spec = state.get("problem_spec") or {}
    if spec.get("template_id"):
        set_agent_episode_template(episode_id, spec["template_id"])
    return {"summary": update.get("_trace_detail"), "model": {
        name: spec.get(name) for name in ("template_id", "objective", "constraints", "decision_variables", "recommended_solver")}}


def solve(context: AgentToolContext, decision: ControllerDecision) -> dict:
    """执行 solve，沿用现有作用域和数学校验约束。"""
    state = context.state
    update = context.run_workflow_node("_solver_node")
    state.update(update)
    state["solver_attempt"] += 1
    state["verification"] = {}
    solved = state["result"]
    # 运行记录先标记为候选，避免预算中止后历史页面出现未经接受的成功解。
    if isinstance(solved.get("run_id"), int):
        pending = dict(solved, status="PENDING_VERIFICATION", solver_status=solved.get("status"),
                       workflow_verification={"passed": False, "scope": "pending_acceptance"})
        update_run_result(solved["run_id"], pending)
    return {"summary": update.get("_trace_detail"), "result_ref": solved.get("run_id"),
            "status": solved.get("status"), "objective_value": solved.get("objective_value"),
            "verification_required": True, "warnings": solved.get("warnings", [])}


def verify(context: AgentToolContext, decision: ControllerDecision) -> dict:
    """执行 verify，沿用现有作用域和数学校验约束。"""
    state = context.state
    update = context.run_workflow_node("_verifier_node")
    state.update(update)
    state["recovery_feedback"] = update["verification"].get("errors", [])
    return {"summary": update.get("_trace_detail"), "verification": update["verification"]}


def compare_plans(context: AgentToolContext, decision: ControllerDecision) -> dict:
    """执行 compare_plans，沿用现有作用域和数学校验约束。"""
    state = context.state
    question = context.question
    requested_dataset_id = context.requested_dataset_id
    user_id = context.user_id
    conversation_id = context.conversation_id
    from api.services.plan_comparison import compare_plans
    current = None
    if (state.get("verification") or {}).get("passed"):
        current = dict(state["result"], requirement_analysis=state["requirement_analysis"], workflow_verification=state["verification"])
    comparison = compare_plans(user_id, conversation_id, current=current)
    # 比较是分析结果，不能覆盖尚未验算的求解结果。
    if current is not None:
        state["result"]["plan_comparison"] = comparison
    else:
        brief = RequirementBrief.model_validate(state["requirement_analysis"])
        state["result"] = build_clarification_result(question=question, brief=brief,
            requested_dataset_id=None, user_id=user_id, conversation_id=conversation_id,
            response_status="COMPARISON", message=comparison["message"], comparison=comparison)
    return {"summary": comparison["message"], "comparison": comparison}


def retrieve_memory(context: AgentToolContext, decision: ControllerDecision) -> dict:
    """执行 retrieve_memory，沿用现有作用域和数学校验约束。"""
    question = context.question
    user_id = context.user_id
    conversation_id = context.conversation_id
    memories = context.memories
    found = search_memories(user_id=user_id, conversation_id=conversation_id, query=decision.query or question) if conversation_id is not None else []
    memories[:] = found
    return {"summary": f"检索到 {len(found)} 条显式记忆", "memory_ids": [row["id"] for row in found]}


def retrieve_knowledge(context: AgentToolContext, decision: ControllerDecision) -> dict:
    """执行 retrieve_knowledge，沿用现有作用域和数学校验约束。"""
    question = context.question
    docs = retrieve(decision.query or question, top_k=2)
    return {"summary": "已检索本地知识", "documents": [
        {"title": doc.title, "source": doc.source, "content": doc.content[:1400]} for doc in docs]}
