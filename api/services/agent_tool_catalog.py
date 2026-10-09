"""内置工具组装与业务前置条件，不承载主控循环。"""

from __future__ import annotations

from api.services.agent_decision import (
    ClarificationParameters, ExplanationParameters, NoParameters,
    PatchParameters, QueryParameters, ResultReferenceParameters,
)
from api.services.agent_tool_registry import AgentTool
from api.services import agent_tools as handlers
from api.services.result_evidence import explanation_request


ACTION_DESCRIPTIONS = {
    "analyze_requirements": "分析本轮原话并更新有效需求版本；每轮一次，不会自行改写用户条件",
    "read_requirement": "按 query（实体名称或字段名）读取当前背包/运输可编辑字段和版本，每次最多二十项",
    "apply_requirement_patch": "使用 patch 原子修改当前背包/运输数值；完整覆盖本轮原文，提供字符位置、旧版本和精确单位",
    "read_verified_result": "读取当前会话已验算结果；仅在用户明确指定方案 #编号时填 run_id，query 可筛选决策实体",
    "explain_result": "使用 fact_ids 选择已读取事实，程序生成解释和来源；不允许编造数字或因果结论",
    "inspect_data": "读取当前用户和会话的数据元信息；大表保留在本地",
    "build_model": "按有效需求构建当前支持的模板模型，携带上次验算反馈",
    "solve": "对当前有效版本调用本地求解器；结果仍需 verify",
    "verify": "独立复算当前求解结果与响应合同，给出可信度证据",
    "compare_plans": "比较当前会话内已验算的方案，先分析本轮意图",
    "retrieve_memory": "按 query 检索当前用户和会话的显式长期记忆，仅作为参考",
    "retrieve_knowledge": "按 query 检索本地运筹知识，不自动修改任务",
    "clarify": "使用 message 向用户追问；保存任务并暂停，绝不接受未验算解",
    "finish": "返回程序生成的结果；成功解必须经过本轮 verify，不能自行编造结论",
}
ACTION_LABELS = {
    "controller": "规划下一步", "analyze_requirements": "整理需求", "inspect_data": "检查数据",
    "build_model": "建立模型", "solve": "求解方案", "verify": "独立验算",
    "compare_plans": "比较方案", "retrieve_memory": "查阅记忆", "retrieve_knowledge": "查阅知识",
    "clarify": "追问与暂停", "finish": "返回结论",
    "read_requirement": "读取有效需求", "apply_requirement_patch": "核对并保存修改",
    "read_verified_result": "读取已验算方案", "explain_result": "解释方案依据",
}

def _always(state, counts) -> bool:
    """读取与澄清不依赖数学候选的成功状态。"""
    return True


def _editable(state, counts) -> bool:
    """只开放已实现原文核对能力的模板字段。"""
    contract = (state.get("requirement_analysis") or {}).get("dialogue_contract") or {}
    return bool(contract.get("data") and contract.get("template_id") in {"knapsack", "transportation"})


def _can_patch(state, counts) -> bool:
    """解释轮禁止修改，需求分析完成后本轮也不再改写。"""
    return _editable(state, counts) and not state.get("requirements_done") and not state.get("analysis_only")


def _can_read_result(state, counts) -> bool:
    """只在明确解释意图或解释证据失效后读取源方案。"""
    if state.get("analysis_only"):
        # 证据已经读取后必须转入解释，避免主控重复读取同一方案消耗预算。
        return not state.get("result") and not state.get("result_evidence")
    return bool(explanation_request(state.get("original_question", ""))[0]
                and not state.get("result") and not state.get("requirements_done"))


def _can_explain(state, counts) -> bool:
    """解释必须先持有工具读取的事实证据。"""
    return bool(state.get("result_evidence") and not state.get("result"))


def _can_analyze(state, counts) -> bool:
    """每轮只完成一次需求分析，解释不重启求解。"""
    return not state.get("requirements_done") and not state.get("analysis_only")


def _ready_to_solve(state) -> bool:
    """共享求解条件，保留运输不可行和验算通过后的终止门控。"""
    brief = state.get("requirement_analysis") or {}
    contract = brief.get("dialogue_contract") or {}
    verification = state.get("verification") or {}
    terminal = verification.get("terminal_failure") or (
        brief.get("template_id") == "transportation" and (state.get("result") or {}).get("status") == "INFEASIBLE")
    return bool(state.get("requirements_done") and not state.get("analysis_only")
                and brief.get("readiness") == "ready_to_solve" and contract.get("action") not in {"hold", "compare"}
                and not terminal and not verification.get("passed"))


def _can_build(state, counts) -> bool:
    """建模最多两次，不能绕过本轮分析。"""
    return _ready_to_solve(state) and counts["build_model"] < 2


def _can_solve(state, counts) -> bool:
    """求解必须先持有模型且仍有重试预算。"""
    return bool(_ready_to_solve(state) and state.get("problem_spec") and counts["solve"] < 2)


def _can_verify(state, counts) -> bool:
    """成功候选或本轮求解失败记录可验算，解释引用使用独立门控。"""
    result = state.get("result") or {}
    return bool(not state.get("analysis_only") and state.get("requirements_done") and result
                and not state.get("verification")
                and (result.get("status") in {"OPTIMAL", "NEAR_OPTIMAL", "FEASIBLE"} or counts["solve"]))


def _can_finish(state, counts) -> bool:
    """成功候选必须通过本轮验算，失败和纯分析记录可直接返回。"""
    result = state.get("result")
    if state.get("analysis_only"):
        return bool(result)
    if not state.get("requirements_done"):
        return False
    if result:
        return bool(result.get("status") not in {"OPTIMAL", "NEAR_OPTIMAL", "FEASIBLE"}
                    or (state.get("verification") or {}).get("passed"))
    return (state.get("requirement_analysis") or {}).get("readiness") == "ready_for_analysis"


def _can_compare(state, counts) -> bool:
    """比较不能覆盖未验算候选，只在已分析的任务上开放。"""
    return bool(not state.get("analysis_only") and state.get("requirements_done")
                and ((state.get("requirement_analysis") or {}).get("readiness") == "ready_for_analysis"
                     or (state.get("verification") or {}).get("passed")))


def builtin_tools() -> list[AgentTool]:
    """每项绑定描述、参数、条件与处理函数，新增工具无需改控制器分支。"""
    bindings = {
        "analyze_requirements": (NoParameters, _can_analyze, handlers.analyze_requirements),
        "read_requirement": (QueryParameters, _editable, handlers.read_requirement),
        "apply_requirement_patch": (PatchParameters, _can_patch, handlers.apply_requirement_patch),
        "read_verified_result": (ResultReferenceParameters, _can_read_result, handlers.read_verified_result),
        "explain_result": (ExplanationParameters, _can_explain, handlers.explain_result),
        "inspect_data": (NoParameters, _always, handlers.inspect_data),
        "build_model": (NoParameters, _can_build, handlers.build_model),
        "solve": (NoParameters, _can_solve, handlers.solve),
        "verify": (NoParameters, _can_verify, handlers.verify),
        "compare_plans": (NoParameters, _can_compare, handlers.compare_plans),
        "retrieve_memory": (QueryParameters, _always, handlers.retrieve_memory),
        "retrieve_knowledge": (QueryParameters, _always, handlers.retrieve_knowledge),
        "clarify": (ClarificationParameters, _always, None),
        "finish": (NoParameters, _can_finish, None),
    }
    return [AgentTool(name, ACTION_LABELS[name], description, *bindings[name], terminal=name in {"finish", "clarify"})
            for name, description in ACTION_DESCRIPTIONS.items()]
