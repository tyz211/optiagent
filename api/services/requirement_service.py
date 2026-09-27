from __future__ import annotations

from api.database import (
    get_active_dataset_id_or_none,
    get_active_llm_config,
    get_conversation_requirement,
    list_runs,
    list_uploaded_files,
    save_conversation_requirement,
    save_run,
    touch_conversation,
)
from optiagent.llm import llm_config_from_record
from optiagent.requirement_analysis import RequirementBrief, analyze_requirements


def analyze_requirement_turn(
    *,
    question: str,
    requested_dataset_id: int | None,
    user_id: int | None,
    conversation_id: int | None,
) -> RequirementBrief:
    """读取会话记忆、分析本轮补充，并保存新的需求状态。"""

    dataset_id = requested_dataset_id or get_active_dataset_id_or_none(
        user_id=user_id,
        conversation_id=conversation_id,
    )
    files = list_uploaded_files(limit=30, user_id=user_id, conversation_id=conversation_id)
    previous = get_conversation_requirement(conversation_id, user_id=user_id)
    history = list_runs(limit=100, user_id=user_id, conversation_id=conversation_id)[-6:]
    llm_config = llm_config_from_record(get_active_llm_config(user_id))
    brief = analyze_requirements(
        question,
        previous=previous,
        history=history,
        uploaded_files=files,
        dataset_id=dataset_id,
        llm_config=llm_config,
    )
    save_conversation_requirement(conversation_id, brief.model_dump(mode="json"), user_id=user_id)
    return brief


def build_clarification_result(
    *,
    question: str,
    brief: RequirementBrief,
    requested_dataset_id: int | None,
    user_id: int | None,
    conversation_id: int | None,
    response_status: str = "NEEDS_CLARIFICATION",
    message: str | None = None,
    comparison: dict | None = None,
) -> dict:
    """把未完成需求转换为兼容聊天界面的结构化追问并记录本轮。"""

    questions = brief.clarification_questions or ["请补充目标、约束和可用于求解的数据。"]
    conclusion = "我先整理需求，当前信息还不足以安全进入求解。"
    answer = "\n".join(
        [
            conclusion,
            f"已理解：{brief.summary}",
            "请继续补充：",
            *[f"{index}. {item}" for index, item in enumerate(questions, start=1)],
        ]
    )
    # 比较与仅保存修改也结束当前轮次，但不触发求解器。
    if message is not None:
        conclusion = message
        answer = message
        questions = []
    result = {
        "answer": answer,
        "structured_answer": {
            "conclusion": conclusion,
            "metrics": {
                "objective_label": "需求状态",
                "extra": [
                    {"label": "对话轮次", "value": brief.turn_count},
                    {"label": "已确认约束", "value": len(brief.constraints)},
                    {"label": "待确认信息", "value": len(brief.missing_information)},
                ],
            },
            "recommendations": questions,
            "risks": ["在目标、约束或数据未确认前调用求解器，可能得到错误但形式完整的方案。"] if message is None else [],
            "evidence": [*brief.known_facts[:3], *brief.data_sources[:3]],
            "raw_answer": answer,
        },
        "requirement_analysis": brief.model_dump(mode="json"),
        "problem_spec": None,
        "problem_summary": [],
        "rag_context": {},
        "generic_result": None,
        "question": question,
        "status": response_status,
        "plan_comparison": comparison,
        "objective_value": None,
        "transport_cost": None,
        "fixed_cost": None,
        "baseline_objective": None,
        "open_warehouses": [],
        "scenario_changes": [],
        "warnings": brief.missing_information,
        "explanation": [],
        "rag_notes": [],
        "rag_docs": [],
        "tool_names": ["conversation_memory", "requirement_analyzer"],
        "agent_steps": [
            {
                "step": "需求分析",
                "tool": "requirement_analyzer",
                "output": brief.summary,
            },
            {
                "step": "求解门控",
                "tool": "requirement_gate",
                "output": "信息不足，未调用 Solver。" if message is None else "按本轮请求返回记录，未调用 Solver。",
            },
        ],
        "mcp_status": "需求澄清阶段未调用 MCP。" if message is None else "本轮未调用求解器。",
        "warehouse_summary": [],
        "allocations": [],
        "conversation_id": conversation_id,
    }
    run_id = save_run(
        requested_dataset_id,
        question,
        answer,
        result,
        user_id=user_id,
        conversation_id=conversation_id,
    )
    result["run_id"] = run_id
    touch_conversation(conversation_id, question)
    return result
