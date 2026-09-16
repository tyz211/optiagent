from __future__ import annotations

import json
import re
from typing import Any, Literal

from pydantic import BaseModel, Field, ValidationError

from optiagent.llm import LLMConfig, call_openai_compatible_chat, parse_json_object
from optiagent.templates.registry import get_template, rank_templates


RequirementIntent = Literal["explore", "analyze", "solve", "what_if"]
RequirementReadiness = Literal["needs_clarification", "ready_for_analysis", "ready_to_solve"]


class RequirementBrief(BaseModel):
    """跨轮保存的结构化需求摘要，不包含模型密钥或隐藏推理。"""

    schema_version: str = "1.0"
    turn_count: int = Field(default=1, ge=1)
    source: Literal["local_rule", "llm"] = "local_rule"
    intent: RequirementIntent = "explore"
    template_id: str | None = None
    problem_type: str | None = None
    objective: str | None = None
    constraints: list[str] = Field(default_factory=list)
    known_facts: list[str] = Field(default_factory=list)
    data_sources: list[str] = Field(default_factory=list)
    assumptions: list[str] = Field(default_factory=list)
    missing_information: list[str] = Field(default_factory=list)
    clarification_questions: list[str] = Field(default_factory=list)
    readiness: RequirementReadiness = "needs_clarification"
    summary: str = ""
    resolved_request: str = ""


def analyze_requirements(
    question: str,
    *,
    previous: dict[str, Any] | RequirementBrief | None = None,
    history: list[dict[str, Any]] | None = None,
    uploaded_files: list[dict[str, Any]] | None = None,
    dataset_id: int | None = None,
    llm_config: LLMConfig | None = None,
) -> RequirementBrief:
    """合并当前输入与历史需求，并在必要时使用 LLM 细化需求摘要。"""

    previous_brief = _parse_previous(previous)
    files = uploaded_files or []
    local = _local_analysis(
        question,
        previous=previous_brief,
        uploaded_files=files,
        dataset_id=dataset_id,
    )
    if not llm_config or not llm_config.enabled or not _needs_llm_analysis(local, previous_brief):
        return local
    try:
        candidate = _llm_analysis(
            question,
            previous=previous_brief,
            history=history or [],
            uploaded_files=files,
            dataset_id=dataset_id,
            llm_config=llm_config,
        )
    except Exception:
        return local
    return _guard_llm_analysis(candidate, local, has_runtime_data=_has_runtime_data(question, files, dataset_id))


def _local_analysis(
    question: str,
    *,
    previous: RequirementBrief | None,
    uploaded_files: list[dict[str, Any]],
    dataset_id: int | None,
) -> RequirementBrief:
    clean_question = " ".join((question or "").strip().split())
    previous_request = previous.resolved_request if previous else ""
    resolved_request = (
        f"{previous_request}\n用户补充：{clean_question}".strip()
        if previous_request
        else clean_question
    )
    template_id = _detect_template(resolved_request) or (previous.template_id if previous else None)
    template = get_template(template_id) if template_id else None
    intent = _detect_intent(clean_question, previous)
    objective = _extract_objective(clean_question) or (previous.objective if previous else None)
    assumptions = list(previous.assumptions if previous else [])
    has_runtime_data = _has_runtime_data(clean_question, uploaded_files, dataset_id)
    if objective is None and template is not None and has_runtime_data and intent in {"solve", "what_if"}:
        objective = template.build_spec(resolved_request, None).objective
        assumptions = _unique(
            [*assumptions, f"用户未重复声明目标，暂按标准{template.display_name}目标处理。"]
        )

    constraints = _unique(
        [
            *(previous.constraints if previous else []),
            *_extract_constraint_sentences(clean_question),
        ]
    )
    known_facts = _unique(
        [
            *(previous.known_facts if previous else []),
            *_extract_known_facts(clean_question),
        ]
    )
    data_sources = _data_sources(uploaded_files, dataset_id, clean_question)
    if previous:
        data_sources = _unique([*previous.data_sources, *data_sources])

    missing = _missing_information(
        template_id=template_id,
        objective=objective,
        constraints=constraints,
        intent=intent,
        has_runtime_data=has_runtime_data,
    )
    readiness: RequirementReadiness
    if missing:
        readiness = "needs_clarification"
    elif intent in {"solve", "what_if"}:
        readiness = "ready_to_solve"
    else:
        readiness = "ready_for_analysis"
    questions = _clarification_questions(missing, template_id)
    problem_type = template.display_name if template else (previous.problem_type if previous else None)
    summary = _summary(problem_type, objective, constraints, data_sources, readiness)
    return RequirementBrief(
        turn_count=(previous.turn_count + 1) if previous else 1,
        source="local_rule",
        intent=intent,
        template_id=template_id,
        problem_type=problem_type,
        objective=objective,
        constraints=constraints,
        known_facts=known_facts,
        data_sources=data_sources,
        assumptions=assumptions,
        missing_information=missing,
        clarification_questions=questions,
        readiness=readiness,
        summary=summary,
        resolved_request=resolved_request,
    )


def _llm_analysis(
    question: str,
    *,
    previous: RequirementBrief | None,
    history: list[dict[str, Any]],
    uploaded_files: list[dict[str, Any]],
    dataset_id: int | None,
    llm_config: LLMConfig,
) -> RequirementBrief:
    """让已配置模型只输出需求合同，不要求它生成求解结果。"""

    payload = {
        "current_user_message": question,
        "previous_requirement_brief": previous.model_dump(mode="json") if previous else None,
        "recent_turns": [
            {"question": item.get("question"), "answer": str(item.get("answer") or "")[:800]}
            for item in history[-6:]
        ],
        "available_data": {
            "dataset_id": dataset_id,
            "uploaded_files": [
                {
                    "filename": item.get("filename"),
                    "role": item.get("role"),
                    "columns": _safe_columns(item),
                }
                for item in uploaded_files
            ],
        },
    }
    raw = call_openai_compatible_chat(
        llm_config,
        [
            {
                "role": "system",
                "content": (
                    "你是运筹优化需求分析 Agent。请把多轮对话合并为一个结构化需求摘要，"
                    "区分用户已确认的信息、系统假设和仍需澄清的信息。不要求解，不要编造数据。"
                    "template_id 只能是 facility_location、knapsack、assignment、tsp、"
                    "job_shop_scheduling、production_mix 或 null。"
                    "readiness 只能是 needs_clarification、ready_for_analysis、ready_to_solve；"
                    "只有目标、关键约束和求解数据都足够时才能使用 ready_to_solve。"
                    "只输出 JSON，不要输出 Markdown 或思维过程。字段必须包含："
                    "intent, template_id, problem_type, objective, constraints, known_facts, data_sources, "
                    "assumptions, missing_information, clarification_questions, readiness, summary, resolved_request。"
                ),
            },
            {"role": "user", "content": json.dumps(payload, ensure_ascii=False)},
        ],
    )
    parsed = parse_json_object(raw, error_message="需求分析模型未返回 JSON 对象。")
    parsed["schema_version"] = "1.0"
    parsed["turn_count"] = (previous.turn_count + 1) if previous else 1
    parsed["source"] = "llm"
    return RequirementBrief.model_validate(parsed)


def _guard_llm_analysis(
    candidate: RequirementBrief,
    local: RequirementBrief,
    *,
    has_runtime_data: bool,
) -> RequirementBrief:
    """使用确定性条件约束 LLM readiness，避免在缺少数据时直接进入 Solver。"""

    allowed_templates = {
        "facility_location",
        "knapsack",
        "assignment",
        "tsp",
        "job_shop_scheduling",
        "production_mix",
    }
    if candidate.template_id not in allowed_templates:
        candidate.template_id = local.template_id
        candidate.problem_type = local.problem_type
    candidate.data_sources = _unique([*local.data_sources, *candidate.data_sources])
    candidate.constraints = _unique(candidate.constraints)
    candidate.known_facts = _unique(candidate.known_facts)
    candidate.assumptions = _unique(candidate.assumptions)
    candidate.missing_information = _unique(candidate.missing_information)
    candidate.clarification_questions = _unique(candidate.clarification_questions)[:4]
    if candidate.intent in {"solve", "what_if"} and not has_runtime_data:
        candidate.readiness = "needs_clarification"
        candidate.missing_information = _unique(
            [*candidate.missing_information, "缺少可执行求解所需的数据或参数"]
        )
        candidate.clarification_questions = _unique(
            [*candidate.clarification_questions, "请上传数据文件，或在消息中提供完整参数。"]
        )[:4]
    if candidate.readiness == "ready_to_solve" and (not candidate.template_id or not candidate.objective):
        candidate.readiness = "needs_clarification"
    if not candidate.resolved_request.strip():
        candidate.resolved_request = local.resolved_request
    if not candidate.summary.strip():
        candidate.summary = local.summary
    return candidate


def _parse_previous(value: dict[str, Any] | RequirementBrief | None) -> RequirementBrief | None:
    if isinstance(value, RequirementBrief):
        return value
    if not isinstance(value, dict) or not value:
        return None
    try:
        return RequirementBrief.model_validate(value)
    except ValidationError:
        return None


def _detect_template(text: str) -> str | None:
    ranked = rank_templates(text, None)
    if not ranked:
        return None
    best = ranked[0]
    return best.template_id if best.score(text, None) >= 0.2 else None


def _detect_intent(question: str, previous: RequirementBrief | None) -> RequirementIntent:
    lowered = question.lower()
    if any(keyword in lowered for keyword in ("如果", "假设", "调整", "变化", "what-if", "what if")):
        return "what_if"
    if any(keyword in lowered for keyword in ("求解", "解决", "最优", "最小化", "最大化", "排程", "分配方案", "solve")):
        return "solve"
    if any(keyword in lowered for keyword in ("分析", "评估", "诊断", "建议", "比较", "规划", "analyze")):
        return "analyze"
    if previous and previous.intent in {"solve", "what_if", "analyze"}:
        return previous.intent
    return "explore"


def _extract_objective(question: str) -> str | None:
    objective_words = ("目标", "最小化", "最大化", "降低", "提高", "尽量少", "尽量多", "优化")
    for sentence in _sentences(question):
        if any(word in sentence for word in objective_words):
            return sentence[:300]
    return None


def _extract_constraint_sentences(question: str) -> list[str]:
    markers = ("必须", "至少", "至多", "最多", "不能", "不得", "每个", "容量", "预算", "限制", "约束", "保证")
    return [sentence[:300] for sentence in _sentences(question) if any(marker in sentence for marker in markers)]


def _extract_known_facts(question: str) -> list[str]:
    return [
        sentence[:300]
        for sentence in _sentences(question)
        if re.search(r"\d", sentence) and "{" not in sentence and "}" not in sentence
    ]


def _sentences(text: str) -> list[str]:
    # 中文需求常用逗号串联目标和约束，需要按子句分开才能独立累积。
    return [item.strip() for item in re.split(r"[。！？!?；;，\n]+", text or "") if item.strip()]


def _has_runtime_data(question: str, files: list[dict[str, Any]], dataset_id: int | None) -> bool:
    return dataset_id is not None or bool(files) or ("{" in question and "}" in question)


def _data_sources(files: list[dict[str, Any]], dataset_id: int | None, question: str) -> list[str]:
    sources: list[str] = []
    if dataset_id is not None:
        sources.append(f"结构化数据集 #{dataset_id}")
    sources.extend(
        f"上传文件：{item.get('filename')}（{item.get('role') or '未分类'}）"
        for item in files
        if item.get("filename")
    )
    if "{" in question and "}" in question:
        sources.append("当前消息内联 JSON")
    return _unique(sources)


def _missing_information(
    *,
    template_id: str | None,
    objective: str | None,
    constraints: list[str],
    intent: RequirementIntent,
    has_runtime_data: bool,
) -> list[str]:
    missing: list[str] = []
    if template_id is None:
        missing.append("需要明确要优化的业务对象或问题类型")
    if not objective:
        missing.append("需要明确优化目标及其方向")
    if intent in {"solve", "what_if"} and not has_runtime_data:
        missing.append("缺少可执行求解所需的数据或参数")
    if intent in {"solve", "what_if"} and not constraints and not has_runtime_data:
        missing.append("需要确认关键业务约束")
    return missing


def _clarification_questions(missing: list[str], template_id: str | None) -> list[str]:
    questions: list[str] = []
    for item in missing:
        if "问题类型" in item:
            questions.append("你希望优化的对象是什么，例如选址、排班、路径、指派还是生产计划？")
        elif "优化目标" in item:
            questions.append("你的首要目标是什么，例如最小化成本、最短时间或最大化收益？")
        elif "数据或参数" in item:
            if template_id:
                spec = get_template(template_id).build_spec("", None)
                tables = "；".join(
                    f"{requirement.table}({', '.join(requirement.columns)})"
                    for requirement in spec.data_requirements
                )
                questions.append(f"请上传或提供这些求解数据：{tables}。")
            else:
                questions.append("请上传数据文件，或在消息中提供实体、参数和关系数据。")
        elif "关键业务约束" in item:
            questions.append("有哪些必须满足的限制，例如容量、覆盖、预算、时间窗或人员资格？")
    return _unique(questions)[:4]


def _summary(
    problem_type: str | None,
    objective: str | None,
    constraints: list[str],
    data_sources: list[str],
    readiness: RequirementReadiness,
) -> str:
    parts = [f"问题：{problem_type or '待确认'}", f"目标：{objective or '待确认'}"]
    parts.append(f"约束：已确认 {len(constraints)} 条")
    parts.append(f"数据：{'已提供' if data_sources else '待提供'}")
    labels = {
        "needs_clarification": "需要继续澄清",
        "ready_for_analysis": "可以进入需求分析",
        "ready_to_solve": "可以进入建模求解",
    }
    parts.append(labels[readiness])
    return "；".join(parts) + "。"


def _safe_columns(item: dict[str, Any]) -> list[str]:
    value = item.get("columns_json")
    if isinstance(value, list):
        return [str(column) for column in value]
    if isinstance(value, str):
        try:
            parsed = json.loads(value)
            return [str(column) for column in parsed] if isinstance(parsed, list) else []
        except json.JSONDecodeError:
            return []
    return []


def _needs_llm_analysis(local: RequirementBrief, previous: RequirementBrief | None) -> bool:
    return local.readiness == "needs_clarification" or previous is not None


def _unique(values: list[str]) -> list[str]:
    result: list[str] = []
    seen: set[str] = set()
    for value in values:
        cleaned = str(value).strip()
        if cleaned and cleaned not in seen:
            seen.add(cleaned)
            result.append(cleaned)
    return result
