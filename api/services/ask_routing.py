from __future__ import annotations

import json

from api.services.uploaded_data_service import (
    EXECUTABLE_TEMPLATE_IDS,
    facility_data_from_uploaded_files,
    has_executable_generic_upload,
)
from optiagent.llm import LLMConfig, call_openai_compatible_chat, clamp_probability, parse_json_object


def build_agent_plan(
    question: str,
    files: list[dict],
    llm_config: LLMConfig | None,
) -> tuple[dict | None, str | None]:
    """让 LLM 选择问题模板与工具链，并对输出执行确定性校验。"""

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
            "mcp_gateway",
            "data_validate_problem",
            "solver_solve_problem",
            "uploaded_file_context",
        ],
    }
    try:
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
                        "当用户要求求解优化问题时，tool_chain 必须依次包含 "
                        "mcp_gateway、data_validate_problem 和 solver_solve_problem。"
                        "工具计划必须要求求解器返回可证明最优解；如果工具只能给启发式可行解，必须在结果中标记未证明最优。"
                        "只输出 JSON，不要输出 Markdown。"
                        "JSON 字段：template_id, confidence, objective, selected_file, tool_chain, reasoning, needs_solver, data_gaps。"
                        "template_id 只能是 knapsack、assignment、tsp、job_shop_scheduling、production_mix、facility_location、file_answer。"
                    ),
                },
                {"role": "user", "content": json.dumps(prompt, ensure_ascii=False)},
            ],
        )
        plan = parse_json_object(raw, error_message="LLM 路由未返回 JSON 对象。")
    except Exception as exc:
        return None, f"LLM 路由暂时不可用，已回退到本地规则路由（{type(exc).__name__}）。"

    template_id = str(plan.get("template_id", "")).strip()
    if template_id not in {*EXECUTABLE_TEMPLATE_IDS, "facility_location", "file_answer"}:
        return None, "LLM 路由返回了未知问题类型，已回退到本地规则路由。"
    tool_chain = plan.get("tool_chain")
    if not isinstance(tool_chain, list) or not tool_chain:
        plan["tool_chain"] = _default_tool_chain(template_id)
    has_executable_upload = bool(
        facility_data_from_uploaded_files(files) or has_executable_generic_upload(files, template_id)
    )
    if needs_external_data(question) and not has_executable_upload and "web_search_tool" not in plan["tool_chain"]:
        insert_at = 1 if plan["tool_chain"] else 0
        plan["tool_chain"].insert(insert_at, "web_search_tool")
    plan["template_id"] = template_id
    plan["confidence"] = clamp_probability(plan.get("confidence"))
    plan["llm_used"] = True
    return plan, None


def should_solve_optimization(question: str) -> bool:
    """判断用户当前是在要求求解，还是仅分析已有数据。"""

    text = (question or "").lower()
    strong_solve_keywords = [
        "求解",
        "解决",
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


def needs_external_data(question: str) -> bool:
    """判断问题是否明确依赖当前或公开的外部事实。"""

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


def external_data_gaps(question: str, files: list[dict]) -> list[str]:
    """根据问题类型列出执行优化仍缺少的事实参数。"""

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


def is_facility_question(question: str, preferred_template: str | None = None) -> bool:
    """识别仓库选址及客户分配类问题。"""

    if preferred_template == "facility_location":
        return True
    lowered = question.lower()
    keywords = ["仓库", "仓", "选址", "facility", "warehouse", "固定成本", "运输成本", "客户分配", "利用率"]
    return any(keyword in lowered for keyword in keywords)


def _default_tool_chain(template_id: str) -> list[str]:
    """为 LLM 未给出工具链时提供确定性的默认计划。"""

    if template_id in {*EXECUTABLE_TEMPLATE_IDS, "facility_location"}:
        return [
            "problem_spec_tool",
            "rag_context_pack_tool",
            "mcp_gateway",
            "data_validate_problem",
            "solver_solve_problem",
            "result_formatter",
        ]
    return ["uploaded_file_context"]
