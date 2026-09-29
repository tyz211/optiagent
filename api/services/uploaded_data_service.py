from __future__ import annotations

from io import StringIO
import json
import math
import re

import pandas as pd

from optiagent.data import SupplyChainData
from optiagent.llm import LLMConfig, call_openai_compatible_chat, clamp_probability, parse_json_object
from optiagent.optimization_gateway import solve_generic_via_gateway, solve_question_via_gateway
from optiagent.problem_spec import infer_problem_spec
from optiagent.schema_mapping import (
    TableMapping,
    apply_table_mapping,
    assemble_facility_data,
    infer_facility_table,
    mapping_summary,
)


# 当前已具备完整数据适配与求解链路的通用优化模板。
EXECUTABLE_TEMPLATE_IDS = {
    "linear_program",
    "knapsack",
    "assignment",
    "tsp",
    "job_shop_scheduling",
    "production_mix",
}


def problem_spec_for_template(question: str, template_id: str):
    """优先使用指定模板构建 ProblemSpec，模板不存在时回退到本地推断。"""

    from optiagent.templates.registry import get_template

    if template_id:
        try:
            if template_id == "linear_program":
                # 展示当前执行合同，避免数学模型结果仍显示通用占位目标。
                from dataclasses import replace
                from optiagent.linear_solver import describe_linear_model, extract_linear_data
                data, _, _ = extract_linear_data(question)
                objective, variables, constraints = describe_linear_model(data)
                return replace(get_template(template_id).build_spec(question, None),
                               objective=objective, decision_variables=variables, constraints=constraints)
            return get_template(template_id).build_spec(question, None)
        except KeyError:
            pass
    return infer_problem_spec(question, None)


def solve_uploaded_generic(question: str, files: list[dict], preferred_template: str | None = None):
    """把上传表格适配为标准模板输入，并通过统一 Gateway 求解。"""

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
                return solve_generic_via_gateway(
                    "knapsack",
                    data,
                    data_source=f"上传文件：{filename}",
                    question=question,
                )
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
                return solve_generic_via_gateway(
                    "assignment",
                    data,
                    data_source=f"上传文件：{filename}",
                    question=question,
                )
    if _should_try_template(preferred_template, lowered, "tsp", ["旅行商", "tsp", "巡回", "最短路径"], roles):
        for frame, filename in frames:
            normalized = _normalize_generic_frame(frame)
            columns = {str(column).strip().lower() for column in normalized.columns}
            if {"from", "to", "distance"}.issubset(columns):
                data = {"distances": normalized.to_dict(orient="records")}
                return solve_generic_via_gateway(
                    "tsp",
                    data,
                    data_source=f"上传文件：{filename}",
                    question=question,
                )
            coordinate_data = _coordinates_to_tsp_data(normalized)
            if coordinate_data:
                return solve_generic_via_gateway(
                    "tsp",
                    coordinate_data,
                    data_source=f"上传文件：{filename}",
                    question=question,
                    warnings=["上传文件为坐标表，已按欧氏距离构造 TSP 距离矩阵。"],
                )
    if _should_try_template(
        preferred_template,
        lowered,
        "job_shop_scheduling",
        ["调度", "排产", "工序", "job", "schedule"],
        roles,
    ):
        for frame, filename in frames:
            normalized = _normalize_generic_frame(frame)
            columns = {str(column).strip().lower() for column in normalized.columns}
            if {"job", "machine", "duration"}.issubset(columns):
                data = {"tasks": normalized.to_dict(orient="records")}
                return solve_generic_via_gateway(
                    "job_shop_scheduling",
                    data,
                    data_source=f"上传文件：{filename}",
                    question=question,
                )
    if _should_try_template(
        preferred_template,
        lowered,
        "production_mix",
        ["产品组合", "生产计划", "利润", "资源约束", "milp"],
        roles,
    ):
        production = _extract_production_mix_from_files(frames, question)
        if production:
            data, filename = production
            return solve_generic_via_gateway(
                "production_mix",
                data,
                data_source=f"上传文件：{filename}",
                question=question,
            )
    return None


def facility_data_from_uploaded_files(files: list[dict]) -> SupplyChainData | None:
    """使用本地字段规则组装仓库选址数据。"""

    frames: list[tuple[pd.DataFrame, str, str | None]] = []
    for item in files:
        try:
            frame, filename = _uploaded_item_to_frame(item)
        except Exception:
            continue
        frames.append((frame, filename, item.get("role")))
    return assemble_facility_data(frames).data


def facility_data_from_llm_mapping(
    question: str,
    files: list[dict],
    llm_config: LLMConfig | None,
) -> tuple[SupplyChainData | None, str | None]:
    """在本地映射失败时，请 LLM 只做字段映射并再次执行数据校验。"""

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
        payload = parse_json_object(raw, error_message="LLM 字段映射未返回 JSON 对象。")
    except Exception as exc:
        return None, f"LLM 字段映射失败：{type(exc).__name__}。"

    confidence = clamp_probability(payload.get("confidence"))
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


def facility_schema_diagnostics(files: list[dict]) -> str | None:
    """解释仓库选址数据为何未通过本地 Schema 组装。"""

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
        *assembly.warnings[:4],
    ]
    return "\n".join(lines)


def has_executable_generic_upload(files: list[dict], preferred_template: str | None = None) -> bool:
    """检查上传文件是否包含任一通用模板所需的最小字段集合。"""

    roles = {str(item.get("role") or "") for item in files}
    if preferred_template in EXECUTABLE_TEMPLATE_IDS and preferred_template in roles:
        return True
    if roles & EXECUTABLE_TEMPLATE_IDS:
        return True
    required_column_sets = (
        {"item", "value", "weight"},
        {"resource", "task", "cost"},
        {"from", "to", "distance"},
        {"city", "x", "y"},
        {"job", "machine", "duration"},
        {"product", "profit"},
    )
    for item in files:
        try:
            frame, _filename = _uploaded_item_to_frame(item)
        except Exception:
            continue
        normalized = _normalize_generic_frame(frame)
        columns = {str(column).strip().lower() for column in normalized.columns}
        if any(required.issubset(columns) for required in required_column_sets):
            return True
    return False


def solve_json_generic(question: str, preferred_template: str | None = None):
    """解析问题中的内联 JSON，并通过统一 Gateway 尝试求解。"""

    if preferred_template in EXECUTABLE_TEMPLATE_IDS:
        problem_spec = problem_spec_for_template(question, preferred_template)
    else:
        problem_spec = infer_problem_spec(question, None)
    result = solve_question_via_gateway(question, problem_spec)
    if result and result.template_id == "linear_program":
        # 求解器故障也要保留真实状态，不能回落成误导性的“请上传 CSV”。
        return result
    if result and result.status not in {"ERROR", "INVALID_DATA", "SOLVER_ERROR"}:
        return result
    return None


def local_uploaded_file_answer(files: list[dict]) -> str:
    """在 LLM 不可用时生成确定性的上传文件摘要。"""

    lines = ["我已读取最近上传的 CSV 文件："]
    for item in files:
        columns = json.loads(item["columns_json"])
        csv_text = item.get("content_csv") or item["preview_csv"]
        row_count = max(len(csv_text.splitlines()) - 1, 0)
        lines.append(f"- {item['filename']}：{row_count} 行，字段包括 {', '.join(columns)}。")
    lines.append("请在问题中说明要优化的目标、约束和容量/预算等参数，我可以继续帮你转成优化模型。")
    return "\n".join(lines)


def _should_try_template(
    preferred_template: str | None,
    lowered_question: str,
    template_id: str,
    keywords: list[str],
    uploaded_roles: set[str] | None = None,
) -> bool:
    """按 Planner 结果、文件角色和关键词决定是否尝试某模板。"""

    if preferred_template:
        return preferred_template == template_id
    if uploaded_roles and template_id in uploaded_roles:
        return True
    return any(keyword.lower() in lowered_question for keyword in keywords)


def _coordinates_to_tsp_data(frame: pd.DataFrame) -> dict | None:
    """将城市坐标表转换为完整的欧氏距离矩阵。"""

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
    for source_index, source in enumerate(rows):
        for target_index, target in enumerate(rows):
            if source_index == target_index:
                continue
            distance = math.dist((float(source[1]), float(source[2])), (float(target[1]), float(target[2])))
            distances.append({"from": str(source[0]), "to": str(target[0]), "distance": distance})
    return {"distances": distances}


def _uploaded_item_to_frame(item: dict) -> tuple[pd.DataFrame, str]:
    """将数据库中的上传文件记录还原为 DataFrame。"""

    csv_text = item.get("content_csv") or item["preview_csv"]
    return pd.read_csv(StringIO(csv_text)), item["filename"]


def _extract_capacity(text: str) -> float | None:
    """从自然语言中提取背包容量或预算上限。"""

    match = re.search(r"(?:容量|capacity|预算|限制)[^0-9]{0,8}([0-9]+(?:\.[0-9]+)?)", text, flags=re.IGNORECASE)
    return float(match.group(1)) if match else None


def _normalize_generic_frame(frame: pd.DataFrame) -> pd.DataFrame:
    """把常见中英文字段别名统一为模板标准字段。"""

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
    """从产品表和问题文本中组合生产计划模板输入。"""

    for frame, filename in frames:
        normalized = _normalize_generic_frame(frame)
        columns = {str(column).strip().lower() for column in normalized.columns}
        if {"product", "profit"}.issubset(columns):
            capacities = _extract_capacities(question, normalized)
            if capacities:
                return {"products": normalized.to_dict(orient="records"), "capacities": capacities}, filename
    return None


def _extract_capacities(question: str, products: pd.DataFrame) -> dict[str, float]:
    """从 JSON 片段或自然语言中提取各资源容量。"""

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
