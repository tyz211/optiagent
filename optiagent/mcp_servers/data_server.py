from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any

import pandas as pd
from mcp.server.fastmcp import FastMCP
from pydantic import BaseModel, Field

from optiagent.data import normalize_data
from optiagent.mcp_contracts import (
    ProblemEnvelope,
    SourceReference,
    ValidationReport,
)
from optiagent.mcp_servers.common import json_safe, resolve_readable_path, run_server
from optiagent.mcp_validation import validate_problem_data
from optiagent.optimization_gateway import build_problem_envelope
from optiagent.schema_mapping import assemble_facility_data
from optiagent.templates.registry import get_template, list_templates


class DataProfileResult(BaseModel):
    """数据文件的结构和质量摘要。"""

    path: str
    rows: int
    columns: list[str]
    dtypes: dict[str, str]
    missing_values: dict[str, int]
    duplicate_rows: int
    sample: list[dict[str, Any]] = Field(default_factory=list)
    numeric_summary: dict[str, dict[str, Any]] = Field(default_factory=dict)


mcp = FastMCP(
    "OptiAgent Data MCP",
    instructions=(
        "负责读取和分析结构化数据，将 CSV/Excel/JSON 转为带来源和校验报告的 ProblemEnvelope。"
        "不负责运行优化求解器。"
    ),
    json_response=True,
)


@mcp.tool(title="列出数据模板")
def data_list_templates() -> list[dict[str, Any]]:
    """列出 Data MCP 可标准化的优化问题模板。"""

    return [
        {
            "template_id": template.template_id,
            "display_name": template.display_name,
            "problem_type": template.problem_type,
        }
        for template in list_templates()
    ]


@mcp.tool(title="生成数据画像")
def data_profile(path: str, sheet_name: str | None = None, sample_rows: int = 5) -> DataProfileResult:
    """读取 CSV、Excel 或表格型 JSON，返回列、缺失值、重复行和数值分布。"""

    source = resolve_readable_path(path)
    frame = _read_table(source, sheet_name=sheet_name)
    sample_rows = min(max(sample_rows, 1), 20)
    numeric = frame.select_dtypes(include="number")
    summary: dict[str, dict[str, Any]] = {}
    if not numeric.empty:
        described = numeric.describe().to_dict()
        summary = {
            str(column): {str(metric): json_safe(value) for metric, value in values.items()}
            for column, values in described.items()
        }
    return DataProfileResult(
        path=str(source),
        rows=len(frame),
        columns=[str(column) for column in frame.columns],
        dtypes={str(column): str(dtype) for column, dtype in frame.dtypes.items()},
        missing_values={str(column): int(count) for column, count in frame.isna().sum().items()},
        duplicate_rows=int(frame.duplicated().sum()),
        sample=_records(frame.head(sample_rows)),
        numeric_summary=summary,
    )


@mcp.tool(title="构建优化问题数据")
def data_build_problem(
    template_id: str,
    paths: list[str],
    parameters: dict[str, Any] | None = None,
    sheet_names: list[str] | None = None,
) -> ProblemEnvelope:
    """把本地 CSV/Excel/JSON 数据映射到统一 ProblemEnvelope，并执行前置校验。"""

    if not paths:
        raise ValueError("至少需要一个数据文件。")
    try:
        get_template(template_id)
    except KeyError as exc:
        supported = "、".join(item.template_id for item in list_templates())
        raise ValueError(f"不支持的模板 {template_id}；可选：{supported}") from exc

    parameters = dict(parameters or {})
    sheets = list(sheet_names or [])
    resolved = [resolve_readable_path(path) for path in paths]
    frames: list[tuple[pd.DataFrame, str]] = []
    json_payload: dict[str, Any] | None = None
    for index, source in enumerate(resolved):
        sheet = sheets[index] if index < len(sheets) else None
        if source.suffix.lower() == ".json" and len(resolved) == 1:
            raw = json.loads(source.read_text(encoding="utf-8-sig"))
            if isinstance(raw, dict):
                json_payload = raw.get(template_id, raw)
                continue
        frames.append((_read_table(source, sheet_name=sheet), source.name))

    if json_payload is not None:
        data = {**json_payload, **parameters}
        warnings: list[str] = []
    else:
        data, warnings = _build_template_data(template_id, frames, parameters)

    return build_problem_envelope(
        template_id,
        data,
        sources=[SourceReference(kind="file", name=source.name, uri=str(source)) for source in resolved],
        warnings=warnings,
        metadata={"builder": "OptiAgent Data MCP", "parameters": json_safe(parameters)},
    )


@mcp.tool(title="校验优化数据")
def data_validate_problem(template_id: str, data: dict[str, Any]) -> ValidationReport:
    """按指定模板校验数据完整性与基本可行性。"""

    return validate_problem_data(template_id, data)


def _build_template_data(
    template_id: str,
    frames: list[tuple[pd.DataFrame, str]],
    parameters: dict[str, Any],
) -> tuple[dict[str, Any], list[str]]:
    if not frames:
        return parameters, ["未从文件中读取到表格数据。"]
    if template_id == "facility_location":
        assembly = assemble_facility_data([(frame, name, None) for frame, name in frames])
        if assembly.data is None:
            return parameters, assembly.warnings
        data = normalize_data(assembly.data)
        return {
            "warehouses": _records(data.warehouses),
            "customers": _records(data.customers),
            "costs": _records(data.costs),
            **parameters,
        }, assembly.warnings

    normalized = [(_normalize_columns(frame), name) for frame, name in frames]
    if template_id == "knapsack":
        frame, _ = _find_frame(normalized, {"item", "value", "weight"})
        return {"items": _records(frame), **parameters}, []
    if template_id == "assignment":
        frame, _ = _find_frame(normalized, {"resource", "task", "cost"})
        return {
            "resources": sorted(frame["resource"].astype(str).unique().tolist()),
            "tasks": sorted(frame["task"].astype(str).unique().tolist()),
            "costs": _records(frame),
            **parameters,
        }, []
    if template_id == "tsp":
        for frame, _name in normalized:
            if {"from", "to", "distance"}.issubset(frame.columns):
                return {"distances": _records(frame), **parameters}, []
            if {"city", "x", "y"}.issubset(frame.columns):
                return {
                    "distances": _coordinate_distances(frame),
                    **parameters,
                }, ["坐标表已按欧氏距离转换为 TSP 距离矩阵。"]
        raise ValueError("未找到 from/to/distance 距离表或 city/x/y 坐标表。")
    if template_id == "job_shop_scheduling":
        frame, _ = _find_frame(normalized, {"job", "machine", "duration"})
        if "order" not in frame.columns:
            frame = frame.copy()
            frame["order"] = frame.groupby("job").cumcount()
        return {"tasks": _records(frame), **parameters}, []
    if template_id == "production_mix":
        products, _ = _find_frame(normalized, {"product", "profit"})
        capacities = parameters.get("capacities")
        if capacities is None:
            for frame, _name in normalized:
                if {"resource", "capacity"}.issubset(frame.columns):
                    capacities = dict(zip(frame["resource"].astype(str), frame["capacity"], strict=False))
                    break
        return {"products": _records(products), "capacities": json_safe(capacities or {}), **parameters}, []
    return parameters, [f"暂不支持数据构建：{template_id}"]


def _read_table(path: Path, sheet_name: str | None = None) -> pd.DataFrame:
    suffix = path.suffix.lower()
    if suffix == ".csv":
        for encoding in ("utf-8-sig", "utf-8", "gb18030", "gbk"):
            try:
                return pd.read_csv(path, encoding=encoding)
            except UnicodeDecodeError:
                continue
        raise ValueError(f"无法识别 CSV 编码：{path.name}")
    if suffix in {".xlsx", ".xls"}:
        try:
            return pd.read_excel(path, sheet_name=sheet_name or 0)
        except ImportError as exc:
            raise RuntimeError("读取 Excel 需要安装 openpyxl（.xlsx）或 xlrd（.xls）。") from exc
    if suffix == ".json":
        payload = json.loads(path.read_text(encoding="utf-8-sig"))
        if isinstance(payload, list):
            return pd.DataFrame(payload)
        if isinstance(payload, dict):
            for value in payload.values():
                if isinstance(value, list):
                    return pd.DataFrame(value)
        raise ValueError("JSON 不包含可转换为表格的数组。")
    raise ValueError(f"Data MCP 不支持的文件格式：{suffix}")


def _normalize_columns(frame: pd.DataFrame) -> pd.DataFrame:
    aliases = {
        "item": {"item", "item_id", "name", "物品", "物品编号", "项目"},
        "value": {"value", "profit", "benefit", "价值", "收益", "利润"},
        "weight": {"weight", "cost", "resource", "重量", "成本", "资源消耗"},
        "resource": {"resource", "worker", "person", "employee", "资源", "员工", "人员"},
        "task": {"task", "job", "shift", "任务", "岗位", "班次"},
        "cost": {"cost", "price", "distance", "成本", "费用", "距离"},
        "from": {"from", "source", "origin", "起点", "来源"},
        "to": {"to", "target", "destination", "终点", "目的地"},
        "distance": {"distance", "cost", "length", "距离", "路程", "成本"},
        "job": {"job", "job_id", "order_id", "作业", "工件", "订单"},
        "machine": {"machine", "machine_id", "机器", "机台", "设备"},
        "duration": {"duration", "processing_time", "time", "时长", "加工时间", "工时"},
        "order": {"order", "sequence", "operation", "顺序", "工序", "序号"},
        "product": {"product", "product_id", "name", "产品", "产品编号"},
        "profit": {"profit", "value", "margin", "利润", "收益", "贡献毛利"},
        "capacity": {"capacity", "limit", "容量", "上限"},
        "city": {"city", "node", "name", "城市", "节点", "名称"},
        "x": {"x", "longitude", "lng", "lon", "经度", "横坐标"},
        "y": {"y", "latitude", "lat", "纬度", "纵坐标"},
    }
    output = frame.copy()
    rename: dict[str, str] = {}
    used: set[str] = set()
    for column in output.columns:
        normalized = str(column).strip().lower().lstrip("\ufeff")
        if normalized in aliases and normalized not in used:
            rename[column] = normalized
            used.add(normalized)
            continue
        for target, names in aliases.items():
            if normalized in names and target not in used:
                rename[column] = target
                used.add(target)
                break
    return output.rename(columns=rename)


def _find_frame(frames: list[tuple[pd.DataFrame, str]], required: set[str]) -> tuple[pd.DataFrame, str]:
    for frame, name in frames:
        if required.issubset(frame.columns):
            return frame, name
    raise ValueError("未找到包含字段的数据表：" + "、".join(sorted(required)))


def _records(frame: pd.DataFrame) -> list[dict[str, Any]]:
    # 通过 pandas JSON 通道统一处理 NaN、numpy 数值和时间类型。
    return json.loads(frame.to_json(orient="records", force_ascii=False, date_format="iso"))


def _coordinate_distances(frame: pd.DataFrame) -> list[dict[str, Any]]:
    points = [
        (str(row.city), float(row.x), float(row.y))
        for row in frame[["city", "x", "y"]].itertuples(index=False)
    ]
    distances: list[dict[str, Any]] = []
    for source, source_x, source_y in points:
        for target, target_x, target_y in points:
            if source == target:
                continue
            distances.append(
                {
                    "from": source,
                    "to": target,
                    "distance": math.hypot(source_x - target_x, source_y - target_y),
                }
            )
    return distances


if __name__ == "__main__":
    run_server(mcp, default_port=8102)
