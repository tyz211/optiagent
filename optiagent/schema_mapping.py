from __future__ import annotations

from dataclasses import dataclass, field
from itertools import permutations
import re

import pandas as pd

from optiagent.data import SupplyChainData, normalize_data, validate_data


FACILITY_ROLES = ("warehouses", "customers", "costs")


@dataclass(frozen=True)
class TableMapping:
    role: str | None
    confidence: float
    column_mapping: dict[str, str]
    method: str
    warnings: list[str] = field(default_factory=list)


@dataclass(frozen=True)
class FacilityAssembly:
    data: SupplyChainData | None
    mappings: dict[str, TableMapping]
    warnings: list[str]
    needs_confirmation: bool = False
    confirmation_message: str = ""


ALIASES: dict[str, dict[str, set[str]]] = {
    "warehouses": {
        "warehouse": {
            "warehouse",
            "warehouses",
            "warehouse_id",
            "warehouse_name",
            "facility",
            "facility_id",
            "facility_name",
            "site",
            "site_id",
            "site_name",
            "name",
            "id",
            "仓库",
            "仓库id",
            "仓库编号",
            "仓库名称",
            "候选仓库",
            "设施",
            "设施点",
            "网点",
            "名称",
            "编号",
        },
        "capacity": {
            "capacity",
            "cap",
            "max_load",
            "maxload",
            "load",
            "supply",
            "available",
            "limit",
            "容量",
            "仓库容量",
            "可用容量",
            "产能",
            "供应量",
            "上限",
        },
        "fixed_cost": {
            "fixed_cost",
            "fixed cost",
            "fixedcost",
            "open_cost",
            "opening_cost",
            "setup_cost",
            "startup_fee",
            "fee",
            "fixed",
            "启动成本",
            "启用成本",
            "开仓成本",
            "固定成本",
            "建设成本",
            "运营成本",
            "费用",
        },
        "region": {"region", "area", "zone", "区域", "地区", "城市"},
    },
    "customers": {
        "customer": {
            "customer",
            "customers",
            "customer_id",
            "customer_name",
            "demand_point",
            "client",
            "store",
            "name",
            "id",
            "客户",
            "客户id",
            "客户编号",
            "客户名称",
            "需求点",
            "门店",
            "门店编号",
            "名称",
            "编号",
        },
        "demand": {
            "demand",
            "qty",
            "quantity",
            "amount",
            "volume",
            "order",
            "orders",
            "需求",
            "需求量",
            "订单量",
            "销量",
            "数量",
            "量",
        },
    },
    "costs": {
        "warehouse": {
            "warehouse",
            "warehouse_id",
            "warehouse_name",
            "facility",
            "facility_id",
            "facility_name",
            "site",
            "source",
            "src",
            "from",
            "origin",
            "start",
            "仓库",
            "仓库id",
            "仓库编号",
            "仓库名称",
            "候选仓库",
            "设施",
            "设施点",
            "起点",
            "来源",
            "源点",
        },
        "customer": {
            "customer",
            "customer_id",
            "customer_name",
            "demand_point",
            "client",
            "store",
            "destination",
            "dest",
            "dst",
            "to",
            "target",
            "end",
            "客户",
            "客户id",
            "客户编号",
            "客户名称",
            "需求点",
            "门店",
            "终点",
            "目的地",
        },
        "cost": {
            "cost",
            "unit_cost",
            "transport_cost",
            "shipping_cost",
            "price",
            "rate",
            "distance",
            "fee",
            "运输成本",
            "单位成本",
            "单位运输成本",
            "配送成本",
            "距离",
            "费用",
            "价格",
        },
    },
}


ROLE_FILENAME_HINTS = {
    "warehouses": ("warehouse", "warehouses", "facility", "site", "仓库", "设施", "候选点"),
    "customers": ("customer", "customers", "client", "store", "demand", "客户", "需求", "门店"),
    "costs": ("cost", "costs", "transport", "shipping", "distance", "matrix", "成本", "运费", "距离", "矩阵"),
}


def infer_facility_table(frame: pd.DataFrame, filename: str = "") -> TableMapping:
    candidates = [_score_role(frame, filename, role) for role in FACILITY_ROLES]
    candidates.sort(key=lambda item: item.confidence, reverse=True)
    best = candidates[0]
    second = candidates[1]
    if best.confidence < 0.62:
        return TableMapping(None, best.confidence, best.column_mapping, best.method, best.warnings)
    if best.confidence < 0.78 and best.confidence - second.confidence < 0.08:
        warnings = [*best.warnings, f"表类型在 {best.role} 与 {second.role} 之间存在歧义。"]
        return TableMapping(None, best.confidence, best.column_mapping, best.method, warnings)
    return best


def assemble_facility_data(frames: list[tuple[pd.DataFrame, str, str | None]]) -> FacilityAssembly:
    if len(frames) < 3:
        return FacilityAssembly(None, {}, ["至少需要仓库、客户、成本三类数据。"])

    role_options: list[list[tuple[str, TableMapping, pd.DataFrame, str]]] = []
    for frame, filename, stored_role in frames:
        options = []
        for role in FACILITY_ROLES:
            mapping = _score_role(frame, filename, role)
            if stored_role in FACILITY_ROLES and stored_role == role:
                mapping = TableMapping(
                    role,
                    min(1.0, mapping.confidence + 0.18),
                    mapping.column_mapping,
                    f"{mapping.method}+stored_role",
                    mapping.warnings,
                )
            if mapping.confidence >= 0.48:
                options.append((role, mapping, frame, filename))
        role_options.append(options)

    best_payload = None
    best_score = -1.0
    warnings: list[str] = []
    for picked in _pick_role_combinations(role_options):
        selected = {role: (mapping, frame, filename) for role, mapping, frame, filename in picked}
        if set(selected) != set(FACILITY_ROLES):
            continue
        try:
            data = SupplyChainData(
                warehouses=apply_table_mapping(selected["warehouses"][1], selected["warehouses"][0]),
                customers=apply_table_mapping(selected["customers"][1], selected["customers"][0]),
                costs=apply_table_mapping(selected["costs"][1], selected["costs"][0]),
            )
            normalized = normalize_data(data)
            errors = validate_data(normalized)
        except Exception as exc:
            warnings.append(f"字段映射组合校验失败：{type(exc).__name__}")
            continue
        if errors:
            warnings.extend(errors[:2])
            continue
        score = sum(selected[role][0].confidence for role in FACILITY_ROLES)
        if score > best_score:
            best_score = score
            best_payload = (
                normalized,
                {selected[role][2]: selected[role][0] for role in FACILITY_ROLES},
            )

    if best_payload:
        data, mappings = best_payload
        low_conf = [name for name, mapping in mappings.items() if mapping.confidence < 0.72]
        return FacilityAssembly(
            data=data,
            mappings=mappings,
            warnings=[f"{name} 的字段映射置信度偏低，建议人工确认。" for name in low_conf],
            needs_confirmation=bool(low_conf),
            confirmation_message=_confirmation_text(mappings) if low_conf else "",
        )

    return FacilityAssembly(
        None,
        {},
        sorted(set(warnings))[:6] or ["未能从当前 CSV 自动映射出完整 warehouses/customers/costs 数据。"],
        needs_confirmation=True,
        confirmation_message="无法可靠判断上传 CSV 的表类型和字段含义，请确认仓库表、客户表、成本表及对应字段。",
    )


def apply_table_mapping(frame: pd.DataFrame, mapping: TableMapping) -> pd.DataFrame:
    renamed = frame.copy()
    rename = {
        source: target
        for source, target in mapping.column_mapping.items()
        if source in renamed.columns and target
    }
    return renamed.rename(columns=rename)


def mapping_summary(mapping: TableMapping) -> str:
    if not mapping.role:
        return "未能可靠识别表类型"
    pairs = ", ".join(f"{source}->{target}" for source, target in mapping.column_mapping.items())
    return f"{mapping.role} / {mapping.confidence:.0%} / {pairs or '无字段映射'}"


def _pick_role_combinations(options_by_file):
    indexes = range(len(options_by_file))
    for file_indexes in permutations(indexes, 3):
        buckets = []
        for wanted_role, file_index in zip(FACILITY_ROLES, file_indexes, strict=False):
            options = [item for item in options_by_file[file_index] if item[0] == wanted_role]
            if not options:
                break
            buckets.append(options)
        if len(buckets) != 3:
            continue
        for warehouses_item in buckets[0]:
            for customers_item in buckets[1]:
                for costs_item in buckets[2]:
                    yield [warehouses_item, customers_item, costs_item]


def _score_role(frame: pd.DataFrame, filename: str, role: str) -> TableMapping:
    columns = [str(column).strip() for column in frame.columns]
    lower_to_original = {column.lower().lstrip("\ufeff"): column for column in columns}
    aliases = {
        target: {name.lower() for name in names}
        for target, names in ALIASES[role].items()
    }
    mapping: dict[str, str] = {}
    method_parts: list[str] = []
    warnings: list[str] = []

    for target, names in aliases.items():
        for normalized, original in lower_to_original.items():
            if normalized in names and target not in mapping.values():
                mapping[original] = target
                method_parts.append("alias")
                break

    text_cols, numeric_cols = _profile_columns(frame)
    required = _required_columns(role)
    for target in required:
        if target in mapping.values():
            continue
        inferred = _infer_column_by_profile(target, role, columns, text_cols, numeric_cols, set(mapping))
        if inferred:
            mapping[inferred] = target
            method_parts.append("profile")

    if role == "warehouses" and "fixed_cost" not in mapping.values():
        inferred = _infer_column_by_profile("fixed_cost", role, columns, text_cols, numeric_cols, set(mapping))
        if inferred:
            mapping[inferred] = "fixed_cost"
            method_parts.append("profile")

    required_hits = sum(1 for target in required if target in mapping.values())
    alias_hits = method_parts.count("alias")
    profile_hits = method_parts.count("profile")
    filename_bonus = 0.14 if _filename_matches(filename, role) else 0.0
    shape_bonus = _shape_bonus(role, frame, text_cols, numeric_cols)
    confidence = min(
        0.98,
        required_hits / max(len(required), 1) * 0.58
        + min(alias_hits, len(required)) * 0.09
        + min(profile_hits, len(required)) * 0.06
        + filename_bonus
        + shape_bonus,
    )
    if required_hits < len(required):
        missing = sorted(set(required) - set(mapping.values()))
        warnings.append(f"缺少必要字段映射：{', '.join(missing)}")
    method = "+".join(sorted(set(method_parts))) or "unmapped"
    return TableMapping(role, confidence, mapping, method, warnings)


def _required_columns(role: str) -> tuple[str, ...]:
    return {
        "warehouses": ("warehouse", "capacity"),
        "customers": ("customer", "demand"),
        "costs": ("warehouse", "customer", "cost"),
    }[role]


def _profile_columns(frame: pd.DataFrame) -> tuple[list[str], list[str]]:
    text_cols: list[str] = []
    numeric_cols: list[str] = []
    for column in frame.columns:
        series = frame[column].dropna()
        if series.empty:
            continue
        numeric = pd.to_numeric(series, errors="coerce")
        numeric_ratio = numeric.notna().mean()
        if numeric_ratio >= 0.85:
            numeric_cols.append(str(column))
        else:
            text_cols.append(str(column))
    return text_cols, numeric_cols


def _infer_column_by_profile(
    target: str,
    role: str,
    columns: list[str],
    text_cols: list[str],
    numeric_cols: list[str],
    used: set[str],
) -> str | None:
    available_text = [column for column in text_cols if column not in used]
    available_numeric = [column for column in numeric_cols if column not in used]
    if target in {"warehouse", "customer"}:
        hints = {
            "warehouse": ("warehouse", "facility", "site", "source", "src", "from", "仓", "设施", "起点"),
            "customer": ("customer", "client", "store", "dest", "dst", "to", "需求", "客户", "门店", "终点"),
        }[target]
        hinted = _first_name_match(available_text, hints)
        if hinted:
            return hinted
        if role == "costs" and target == "customer" and len(available_text) >= 2:
            return available_text[1]
        return available_text[0] if available_text else None
    hints = {
        "capacity": ("capacity", "cap", "load", "supply", "limit", "容量", "产能", "供应", "上限"),
        "fixed_cost": ("fixed", "cost", "fee", "setup", "open", "startup", "固定", "成本", "费用", "开仓"),
        "demand": ("demand", "qty", "quantity", "amount", "volume", "需求", "数量", "订单"),
        "cost": ("cost", "price", "rate", "distance", "fee", "成本", "费用", "价格", "距离"),
    }.get(target, ())
    hinted = _first_name_match(available_numeric, hints)
    if hinted:
        return hinted
    if target == "fixed_cost" and len(available_numeric) >= 2:
        return available_numeric[1]
    return available_numeric[0] if available_numeric else None


def _first_name_match(columns: list[str], hints: tuple[str, ...]) -> str | None:
    for column in columns:
        lowered = column.lower()
        if any(hint.lower() in lowered for hint in hints):
            return column
    return None


def _filename_matches(filename: str, role: str) -> bool:
    lowered = filename.lower()
    return any(hint.lower() in lowered for hint in ROLE_FILENAME_HINTS[role])


def _shape_bonus(role: str, frame: pd.DataFrame, text_cols: list[str], numeric_cols: list[str]) -> float:
    if role == "costs" and len(text_cols) >= 2 and len(numeric_cols) >= 1:
        return 0.12
    if role == "warehouses" and len(text_cols) >= 1 and len(numeric_cols) >= 2:
        return 0.12
    if role == "customers" and len(text_cols) >= 1 and len(numeric_cols) == 1:
        return 0.10
    if role == "customers" and len(frame) > 5 and len(text_cols) >= 1 and len(numeric_cols) >= 1:
        return 0.06
    return 0.0


def _confirmation_text(mappings: dict[str, TableMapping]) -> str:
    lines = ["我推测的字段映射如下，但置信度偏低，请确认后再求解："]
    for filename, mapping in mappings.items():
        lines.append(f"- {filename}: {mapping_summary(mapping)}")
    return "\n".join(lines)
