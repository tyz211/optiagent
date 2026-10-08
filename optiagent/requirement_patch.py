"""核对 LLM 提议的数值修改，原文覆盖、实体和数值均通过后才生成新版本。"""

from __future__ import annotations

from copy import deepcopy
import re
from typing import Literal

from pydantic import BaseModel, ConfigDict, Field

from optiagent.dialogue_contract import apply_contract
from optiagent.mcp_validation import validate_problem_data
from optiagent.requirement_analysis import RequirementBrief


class RequirementEdit(BaseModel):
    """字段来自封闭能力表；原文位置以 Python 字符索引计，不允许补造引用。"""

    model_config = ConfigDict(extra="forbid")
    field: Literal["capacity", "item_value", "item_weight", "supply", "demand", "route_cost"]
    entity: str = Field(max_length=80)
    target: str = Field(max_length=80)
    value: float = Field(strict=True, allow_inf_nan=False, ge=0)
    source_quote: str = Field(min_length=1, max_length=500)
    source_start: int = Field(strict=True, ge=0)
    source_end: int = Field(strict=True, ge=1)


class RequirementPatch(BaseModel):
    """一批修改共享基准版本与执行意图，任何一项失败整批不生效。"""

    model_config = ConfigDict(extra="forbid")
    expected_revision: int = Field(strict=True, ge=1)
    template_id: Literal["knapsack", "transportation"]
    mode: Literal["solve", "hold"]
    edits: list[RequirementEdit] = Field(min_length=1, max_length=8)


NUMBER = r"[+-]?(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?"
EDIT_PATTERN = re.compile(
    rf"(?:请)?(?:把|将)?(?P<label>.+?)(?:从\s*(?P<old>{NUMBER})\s*)?"
    rf"(?P<verb>改成|改为|调整为|设为|设置为|调整到|修改为|降低到|降低为|提高到|提高为|增至|降至)"
    rf"\s*(?P<new>{NUMBER})\s*(?P<unit>[^\d\s]*)"
)
HOLD_COMMANDS = {"先不求解", "暂不求解", "只修改", "先不算", "只分析"}
SOLVE_COMMANDS = {"求解", "并求解", "然后求解", "重新求解"}
SEPARATORS = " \t\r\n，,；;。.!！"


def editable_targets(brief: dict, query: str = "") -> dict:
    """按实体查询返回最多二十个目标，完整数据继续保留在本地。"""

    contract = brief.get("dialogue_contract") or {}
    data = contract.get("data") or {}
    rows = []
    if contract.get("template_id") == "knapsack":
        rows.append({"field": "capacity", "entity": "", "target": "", "value": data.get("capacity"), "unit": ""})
        for item in data.get("items", []):
            for field, key in (("item_value", "value"), ("item_weight", "weight")):
                rows.append({"field": field, "entity": item["item"], "target": "", "value": item[key], "unit": ""})
    elif contract.get("template_id") == "transportation":
        for collection, field, key in (("suppliers", "supply", "supply"), ("consumers", "demand", "demand")):
            for item in data.get(collection, []):
                rows.append({"field": field, "entity": item["name"], "target": "", "value": item[key], "unit": data["quantity_unit"]})
        for item in data.get("routes", []):
            rows.append({"field": "route_cost", "entity": item["source"], "target": item["target"],
                         "value": item["cost"], "unit": data["currency"] + "/" + data["quantity_unit"]})
    if query:
        rows = [row for row in rows if query in row["entity"] or query in row["target"] or query == row["field"]]
    return {"revision": contract.get("revision"), "template_id": contract.get("template_id"),
            "targets": rows[:20], "remaining_count": max(0, len(rows) - 20)}


def _target(data: dict, template: str, edit: RequirementEdit) -> tuple[dict, str, set[str], str]:
    """把业务名称解析成已存在字段，禁止新增实体、线路或隐式单位换算。"""

    if template == "knapsack" and edit.field == "capacity" and not edit.entity and not edit.target:
        return data, "capacity", {"容量", "背包容量", "背包的容量"}, ""
    if template == "knapsack" and edit.field in {"item_value", "item_weight"} and not edit.target:
        matches = [row for row in data["items"] if row["item"] == edit.entity]
        key = "value" if edit.field == "item_value" else "weight"
        names = {"价值", "收益"} if key == "value" else {"重量"}
        labels = {prefix + edit.entity + "的" + name for prefix in ("", "物品") for name in names}
        if len(matches) == 1:
            return matches[0], key, labels, ""
    if template == "transportation" and edit.field in {"supply", "demand"} and not edit.target:
        supply = edit.field == "supply"
        matches = [row for row in data["suppliers" if supply else "consumers"] if row["name"] == edit.entity]
        names = {"供给量", "供应量", "供应上限"} if supply else {"需求量", "需求"}
        labels = {prefix + edit.entity + "的" + name for prefix in ("", "供应点" if supply else "需求点") for name in names}
        if len(matches) == 1:
            return matches[0], edit.field, labels, data["quantity_unit"]
    if template == "transportation" and edit.field == "route_cost":
        matches = [row for row in data["routes"] if row["source"] == edit.entity and row["target"] == edit.target]
        labels = {prefix + edit.entity + "到" + edit.target + "的" + name
                  for prefix in ("", "从") for name in ("单位运费", "单位运输成本")}
        if len(matches) == 1:
            return matches[0], "cost", labels, data["currency"] + "/" + data["quantity_unit"]
    raise ValueError("修改目标不存在或不属于当前模板的可编辑字段")


def build_patched_requirement(previous: dict, question: str, patch: RequirementPatch) -> dict:
    """纯函数验证并生成完整候选；调用方用完整旧状态做原子比较提交。"""

    contract = deepcopy(previous.get("dialogue_contract") or {})
    if patch.expected_revision != contract.get("revision") or patch.template_id != contract.get("template_id"):
        raise ValueError("基准版本或模板已变化，请重新读取需求")
    if not contract.get("data"):
        raise ValueError("没有可修改的有效结构化数据")
    if contract.get("error") or contract.get("pending_transportation"):
        raise ValueError("已有待澄清条件或未确认草稿，请先处理后再修改")
    if len(contract.get("versions", [])) >= 100:
        raise ValueError("当前会话已达到 100 个需求版本，请新建对话继续")
    data = contract["data"]
    edits = sorted(patch.edits, key=lambda item: item.source_start)
    end = 0
    gaps = []
    touched = set()
    audit = []
    for edit in edits:
        if edit.source_start < end or edit.source_end > len(question) or question[edit.source_start:edit.source_end] != edit.source_quote:
            raise ValueError("修改依据位置重叠或与本轮原文不一致")
        gaps.append(question[end:edit.source_start])
        end = edit.source_end
        target, key, labels, unit = _target(data, patch.template_id, edit)
        identity = (edit.field, edit.entity, edit.target)
        if identity in touched:
            raise ValueError("同一字段不能在一批修改中重复赋值")
        touched.add(identity)
        match = EDIT_PATTERN.fullmatch(edit.source_quote.strip(SEPARATORS))
        if match is None or match["label"].strip() not in labels:
            raise ValueError("原文必须明确指定已存在实体的字段和替换数值")
        if float(match["new"]) != edit.value or (match["unit"] and match["unit"] != unit):
            raise ValueError("修改数值或单位与原文不一致，不执行隐式单位换算")
        before = target[key]
        if match["old"] is not None and float(match["old"]) != before:
            raise ValueError("原文中的旧数值与有效版本不一致")
        if edit.value == before:
            raise ValueError("数值未变化，无需生成重复版本")
        if match["verb"] in {"降低到", "降低为", "降至"} and edit.value >= before:
            raise ValueError("降低指令与数值方向矛盾")
        if match["verb"] in {"提高到", "提高为", "增至"} and edit.value <= before:
            raise ValueError("提高指令与数值方向矛盾")
        target[key] = edit.value
        audit.append({**edit.model_dump(), "before": before})
    gaps.append(question[end:])
    commands = [part.strip(SEPARATORS) for gap in gaps for part in re.split(r"[，,；;。.!！\n]", gap) if part.strip(SEPARATORS)]
    if any(part not in HOLD_COMMANDS | SOLVE_COMMANDS for part in commands):
        raise ValueError("本轮还有未覆盖条件，不能只保存部分修改")
    hold = any(part in HOLD_COMMANDS for part in commands)
    if hold and any(part in SOLVE_COMMANDS for part in commands):
        raise ValueError("本轮同时要求求解与暂停，请明确执行意图")
    if (patch.mode == "hold") != hold:
        raise ValueError("执行意图与本轮原文不一致")
    report = validate_problem_data(patch.template_id, data)
    if not report.valid:
        raise ValueError("修改后的完整数据未通过模板校验")
    # 沿用现有不可变版本结构，证据连同批次一起提交。
    revision = contract["revision"] + 1
    change = {"operation": "grounded_patch", "turn": previous.get("turn_count", 1) + 1,
              "before_revision": contract["revision"], "after_revision": revision, "edits": audit}
    versions = contract.get("versions", [])
    versions.append({"revision": revision, "parent_revision": contract["revision"],
                     "template_id": patch.template_id, "data": deepcopy(data), "change": change})
    contract.update(revision=revision, versions=versions, changes=[change], action=patch.mode)
    for name in ("error", "parser_attempt", "pending_transportation", "pending_source_quotes"):
        contract.pop(name, None)
    brief = RequirementBrief.model_validate(previous)
    brief.turn_count += 1
    brief.source = "llm"
    return apply_contract(brief, contract).model_dump(mode="json")
