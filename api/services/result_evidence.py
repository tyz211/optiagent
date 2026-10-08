"""从当前作用域已接受的求解记录提取事实，解释文本仅由事实引用生成。"""

from __future__ import annotations

import hashlib
import json
import math
import re

from api.database import get_scoped_run, list_runs


def explanation_request(question: str) -> tuple[bool, int | None]:
    """首版仅接受完整的结果解释指令，附带修改或额外条件时交回需求工具。"""

    match = re.fullmatch(
        r"\s*(?:请)?(?:解释|说明|展示|查看)(?:一下)?\s*(?:上一轮|最新|这个|当前|已验算)?"
        r"(?:结果|方案)(?:\s*#\s*(?P<run>\d+))?(?:的(?:目标值|选择情况|验算依据|容量使用))?\s*[。？！!?]?\s*",
        question,
    )
    return (True, int(match["run"]) if match["run"] else None) if match else (False, None)


def _accepted(row: dict) -> dict | None:
    """同时核对数据库状态、数学验算与工作流接受状态，不借用单个成功标志。"""

    try:
        result = json.loads(row["result_json"])
    except (ValueError, TypeError):
        return None
    if not isinstance(result, dict) or row["status"] not in {"OPTIMAL", "FEASIBLE", "NEAR_OPTIMAL"}:
        return None
    if result.get("status") != row["status"]:
        return None
    objective = result.get("objective_value")
    if isinstance(objective, bool) or not isinstance(objective, (int, float)) or not math.isfinite(objective) or objective != row["objective_value"]:
        return None
    verification = result.get("solution_verification") or {}
    if verification.get("verifiable") is not True or verification.get("passed") is not True:
        return None
    if (result.get("workflow_verification") or {}).get("passed") is not True:
        return None
    return result


def read_result_evidence(*, user_id: int | None, conversation_id: int | None,
                         brief: dict, run_id: int | None, query: str = "") -> dict:
    """默认选择与有效输入相同的最近已验算记录；显式历史引用保留其原版本。"""

    if conversation_id is None:
        raise ValueError("请在有效会话中读取已验算方案")
    current = brief.get("dialogue_contract") or {}
    if run_id is not None:
        row = get_scoped_run(run_id, user_id=user_id, conversation_id=conversation_id)
        candidates = [row] if row else []
    else:
        candidates = reversed(list_runs(limit=100, user_id=user_id, conversation_id=conversation_id))
    chosen = None
    for row in candidates:
        result = _accepted(row)
        if result is None:
            continue
        contract = (result.get("requirement_analysis") or {}).get("dialogue_contract") or {}
        same = bool(current.get("data")) and all(contract.get(key) == current.get(key) for key in ("revision", "template_id", "data"))
        if run_id is None and not same:
            continue
        chosen = row, result, contract, same
        break
    if chosen is None:
        raise ValueError("未找到当前作用域内符合版本要求且已通过验算的方案")
    row, result, contract, same = chosen
    facts = []

    def add(identity: str, text: str, path: str, value) -> None:
        # 字段引用均指向源运行；不采用历史回答或模型消息作为事实。
        facts.append({"id": identity, "text": text, "source_path": path, "value": value})

    add("objective", f"目标值为 {result.get('objective_value')}。", "/objective_value", result.get("objective_value"))
    add("status", f"求解器状态为 {result['status']}。", "/status", result["status"])
    add("verification", "该方案通过独立验算和工作流审核；验算覆盖范围以源报告为准，不代表经统计校准的正确概率。",
        "/solution_verification", result["solution_verification"])
    generic = result.get("generic_result") or {}
    labels = {"capacity": "容量上限", "used_weight": "已用重量", "remaining_capacity": "剩余容量",
              "selected_value": "已选总价值", "selected_count": "已选物品数", "total_supply": "总供给量",
              "total_demand": "总需求量", "mip_gap": "求解器报告的最优间隙"}
    for key, label in labels.items():
        if generic.get("metrics", {}).get(key) is not None:
            value = generic["metrics"][key]
            add("metric:" + key, f"{label}为 {value}。", "/generic_result/metrics/" + key, value)
    for index, item in enumerate(generic.get("decisions", [])):
        if query and query not in json.dumps(item, ensure_ascii=False):
            continue
        if generic.get("template_id") == "knapsack":
            text = f"物品 {item['item']}：{'已选择' if item['selected'] else '未选择'}，重量 {item['weight']}，价值 {item['value']}。"
        elif generic.get("template_id") == "transportation":
            text = "运输决策：" + json.dumps(item, ensure_ascii=False, allow_nan=False) + "。"
        else:
            text = "决策记录：" + json.dumps(item, ensure_ascii=False, allow_nan=False) + "。"
        add("decision:" + str(index), text, "/generic_result/decisions/" + str(index), item)
    # 固定数量上限，规模较大的结果由 query 按实体过滤后读取。
    return {"run_id": row["id"], "requirement_revision": contract.get("revision"),
            "template_id": contract.get("template_id"), "matches_current_requirement": same,
            "fingerprint": hashlib.sha256(row["result_json"].encode("utf-8")).hexdigest(),
            "facts": facts[:24], "remaining_count": max(0, len(facts) - 24)}


def render_result_explanation(evidence: dict, fact_ids: list[str], *, question: str = "") -> tuple[str, list[dict]]:
    """LLM 选择相关事实，程序生成数值与引用，禁止自由文本新增事实或因果断言。"""

    lookup = {fact["id"]: fact for fact in evidence["facts"]}
    if not fact_ids or len(fact_ids) > 12 or len(set(fact_ids)) != len(fact_ids):
        raise ValueError("请选择一到十二个互不重复的事实引用")
    if any(identity not in lookup for identity in fact_ids):
        raise ValueError("解释引用了未读取或不存在的事实")
    required = set()
    if "的目标值" in question:
        required.add("objective")
    if "的验算依据" in question:
        required.add("verification")
    if "的容量使用" in question:
        required.update({"metric:capacity", "metric:used_weight"})
    if not required.issubset(fact_ids) or ("的选择情况" in question and not any(identity.startswith("decision:") for identity in fact_ids)):
        raise ValueError("所选事实尚未覆盖本轮指定的解释内容")
    facts = [lookup[identity] for identity in fact_ids]
    reference = f"方案 #{evidence['run_id']}，需求版本 {evidence['requirement_revision']}"
    note = "" if evidence["matches_current_requirement"] else "（历史方案，与当前有效输入不同）"
    answer = reference + note + "：\n" + "\n".join(f"- {fact['text']}" for fact in facts)
    if evidence.get("remaining_count"):
        answer += f"\n源方案还有 {evidence['remaining_count']} 条事实未读取，可按实体查询。"
    citations = [{"run_id": evidence["run_id"], "requirement_revision": evidence["requirement_revision"],
                  "fact_id": fact["id"], "source_path": fact["source_path"]} for fact in facts]
    return answer, citations
