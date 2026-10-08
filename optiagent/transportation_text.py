"""运输需求的受限中文解析：逐句消费，未识别内容必须澄清。"""

from __future__ import annotations

from copy import deepcopy
import json
import re

from optiagent.transportation import TransportationData


# 名称和数值采用完整匹配，避免只读出句首后忽略剩余业务限制。
NAME = r"[A-Za-z0-9_\u4e00-\u9fff-]+?"
NUMBER = r"\d+(?:\.\d+)?"
UNIT = r"吨|千克|公斤|件|箱"
CURRENCY = r"元|美元"
CONTROL = r"(?:请)?(?:求解|进行求解|计算|继续求解|重新求解|确认并求解|运输分配|运输分配问题|运输问题|最小化运输成本|最小化运输总成本|数据|替换数据|先不求解|暂不求解|只分析|先分析)"


def looks_like_transportation(text: str) -> bool:
    """只触发运输专用入口，不把泛泛的仓库客户描述误判成运输数据。"""

    return bool(re.search(r"运输分配|运输问题|供给点|需求点|单位运费", text))


def _clauses(text: str) -> list[str]:
    """保留原句内容用于审计；冒号仅作为标题分隔。"""

    return [part.strip() for part in re.split(r"[；;，,。\n：:]+", text) if part.strip()]


def _control_only(text: str) -> list[dict]:
    """JSON 以外只允许已知控制语句，额外自然语言约束不能被静默忽略。"""

    evidence = []
    for clause in _clauses(text):
        if not re.fullmatch(CONTROL, clause):
            raise ValueError(f"尚不能完整执行这段描述：{clause}。请将条件写入运输数据，或按支持的格式重新描述。")
        evidence.append({"text": clause, "status": "control"})
    return evidence


def parse_transportation_request(text: str) -> tuple[dict, list[dict]]:
    """支持严格 JSON 与带单位的中文声明，输出可执行数据及逐句来源。"""

    cleaned = text.strip()
    if "{" in cleaned:
        start = cleaned.index("{")
        try:
            payload, end = json.JSONDecoder(object_pairs_hook=_unique_keys).raw_decode(cleaned[start:])
        except json.JSONDecodeError as exc:
            raise ValueError("运输数据必须是完整 JSON 对象") from exc
        # 只去掉 JSON 对象外侧的代码围栏，不能改写字符串里的节点名称。
        prefix = re.sub(r"```(?:json)?\s*$", "", cleaned[:start])
        suffix = re.sub(r"^\s*```", "", cleaned[start + end:])
        surrounding = prefix + "；" + suffix
        evidence = _control_only(surrounding)
        if "transportation" in payload:
            if set(payload) != {"transportation"}:
                raise ValueError("transportation 外层不能混入未处理字段")
            payload = payload["transportation"]
        problem = TransportationData.model_validate(payload)
        return problem.model_dump(), [*evidence, {"text": cleaned[start:start + end], "status": "validated_data"}]

    suppliers, consumers, routes, forbidden = [], [], [], []
    units, currencies, evidence = set(), set(), []
    for clause in _clauses(cleaned):
        supply = re.fullmatch(rf"供给点\s*({NAME})\s*供给\s*({NUMBER})\s*({UNIT})", clause)
        demand = re.fullmatch(rf"需求点\s*({NAME})\s*需求\s*({NUMBER})\s*({UNIT})", clause)
        cost = re.fullmatch(rf"({NAME})\s*(?:到|→|->)\s*({NAME})\s*单位运费\s*({NUMBER})\s*({CURRENCY})\s*/\s*({UNIT})", clause)
        blocked = re.fullmatch(rf"禁止\s*({NAME})\s*(?:到|→|->)\s*({NAME})\s*运输", clause)
        if supply or demand:
            match = supply or demand
            records = suppliers if supply else consumers
            records.append({"name": match[1], "supply" if supply else "demand": float(match[2])})
            units.add(match[3])
            target = f"{'suppliers' if supply else 'consumers'}/{match[1]}"
        elif cost:
            routes.append({"source": cost[1], "target": cost[2], "cost": float(cost[3])})
            currencies.add(cost[4])
            units.add(cost[5])
            target = f"routes/{cost[1]}/{cost[2]}"
        elif blocked:
            forbidden.append({"source": blocked[1], "target": blocked[2]})
            target = f"forbidden_routes/{blocked[1]}/{blocked[2]}"
        elif re.fullmatch(CONTROL, clause):
            evidence.append({"text": clause, "status": "control"})
            continue
        else:
            raise ValueError(f"尚不能完整解析：{clause}。可用“供给点 A 供给 20 吨”“需求点 X 需求 15 吨”“A 到 X 单位运费 2 元/吨”。")
        evidence.append({"text": clause, "field": target, "status": "compiled"})
    if len(units) != 1 or len(currencies) != 1:
        raise ValueError("请明确统一的数量单位及单位运费币种；首版不自动换算混合单位，可改用带 quantity_unit/currency 的 JSON")
    problem = TransportationData.model_validate({
        "suppliers": suppliers, "consumers": consumers, "routes": routes,
        "forbidden_routes": forbidden, "quantity_unit": units.pop(), "currency": currencies.pop(),
    })
    return problem.model_dump(), evidence


def _unique_keys(pairs: list[tuple]) -> dict:
    """拒绝重复 JSON 键，避免后一个值悄悄覆盖用户的另一个声明。"""
    result = {}
    for key, value in pairs:
        if key in result:
            raise ValueError(f"重复字段：{key}")
        result[key] = value
    return result


def edit_transportation(text: str, data: dict) -> tuple[dict, list[dict]]:
    """多项变更全部校验后再交给对话合同提交，失败不污染旧版本。"""

    candidate, changes, touched = deepcopy(data), [], set()
    for clause in _clauses(text):
        if re.fullmatch(CONTROL, clause):
            continue
        amount = re.fullmatch(rf"(?:把|将)?\s*(供给点|需求点)\s*({NAME})\s*(供给|需求)\s*(?:改成|改为|调整为)\s*({NUMBER})\s*({UNIT})", clause)
        cost = re.fullmatch(rf"(?:把|将)?\s*({NAME})\s*(?:到|→|->)\s*({NAME})\s*单位运费\s*(?:改成|改为|调整为)\s*({NUMBER})\s*({CURRENCY})\s*/\s*({UNIT})", clause)
        blocked = re.fullmatch(rf"禁止\s*({NAME})\s*(?:到|→|->)\s*({NAME})\s*运输", clause)
        if amount:
            table, field = ("suppliers", "supply") if amount[1] == "供给点" else ("consumers", "demand")
            if amount[3] != ("供给" if table == "suppliers" else "需求") or amount[5] != data["quantity_unit"]:
                raise ValueError("修改的字段或单位与当前有效数据不一致")
            row = next((r for r in candidate[table] if r["name"] == amount[2]), None)
            if row is None:
                raise ValueError(f"未找到节点：{amount[2]}")
            path, before = f"{table}/{amount[2]}/{field}", row[field]
            row[field] = float(amount[4])
            after = row[field]
        elif cost or blocked:
            match = cost or blocked
            pair = (match[1], match[2])
            row = next((r for r in candidate["routes"] if (r["source"], r["target"]) == pair), None)
            if row is None:
                raise ValueError("未找到可修改线路；恢复禁运线路请提交完整数据并提供运费")
            path, before = f"routes/{pair[0]}/{pair[1]}", deepcopy(row)
            if cost:
                if cost[4] != data["currency"] or cost[5] != data["quantity_unit"]:
                    raise ValueError("运费单位与当前有效数据不一致")
                row["cost"] = float(cost[3])
                after = deepcopy(row)
            else:
                candidate["routes"].remove(row)
                after = {"source": pair[0], "target": pair[1]}
                candidate["forbidden_routes"].append(after)
        else:
            raise ValueError(f"这条运输修改尚不能完整执行：{clause}")
        if path in touched:
            raise ValueError("同一字段在本轮被重复修改，请明确最终值")
        touched.add(path)
        changes.append({"text": clause, "field": path, "before": before, "after": after, "status": "compiled"})
    if not changes:
        raise ValueError("没有识别到运输数据修改")
    return TransportationData.model_validate(candidate).model_dump(), changes
