"""为主控模型组装有预算的上下文，核心需求绝不静默截断。"""

from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any


class ContextBudgetExceeded(ValueError):
    """核心消息超出输入预算时交由运行时明确停止。"""


@dataclass(frozen=True)
class ContextPacket:
    """保存请求消息及不含密钥的预算统计。"""

    messages: list[dict[str, str]]
    stats: dict[str, Any]


def requirement_view(brief: dict[str, Any] | None) -> dict[str, Any]:
    """只取有效需求，完整数据和历史版本由本地工具按引用读取。"""

    brief = brief or {}
    contract = brief.get("dialogue_contract") or {}
    fields = ("template_id", "objective", "constraints", "assumptions", "missing_information",
              "clarification_questions", "readiness", "intent", "summary", "turn_count")
    result = {name: brief.get(name) for name in fields}
    result.update(revision=contract.get("revision"), action=contract.get("action"),
                  has_valid_data=bool(contract.get("data")),
                  pending_draft=bool(contract.get("pending_transportation")),
                  data_sources=brief.get("data_sources", []))
    return result


def build_context(
    *, system_prompt: str, question: str, task: dict[str, Any],
    actions: dict[str, str], observations: list[dict[str, Any]],
    history: list[dict[str, Any]], memories: list[dict[str, Any]],
    max_input_tokens: int = 12000,
    schema_token_reserve: int = 0,
) -> ContextPacket:
    """优先移除较旧可选信息；UTF-8 字节数用作保守的输入预算估计。"""

    # 字节预算不依赖具体模型分词器；预算仅覆盖输入，输出预算另行设置。
    if max_input_tokens < 512:
        raise ValueError("上下文输入预算至少为 512")
    core = {"current_user_message": question, "task": task, "available_actions": actions}
    optional = {"recent_turns": list(history[-6:]), "retrieved_memories": list(memories),
                "tool_observations": list(observations)}
    dropped = {name: 0 for name in optional}

    def packet() -> list[dict[str, str]]:
        return [{"role": "system", "content": system_prompt},
                {"role": "user", "content": json.dumps({**core, **optional}, ensure_ascii=False, allow_nan=False)}]

    def estimate(messages: list[dict[str, str]]) -> int:
        # 加入角色和消息封装余量，完整序列化后计算，避免只统计正文。
        return len(json.dumps(messages, ensure_ascii=False).encode("utf-8")) + 128 + schema_token_reserve

    core_messages = [{"role": "system", "content": system_prompt},
                     {"role": "user", "content": json.dumps(core, ensure_ascii=False, allow_nan=False)}]
    if estimate(core_messages) > max_input_tokens:
        raise ContextBudgetExceeded("当前消息和有效需求超出上下文预算，请缩短描述或通过文件提交数据。")
    while estimate(packet()) > max_input_tokens:
        # 最近一条工具反馈优先保留；早期反馈已在审计轨迹中持久化。
        name = next((name for name in ("recent_turns", "retrieved_memories", "tool_observations")
                     if optional[name] and (name != "tool_observations" or len(optional[name]) > 1)), None)
        if name is None:
            raise ContextBudgetExceeded("最新工具反馈超出上下文预算，请减少输入规模。")
        optional[name].pop(0)
        dropped[name] += 1
    messages = packet()
    return ContextPacket(messages=messages, stats={
        "estimator": "utf8_bytes_upper_estimate", "estimated_input_tokens": estimate(messages),
        "max_input_tokens": max_input_tokens, "dropped_items": dropped,
        "schema_token_reserve": schema_token_reserve,
        "retained_items": {name: len(value) for name, value in optional.items()},
    })
