from __future__ import annotations

from collections.abc import Mapping
from dataclasses import dataclass
import json
import math
import re
from typing import Any

import requests


@dataclass(frozen=True)
class LLMConfig:
    enabled: bool
    api_key: str
    base_url: str
    model: str
    temperature: float = 0.2


@dataclass(frozen=True)
class DataProfile:
    source: str
    summary: str
    warnings: list[str]
    llm_used: bool


def llm_config_from_record(record: Mapping[str, Any] | None) -> LLMConfig | None:
    """将数据库或配置文件记录统一转换为运行时 LLM 配置。"""

    if not record:
        return None
    return LLMConfig(
        enabled=True,
        api_key=str(record["api_key"]),
        base_url=str(record["base_url"]),
        model=str(record["model"]),
        temperature=float(record.get("temperature", 0.2)),
    )


def parse_json_object(text: str, *, error_message: str = "LLM 未返回 JSON 对象。") -> dict[str, Any]:
    """解析 LLM 返回的纯 JSON 或 Markdown JSON 代码块。"""

    cleaned = re.sub(r"^```(?:json)?\s*|\s*```$", "", text.strip(), flags=re.IGNORECASE)
    try:
        value = json.loads(cleaned)
    except json.JSONDecodeError:
        match = re.search(r"\{.*\}", cleaned, flags=re.DOTALL)
        if not match:
            raise ValueError(error_message)
        value = json.loads(match.group(0))
    if not isinstance(value, dict):
        raise ValueError(error_message)
    return value


def clamp_probability(value: Any, *, default: float = 0.0) -> float:
    """把外部返回的置信度安全转换为 0 到 1 的有限数值。"""

    try:
        number = float(value)
    except (TypeError, ValueError):
        return default
    if math.isnan(number) or math.isinf(number):
        return default
    return max(0.0, min(number, 1.0))


def call_openai_compatible_chat(
    config: LLMConfig, messages: list[dict[str, str]], *,
    json_schema: dict[str, Any] | None = None, timeout: float = 20,
    max_tokens: int | None = None,
    strict_schema: bool = False,
) -> str:
    """兼容旧聊天调用；结构化任务显式提交 Schema 和生成预算，不静默降级。"""
    endpoint = config.base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"
    body: dict[str, Any] = {"model": config.model, "messages": messages, "temperature": config.temperature}
    if json_schema is not None:
        body["response_format"] = {"type": "json_schema", "json_schema": {
            "name": "optiagent_requirement", "schema": json_schema,
        }}
        if strict_schema:
            # 主控行动使用封闭 Schema；旧模型草稿接口保持原有兼容行为。
            body["response_format"]["json_schema"]["strict"] = True
    if max_tokens is not None:
        body["max_tokens"] = max_tokens
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json=body,
        timeout=timeout,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]
