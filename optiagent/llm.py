from __future__ import annotations

from dataclasses import dataclass

import pandas as pd
import requests

from optiagent.data import SupplyChainData


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


def profile_supply_chain_data(data: SupplyChainData, config: LLMConfig | None) -> DataProfile:
    fallback = _rule_based_profile(data)
    if not config or not config.enabled:
        return DataProfile(source="本地规则解析", summary=fallback, warnings=[], llm_used=False)

    prompt = _build_profile_prompt(data, fallback)
    try:
        content = call_openai_compatible_chat(
            config=config,
            messages=[
                {
                    "role": "system",
                    "content": "你是供应链优化数据分析助手。请用中文简洁分析数据，不要编造上传数据中不存在的字段。",
                },
                {"role": "user", "content": prompt},
            ],
        )
    except Exception as exc:
        return DataProfile(
            source="本地规则解析",
            summary=fallback,
            warnings=[f"模型服务暂时不可用，已回退到本地规则解析（{type(exc).__name__}）。"],
            llm_used=False,
        )

    return DataProfile(source=f"LLM：{config.model}", summary=content.strip(), warnings=[], llm_used=True)


def call_openai_compatible_chat(config: LLMConfig, messages: list[dict[str, str]]) -> str:
    endpoint = config.base_url.rstrip("/")
    if not endpoint.endswith("/chat/completions"):
        endpoint = f"{endpoint}/chat/completions"
    response = requests.post(
        endpoint,
        headers={
            "Authorization": f"Bearer {config.api_key}",
            "Content-Type": "application/json",
        },
        json={
            "model": config.model,
            "messages": messages,
            "temperature": config.temperature,
        },
        timeout=20,
    )
    response.raise_for_status()
    payload = response.json()
    return payload["choices"][0]["message"]["content"]


def _rule_based_profile(data: SupplyChainData) -> str:
    total_capacity = data.warehouses["capacity"].sum()
    total_demand = data.customers["demand"].sum()
    fixed_cost_total = data.warehouses["fixed_cost"].sum()
    warehouse_count = len(data.warehouses)
    customer_count = len(data.customers)
    candidate_regions = ", ".join(sorted(data.warehouses["region"].astype(str).unique()))
    most_demand = data.customers.sort_values("demand", ascending=False).head(3)
    most_demand_text = "、".join(
        f"{row.customer}({row.demand:.0f})" for row in most_demand.itertuples(index=False)
    )
    cheapest_routes = data.costs.sort_values("cost").head(5)
    route_text = "、".join(
        f"{row.warehouse}->{row.customer}({row.cost:.2f})"
        for row in cheapest_routes.itertuples(index=False)
    )
    ratio = total_demand / total_capacity if total_capacity else 0
    return (
        f"数据包含 {warehouse_count} 个候选仓库、{customer_count} 个客户点，候选区域为 {candidate_regions}。"
        f"总容量 {total_capacity:,.0f}，总需求 {total_demand:,.0f}，需求/容量比为 {ratio:.1%}。"
        f"候选仓库固定成本合计 {fixed_cost_total:,.0f}。需求最高的客户为 {most_demand_text}。"
        f"最低成本线路包括 {route_text}。"
    )


def _build_profile_prompt(data: SupplyChainData, fallback: str) -> str:
    return (
        "请先阅读以下供应链数据解析和表格样本，然后输出 5-8 条中文要点，"
        "说明数据规模、容量是否充足、需求集中点、固定成本特点、潜在风险，以及适合用什么优化模型。\n\n"
        f"本地规则摘要：{fallback}\n\n"
        f"仓库表：\n{_sample_table(data.warehouses)}\n\n"
        f"客户表：\n{_sample_table(data.customers)}\n\n"
        f"运输成本表样本：\n{_sample_table(data.costs, rows=12)}"
    )


def _sample_table(frame: pd.DataFrame, rows: int = 8) -> str:
    return frame.head(rows).to_csv(index=False)
