from __future__ import annotations

import asyncio
from dataclasses import dataclass
import json
from typing import Any

import pandas as pd
from langchain.agents import create_agent
from langchain_core.tools import tool
from langchain_openai import ChatOpenAI

from optiagent.data import SupplyChainData
from optiagent.generic_solvers import solve_by_problem_spec
from optiagent.llm import LLMConfig
from optiagent.problem_spec import infer_problem_spec, spec_summary
from optiagent.rag import rag_context_pack, rag_summary
from optiagent.scenario import apply_what_if, explain_result
from optiagent.solver import solve_facility_location
from optiagent.web_research import web_search


@dataclass(frozen=True)
class AgentRunResult:
    answer: str
    tool_names: list[str]
    mcp_status: str
    raw: dict[str, Any] | None


def run_configured_multi_agent(
    query: str,
    data: SupplyChainData,
    config: LLMConfig | None,
    mcp_config_json: str = "",
) -> AgentRunResult:
    tools = build_builtin_tools(data)
    mcp_tools, mcp_status = _load_mcp_tools_sync(mcp_config_json)
    tools.extend(mcp_tools)

    if not config or not config.enabled:
        return AgentRunResult(
            answer=_local_router_answer(query, data),
            tool_names=[item.name for item in tools],
            mcp_status=mcp_status,
            raw=None,
        )

    llm = ChatOpenAI(
        model=config.model,
        api_key=config.api_key,
        base_url=config.base_url,
        temperature=config.temperature,
    )
    system_prompt = (
        "你是通用运筹优化多智能体系统的 Supervisor。回答必须使用中文，"
        "先给业务结论，再简要说明调用了哪些工具和依据来源。\n"
        "核心流程：面对任何优化问题，优先调用 problem_spec_tool 生成 ProblemSpec；"
        "涉及建模方法、数据字段、代码模板或求解器选择时调用 rag_context_pack_tool；"
        "ProblemSpec 为 knapsack、assignment、tsp、job_shop_scheduling 或 production_mix 时调用 generic_optimizer_tool；"
        "仓库选址、启用仓库、成本、需求变化或关闭仓库等结构化供应链问题调用 gurobi_facility_location_tool。\n"
        "真实数据规则：当用户要求真实、当前、公开网页、市场、城市、物流、仓储、地理位置等外部事实，"
        "或仓库/门店/设施选址缺少事实支撑时，必须调用 web_search_tool；涉及中国城市人口、GDP、经纬度时可同时调用 city_reference_tool。"
        "web_search_tool 只提供来源证据，不能替代优化数据表。\n"
        "实体约束：优先使用用户上传数据，其次使用本地城市参考数据和带 URL 的网页证据。"
        "不得无依据新增候选仓库、客户、需求、容量、固定成本、运输成本或距离矩阵。"
        "如果缺少完整优化参数，只能列出数据缺口、引用已找到来源，并请用户补充，不能伪造可求解模型。"
        "所有来自网页的信息必须在回答中附来源标题或 URL。"
        "如果问题涉及规则、模型、Gurobi、OR-Tools、RAG 或多 Agent，调用 rag_search_tool。"
    )
    agent = create_agent(model=llm, tools=tools, system_prompt=system_prompt)
    try:
        raw = agent.invoke({"messages": [{"role": "user", "content": query}]})
        answer = _extract_answer(raw)
        return AgentRunResult(
            answer=answer,
            tool_names=_extract_called_tool_names(raw) or [item.name for item in tools],
            mcp_status=mcp_status,
            raw=_jsonable(raw),
        )
    except Exception as exc:
        return AgentRunResult(
            answer=(
                "模型服务暂时不可用，已自动回退到本地工具回答。\n\n"
                f"{_local_router_answer(query, data)}"
            ),
            tool_names=[item.name for item in tools],
            mcp_status=mcp_status,
            raw={"error": str(exc)},
        )


def build_builtin_tools(data: SupplyChainData):
    @tool
    def problem_spec_tool(question: str) -> str:
        """把自然语言优化问题解析为通用 ProblemSpec，用于选择模板、数据要求和求解器。"""
        spec = infer_problem_spec(question, data)
        payload = spec.to_dict()
        payload["summary"] = spec_summary(spec)
        return json.dumps(payload, ensure_ascii=False)

    @tool
    def rag_context_pack_tool(question: str) -> str:
        """按建模知识、数据 Schema、代码模板和求解策略检索 RAG 上下文包。"""
        return json.dumps(rag_context_pack(question), ensure_ascii=False)

    @tool
    def generic_optimizer_tool(question: str) -> str:
        """对已支持的通用优化模板进行实际求解，当前支持背包、指派、TSP、作业车间调度和产品组合/MILP。"""
        spec = infer_problem_spec(question, data)
        result = solve_by_problem_spec(question, spec)
        if result is None:
            return json.dumps(
                {
                    "status": "UNSUPPORTED",
                    "message": f"当前模板 {spec.template_id} 由专用链路处理，或暂未暴露为通用求解器。",
                    "problem_spec": spec.to_dict(),
                },
                ensure_ascii=False,
            )
        return json.dumps({"problem_spec": spec.to_dict(), "result": result.to_dict()}, ensure_ascii=False)

    @tool
    def rag_search_tool(question: str) -> str:
        """检索本地 RAG 知识库，回答模型规则、求解器选择、RAG 和多 Agent 相关问题。"""
        notes, docs = rag_summary(question, top_k=4)
        return json.dumps(
            {
                "notes": notes,
                "documents": [
                    {"title": doc.title, "score": round(doc.score, 4), "content": doc.content}
                    for doc in docs
                ],
            },
            ensure_ascii=False,
        )

    @tool
    def web_search_tool(query: str) -> str:
        """搜索公开网页以获取真实数据来源。只返回来源证据，不自动生成优化实体或参数。"""
        try:
            results = web_search(query, max_results=5)
        except Exception as exc:
            return json.dumps(
                {
                    "status": "ERROR",
                    "message": f"web 搜索失败：{type(exc).__name__}",
                    "policy": "搜索失败时不得编造外部数据或优化实体。",
                },
                ensure_ascii=False,
            )
        return json.dumps(
            {
                "status": "OK",
                "policy": "仅作为外部证据；不得无依据新增仓库、客户、容量、需求、成本或距离参数。",
                "results": [item.to_dict() for item in results],
            },
            ensure_ascii=False,
        )

    @tool
    def data_profile_tool(_: str = "") -> str:
        """总结当前供应链数据的规模、容量、需求和低成本路线。"""
        total_capacity = float(data.warehouses["capacity"].sum())
        total_demand = float(data.customers["demand"].sum())
        top_customers = data.customers.sort_values("demand", ascending=False).head(5)
        cheapest_routes = data.costs.sort_values("cost").head(8)
        payload = {
            "warehouse_count": len(data.warehouses),
            "customer_count": len(data.customers),
            "total_capacity": total_capacity,
            "total_demand": total_demand,
            "demand_capacity_ratio": total_demand / total_capacity if total_capacity else None,
            "top_customers": top_customers.to_dict(orient="records"),
            "cheapest_routes": cheapest_routes.to_dict(orient="records"),
        }
        return json.dumps(payload, ensure_ascii=False)

    @tool
    def gurobi_facility_location_tool(question: str) -> str:
        """根据中文 what-if 问题修改场景，并调用 Gurobi 求解仓库选址 MILP。"""
        baseline = solve_facility_location(data)
        scenario = apply_what_if(question, data)
        result = solve_facility_location(scenario.data)
        open_warehouses = []
        warehouse_summary = []
        if not result.warehouse_summary.empty:
            open_warehouses = result.warehouse_summary.loc[
                result.warehouse_summary["is_open"] == 1,
                "warehouse",
            ].tolist()
            warehouse_summary = result.warehouse_summary.to_dict(orient="records")
        payload = {
            "scenario_changes": scenario.changes,
            "warnings": scenario.warnings,
            "status": result.status,
            "objective_value": result.objective_value,
            "transport_cost": result.transport_cost,
            "fixed_cost": result.fixed_cost,
            "baseline_objective": baseline.objective_value,
            "open_warehouses": open_warehouses,
            "explanation": explain_result(baseline, result, scenario.changes),
            "warehouse_summary": warehouse_summary,
        }
        return json.dumps(payload, ensure_ascii=False, default=str)

    @tool
    def city_reference_tool(question: str) -> str:
        """查询本地公开中国城市参考数据，支持城市名、人口、GDP、经纬度等问题。"""
        try:
            cities = pd.read_csv("data/china_city_reference.csv")
        except FileNotFoundError:
            return "未找到 data/china_city_reference.csv。"

        lowered = question.lower()
        mask = pd.Series(False, index=cities.index)
        for column in ["cityName_CN", "cityName_EN"]:
            mask = mask | cities[column].astype(str).str.lower().apply(lambda item: item in lowered)
        matched = cities[mask]
        if matched.empty:
            matched = cities.sort_values("GDP", ascending=False).head(10)
        return matched.head(20).to_json(orient="records", force_ascii=False)

    return [
        problem_spec_tool,
        rag_context_pack_tool,
        generic_optimizer_tool,
        rag_search_tool,
        web_search_tool,
        data_profile_tool,
        gurobi_facility_location_tool,
        city_reference_tool,
    ]


def _local_router_answer(query: str, data: SupplyChainData) -> str:
    notes, docs = rag_summary(query, top_k=3)
    spec = infer_problem_spec(query, data)
    generic_result = solve_by_problem_spec(query, spec)
    if generic_result:
        return "\n".join(
            [
                "未配置 LLM，已使用本地通用优化工具给出回答。",
                f"ProblemSpec：{spec.display_name} / {spec.problem_type}；推荐求解器：{spec.recommended_solver}。",
                f"求解状态：{generic_result.status}；{generic_result.summary}",
                f"数据来源：{generic_result.data_source}",
                "RAG 依据：" + "；".join(notes[:2]),
                "命中文档：" + "、".join(doc.title for doc in docs),
            ]
        )
    baseline = solve_facility_location(data)
    scenario = apply_what_if(query, data)
    result = solve_facility_location(scenario.data)
    explanation = explain_result(baseline, result, scenario.changes)
    open_warehouses = []
    if not result.warehouse_summary.empty:
        open_warehouses = result.warehouse_summary.loc[
            result.warehouse_summary["is_open"] == 1,
            "warehouse",
        ].tolist()
    return "\n".join(
        [
            "未配置 LLM，已使用本地工具路由给出回答。",
            f"ProblemSpec：{spec.display_name} / {spec.problem_type}；推荐求解器：{spec.recommended_solver}。",
            f"识别到的场景修改：{'；'.join(scenario.changes)}",
            f"Gurobi 状态：{result.status}，总成本：{result.objective_value:,.2f}" if result.objective_value is not None else result.message,
            f"建议启用仓库：{', '.join(open_warehouses) if open_warehouses else '无'}",
            "优化解释：" + "；".join(explanation[:4]),
            "RAG 依据：" + "；".join(notes[:2]),
            "命中文档：" + "、".join(doc.title for doc in docs),
        ]
    )


def _load_mcp_tools_sync(mcp_config_json: str):
    if not mcp_config_json.strip():
        return [], "未配置 MCP，使用内置工具。"
    try:
        config = json.loads(mcp_config_json)
    except json.JSONDecodeError as exc:
        return [], f"MCP JSON 解析失败：{exc}"

    try:
        return asyncio.run(_load_mcp_tools(config)), "MCP 工具加载成功。"
    except Exception as exc:
        return [], f"MCP 工具加载失败：{exc}"


async def _load_mcp_tools(config: dict[str, Any]):
    from langchain_mcp_adapters.client import MultiServerMCPClient

    client = MultiServerMCPClient(config)
    return await client.get_tools()


def _extract_answer(raw: Any) -> str:
    messages = raw.get("messages", []) if isinstance(raw, dict) else []
    if not messages:
        return str(raw)
    last = messages[-1]
    content = getattr(last, "content", None)
    if content is None and isinstance(last, dict):
        content = last.get("content")
    return str(content)


def _extract_called_tool_names(raw: Any) -> list[str]:
    messages = raw.get("messages", []) if isinstance(raw, dict) else []
    names: list[str] = []
    seen: set[str] = set()
    for message in messages:
        tool_calls = getattr(message, "tool_calls", None)
        if tool_calls is None and isinstance(message, dict):
            tool_calls = message.get("tool_calls")
        if isinstance(tool_calls, list):
            for call in tool_calls:
                name = ""
                if isinstance(call, dict):
                    name = str(call.get("name") or call.get("function", {}).get("name") or "")
                else:
                    name = str(getattr(call, "name", "") or "")
                if name and name not in seen:
                    seen.add(name)
                    names.append(name)
        name = getattr(message, "name", None)
        if name is None and isinstance(message, dict):
            name = message.get("name")
        if name and str(name) not in seen:
            seen.add(str(name))
            names.append(str(name))
    return names


def _jsonable(value: Any) -> Any:
    try:
        json.dumps(value, ensure_ascii=False, default=str)
        return value
    except TypeError:
        return json.loads(json.dumps(value, ensure_ascii=False, default=str))
