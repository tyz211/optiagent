from __future__ import annotations

import json
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import pandas as pd

from api.services.answer_presenter import (
    facility_profile,
    generic_agent_steps,
    generic_answer_text,
    rag_context_preview,
    structured_answer,
)
from api.services.ask_routing import (
    build_agent_plan,
    external_data_gaps,
    is_facility_question,
    needs_external_data,
    should_solve_optimization,
)
from optiagent.data import SupplyChainData
from optiagent.llm import LLMConfig


class AskRoutingTests(unittest.TestCase):
    """验证拆分后的 Agent 路由仍保留确定性保护规则。"""

    def setUp(self) -> None:
        self.llm_config = LLMConfig(
            enabled=True,
            api_key="test-key",
            base_url="https://example.test/v1",
            model="test-model",
        )

    def test_missing_llm_uses_local_fallback(self) -> None:
        plan, warning = build_agent_plan("求解背包问题", [], None)

        self.assertIsNone(plan)
        self.assertIn("本地规则路由", warning)

    def test_llm_plan_is_validated_and_external_data_tool_is_inserted(self) -> None:
        raw_plan = json.dumps(
            {
                "template_id": "facility_location",
                "confidence": 1.8,
                "tool_chain": [],
                "needs_solver": True,
            }
        )
        with (
            patch("api.services.ask_routing.call_openai_compatible_chat", return_value=raw_plan),
            patch("api.services.ask_routing.facility_data_from_uploaded_files", return_value=None),
            patch("api.services.ask_routing.has_executable_generic_upload", return_value=False),
        ):
            plan, warning = build_agent_plan("使用最新公开数据进行城市选址", [], self.llm_config)

        self.assertIsNone(warning)
        self.assertEqual(1.0, plan["confidence"])
        self.assertTrue(plan["llm_used"])
        self.assertEqual("problem_spec_tool", plan["tool_chain"][0])
        self.assertEqual("web_search_tool", plan["tool_chain"][1])
        self.assertIn("solver_solve_problem", plan["tool_chain"])

    def test_unknown_template_is_rejected(self) -> None:
        raw_plan = json.dumps({"template_id": "unknown_problem", "confidence": 0.9})
        with patch("api.services.ask_routing.call_openai_compatible_chat", return_value=raw_plan):
            plan, warning = build_agent_plan("处理一个未知问题", [], self.llm_config)

        self.assertIsNone(plan)
        self.assertIn("未知问题类型", warning)

    def test_intent_and_data_gap_rules_cover_solve_analysis_and_web(self) -> None:
        self.assertTrue(should_solve_optimization("请解决这个仓库选址问题"))
        self.assertFalse(should_solve_optimization("分析一下当前仓库数据"))
        self.assertTrue(needs_external_data("搜索最新物流园公开数据"))
        self.assertFalse(needs_external_data("根据上传数据求解背包问题"))
        self.assertTrue(is_facility_question("请进行客户分配"))

        gaps = external_data_gaps("根据真实数据进行仓库选址", [{"filename": "warehouses.csv"}])
        self.assertEqual(4, len(gaps))
        self.assertTrue(all(item.startswith("已读取当前对话上传文件") for item in gaps))


class AnswerPresenterTests(unittest.TestCase):
    """验证展示层输出字段和关键业务含义保持稳定。"""

    def setUp(self) -> None:
        self.generic_result = SimpleNamespace(
            template_id="knapsack",
            display_name="0-1 背包",
            decisions=[
                {"item": "1", "selected": 1},
                {"item": "2", "selected": 0},
            ],
            metrics={
                "selected_value": 13.0,
                "used_weight": 5.0,
                "capacity": 5.0,
                "remaining_capacity": 0.0,
                "selected_count": 1,
                "optimality_proven": True,
                "mip_gap": 0.0,
            },
            objective_value=13.0,
            objective_label="最大总价值",
            solver_name="Gurobi",
            status="OPTIMAL",
            data_source="测试数据",
            summary="选择物品 1，最大价值为 13。",
            warnings=[],
        )
        self.problem_spec = SimpleNamespace(
            display_name="0-1 背包",
            problem_type="integer_programming",
            recommended_solver="Gurobi",
        )

    def test_generic_answer_and_structured_metrics_remain_consistent(self) -> None:
        answer = generic_answer_text(self.generic_result, self.problem_spec, ["背包模型证据"])
        payload = structured_answer(
            answer,
            result=None,
            baseline=None,
            changes=[],
            rag_notes=["背包模型证据"],
            open_warehouses=[],
            problem_spec=self.problem_spec,
            generic_result=self.generic_result,
        )

        self.assertIn("物品 1", answer)
        self.assertIn("13.00", answer)
        self.assertEqual(13.0, payload["metrics"]["total_cost"])
        self.assertTrue(payload["metrics"]["optimality_proven"])
        self.assertEqual("背包模型证据", payload["evidence"][0])

    def test_generic_steps_include_gateway_validation_and_optimality_check(self) -> None:
        steps = generic_agent_steps(self.generic_result)
        tools = [item["tool"] for item in steps]

        self.assertIn("mcp_gateway", tools)
        self.assertIn("data_validate_problem", tools)
        self.assertIn("solver_solve_problem", tools)
        self.assertIn("optimality_checker", tools)
        self.assertIn("已按工具能力证明", next(item["output"] for item in steps if item["tool"] == "optimality_checker"))

    def test_facility_profile_calculates_capacity_and_risk(self) -> None:
        data = SupplyChainData(
            warehouses=pd.DataFrame(
                [
                    {"warehouse": "武汉", "capacity": 60, "fixed_cost": 100},
                    {"warehouse": "南昌", "capacity": 40, "fixed_cost": 80},
                ]
            ),
            customers=pd.DataFrame(
                [
                    {"customer": "客户A", "demand": 50},
                    {"customer": "客户B", "demand": 45},
                ]
            ),
            costs=pd.DataFrame(
                [
                    {"warehouse": "武汉", "customer": "客户A", "cost": 10},
                    {"warehouse": "南昌", "customer": "客户B", "cost": 30},
                ]
            ),
        )

        profile = facility_profile(data)

        self.assertEqual(100.0, profile["total_capacity"])
        self.assertEqual(95.0, profile["total_demand"])
        self.assertAlmostEqual(0.95, profile["demand_capacity_ratio"])
        self.assertTrue(any("容量" in risk for risk in profile["risks"]))

    def test_rag_preview_does_not_expose_full_document_content(self) -> None:
        preview = rag_context_preview(
            {
                "建模知识": [
                    {
                        "title": "背包模型",
                        "score": 0.91,
                        "source": "knowledge.md",
                        "content": "完整正文不应进入预览。",
                    }
                ]
            }
        )

        self.assertEqual("背包模型", preview["建模知识"][0]["title"])
        self.assertNotIn("content", preview["建模知识"][0])


if __name__ == "__main__":
    unittest.main()
