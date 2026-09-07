from __future__ import annotations

import json
from pathlib import Path
import unittest

import pandas as pd

from api.services.ask_service import _solve_json_generic, _solve_uploaded_generic
from optiagent.data import SupplyChainData, normalize_data
from optiagent.optimization_gateway import (
    IN_PROCESS_MCP_STATUS,
    solve_facility_via_gateway,
    solve_generic_via_gateway,
)


PROJECT_ROOT = Path(__file__).resolve().parent.parent


class GatewayMigrationTests(unittest.TestCase):
    """验证原有直连路径已经迁移到统一 MCP Gateway。"""

    def test_inline_json_uses_gateway_contract(self) -> None:
        question = json.dumps(
            {
                "capacity": 5,
                "items": [
                    {"item": "A", "value": 8, "weight": 3},
                    {"item": "B", "value": 5, "weight": 2},
                ],
            },
            ensure_ascii=False,
        )
        result = _solve_json_generic(f"请求解背包问题\n{question}", "knapsack")
        self.assertIsNotNone(result)
        self.assertEqual("OPTIMAL", result.status)
        self.assertEqual(13.0, result.objective_value)
        self.assertTrue(result.solution_verification["passed"])

    def test_uploaded_csv_uses_gateway_contract(self) -> None:
        csv_text = "item,value,weight\nA,8,3\nB,5,2\n"
        files = [
            {
                "filename": "items.csv",
                "role": "knapsack",
                "columns_json": json.dumps(["item", "value", "weight"]),
                "preview_csv": csv_text,
                "content_csv": csv_text,
            }
        ]
        result = _solve_uploaded_generic("背包容量 5，请求最大价值", files, "knapsack")
        self.assertIsNotNone(result)
        self.assertEqual("OPTIMAL", result.status)
        self.assertEqual(13.0, result.objective_value)

    def test_facility_gateway_preserves_legacy_result_shape(self) -> None:
        data = normalize_data(
            SupplyChainData(
                warehouses=pd.read_csv(PROJECT_ROOT / "data/facility_location_warehouses.csv"),
                customers=pd.read_csv(PROJECT_ROOT / "data/facility_location_customers.csv"),
                costs=pd.read_csv(PROJECT_ROOT / "data/facility_location_costs.csv"),
            )
        )
        result = solve_facility_via_gateway(data, data_source="测试数据", time_limit=5)
        self.assertEqual("OPTIMAL", result.status)
        self.assertEqual(1_021_700.0, result.objective_value)
        self.assertFalse(result.warehouse_summary.empty)
        self.assertFalse(result.allocations.empty)
        self.assertTrue(result.solution_verification["passed"])

    def test_invalid_data_is_blocked_before_solver(self) -> None:
        result = solve_generic_via_gateway(
            "knapsack",
            {"capacity": 0, "items": [{"item": "A", "value": 8, "weight": 3}]},
        )
        self.assertEqual("INVALID_DATA", result.status)
        self.assertIsNone(result.objective_value)
        self.assertTrue(any("capacity" in warning for warning in result.warnings))

    def test_orchestration_has_no_direct_solver_imports(self) -> None:
        ask_source = (PROJECT_ROOT / "api/services/ask_service.py").read_text(encoding="utf-8")
        agent_source = (PROJECT_ROOT / "optiagent/langchain_agents.py").read_text(encoding="utf-8")
        server_source = (PROJECT_ROOT / "optiagent/mcp_servers/solver_server.py").read_text(encoding="utf-8")
        self.assertNotIn("from optiagent.generic_solvers import", ask_source)
        self.assertNotIn("from optiagent.solver import", ask_source)
        self.assertNotIn("from optiagent.generic_solvers import", agent_source)
        self.assertNotIn("from optiagent.solver import", agent_source)
        self.assertNotIn("from optiagent.solver import", server_source)
        self.assertIn("solve_problem_envelope", server_source)
        self.assertIn("已通过 MCP Gateway", IN_PROCESS_MCP_STATUS)


if __name__ == "__main__":
    unittest.main()
