from __future__ import annotations

import asyncio
import sys
import unittest

from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from optiagent.mcp_client import load_mcp_tools_sync, parse_mcp_config
from optiagent.mcp_servers.common import resolve_readable_path
from optiagent.mcp_servers.data_server import data_build_problem, validate_problem_data
from optiagent.mcp_servers.document_server import document_search_knowledge
from optiagent.mcp_servers.solver_server import solver_solve_problem


class MCPContractTests(unittest.TestCase):
    """验证三类 MCP 之间的公共合同和安全边界。"""

    def test_default_config_contains_three_builtin_servers(self) -> None:
        config = parse_mcp_config("")
        self.assertEqual({"document", "data", "solver"}, set(config))
        self.assertTrue(all(item["transport"] == "stdio" for item in config.values()))

    def test_path_outside_project_is_rejected(self) -> None:
        with self.assertRaises(ValueError):
            resolve_readable_path("/etc/hosts")

    def test_document_knowledge_search_returns_evidence(self) -> None:
        result = document_search_knowledge("旅行商问题如何选择求解器", top_k=2)
        self.assertEqual(2, len(result.documents))
        self.assertTrue(all("source" in item for item in result.documents))

    def test_tsp_file_runs_through_data_and_solver_contract(self) -> None:
        problem = data_build_problem("tsp", ["data/tsp.csv"])
        self.assertTrue(problem.validation.valid)
        self.assertEqual("1.0", problem.schema_version)
        result = solver_solve_problem(problem, time_limit=5)
        self.assertEqual("OPTIMAL", result.status)
        self.assertGreater(result.objective_value or 0, 0)
        self.assertTrue(result.provenance)

    def test_invalid_knapsack_stops_before_solver(self) -> None:
        report = validate_problem_data(
            "knapsack",
            {"capacity": 0, "items": [{"item": "A", "value": 10, "weight": 2}]},
        )
        self.assertFalse(report.valid)
        self.assertTrue(any("capacity" in error for error in report.errors))


class MCPProtocolTests(unittest.TestCase):
    """通过真实 stdio 连接验证服务发现。"""

    def test_builtin_servers_are_discoverable(self) -> None:
        result = load_mcp_tools_sync("")
        self.assertFalse(result.errors)
        self.assertEqual({"document", "data", "solver"}, set(result.connected_servers))
        names = {tool.name for tool in result.tools}
        self.assertIn("document_document_read", names)
        self.assertIn("data_data_build_problem", names)
        self.assertIn("solver_solver_solve_problem", names)

    def test_data_to_solver_stdio_round_trip(self) -> None:
        """验证 Data MCP 的标准输出可直接作为 Solver MCP 的标准输入。"""

        async def run_round_trip() -> None:
            # 分别启动两个独立 MCP 进程，避免测试退化为进程内函数调用。
            data_server = StdioServerParameters(
                command=sys.executable,
                args=["-m", "optiagent.mcp_servers.data_server"],
            )
            solver_server = StdioServerParameters(
                command=sys.executable,
                args=["-m", "optiagent.mcp_servers.solver_server"],
            )
            async with stdio_client(data_server) as (data_read, data_write):
                async with ClientSession(data_read, data_write) as data_session:
                    await data_session.initialize()
                    problem_result = await data_session.call_tool(
                        "data_build_problem",
                        {"template_id": "tsp", "paths": ["data/tsp.csv"]},
                    )
                    self.assertFalse(problem_result.isError)
                    self.assertIsInstance(problem_result.structuredContent, dict)

            async with stdio_client(solver_server) as (solver_read, solver_write):
                async with ClientSession(solver_read, solver_write) as solver_session:
                    await solver_session.initialize()
                    solve_result = await solver_session.call_tool(
                        "solver_solve_problem",
                        {"problem": problem_result.structuredContent, "time_limit": 5},
                    )
                    self.assertFalse(solve_result.isError)
                    self.assertEqual("OPTIMAL", solve_result.structuredContent["status"])
                    self.assertGreater(solve_result.structuredContent["objective_value"], 0)

        asyncio.run(run_round_trip())


if __name__ == "__main__":
    unittest.main()
