from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from fastapi.testclient import TestClient

import api.database as database
from api.database import init_db, list_runs
from api.main import app
from api.services.agent_workflow import NODE_DEFINITIONS, run_agent_workflow


class AgentWorkflowTests(unittest.TestCase):
    """验证 LangGraph 状态、实时事件和最终轨迹持久化。"""

    def setUp(self) -> None:
        self._original_db_path = database.DB_PATH
        self._temporary_directory = tempfile.TemporaryDirectory()
        # 每项测试使用独立数据库，避免修改开发环境中的对话记录。
        database.DB_PATH = Path(self._temporary_directory.name) / "agent-workflow.sqlite3"
        init_db()

    def tearDown(self) -> None:
        database.DB_PATH = self._original_db_path
        self._temporary_directory.cleanup()

    def test_graph_emits_ordered_node_events_and_persists_trace(self) -> None:
        events: list[dict] = []
        result = run_agent_workflow(
            question=_knapsack_question(),
            requested_dataset_id=None,
            mcp_config="",
            user_id=None,
            conversation_id=None,
            event_callback=events.append,
        )

        expected_ids = [item[0] for item in NODE_DEFINITIONS]
        completed_ids = [event["node_id"] for event in events if event["status"] == "completed"]
        self.assertEqual("OPTIMAL", result["status"])
        self.assertEqual(expected_ids, completed_ids)
        self.assertEqual(len(expected_ids) * 2, len(events))
        self.assertTrue(result["workflow_verification"]["passed"])
        self.assertEqual(expected_ids, [node["node_id"] for node in result["agent_graph"]["nodes"]])

        persisted = json.loads(list_runs(limit=1)[0]["result_json"])
        self.assertEqual("LangGraph", persisted["agent_graph"]["engine"])
        self.assertEqual(6, len(persisted["agent_graph"]["nodes"]))

    def test_stream_api_contains_live_agent_step_events(self) -> None:
        with TestClient(app) as client:
            response = client.post("/api/ask/stream", json={"question": _knapsack_question()})

        event_types = [
            line.removeprefix("event: ")
            for line in response.text.splitlines()
            if line.startswith("event:")
        ]
        self.assertEqual(200, response.status_code)
        self.assertEqual(12, event_types.count("agent_step"))
        self.assertIn("answer_delta", event_types)
        self.assertEqual("final", event_types[-1])

    def test_natural_solve_wording_enters_solver_path(self) -> None:
        """验证“解决”这种自然表达也会进入优化求解路径。"""

        question = _knapsack_question().replace("请进行求解", "请解决")
        result = run_agent_workflow(
            question=question,
            requested_dataset_id=None,
            mcp_config="",
            user_id=None,
            conversation_id=None,
        )

        self.assertEqual("OPTIMAL", result["status"])
        planner = result["agent_graph"]["nodes"][0]
        self.assertIn("knapsack", planner["detail"])
        self.assertIn("执行优化", planner["detail"])


def _knapsack_question() -> str:
    payload = {
        "capacity": 5,
        "items": [
            {"item": "A", "value": 8, "weight": 3},
            {"item": "B", "value": 5, "weight": 2},
        ],
    }
    return "请进行求解：背包问题\n" + json.dumps(payload, ensure_ascii=False)


if __name__ == "__main__":
    unittest.main()
