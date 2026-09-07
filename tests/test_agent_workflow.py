from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from fastapi import HTTPException
from fastapi.testclient import TestClient

import api.database as database
from api.database import (
    clear_runs,
    complete_agent_episode,
    create_agent_episode,
    get_agent_episode,
    get_training_episode,
    init_db,
    list_agent_episodes,
    list_runs,
)
from api.main import app
from api.services.agent_workflow import NODE_DEFINITIONS, _build_agent_graph, _deterministic_reward, run_agent_workflow
from optiagent.agent_policy import decide_after_verification


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
        self.assertTrue(result["workflow_verification"]["mathematical"]["passed"])
        self.assertEqual(1.0, result["workflow_verification"]["reward"]["total"])
        self.assertEqual(expected_ids, [node["node_id"] for node in result["agent_graph"]["nodes"]])

        persisted = json.loads(list_runs(limit=1)[0]["result_json"])
        self.assertEqual("LangGraph", persisted["agent_graph"]["engine"])
        self.assertEqual(len(expected_ids), len(persisted["agent_graph"]["nodes"]))
        self.assertTrue(persisted["solution_verification"]["passed"])
        self.assertEqual(1.0, persisted["workflow_verification"]["reward"]["total"])
        self.assertEqual("accept_solution", persisted["agent_policy"]["decisions"][0]["selected_action"])

        episode = get_agent_episode(result["agent_episode_id"], user_id=None)
        self.assertEqual("completed", episode["status"])
        self.assertEqual("knapsack", episode["template_id"])
        self.assertEqual(1.0, episode["total_reward"])
        self.assertEqual(expected_ids, [step["node_id"] for step in episode["steps"]])
        self.assertTrue(all(step["status"] == "completed" for step in episode["steps"]))
        self.assertTrue(all(step["state"] and step["action"] and step["observation"] for step in episode["steps"]))

        training = get_training_episode(result["agent_episode_id"], user_id=None)
        self.assertEqual(len(expected_ids), len(training["transitions"]))
        self.assertTrue(training["transitions"][-1]["done"])
        self.assertEqual(1.0, training["transitions"][-1]["reward"])
        self.assertTrue(all(item["reward"] == 0.0 for item in training["transitions"][:-1]))
        policy_action = next(item["action"] for item in training["transitions"] if item["node_id"] == "policy")
        self.assertEqual("accept_solution", policy_action["selected_action"])
        self.assertEqual(len(policy_action["candidate_ids"]), len(policy_action["action_mask"]))
        self.assertEqual(1, len(training["decision_transitions"]))
        self.assertEqual("accept_solution", training["decision_transitions"][0]["action"])
        self.assertEqual(1.0, training["decision_transitions"][0]["reward"])
        self.assertTrue(training["decision_transitions"][0]["done"])

    def test_stream_api_contains_live_agent_step_events(self) -> None:
        with TestClient(app) as client:
            response = client.post("/api/ask/stream", json={"question": _knapsack_question()})

        event_types = [
            line.removeprefix("event: ")
            for line in response.text.splitlines()
            if line.startswith("event:")
        ]
        self.assertEqual(200, response.status_code)
        self.assertEqual(len(NODE_DEFINITIONS) * 2, event_types.count("agent_step"))
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

    def test_infeasible_solution_receives_negative_reward(self) -> None:
        """确保求解器自报成功不能覆盖数学验证失败。"""

        reward = _deterministic_reward(
            {"status": "FEASIBLE"},
            {"has_answer": True, "has_status": True},
            {"feasible": False, "objective_consistent": True},
        )

        self.assertLess(reward["total"], 0)
        self.assertEqual(0.0, reward["components"]["terminal_success"])

        unsolved_reward = _deterministic_reward(
            {"status": "INFEASIBLE"},
            {"has_answer": True, "has_status": True},
            {"verifiable": False, "feasible": None, "objective_consistent": None},
        )
        self.assertLess(unsolved_reward["total"], 0)

    def test_failed_workflow_persists_replayable_episode(self) -> None:
        """节点异常也必须留下失败 step 和 episode，而不是丢失负样本。"""

        with self.assertRaises(HTTPException):
            run_agent_workflow(
                question="请分析并解决这个优化问题",
                requested_dataset_id=None,
                mcp_config="",
                user_id=None,
                conversation_id=None,
            )

        summaries = list_agent_episodes(user_id=None)
        self.assertEqual(1, len(summaries))
        episode = get_agent_episode(summaries[0]["episode_id"], user_id=None)
        self.assertEqual("failed", episode["status"])
        self.assertEqual("facility_location", episode["template_id"])
        self.assertEqual(-1.0, episode["total_reward"])
        self.assertIn("HTTPException", episode["error"])
        self.assertEqual("solver", episode["steps"][-1]["node_id"])
        self.assertEqual("failed", episode["steps"][-1]["status"])
        self.assertEqual(-1.0, episode["steps"][-1]["reward"]["total"])
        self.assertTrue(episode["steps"][-1]["observation"]["message"])
        training = get_training_episode(episode["episode_id"], user_id=None)
        self.assertEqual("facility_location", training["task"]["template_id"])
        self.assertEqual(-1.0, training["transitions"][-1]["reward"])
        self.assertTrue(training["transitions"][-1]["done"])

    def test_trajectory_query_and_training_export_api(self) -> None:
        """验证审计详情与批量训练数据可以通过只读 API 获取。"""

        result = run_agent_workflow(
            question=_knapsack_question(),
            requested_dataset_id=None,
            mcp_config="",
            user_id=None,
            conversation_id=None,
        )
        episode_id = result["agent_episode_id"]

        with TestClient(app) as client:
            listing = client.get("/api/agent/episodes")
            detail = client.get(f"/api/agent/episodes/{episode_id}")
            training = client.get(f"/api/agent/episodes/{episode_id}/training")
            batch = client.get("/api/agent/training-data")

        self.assertEqual(200, listing.status_code)
        self.assertEqual(episode_id, listing.json()["episodes"][0]["episode_id"])
        self.assertEqual(len(NODE_DEFINITIONS), len(detail.json()["steps"]))
        self.assertEqual(len(NODE_DEFINITIONS), len(training.json()["transitions"]))
        self.assertEqual(episode_id, batch.json()["episodes"][0]["episode_id"])

    def test_clearing_runs_also_clears_trajectory(self) -> None:
        """对话历史清理后不应留下失去父运行记录的孤立轨迹。"""

        run_agent_workflow(
            question=_knapsack_question(),
            requested_dataset_id=None,
            mcp_config="",
            user_id=None,
            conversation_id=None,
        )
        self.assertEqual(1, len(list_agent_episodes(user_id=None)))

        clear_runs(user_id=None, conversation_id=None)

        self.assertEqual([], list_agent_episodes(user_id=None))

    def test_trajectory_queries_respect_user_and_conversation_scope(self) -> None:
        """轨迹列表和详情必须与用户及会话范围保持一致。"""

        anonymous_episode = create_agent_episode("匿名任务", None, None)
        first_episode = create_agent_episode("用户任务 A", 101, 11)
        second_episode = create_agent_episode("用户任务 B", 101, 12)
        other_user_episode = create_agent_episode("其他用户任务", 202, 11)

        self.assertEqual([anonymous_episode], [item["episode_id"] for item in list_agent_episodes(user_id=None)])
        scoped = list_agent_episodes(user_id=101, conversation_id=11)
        self.assertEqual([first_episode], [item["episode_id"] for item in scoped])
        self.assertIsNone(get_agent_episode(first_episode, user_id=202))
        self.assertIsNotNone(get_agent_episode(second_episode, user_id=101))
        self.assertIsNotNone(get_agent_episode(other_user_episode, user_id=202))

    def test_policy_exposes_rebuild_and_terminate_actions(self) -> None:
        """数学验证异常应先返回 Modeler，预算耗尽后必须终止。"""

        verification = {
            "passed": False,
            "mathematical": {"verifiable": True, "passed": False},
        }
        rebuild = decide_after_verification(verification, model_attempt=1, solver_attempt=1)
        terminate = decide_after_verification(verification, model_attempt=2, solver_attempt=2)

        self.assertEqual("rebuild_model", rebuild.selected_action)
        self.assertTrue(rebuild.action_mask[2])
        self.assertEqual("terminate", terminate.selected_action)
        self.assertEqual([False, False, False, True], terminate.action_mask)

    def test_graph_retries_solver_and_preserves_chronological_transitions(self) -> None:
        """条件边应让一次未求解状态返回 Solver，而不是直接进入 Explainer。"""

        solver_calls: list[int] = []

        def fake_solver(state: dict) -> dict:
            # 第一次模拟暂时未求解，第二次返回已验证的可行解。
            solver_calls.append(int(state.get("solver_attempt", 0)) + 1)
            if len(solver_calls) == 1:
                return {
                    "result": {
                        "answer": "暂未得到可行解",
                        "structured_answer": {},
                        "status": "ERROR",
                        "objective_value": None,
                        "solution_verification": {
                            "verifiable": False,
                            "passed": False,
                            "feasible": None,
                            "objective_consistent": None,
                        },
                    },
                    "_trace_detail": "ERROR",
                }
            return {
                "result": {
                    "answer": "已得到可行解",
                    "structured_answer": {},
                    "status": "OPTIMAL",
                    "objective_value": 13.0,
                    "solution_verification": {
                        "verifiable": True,
                        "passed": True,
                        "feasible": True,
                        "objective_consistent": True,
                        "violations": [],
                    },
                },
                "_trace_detail": "OPTIMAL · 目标值 13.00",
            }

        episode_id = create_agent_episode("恢复测试", None, None)
        graph = _build_agent_graph({"solver": fake_solver})
        final_state = graph.invoke(
            {
                "question": _knapsack_question(),
                "requested_dataset_id": None,
                "mcp_config": "",
                "user_id": None,
                "conversation_id": None,
                "episode_id": episode_id,
                "model_attempt": 0,
                "solver_attempt": 0,
                "policy_decisions": [],
                "trace_nodes": [],
            }
        )

        self.assertEqual([1, 2], solver_calls)
        self.assertEqual(
            ["retry_solver", "accept_solution"],
            [item["selected_action"] for item in final_state["policy_decisions"]],
        )
        episode = get_agent_episode(episode_id, user_id=None)
        self.assertEqual(list(range(1, len(episode["steps"]) + 1)), [step["sequence"] for step in episode["steps"]])
        solver_steps = [step for step in episode["steps"] if step["node_id"] == "solver"]
        self.assertEqual([1, 2], [step["attempt"] for step in solver_steps])
        complete_agent_episode(
            episode_id,
            status="completed",
            template_id="knapsack",
            reward=final_state["verification"]["reward"],
            result=final_state["result"],
        )
        training = get_training_episode(episode_id, user_id=None)
        self.assertEqual(["retry_solver", "accept_solution"], [item["action"] for item in training["decision_transitions"]])
        self.assertEqual([0.0, 1.0], [item["reward"] for item in training["decision_transitions"]])
        self.assertTrue(training["decision_transitions"][-1]["done"])

    def test_graph_rebuilds_model_from_verifier_feedback(self) -> None:
        """数学验算失败时应将异常反馈带回第二次建模。"""

        solver_calls = 0

        def fake_solver(state: dict) -> dict:
            nonlocal solver_calls
            solver_calls += 1
            passed = solver_calls > 1
            return {
                "result": {
                    "answer": "验证测试结果",
                    "structured_answer": {},
                    "status": "OPTIMAL",
                    "objective_value": 13.0,
                    "solution_verification": {
                        "verifiable": True,
                        "passed": passed,
                        "feasible": passed,
                        "objective_consistent": passed,
                        "violations": [] if passed else ["容量约束违反"],
                    },
                },
                "_trace_detail": "OPTIMAL",
            }

        episode_id = create_agent_episode("重建模测试", None, None)
        final_state = _build_agent_graph({"solver": fake_solver}).invoke(
            {
                "question": _knapsack_question(),
                "requested_dataset_id": None,
                "mcp_config": "",
                "user_id": None,
                "conversation_id": None,
                "episode_id": episode_id,
                "model_attempt": 0,
                "solver_attempt": 0,
                "policy_decisions": [],
                "trace_nodes": [],
            }
        )

        self.assertEqual(["rebuild_model", "accept_solution"], [item["selected_action"] for item in final_state["policy_decisions"]])
        modeler_steps = [item for item in final_state["trace_nodes"] if item["node_id"] == "modeler"]
        self.assertEqual([1, 2], [item["attempt"] for item in modeler_steps])
        second_spec = final_state["problem_spec"]
        self.assertTrue(any("容量约束违反" in note for note in second_spec["notes"]))


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
