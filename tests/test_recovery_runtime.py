from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import Mock, patch

import api.database as database
from api.services import agent_workflow as workflow
from optiagent.recovery_runtime import RecoveryPolicyRuntime, load_recovery_runtime, workflow_observation
from optiagent.rl.dqn import DQNConfig, MaskedDoubleDQNAgent
from optiagent.rl.state_encoder import RecoveryStateEncoder, TransportAwareRecoveryStateEncoder


class RecoveryRuntimeTests(unittest.TestCase):
    """验证实际执行与轨迹一致，并保证网络不能绕过数学验证或预算。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.original_db = database.DB_PATH
        database.DB_PATH = Path(self.temporary.name) / "runtime.sqlite3"
        database.init_db()

    def tearDown(self) -> None:
        database.DB_PATH = self.original_db
        self.temporary.cleanup()

    def test_network_cannot_accept_invalid_solution_or_exceed_budget(self) -> None:
        agent = Mock()
        runtime = RecoveryPolicyRuntime(agent, checkpoint_sha256="test")
        for action, attempt, expected in [("accept_solution", 1, "rebuild_model"), ("retry_solver", 2, "terminate")]:
            with self.subTest(action=action):
                agent.policy.return_value = action
                decision = runtime.decide(_state(attempt))
                self.assertEqual(expected, decision.selected_action)
                self.assertEqual("invalid_or_masked_action", decision.metadata["fallback_reason"])
                self.assertEqual("deterministic_recovery_baseline", decision.policy_name)

    def test_inference_error_falls_back_without_persisting_exception_text(self) -> None:
        agent = Mock()
        agent.policy.side_effect = RuntimeError("sensitive-runtime-details")
        decision = RecoveryPolicyRuntime(agent).decide(_state())
        self.assertEqual("rebuild_model", decision.selected_action)
        self.assertEqual("inference_error", decision.metadata["fallback_reason"])
        self.assertNotIn("sensitive-runtime-details", decision.model_dump_json())

    def test_observation_uses_completed_attempts_and_previous_cost(self) -> None:
        state = _state()
        state["trace_nodes"] = [{"node_id": "solver", "elapsed_ms": 10}]
        first = RecoveryPolicyRuntime().decide(state).model_dump(mode="json")
        state["policy_decisions"] = [first]
        observation = workflow_observation(state)
        self.assertEqual("rebuild_model", observation["last_action"])
        self.assertEqual(first["metadata"]["observation"]["features"]["rebuild_cost_estimate"], observation["cumulative_cost"])
        self.assertEqual(1, observation["remaining_solver_attempts"])

    def test_checkpoint_load_and_explicit_rule_override(self) -> None:
        checkpoint = Path(self.temporary.name) / "policy.pt"
        MaskedDoubleDQNAgent(RecoveryStateEncoder(), DQNConfig()).save(checkpoint)
        with patch.dict("os.environ", {"OPTIAGENT_RECOVERY_CHECKPOINT": str(checkpoint)}):
            runtime = load_recovery_runtime()
            self.assertEqual("masked_double_dqn", runtime.name)
            self.assertEqual(64, len(runtime.checkpoint_sha256))
            self.assertIs(runtime, load_recovery_runtime())
            self.assertEqual("deterministic_recovery_baseline", load_recovery_runtime("").name)
        transport_checkpoint = Path(self.temporary.name) / "transport.pt"
        MaskedDoubleDQNAgent(TransportAwareRecoveryStateEncoder(), DQNConfig()).save(transport_checkpoint)
        with self.assertRaisesRegex(ValueError, "transport"):
            load_recovery_runtime(transport_checkpoint)
        with self.assertRaises(FileNotFoundError):
            load_recovery_runtime(Path(self.temporary.name) / "absent.pt")

    def test_graph_records_learned_action_once_and_exports_exact_observation(self) -> None:
        # 规则遇到目标不一致会重建模，测试网络选择重试，以暴露旧的重复规则记录问题。
        agent = Mock()
        agent.policy.side_effect = ["retry_solver", "accept_solution"]
        runtime = RecoveryPolicyRuntime(agent, checkpoint_sha256="test-checkpoint")
        calls = []

        def solver(state: dict) -> dict:
            calls.append(1)
            passed = len(calls) > 1
            return {"result": {
                "answer": "恢复测试结果", "structured_answer": {}, "status": "OPTIMAL", "objective_value": 13.0,
                "solution_verification": {"verifiable": True, "passed": passed, "feasible": True,
                                          "objective_consistent": passed, "violations": []},
            }}

        graph = workflow._build_agent_graph({"solver": solver})
        question = "请进行求解背包问题，最大化价值。数据：" + json.dumps({
            "capacity": 5, "items": [{"item": "A", "weight": 3, "value": 8}, {"item": "B", "weight": 2, "value": 5}],
        })
        with patch.object(workflow, "AGENT_GRAPH", graph), patch.object(workflow, "load_recovery_runtime", return_value=runtime):
            result = workflow.run_agent_workflow(question=question, requested_dataset_id=None,
                                                 mcp_config="", user_id=None, conversation_id=None)
        training = database.get_training_episode(result["agent_episode_id"], user_id=None)
        decisions = result["agent_policy"]["decisions"]
        transitions = training["decision_transitions"]
        self.assertEqual(2, agent.policy.call_count)
        self.assertEqual(["retry_solver", "accept_solution"], [item["action"] for item in transitions])
        self.assertEqual([item["selected_action"] for item in decisions], [item["action"] for item in transitions])
        self.assertEqual("masked_double_dqn", training["policy"]["name"])
        for index, transition in enumerate(transitions):
            self.assertEqual(agent.policy.call_args_list[index].args[0], transition["policy_observation"])
            self.assertEqual("masked_double_dqn", transition["policy"]["name"])
        self.assertEqual(transitions[1]["policy_observation"], transitions[0]["next_policy_observation"])
        self.assertTrue(transitions[-1]["done"])
        self.assertEqual(1, sum(node["node_id"] == "modeler" for node in result["agent_graph"]["nodes"]))


def _state(attempt: int = 1) -> dict:
    """构造尚未通过验证的状态，保留合法恢复候选。"""

    return {"model_attempt": attempt, "solver_attempt": attempt,
            "verification": {"passed": False, "mathematical": {"verifiable": True, "passed": False}},
            "problem_spec": {"template_id": "knapsack"}}
