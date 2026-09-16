from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from optiagent.rl.dqn import DQNConfig, MaskedDoubleDQNAgent
from optiagent.rl.state_encoder import CostAwareRecoveryStateEncoder, TransportAwareRecoveryStateEncoder
from optiagent.rl.transport_environment import (
    MCPTransportRecoveryEnv,
    collect_transport_recovery_tasks,
    evaluate_transport_policy,
    transport_teacher_policy,
)
from optiagent.rl.transport_training import train_transport_recovery_policy


class MCPTransportEnvironmentTests(unittest.TestCase):
    """验证真实 MCP 协议故障采集、状态编码和恢复转移。"""

    @classmethod
    def setUpClass(cls) -> None:
        cls.tasks = collect_transport_recovery_tasks(
            seed=31,
            template_ids=("knapsack",),
            timeout_ms=80,
            delay_ms=240,
        )

    def test_real_stdio_traces_cover_four_transport_outcomes(self) -> None:
        traces = {task.transport_trace.fault: task.transport_trace for task in self.tasks}

        self.assertEqual({"normal", "timeout", "disconnect", "invalid_return"}, set(traces))
        self.assertTrue(traces["normal"].success)
        self.assertTrue(traces["normal"].response_schema_valid)
        for fault in ("timeout", "disconnect", "invalid_return"):
            self.assertFalse(traces[fault].success)
            self.assertTrue(traces[fault].error_type)

    def test_transport_encoder_adds_observed_signals_without_label_leakage(self) -> None:
        task = self._task("timeout", "train")
        observation, _ = MCPTransportRecoveryEnv([task]).reset(task_id=task.task_id)
        encoder = TransportAwareRecoveryStateEncoder()
        altered = {**observation, "scenario": "不应进入模型的场景标签"}

        self.assertEqual(CostAwareRecoveryStateEncoder().dimension + 5, encoder.dimension)
        self.assertEqual(encoder.dimension, encoder.encode(observation).shape[0])
        self.assertTrue((encoder.encode(observation) == encoder.encode(altered)).all())

    def test_timeout_retries_but_invalid_return_rebuilds_contract(self) -> None:
        timeout_task = self._task("timeout", "test")
        timeout_env = MCPTransportRecoveryEnv([timeout_task])
        timeout_observation, _ = timeout_env.reset(task_id=timeout_task.task_id)
        self.assertEqual("retry_solver", transport_teacher_policy(timeout_observation))
        timeout_recovered, _, _, _, _ = timeout_env.step("retry_solver")
        self.assertTrue(timeout_recovered["verification"]["passed"])

        invalid_task = self._task("invalid_return", "test")
        invalid_env = MCPTransportRecoveryEnv([invalid_task])
        invalid_observation, _ = invalid_env.reset(task_id=invalid_task.task_id)
        self.assertEqual("rebuild_model", transport_teacher_policy(invalid_observation))
        invalid_recovered, _, _, _, _ = invalid_env.step("rebuild_model")
        self.assertTrue(invalid_recovered["verification"]["passed"])

    def test_persistent_timeout_terminates_after_retry_budget(self) -> None:
        task = self._task("persistent_timeout", "test")
        environment = MCPTransportRecoveryEnv([task])
        observation, _ = environment.reset(task_id=task.task_id)
        observation, _, _, _, _ = environment.step(transport_teacher_policy(observation))
        _, _, terminated, _, info = environment.step(transport_teacher_policy(observation))

        self.assertTrue(terminated)
        self.assertFalse(info["success"])
        self.assertEqual("recovery_budget_exhausted", info["terminal_reason"])

    def _task(self, scenario: str, split: str):
        return next(item for item in self.tasks if item.scenario == scenario and item.split == split)


class MCPTransportTrainingTests(unittest.TestCase):
    """用小规模 transport 任务验证学习、测试集泛化和 v3 checkpoint。"""

    @classmethod
    def setUpClass(cls) -> None:
        import torch

        torch.set_num_threads(1)
        cls.tasks = collect_transport_recovery_tasks(
            seed=37,
            template_ids=("knapsack",),
            timeout_ms=80,
            delay_ms=240,
        )
        cls.training = train_transport_recovery_policy(
            cls.tasks,
            DQNConfig(
                seed=37,
                train_episodes=240,
                bc_epochs=100,
                batch_size=16,
                warmup_transitions=32,
                target_sync_interval=40,
                epsilon_decay_steps=300,
            ),
        )

    def test_learned_policy_recovers_all_recoverable_transport_faults(self) -> None:
        learned = self.training.report["evaluation"]["learned_policy"]["test"]
        teacher = self.training.report["evaluation"]["transport_teacher"]["test"]

        self.assertEqual(1.0, learned["recoverable_success_rate"])
        self.assertEqual(0.0, learned["invalid_action_rate"])
        self.assertEqual(teacher["average_return"], learned["average_return"])

    def test_transport_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "transport_policy.pt"
            self.training.agent.save(path)
            restored = MaskedDoubleDQNAgent.load(path)

        self.assertIsInstance(restored.state_encoder, TransportAwareRecoveryStateEncoder)
        test_tasks = [item for item in self.tasks if item.split == "test"]
        metrics = evaluate_transport_policy(test_tasks, restored.policy, seed=37)["metrics"]
        self.assertEqual(1.0, metrics["recoverable_success_rate"])


if __name__ == "__main__":
    unittest.main()
