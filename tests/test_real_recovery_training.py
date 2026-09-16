from __future__ import annotations

from pathlib import Path
import tempfile
import unittest

from optiagent.rl.dqn import DQNConfig, MaskedDoubleDQNAgent
from optiagent.rl.real_environment import (
    RealGatewayRecoveryEnv,
    collect_real_recovery_tasks,
    cost_aware_teacher_policy,
    evaluate_real_policy,
)
from optiagent.rl.real_training import train_real_recovery_policy
from optiagent.rl.rollout import recovery_baseline_policy
from optiagent.rl.state_encoder import CostAwareRecoveryStateEncoder, RecoveryStateEncoder


class RealRecoveryEnvironmentTests(unittest.TestCase):
    """验证真实求解轨迹、成本奖励和恢复动作差异。"""

    @classmethod
    def setUpClass(cls) -> None:
        # 单一模板已足够覆盖五类故障和三个隔离 split。
        cls.tasks = collect_real_recovery_tasks(seed=19, template_ids=("knapsack",))

    def test_collection_uses_real_verified_solver_outputs(self) -> None:
        self.assertEqual(15, len(self.tasks))
        self.assertTrue(all(item.clean_verification["passed"] for item in self.tasks))
        self.assertTrue(all(item.solver_latency_ms >= 0.0 for item in self.tasks))
        self.assertEqual(
            {"train", "validation", "test"},
            {item.split for item in self.tasks},
        )

    def test_cost_encoder_extends_v1_without_scenario_leakage(self) -> None:
        task = self._task("tampered_objective", "train")
        observation, _ = RealGatewayRecoveryEnv([task]).reset(task_id=task.task_id)
        encoder = CostAwareRecoveryStateEncoder()
        altered = {**observation, "scenario": "不应进入状态编码的标签"}

        self.assertEqual(RecoveryStateEncoder().dimension + 6, encoder.dimension)
        self.assertEqual(encoder.dimension, encoder.encode(observation).shape[0])
        self.assertTrue((encoder.encode(observation) == encoder.encode(altered)).all())

    def test_retry_is_cheaper_for_tampered_result(self) -> None:
        task = self._task("tampered_objective", "test")
        teacher = evaluate_real_policy([task], cost_aware_teacher_policy, seed=19)
        rule = evaluate_real_policy([task], recovery_baseline_policy, seed=19)

        self.assertEqual(1.0, teacher["metrics"]["success_rate"])
        self.assertEqual(1.0, rule["metrics"]["success_rate"])
        self.assertGreater(teacher["metrics"]["average_return"], rule["metrics"]["average_return"])

    def test_model_error_requires_rebuild_instead_of_retry(self) -> None:
        task = self._task("repairable_model_error", "test")
        environment = RealGatewayRecoveryEnv([task])
        observation, _ = environment.reset(task_id=task.task_id)

        self.assertEqual("rebuild_model", cost_aware_teacher_policy(observation))
        recovered, _, _, _, _ = environment.step("rebuild_model")
        self.assertTrue(recovered["verification"]["passed"])

    def test_model_error_fault_is_detected_for_all_templates(self) -> None:
        tasks = collect_real_recovery_tasks(
            seed=29,
            splits=("test",),
            scenarios=("repairable_model_error",),
        )

        self.assertEqual(6, len(tasks))
        self.assertTrue(all(not item.model_error_verification["passed"] for item in tasks))
        self.assertTrue(
            all(item.model_error_verification["mathematical"]["feasible"] is False for item in tasks)
        )

    def _task(self, scenario: str, split: str):
        return next(item for item in self.tasks if item.scenario == scenario and item.split == split)


class RealRecoveryTrainingTests(unittest.TestCase):
    """用小规模真实轨迹训练一次，验证学习策略和 v2 checkpoint。"""

    @classmethod
    def setUpClass(cls) -> None:
        import torch

        torch.set_num_threads(1)
        cls.tasks = collect_real_recovery_tasks(seed=23, template_ids=("knapsack",))
        cls.training = train_real_recovery_policy(
            cls.tasks,
            DQNConfig(
                seed=23,
                train_episodes=180,
                bc_epochs=80,
                batch_size=16,
                warmup_transitions=32,
                target_sync_interval=40,
                epsilon_decay_steps=240,
            ),
        )

    def test_learned_policy_recovers_and_beats_rule_cost(self) -> None:
        learned = self.training.report["evaluation"]["learned_policy"]["test"]
        rule = self.training.report["evaluation"]["rule_policy"]["test"]

        self.assertEqual(1.0, learned["recoverable_success_rate"])
        self.assertGreater(learned["average_return"], rule["average_return"])
        self.assertLess(learned["average_action_cost"], rule["average_action_cost"])

    def test_cost_aware_checkpoint_roundtrip(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "real_policy.pt"
            self.training.agent.save(path)
            restored = MaskedDoubleDQNAgent.load(path)

        self.assertIsInstance(restored.state_encoder, CostAwareRecoveryStateEncoder)
        self.assertEqual(
            self.training.agent.state_encoder.dimension,
            restored.state_encoder.dimension,
        )


if __name__ == "__main__":
    unittest.main()
