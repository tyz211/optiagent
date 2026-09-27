from __future__ import annotations

from importlib.util import find_spec
from pathlib import Path
import tempfile
import unittest

import numpy as np

from optiagent.rl.benchmark import generate_benchmark
from optiagent.rl.environment import ACTION_IDS, OptimizationAgentEnv
from optiagent.rl.state_encoder import RecoveryStateEncoder


TORCH_AVAILABLE = find_spec("torch") is not None
if TORCH_AVAILABLE:
    from optiagent.rl.dqn import DQNConfig, MaskedDoubleDQNAgent
    from optiagent.rl.training import train_recovery_policy


class RecoveryStateEncoderTests(unittest.TestCase):
    """验证状态编码稳定且不使用 benchmark 答案标签。"""

    def test_encoder_has_stable_dimension_and_mask_order(self) -> None:
        task = generate_benchmark(seed=5, variants_per_template=3)[0]
        observation, _ = OptimizationAgentEnv([task]).reset(task_id=task.task_id)
        encoder = RecoveryStateEncoder()

        encoded = encoder.encode(observation)
        mask = encoder.encode_mask(observation)

        self.assertEqual((encoder.dimension,), encoded.shape)
        self.assertEqual(np.float32, encoded.dtype)
        self.assertEqual((len(ACTION_IDS),), mask.shape)
        self.assertTrue(mask.any())

    def test_encoder_does_not_leak_scenario_label(self) -> None:
        task = generate_benchmark(seed=5, variants_per_template=3)[0]
        observation, _ = OptimizationAgentEnv([task]).reset(task_id=task.task_id)
        altered = {**observation, "scenario": "不应被模型读取的答案"}
        encoder = RecoveryStateEncoder()

        np.testing.assert_array_equal(encoder.encode(observation), encoder.encode(altered))


@unittest.skipUnless(TORCH_AVAILABLE, "未安装可选 PyTorch RL 依赖")
class LearnedRecoveryPolicyTests(unittest.TestCase):
    """小规模训练一次，验证 BC+DQN 、mask 和 checkpoint。"""

    @classmethod
    def setUpClass(cls) -> None:
        import torch

        # 小网络单线程训练更快，也能降低 CI 调度造成的波动。
        torch.set_num_threads(1)
        cls.tasks = generate_benchmark(seed=7, variants_per_template=12)
        cls.config = DQNConfig(
            seed=7,
            train_episodes=150,
            bc_epochs=60,
            warmup_transitions=32,
            batch_size=16,
            target_sync_interval=40,
            epsilon_decay_steps=200,
        )
        cls.training = train_recovery_policy(cls.tasks, cls.config)

    def test_checkpoint_selection_uses_validation_and_preserves_bc_baseline(self) -> None:
        """所选 checkpoint 对应最优验证回报，BC 对照在预热结束时冻结。"""

        selection = self.training.report["training"]["selection"]
        self.assertEqual("validation", selection["split"])
        best = max(selection["history"], key=lambda row: row["average_return"])
        self.assertEqual(best["episode"], selection["selected_episode"])
        self.assertEqual(self.training.agent.optimization_steps, selection["selected_optimization_steps"])
        self.assertIn("bc_policy", self.training.report["evaluation"])
        self.assertIn("dqn_final_policy", self.training.report["evaluation"])
        self.assertEqual(0, self.training.bc_agent.optimization_steps)
        self.assertEqual(self.training.report["training"]["optimization_steps"], self.training.final_agent.optimization_steps)
        with tempfile.TemporaryDirectory() as directory:
            paths = self.training.save_comparisons(Path(directory), run_id="comparison-test")
            restored = MaskedDoubleDQNAgent.load(paths["dqn_final_checkpoint"])
            self.assertEqual(self.training.final_agent.optimization_steps, restored.optimization_steps)

    def test_behavior_cloning_reduces_teacher_action_loss(self) -> None:
        training = self.training.report["training"]

        self.assertGreater(training["bc_initial_loss"], training["bc_final_loss"])
        self.assertLess(training["bc_final_loss"], 0.01)
        self.assertGreater(training["optimization_steps"], 0)

    def test_learned_policy_generalizes_to_held_out_tasks(self) -> None:
        learned = self.training.report["evaluation"]["learned_policy"]["test"]
        random_valid = self.training.report["evaluation"]["random_valid_policy"]["test"]

        self.assertEqual(1.0, learned["recoverable_success_rate"])
        self.assertEqual(0.0, learned["invalid_action_rate"])
        self.assertGreater(learned["average_return"], random_valid["average_return"])

    def test_epsilon_exploration_never_selects_masked_action(self) -> None:
        environment = OptimizationAgentEnv(self.tasks)
        for task in self.tasks[:20]:
            observation, _ = environment.reset(task_id=task.task_id)
            mask = self.training.agent.state_encoder.encode_mask(observation)
            for _ in range(5):
                action_index = self.training.agent.select_action_index(observation, epsilon=1.0)
                self.assertTrue(mask[action_index])

    def test_checkpoint_roundtrip_preserves_policy(self) -> None:
        task = next(item for item in self.tasks if item.split == "test")
        observation, _ = OptimizationAgentEnv([task]).reset(task_id=task.task_id)
        expected_action = self.training.agent.policy(observation)
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "policy.pt"
            self.training.agent.save(checkpoint, metadata={"test": True})
            restored = MaskedDoubleDQNAgent.load(checkpoint)

        self.assertEqual(expected_action, restored.policy(observation))


if __name__ == "__main__":
    unittest.main()
