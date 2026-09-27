from __future__ import annotations

from copy import deepcopy
from dataclasses import replace
import hashlib
import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

import torch

from optiagent.recovery_runtime import RecoveryPolicyRuntime
from optiagent.rl.dqn import DQNConfig, MaskedDoubleDQNAgent
from optiagent.rl.offline_dataset import OfflineDataset, load_offline_dataset, transition_reward
from optiagent.rl.offline_training import OfflineConfig, conservative_penalty, double_dqn_targets, encode_episodes, train_offline_policy
from optiagent.rl.state_encoder import CostAwareRecoveryStateEncoder
from optiagent.rl.trajectory_dataset import export_trajectory_dataset


class OfflineDatasetTests(unittest.TestCase):
    """离线消费端必须独立拒绝损坏文件与伪造合同，而不是只相信 manifest。"""

    def setUp(self) -> None:
        self.temporary = tempfile.TemporaryDirectory()
        self.path = Path(self.temporary.name) / "dataset"
        export_trajectory_dataset(_episodes(), self.path, seed=57, source="controlled_workflow_benchmark")

    def tearDown(self) -> None:
        self.temporary.cleanup()

    def test_loader_accepts_complete_split_and_detects_file_tampering(self) -> None:
        dataset = load_offline_dataset(self.path)
        self.assertTrue(all(dataset.splits.values()))
        with (self.path / "train.jsonl").open("a") as stream:
            stream.write("\n")
        with self.assertRaisesRegex(ValueError, "摘要"):
            load_offline_dataset(self.path)

    def test_rehashed_invalid_action_is_rejected(self) -> None:
        rows = _read_rows(self.path / "train.jsonl")
        rows[0]["transitions"][0]["action"] = "rebuild_model"
        _replace_rows_and_hash(self.path, "train", rows)
        with self.assertRaises(ValueError):
            load_offline_dataset(self.path)

    def test_cross_split_copy_is_rejected_even_with_updated_file_hash(self) -> None:
        row = deepcopy(_read_rows(self.path / "train.jsonl")[0])
        row["split"] = "test"
        row["episode_ref"] = "f" * 64
        _replace_rows_and_hash(self.path, "test", _read_rows(self.path / "test.jsonl") + [row])
        with self.assertRaisesRegex(ValueError, "分组"):
            load_offline_dataset(self.path)

    def test_failed_outcome_reward_cannot_become_positive_from_format_bonus(self) -> None:
        dataset = load_offline_dataset(self.path)
        episode = next(row for rows in dataset.splits.values() for row in rows if row["outcome"] == "terminated_failure")
        transition = deepcopy(episode["transitions"][-1])
        transition["reward"] = 0.08
        self.assertEqual(0.08, transition_reward(episode, transition, "sparse"))
        self.assertEqual(-1.0, transition_reward(episode, transition, "verified_cost"))


class OfflineLearningTests(unittest.TestCase):
    """检查保守项、自举边界、测试集隔离和 checkpoint 推理兼容性。"""

    @classmethod
    def setUpClass(cls) -> None:
        torch.set_num_threads(1)
        cls.temporary = tempfile.TemporaryDirectory()
        directory = Path(cls.temporary.name) / "data"
        export_trajectory_dataset(_episodes(), directory, seed=57, source="controlled_workflow_benchmark")
        cls.dataset = load_offline_dataset(directory)

    @classmethod
    def tearDownClass(cls) -> None:
        cls.temporary.cleanup()

    def test_cql_penalty_ignores_illegal_q_and_pushes_down_unobserved_legal_actions(self) -> None:
        q = torch.tensor([[1.0, 3.0, 9999.0, -2.0]], requires_grad=True)
        mask = torch.tensor([[True, True, False, False]])
        penalty = conservative_penalty(q, torch.tensor([0]), mask)
        expected = torch.logsumexp(torch.tensor([1.0, 3.0]), 0) - 1.0
        self.assertAlmostEqual(float(expected), float(penalty.detach()), places=6)
        penalty.backward()
        self.assertLess(float(q.grad[0, 0]), 0)
        self.assertGreater(float(q.grad[0, 1]), 0)
        self.assertEqual(0.0, float(q.grad[0, 2]))

    def test_terminal_states_never_bootstrap(self) -> None:
        agent = MaskedDoubleDQNAgent(CostAwareRecoveryStateEncoder(), DQNConfig())
        batch = {"rewards": torch.tensor([-1.0, 1.0]), "dones": torch.tensor([True, True]),
                 "next_states": torch.full((2, 35), float("nan")), "next_masks": torch.zeros((2, 4), dtype=torch.bool)}
        with patch.object(agent.policy_network, "forward", side_effect=AssertionError("终局不能读取下一状态")):
            torch.testing.assert_close(batch["rewards"], double_dqn_targets(agent, batch, 0.95))

    def test_bootstrap_selects_legal_online_action_and_uses_target_value(self) -> None:
        """非法动作即使分数最大也不能进入 Bellman 目标。"""

        agent = MaskedDoubleDQNAgent(CostAwareRecoveryStateEncoder(), DQNConfig())
        batch = {"rewards": torch.tensor([-0.1]), "dones": torch.tensor([False]),
                 "next_states": torch.zeros((1, 35)), "next_masks": torch.tensor([[True, True, False, True]])}
        with patch.object(agent.policy_network, "forward", return_value=torch.tensor([[0., 5., 1000., 3.]])), \
             patch.object(agent.target_network, "forward", return_value=torch.tensor([[10., 20., 9000., 30.]])):
            self.assertAlmostEqual(-0.1 + 0.95 * 20, float(double_dqn_targets(agent, batch, 0.95)[0]), places=5)

    def test_test_rewards_do_not_change_training_or_model_selection(self) -> None:
        config = OfflineConfig(gradient_steps=40, bc_epochs=10, validation_interval=10, reward_mode="sparse")
        # 修改测试奖励后重新运行，训练与验证必须逐位相同。
        altered_splits = deepcopy(self.dataset.splits)
        for episode in altered_splits["test"]:
            episode["transitions"][-1]["reward"] = -0.3
        altered = OfflineDataset(altered_splits, self.dataset.manifest, self.dataset.manifest_sha256)
        with patch("optiagent.rl.training.train_masked_dqn_core", side_effect=AssertionError("离线训练不能进入环境交互内核")):
            first = train_offline_policy(self.dataset, config)
            second = train_offline_policy(altered, config)
        self.assertEqual(first.report["training"]["selection"], second.report["training"]["selection"])
        self.assertEqual(0, first.report["training"]["environment_steps"])
        self.assertEqual(40, first.final_agent.optimization_steps)
        for name, weights in first.agent.policy_network.state_dict().items():
            torch.testing.assert_close(weights, second.agent.policy_network.state_dict()[name], rtol=0, atol=0)
        self.assertNotEqual(first.report["evaluation"]["learned_policy"]["test"]["logged_return_mse"],
                            second.report["evaluation"]["learned_policy"]["test"]["logged_return_mse"])
        path = Path(self.temporary.name) / "offline.pt"
        first.agent.save(path, metadata={"stage": "offline_cql"})
        loaded = MaskedDoubleDQNAgent.load(path)
        observation = self.dataset.splits["test"][0]["transitions"][0]["state"]
        self.assertEqual(first.agent.policy(observation), loaded.policy(observation))

    def test_invalid_hyperparameters_fail_before_training(self) -> None:
        for config in (replace(OfflineConfig(), cql_alpha=float("nan")), replace(OfflineConfig(), gradient_steps=-1),
                       replace(OfflineConfig(), reward_mode="unknown")):
            with self.assertRaises(ValueError):
                train_offline_policy(self.dataset, config)


def _episodes() -> list[dict]:
    """生成不同内容的合法终局样本，覆盖成功/失败并自然形成三个哈希集合。"""

    episodes = []
    for index in range(40):
        failed = index % 2 == 1
        state = {"question": json.dumps({"capacity": 5 + index, "items": [{"weight": 3, "value": 8}]}),
                 "data_context": {"source_type": "inline_json"}, "problem_spec": {"template_id": "knapsack"},
                 "model_attempt": 2, "solver_attempt": 2,
                 "verification": {"passed": not failed, "errors": [],
                                  "mathematical": {"passed": not failed, "verifiable": True, "feasible": not failed,
                                                   "objective_consistent": True}}}
        decision = RecoveryPolicyRuntime().decide(state).model_dump(mode="json")
        observation = decision["metadata"]["observation"]
        reward = -1.0 if failed else 1.0
        episodes.append({"episode_id": str(index), "trajectory_source": "controlled_workflow_benchmark", "status": "completed",
                         "task": {"template_id": "knapsack", "instance_fingerprint": observation["instance_fingerprint"]},
                         "total_reward": reward,
                         "decision_transitions": [{"policy_observation": observation, "next_policy_observation": None,
                                                   "candidate_actions": decision["candidate_ids"], "action_mask": decision["action_mask"],
                                                   "action": decision["selected_action"], "reward": reward, "done": True, "status": "completed",
                                                   "policy": {"name": decision["policy_name"], "version": decision["policy_version"]}}]})
    return episodes


def _read_rows(path: Path) -> list[dict]:
    """读取测试数据以制造需要被消费端拒绝的破坏。"""

    return [json.loads(line) for line in path.read_text().splitlines()]


def _replace_rows_and_hash(directory: Path, split: str, rows: list[dict]) -> None:
    """同步修改摘要以确认内容校验独立于文件校验。"""

    path = directory / f"{split}.jsonl"
    path.write_text("".join(json.dumps(row) + "\n" for row in rows))
    manifest_path = directory / "manifest.json"
    manifest = json.loads(manifest_path.read_text())
    manifest["files"][path.name] = {"bytes": path.stat().st_size, "sha256": hashlib.sha256(path.read_bytes()).hexdigest()}
    manifest_path.write_text(json.dumps(manifest))
