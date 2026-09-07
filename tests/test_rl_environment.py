from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from optiagent.rl import (
    ACTION_IDS,
    OptimizationAgentEnv,
    evaluate_policy,
    export_rollouts_jsonl,
    generate_benchmark,
)


class RLEnvironmentTests(unittest.TestCase):
    """验证 RL 环境动作合同、故障转移与数据导出。"""

    def setUp(self) -> None:
        self.tasks = generate_benchmark(seed=7, variants_per_template=12)

    def test_benchmark_is_reproducible_and_stratified(self) -> None:
        repeated = generate_benchmark(seed=7, variants_per_template=12)
        changed = generate_benchmark(seed=8, variants_per_template=12)

        self.assertEqual(self.tasks, repeated)
        self.assertNotEqual(self.tasks, changed)
        for template_id in {item.template_id for item in self.tasks}:
            splits = {item.split for item in self.tasks if item.template_id == template_id}
            self.assertEqual({"train", "validation", "test"}, splits)
        minimal = generate_benchmark(seed=7, variants_per_template=3)
        for template_id in {item.template_id for item in minimal}:
            splits = {item.split for item in minimal if item.template_id == template_id}
            self.assertEqual({"train", "validation", "test"}, splits)

    def test_direct_success_accepts_verified_solution(self) -> None:
        task = self._task("direct_success")
        environment = OptimizationAgentEnv([task])
        observation, _ = environment.reset(task_id=task.task_id)

        self.assertEqual([True, False, False, False], observation["action_mask"])
        _, reward, terminated, truncated, info = environment.step("accept_solution")

        self.assertEqual(1.0, reward)
        self.assertTrue(terminated)
        self.assertFalse(truncated)
        self.assertTrue(info["success"])

    def test_transient_solver_failure_requires_retry(self) -> None:
        task = self._task("transient_solver_failure")
        environment = OptimizationAgentEnv([task])
        observation, _ = environment.reset(task_id=task.task_id)

        self.assertTrue(observation["action_mask"][ACTION_IDS.index("retry_solver")])
        recovered, reward, terminated, truncated, _ = environment.step("retry_solver")

        self.assertEqual(-0.05, reward)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertTrue(recovered["verification"]["passed"])
        self.assertEqual(2, recovered["solver_attempt"])

    def test_model_error_requires_rebuild(self) -> None:
        task = self._task("repairable_model_error")
        environment = OptimizationAgentEnv([task])
        observation, _ = environment.reset(task_id=task.task_id)

        self.assertTrue(observation["action_mask"][ACTION_IDS.index("rebuild_model")])
        recovered, _, _, _, _ = environment.step("rebuild_model")

        self.assertTrue(recovered["verification"]["passed"])
        self.assertEqual(2, recovered["model_attempt"])
        self.assertEqual(2, recovered["solver_attempt"])

    def test_masked_action_ends_episode_with_penalty(self) -> None:
        task = self._task("direct_success")
        environment = OptimizationAgentEnv([task])
        environment.reset(task_id=task.task_id)

        _, reward, terminated, _, info = environment.step("retry_solver")

        self.assertEqual(-1.0, reward)
        self.assertTrue(terminated)
        self.assertEqual("invalid_action", info["terminal_reason"])

    def test_baseline_recovers_all_recoverable_tasks(self) -> None:
        result = evaluate_policy(self.tasks, seed=7)

        self.assertEqual(0.0, result["metrics"]["invalid_action_rate"])
        self.assertEqual(1.0, result["metrics"]["recoverable_success_rate"])
        persistent = [item for item in result["episodes"] if not item["recoverable"]]
        self.assertTrue(all(item["terminal_reason"] == "recovery_budget_exhausted" for item in persistent))

    def test_jsonl_export_contains_one_replayable_episode_per_line(self) -> None:
        selected = self.tasks[:4]
        result = evaluate_policy(selected, seed=7)
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = export_rollouts_jsonl(result, Path(temporary_directory) / "rollouts.jsonl")
            rows = [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines()]

        self.assertEqual(len(selected), len(rows))
        self.assertTrue(all(item["transitions"] for item in rows))
        self.assertTrue(all("action_mask" in item["transitions"][0] for item in rows))

    def _task(self, scenario: str):
        """从固定 benchmark 中选取指定故障场景。"""

        return next(item for item in self.tasks if item.scenario == scenario)


if __name__ == "__main__":
    unittest.main()
