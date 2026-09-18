from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import random
import tempfile
import unittest

from optiagent.rl import OptimizationAgentEnv, evaluate_policy, generate_benchmark
from optiagent.rl.q_learning import QLearningPolicy, state_key
from scripts.train_recovery_policy import train


class QLearningTests(unittest.TestCase):
    def setUp(self):
        self.tasks = generate_benchmark(seed=42)
        self.env = OptimizationAgentEnv(self.tasks)

    def observation(self, scenario):
        task = next(t for t in self.tasks if t.scenario == scenario)
        return self.env.reset(task_id=task.task_id)[0]

    def test_terminal_update_does_not_bootstrap(self):
        state = self.observation("direct_success")
        policy = QLearningPolicy(alpha=1)
        policy.q_table[state_key(state)] = [100.0] * 4
        policy.update(state, "accept_solution", 1.0, state, True)
        self.assertEqual(1.0, policy.values(state)[0])

    def test_bootstrap_and_selection_ignore_masked_actions(self):
        state = self.observation("transient_solver_failure")
        next_state, reward, _, _, _ = self.env.step("retry_solver")
        policy = QLearningPolicy(alpha=1)
        policy.q_table[state_key(next_state)] = [1.0, 1000.0, 1000.0, 1000.0]
        policy.update(state, "retry_solver", reward, next_state, False)
        self.assertAlmostEqual(0.95, policy.values(state)[1])
        self.assertEqual("accept_solution", policy(next_state))
        rng = random.Random(1)
        self.assertTrue(all(policy.explore(next_state, 1.0, rng) == "accept_solution" for _ in range(50)))

    def test_identity_and_hidden_labels_cannot_change_policy(self):
        state = self.observation("repairable_model_error")
        other = deepcopy(state)
        other.update(task_id="unseen", split="test", scenario="direct_success", template_id="other", features={})
        self.assertEqual(state_key(state), state_key(other))

    def test_training_learns_without_using_test_data_and_roundtrips(self):
        training = [t for t in self.tasks if t.split == "train"]
        validation = [t for t in self.tasks if t.split == "validation"]
        testing = [t for t in self.tasks if t.split == "test"]
        with tempfile.TemporaryDirectory() as tmp:
            kwargs = dict(seed=11, episodes=1000, interval=100)
            first, _, updates = train(training, validation, output=Path(tmp) / "a", **kwargs)
            second, _, _ = train(training, validation, output=Path(tmp) / "b", **kwargs)
            self.assertGreater(updates, 1000)
            self.assertEqual(first.q_table, second.q_table)
            before = deepcopy(first.q_table)
            metrics = evaluate_policy(testing, first)["metrics"]
            self.assertEqual(1.0, metrics["recoverable_success_rate"])
            self.assertGreater(metrics["average_return"], evaluate_policy(testing, QLearningPolicy())["metrics"]["average_return"])
            self.assertEqual(before, first.q_table)
            first.save(Path(tmp) / "saved.json")
            loaded = QLearningPolicy.load(Path(tmp) / "saved.json")
            self.assertEqual(first.q_table, loaded.q_table)
            with self.assertRaises(ValueError):
                train(testing, validation, output=Path(tmp) / "bad", **kwargs)

    def test_continuation_preserves_parent_and_accumulates_updates(self):
        training = [t for t in self.tasks if t.split == "train"]
        validation = [t for t in self.tasks if t.split == "validation"]
        with tempfile.TemporaryDirectory() as tmp:
            parent_path = Path(tmp) / "parent.json"
            parent = QLearningPolicy()
            parent.q_table["sentinel-unvisited-state"] = [0.1, 0.2, 0.3, 0.4]
            parent.updates = 123
            parent.save(parent_path, metadata={"seed": 11, "episode": 100})
            digest = hashlib.sha256(parent_path.read_bytes()).hexdigest()
            output = Path(tmp) / "continued"
            train(training, validation, seed=11, episodes=20, interval=10,
                  output=output, resume=parent_path)
            continued = QLearningPolicy.load(output / "last_policy.json")
            info = json.loads((output / "continuation.json").read_text())
            self.assertEqual(parent.q_table["sentinel-unvisited-state"], continued.q_table["sentinel-unvisited-state"])
            self.assertEqual(120, info["cumulative_episodes"])
            self.assertGreaterEqual(continued.updates, 143)
            self.assertEqual(continued.updates - 123, info["additional_updates"])
            self.assertEqual(digest, info["parent_sha256"])
            self.assertEqual(digest, hashlib.sha256(parent_path.read_bytes()).hexdigest())


if __name__ == "__main__":
    unittest.main()
