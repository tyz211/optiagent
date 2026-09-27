from __future__ import annotations

from copy import deepcopy
import hashlib
import json
from pathlib import Path
import tempfile
from types import SimpleNamespace
import unittest
from unittest.mock import patch

import numpy as np

from optiagent.instance_identity import audit_instance_splits, instance_fingerprint
from optiagent.recovery_runtime import RecoveryPolicyRuntime
from optiagent.rl.benchmark import TEMPLATE_IDS
from optiagent.rl.instances import generate_recovery_instance
from optiagent.rl.real_environment import collect_real_recovery_tasks
from optiagent.rl.state_encoder import CostAwareRecoveryStateEncoder
from optiagent.rl.trajectory_dataset import export_trajectory_dataset, read_database_episodes, split_for_instance


class ContentSplitTests(unittest.TestCase):
    """以实际数据而不是随机种子或任务 ID 验证训练隔离。"""

    def test_fingerprint_normalizes_rows_keys_and_numeric_types(self) -> None:
        first = {"capacity": 5, "items": [{"item": "A", "weight": 2}, {"item": "B", "weight": 3}]}
        second = {"items": [{"weight": 3.0, "item": "B"}, {"item": "A", "weight": 2.0}], "capacity": 5.0}
        self.assertEqual(instance_fingerprint("knapsack", first), instance_fingerprint("knapsack", second))
        second["capacity"] = 6
        self.assertNotEqual(instance_fingerprint("knapsack", first), instance_fingerprint("knapsack", second))
        with self.assertRaises(ValueError):
            instance_fingerprint("knapsack", {"capacity": float("nan")})

    def test_new_generator_has_unique_contents_across_splits_and_stable_coordinates(self) -> None:
        tasks = []
        for template in TEMPLATE_IDS:
            for split in ("train", "validation", "test"):
                for index in range(4):
                    instance = generate_recovery_instance(template, seed=55, split=split, index=index)
                    repeated = generate_recovery_instance(template, seed=55, split=split, index=index)
                    self.assertEqual(instance.data, repeated.data)
                    tasks.append(SimpleNamespace(template_id=template, split=split, instance_data=instance.data,
                                                 instance_fingerprint=instance_fingerprint(template, instance.data)))
        audit = audit_instance_splits(tasks)
        self.assertEqual(72, audit["unique_instance_count"])
        self.assertEqual({"train": 24, "validation": 24, "test": 24}, audit["unique_instances_by_split"])
        duplicated = deepcopy(tasks[0])
        duplicated.split = "test"
        with self.assertRaisesRegex(ValueError, "跨集合"):
            audit_instance_splits(tasks + [duplicated])
        tasks[0].instance_data["extra"] = 123
        with self.assertRaisesRegex(ValueError, "不一致"):
            audit_instance_splits(tasks)

    def test_fault_variants_share_content_and_stay_in_one_split(self) -> None:
        tasks = collect_real_recovery_tasks(seed=31, template_ids=("knapsack",))
        self.assertEqual(15, len(tasks))
        self.assertEqual(3, audit_instance_splits(tasks)["unique_instance_count"])
        for split in ("train", "validation", "test"):
            variants = [item for item in tasks if item.split == split]
            self.assertEqual(1, len({item.instance_fingerprint for item in variants}))
            self.assertEqual(1, len({item.solver_latency_ms for item in variants}))


class TrajectoryExportTests(unittest.TestCase):
    """确保导出保留失败、排除泄露字段，并拒绝缺失或伪造的恢复决策。"""

    def test_export_keeps_failure_and_preserves_encoded_state(self) -> None:
        success, failure = _episode("success"), _episode("failure", failed=True)
        with tempfile.TemporaryDirectory() as directory:
            output = Path(directory) / "dataset"
            report = export_trajectory_dataset([success, failure], output)
            content = "".join(path.read_text() for path in output.iterdir())
            rows = [json.loads(line) for path in output.glob("*.jsonl") if path.name != "quarantine.jsonl" for line in path.read_text().splitlines()]
            self.assertEqual(2, report["accepted_episodes"])
            self.assertEqual(1, report["outcomes"]["terminated_failure"])
            self.assertEqual(1, len({row["split"] for row in rows}))
            self.assertNotIn("sk-do-not-export-this-key", content)
            self.assertNotIn("私人问题", content)
            self.assertNotIn("敏感异常", content)
            exported = next(row for row in rows if row["outcome"] == "verified_success")
            encoder = CostAwareRecoveryStateEncoder()
            np.testing.assert_array_equal(encoder.encode(success["decision_transitions"][0]["policy_observation"]),
                                          encoder.encode(exported["transitions"][0]["state"]))
            with self.assertRaises(FileExistsError):
                export_trajectory_dataset([], output)

    def test_quarantine_rejects_mask_missing_state_and_source_mixing(self) -> None:
        cases = []
        for name in ("mask", "state", "source", "reward", "unfinished"):
            episode = _episode(name)
            row = episode["decision_transitions"][0]
            if name == "mask":
                row["action"] = "retry_solver"
            elif name == "state":
                row["policy_observation"] = None
            elif name == "source":
                episode["trajectory_source"] = "controlled_workflow_benchmark"
            elif name == "reward":
                row["reward"] = float("nan")
            else:
                episode["status"] = "running"
            cases.append(episode)
        with tempfile.TemporaryDirectory() as directory:
            manifest = export_trajectory_dataset(cases, Path(directory) / "dataset")
        self.assertEqual(0, manifest["accepted_episodes"])
        self.assertEqual(5, manifest["rejected_episodes"])
        self.assertFalse(manifest["all_splits_nonempty"])

    def test_split_does_not_depend_on_episode_id_or_order(self) -> None:
        fingerprint = _episode("a")["task"]["instance_fingerprint"]
        self.assertEqual(split_for_instance(fingerprint, 4), split_for_instance(fingerprint, 4))
        episodes = [_episode("a"), _episode("b")]
        with tempfile.TemporaryDirectory() as directory:
            first = export_trajectory_dataset(episodes, Path(directory) / "first", seed=4)
            second = export_trajectory_dataset(reversed(episodes), Path(directory) / "second", seed=4)
        self.assertEqual(first["unique_instances_by_split"], second["unique_instances_by_split"])

    def test_retry_then_exception_is_exported_as_failed_action_transition(self) -> None:
        """后续工具抛错也要保留已经执行的恢复动作及其负奖励。"""

        import api.database as database
        from api.services import agent_workflow as workflow

        calls = []

        def solver(state: dict) -> dict:
            calls.append(1)
            if len(calls) > 1:
                raise TimeoutError("敏感异常，不应进入导出文件")
            return {"result": {"answer": "暂未得到解", "structured_answer": {}, "status": "ERROR", "solution_verification": None}}

        question = '请进行背包求解，最大化价值。数据：{"capacity":5,"items":[{"item":"A","value":8,"weight":3}]}'
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "source.sqlite3"
            with patch.object(database, "DB_PATH", path):
                database.init_db()
                database.create_agent_episode("其他用户", 999, None)
                with patch.object(workflow, "AGENT_GRAPH", workflow._build_agent_graph({"solver": solver})):
                    with self.assertRaises(TimeoutError):
                        workflow.run_agent_workflow(question=question, requested_dataset_id=None, mcp_config="", user_id=None,
                                                    conversation_id=None, recovery_checkpoint="",
                                                    trajectory_source="controlled_workflow_benchmark")
            before = hashlib.sha256(path.read_bytes()).hexdigest()
            episodes = list(read_database_episodes(path, user_id=None))
            self.assertEqual(before, hashlib.sha256(path.read_bytes()).hexdigest())
            self.assertEqual(1, len(episodes))
            self.assertEqual("execution_failed", episodes[0]["outcome"])
            manifest = export_trajectory_dataset(episodes, Path(directory) / "export", source="controlled_workflow_benchmark")
            self.assertEqual(1, manifest["accepted_episodes"])
            self.assertEqual({"execution_failed": 1}, manifest["outcomes"])
            rows = [json.loads(line) for file in (Path(directory) / "export").glob("*.jsonl") for line in file.read_text().splitlines()]
            self.assertEqual("retry_solver", rows[0]["transitions"][-1]["action"])
            self.assertEqual(-1.0, rows[0]["transitions"][-1]["reward"])
            self.assertTrue(rows[0]["transitions"][-1]["done"])


def _episode(identifier: str, *, failed: bool = False) -> dict:
    """生成符合实际状态合同的终局决策，夹带文本以验证导出白名单。"""

    state = {"question": '私人问题 {"capacity": 5, "items": [{"value": 8, "weight": 3}]}',
             "data_context": {"source_type": "inline_json"}, "problem_spec": {"template_id": "knapsack"},
             "model_attempt": 2, "solver_attempt": 2,
             "verification": {"passed": not failed, "errors": ["敏感异常"] if failed else [],
                              "mathematical": {"passed": not failed, "verifiable": True, "feasible": not failed, "objective_consistent": True}}}
    decision = RecoveryPolicyRuntime().decide(state).model_dump(mode="json")
    observation = decision["metadata"]["observation"]
    observation["api_key"] = "sk-do-not-export-this-key"
    return {"episode_id": identifier, "trajectory_source": "live_workflow", "status": "completed",
            "task": {"template_id": "knapsack", "question": "私人问题", "instance_fingerprint": observation["instance_fingerprint"]},
            "total_reward": -1.0 if failed else 1.0,
            "decision_transitions": [{"policy_observation": observation, "next_policy_observation": None,
                                      "candidate_actions": decision["candidate_ids"], "action_mask": decision["action_mask"],
                                      "action": decision["selected_action"], "done": True, "status": "completed",
                                      "reward": -1.0 if failed else 1.0,
                                      "policy": {"name": decision["policy_name"], "version": decision["policy_version"]}}]}
