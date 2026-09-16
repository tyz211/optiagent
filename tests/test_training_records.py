from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest

from optiagent.rl.experiment_log import TrainingRunRecorder


class TrainingRunRecorderTests(unittest.TestCase):
    """验证每次训练都生成独立记录，且不会把凭据写入产物。"""

    def test_completed_run_writes_manifest_index_and_latest_pointer(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recorder = TrainingRunRecorder(
                temporary_directory,
                seed=43,
                algorithm="test_algorithm",
                parameters={"seed": 43, "api_key": "sk-should-never-be-written"},
                run_id="test_run",
            )
            recorder.paths.checkpoint.write_bytes(b"checkpoint")
            recorder.paths.report.write_text("{}", encoding="utf-8")
            recorder.complete(_fake_report())

            manifest_text = recorder.paths.manifest.read_text(encoding="utf-8")
            manifest = json.loads(manifest_text)
            index_records = [json.loads(line) for line in recorder.paths.index.read_text(encoding="utf-8").splitlines()]
            latest = json.loads(recorder.paths.latest.read_text(encoding="utf-8"))

        self.assertEqual("completed", manifest["status"])
        self.assertNotIn("api_key", manifest_text)
        self.assertNotIn("sk-should-never-be-written", manifest_text)
        self.assertEqual("test_run", index_records[0]["run_id"])
        self.assertEqual("test_run", latest["run_id"])
        self.assertTrue(manifest["artifacts"]["checkpoint"]["sha256"])

    def test_failed_run_is_also_appended_to_history(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            recorder = TrainingRunRecorder(
                temporary_directory,
                seed=44,
                algorithm="test_algorithm",
                parameters={"seed": 44},
                run_id="failed_run",
            )
            recorder.fail(RuntimeError("模拟训练失败"))
            manifest = json.loads(recorder.paths.manifest.read_text(encoding="utf-8"))
            index_record = json.loads(recorder.paths.index.read_text(encoding="utf-8").strip())

        self.assertEqual("failed", manifest["status"])
        self.assertEqual("RuntimeError", manifest["error"]["type"])
        self.assertEqual("failed", index_record["status"])

    def test_duplicate_run_id_cannot_overwrite_existing_record(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            TrainingRunRecorder(
                temporary_directory,
                seed=45,
                algorithm="test_algorithm",
                parameters={"seed": 45},
                run_id="same_run",
            )
            with self.assertRaises(FileExistsError):
                TrainingRunRecorder(
                    temporary_directory,
                    seed=45,
                    algorithm="test_algorithm",
                    parameters={"seed": 45},
                    run_id="same_run",
                )


def _fake_report() -> dict:
    """构造不依赖 PyTorch 的最小训练报告。"""

    return {
        "training": {
            "demonstration_count": 10,
            "environment_steps": 20,
            "optimization_steps": 12,
            "bc_final_loss": 0.1,
            "dqn_final_loss": 0.2,
        },
        "evaluation": {
            "learned_policy": {"test": {"success_rate": 1.0}},
            "rule_policy": {"test": {"success_rate": 1.0}},
        },
    }


if __name__ == "__main__":
    unittest.main()
