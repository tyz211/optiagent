from __future__ import annotations

from copy import deepcopy
import unittest

from optiagent.rl.data_repair import DataRepairHarness, corrupt_mapping
from optiagent.rl.e2e_benchmark import build_smoke_instances


class DataRepairTests(unittest.TestCase):
    """必须真的求解错误输入并独立复算；重复求解不能自动清除持续映射错误。"""

    def test_mapping_corruption_does_not_mutate_authoritative_source(self):
        for instance in build_smoke_instances():
            before = deepcopy(instance.data)
            changed = corrupt_mapping(instance.template_id, instance.data)
            self.assertNotEqual(before, changed)
            self.assertEqual(before, instance.data)

    def test_all_templates_detect_wrong_mapping_and_verify_restored_source(self):
        for instance in build_smoke_instances():
            with self.subTest(template=instance.template_id):
                harness = DataRepairHarness(instance, "model_error")
                harness.bound = deepcopy(harness.corrupted)
                failed = harness.solver({"solver_attempt": 0})["result"]["solution_verification"]
                retried = harness.solver({"solver_attempt": 1})["result"]["solution_verification"]
                self.assertFalse(failed["passed"])
                self.assertFalse(retried["passed"])
                harness.bound = deepcopy(harness.source)
                repaired = harness.solver({"solver_attempt": 1})["result"]["solution_verification"]
                self.assertTrue(repaired["passed"])
                self.assertNotEqual(harness.events[0]["input"], harness.events[-1]["input"])

    def test_transient_wrong_mapping_recovers_on_retry(self):
        harness = DataRepairHarness(build_smoke_instances()[0], "tampered_objective")
        self.assertFalse(harness.solver({"solver_attempt": 0})["result"]["solution_verification"]["passed"])
        self.assertTrue(harness.solver({"solver_attempt": 1})["result"]["solution_verification"]["passed"])

    def test_rebuild_consumes_feedback_and_persistent_source_remains_corrupt(self):
        from unittest.mock import patch
        for scenario, restored in (("model_error", True), ("persistent_failure", False)):
            harness = DataRepairHarness(build_smoke_instances()[0], scenario)
            with patch("api.services.agent_workflow._modeler_node", return_value={"problem_spec": {}}):
                harness.modeler({})
                self.assertEqual(harness.corrupted, harness.bound)
                harness.modeler({"recovery_feedback": ["独立验算未通过"]})
            self.assertEqual(restored, harness.bound == harness.source)


if __name__ == "__main__":
    unittest.main()
