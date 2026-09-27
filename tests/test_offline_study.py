from __future__ import annotations

from collections import Counter
from copy import deepcopy
from types import SimpleNamespace
import unittest

from optiagent.instance_identity import instance_fingerprint
from optiagent.rl.study import SCENARIOS, audit_dataset_against_plan, build_study_plan, paired_comparison, summarize_study, workflow_instance_data
from optiagent.rl.instances import generate_recovery_instance
from optiagent.rl.trajectory_dataset import split_for_instance


class OfflineStudyTests(unittest.TestCase):
    """验证分组先于评测、独立实例配对和多种子验收规则。"""

    def test_facility_default_columns_share_execution_fingerprint(self):
        instance = generate_recovery_instance("facility_location", seed=1, split="test", index=0)
        explicit = deepcopy(instance.data)
        for row in explicit["warehouses"]:
            row.update(min_open_ratio=0.0, force_open=0, force_closed=0)
        # 原始数据是否显式写默认值，不应影响最终执行实例的身份。
        left = workflow_instance_data(instance.template_id, instance.data)
        right = workflow_instance_data(instance.template_id, explicit)
        self.assertEqual(instance_fingerprint(instance.template_id, left), instance_fingerprint(instance.template_id, right))

    def test_frozen_plan_has_exact_template_quotas_and_no_content_overlap(self):
        kwargs = dict(seed=70, split_seed=71, train_count=3, validation_count=2, test_count=2, external_count=2)
        plan = build_study_plan(**kwargs)
        self.assertEqual(plan, build_study_plan(**kwargs))
        fingerprints = [row["fingerprint"] for row in plan["instances"]]
        self.assertEqual(len(fingerprints), len(set(fingerprints)))
        counts = Counter((row["group"], row["instance"]["template_id"]) for row in plan["instances"])
        for row in plan["instances"]:
            self.assertEqual(row["fingerprint"], instance_fingerprint(row["instance"]["template_id"], row["instance"]["data"]))
            self.assertEqual(plan["quotas_per_template"][row["group"]], counts[row["group"], row["instance"]["template_id"]])
            if row["group"] != "external":
                self.assertEqual(row["group"], split_for_instance(row["fingerprint"], 71))
        excluded = {fingerprints[0], fingerprints[-1]}
        new_plan = build_study_plan(**kwargs, excluded=excluded)
        self.assertFalse(excluded & {row["fingerprint"] for row in new_plan["instances"]})

    def test_missing_template_coverage_rejected_even_when_splits_nonempty(self):
        plan = build_study_plan(seed=70, split_seed=71, train_count=1, validation_count=1, test_count=1, external_count=1)
        splits = {name: [{"template_id": row["instance"]["template_id"], "instance_fingerprint": row["fingerprint"]}
                         for row in plan["instances"] if row["group"] == name] for name in ("train", "validation", "test")}
        dataset = SimpleNamespace(splits=splits, manifest={"rejected_episodes": 0})
        self.assertEqual(6, len(audit_dataset_against_plan(dataset, plan)["test"]))
        dataset.splits["test"].pop()
        with self.assertRaisesRegex(ValueError, "冻结"):
            audit_dataset_against_plan(dataset, plan)

    def test_paired_statistics_count_instances_not_fault_variants(self):
        baseline = _cases()
        candidate = deepcopy(baseline)
        for row in candidate:
            if row["instance_fingerprint"] == "a":
                row["model_calls"] -= 1
        report = paired_comparison(baseline, candidate)
        self.assertEqual(2, report["instance_count"])
        self.assertEqual(10, report["case_count"])
        self.assertEqual(-0.5, report["candidate_minus_reference"]["model_calls"]["mean_delta"])
        self.assertEqual(1, report["candidate_minus_reference"]["model_calls"]["improved_instances"])

    def test_unpaired_missing_or_duplicate_cases_are_rejected(self):
        baseline = _cases()
        altered = deepcopy(baseline)
        for row in altered:
            if row["instance_fingerprint"] == "a":
                row["instance_fingerprint"] = "c"
        for candidate in (altered, baseline[:-1], baseline + baseline[:1]):
            with self.assertRaises(ValueError):
                paired_comparison(baseline, candidate)

    def test_one_regressed_seed_blocks_gate_even_if_average_improves(self):
        metric = {"recoverable_success_rate": 1.0, "average_model_calls": 1.4, "average_solver_calls": 1.8,
                  "invalid_case_count": 0, "fallback_count": 0, "unverified_accept_count": 0}
        runs = [{"seed": seed, "metrics": {name: dict(metric) for name in
                 ("rule_policy", "bc_policy", "cql_final_policy", "selected_policy")}} for seed in range(5)]
        for run in runs:
            run["metrics"]["selected_policy"]["average_model_calls"] = 1.2
        self.assertTrue(summarize_study(runs)["controlled_benchmark_gate_passed"])
        self.assertFalse(summarize_study(runs[:1])["controlled_benchmark_gate_passed"])
        runs[0]["metrics"]["selected_policy"]["recoverable_success_rate"] = 0.99
        self.assertFalse(summarize_study(runs)["controlled_benchmark_gate_passed"])
        runs[0]["metrics"]["selected_policy"]["recoverable_success_rate"] = 1.0
        runs[0]["metrics"]["selected_policy"]["fallback_count"] = 1
        self.assertFalse(summarize_study(runs)["controlled_benchmark_gate_passed"])


def _cases():
    """两个独立实例各包含完整五种受控场景。"""

    return [{"instance_fingerprint": fingerprint, "scenario": scenario, "success": scenario != "persistent_failure",
             "model_calls": 2, "solver_calls": 2, "estimated_recovery_cost": 0.2, "elapsed_seconds": 0.1}
            for fingerprint in ("a", "b") for scenario in sorted(SCENARIOS)]


if __name__ == "__main__":
    unittest.main()
