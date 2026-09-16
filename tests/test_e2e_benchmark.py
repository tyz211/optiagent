from __future__ import annotations

import unittest

from optiagent.rl.e2e_benchmark import build_smoke_instances, run_end_to_end_benchmark


class EndToEndBenchmarkTests(unittest.TestCase):
    """验证六类问题的真实 Gateway/Solver/Verifier 测试矩阵。"""

    @classmethod
    def setUpClass(cls) -> None:
        # 全部断言共用一次真实求解，避免重复消耗 Gurobi 许可证和时间。
        cls.result = run_end_to_end_benchmark(seed=42)

    def test_smoke_instances_cover_six_templates(self) -> None:
        instances = build_smoke_instances(seed=42)

        self.assertEqual(6, len(instances))
        self.assertEqual(
            {
                "knapsack",
                "assignment",
                "tsp",
                "job_shop_scheduling",
                "production_mix",
                "facility_location",
            },
            {item.template_id for item in instances},
        )

    def test_clean_cases_return_verified_solutions(self) -> None:
        clean = [item for item in self.result["cases"] if item["fault"] == "clean"]

        self.assertEqual(6, len(clean))
        self.assertTrue(all(item["passed"] for item in clean))
        self.assertTrue(all(item["validation_valid"] for item in clean))
        self.assertTrue(all(item["solution_verified"] for item in clean))

    def test_missing_schema_is_rejected_before_solving(self) -> None:
        missing = [item for item in self.result["cases"] if item["fault"] == "missing_schema"]

        self.assertEqual(6, len(missing))
        self.assertTrue(all(item["solver_status"] == "INVALID_DATA" for item in missing))
        self.assertTrue(all(item["fault_detected"] for item in missing))

    def test_tampered_objective_is_detected_by_solution_verifier(self) -> None:
        tampered = [item for item in self.result["cases"] if item["fault"] == "tampered_objective"]

        self.assertEqual(6, len(tampered))
        self.assertTrue(all(item["solution_verifiable"] for item in tampered))
        self.assertTrue(all(not item["solution_verified"] for item in tampered))
        self.assertTrue(all(item["fault_detected"] for item in tampered))

    def test_full_matrix_reaches_expected_detection_rates(self) -> None:
        self.assertEqual(18, self.result["case_count"])
        self.assertEqual(1.0, self.result["metrics"]["pass_rate"])
        self.assertEqual(1.0, self.result["metrics"]["clean_success_rate"])
        self.assertEqual(1.0, self.result["metrics"]["fault_detection_rate"])


if __name__ == "__main__":
    unittest.main()
