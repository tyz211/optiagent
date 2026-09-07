from __future__ import annotations

import unittest

from optiagent.mcp_contracts import SolveEnvelope
from optiagent.optimization_gateway import build_problem_envelope, solve_problem_envelope
from optiagent.solution_verifier import verify_solution


class SolutionVerifierTests(unittest.TestCase):
    """验证六类模板都能独立复算正确解并拒绝被篡改的解。"""

    def test_six_templates_accept_valid_solutions(self) -> None:
        cases = _valid_cases()
        for template_id, (data, solution) in cases.items():
            with self.subTest(template_id=template_id):
                problem = build_problem_envelope(template_id, data)
                report = verify_solution(problem, solution)
                self.assertTrue(report.verifiable)
                self.assertTrue(report.feasible)
                self.assertTrue(report.objective_consistent)
                self.assertTrue(report.passed)
                self.assertEqual(solution.objective_value, report.recomputed_objective)

    def test_six_templates_reject_tampered_solutions(self) -> None:
        cases = _invalid_cases()
        for template_id, (data, solution) in cases.items():
            with self.subTest(template_id=template_id):
                problem = build_problem_envelope(template_id, data)
                report = verify_solution(problem, solution)
                self.assertTrue(report.verifiable)
                self.assertFalse(report.passed)
                self.assertTrue(report.violations)

    def test_gateway_always_attaches_verification_report(self) -> None:
        problem = build_problem_envelope(
            "knapsack",
            {
                "capacity": 5,
                "items": [
                    {"item": "A", "value": 8, "weight": 3},
                    {"item": "B", "value": 5, "weight": 2},
                ],
            },
        )
        result = solve_problem_envelope(problem)

        self.assertEqual("OPTIMAL", result.status)
        self.assertIsNotNone(result.solution_verification)
        self.assertTrue(result.solution_verification.passed)

    def test_non_solution_status_is_marked_unverifiable(self) -> None:
        """没有可行决策时应明确标记不可验证，而不是伪造通过结论。"""

        data, _solution = _valid_cases()["knapsack"]
        problem = build_problem_envelope("knapsack", data)
        result = SolveEnvelope(
            template_id="knapsack",
            status="INFEASIBLE",
            summary="模型不可行",
        )
        report = verify_solution(problem, result)

        self.assertFalse(report.verifiable)
        self.assertFalse(report.passed)
        self.assertIsNone(report.feasible)

    def test_template_mismatch_is_rejected(self) -> None:
        """禁止把其他问题类型的决策伪装成当前模板结果。"""

        data, solution = _valid_cases()["knapsack"]
        problem = build_problem_envelope("knapsack", data)
        mismatched = solution.model_copy(update={"template_id": "assignment"})
        report = verify_solution(problem, mismatched)

        self.assertFalse(report.passed)
        self.assertTrue(any("模板" in item for item in report.violations))


def _solve(template_id: str, objective: float, decisions: list[dict], metrics: dict | None = None) -> SolveEnvelope:
    """构造不依赖真实求解器的固定结果，便于隔离测试验证逻辑。"""

    return SolveEnvelope(
        template_id=template_id,
        status="FEASIBLE",
        objective_value=objective,
        summary="测试结果",
        decisions=decisions,
        metrics=metrics or {},
    )


def _valid_cases() -> dict[str, tuple[dict, SolveEnvelope]]:
    knapsack_data = {
        "capacity": 5,
        "items": [
            {"item": "A", "value": 8, "weight": 3},
            {"item": "B", "value": 5, "weight": 2},
        ],
    }
    assignment_data = {
        "resources": ["R1", "R2"],
        "tasks": ["T1", "T2"],
        "costs": [
            {"resource": "R1", "task": "T1", "cost": 2},
            {"resource": "R1", "task": "T2", "cost": 8},
            {"resource": "R2", "task": "T1", "cost": 7},
            {"resource": "R2", "task": "T2", "cost": 3},
        ],
    }
    tsp_data = {
        "nodes": ["A", "B", "C"],
        "distance_matrix": [[0, 2, 4], [2, 0, 3], [4, 3, 0]],
    }
    job_shop_data = {
        "tasks": [
            {"job": "J1", "machine": "M1", "duration": 2, "order": 1},
            {"job": "J1", "machine": "M2", "duration": 1, "order": 2},
            {"job": "J2", "machine": "M1", "duration": 2, "order": 1},
        ]
    }
    production_data = {
        "products": [
            {"product": "P1", "profit": 5, "labor": 2, "max_qty": 3},
            {"product": "P2", "profit": 4, "labor": 1},
        ],
        "capacities": {"labor": 5},
        "integer": True,
    }
    facility_data = {
        "warehouses": [
            {"warehouse": "W1", "capacity": 10, "fixed_cost": 5},
            {"warehouse": "W2", "capacity": 10, "fixed_cost": 4},
        ],
        "customers": [{"customer": "C1", "demand": 6}],
        "costs": [
            {"warehouse": "W1", "customer": "C1", "cost": 2},
            {"warehouse": "W2", "customer": "C1", "cost": 3},
        ],
    }
    facility_decisions = [{
        "allocations": [{"warehouse": "W1", "customer": "C1", "quantity": 6}],
        "warehouses": [
            {"warehouse": "W1", "is_open": 1},
            {"warehouse": "W2", "is_open": 0},
        ],
        "customers": [{"customer": "C1", "received": 6}],
    }]
    return {
        "knapsack": (knapsack_data, _solve("knapsack", 13, [
            {"item": "A", "selected": 1}, {"item": "B", "selected": 1},
        ])),
        "assignment": (assignment_data, _solve("assignment", 5, [
            {"resource": "R1", "task": "T1"}, {"resource": "R2", "task": "T2"},
        ])),
        "tsp": (tsp_data, _solve("tsp", 9, [], {"route": ["A", "B", "C", "A"]})),
        "job_shop_scheduling": (job_shop_data, _solve("job_shop_scheduling", 4, [
            {"job": "J1", "machine": "M1", "order": 1, "start": 0, "end": 2},
            {"job": "J1", "machine": "M2", "order": 2, "start": 2, "end": 3},
            {"job": "J2", "machine": "M1", "order": 1, "start": 2, "end": 4},
        ])),
        "production_mix": (production_data, _solve("production_mix", 17, [
            {"product": "P1", "quantity": 1}, {"product": "P2", "quantity": 3},
        ])),
        "facility_location": (facility_data, _solve("facility_location", 17, facility_decisions)),
    }


def _invalid_cases() -> dict[str, tuple[dict, SolveEnvelope]]:
    valid = _valid_cases()
    return {
        "knapsack": (valid["knapsack"][0], _solve("knapsack", 13, [
            {"item": "A", "selected": 1}, {"item": "B", "selected": 2},
        ])),
        "assignment": (valid["assignment"][0], _solve("assignment", 4, [
            {"resource": "R1", "task": "T1"}, {"resource": "R1", "task": "T2"},
        ])),
        "tsp": (valid["tsp"][0], _solve("tsp", 4, [], {"route": ["A", "B", "A"]})),
        "job_shop_scheduling": (valid["job_shop_scheduling"][0], _solve("job_shop_scheduling", 3, [
            {"job": "J1", "machine": "M1", "order": 1, "start": 0, "end": 2},
            {"job": "J1", "machine": "M2", "order": 2, "start": 2, "end": 3},
            {"job": "J2", "machine": "M1", "order": 1, "start": 1, "end": 3},
        ])),
        "production_mix": (valid["production_mix"][0], _solve("production_mix", 25, [
            {"product": "P1", "quantity": 5}, {"product": "P2", "quantity": 0},
        ])),
        "facility_location": (valid["facility_location"][0], _solve("facility_location", 17, [{
            "allocations": [{"warehouse": "W1", "customer": "C1", "quantity": 5}],
            "warehouses": [
                {"warehouse": "W1", "is_open": 1},
                {"warehouse": "W2", "is_open": 0},
            ],
        }])),
    }


if __name__ == "__main__":
    unittest.main()
