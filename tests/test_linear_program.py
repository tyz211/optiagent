from __future__ import annotations

from copy import deepcopy
from itertools import product
from pathlib import Path
import re
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient
from api import database
from api.main import app
from optiagent.linear_model import LinearModel, parse_linear_text
from optiagent.optimization_gateway import build_problem_envelope, solve_problem_envelope
from optiagent.recovery_runtime import RecoveryPolicyRuntime
from optiagent.requirement_analysis import analyze_requirements
from optiagent.solution_verifier import verify_solution


# 回归样本直接取用户提供的数学模型，确保不依赖手工转换的 JSON。
SAMPLE = re.search(r'```latex\n(.*?)```', (Path(__file__).resolve().parents[1] / 'examples/linear_program_sample.md').read_text(), re.S)[1]


def solve(text):
    """测试完整的解析、合同校验、求解和独立验算链路。"""
    problem = build_problem_envelope('linear_program', parse_linear_text(text))
    return problem, solve_problem_envelope(problem)


class LinearParserTests(unittest.TestCase):
    """解析失败必须明确报错，不能只求解成功识别的一部分公式。"""

    def test_original_latex_preserves_every_variable_and_constraint(self):
        data = parse_linear_text(SAMPLE)
        self.assertEqual(10, len(data['variables']))
        self.assertEqual(8, len(data['constraints']))
        self.assertEqual({'x5': 1, 'x6': -1, 'x7': -1}, data['constraints'][4]['coefficients'])
        self.assertEqual([0, 1, 2, 3], next(v['values'] for v in data['variables'] if v['name'] == 'x7'))

    def test_plain_text_fraction_parentheses_constant_and_rhs_variables(self):
        text = 'x1 in R\nx2 in {0,1}\nmin \\frac{1}{2}x1 + 2x2 + 3\n2(x1+1) >= x2+6\nx1 <= 5'
        _, result = solve(text)
        self.assertEqual(4, result.objective_value)
        self.assertTrue(result.solution_verification.passed)
        self.assertEqual({'x1': 2, 'x2': 0}, {r['variable']: r['value'] for r in result.decisions})

    def test_chain_bounds_unicode_and_free_real_variables(self):
        _, result = solve('x ∈ R\n最小化 x\n−2 ≤ x ≤ 3')
        self.assertEqual(-2, result.objective_value)
        self.assertTrue(result.solution_verification.passed)

    def test_reject_nonlinearity_unknown_fragments_and_missing_information(self):
        cases = [
            'x1,x2 in {0,1}\nmax x1*x2',
            'x1 in {0,1}\nmax x1^2',
            'x1,x2 in {0,1}\nmax x1/x2',
            'x1 in {0,1}\nmax 1 2x1',
            'x1 in {0,1}\nmax x1\n另有一个未说明的逻辑条件',
            'x1 in {0,1}\nmax x1+x2-x2',
            'x1 in {0,1}\nmax x1\nmin x1',
            'max x1', 'x1 in {0,1}',
            'x1 in {0,1.5}\nmax x1',
            'x1 in {0,1}\nx1 in {0,1}\nmax x1',
            'x1 in {0,1}\nmax 1e999*x1',
            'x1 in {0,1}\nmax x1\n__import__("os").system("false")',
        ]
        for text in cases:
            with self.subTest(text=text), self.assertRaises(ValueError):
                parse_linear_text(text)

    def test_unknown_json_fields_and_references_are_rejected(self):
        data = parse_linear_text(SAMPLE)
        for mutate in (lambda d: d.update(hidden_constraint='x1=0'),
                       lambda d: d['objective']['coefficients'].update(unknown=1),
                       lambda d: d['variables'][0].update(lower=2, upper=1)):
            candidate = deepcopy(data)
            mutate(candidate)
            with self.assertRaises(ValueError):
                LinearModel.model_validate(candidate)
            problem = build_problem_envelope('linear_program', candidate)
            with patch('optiagent.optimization_gateway.get_generic_solver') as registry:
                result = solve_problem_envelope(problem)
                registry.assert_not_called()
            self.assertEqual('INVALID_DATA', result.status)

    def test_invalid_replacement_preserves_last_valid_revision(self):
        first = analyze_requirements(SAMPLE)
        invalid = analyze_requirements('x1 in {0,1}\nmax x1^2', previous=first)
        self.assertEqual('needs_clarification', invalid.readiness)
        self.assertEqual(first.dialogue_contract['data'], invalid.dialogue_contract['data'])
        resumed = analyze_requirements('继续求解', previous=invalid)
        self.assertEqual('ready_to_solve', resumed.readiness)
        self.assertEqual(1, resumed.dialogue_contract['revision'])


class LinearSolverTests(unittest.TestCase):
    """用独立穷举对照最优解，并主动篡改输出检查验证器。"""

    def test_original_optimum_matches_independent_enumeration(self):
        # 枚举直接编码原题条件，不复用解析器的系数矩阵作为真值。
        objective = [10, 8, 15, 4, 7, 9, 4, 5, 6, 6]
        capacity_a = [4, 3, 6, 1, 2, 4, 2, 1, 5, 2]
        capacity_b = [3, 3, 4, 1, 2, 3, 2, 1, 4, 1]
        best = -1
        optimal = []
        domains = [range(2)] * 5 + [range(3), range(4), range(3), range(2), range(2)]
        for x in product(*domains):
            if sum(a*b for a, b in zip(capacity_a, x)) > 15 or sum(a*b for a, b in zip(capacity_b, x)) > 12:
                continue
            if not (x[0]+x[1] <= 1 and x[2] <= x[3] and x[4] <= x[5]+x[6] and
                    x[5] >= x[0]+x[2] and x[7] >= x[0] and x[7] >= x[1]):
                continue
            value = sum(a*b for a, b in zip(objective, x))
            if value > best:
                best, optimal = value, [x]
            elif value == best:
                optimal.append(x)
        _, result = solve(SAMPLE)
        self.assertEqual(46, best)
        self.assertEqual(best, result.objective_value)
        self.assertIn(tuple(row['value'] for row in result.decisions), optimal)
        self.assertTrue(result.solution_verification.passed)
        self.assertEqual(8, len(result.metrics['constraint_activity']))

    def test_noncontiguous_domain_is_not_relaxed_to_an_interval(self):
        problem, result = solve('x in {0,2}\nmax x\nx <= 1')
        self.assertEqual(0, result.objective_value)
        tampered = result.model_copy(deep=True)
        tampered.decisions[0]['value'] = 1
        tampered.objective_value = 1
        report = verify_solution(problem, tampered)
        self.assertFalse(report.passed)
        self.assertFalse(report.checks['domain:x'])

    def test_verifier_rejects_tampered_objective_missing_variables_and_bad_constraints(self):
        problem, result = solve(SAMPLE)
        altered = result.model_copy(deep=True)
        altered.objective_value += 1
        self.assertFalse(verify_solution(problem, altered).objective_consistent)
        altered = result.model_copy(deep=True)
        altered.decisions.pop()
        self.assertFalse(verify_solution(problem, altered).passed)
        altered = result.model_copy(deep=True)
        next(row for row in altered.decisions if row['variable'] == 'x2')['value'] = 1
        altered.objective_value += 8
        report = verify_solution(problem, altered)
        self.assertFalse(report.feasible)
        self.assertTrue(report.objective_consistent)
        self.assertFalse(report.checks['constraint:3'])

    def test_infeasible_and_unbounded_are_reported(self):
        for text, status in [('x in {0,1}\nmax x\nx >= 2', 'INFEASIBLE'), ('x in R\nmax x', 'UNBOUNDED')]:
            _, result = solve(text)
            self.assertEqual(status, result.status)
            self.assertIsNone(result.objective_value)
            self.assertFalse(result.solution_verification.passed)

    def test_new_template_does_not_use_six_template_checkpoint(self):
        agent = Mock()
        runtime = RecoveryPolicyRuntime(agent)
        decision = runtime.decide({'problem_spec': {'template_id': 'linear_program'},
                                   'verification': {'passed': True}})
        agent.policy.assert_not_called()
        self.assertEqual('accept_solution', decision.selected_action)
        self.assertEqual('unsupported_template', decision.metadata['fallback_reason'])


class LinearWorkflowTests(unittest.TestCase):
    """用临时数据库验证 HTTP/SSE、暂停、替换、撤销和错误澄清。"""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(database, 'DB_PATH', Path(self.temporary.name) / 'linear.sqlite3')
        self.db_patch.start()
        self.client = TestClient(app)
        self.client.__enter__()
        self.cid = self.client.post('/api/conversations', json={'title': '数学模型验收'}).json()['conversation']['id']

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.db_patch.stop()
        self.temporary.cleanup()

    def ask(self, question):
        response = self.client.post('/api/ask', json={'question': question, 'conversation_id': self.cid})
        self.assertEqual(200, response.status_code, response.text)
        return response.json()

    def test_http_original_input_without_upload_or_llm(self):
        result = self.ask(SAMPLE)
        self.assertEqual('OPTIMAL', result['status'])
        self.assertEqual(46, result['objective_value'])
        self.assertTrue(result['solution_verification']['passed'])
        self.assertEqual(8, len(result['problem_spec']['constraints']))
        self.assertIn('10*x1', result['problem_spec']['objective'])

    def test_sse_original_input(self):
        response = self.client.post('/api/ask/stream', json={'question': SAMPLE, 'conversation_id': self.cid})
        self.assertEqual(200, response.status_code)
        self.assertIn('event: final', response.text)
        self.assertIn('"objective_value": 46.0', response.text)
        self.assertNotIn('event: error', response.text)

    def test_hold_replace_undo_and_invalid_model_do_not_lose_data(self):
        first = self.ask(SAMPLE + '\n先不求解')
        self.assertEqual('REQUIREMENT_UPDATED', first['status'])
        self.assertNotIn('solver', [node['node_id'] for node in first['agent_graph']['nodes']])
        self.assertEqual(46, self.ask('继续求解')['objective_value'])
        second = self.ask('x in {0,1}\nmax 3x')
        self.assertEqual(3, second['objective_value'])
        invalid = self.ask('x in {0,1}\nmax x*x')
        self.assertEqual('NEEDS_CLARIFICATION', invalid['status'])
        self.assertNotIn('solver', [node['node_id'] for node in invalid['agent_graph']['nodes']])
        self.assertEqual(3, self.ask('继续求解')['objective_value'])
        self.assertEqual(46, self.ask('撤销上次修改')['objective_value'])

    def test_solver_failure_is_not_misreported_as_missing_upload(self):
        with patch('gurobipy.Model', side_effect=RuntimeError('模拟许可或求解器失败')):
            result = self.ask(SAMPLE)
        self.assertEqual('SOLVER_ERROR', result['status'])
        self.assertNotIn('请先上传', result['answer'])


if __name__ == '__main__':
    unittest.main()
