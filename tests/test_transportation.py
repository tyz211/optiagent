"""运输增量回归：输入语义、真实求解、独立验算、多轮状态与本地模型合同。"""

from copy import deepcopy
import json
from pathlib import Path
import sys
import tempfile
import unittest
from unittest.mock import Mock, patch

from fastapi.testclient import TestClient

from api import database
from api.main import app
from optiagent.llm import LLMConfig, call_openai_compatible_chat
from optiagent.mcp_servers.data_server import data_build_problem
from optiagent.mcp_servers.solver_server import solver_list_capabilities
from optiagent.optimization_gateway import build_problem_envelope, solve_problem_envelope, solve_question_via_gateway
from optiagent.requirement_analysis import analyze_requirements
from optiagent.solution_verifier import verify_solution
from optiagent.templates.capabilities import CAPABILITIES
from optiagent.templates.registry import get_template, template_ids
from optiagent.transportation import TransportationData, diagnose_transportation_infeasibility
from optiagent.transportation_text import parse_transportation_request


ROOT = Path(__file__).resolve().parents[1]
DATA = json.loads((ROOT / 'examples/transportation_sample.json').read_text())['transportation']
TEXT = ('运输分配；供给点 A 供给 20 吨；供给点 B 供给 30 吨；'
        '需求点 X 需求 10 吨；需求点 Y 需求 30 吨；'
        'A 到 X 单位运费 2 元/吨；A 到 Y 单位运费 5 元/吨；'
        'B 到 X 单位运费 4 元/吨；B 到 Y 单位运费 1 元/吨；最小化运输成本')


class TransportationContractTests(unittest.TestCase):
    """所有未消费的条件、单位和未知字段都应触发澄清。"""

    def test_text_and_json_have_same_semantics(self):
        parsed, evidence = parse_transportation_request(TEXT)
        self.assertEqual(DATA, parsed)
        self.assertEqual(8, sum(row['status'] == 'compiled' for row in evidence))
        brief = analyze_requirements(TEXT)
        self.assertEqual('transportation', brief.template_id)
        self.assertEqual('ready_to_solve', brief.readiness)
        self.assertIn('X 收货量恰好为 10 吨', brief.constraints)

    def test_schema_rejects_ambiguous_and_unsupported_data(self):
        invalid = []
        for field, value in [('quantity_unit', ''), ('currency', None), ('integer', True)]:
            invalid.append({**deepcopy(DATA), field: value})
        duplicate = deepcopy(DATA)
        duplicate['suppliers'].append(duplicate['suppliers'][0])
        invalid.append(duplicate)
        for value in (float('nan'), float('inf'), -1, True, '20'):
            bad = deepcopy(DATA)
            bad['suppliers'][0]['supply'] = value
            invalid.append(bad)
        for candidate in invalid:
            with self.subTest(candidate=candidate):
                report = build_problem_envelope('transportation', candidate).validation
                self.assertFalse(report.valid)

    def test_missing_unknown_duplicate_and_conflicting_routes_are_rejected(self):
        variants = []
        missing = deepcopy(DATA)
        missing['routes'].pop()
        variants.append(missing)
        unknown = deepcopy(DATA)
        unknown['routes'][0]['source'] = 'UNKNOWN'
        variants.append(unknown)
        duplicate = deepcopy(DATA)
        duplicate['routes'].append(duplicate['routes'][0])
        variants.append(duplicate)
        blocked = deepcopy(DATA)
        blocked['forbidden_routes'] = [{'source': 'A', 'target': 'X'}]
        variants.append(blocked)
        for candidate in variants:
            with self.subTest(candidate=candidate), self.assertRaises(ValueError):
                TransportationData.model_validate(candidate)

    def test_extra_conditions_never_reach_solver(self):
        for suffix in ['最多使用两辆车', 'A 到 X 最多运 5 吨', '允许缺货', '运输量必须是整数',
                       '最大化运输成本', '忽略供给上限', '至少满足一半需求']:
            for text in [TEXT + '；' + suffix, '运输分配；' + json.dumps(DATA) + '；' + suffix]:
                with self.subTest(suffix=suffix):
                    brief = analyze_requirements(text)
                    self.assertEqual('needs_clarification', brief.readiness)
                    self.assertFalse(brief.dialogue_contract.get('data'))

    def test_mixed_units_and_missing_data_clarify(self):
        for text in [TEXT.replace('20 吨', '20000 千克'), TEXT.replace('1 元/吨', '1 美元/吨'),
                     '运输分配；供给点 A 供给 20 吨', TEXT.replace('；A 到 X 单位运费 2 元/吨', '')]:
            with self.subTest(text=text):
                self.assertEqual('needs_clarification', analyze_requirements(text).readiness)

    def test_duplicate_json_keys_and_multiple_wrappers_are_rejected(self):
        repeated = json.dumps(DATA).replace('"supply": 20', '"supply": 10, "supply": 20')
        mixed = json.dumps({'transportation': DATA, 'knapsack': {'capacity': 1, 'items': []}})
        for text in (repeated, mixed):
            with self.subTest(text=text):
                self.assertEqual('needs_clarification', analyze_requirements(text).readiness)

    def test_json_fences_do_not_rewrite_entity_names(self):
        data = deepcopy(DATA)
        data['suppliers'][0]['name'] = 'A```'
        for route in data['routes']:
            if route['source'] == 'A':
                route['source'] = 'A```'
        parsed, _ = parse_transportation_request('运输分配；```json\n' + json.dumps(data) + '\n```')
        self.assertEqual(data, parsed)

    def test_authored_parser_regression_cases(self):
        from scripts.evaluate_requirement_parser import evaluate
        cases = json.loads((ROOT / 'tests/fixtures/requirement_cases.json').read_text())
        self.assertEqual(40, len({case['id'] for case in cases}))
        report = evaluate(cases)
        self.assertEqual(40, report['passed_count'], [r for r in report['cases'] if not r['passed']])
        self.assertEqual(0, report['real_llm_calls'])

    def test_compound_edit_is_atomic_and_undoable(self):
        first = analyze_requirements(TEXT)
        second = analyze_requirements('把供给点 B 供给改成 20 吨；把需求点 Y 需求改成 20 吨；先不求解', previous=first)
        self.assertEqual('hold', second.dialogue_contract['action'])
        self.assertEqual(2, second.dialogue_contract['revision'])
        self.assertEqual(20, second.dialogue_contract['data']['consumers'][1]['demand'])
        bad = analyze_requirements('把供给点 A 供给改成 100 吨；最多用两辆车', previous=second)
        self.assertEqual('needs_clarification', bad.readiness)
        self.assertEqual(second.dialogue_contract['data'], bad.dialogue_contract['data'])
        self.assertEqual(2, bad.dialogue_contract['revision'])
        restored = analyze_requirements('撤销上次修改', previous=bad)
        self.assertEqual(DATA, restored.dialogue_contract['data'])

    def test_repeated_edits_and_unknown_units_cannot_override_state(self):
        first = analyze_requirements(TEXT)
        for text in ['把供给点 A 供给改成 30 吨；把供给点 A 供给改成 40 吨',
                     '把供给点 A 供给改成 20000 千克', '把供给点 Z 供给改成 10 吨']:
            result = analyze_requirements(text, previous=first)
            self.assertEqual('needs_clarification', result.readiness)
            self.assertEqual(DATA, result.dialogue_contract['data'])

    def test_capability_catalog_and_solver_registry_are_in_sync(self):
        self.assertEqual(set(template_ids()), set(CAPABILITIES))
        capabilities = {row.template_id: row for row in solver_list_capabilities()}
        self.assertEqual(set(template_ids()), set(capabilities))
        self.assertIn('quantity_unit', capabilities['transportation'].input_keys)


class TransportationSolverTests(unittest.TestCase):
    """在真实求解器上验证已知目标，并破坏决策检测验证器是否独立。"""

    def test_known_optimum_and_mcp_file_input(self):
        problem = data_build_problem('transportation', ['examples/transportation_sample.json'])
        result = solve_problem_envelope(problem)
        self.assertEqual('OPTIMAL', result.status)
        # X 的最低单位成本是 2，Y 是 1；对应供给充足，因此下界 10*2+30*1 可达到。
        self.assertEqual(50, result.objective_value)
        self.assertTrue(result.solution_verification.passed)
        self.assertEqual(4, len(result.decisions))

    def test_forbidden_route_changes_optimum(self):
        data = deepcopy(DATA)
        data['routes'] = [r for r in data['routes'] if (r['source'], r['target']) != ('A', 'X')]
        data['forbidden_routes'] = [{'source': 'A', 'target': 'X'}]
        result = solve_problem_envelope(build_problem_envelope('transportation', data))
        # B 给 X 供货 10，余下 20 给 Y；A 向 Y 供货 10，总成本为 110。
        self.assertEqual(110, result.objective_value)
        self.assertTrue(result.solution_verification.passed)

    def test_infeasibility_is_not_invalid_data_or_success(self):
        data = deepcopy(DATA)
        data['suppliers'][1]['supply'] = 1
        problem = build_problem_envelope('transportation', data)
        self.assertTrue(problem.validation.valid)
        result = solve_problem_envelope(problem)
        self.assertEqual('INFEASIBLE', result.status)
        self.assertIsNone(result.objective_value)
        self.assertFalse(result.solution_verification.passed)

    def test_infeasibility_includes_recomputable_network_diagnosis(self):
        data = deepcopy(DATA)
        data['suppliers'][1]['supply'] = 1
        diagnosis = diagnose_transportation_infeasibility(TransportationData.model_validate(data))
        self.assertEqual('transportation_infeasibility', diagnosis['kind'])
        self.assertEqual(21, diagnosis['total_supply'])
        self.assertTrue(any(item['type'] == 'global_supply_shortfall' for item in diagnosis['issues']))
        result = solve_problem_envelope(build_problem_envelope('transportation', data))
        self.assertEqual(diagnosis, result.metrics['infeasibility_diagnosis'])
        self.assertIn('总供给不足', result.summary)
        self.assertTrue(any('运输模型不可行' in warning for warning in result.warnings))

    def test_disconnected_and_zero_demand_networks(self):
        data = deepcopy(DATA)
        data['forbidden_routes'] = [{k: r[k] for k in ('source', 'target')} for r in data['routes']]
        data['routes'] = []
        result = solve_problem_envelope(build_problem_envelope('transportation', data))
        self.assertEqual('INFEASIBLE', result.status)
        for row in data['consumers']:
            row['demand'] = 0
        result = solve_problem_envelope(build_problem_envelope('transportation', data))
        self.assertEqual(0, result.objective_value)
        self.assertTrue(result.solution_verification.passed)

    def test_corrupted_decisions_and_objectives_fail_verification(self):
        problem = build_problem_envelope('transportation', DATA)
        result = solve_problem_envelope(problem)
        corrupted = []
        wrong_objective = result.model_copy(deep=True)
        wrong_objective.objective_value = 0
        corrupted.append(wrong_objective)
        for field, value in [('quantity', -1), ('quantity', float('nan')), ('quantity', True), ('quantity', 100), ('source', 'Z')]:
            wrong = result.model_copy(deep=True)
            wrong.decisions[0][field] = value
            corrupted.append(wrong)
        duplicate = result.model_copy(deep=True)
        duplicate.decisions.append(duplicate.decisions[0])
        corrupted.append(duplicate)
        for wrong in corrupted:
            self.assertFalse(verify_solution(problem, wrong).passed)

    def test_direct_question_gateway_cannot_ignore_extra_conditions(self):
        with self.assertRaises(ValueError):
            solve_question_via_gateway('运输分配；' + json.dumps(DATA) + '；最多两辆车', get_template('transportation').build_spec('', None))


class TransportationLocalModelTests(unittest.TestCase):
    """模拟接口只验证控制逻辑，不将模拟输出计作真实模型效果。"""

    config = LLMConfig(True, '', 'http://127.0.0.1:11434/v1', 'test-local', 0)
    question = '运输分配：工厂A和B分别供应20吨和30吨，X和Y需要10吨和30吨。'

    def draft(self, **changes):
        return json.dumps({'data': DATA, 'missing_information': [], 'unsupported_requirements': [],
                           'source_quotes': [self.question], **changes}, ensure_ascii=False)

    @patch('optiagent.transportation_llm.call_openai_compatible_chat')
    def test_model_draft_requires_specific_confirmation(self, call):
        call.return_value = self.draft()
        first = analyze_requirements(self.question, llm_config=self.config)
        self.assertEqual('needs_clarification', first.readiness)
        self.assertFalse(first.dialogue_contract.get('data'))
        self.assertEqual(DATA, first.dialogue_contract['pending_transportation'])
        ignored = analyze_requirements('继续求解', previous=first)
        self.assertEqual('needs_clarification', ignored.readiness)
        confirmed = analyze_requirements('确认运输草稿并求解', previous=ignored)
        self.assertEqual('ready_to_solve', confirmed.readiness)
        self.assertEqual(DATA, confirmed.dialogue_contract['data'])
        self.assertEqual(1, call.call_count)
        self.assertIn('json_schema', call.call_args.kwargs)

    @patch('optiagent.transportation_llm.call_openai_compatible_chat')
    def test_unsupported_missing_and_bad_model_responses_stay_blocked(self, call):
        for raw in [self.draft(unsupported_requirements=['两辆车']), self.draft(missing_information=['运费']),
                    self.draft(source_quotes=['不存在的原文']), 'not json']:
            call.return_value = raw
            brief = analyze_requirements(self.question, llm_config=self.config)
            self.assertEqual('needs_clarification', brief.readiness)
            self.assertFalse(brief.dialogue_contract.get('pending_transportation'))

    @patch('optiagent.transportation_llm.call_openai_compatible_chat')
    def test_remote_endpoint_is_not_used_as_fallback(self, call):
        remote = LLMConfig(True, 'not-a-real-key', 'https://example.com/v1', 'remote')
        brief = analyze_requirements(self.question, llm_config=remote)
        call.assert_not_called()
        self.assertEqual('needs_clarification', brief.readiness)

    @patch('optiagent.llm.requests.post')
    def test_compatible_client_passes_schema_and_budget(self, post):
        post.return_value = Mock(json=lambda: {'choices': [{'message': {'content': '{}'}}]})
        call_openai_compatible_chat(self.config, [], json_schema={'type': 'object'}, timeout=60, max_tokens=3000)
        kwargs = post.call_args.kwargs
        self.assertEqual('json_schema', kwargs['json']['response_format']['type'])
        self.assertEqual(3000, kwargs['json']['max_tokens'])
        self.assertEqual(60, kwargs['timeout'])

    @patch('scripts.evaluate_transportation_model.propose_transportation')
    def test_model_evaluator_does_not_accept_hallucinated_parameters(self, propose):
        from scripts.evaluate_transportation_model import evaluate
        from optiagent.transportation_llm import TransportationDraft
        case = json.loads((ROOT / 'tests/fixtures/transportation_model_cases.json').read_text())[0]
        wrong = deepcopy(case['data'])
        wrong['suppliers'][0]['supply'] = 999
        propose.return_value = TransportationDraft(data=TransportationData.model_validate(wrong), source_quotes=[case['question']])
        report = evaluate([case], self.config)
        self.assertEqual(0, report['passed_count'])
        self.assertEqual(1, report['llm_attempts'])


class TransportationAPITests(unittest.TestCase):
    """隔离数据库，通过实际 API 验证求解、暂停、撤销、比较及澄清门控。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db = patch.object(database, 'DB_PATH', Path(self.temp.name) / 'transport.sqlite3')
        self.db.start()
        self.client = TestClient(app)
        self.client.__enter__()
        self.cid = self.client.post('/api/conversations', json={'title': '运输验收'}).json()['conversation']['id']

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.db.stop()
        self.temp.cleanup()

    def ask(self, question):
        response = self.client.post('/api/ask', json={'question': question, 'conversation_id': self.cid})
        self.assertEqual(200, response.status_code, response.text)
        return response.json()

    def test_real_dialogue_round_trip(self):
        first = self.ask(TEXT)
        self.assertEqual(50, first['objective_value'])
        self.assertTrue(first['solution_verification']['passed'])
        held = self.ask('把供给点 B 供给改成 20 吨；先不求解')
        self.assertEqual('REQUIREMENT_UPDATED', held['status'])
        self.assertNotIn('solver', [n['node_id'] for n in held['agent_graph']['nodes']])
        second = self.ask('继续求解')
        self.assertEqual(90, second['objective_value'])
        self.assertTrue(second['solution_verification']['passed'])
        compared = self.ask('比较最近两个方案')
        self.assertEqual(40, compared['plan_comparison']['delta'])
        restored = self.ask('撤销上次修改')
        self.assertEqual(50, restored['objective_value'])

    def test_unknown_constraint_never_calls_solver(self):
        result = self.ask(TEXT + '；最多使用两辆车')
        self.assertEqual('NEEDS_CLARIFICATION', result['status'])
        self.assertNotIn('solver', [n['node_id'] for n in result['agent_graph']['nodes']])

    def test_infeasible_transportation_is_not_retried(self):
        result = self.ask(TEXT.replace('B 供给 30 吨', 'B 供给 1 吨'))
        self.assertEqual('INFEASIBLE', result['status'])
        self.assertEqual(1, sum(n['node_id'] == 'solver' for n in result['agent_graph']['nodes']))


class TransportationMCPTests(unittest.IsolatedAsyncioTestCase):
    """实际启动两个独立 MCP 进程，检查运输合同是否可跨进程传递。"""

    async def test_transportation_stdio_round_trip(self):
        from mcp import ClientSession, StdioServerParameters
        from mcp.client.stdio import stdio_client

        async def invoke(module, name, arguments):
            server = StdioServerParameters(command=sys.executable, args=['-m', module])
            async with stdio_client(server) as (read, write):
                async with ClientSession(read, write) as session:
                    await session.initialize()
                    result = await session.call_tool(name, arguments)
                    self.assertFalse(result.isError)
                    return result.structuredContent

        problem = await invoke('optiagent.mcp_servers.data_server', 'data_build_problem',
                               {'template_id': 'transportation', 'paths': ['examples/transportation_sample.json']})
        with self.assertNoLogs('mcp.client.stdio', level='ERROR'):
            result = await invoke('optiagent.mcp_servers.solver_server', 'solver_solve_problem',
                                  {'problem': problem, 'time_limit': 5})
        self.assertEqual('OPTIMAL', result['status'])
        self.assertEqual(50, result['objective_value'])
        self.assertTrue(result['solution_verification']['passed'])


if __name__ == '__main__':
    unittest.main()
