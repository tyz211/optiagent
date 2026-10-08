"""主控决策使用模拟响应，工具使用真实求解；不计作真实模型效果评测。"""

from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient

from api import database
from api.database import create_conversation, get_agent_episode, init_db, save_llm_config
from api.main import app
from api.services.agent_memory import add_memory, delete_memory, load_task, search_memories
from api.services.agent_workflow import run_agent_workflow
from api.services.llm_controller import ControllerDecision, ControllerLimits, run_llm_controller
from optiagent.agent_context import ContextBudgetExceeded, build_context
from optiagent.llm import LLMConfig


QUESTION = '求解背包：' + json.dumps({'knapsack': {'capacity': 5, 'items': [
    {'item': 'A', 'value': 8, 'weight': 3}, {'item': 'B', 'value': 5, 'weight': 2}]}})
SOLVE_ACTIONS = ['analyze_requirements', 'build_model', 'solve', 'verify', 'finish']


def decision(action, message='', query=''):
    """生成符合真实接口合同的脚本化行动。"""
    return ControllerDecision(action=action, reason='测试所需的下一步行动', query=query, message=message)


class ControllerTests(unittest.TestCase):
    """验证动态顺序、数学门控、跨轮状态和明确失败出口。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(database, 'DB_PATH', Path(self.temp.name) / 'controller.sqlite3')
        self.db_patch.start()
        init_db()
        self.conversation = create_conversation('主控测试')['id']
        self.config = LLMConfig(True, 'test-secret', 'https://example.test/v1', 'scripted-model')

    def tearDown(self):
        self.db_patch.stop()
        self.temp.cleanup()

    def run_script(self, actions, *, question=QUESTION, limits=None):
        """仅替换模型选择，需求、求解、验算和持久化仍执行真实实现。"""
        outputs = [item if isinstance(item, ControllerDecision) else decision(item) for item in actions]
        with patch('api.services.llm_controller.choose_action', side_effect=outputs) as choose:
            result = run_llm_controller(question=question, requested_dataset_id=None, mcp_config='',
                user_id=None, conversation_id=self.conversation, llm_config=self.config, limits=limits)
        return result, choose

    def test_model_selects_dynamic_order_and_feedback_is_in_next_context(self):
        result, choose = self.run_script(['inspect_data', decision('retrieve_knowledge', query='背包'), *SOLVE_ACTIONS])
        self.assertEqual('OPTIMAL', result['status'])
        self.assertEqual(13, result['objective_value'])
        self.assertTrue(result['workflow_verification']['passed'])
        self.assertEqual('LLMController', result['agent_graph']['engine'])
        self.assertEqual(['inspect_data', 'retrieve_knowledge', *SOLVE_ACTIONS],
                         [item['action'] for item in result['agent_policy']['decisions']])
        second_context = json.loads(choose.call_args_list[1].args[1][1]['content'])
        self.assertIn('data', second_context['tool_observations'][0])
        episode = get_agent_episode(result['agent_episode_id'], user_id=None)
        self.assertEqual('completed', episode['status'])
        self.assertEqual('knapsack', episode['template_id'])
        self.assertIn('verify', [step['node_id'] for step in episode['steps']])
        self.assertNotIn('test-secret', json.dumps(episode))

    def test_unverified_finish_is_blocked_then_model_can_verify(self):
        actions = ['analyze_requirements', 'build_model', 'solve', 'finish', 'verify', 'finish']
        result, choose = self.run_script(actions)
        self.assertTrue(result['workflow_verification']['passed'])
        feedback = json.loads(choose.call_args_list[4].args[1][1]['content'])['tool_observations']
        self.assertTrue(any('不满足前置条件' in item.get('summary', '') for item in feedback))
        episode = get_agent_episode(result['agent_episode_id'], user_id=None)
        self.assertEqual(1, sum(step['node_id'] == 'solve' for step in episode['steps']))

    def test_unsupported_extra_condition_cannot_reach_solver(self):
        result, _ = self.run_script(['analyze_requirements', 'solve', decision('clarify', '请明确额外条件。')],
                                    question=QUESTION + '；A 和 B 不能同时选择')
        self.assertEqual('NEEDS_CLARIFICATION', result['status'])
        episode = get_agent_episode(result['agent_episode_id'], user_id=None)
        self.assertNotIn('solve', [step['node_id'] for step in episode['steps']])
        self.assertEqual('waiting_for_user', load_task(None, self.conversation)['status'])

    def test_clarification_resumes_with_persisted_requirement(self):
        first, _ = self.run_script(['analyze_requirements', decision('clarify', '请提供物品和容量。')], question='我想求解背包问题')
        self.assertEqual('NEEDS_CLARIFICATION', first['status'])
        second, choose = self.run_script(SOLVE_ACTIONS)
        payload = json.loads(choose.call_args_list[0].args[1][1]['content'])
        self.assertEqual('waiting_for_user', payload['task']['previous_task']['status'])
        self.assertTrue(second['solution_verification']['passed'])
        self.assertEqual(second['run_id'], load_task(None, self.conversation)['last_verified_run_id'])

    def test_follow_up_changes_active_version_without_stale_json(self):
        first, _ = self.run_script(SOLVE_ACTIONS)
        second, _ = self.run_script(SOLVE_ACTIONS, question='把容量改成 3')
        self.assertEqual(13, first['objective_value'])
        self.assertEqual(8, second['objective_value'])
        self.assertEqual(2, load_task(None, self.conversation)['requirement_revision'])
        self.assertEqual(3, second['requirement_analysis']['dialogue_contract']['data']['capacity'])

    def test_new_turn_cannot_solve_using_old_readiness_before_analysis(self):
        self.run_script(SOLVE_ACTIONS)
        result, choose = self.run_script(['solve', *SOLVE_ACTIONS], question='把容量改成 3')
        initial = json.loads(choose.call_args_list[0].args[1][1]['content'])
        self.assertNotIn('solve', initial['available_actions'])
        self.assertEqual(8, result['objective_value'])

    def test_budget_exhaustion_is_not_misreported_as_success(self):
        result, _ = self.run_script(['inspect_data', 'inspect_data'], limits=ControllerLimits(max_decisions=2))
        self.assertEqual('AGENT_BUDGET_EXCEEDED', result['status'])
        self.assertFalse(result['workflow_verification']['passed'])
        self.assertEqual('failed', get_agent_episode(result['agent_episode_id'], user_id=None)['status'])

    def test_budget_after_solver_rejects_candidate_in_history(self):
        result, _ = self.run_script(['analyze_requirements', 'build_model', 'solve'],
                                    limits=ControllerLimits(max_decisions=3))
        self.assertEqual('AGENT_BUDGET_EXCEEDED', result['status'])
        rows = database.list_runs(user_id=None, conversation_id=self.conversation)
        self.assertIn('UNACCEPTED_RESULT', [row['status'] for row in rows])
        self.assertNotIn('OPTIMAL', [row['status'] for row in rows])

    def test_model_error_does_not_leak_secrets_or_run_rule_fallback(self):
        with patch('api.services.llm_controller.choose_action', side_effect=RuntimeError('test-secret')):
            result = run_llm_controller(question=QUESTION, requested_dataset_id=None, mcp_config='',
                user_id=None, conversation_id=self.conversation, llm_config=self.config)
        self.assertEqual('AGENT_ERROR', result['status'])
        self.assertNotIn('test-secret', json.dumps(result))
        self.assertIsNone(result['objective_value'])

    def test_failed_mathematical_verification_cannot_finish_successfully(self):
        bad_result = {'answer': '候选方案', 'status': 'OPTIMAL', 'objective_value': 13,
                      'structured_answer': {}, 'solution_verification': {'verifiable': True, 'passed': False}}
        with patch('api.services.agent_workflow._solver_node', return_value={'result': bad_result}):
            result, choose = self.run_script(['analyze_requirements', 'build_model', 'solve', 'verify',
                                             'finish', decision('clarify', '验算失败，需要检查模型。')])
        allowed = json.loads(choose.call_args_list[4].args[1][1]['content'])['available_actions']
        self.assertNotIn('finish', allowed)
        self.assertEqual('NEEDS_CLARIFICATION', result['status'])
        self.assertIsNone(result['objective_value'])

    def test_context_uses_explicit_memory_without_overriding_current_requirement(self):
        add_memory(user_id=None, conversation_id=self.conversation, content='背包容量通常为 100', kind='business_rule')
        result, choose = self.run_script(SOLVE_ACTIONS)
        payload = json.loads(choose.call_args_list[0].args[1][1]['content'])
        self.assertEqual(1, len(payload['retrieved_memories']))
        self.assertEqual(5, result['requirement_analysis']['dialogue_contract']['data']['capacity'])

    def test_strict_schema_and_output_budget_are_sent_to_compatible_api(self):
        from unittest.mock import Mock
        from api.services.llm_controller import choose_action
        response = Mock()
        response.json.return_value = {'choices': [{'message': {'content': decision('inspect_data').model_dump_json()}}]}
        with patch('optiagent.llm.requests.post', return_value=response) as post:
            selected = choose_action(self.config, [{'role': 'user', 'content': '检查数据'}], timeout=3, max_output_tokens=700)
        self.assertEqual('inspect_data', selected.action)
        body = post.call_args.kwargs['json']
        schema = body['response_format']['json_schema']
        self.assertTrue(schema['strict'])
        self.assertFalse(schema['schema']['additionalProperties'])
        self.assertEqual(set(schema['schema']['properties']), set(schema['schema']['required']))
        self.assertEqual(700, body['max_tokens'])

    def test_infeasible_transportation_cannot_retry_by_rebuilding(self):
        from collections import Counter
        from api.services.llm_controller import available_actions
        state = {'requirements_done': True, 'requirement_analysis': {
            'template_id': 'transportation', 'readiness': 'ready_to_solve', 'dialogue_contract': {'action': 'solve'}},
            'problem_spec': {'template_id': 'transportation'}, 'result': {'status': 'INFEASIBLE'}}
        actions = available_actions(state, Counter({'solve': 1, 'build_model': 1}))
        self.assertNotIn('build_model', actions)
        self.assertNotIn('solve', actions)
        self.assertIn('verify', actions)

    def test_clear_runs_removes_task_but_retains_explicit_long_term_memory(self):
        self.run_script(SOLVE_ACTIONS)
        add_memory(user_id=None, conversation_id=self.conversation, content='单位使用吨', kind='terminology')
        database.clear_runs(user_id=None, conversation_id=self.conversation)
        self.assertEqual({}, load_task(None, self.conversation))
        self.assertEqual(1, len(search_memories(user_id=None, conversation_id=self.conversation)))

    def test_context_overflow_stops_before_model_call(self):
        with patch('api.services.llm_controller.choose_action') as choose:
            result = run_llm_controller(question='约束' * 10000, requested_dataset_id=None, mcp_config='',
                user_id=None, conversation_id=self.conversation, llm_config=self.config)
        choose.assert_not_called()
        self.assertEqual('context_budget_exhausted', result['agent_controller']['stop_reason'])

    def test_configured_http_and_sse_enter_controller(self):
        save_llm_config({'name': '测试', 'base_url': self.config.base_url, 'model': self.config.model,
                         'api_key': self.config.api_key, 'temperature': 0.2})
        with TestClient(app) as client:
            with patch('api.services.llm_controller.choose_action', side_effect=[decision(a) for a in SOLVE_ACTIONS]):
                response = client.post('/api/ask', json={'question': QUESTION, 'conversation_id': self.conversation})
            self.assertEqual(200, response.status_code)
            self.assertEqual('llm', response.json()['agent_controller']['mode'])
            with patch('api.services.llm_controller.choose_action', side_effect=[decision(a) for a in SOLVE_ACTIONS]):
                stream = client.post('/api/ask/stream', json={'question': '把容量改成 3', 'conversation_id': self.conversation})
            self.assertIn('event: agent_step', stream.text)
            self.assertIn('event: final', stream.text)
            self.assertIn('LLMController', stream.text)

    def test_explicit_llm_mode_without_configuration_is_rejected(self):
        with TestClient(app) as client:
            response = client.post('/api/ask', json={'question': QUESTION, 'agent_mode': 'llm'})
        self.assertEqual(400, response.status_code)


class ContextTests(unittest.TestCase):
    """上下文预算只能移除可选历史，不能切断有效约束。"""

    def test_optional_history_is_dropped_and_core_conditions_are_preserved(self):
        packet = build_context(system_prompt='规则', question='预算改为 150000 元',
            task={'constraints': ['A 与 B 不能同时选择']}, actions={'clarify': '追问'}, observations=[],
            history=[{'answer': '旧内容' * 1000}], memories=[], max_input_tokens=600)
        payload = json.loads(packet.messages[1]['content'])
        self.assertEqual('预算改为 150000 元', payload['current_user_message'])
        self.assertEqual(['A 与 B 不能同时选择'], payload['task']['constraints'])
        self.assertEqual(1, packet.stats['dropped_items']['recent_turns'])

    def test_oversized_core_is_explicitly_rejected(self):
        with self.assertRaises(ContextBudgetExceeded):
            build_context(system_prompt='规则', question='供给 20 吨' * 1000, task={}, actions={},
                          observations=[], history=[], memories=[], max_input_tokens=600)


class MemoryTests(unittest.TestCase):
    """验证用户隔离、匿名会话隔离、删除和清理生命周期。"""

    def setUp(self):
        self.temp = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(database, 'DB_PATH', Path(self.temp.name) / 'memory.sqlite3')
        self.db_patch.start()
        init_db()

    def tearDown(self):
        self.db_patch.stop()
        self.temp.cleanup()

    def test_anonymous_memories_are_scoped_to_conversation(self):
        first = create_conversation('第一会话')['id']
        second = create_conversation('第二会话')['id']
        memory = add_memory(user_id=None, conversation_id=first, content='运输单位使用吨', kind='terminology')
        self.assertEqual([memory['id']], [row['id'] for row in search_memories(user_id=None, conversation_id=first, query='运输')])
        self.assertEqual([], search_memories(user_id=None, conversation_id=second))
        self.assertFalse(delete_memory(memory['id'], None, second))
        self.assertTrue(delete_memory(memory['id'], None, first))
        with self.assertRaises(PermissionError):
            search_memories(user_id=None, conversation_id=None)

    def test_user_memory_isolated_and_conversation_deletion_removes_local_memory(self):
        first_user = database.login_user('用户甲')['id']
        second_user = database.login_user('用户乙')['id']
        conversation = create_conversation('甲任务', user_id=first_user)['id']
        add_memory(user_id=first_user, conversation_id=None, content='优先解释成本', kind='preference')
        add_memory(user_id=first_user, conversation_id=conversation, content='本项目单位为吨', kind='terminology')
        self.assertEqual(2, len(search_memories(user_id=first_user, conversation_id=conversation)))
        self.assertEqual([], search_memories(user_id=second_user, conversation_id=None))
        with self.assertRaises(PermissionError):
            load_task(second_user, conversation)
        database.delete_conversation(conversation, user_id=first_user)
        self.assertEqual(1, len(search_memories(user_id=first_user, conversation_id=None)))

    def test_memory_api_and_invalid_token(self):
        conversation = create_conversation('匿名任务')['id']
        with TestClient(app) as client:
            response = client.post('/api/agent/memories', json={'conversation_id': conversation, 'content': '运输使用吨'})
            self.assertEqual(200, response.status_code)
            memory_id = response.json()['memory']['id']
            self.assertEqual(1, len(client.get('/api/agent/memories', params={'conversation_id': conversation}).json()['memories']))
            self.assertEqual(401, client.get('/api/agent/memories', params={'conversation_id': conversation}, headers={'x-session-token': 'invalid'}).status_code)
            self.assertEqual(200, client.delete(f'/api/agent/memories/{memory_id}', params={'conversation_id': conversation}).status_code)
