from __future__ import annotations

import json
from pathlib import Path
import tempfile
import unittest
from unittest.mock import patch

from fastapi.testclient import TestClient
from api import database
from api.main import app
from optiagent.requirement_analysis import analyze_requirements


DATA = {'capacity': 5, 'items': [{'item': 'A', 'value': 8, 'weight': 3}, {'item': 'B', 'value': 5, 'weight': 2}]}
QUESTION = '求解背包问题，最大化价值。数据：' + json.dumps(DATA, ensure_ascii=False)


class DialogueContractTests(unittest.TestCase):
    """覆盖有效输入继承、原子更新、撤销链与不支持编辑的求解门控。"""

    def test_latest_json_replaces_old_data_and_repeated_solves_inherit_it(self):
        first = analyze_requirements(QUESTION)
        changed = {**DATA, 'capacity': 3}
        second = analyze_requirements('替换数据：' + json.dumps(changed), previous=first)
        third = analyze_requirements('继续求解', previous=second)
        self.assertEqual(3, third.dialogue_contract['data']['capacity'])
        self.assertEqual(2, third.dialogue_contract['revision'])
        self.assertEqual(1, third.resolved_request.count('"capacity"'))
        self.assertEqual('ready_to_solve', third.readiness)
        self.assertEqual(5, first.dialogue_contract['data']['capacity'])

    def test_invalid_and_ambiguous_edits_preserve_version(self):
        first = analyze_requirements(QUESTION)
        for text in ['把容量改成 -1', '把容量从 7 改成 3', '把容量改成 3 万',
                     '删除容量约束', '必须选择 B', '把容量改成 3，必须选择 B', '{"capacity": 3}']:
            with self.subTest(text=text):
                result = analyze_requirements(text, previous=first)
                self.assertEqual('needs_clarification', result.readiness)
                self.assertEqual(5, result.dialogue_contract['data']['capacity'])
                self.assertEqual(1, result.dialogue_contract['revision'])

    def test_two_consecutive_undos_restore_parent_chain(self):
        brief = analyze_requirements(QUESTION)
        for text in ['把容量改成 4', '把容量改成 3', '撤销上次修改']:
            brief = analyze_requirements(text, previous=brief)
        self.assertEqual(4, brief.dialogue_contract['data']['capacity'])
        brief = analyze_requirements('撤销上次修改', previous=brief)
        self.assertEqual(5, brief.dialogue_contract['data']['capacity'])
        brief = analyze_requirements('撤销上次修改', previous=brief)
        self.assertEqual('needs_clarification', brief.readiness)
        self.assertEqual(5, brief.dialogue_contract['revision'])


class DialogueIntegrationTests(unittest.TestCase):
    """通过 API 使用真实求解器，确认参数变化真正改变解而非仅改变页面文字。"""

    def setUp(self):
        self.temporary = tempfile.TemporaryDirectory()
        self.db_patch = patch.object(database, 'DB_PATH', Path(self.temporary.name) / 'dialogue.sqlite3')
        self.db_patch.start()
        self.client = TestClient(app)
        self.client.__enter__()
        self.cid = self.client.post('/api/conversations', json={'title': '多轮验收'}).json()['conversation']['id']

    def tearDown(self):
        self.client.__exit__(None, None, None)
        self.db_patch.stop()
        self.temporary.cleanup()

    def ask(self, question, cid=None):
        response = self.client.post('/api/ask', json={'question': question, 'conversation_id': cid or self.cid})
        self.assertEqual(200, response.status_code, response.text)
        return response.json()

    def test_solve_edit_hold_resume_compare_undo_and_reload(self):
        first = self.ask(QUESTION)
        self.assertEqual(13, first['objective_value'])
        held = self.ask('把容量改成 3，先不求解')
        self.assertEqual('REQUIREMENT_UPDATED', held['status'])
        self.assertNotIn('solver', [node['node_id'] for node in held['agent_graph']['nodes']])
        second = self.ask('继续求解')
        self.assertEqual(8, second['objective_value'])
        self.assertEqual(-5, second['plan_comparison']['delta'])
        self.assertEqual(['capacity'], second['plan_comparison']['changed_fields'])
        compared = self.ask('比较最近两个方案')
        self.assertEqual('COMPARISON', compared['status'])
        self.assertNotIn('solver', [node['node_id'] for node in compared['agent_graph']['nodes']])
        self.assertEqual(first['run_id'], compared['plan_comparison']['before_run_id'])
        invalid = self.ask('删除容量约束')
        self.assertEqual('NEEDS_CLARIFICATION', invalid['status'])
        self.assertNotIn('solver', [node['node_id'] for node in invalid['agent_graph']['nodes']])
        restored = self.ask('撤销上次修改')
        self.assertEqual(13, restored['objective_value'])
        state = self.client.get(f'/api/conversations/{self.cid}/requirements').json()['requirement_analysis']
        self.assertEqual(5, state['dialogue_contract']['data']['capacity'])
        rows = database.list_runs(user_id=None, conversation_id=self.cid)
        saved = json.loads(rows[-1]['result_json'])
        self.assertEqual(restored['plan_comparison'], saved['plan_comparison'])

    def test_conversations_do_not_share_inputs_or_comparison(self):
        self.ask(QUESTION)
        other = self.client.post('/api/conversations', json={'title': '隔离'}).json()['conversation']['id']
        result = self.ask('继续求解', cid=other)
        self.assertEqual('NEEDS_CLARIFICATION', result['status'])
        self.assertFalse(result['requirement_analysis']['dialogue_contract'])
        isolated = self.ask(QUESTION, cid=other)
        self.assertFalse(isolated['plan_comparison']['available'])

    def test_second_full_json_is_consumed_by_solver(self):
        self.ask(QUESTION)
        second = self.ask('替换数据：' + json.dumps({**DATA, 'capacity': 2}))
        self.assertEqual(5, second['objective_value'])
        self.assertTrue(second['solution_verification']['passed'])

    def test_run_limit_returns_recent_records_in_chronological_order(self):
        for i in range(7):
            database.save_run(None, str(i), '', {}, conversation_id=self.cid)
        self.assertEqual(['4', '5', '6'], [r['question'] for r in database.list_runs(limit=3, conversation_id=self.cid)])

    def test_all_five_inline_templates_reuse_current_input(self):
        from optiagent.rl.e2e_benchmark import build_smoke_instances
        for instance in build_smoke_instances():
            if instance.template_id == 'facility_location':
                continue
            with self.subTest(template=instance.template_id):
                cid = self.client.post('/api/conversations', json={'title': instance.template_id}).json()['conversation']['id']
                first = self.ask(instance.question + '\n' + json.dumps(instance.data), cid=cid)
                second = self.ask('继续求解', cid=cid)
                self.assertTrue(first['solution_verification']['passed'])
                self.assertTrue(second['solution_verification']['passed'])
                self.assertEqual(first['objective_value'], second['objective_value'])
                self.assertEqual(instance.template_id, second['problem_spec']['template_id'])

    def test_unrelated_dataset_cannot_override_inline_revision(self):
        # 不存在的数据集编号也不应被加载：当前显式输入具有更高优先级。
        response = self.client.post('/api/ask', json={'question': QUESTION, 'conversation_id': self.cid, 'dataset_id': 99999})
        self.assertEqual(200, response.status_code)
        self.assertEqual(13, response.json()['objective_value'])


class ConversationGuardTests(unittest.TestCase):
    """在同一进程中，重叠轮次必须被拒绝且异常后释放占用。"""

    def test_reentrant_same_conversation_rejected_other_conversation_allowed(self):
        from fastapi import HTTPException
        from api.services.conversation_guard import serialize_conversation

        @serialize_conversation
        def execute(*, conversation_id, user_id=None, recurse=False):
            if recurse:
                self.assertEqual(2, execute(conversation_id=2))
                with self.assertRaises(HTTPException) as caught:
                    execute(conversation_id=conversation_id)
                self.assertEqual(409, caught.exception.status_code)
                raise ValueError('模拟工作流异常')
            return conversation_id

        with self.assertRaises(ValueError):
            execute(conversation_id=1, recurse=True)
        self.assertEqual(1, execute(conversation_id=1))


if __name__ == '__main__':
    unittest.main()
