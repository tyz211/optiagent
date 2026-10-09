"""在临时数据库演示 LLM 主控；默认脚本化响应，--live 复用项目已有配置。"""

from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import sqlite3
import sys
import tempfile
from unittest.mock import patch


# 演示不修改开发环境的对话、需求版本、配置或运行记录。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from api import database
from api.database import create_conversation, init_db
from api.services.llm_controller import ControllerDecision, ControllerLimits, run_llm_controller
from optiagent.llm import LLMConfig, list_openai_compatible_models, llm_config_from_record
from optiagent.requirement_patch import RequirementEdit, RequirementPatch


QUESTION = '请求解背包：' + json.dumps({'knapsack': {'capacity': 5, 'items': [
    {'item': 'A', 'value': 8, 'weight': 3}, {'item': 'B', 'value': 5, 'weight': 2}]}})


def main() -> int:
    """仅输出安全的效果摘要，真实调用与模拟调用分别标记。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--live', action='store_true', help='使用已经授权复用的现有模型配置执行合成案例')
    parser.add_argument('--user-id', type=int, help='读取指定用户的已有配置；省略时使用匿名作用域配置')
    parser.add_argument('--followup', action='store_true', help='追加结构化修改和已验算方案解释的三轮演示')
    parser.add_argument('--list-models', action='store_true', help='只查询所选账户服务的真实模型目录')
    parser.add_argument('--model', help='真实测试时临时选用指定模型，不改动账户配置')
    args = parser.parse_args()
    if (args.list_models or args.model) and not args.live:
        parser.error('--list-models 和 --model 必须与 --live 一起使用')
    config = LLMConfig(True, 'demo-placeholder', 'https://example.test/v1', 'scripted-model')
    if args.live:
        path = ROOT / 'data/optiagent.sqlite3'
        with sqlite3.connect(f'file:{path}?mode=ro', uri=True) as conn:
            conn.row_factory = sqlite3.Row
            clause = 'user_id IS NULL' if args.user_id is None else 'user_id = ?'
            params = () if args.user_id is None else (args.user_id,)
            row = conn.execute(f'SELECT * FROM llm_configs WHERE is_active=1 AND {clause} ORDER BY id DESC LIMIT 1', params).fetchone()
        config = llm_config_from_record(dict(row) if row else None)
        if config is None:
            print(json.dumps({'mode': 'live', 'error': '当前作用域没有已有模型配置'}, ensure_ascii=False))
            return 1
        if args.list_models:
            # 网络错误只输出类型和 HTTP 状态，禁止回显供应商响应及配置凭据。
            try:
                models = list_openai_compatible_models(config)
            except Exception as exc:
                response = getattr(exc, 'response', None)
                print(json.dumps({'mode': 'live', 'error_type': type(exc).__name__,
                                  'http_status': getattr(response, 'status_code', None)}, ensure_ascii=False))
                return 1
            print(json.dumps({'mode': 'live', 'configured_model': config.model, 'models': models}, ensure_ascii=False))
            return 0
        if args.model:
            config = replace(config, model=args.model)
    with tempfile.TemporaryDirectory() as directory, patch.object(database, 'DB_PATH', Path(directory) / 'demo.sqlite3'):
        init_db()
        conversation_id = create_conversation('主控合成演示')['id']
        turns = [(QUESTION, 13), ('将背包容量调整为 3' if args.followup else '把容量改成 3', 8)]
        if args.followup:
            turns.append(('解释上一轮方案的容量使用', None))
        for turn, (question, expected) in enumerate(turns, start=1):
            outputs = [ControllerDecision(action=action, reason='演示行动', query='', message='')
                       for action in ('inspect_data', 'analyze_requirements', 'build_model', 'solve', 'verify', 'finish')]
            if args.followup and turn == 2:
                # 用精确原文和字符位置提出修改，完整版本提交仍由工具验证。
                proposal = RequirementPatch(expected_revision=1, template_id='knapsack', mode='solve', edits=[
                    RequirementEdit(field='capacity', entity='', target='', value=3,
                        source_quote=question, source_start=0, source_end=len(question))])
                outputs = [ControllerDecision(action=action, reason='演示原文修改', query='', message='',
                    patch=proposal if action == 'apply_requirement_patch' else None)
                    for action in ('read_requirement', 'apply_requirement_patch', 'build_model', 'solve', 'verify', 'finish')]
            elif args.followup and turn == 3:
                # 解释只选择真实工具返回的事实，由程序渲染目标、容量与引用。
                outputs = [ControllerDecision(action=action, reason='演示方案依据', query='', message='',
                    fact_ids=['objective', 'metric:capacity', 'metric:used_weight', 'decision:0', 'verification']
                    if action == 'explain_result' else None)
                    for action in ('read_verified_result', 'explain_result', 'finish')]
            def run():
                # 真实模式只发送此处的合成数据，读取配置后立即切换临时数据库。
                return run_llm_controller(question=question, requested_dataset_id=None, mcp_config='',
                    user_id=None, conversation_id=conversation_id, llm_config=config,
                    limits=ControllerLimits(max_decisions=10))
            if args.live:
                result = run()
            else:
                with patch('api.services.llm_controller.choose_action', side_effect=outputs):
                    result = run()
            passed = result.get('objective_value') == expected and bool(result.get('workflow_verification', {}).get('passed'))
            if expected is None:
                passed = passed and result['status'] == 'ANALYSIS' and bool(result.get('fact_citations'))
            print(json.dumps({'mode': 'live' if args.live else 'scripted', 'turn': turn,
                'status': result['status'], 'objective_value': result.get('objective_value'),
                'verified': result.get('workflow_verification', {}).get('passed'), 'passed': passed,
                'decision_count': result['agent_controller']['decision_count'],
                'actions': [item['action'] for item in result['agent_policy']['decisions']],
                'stop_reason': result['agent_controller']['stop_reason'],
                'requirement_revision': result.get('requirement_analysis', {}).get('dialogue_contract', {}).get('revision'),
                'source_run_id': result.get('result_reference', {}).get('run_id'),
                'fact_citation_count': len(result.get('fact_citations', [])),
                'error': result['agent_controller'].get('error')}, ensure_ascii=False))
            if not passed:
                return 1
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
