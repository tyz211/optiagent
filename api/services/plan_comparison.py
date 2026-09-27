from __future__ import annotations

import json

from api.database import list_runs


def compare_plans(user_id, conversation_id, current=None):
    """只比较当前会话同一模板、验算通过的方案；澄清和失败记录不冒充基准。"""
    if conversation_id is None:
        return {'available': False, 'message': '请在同一对话中至少完成两次有效求解。'}
    candidates = []
    for row in list_runs(limit=100, user_id=user_id, conversation_id=conversation_id):
        try:
            result = json.loads(row.get('result_json') or '{}')
        except (ValueError, TypeError):
            continue
        if not isinstance(result, dict):
            continue
        if current and row['id'] == current.get('run_id'):
            continue
        verification = result.get('solution_verification') or {}
        if verification.get('passed') and result.get('workflow_verification', {}).get('passed') and result.get('objective_value') is not None:
            result['run_id'] = row['id']
            candidates.append(result)
    if current is None:
        if not candidates:
            return {'available': False, 'message': '当前会话尚无通过验算的方案，请先完成求解。'}
        current = candidates.pop()
    template = (current.get('problem_spec') or {}).get('template_id')
    before = next((item for item in reversed(candidates) if (item.get('problem_spec') or {}).get('template_id') == template), None)
    if before is None:
        return {'available': False, 'message': '当前已有一个有效方案，再修改数据并求解即可对比。'}
    old_value, new_value = before['objective_value'], current['objective_value']
    old_contract = (before.get('requirement_analysis') or {}).get('dialogue_contract') or {}
    new_contract = (current.get('requirement_analysis') or {}).get('dialogue_contract') or {}
    fields = sorted(set(old_contract.get('data', {})) | set(new_contract.get('data', {})))
    changed = [name for name in fields if old_contract.get('data', {}).get(name) != new_contract.get('data', {}).get(name)]
    delta = new_value - old_value
    return {'available': True, 'template_id': template, 'before_run_id': before['run_id'],
            'after_run_id': current['run_id'], 'before_revision': old_contract.get('revision'),
            'after_revision': new_contract.get('revision'), 'before_objective': old_value, 'after_objective': new_value,
            'delta': delta, 'changed_fields': changed,
            'input_diff_available': bool(old_contract.get('data') and new_contract.get('data')),
            'message': f'方案 #{before["run_id"]} → #{current["run_id"]}：目标值 {old_value:g} → {new_value:g}，变化 {delta:+g}。',
            'note': '两次结果均通过各自输入的验算；参数或约束可能不同，目标值变化不等于算法性能提升。'}
