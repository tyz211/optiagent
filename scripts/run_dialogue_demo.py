from __future__ import annotations

import argparse
from datetime import datetime
import json
from pathlib import Path
import sys
import tempfile
from unittest.mock import patch

# 从项目根目录直接执行；验收仅使用临时数据库，不触碰已有对话。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))


def main():
    """运行可重复的多轮验收，将实际求解与门控结果保存为独立报告。"""
    parser = argparse.ArgumentParser(description='验收多轮修改、求解、比较与撤销')
    parser.add_argument('--output', default=f'artifacts/rl/dialogue-demo/{datetime.now():%Y%m%d-%H%M%S}.json')
    args = parser.parse_args()
    from api import database
    from api.services.agent_workflow import run_agent_workflow
    data = {'capacity': 5, 'items': [{'item': 'A', 'value': 8, 'weight': 3}, {'item': 'B', 'value': 5, 'weight': 2}]}
    steps = [
        ('我想解决背包选择问题，目标是最大化总价值，总重量不能超过容量。', 'NEEDS_CLARIFICATION', None),
        ('数据如下：' + json.dumps(data, ensure_ascii=False), 'OPTIMAL', 13),
        ('把容量改成 3，先不求解', 'REQUIREMENT_UPDATED', None),
        ('继续求解', 'OPTIMAL', 8),
        ('比较最近两个方案', 'COMPARISON', None),
        ('删除容量约束', 'NEEDS_CLARIFICATION', None),
        ('撤销上次修改', 'OPTIMAL', 13),
    ]
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # 使用独占创建，防止覆盖上一份验收证据。
    with output.open('x', encoding='utf-8') as report_file:
        records = []
        try:
            with tempfile.TemporaryDirectory() as directory, patch.object(database, 'DB_PATH', Path(directory) / 'demo.sqlite3'):
                database.init_db()
                cid = database.create_conversation('多轮 Demo 验收')['id']
                for question, expected_status, expected_value in steps:
                    result = run_agent_workflow(question=question, requested_dataset_id=None, mcp_config='',
                                                user_id=None, conversation_id=cid, recovery_checkpoint='',
                                                trajectory_source='controlled_workflow_benchmark')
                    solver_called = any(node['node_id'] == 'solver' for node in result['agent_graph']['nodes'])
                    checked = (result.get('solution_verification') or {}).get('passed', False)
                    passed = (result['status'] == expected_status and result['objective_value'] == expected_value
                              and solver_called == (expected_value is not None)
                              and (expected_value is None or checked))
                    comparison = result.get('plan_comparison')
                    if expected_status == 'COMPARISON':
                        passed = passed and comparison['available'] and comparison['delta'] == -5
                    records.append({'question': question, 'status': result['status'], 'objective_value': result['objective_value'],
                                    'solver_called': solver_called, 'verified': checked, 'passed': passed,
                                    'revision': result['requirement_analysis']['dialogue_contract'].get('revision'),
                                    'comparison': comparison})
            report = {'status': 'passed' if all(row['passed'] for row in records) else 'failed',
                      'mode': 'generated_demo_data', 'llm_used': False, 'steps': records}
        except Exception as exc:
            report = {'status': 'failed', 'error_type': type(exc).__name__, 'steps': records}
        json.dump(report, report_file, ensure_ascii=False, indent=2)
    print(json.dumps({'status': report['status'], 'report': str(output), 'step_count': len(records)}, ensure_ascii=False))
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
