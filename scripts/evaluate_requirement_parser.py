"""可复现需求解析基线；可选真实求解，不将规则或模拟结果计作模型成绩。"""

from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import sys
from time import perf_counter


# 允许从任意目录执行，样本与输出均使用明确路径。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optiagent.optimization_gateway import build_problem_envelope, solve_problem_envelope
from optiagent.requirement_analysis import analyze_requirements


def evaluate(cases: list[dict], *, solve: bool = False) -> dict:
    """逐条比较人工预期，不因某条失败中断整份报告。"""

    rows = []
    for case in cases:
        question = (ROOT / case['question_file']).read_text() if 'question_file' in case else case['question']
        start = perf_counter()
        row = {'id': case['id'], 'group': case['group'], 'passed': False}
        try:
            brief = analyze_requirements(question)
            expected = case['expected_readiness']
            checks = {'readiness': brief.readiness == expected}
            if 'expected_template' in case:
                checks['template'] = brief.template_id == case['expected_template']
            if expected == 'needs_clarification':
                checks['no_executable_data'] = not bool(brief.dialogue_contract.get('data'))
            row.update(template_id=brief.template_id, readiness=brief.readiness, source=brief.source, checks=checks)
            row['parser_elapsed_ms'] = round((perf_counter() - start) * 1000, 3)
            if solve and 'expected_objective' in case:
                data = brief.dialogue_contract.get('data')
                if data and brief.readiness == 'ready_to_solve':
                    result = solve_problem_envelope(build_problem_envelope(brief.template_id, data))
                    checks['solution'] = bool(result.solution_verification and result.solution_verification.passed)
                    checks['objective'] = result.objective_value is not None and math.isclose(result.objective_value, case['expected_objective'], abs_tol=1e-6)
                    row.update(solver_status=result.status, objective_value=result.objective_value)
                else:
                    checks['solution'] = False
            row['passed'] = all(checks.values())
        except Exception as exc:
            row['error_type'] = type(exc).__name__
        row['elapsed_ms'] = round((perf_counter() - start) * 1000, 3)
        rows.append(row)
    groups = Counter(row['group'] for row in rows)
    return {
        'schema_version': '1.0', 'mode': 'local_rule', 'real_llm_calls': 0,
        'real_solver_enabled': solve, 'created_at': datetime.now(timezone.utc).isoformat(),
        'python': platform.python_version(), 'platform': platform.platform(),
        'case_count': len(rows), 'passed_count': sum(row['passed'] for row in rows),
        'groups': {group: {'count': count, 'passed': sum(r['passed'] for r in rows if r['group'] == group)} for group, count in groups.items()},
        'limitations': ['人工编写的开发回归集，不是冻结测试集或真实业务泛化成绩。',
                        '仅检查已标注的路由、门控与参考目标，不衡量任意自然语言语义等价。'],
        'cases': rows,
    }


def main():
    """执行开发基线并保存样本摘要；失败用非零退出码报告。"""

    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--cases', type=Path, default=ROOT / 'tests/fixtures/requirement_cases.json')
    parser.add_argument('--output', type=Path, default=ROOT / 'data/benchmarks/requirement-parser.json')
    parser.add_argument('--solve', action='store_true', help='对带参考目标的案例调用真实求解器')
    args = parser.parse_args()
    raw = args.cases.read_bytes()
    report = evaluate(json.loads(raw), solve=args.solve)
    report['dataset_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('mode', 'case_count', 'passed_count', 'real_llm_calls')}, ensure_ascii=False))
    for case in report['cases']:
        if not case['passed']:
            print(json.dumps(case, ensure_ascii=False))
    return 0 if report['passed_count'] == report['case_count'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
