from __future__ import annotations

import argparse
from datetime import datetime, timezone
from importlib import metadata
from importlib.util import find_spec
import json
import os
from pathlib import Path
import platform
import shutil
import subprocess
import sys
import tempfile
import unittest
from unittest.mock import patch

# 从任意工作目录调用时，仍使用当前源码包和当前 Python 解释器。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from scripts.release_support import check_models, configuration, locked_versions, verify_snapshot


# 基础验收明确排除神经网络训练模块；完整验收必须执行全部测试且不能跳过。
RL_TESTS = {'test_learned_policy.py', 'test_mcp_transport_training.py',
            'test_real_recovery_training.py', 'test_recovery_runtime.py', 'test_offline_training.py'}


def require(condition: bool, message: str) -> None:
    """使用显式异常，避免 Python 优化模式关闭验收断言。"""
    if not condition:
        raise RuntimeError(message)


def environment_check(profile: str) -> dict:
    """检查实际依赖版本；未安装或版本漂移不能被记录为复现成功。"""
    lock = 'requirements-rl.lock' if profile == 'full' else 'requirements-demo.lock'
    expected = locked_versions(ROOT, lock)
    mismatch = []
    for name, version in expected.items():
        try:
            actual = metadata.version(name)
        except metadata.PackageNotFoundError:
            actual = 'missing'
        if actual != version:
            mismatch.append(f'{name}: expected={version}, actual={actual}')
    require(not mismatch, '依赖不符合锁定版本：' + '; '.join(mismatch))
    require(sys.version_info[:2] == (3, 14), '本次交付基线要求 Python 3.14；其他版本需另行验证。')
    subprocess.run([sys.executable, '-m', 'pip', 'check'], cwd=ROOT, check=True)
    result = {'python': platform.python_version(), 'system': platform.system(),
              'machine': platform.machine(), 'locked_packages': len(expected),
              'torch_installed': find_spec('torch') is not None, 'snapshot': verify_snapshot()}
    if profile == 'full':
        result['checkpoints'] = check_models()
    return result


def regression_check(profile: str) -> dict:
    """测试只使用临时数据库；核心与完整模式的覆盖范围进入报告。"""
    from api import database
    files = sorted((ROOT / 'tests').glob('test_*.py'))
    excluded = sorted(path.name for path in files if profile == 'core' and path.name in RL_TESTS)
    suite = unittest.TestSuite()
    loader = unittest.TestLoader()
    for path in files:
        if path.name not in excluded:
            suite.addTests(loader.discover(str(ROOT / 'tests'), pattern=path.name))
    with tempfile.TemporaryDirectory(prefix='optiagent-regression-') as directory:
        with patch.object(database, 'DB_PATH', Path(directory) / 'regression.sqlite3'):
            result = unittest.TextTestRunner(verbosity=1).run(suite)
    details = {'tests_run': result.testsRun, 'failures': len(result.failures),
               'errors': len(result.errors), 'skipped': len(result.skipped), 'excluded_modules': excluded}
    print(json.dumps(details, ensure_ascii=False))
    require(result.wasSuccessful() and not result.skipped, '回归测试失败或出现未声明的跳过。')
    return details


def events(text: str) -> list[tuple[str, dict]]:
    """按 SSE 的事件与数据字段解析，验证公开 HTTP 合同。"""
    result = []
    for block in text.replace('\r\n', '\n').split('\n\n'):
        kind = next((line[7:] for line in block.splitlines() if line.startswith('event: ')), None)
        data = next((line[6:] for line in block.splitlines() if line.startswith('data: ')), None)
        if kind and data:
            result.append((kind, json.loads(data)))
    return result


def http_check(profile: str) -> dict:
    """通过 ASGI HTTP 接口检查页面、对话流和真实修复子进程，不连接用户数据库。"""
    from api import database
    from api.main import app
    from fastapi.testclient import TestClient

    repair_cases = []
    with tempfile.TemporaryDirectory(prefix='optiagent-http-') as directory:
        with patch.object(database, 'DB_PATH', Path(directory) / 'http.sqlite3'), TestClient(app) as client:
            for route in ('/', '/repair-demo', '/static/app.js', '/static/repair-demo.js', '/api/health'):
                require(client.get(route).status_code == 200, f'页面或接口不可用：{route}')
            catalog = client.get('/api/demo/repair/catalog').json()
            available = {item['id']: item['available'] for item in catalog['policies']}
            require(available['rule'], '规则策略必须始终可用。')
            if profile == 'full':
                require(all(available.values()), '完整验收必须提供两份学习策略检查点。')
            response = client.post('/api/conversations', json={'title': '隔离交付验收'})
            require(response.status_code == 200, '创建对话失败。')
            conversation_id = response.json()['conversation']['id']
            payload = {'capacity': 5, 'items': [{'item': 'A', 'value': 8, 'weight': 3},
                                                {'item': 'B', 'value': 5, 'weight': 2}]}
            response = client.post('/api/ask/stream', json={
                'question': '求解背包，最大化总价值。数据如下：' + json.dumps(payload, ensure_ascii=False),
                'conversation_id': conversation_id})
            require(response.status_code == 200, '主对话流接口失败。')
            final = [data for kind, data in events(response.text) if kind == 'final']
            require(len(final) == 1 and final[0]['objective_value'] == 13
                    and final[0]['solution_verification']['passed'], '主对话未返回验算通过的目标值 13。')
            for policy in (('rule', 'previous', 'learned') if profile == 'full' else ('rule',)):
                for scenario in ('model_error', 'persistent_failure'):
                    response = client.post('/api/demo/repair/stream', json={
                        'policy': policy, 'scenario': scenario, 'template_id': 'knapsack',
                        'seed': 7300, 'instance_index': 0})
                    require(response.status_code == 200, f'修复接口失败：{policy}/{scenario}')
                    stream = events(response.text)
                    require(not any(kind == 'error' for kind, _ in stream), '修复子进程返回错误。')
                    finals = [data for kind, data in stream if kind == 'final']
                    require(len(finals) == 1, '修复流缺少唯一终局结果。')
                    case = finals[0]['case']
                    require(case['valid_actions'] and not case['unverified_accept'] and not case['fallbacks'],
                            '恢复动作不合法、发生回退或接受了未验证的结果。')
                    expected_success = scenario == 'model_error' and policy != 'previous'
                    require(case['success'] == expected_success, f'固定演示案例的行为发生变化：{policy}/{scenario}')
                    repair_cases.append({'policy': policy, 'scenario': scenario, 'success': case['success'],
                                         'actions': case['actions'], 'checkpoint_sha256': finals[0]['checkpoint_sha256']})
    return {'main_dialogue_objective': 13, 'repair_cases': repair_cases, 'llm_used': False}


def worker(stage: str, profile: str, result_path: Path) -> None:
    """每个阶段独立执行，错误与成功均生成机器可读结果。"""
    try:
        if stage == 'environment':
            details = environment_check(profile)
        elif stage == 'tests':
            details = regression_check(profile)
        elif stage == 'http':
            details = http_check(profile)
        elif stage == 'dialogue':
            subprocess.run([sys.executable, 'scripts/run_dialogue_demo.py', '--output', str(result_path)],
                           cwd=ROOT, check=True)
            return
        else:
            # Node 仅用于静态语法检查；缺失时清楚记录，不能伪称已检查。
            node = shutil.which('node')
            if node:
                for name in ('web/app.js', 'web/repair-demo.js'):
                    subprocess.run([node, '--check', name], cwd=ROOT, check=True)
            subprocess.run(['bash', '-n', 'start.sh'], cwd=ROOT, check=True)
            details = {'javascript': 'passed' if node else 'not_checked_node_unavailable', 'shell': 'passed'}
        result = {'status': 'passed', 'details': details}
    except Exception as exc:
        result_path.write_text(json.dumps({'status': 'failed', 'error': str(exc)}, ensure_ascii=False, indent=2), encoding='utf-8')
        raise
    result_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')


def main() -> None:
    """统一验收不重跑研究训练、不调用 LLM，不修改已有验收记录或固定权重。"""
    parser = argparse.ArgumentParser(description='验收 OptiAgent 可复现 Demo')
    parser.add_argument('--profile', choices=('core', 'full'), default='core')
    parser.add_argument('--output', type=Path)
    parser.add_argument('--worker', choices=('environment', 'syntax', 'tests', 'dialogue', 'http'), help=argparse.SUPPRESS)
    args = parser.parse_args()
    os.chdir(ROOT)
    if args.worker:
        worker(args.worker, args.profile, args.output)
        return
    output = (args.output or ROOT / 'artifacts/validation' / datetime.now().strftime('%Y%m%d-%H%M%S-%f')).resolve()
    output.mkdir(parents=True, exist_ok=False)
    # 固定演示配置，避免操作者原有环境变量改变验收目标或连接远程 MCP。
    environment = {key: value for key, value in os.environ.items() if not key.startswith('OPTIAGENT_')}
    environment.update(OPTIAGENT_RECOVERY_CHECKPOINT='', OPTIAGENT_SOLVER_THREADS='1',
                       PYTHONNOUSERSITE='1', LANGSMITH_TRACING='false', LANGCHAIN_TRACING_V2='false')
    environment.pop('PYTHONPATH', None)
    for name, record in configuration()['checkpoints'].items():
        key = 'OPTIAGENT_DEMO_CHECKPOINT' if name == 'learned' else 'OPTIAGENT_DEMO_PREVIOUS_CHECKPOINT'
        environment[key] = str(ROOT / record['path']) if args.profile == 'full' else str(output / 'not-configured.pt')
    report = {'schema_version': '1.0', 'profile': args.profile,
              'started_at': datetime.now(timezone.utc).isoformat(), 'status': 'running', 'stages': []}
    for stage in ('environment', 'syntax', 'tests', 'dialogue', 'http'):
        print(f'正在验收：{stage}', flush=True)
        path = output / f'{stage}.json'
        with (output / f'{stage}.log').open('x', encoding='utf-8') as log:
            try:
                run = subprocess.run([sys.executable, str(Path(__file__).resolve()), '--profile', args.profile,
                                      '--worker', stage, '--output', str(path)], cwd=ROOT, env=environment,
                                     stdout=log, stderr=subprocess.STDOUT, timeout=300)
                code = run.returncode
            except subprocess.TimeoutExpired:
                code = 124
        result = json.loads(path.read_text(encoding='utf-8')) if path.exists() else {'status': 'failed', 'error': '阶段超时或未返回结果'}
        report['stages'].append({'name': stage, 'returncode': code, **result})
        if code or result['status'] != 'passed':
            report['status'] = 'failed'
            break
    else:
        report['status'] = 'passed'
    report['completed_at'] = datetime.now(timezone.utc).isoformat()
    (output / 'summary.json').write_text(json.dumps(report, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({'status': report['status'], 'report': str(output / 'summary.json')}, ensure_ascii=False))
    if report['status'] != 'passed':
        raise SystemExit(1)


if __name__ == '__main__':
    main()
