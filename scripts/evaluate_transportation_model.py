"""针对显式指定的本机模型运行运输抽取评测，不启用远程兜底。"""

from __future__ import annotations

import argparse
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import platform
import statistics
import sys
from time import perf_counter
from urllib.parse import urlparse


# 与应用共用抽取器，确保评测覆盖真实接入协议。
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))
from optiagent.llm import LLMConfig
from optiagent.transportation import TransportationData
from optiagent.transportation_llm import propose_transportation


def normalized(data: dict) -> dict:
    """对集合型行排序后比较，避免把等价的节点顺序误判成语义错误。"""
    result = TransportationData.model_validate(data).model_dump()
    for field in ('suppliers', 'consumers', 'routes', 'forbidden_routes'):
        result[field] = sorted(result[field], key=lambda row: json.dumps(row, sort_keys=True, ensure_ascii=False))
    return result


def evaluate(cases: list[dict], config: LLMConfig) -> dict:
    """分别计算完整数据抽取、缺失信息与不支持条件的正确处理。"""
    rows = []
    for case in cases:
        start = perf_counter()
        row = {'id': case['id'], 'expected': case['expected'], 'passed': False}
        try:
            draft = propose_transportation(case['question'], config)
            row['output'] = draft.model_dump()
            if case['expected'] == 'draft':
                expected = json.loads((ROOT / case['data_file']).read_text())['transportation'] if 'data_file' in case else case['data']
                row['passed'] = bool(draft.data and not draft.missing_information and not draft.unsupported_requirements
                                     and normalized(draft.data.model_dump()) == normalized(expected))
            elif case['expected'] == 'missing':
                row['passed'] = draft.data is None and bool(draft.missing_information)
            elif case['expected'] == 'unsupported':
                row['passed'] = bool(draft.unsupported_requirements)
        except Exception as exc:
            # 报告只保存故障类型，不记录服务响应中的潜在凭据。
            row['error_type'] = type(exc).__name__
        row['elapsed_ms'] = round((perf_counter() - start) * 1000, 3)
        rows.append(row)
    latencies = sorted(row['elapsed_ms'] for row in rows)
    return {
        'schema_version': '1.0', 'mode': 'local_llm', 'model': config.model, 'temperature': config.temperature,
        'created_at': datetime.now(timezone.utc).isoformat(), 'platform': platform.platform(),
        'case_count': len(rows), 'passed_count': sum(row['passed'] for row in rows),
        'llm_attempts': len(rows), 'remote_fallback': False,
        'p50_ms': statistics.median(latencies) if latencies else None,
        'p95_ms': latencies[min(len(latencies) - 1, math.ceil(len(latencies) * .95) - 1)] if latencies else None,
        'limitations': ['八条开发样例，不是冻结测试集。', '未测显存或内存峰值；延迟包括连接、生成与校验，未区分冷启动。',
                        '抽取正确不代表模型已证明任意自然语言等价；应用仍要求核对模型草稿。'],
        'cases': rows,
    }


def main():
    """只测用户显式指定的本机模型，不自动下载或选取模型。"""
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--base-url', default='http://127.0.0.1:11434/v1')
    parser.add_argument('--model', required=True)
    parser.add_argument('--output', type=Path, default=ROOT / 'data/benchmarks/transportation-model.json')
    args = parser.parse_args()
    if urlparse(args.base_url).hostname not in {'localhost', '127.0.0.1', '::1'}:
        parser.error('首版评测只允许本机推理服务')
    fixture = ROOT / 'tests/fixtures/transportation_model_cases.json'
    raw = fixture.read_bytes()
    report = evaluate(json.loads(raw), LLMConfig(True, '', args.base_url, args.model, 0))
    report['dataset_sha256'] = hashlib.sha256(raw).hexdigest()
    args.output.parent.mkdir(parents=True, exist_ok=True)
    args.output.write_text(json.dumps(report, ensure_ascii=False, indent=2) + '\n')
    print(json.dumps({key: report[key] for key in ('model', 'case_count', 'passed_count', 'p50_ms', 'p95_ms')}, ensure_ascii=False))
    return 0 if report['passed_count'] == report['case_count'] else 1


if __name__ == '__main__':
    raise SystemExit(main())
