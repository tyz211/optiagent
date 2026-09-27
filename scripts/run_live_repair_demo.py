from __future__ import annotations

import json
from pathlib import Path
import sys

# 子进程具有自己的模块、临时数据库与故障映射，不修改 API 进程的全局图。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))


def emit(kind: str, payload: dict):
    """前缀区分协议事件与第三方求解器的启动提示。"""
    print('OPTIAGENT_DEMO ' + json.dumps({'event': kind, 'data': payload}, ensure_ascii=False, allow_nan=False), flush=True)


def main():
    """只执行一次真实案例，实时返回节点、修复证据及最终结果。"""
    from api.services.repair_demo import RepairDemoRequest, checkpoints, policy_available
    from optiagent.rl.instances import generate_recovery_instance
    from optiagent.rl.e2e_benchmark import build_smoke_instances
    from optiagent.rl.study import workflow_instance_data
    from optiagent.rl.data_repair import corrupt_mapping
    from optiagent.instance_identity import instance_fingerprint
    from scripts.evaluate_workflow_policy import evaluate
    request = RepairDemoRequest.model_validate_json(sys.argv[1])
    if not policy_available(request.policy):
        raise ValueError('所选检查点不可用。')
    if request.policy != 'rule':
        import torch
        torch.set_num_threads(1)
    instance = generate_recovery_instance(request.template_id, seed=request.seed, split='test', index=request.instance_index)
    question = next(item.question for item in build_smoke_instances() if item.template_id == request.template_id)
    instance = instance.model_copy(update={'question': question, 'data': workflow_instance_data(instance.template_id, instance.data)})
    # 界面可查看本次生成的数据和错误映射；这两个对象都不会混入策略网络的输入。
    altered = corrupt_mapping(instance.template_id, instance.data)
    emit('input', {'task_id': instance.task_id, 'source_data': instance.data, 'corrupted_data': altered,
                   'source_fingerprint': instance_fingerprint(instance.template_id, instance.data),
                   'corrupted_fingerprint': instance_fingerprint(instance.template_id, altered)})
    report = evaluate('', request.seed, instances=[instance], policy_checkpoints={request.policy: checkpoints()[request.policy]},
                      benchmark_profile='data_repair_v1', selected_scenarios=(request.scenario,), retain_trajectories=False,
                      event_callback=lambda event: emit('node', event), evidence_callback=lambda event: emit('evidence', event))
    emit('final', {'run_id': sys.argv[2], 'mode': 'live_controlled_demo', 'request': request.model_dump(),
                   'case': report['cases'][0], 'checkpoint_sha256': report['checkpoint_hashes'][request.policy],
                   'llm_used': False, 'persisted_to_live_database': False})


if __name__ == '__main__':
    try:
        main()
    except Exception as exc:
        # 向界面返回明确但脱敏的错误；不泄露服务器路径或原始异常内容。
        emit('error', {'message': '演示执行失败，请检查模型或求解器配置。', 'error_type': type(exc).__name__})
        raise SystemExit(1)
