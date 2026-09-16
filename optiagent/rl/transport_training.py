from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from typing import Any

from optiagent.rl.dqn import DQNConfig
from optiagent.rl.environment import ACTION_IDS
from optiagent.rl.rollout import RandomValidPolicy, recovery_baseline_policy
from optiagent.rl.state_encoder import TransportAwareRecoveryStateEncoder
from optiagent.rl.training import (
    RecoveryPolicyTrainingResult,
    evaluate_policy_splits,
    train_masked_dqn_core,
)
from optiagent.rl.transport_environment import (
    TRANSPORT_ENVIRONMENT_VERSION,
    MCPTransportRecoveryEnv,
    MCPTransportRecoveryTask,
    evaluate_transport_policy,
    transport_teacher_policy,
)


def train_transport_recovery_policy(
    tasks: list[MCPTransportRecoveryTask],
    config: DQNConfig | None = None,
) -> RecoveryPolicyTrainingResult:
    """使用真实 MCP stdio 故障轨迹训练 transport 恢复策略。"""

    selected_config = config or DQNConfig()
    encoder = TransportAwareRecoveryStateEncoder()
    core = train_masked_dqn_core(
        tasks,
        encoder=encoder,
        environment_factory=MCPTransportRecoveryEnv,
        teacher_policy=transport_teacher_policy,
        config=selected_config,
        split_error="MCP transport 训练需要非空的 train/validation/test 划分。",
    )
    agent = core.agent

    report = {
        "schema_version": "1.0",
        "algorithm": "behavior_cloning+action_masked_double_dqn+mcp_transport",
        "environment_version": TRANSPORT_ENVIRONMENT_VERSION,
        "state_encoder_version": encoder.version,
        "state_dimension": encoder.dimension,
        "action_ids": list(ACTION_IDS),
        "config": asdict(selected_config),
        "dataset": _dataset_summary(tasks),
        "training": core.metrics,
        "evaluation": {
            "learned_policy": evaluate_policy_splits(
                tasks,
                evaluate_transport_policy,
                agent.policy,
                selected_config.seed,
            ),
            "transport_teacher": evaluate_policy_splits(
                tasks,
                evaluate_transport_policy,
                transport_teacher_policy,
                selected_config.seed,
            ),
            "rule_policy": evaluate_policy_splits(
                tasks,
                evaluate_transport_policy,
                recovery_baseline_policy,
                selected_config.seed,
            ),
            "random_valid_policy": evaluate_policy_splits(
                tasks,
                evaluate_transport_policy,
                RandomValidPolicy(selected_config.seed),
                selected_config.seed,
            ),
        },
    }
    return RecoveryPolicyTrainingResult(agent=agent, report=report)


def _dataset_summary(tasks: list[MCPTransportRecoveryTask]) -> dict[str, Any]:
    """汇总真实 MCP transport 轨迹覆盖和延迟。"""

    return {
        "task_count": len(tasks),
        "templates": sorted({item.template_id for item in tasks}),
        "scenarios": sorted({item.scenario for item in tasks}),
        "split_counts": {
            split: sum(item.split == split for item in tasks)
            for split in ("train", "validation", "test")
        },
        "transport_trace_count": len({item.transport_trace.trace_id for item in tasks}),
        "average_transport_elapsed_ms": round(mean(item.transport_trace.elapsed_ms for item in tasks), 6),
        "error_types": sorted(
            {
                item.transport_trace.error_type
                for item in tasks
                if item.transport_trace.error_type is not None
            }
        ),
    }
