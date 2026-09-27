from __future__ import annotations

from dataclasses import asdict
from statistics import mean
from typing import Any

from optiagent.rl.dqn import DQNConfig
from optiagent.instance_identity import audit_instance_splits
from optiagent.rl.environment import ACTION_IDS
from optiagent.rl.real_environment import (
    REAL_ENVIRONMENT_VERSION,
    RealGatewayRecoveryEnv,
    RealRecoveryTask,
    cost_aware_teacher_policy,
    evaluate_real_policy,
)
from optiagent.rl.rollout import RandomValidPolicy, recovery_baseline_policy
from optiagent.rl.state_encoder import CostAwareRecoveryStateEncoder
from optiagent.rl.training import (
    RecoveryPolicyTrainingResult,
    evaluate_policy_splits,
    train_masked_dqn_core,
)


def train_real_recovery_policy(
    tasks: list[RealRecoveryTask],
    config: DQNConfig | None = None,
) -> RecoveryPolicyTrainingResult:
    """在真实 Gateway 求解轨迹上训练成本感知恢复策略。"""

    selected_config = config or DQNConfig()
    audit_instance_splits(tasks)
    encoder = CostAwareRecoveryStateEncoder()
    core = train_masked_dqn_core(
        tasks,
        encoder=encoder,
        environment_factory=RealGatewayRecoveryEnv,
        teacher_policy=cost_aware_teacher_policy,
        config=selected_config,
        split_error="真实恢复训练需要非空的 train/validation/test 划分。",
    )
    agent = core.agent

    report = {
        "schema_version": "1.0",
        "algorithm": "behavior_cloning+action_masked_double_dqn+real_gateway_cost",
        "environment_version": REAL_ENVIRONMENT_VERSION,
        "state_encoder_version": encoder.version,
        "state_dimension": encoder.dimension,
        "action_ids": list(ACTION_IDS),
        "config": asdict(selected_config),
        "dataset": _dataset_summary(tasks),
        "training": core.metrics,
        "evaluation": {
            "bc_policy": evaluate_policy_splits(tasks, evaluate_real_policy, core.bc_agent.policy, selected_config.seed),
            "dqn_final_policy": evaluate_policy_splits(tasks, evaluate_real_policy, core.final_agent.policy, selected_config.seed),
            "learned_policy": evaluate_policy_splits(tasks, evaluate_real_policy, agent.policy, selected_config.seed),
            "cost_aware_teacher": evaluate_policy_splits(
                tasks,
                evaluate_real_policy,
                cost_aware_teacher_policy,
                selected_config.seed,
            ),
            "rule_policy": evaluate_policy_splits(
                tasks,
                evaluate_real_policy,
                recovery_baseline_policy,
                selected_config.seed,
            ),
            "random_valid_policy": evaluate_policy_splits(
                tasks,
                evaluate_real_policy,
                RandomValidPolicy(selected_config.seed),
                selected_config.seed,
            ),
        },
    }
    return RecoveryPolicyTrainingResult(agent=agent, report=report, bc_agent=core.bc_agent, final_agent=core.final_agent)


def _dataset_summary(tasks: list[RealRecoveryTask]) -> dict[str, Any]:
    """记录真实轨迹覆盖范围和耗时统计。"""

    return {
        "content_split_audit": audit_instance_splits(tasks),
        "task_count": len(tasks),
        "templates": sorted({item.template_id for item in tasks}),
        "scenarios": sorted({item.scenario for item in tasks}),
        "split_counts": {
            split: sum(item.split == split for item in tasks)
            for split in ("train", "validation", "test")
        },
        "average_solver_latency_ms": round(mean(item.solver_latency_ms for item in tasks), 6),
        "max_solver_latency_ms": round(max(item.solver_latency_ms for item in tasks), 6),
    }
