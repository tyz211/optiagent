from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from collections.abc import Callable
from typing import Any

import numpy as np

from optiagent.rl.benchmark import BenchmarkTask
from optiagent.rl.dqn import DQNConfig, MaskedDoubleDQNAgent, ReplayBuffer, ReplayTransition, epsilon_at_step
from optiagent.rl.environment import ACTION_IDS, ACTION_TO_INDEX, OptimizationAgentEnv
from optiagent.rl.rollout import RandomValidPolicy, evaluate_policy, recovery_baseline_policy
from optiagent.rl.state_encoder import RecoveryStateEncoder


@dataclass
class RecoveryPolicyTrainingResult:
    """训练后的可调用 Agent 与可 JSON 序列化实验报告。"""

    agent: MaskedDoubleDQNAgent
    report: dict[str, Any]


@dataclass
class TrainingCoreResult:
    """三类恢复环境共用的训练内核输出。"""

    agent: MaskedDoubleDQNAgent
    metrics: dict[str, Any]


def train_recovery_policy(
    tasks: list[BenchmarkTask],
    config: DQNConfig | None = None,
) -> RecoveryPolicyTrainingResult:
    """使用 BC 预热和 Masked Double DQN 训练高层恢复策略。"""

    selected_config = config or DQNConfig()
    encoder = RecoveryStateEncoder()
    core = train_masked_dqn_core(
        tasks,
        encoder=encoder,
        environment_factory=OptimizationAgentEnv,
        teacher_policy=recovery_baseline_policy,
        config=selected_config,
        split_error="RL 训练需要非空的 train/validation/test 划分。",
    )
    agent = core.agent
    report = {
        "schema_version": "1.0",
        "algorithm": "behavior_cloning+action_masked_double_dqn",
        "state_encoder_version": encoder.version,
        "action_ids": list(ACTION_IDS),
        "config": asdict(selected_config),
        "training": core.metrics,
        "evaluation": {
            "learned_policy": evaluate_policy_splits(tasks, evaluate_policy, agent.policy, selected_config.seed),
            "rule_policy": evaluate_policy_splits(tasks, evaluate_policy, recovery_baseline_policy, selected_config.seed),
            "random_valid_policy": evaluate_policy_splits(
                tasks,
                evaluate_policy,
                RandomValidPolicy(selected_config.seed),
                selected_config.seed,
            ),
        },
    }
    return RecoveryPolicyTrainingResult(agent=agent, report=report)


def train_masked_dqn_core(
    tasks: list[Any],
    *,
    encoder: RecoveryStateEncoder,
    environment_factory: Callable[..., Any],
    teacher_policy: Callable[[dict[str, Any]], str],
    config: DQNConfig,
    split_error: str,
) -> TrainingCoreResult:
    """统一执行 BC 预热、环境交互与 Masked Double DQN 更新。"""

    train_tasks = [item for item in tasks if item.split == "train"]
    if not train_tasks or not any(item.split == "validation" for item in tasks) or not any(
        item.split == "test" for item in tasks
    ):
        raise ValueError(split_error)
    demonstration_states, demonstration_actions, demonstration_masks = _collect_teacher_demonstrations(
        train_tasks,
        encoder,
        environment_factory,
        teacher_policy,
    )
    agent = MaskedDoubleDQNAgent(encoder, config)
    bc_losses = agent.behavior_clone(
        demonstration_states,
        demonstration_actions,
        demonstration_masks,
    )
    environment = environment_factory(train_tasks, seed=config.seed)
    replay_buffer = ReplayBuffer(config.replay_capacity, config.seed)
    dqn_losses: list[float] = []
    environment_steps = 0
    for _ in range(config.train_episodes):
        observation, _ = environment.reset()
        terminated = False
        truncated = False
        while not (terminated or truncated):
            epsilon = epsilon_at_step(config, environment_steps)
            action_index = agent.select_action_index(observation, epsilon=epsilon)
            next_observation, reward, terminated, truncated, _ = environment.step(ACTION_IDS[action_index])
            replay_buffer.add(
                ReplayTransition(
                    state=encoder.encode(observation),
                    action=action_index,
                    reward=reward,
                    next_state=encoder.encode(next_observation),
                    done=terminated or truncated,
                    next_action_mask=encoder.encode_mask(next_observation),
                )
            )
            loss = agent.optimize(replay_buffer)
            if loss is not None:
                dqn_losses.append(loss)
            observation = next_observation
            environment_steps += 1
    metrics = {
        "demonstration_count": int(len(demonstration_states)),
        "environment_steps": environment_steps,
        "optimization_steps": agent.optimization_steps,
        "replay_size": len(replay_buffer),
        "bc_initial_loss": bc_losses[0] if bc_losses else None,
        "bc_final_loss": bc_losses[-1] if bc_losses else None,
        "dqn_average_loss": round(mean(dqn_losses), 8) if dqn_losses else None,
        "dqn_final_loss": dqn_losses[-1] if dqn_losses else None,
    }
    return TrainingCoreResult(agent=agent, metrics=metrics)


def _collect_teacher_demonstrations(
    tasks: list[Any],
    encoder: RecoveryStateEncoder,
    environment_factory: Callable[..., Any],
    teacher_policy: Callable[[dict[str, Any]], str],
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """从指定教师策略收集仅属于训练集的 BC 样本。"""

    environment = environment_factory(tasks, seed=0)
    states: list[np.ndarray] = []
    actions: list[int] = []
    masks: list[np.ndarray] = []
    for task in tasks:
        observation, _ = environment.reset(task_id=task.task_id)
        terminated = False
        truncated = False
        while not (terminated or truncated):
            action = teacher_policy(observation)
            states.append(encoder.encode(observation))
            actions.append(ACTION_TO_INDEX[action])
            masks.append(encoder.encode_mask(observation))
            observation, _, terminated, truncated, _ = environment.step(action)
    return (
        np.stack(states).astype(np.float32),
        np.asarray(actions, dtype=np.int64),
        np.stack(masks).astype(np.bool_),
    )


def evaluate_policy_splits(
    tasks: list[Any],
    evaluator: Callable[..., dict[str, Any]],
    policy: Any,
    seed: int,
) -> dict[str, Any]:
    """在三个隔离 split 上调用同一环境评估函数。"""

    metrics = {}
    for split in ("train", "validation", "test"):
        split_tasks = [item for item in tasks if item.split == split]
        metrics[split] = evaluator(split_tasks, policy, seed=seed)["metrics"]
    return metrics
