from __future__ import annotations

from dataclasses import asdict, dataclass
from statistics import mean
from collections.abc import Callable
from copy import deepcopy
from pathlib import Path
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
    bc_agent: MaskedDoubleDQNAgent
    final_agent: MaskedDoubleDQNAgent

    def save_comparisons(self, directory: Path, *, run_id: str) -> dict[str, Path]:
        """保留 BC 与末轮 DQN 权重，使验证选模结果可独立复核。"""

        return {
            "bc_checkpoint": self.bc_agent.save(directory / "bc_policy.pt", metadata={"run_id": run_id, "stage": "bc"}),
            "dqn_final_checkpoint": self.final_agent.save(directory / "dqn_final_policy.pt", metadata={"run_id": run_id, "stage": "dqn"}),
        }


@dataclass
class TrainingCoreResult:
    """三类恢复环境共用的训练内核输出。"""

    agent: MaskedDoubleDQNAgent
    metrics: dict[str, Any]
    bc_agent: MaskedDoubleDQNAgent
    final_agent: MaskedDoubleDQNAgent


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
            "bc_policy": evaluate_policy_splits(tasks, evaluate_policy, core.bc_agent.policy, selected_config.seed),
            "dqn_final_policy": evaluate_policy_splits(tasks, evaluate_policy, core.final_agent.policy, selected_config.seed),
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
    return RecoveryPolicyTrainingResult(agent=agent, report=report, bc_agent=core.bc_agent, final_agent=core.final_agent)


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
    validation_tasks = [item for item in tasks if item.split == "validation"]
    if config.train_episodes < 0 or config.bc_epochs < 0 or config.validation_interval <= 0:
        raise ValueError("训练轮数不能为负，验证间隔必须为正。")
    task_ids = [item.task_id for item in tasks]
    if len(task_ids) != len(set(task_ids)):
        raise ValueError("任务标识必须唯一，不能跨训练、验证和测试集重复。")
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
    # 保留独立 BC 快照；验证集选模从预热结束开始，测试集不参与选择。
    bc_agent = deepcopy(agent)
    best_agent = deepcopy(agent)
    best_return = _validation_return(validation_tasks, environment_factory, agent)
    selected_episode = 0
    validation_history = [{"episode": 0, "average_return": best_return}]
    environment = environment_factory(train_tasks, seed=config.seed)
    replay_buffer = ReplayBuffer(config.replay_capacity, config.seed)
    dqn_losses: list[float] = []
    environment_steps = 0
    for episode in range(1, config.train_episodes + 1):
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
        if episode % config.validation_interval == 0 or episode == config.train_episodes:
            score = _validation_return(validation_tasks, environment_factory, agent)
            validation_history.append({"episode": episode, "average_return": score})
            # 同分保留更早的 checkpoint，防止无证据地声称 RL 优于 BC。
            if score > best_return + 1e-9:
                best_return = score
                best_agent = deepcopy(agent)
                selected_episode = episode
    metrics = {
        "demonstration_count": int(len(demonstration_states)),
        "environment_steps": environment_steps,
        "optimization_steps": agent.optimization_steps,
        "replay_size": len(replay_buffer),
        "bc_initial_loss": bc_losses[0] if bc_losses else None,
        "bc_final_loss": bc_losses[-1] if bc_losses else None,
        "dqn_average_loss": round(mean(dqn_losses), 8) if dqn_losses else None,
        "dqn_final_loss": dqn_losses[-1] if dqn_losses else None,
        "selection": {
            "split": "validation", "metric": "average_return",
            "selected_episode": selected_episode,
            "selected_stage": "bc" if selected_episode == 0 else "dqn",
            "selected_optimization_steps": best_agent.optimization_steps,
            "history": validation_history,
        },
    }
    return TrainingCoreResult(agent=best_agent, metrics=metrics, bc_agent=bc_agent, final_agent=agent)


def _validation_return(tasks: list[Any], environment_factory: Callable[..., Any], agent: MaskedDoubleDQNAgent) -> float:
    """独立环境做贪心验证，不向训练回放池添加任何验证样本。"""

    environment = environment_factory(tasks, seed=agent.config.seed)
    returns = []
    for task in tasks:
        observation, _ = environment.reset(task_id=task.task_id)
        total = 0.0
        terminated = truncated = False
        while not (terminated or truncated):
            observation, reward, terminated, truncated, _ = environment.step(agent.policy(observation))
            total += reward
        returns.append(total)
    return float(mean(returns))


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
