from __future__ import annotations

from collections import Counter
from copy import deepcopy
from dataclasses import asdict, dataclass
import math
import random
from statistics import mean

import numpy as np
import torch
from torch.nn import functional as F

from optiagent.rl.dqn import DQNConfig, MaskedDoubleDQNAgent
from optiagent.rl.environment import ACTION_IDS
from optiagent.rl.offline_dataset import OfflineDataset, transition_reward
from optiagent.rl.state_encoder import CostAwareRecoveryStateEncoder
from optiagent.rl.training import RecoveryPolicyTrainingResult


@dataclass(frozen=True)
class OfflineConfig:
    """固定数据上的 BC 与离散 CQL 超参数；不启动环境交互。"""

    seed: int = 57
    bc_epochs: int = 160
    gradient_steps: int = 1500
    batch_size: int = 32
    learning_rate: float = 0.001
    gamma: float = 0.95
    cql_alpha: float = 0.1
    target_sync_interval: int = 100
    validation_interval: int = 100
    reward_mode: str = "verified_cost"


def encode_episodes(episodes: list[dict], encoder: CostAwareRecoveryStateEncoder, config: OfflineConfig) -> dict:
    """编码固定轨迹并复算记录动作的折扣回报；终止状态不填造合法下一动作。"""

    rows, returns = [], []
    for episode in episodes:
        discounted = 0.0
        episode_returns = []
        for row in reversed(episode["transitions"]):
            discounted = transition_reward(episode, row, config.reward_mode) + config.gamma * discounted
            episode_returns.append(discounted)
        returns.extend(reversed(episode_returns))
        rows.extend((episode, row) for row in episode["transitions"])
    states = np.stack([encoder.encode(row["state"]) for _, row in rows])
    masks = np.stack([encoder.encode_mask(row["state"]) for _, row in rows])
    next_states = np.stack([encoder.encode(row["next_state"]) if not row["done"] else np.zeros(encoder.dimension, dtype=np.float32) for _, row in rows])
    next_masks = np.stack([encoder.encode_mask(row["next_state"]) if not row["done"] else np.zeros(len(ACTION_IDS), dtype=np.bool_) for _, row in rows])
    return {"states": torch.from_numpy(states), "masks": torch.from_numpy(masks),
            "actions": torch.tensor([ACTION_IDS.index(row["action"]) for _, row in rows]),
            "rewards": torch.tensor([transition_reward(episode, row, config.reward_mode) for episode, row in rows], dtype=torch.float32),
            "next_states": torch.from_numpy(next_states), "next_masks": torch.from_numpy(next_masks),
            "dones": torch.tensor([row["done"] for _, row in rows], dtype=torch.bool),
            "returns": torch.tensor(returns, dtype=torch.float32)}


def conservative_penalty(q_values: torch.Tensor, actions: torch.Tensor, masks: torch.Tensor) -> torch.Tensor:
    """CQL 离散惩罚只覆盖合法动作，避免非法动作的大 Q 值污染训练。"""

    legal_q = q_values.masked_fill(~masks, -torch.inf)
    return (torch.logsumexp(legal_q, dim=1) - q_values.gather(1, actions[:, None]).squeeze(1)).mean()


def double_dqn_targets(agent: MaskedDoubleDQNAgent, batch: dict, gamma: float) -> torch.Tensor:
    """只对非终止样本计算自举，终局 next_state=None 不参与 argmax。"""

    with torch.no_grad():
        targets = batch["rewards"].clone()
        active = ~batch["dones"]
        if active.any():
            next_states = batch["next_states"][active]
            online = agent.policy_network(next_states).masked_fill(~batch["next_masks"][active], -torch.inf)
            actions = online.argmax(1)
            target_q = agent.target_network(next_states).gather(1, actions[:, None]).squeeze(1)
            targets[active] += gamma * target_q
        return targets


def logged_diagnostics(agent: MaskedDoubleDQNAgent, data: dict) -> dict:
    """日志诊断不是策略价值估计；分别报告强制动作与有选择状态的动作一致性。"""

    with torch.no_grad():
        q = agent.policy_network(data["states"])
        actions = q.masked_fill(~data["masks"], -torch.inf).argmax(1)
        agreement = actions == data["actions"]
        choice = data["masks"].sum(1) > 1
        predicted = q.gather(1, data["actions"][:, None]).squeeze(1)
        mse = F.mse_loss(predicted, data["returns"])
    if not torch.isfinite(mse):
        raise ValueError("离线评估出现非有限 Q 值。")
    return {"transition_count": len(actions), "choice_state_count": int(choice.sum()),
            "logged_action_agreement": float(agreement.float().mean()),
            "choice_action_agreement": float(agreement[choice].float().mean()) if choice.any() else None,
            "logged_return_mse": float(mse),
            "invalid_action_rate": float((~data["masks"].gather(1, actions[:, None]).squeeze(1)).float().mean())}


def train_offline_policy(dataset: OfflineDataset, config: OfflineConfig) -> RecoveryPolicyTrainingResult:
    """从训练集合做 BC 预热及保守 Q 更新；验证只用于诊断选模，测试只在最终报告使用。"""

    if (config.gradient_steps < 0 or config.bc_epochs < 0 or config.batch_size < 1
            or config.validation_interval < 1 or config.target_sync_interval < 1
            or not 0 <= config.gamma <= 1 or not math.isfinite(config.cql_alpha) or config.cql_alpha < 0
            or not math.isfinite(config.learning_rate) or config.learning_rate <= 0
            or config.reward_mode not in {"sparse", "verified_cost"}):
        raise ValueError("离线训练超参数不合法。")
    encoder = CostAwareRecoveryStateEncoder()
    train = encode_episodes(dataset.splits["train"], encoder, config)
    validation = encode_episodes(dataset.splits["validation"], encoder, config)
    agent = MaskedDoubleDQNAgent(encoder, DQNConfig(seed=config.seed, learning_rate=config.learning_rate,
                                                  bc_epochs=config.bc_epochs, gamma=config.gamma, train_episodes=0))
    bc_losses = agent.behavior_clone(train["states"].numpy(), train["actions"].numpy(), train["masks"].numpy())
    bc_agent = deepcopy(agent)
    best_agent = deepcopy(agent)
    best_score = logged_diagnostics(agent, validation)["logged_return_mse"]
    selected_step = 0
    history = [{"step": 0, "logged_return_mse": best_score}]
    randomizer = random.Random(config.seed)
    losses = []
    for step in range(1, config.gradient_steps + 1):
        indices = randomizer.sample(range(len(train["actions"])), min(config.batch_size, len(train["actions"])))
        batch = {key: value[indices] for key, value in train.items()}
        q = agent.policy_network(batch["states"])
        predicted = q.gather(1, batch["actions"][:, None]).squeeze(1)
        targets = double_dqn_targets(agent, batch, config.gamma)
        td_loss = F.smooth_l1_loss(predicted, targets)
        conservative = conservative_penalty(q, batch["actions"], batch["masks"])
        loss = td_loss + config.cql_alpha * conservative
        if not torch.isfinite(loss):
            raise ValueError("离线训练出现非有限损失。")
        agent.optimizer.zero_grad()
        loss.backward()
        torch.nn.utils.clip_grad_norm_(agent.policy_network.parameters(), 5.0)
        agent.optimizer.step()
        agent.optimization_steps += 1
        if step % config.target_sync_interval == 0:
            agent.target_network.load_state_dict(agent.policy_network.state_dict())
        losses.append({"total": float(loss.detach()), "td": float(td_loss.detach()), "cql": float(conservative.detach())})
        if step % config.validation_interval == 0 or step == config.gradient_steps:
            score = logged_diagnostics(agent, validation)["logged_return_mse"]
            history.append({"step": step, "logged_return_mse": score})
            if score < best_score - 1e-9:
                best_score, selected_step, best_agent = score, step, deepcopy(agent)
    # 参数选择完成后才创建测试张量，测试统计不反馈到梯度或选模。
    test = encode_episodes(dataset.splits["test"], encoder, config)
    evaluation = {name: {split: logged_diagnostics(policy, data) for split, data in (("train", train), ("validation", validation), ("test", test))}
                  for name, policy in (("bc_policy", bc_agent), ("cql_final_policy", agent), ("learned_policy", best_agent))}
    report = {
        "schema_version": "1.0", "algorithm": "behavior_cloning+masked_discrete_cql_double_dqn",
        "config": asdict(config), "state_encoder_version": encoder.version, "action_ids": list(ACTION_IDS),
        "reward": {"source_version": dataset.manifest["reward_version"],
                   "training_version": "workflow-verified-cost-v1" if config.reward_mode == "verified_cost" else dataset.manifest["reward_version"]},
        "dataset": {"manifest_sha256": dataset.manifest_sha256, "trajectory_source": dataset.manifest["trajectory_source"],
                    "unique_instances_by_split": dataset.manifest["unique_instances_by_split"],
                    "split_transition_counts": {key: len(data["actions"]) for key, data in (("train", train), ("validation", validation), ("test", test))},
                    "train_action_counts": dict(Counter(ACTION_IDS[index] for index in train["actions"].tolist()))},
        "training": {"demonstration_count": len(train["actions"]), "environment_steps": 0, "optimization_steps": agent.optimization_steps,
                     "bc_initial_loss": bc_losses[0] if bc_losses else None, "bc_final_loss": bc_losses[-1] if bc_losses else None,
                     "average_loss": mean(item["total"] for item in losses) if losses else None,
                     "final_losses": losses[-1] if losses else None,
                     "selection": {"split": "validation", "metric": "logged_return_mse", "direction": "minimize",
                                   "selected_step": selected_step, "selected_stage": "bc" if selected_step == 0 else "offline_cql", "history": history}},
        "evaluation": evaluation,
        "limitations": ["日志回报误差及动作一致性是诊断指标，不估计新策略成功率或期望回报。",
                        "BC 输出作为动作分类分数，预热 Q 值尚未校准；误差下降本身不能证明策略改进。",
                        "缺少行为概率，未进行重要性采样；部署前需要独立主流程评测。"],
    }
    return RecoveryPolicyTrainingResult(best_agent, report, bc_agent, agent)
