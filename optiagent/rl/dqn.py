from __future__ import annotations

from dataclasses import asdict, dataclass
from pathlib import Path
import random
from typing import Any

import numpy as np
import torch
from torch import nn
from torch.nn import functional as functional

from optiagent.rl.environment import ACTION_IDS
from optiagent.rl.state_encoder import (
    COST_AWARE_STATE_ENCODER_VERSION,
    TRANSPORT_AWARE_STATE_ENCODER_VERSION,
    CostAwareRecoveryStateEncoder,
    RecoveryStateEncoder,
    STATE_ENCODER_VERSION,
    TransportAwareRecoveryStateEncoder,
)


@dataclass(frozen=True)
class DQNConfig:
    """Masked Double DQN 的可复现训练超参数。"""

    seed: int = 42
    hidden_dim: int = 64
    learning_rate: float = 1e-3
    gamma: float = 0.95
    batch_size: int = 32
    replay_capacity: int = 10_000
    warmup_transitions: int = 64
    target_sync_interval: int = 100
    epsilon_start: float = 0.8
    epsilon_end: float = 0.05
    epsilon_decay_steps: int = 800
    gradient_clip_norm: float = 5.0
    bc_epochs: int = 120
    bc_learning_rate: float = 2e-3
    train_episodes: int = 800
    device: str = "cpu"


@dataclass(frozen=True)
class ReplayTransition:
    """经状态编码后的单步 off-policy transition。"""

    state: np.ndarray
    action: int
    reward: float
    next_state: np.ndarray
    done: bool
    next_action_mask: np.ndarray


class ReplayBuffer:
    """固定容量经验回放池，使用独立随机源保证可复现。"""

    def __init__(self, capacity: int, seed: int) -> None:
        if capacity <= 0:
            raise ValueError("replay capacity 必须为正数。")
        self.capacity = capacity
        self._items: list[ReplayTransition] = []
        self._cursor = 0
        self._randomizer = random.Random(seed)

    def __len__(self) -> int:
        return len(self._items)

    def add(self, transition: ReplayTransition) -> None:
        if len(self._items) < self.capacity:
            self._items.append(transition)
        else:
            self._items[self._cursor] = transition
        self._cursor = (self._cursor + 1) % self.capacity

    def sample(self, batch_size: int) -> list[ReplayTransition]:
        if batch_size > len(self._items):
            raise ValueError("回放池样本不足。")
        return self._randomizer.sample(self._items, batch_size)


class DuelingQNetwork(nn.Module):
    """分离状态价值与动作优势的轻量 Q 网络。"""

    def __init__(self, input_dim: int, hidden_dim: int, action_dim: int) -> None:
        super().__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
        )
        self.value_head = nn.Linear(hidden_dim, 1)
        self.advantage_head = nn.Linear(hidden_dim, action_dim)

    def forward(self, states: torch.Tensor) -> torch.Tensor:
        hidden = self.encoder(states)
        value = self.value_head(hidden)
        advantage = self.advantage_head(hidden)
        return value + advantage - advantage.mean(dim=1, keepdim=True)


class MaskedDoubleDQNAgent:
    """支持非法动作 mask、Double DQN target 和 BC 预热的策略。"""

    def __init__(self, encoder: RecoveryStateEncoder, config: DQNConfig) -> None:
        self.state_encoder = encoder
        self.config = config
        self.device = torch.device(config.device)
        self._randomizer = random.Random(config.seed)
        torch.manual_seed(config.seed)
        np.random.seed(config.seed)
        self.policy_network = DuelingQNetwork(encoder.dimension, config.hidden_dim, len(ACTION_IDS)).to(self.device)
        self.target_network = DuelingQNetwork(encoder.dimension, config.hidden_dim, len(ACTION_IDS)).to(self.device)
        self.target_network.load_state_dict(self.policy_network.state_dict())
        self.target_network.eval()
        self.optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=config.learning_rate)
        self.optimization_steps = 0

    def select_action_index(self, observation: dict[str, Any], *, epsilon: float = 0.0) -> int:
        """仅在 mask 允许的动作中执行 epsilon-greedy 选择。"""

        mask = self.state_encoder.encode_mask(observation)
        valid_indices = np.flatnonzero(mask).tolist()
        if self._randomizer.random() < epsilon:
            return int(self._randomizer.choice(valid_indices))
        state = torch.as_tensor(self.state_encoder.encode(observation), device=self.device).unsqueeze(0)
        with torch.no_grad():
            q_values = self.policy_network(state).squeeze(0)
            mask_tensor = torch.as_tensor(mask, dtype=torch.bool, device=self.device)
            q_values = q_values.masked_fill(~mask_tensor, torch.finfo(q_values.dtype).min)
            return int(q_values.argmax().item())

    def policy(self, observation: dict[str, Any]):
        """以 rollout 接口所需的字符串动作返回贪心策略。"""

        return ACTION_IDS[self.select_action_index(observation, epsilon=0.0)]

    def behavior_clone(
        self,
        states: np.ndarray,
        actions: np.ndarray,
        action_masks: np.ndarray,
    ) -> list[float]:
        """使用规则或教师轨迹对 Q 网络做分类预热。"""

        if len(states) == 0:
            return []
        optimizer = torch.optim.Adam(self.policy_network.parameters(), lr=self.config.bc_learning_rate)
        state_tensor = torch.as_tensor(states, dtype=torch.float32, device=self.device)
        action_tensor = torch.as_tensor(actions, dtype=torch.long, device=self.device)
        mask_tensor = torch.as_tensor(action_masks, dtype=torch.bool, device=self.device)
        losses: list[float] = []
        self.policy_network.train()
        for _ in range(self.config.bc_epochs):
            logits = self.policy_network(state_tensor)
            logits = logits.masked_fill(~mask_tensor, -1e9)
            loss = functional.cross_entropy(logits, action_tensor)
            optimizer.zero_grad()
            loss.backward()
            nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config.gradient_clip_norm)
            optimizer.step()
            losses.append(float(loss.item()))
        self.target_network.load_state_dict(self.policy_network.state_dict())
        return losses

    def optimize(self, replay_buffer: ReplayBuffer) -> float | None:
        """执行一次 action-masked Double DQN 更新。"""

        required = max(self.config.batch_size, self.config.warmup_transitions)
        if len(replay_buffer) < required:
            return None
        batch = replay_buffer.sample(self.config.batch_size)
        states = torch.as_tensor(np.stack([item.state for item in batch]), device=self.device)
        actions = torch.as_tensor([item.action for item in batch], dtype=torch.long, device=self.device)
        rewards = torch.as_tensor([item.reward for item in batch], dtype=torch.float32, device=self.device)
        next_states = torch.as_tensor(np.stack([item.next_state for item in batch]), device=self.device)
        dones = torch.as_tensor([item.done for item in batch], dtype=torch.float32, device=self.device)
        next_masks = torch.as_tensor(
            np.stack([item.next_action_mask for item in batch]),
            dtype=torch.bool,
            device=self.device,
        )

        predicted = self.policy_network(states).gather(1, actions.unsqueeze(1)).squeeze(1)
        with torch.no_grad():
            # Double DQN 用在线网络选动作，用 target 网络估值，降低过估计。
            next_online = self.policy_network(next_states).masked_fill(~next_masks, -1e9)
            next_actions = next_online.argmax(dim=1)
            next_values = self.target_network(next_states).gather(1, next_actions.unsqueeze(1)).squeeze(1)
            targets = rewards + self.config.gamma * (1.0 - dones) * next_values
        loss = functional.smooth_l1_loss(predicted, targets)
        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.policy_network.parameters(), self.config.gradient_clip_norm)
        self.optimizer.step()
        self.optimization_steps += 1
        if self.optimization_steps % self.config.target_sync_interval == 0:
            self.target_network.load_state_dict(self.policy_network.state_dict())
        return float(loss.item())

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> Path:
        """保存带动作与状态版本的 checkpoint。"""

        checkpoint_path = Path(path)
        checkpoint_path.parent.mkdir(parents=True, exist_ok=True)
        torch.save(
            {
                "schema_version": "1.0",
                "state_encoder_version": self.state_encoder.version,
                "state_dim": self.state_encoder.dimension,
                "action_ids": list(ACTION_IDS),
                "config": asdict(self.config),
                "policy_state_dict": self.policy_network.state_dict(),
                "target_state_dict": self.target_network.state_dict(),
                "optimizer_state_dict": self.optimizer.state_dict(),
                "optimization_steps": self.optimization_steps,
                "metadata": metadata or {},
            },
            checkpoint_path,
        )
        return checkpoint_path

    @classmethod
    def load(cls, path: str | Path, *, device: str = "cpu") -> "MaskedDoubleDQNAgent":
        """校验合同版本后恢复 checkpoint。"""

        payload = torch.load(Path(path), map_location=device, weights_only=True)
        encoder_classes = {
            STATE_ENCODER_VERSION: RecoveryStateEncoder,
            COST_AWARE_STATE_ENCODER_VERSION: CostAwareRecoveryStateEncoder,
            TRANSPORT_AWARE_STATE_ENCODER_VERSION: TransportAwareRecoveryStateEncoder,
        }
        encoder_class = encoder_classes.get(payload.get("state_encoder_version"))
        if encoder_class is None:
            raise ValueError("状态编码器版本不兼容。")
        encoder = encoder_class()
        if payload.get("state_dim") != encoder.dimension or payload.get("action_ids") != list(ACTION_IDS):
            raise ValueError("checkpoint 的状态或动作合同不兼容。")
        config = DQNConfig(**{**payload["config"], "device": device})
        agent = cls(encoder, config)
        agent.policy_network.load_state_dict(payload["policy_state_dict"])
        agent.target_network.load_state_dict(payload["target_state_dict"])
        agent.optimizer.load_state_dict(payload["optimizer_state_dict"])
        agent.optimization_steps = int(payload.get("optimization_steps", 0))
        return agent


def epsilon_at_step(config: DQNConfig, step: int) -> float:
    """线性退火 exploration rate，不低于 epsilon_end。"""

    progress = min(max(step, 0) / max(config.epsilon_decay_steps, 1), 1.0)
    return config.epsilon_start + progress * (config.epsilon_end - config.epsilon_start)
