"""Masked tabular Q-learning for the recovery micro-environment, without LLM training."""
from __future__ import annotations

import json
import math
import random
from pathlib import Path
from typing import Any

from optiagent.agent_policy import PolicyAction
from optiagent.rl.environment import ACTION_IDS


ENCODER_VERSION = "recovery-observation-v1"


def state_key(observation: dict[str, Any]) -> str:
    """Use visible feedback and budgets; exclude IDs, split and hidden fault labels.

    Instance features are deliberately omitted: they have no effect on v1 dynamics.
    This abstraction is specific to the synthetic recovery environment.
    """
    verification = observation["verification"]
    mathematical = verification.get("mathematical") or {}
    return json.dumps([
        bool(verification.get("passed")),
        bool(mathematical.get("verifiable")),
        bool(mathematical.get("passed")),
        mathematical.get("feasible"),
        mathematical.get("objective_consistent"),
        observation["remaining_model_attempts"],
        observation["remaining_solver_attempts"],
        observation["step_count"],
        observation["action_mask"],
    ], separators=(",", ":"))


def allowed_actions(observation: dict[str, Any]) -> list[int]:
    if tuple(observation["candidate_actions"]) != ACTION_IDS:
        raise ValueError("Unsupported action order")
    mask = observation["action_mask"]
    if len(mask) != len(ACTION_IDS):
        raise ValueError("Invalid action mask length")
    indices = [i for i, enabled in enumerate(mask) if enabled]
    if not indices:
        raise ValueError("No legal action")
    return indices


class QLearningPolicy:
    def __init__(self, *, alpha: float = 0.15, gamma: float = 1.0) -> None:
        if not 0 < alpha <= 1 or not 0 <= gamma <= 1:
            raise ValueError("Require 0 < alpha <= 1 and 0 <= gamma <= 1")
        self.alpha = alpha
        self.gamma = gamma
        self.q_table: dict[str, list[float]] = {}
        self.updates = 0

    def values(self, observation: dict[str, Any]) -> list[float]:
        # Evaluation must not mutate the checkpoint by inserting unseen states.
        return self.q_table.get(state_key(observation), [0.0] * len(ACTION_IDS))

    def __call__(self, observation: dict[str, Any]) -> PolicyAction:
        values = self.values(observation)
        index = max(allowed_actions(observation), key=lambda i: values[i])
        return ACTION_IDS[index]

    def explore(self, observation: dict[str, Any], epsilon: float, rng: random.Random) -> PolicyAction:
        legal = allowed_actions(observation)
        values = self.values(observation)
        if rng.random() < epsilon:
            return ACTION_IDS[rng.choice(legal)]
        best = max(values[i] for i in legal)
        return ACTION_IDS[rng.choice([i for i in legal if values[i] == best])]

    def update(self, state: dict[str, Any], action: PolicyAction, reward: float,
               next_state: dict[str, Any], done: bool) -> None:
        index = ACTION_IDS.index(action)
        if index not in allowed_actions(state):
            raise ValueError("Cannot train on a masked action")
        target = reward
        if not done:
            next_values = self.values(next_state)
            target += self.gamma * max(next_values[i] for i in allowed_actions(next_state))
        values = self.q_table.setdefault(state_key(state), [0.0] * len(ACTION_IDS))
        values[index] += self.alpha * (target - values[index])
        self.updates += 1

    def save(self, path: str | Path, *, metadata: dict[str, Any] | None = None) -> None:
        destination = Path(path)
        destination.parent.mkdir(parents=True, exist_ok=True)
        destination.write_text(json.dumps({
            "schema_version": "1.0", "algorithm": "tabular_q_learning",
            "encoder_version": ENCODER_VERSION, "action_ids": ACTION_IDS,
            "alpha": self.alpha, "gamma": self.gamma, "updates": self.updates,
            "q_table": self.q_table, "metadata": metadata or {},
        }, ensure_ascii=False, indent=2, allow_nan=False), encoding="utf-8")

    @classmethod
    def load(cls, path: str | Path) -> QLearningPolicy:
        payload = json.loads(Path(path).read_text(encoding="utf-8"))
        if (payload["schema_version"] != "1.0"
                or payload["algorithm"] != "tabular_q_learning"
                or payload["encoder_version"] != ENCODER_VERSION
                or tuple(payload["action_ids"]) != ACTION_IDS):
            raise ValueError("Incompatible checkpoint contract")
        policy = cls(alpha=payload["alpha"], gamma=payload["gamma"])
        for key, values in payload["q_table"].items():
            if len(values) != len(ACTION_IDS) or not all(math.isfinite(v) for v in values):
                raise ValueError("Invalid Q values")
            policy.q_table[key] = list(values)
        policy.updates = payload["updates"]
        return policy
