from __future__ import annotations

from typing import Any

import numpy as np

from optiagent.rl.benchmark import TEMPLATE_IDS
from optiagent.rl.environment import ACTION_IDS


STATE_ENCODER_VERSION = "1.0"
COST_AWARE_STATE_ENCODER_VERSION = "2.0"
TRANSPORT_AWARE_STATE_ENCODER_VERSION = "3.0"


class RecoveryStateEncoder:
    """将结构化 Policy observation 转换为稳定、可版本化的数值向量。"""

    def __init__(self) -> None:
        self.version = STATE_ENCODER_VERSION
        self.feature_names = [
            *[f"template:{name}" for name in TEMPLATE_IDS],
            "verification:passed",
            "verification:verifiable",
            "feasible:unknown",
            "feasible:false",
            "feasible:true",
            "objective_consistent:unknown",
            "objective_consistent:false",
            "objective_consistent:true",
            "error_count",
            "violation_count",
            "instance_size",
            "data_density",
            "difficulty",
            "model_attempt",
            "solver_attempt",
            "remaining_model_attempts",
            "remaining_solver_attempts",
            "step_count",
            "last_action:none",
            *[f"last_action:{name}" for name in ACTION_IDS],
        ]

    @property
    def dimension(self) -> int:
        return len(self.feature_names)

    def encode(self, observation: dict[str, Any]) -> np.ndarray:
        """编码单个状态，不使用 scenario 或标准答案等泄露字段。"""

        template_id = str(observation.get("template_id") or "")
        verification = observation.get("verification") or {}
        mathematical = verification.get("mathematical") or {}
        features = observation.get("features") or {}
        last_action = observation.get("last_action")
        values: list[float] = []
        values.extend(float(template_id == name) for name in TEMPLATE_IDS)
        values.extend(
            [
                float(bool(verification.get("passed"))),
                float(bool(mathematical.get("verifiable"))),
            ]
        )
        values.extend(_tri_state(mathematical.get("feasible")))
        values.extend(_tri_state(mathematical.get("objective_consistent")))
        values.extend(
            [
                min(len(verification.get("errors") or []), 5) / 5.0,
                min(len(mathematical.get("violations") or []), 5) / 5.0,
                min(np.log1p(max(float(features.get("instance_size", 0.0)), 0.0)) / 6.0, 1.0),
                _clip_unit(features.get("data_density")),
                _clip_unit(features.get("difficulty")),
                min(max(float(observation.get("model_attempt", 0)), 0.0) / 4.0, 1.0),
                min(max(float(observation.get("solver_attempt", 0)), 0.0) / 4.0, 1.0),
                min(max(float(observation.get("remaining_model_attempts", 0)), 0.0) / 4.0, 1.0),
                min(max(float(observation.get("remaining_solver_attempts", 0)), 0.0) / 4.0, 1.0),
                min(max(float(observation.get("step_count", 0)), 0.0) / 4.0, 1.0),
                float(last_action is None),
            ]
        )
        values.extend(float(last_action == name) for name in ACTION_IDS)
        encoded = np.asarray(values, dtype=np.float32)
        if self.version == STATE_ENCODER_VERSION and encoded.shape != (self.dimension,):
            raise ValueError(f"状态维度异常：期望 {self.dimension}，实际 {encoded.shape}")
        return encoded

    def encode_mask(self, observation: dict[str, Any]) -> np.ndarray:
        """按固定 ACTION_IDS 顺序返回合法动作 mask。"""

        candidate_ids = list(observation.get("candidate_actions") or [])
        candidate_mask = list(observation.get("action_mask") or [])
        enabled = dict(zip(candidate_ids, candidate_mask, strict=False))
        mask = np.asarray([bool(enabled.get(action_id, False)) for action_id in ACTION_IDS], dtype=np.bool_)
        if not mask.any():
            raise ValueError("当前状态没有任何合法动作。")
        return mask


class CostAwareRecoveryStateEncoder(RecoveryStateEncoder):
    """在基础验证状态上加入真实求解耗时、动作成本与剩余预算。"""

    def __init__(self) -> None:
        super().__init__()
        self.version = COST_AWARE_STATE_ENCODER_VERSION
        self.feature_names.extend(
            [
                "solver_latency_ms",
                "retry_cost_estimate",
                "rebuild_cost_estimate",
                "cumulative_cost_ratio",
                "remaining_cost_ratio",
                "last_action_cost",
            ]
        )

    def encode(self, observation: dict[str, Any]) -> np.ndarray:
        """编码成本感知状态，同时保持 v1 特征顺序完全不变。"""

        base = super().encode(observation)
        features = observation.get("features") or {}
        cost_budget = max(float(observation.get("cost_budget", 1.0) or 1.0), 1e-6)
        cumulative_cost = max(float(observation.get("cumulative_cost", 0.0) or 0.0), 0.0)
        remaining_cost = max(float(observation.get("remaining_cost_budget", cost_budget) or 0.0), 0.0)
        additions = np.asarray(
            [
                min(np.log1p(max(float(features.get("solver_latency_ms", 0.0)), 0.0)) / 8.0, 1.0),
                min(max(float(features.get("retry_cost_estimate", 0.0)), 0.0) / 0.5, 1.0),
                min(max(float(features.get("rebuild_cost_estimate", 0.0)), 0.0) / 0.5, 1.0),
                min(cumulative_cost / cost_budget, 1.0),
                min(remaining_cost / cost_budget, 1.0),
                min(max(float(observation.get("last_action_cost", 0.0) or 0.0), 0.0) / 0.5, 1.0),
            ],
            dtype=np.float32,
        )
        encoded = np.concatenate([base, additions]).astype(np.float32)
        if self.version == COST_AWARE_STATE_ENCODER_VERSION and encoded.shape != (self.dimension,):
            raise ValueError(f"成本状态维度异常：期望 {self.dimension}，实际 {encoded.shape}")
        return encoded


class TransportAwareRecoveryStateEncoder(CostAwareRecoveryStateEncoder):
    """加入 MCP transport 状态，使策略能区分超时、断连和合同错误。"""

    def __init__(self) -> None:
        super().__init__()
        self.version = TRANSPORT_AWARE_STATE_ENCODER_VERSION
        self.feature_names.extend(
            [
                "transport:timeout",
                "transport:disconnected",
                "transport:response_schema_valid",
                "transport:elapsed_ms",
                "transport:failure_count",
            ]
        )

    def encode(self, observation: dict[str, Any]) -> np.ndarray:
        """编码真实 MCP 连接观测，不读取任务的故障答案标签。"""

        base = super().encode(observation)
        features = observation.get("features") or {}
        additions = np.asarray(
            [
                _clip_unit(features.get("transport_timeout")),
                _clip_unit(features.get("transport_disconnected")),
                _clip_unit(features.get("transport_response_schema_valid")),
                min(np.log1p(max(float(features.get("transport_elapsed_ms", 0.0)), 0.0)) / 8.0, 1.0),
                min(max(float(observation.get("transport_failure_count", 0.0)), 0.0) / 3.0, 1.0),
            ],
            dtype=np.float32,
        )
        encoded = np.concatenate([base, additions]).astype(np.float32)
        if encoded.shape != (self.dimension,):
            raise ValueError(f"MCP transport 状态维度异常：期望 {self.dimension}，实际 {encoded.shape}")
        return encoded


def _tri_state(value: Any) -> list[float]:
    """使用 unknown/false/true 三分类，避免把未知错当成 False。"""

    if value is None:
        return [1.0, 0.0, 0.0]
    if bool(value):
        return [0.0, 0.0, 1.0]
    return [0.0, 1.0, 0.0]


def _clip_unit(value: Any) -> float:
    try:
        return min(max(float(value), 0.0), 1.0)
    except (TypeError, ValueError):
        return 0.0
