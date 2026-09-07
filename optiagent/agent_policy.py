from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field


PolicyAction = Literal["accept_solution", "retry_solver", "rebuild_model", "terminate"]


class CandidateAction(BaseModel):
    """策略在当前状态下可以考虑的单个高层动作。"""

    action_id: PolicyAction
    enabled: bool
    reason: str


class PolicyDecision(BaseModel):
    """可直接用于行为克隆或离线 RL 的结构化策略决策。"""

    schema_version: str = "1.0"
    type: Literal["route"] = "route"
    policy_name: str = "deterministic_recovery_baseline"
    policy_version: str = "1.0"
    selected_action: PolicyAction
    candidates: list[CandidateAction]
    candidate_ids: list[PolicyAction]
    action_mask: list[bool]
    reason: str
    metadata: dict[str, Any] = Field(default_factory=dict)


def decide_after_verification(
    verification: dict[str, Any],
    *,
    model_attempt: int,
    solver_attempt: int,
    max_model_attempts: int = 2,
    max_solver_attempts: int = 2,
) -> PolicyDecision:
    """
    根据 Verifier 反馈选择接受、重建模、重试求解或终止。

    这是第一个确定性 baseline：后续学习策略可在不改变轨迹合同的
    前提下替换此函数。
    """

    passed = bool(verification.get("passed"))
    mathematical = verification.get("mathematical") or {}
    mathematical_failed = bool(mathematical.get("verifiable")) and not bool(mathematical.get("passed"))
    # 重建模后仍需再求解，因此必须同时保留建模和求解预算。
    can_rebuild = (
        not passed
        and mathematical_failed
        and model_attempt < max_model_attempts
        and solver_attempt < max_solver_attempts
    )
    can_retry = not passed and solver_attempt < max_solver_attempts
    can_terminate = not passed

    candidates = [
        CandidateAction(
            action_id="accept_solution",
            enabled=passed,
            reason="Verifier 已通过。" if passed else "Verifier 尚未通过。",
        ),
        CandidateAction(
            action_id="retry_solver",
            enabled=can_retry,
            reason=(
                f"求解尝试 {solver_attempt}/{max_solver_attempts}，仍可重试。"
                if can_retry
                else "求解已成功或重试预算已用完。"
            ),
        ),
        CandidateAction(
            action_id="rebuild_model",
            enabled=can_rebuild,
            reason=(
                f"数学验证失败，建模尝试 {model_attempt}/{max_model_attempts}。"
                if can_rebuild
                else "未发现可修复的数学异常，或建模/求解预算已用完。"
            ),
        ),
        CandidateAction(
            action_id="terminate",
            enabled=can_terminate,
            reason="保留当前结果并终止。" if can_terminate else "结果已通过，无需异常终止。",
        ),
    ]

    if passed:
        selected: PolicyAction = "accept_solution"
        reason = "Verifier 通过，接受当前解。"
    elif can_rebuild:
        selected = "rebuild_model"
        reason = "数学验证发现可行性或目标值异常，优先返回 Modeler。"
    elif can_retry:
        selected = "retry_solver"
        reason = "未得到可接受的解，且仍有求解预算。"
    else:
        selected = "terminate"
        reason = "恢复预算已用完，终止循环并保留失败样本。"

    return PolicyDecision(
        selected_action=selected,
        candidates=candidates,
        candidate_ids=[item.action_id for item in candidates],
        action_mask=[item.enabled for item in candidates],
        reason=reason,
        metadata={
            "model_attempt": model_attempt,
            "solver_attempt": solver_attempt,
            "max_model_attempts": max_model_attempts,
            "max_solver_attempts": max_solver_attempts,
        },
    )
