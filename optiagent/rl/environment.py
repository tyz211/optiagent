from __future__ import annotations

import random
from typing import Any

from optiagent.agent_policy import PolicyAction, decide_after_verification
from optiagent.rl.benchmark import BenchmarkTask


ACTION_IDS: tuple[PolicyAction, ...] = (
    "accept_solution",
    "retry_solver",
    "rebuild_model",
    "terminate",
)
ACTION_TO_INDEX = {action_id: index for index, action_id in enumerate(ACTION_IDS)}


class OptimizationAgentEnv:
    """
    不依赖 Gymnasium 的轻量 RL 环境。

    环境从首次 Verifier 输出开始，专注于训练接受、重试、
    重建模和终止这四类高层决策。
    """

    def __init__(
        self,
        tasks: list[BenchmarkTask],
        *,
        seed: int = 42,
        max_model_attempts: int = 2,
        max_solver_attempts: int = 2,
        max_steps: int = 4,
    ) -> None:
        if not tasks:
            raise ValueError("RL 环境至少需要一个 benchmark 任务。")
        self.tasks = list(tasks)
        self.max_model_attempts = max_model_attempts
        self.max_solver_attempts = max_solver_attempts
        self.max_steps = max_steps
        self._randomizer = random.Random(seed)
        self._task: BenchmarkTask | None = None
        self._verification: dict[str, Any] = {}
        self._model_attempt = 1
        self._solver_attempt = 1
        self._step_count = 0
        self._last_action: PolicyAction | None = None
        self._finished = False

    def reset(
        self,
        *,
        seed: int | None = None,
        task_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """选择任务并返回首个 Policy 决策状态。"""

        if seed is not None:
            self._randomizer.seed(seed)
        if task_id is None:
            self._task = self._randomizer.choice(self.tasks)
        else:
            self._task = next((item for item in self.tasks if item.task_id == task_id), None)
            if self._task is None:
                raise KeyError(f"未找到 benchmark 任务：{task_id}")
        self._verification = _scenario_verification(self._task.scenario, stage="initial")
        self._model_attempt = 1
        self._solver_attempt = 1
        self._step_count = 0
        self._last_action = None
        self._finished = False
        observation = self._observation()
        return observation, self._info(observation)

    def step(
        self,
        action: int | PolicyAction,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """执行一个受 action mask 约束的高层动作。"""

        if self._task is None:
            raise RuntimeError("请先调用 reset()。")
        if self._finished:
            raise RuntimeError("当前 episode 已终止，请重新调用 reset()。")
        action_id = self._normalize_action(action)
        decision = self._decision()
        enabled = dict(zip(decision.candidate_ids, decision.action_mask, strict=True))
        self._step_count += 1
        self._last_action = action_id

        if not enabled[action_id]:
            self._finished = True
            observation = self._observation()
            return observation, -1.0, True, False, self._info(observation, "invalid_action")

        if action_id == "accept_solution":
            self._finished = True
            observation = self._observation()
            return observation, 1.0, True, False, self._info(observation, "verified_solution_accepted")

        if action_id == "terminate":
            self._finished = True
            has_recovery = any(
                enabled.get(candidate, False) for candidate in ("retry_solver", "rebuild_model")
            )
            reward = -0.6 if has_recovery else -0.2
            observation = self._observation()
            reason = "premature_termination" if has_recovery else "recovery_budget_exhausted"
            return observation, reward, True, False, self._info(observation, reason)

        if action_id == "retry_solver":
            self._solver_attempt += 1
            self._verification = _scenario_verification(self._task.scenario, stage="retry_solver")
        elif action_id == "rebuild_model":
            self._model_attempt += 1
            self._solver_attempt += 1
            self._verification = _scenario_verification(self._task.scenario, stage="rebuild_model")

        truncated = self._step_count >= self.max_steps
        self._finished = truncated
        observation = self._observation()
        reward = -0.75 if truncated else -0.05
        reason = "step_budget_exhausted" if truncated else "environment_transition"
        return observation, reward, False, truncated, self._info(observation, reason)

    def _decision(self):
        return decide_after_verification(
            self._verification,
            model_attempt=self._model_attempt,
            solver_attempt=self._solver_attempt,
            max_model_attempts=self.max_model_attempts,
            max_solver_attempts=self.max_solver_attempts,
        )

    def _observation(self) -> dict[str, Any]:
        if self._task is None:
            raise RuntimeError("环境尚未初始化任务。")
        decision = self._decision()
        return {
            "schema_version": "1.0",
            "task_id": self._task.task_id,
            "template_id": self._task.template_id,
            "split": self._task.split,
            "features": self._task.features,
            "verification": self._verification,
            "model_attempt": self._model_attempt,
            "solver_attempt": self._solver_attempt,
            "remaining_model_attempts": max(0, self.max_model_attempts - self._model_attempt),
            "remaining_solver_attempts": max(0, self.max_solver_attempts - self._solver_attempt),
            "step_count": self._step_count,
            "last_action": self._last_action,
            "candidate_actions": decision.candidate_ids,
            "action_mask": decision.action_mask,
        }

    def _info(self, observation: dict[str, Any], terminal_reason: str | None = None) -> dict[str, Any]:
        if self._task is None:
            return {}
        return {
            "task_id": self._task.task_id,
            "scenario": self._task.scenario,
            "terminal_reason": terminal_reason,
            "success": terminal_reason == "verified_solution_accepted",
            "action_mask": observation["action_mask"],
        }

    @staticmethod
    def _normalize_action(action: int | PolicyAction) -> PolicyAction:
        if isinstance(action, int):
            if action < 0 or action >= len(ACTION_IDS):
                raise ValueError(f"动作编号超出范围：{action}")
            return ACTION_IDS[action]
        if action not in ACTION_TO_INDEX:
            raise ValueError(f"未知动作：{action}")
        return action


def _scenario_verification(scenario: str, *, stage: str) -> dict[str, Any]:
    """把可复现故障场景转换为与生产 Verifier 一致的观察。"""

    if scenario == "direct_success":
        return _successful_verification()
    if scenario == "transient_solver_failure":
        return _successful_verification() if stage == "retry_solver" else _unsolved_verification()
    if scenario == "repairable_model_error":
        return _successful_verification() if stage == "rebuild_model" else _invalid_model_verification()
    if scenario == "persistent_failure":
        return _unsolved_verification()
    raise ValueError(f"未知故障场景：{scenario}")


def _successful_verification() -> dict[str, Any]:
    return {
        "passed": True,
        "errors": [],
        "mathematical": {
            "verifiable": True,
            "passed": True,
            "feasible": True,
            "objective_consistent": True,
            "violations": [],
        },
    }


def _unsolved_verification() -> dict[str, Any]:
    return {
        "passed": False,
        "errors": ["未得到可验证的求解结果。"],
        "mathematical": {
            "verifiable": False,
            "passed": False,
            "feasible": None,
            "objective_consistent": None,
            "violations": [],
        },
    }


def _invalid_model_verification() -> dict[str, Any]:
    return {
        "passed": False,
        "errors": ["数学 SolutionVerifier 未通过。"],
        "mathematical": {
            "verifiable": True,
            "passed": False,
            "feasible": False,
            "objective_consistent": False,
            "violations": ["故障注入：模型约束与解不一致。"],
        },
    }
