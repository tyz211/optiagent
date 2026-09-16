from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
import math
import random
from statistics import mean
from typing import Any, Literal

from pydantic import BaseModel

from optiagent.agent_policy import PolicyAction, decide_after_verification
from optiagent.rl.benchmark import BenchmarkSplit, TEMPLATE_IDS
from optiagent.rl.environment import ACTION_IDS
from optiagent.rl.mcp_transport import MCPTransportTrace, collect_mcp_transport_traces
from optiagent.rl.real_environment import collect_real_recovery_tasks


TRANSPORT_ENVIRONMENT_VERSION = "mcp-transport-recovery-v1"
MCPTransportScenario = Literal[
    "normal",
    "timeout",
    "disconnect",
    "invalid_return",
    "persistent_timeout",
]
MCP_TRANSPORT_SCENARIOS: tuple[MCPTransportScenario, ...] = (
    "normal",
    "timeout",
    "disconnect",
    "invalid_return",
    "persistent_timeout",
)


class MCPTransportRecoveryTask(BaseModel):
    """将真实 MCP transport 观测与真实求解参考结果组合为恢复任务。"""

    schema_version: str = "1.0"
    task_id: str
    template_id: str
    split: BenchmarkSplit
    scenario: MCPTransportScenario
    recoverable: bool
    transport_trace: MCPTransportTrace
    features: dict[str, float]
    clean_verification: dict[str, Any]
    contract_error_verification: dict[str, Any]


def collect_transport_recovery_tasks(
    *,
    seed: int = 52,
    template_ids: Iterable[str] = TEMPLATE_IDS,
    splits: tuple[BenchmarkSplit, ...] = ("train", "validation", "test"),
    timeout_ms: int = 80,
    delay_ms: int = 240,
    time_limit: int = 10,
) -> list[MCPTransportRecoveryTask]:
    """采集真实 stdio MCP 故障，并为六类优化模板构建隔离任务集。"""

    selected_templates = tuple(template_ids)
    base_tasks = collect_real_recovery_tasks(
        seed=seed,
        template_ids=selected_templates,
        splits=splits,
        scenarios=("direct_success",),
        time_limit=time_limit,
    )
    base_by_key = {(item.split, item.template_id): item for item in base_tasks}
    tasks: list[MCPTransportRecoveryTask] = []
    for split in splits:
        traces = collect_mcp_transport_traces(
            timeout_ms=timeout_ms,
            delay_ms=delay_ms,
            nonce_prefix=f"{split}-s{seed}",
        )
        trace_by_fault = {item.fault: item for item in traces}
        for template_id in selected_templates:
            base = base_by_key[(split, template_id)]
            for scenario in MCP_TRANSPORT_SCENARIOS:
                trace_fault = "timeout" if scenario == "persistent_timeout" else scenario
                trace = trace_by_fault[trace_fault]
                tasks.append(_build_transport_task(base, trace, scenario))
    return tasks


class MCPTransportRecoveryEnv:
    """学习在 MCP 超时、断连和非法返回后重试、重建请求或终止。"""

    def __init__(
        self,
        tasks: list[MCPTransportRecoveryTask],
        *,
        seed: int = 52,
        max_model_attempts: int = 2,
        max_solver_attempts: int = 2,
        max_steps: int = 4,
        cost_budget: float = 1.0,
    ) -> None:
        if not tasks:
            raise ValueError("MCP transport 恢复环境至少需要一个任务。")
        self.tasks = list(tasks)
        self.max_model_attempts = max_model_attempts
        self.max_solver_attempts = max_solver_attempts
        self.max_steps = max_steps
        self.cost_budget = cost_budget
        self._randomizer = random.Random(seed)
        self._task: MCPTransportRecoveryTask | None = None
        self._verification: dict[str, Any] = {}
        self._model_attempt = 1
        self._solver_attempt = 1
        self._step_count = 0
        self._transport_failure_count = 0
        self._last_action: PolicyAction | None = None
        self._last_action_cost = 0.0
        self._cumulative_cost = 0.0
        self._finished = False

    def reset(
        self,
        *,
        seed: int | None = None,
        task_id: str | None = None,
    ) -> tuple[dict[str, Any], dict[str, Any]]:
        """选择任务并返回包含 transport 观测、但不含场景标签的状态。"""

        if seed is not None:
            self._randomizer.seed(seed)
        if task_id is None:
            self._task = self._randomizer.choice(self.tasks)
        else:
            self._task = next((item for item in self.tasks if item.task_id == task_id), None)
            if self._task is None:
                raise KeyError(f"未找到 MCP transport 任务：{task_id}")
        self._verification = self._initial_verification(self._task)
        self._model_attempt = 1
        self._solver_attempt = 1
        self._step_count = 0
        self._transport_failure_count = int(self._task.scenario != "normal")
        self._last_action = None
        self._last_action_cost = 0.0
        self._cumulative_cost = 0.0
        self._finished = False
        observation = self._observation()
        return observation, self._info(observation)

    def step(
        self,
        action: int | PolicyAction,
    ) -> tuple[dict[str, Any], float, bool, bool, dict[str, Any]]:
        """执行 transport 恢复动作并累计真实连接耗时对应的成本。"""

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
            self._last_action_cost = 0.0
            observation = self._observation()
            return observation, -1.0, True, False, self._info(observation, "invalid_action")
        if action_id == "accept_solution":
            self._finished = True
            self._last_action_cost = 0.0
            observation = self._observation()
            return observation, 1.0, True, False, self._info(observation, "verified_solution_accepted")
        if action_id == "terminate":
            self._finished = True
            self._last_action_cost = 0.0
            has_recovery = any(enabled.get(candidate, False) for candidate in ("retry_solver", "rebuild_model"))
            reward = -0.6 if has_recovery else -0.2
            reason = "premature_termination" if has_recovery else "recovery_budget_exhausted"
            observation = self._observation()
            return observation, reward, True, False, self._info(observation, reason)

        self._last_action_cost = self._action_cost(action_id)
        self._cumulative_cost += self._last_action_cost
        if action_id == "retry_solver":
            self._solver_attempt += 1
            self._verification = self._after_retry(self._task)
        elif action_id == "rebuild_model":
            self._model_attempt += 1
            self._solver_attempt += 1
            self._verification = self._after_rebuild(self._task)
        if not self._verification.get("passed"):
            self._transport_failure_count += 1

        truncated = self._step_count >= self.max_steps or self._cumulative_cost > self.cost_budget
        self._finished = truncated
        observation = self._observation()
        reward = -0.75 - self._last_action_cost if truncated else -self._last_action_cost
        reason = "cost_or_step_budget_exhausted" if truncated else "environment_transition"
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
            "environment_version": TRANSPORT_ENVIRONMENT_VERSION,
            "task_id": self._task.task_id,
            "template_id": self._task.template_id,
            "split": self._task.split,
            "features": self._task.features,
            "verification": deepcopy(self._verification),
            "model_attempt": self._model_attempt,
            "solver_attempt": self._solver_attempt,
            "remaining_model_attempts": max(0, self.max_model_attempts - self._model_attempt),
            "remaining_solver_attempts": max(0, self.max_solver_attempts - self._solver_attempt),
            "step_count": self._step_count,
            "transport_failure_count": self._transport_failure_count,
            "last_action": self._last_action,
            "last_action_cost": round(self._last_action_cost, 6),
            "cumulative_cost": round(self._cumulative_cost, 6),
            "cost_budget": self.cost_budget,
            "remaining_cost_budget": round(max(0.0, self.cost_budget - self._cumulative_cost), 6),
            "candidate_actions": decision.candidate_ids,
            "action_mask": decision.action_mask,
        }

    def _info(self, observation: dict[str, Any], terminal_reason: str | None = None) -> dict[str, Any]:
        if self._task is None:
            return {}
        return {
            "task_id": self._task.task_id,
            "scenario": self._task.scenario,
            "recoverable": self._task.recoverable,
            "terminal_reason": terminal_reason,
            "success": terminal_reason == "verified_solution_accepted",
            "action_mask": observation["action_mask"],
            "cumulative_cost": round(self._cumulative_cost, 6),
            "transport_elapsed_ms": self._task.transport_trace.elapsed_ms,
        }

    def _action_cost(self, action: PolicyAction) -> float:
        if self._task is None:
            return 0.0
        key = "retry_cost_estimate" if action == "retry_solver" else "rebuild_cost_estimate"
        return float(self._task.features[key])

    @staticmethod
    def _initial_verification(task: MCPTransportRecoveryTask) -> dict[str, Any]:
        if task.scenario == "normal":
            return deepcopy(task.clean_verification)
        if task.scenario == "invalid_return":
            return deepcopy(task.contract_error_verification)
        return _transport_failure_verification()

    @staticmethod
    def _after_retry(task: MCPTransportRecoveryTask) -> dict[str, Any]:
        if task.scenario in {"timeout", "disconnect"}:
            return deepcopy(task.clean_verification)
        if task.scenario == "invalid_return":
            return deepcopy(task.contract_error_verification)
        return _transport_failure_verification()

    @staticmethod
    def _after_rebuild(task: MCPTransportRecoveryTask) -> dict[str, Any]:
        if task.scenario == "invalid_return":
            return deepcopy(task.clean_verification)
        return _transport_failure_verification()

    @staticmethod
    def _normalize_action(action: int | PolicyAction) -> PolicyAction:
        if isinstance(action, int):
            if action < 0 or action >= len(ACTION_IDS):
                raise ValueError(f"动作编号超出范围：{action}")
            return ACTION_IDS[action]
        if action not in ACTION_IDS:
            raise ValueError(f"未知动作：{action}")
        return action


def transport_teacher_policy(observation: dict[str, Any]) -> PolicyAction:
    """根据真实 transport 信号选择重试、重建合同或终止。"""

    decision = decide_after_verification(
        observation["verification"],
        model_attempt=int(observation["model_attempt"]),
        solver_attempt=int(observation["solver_attempt"]),
    )
    return decision.selected_action


def evaluate_transport_policy(
    tasks: list[MCPTransportRecoveryTask],
    policy: Callable[[dict[str, Any]], PolicyAction],
    *,
    seed: int = 52,
) -> dict[str, Any]:
    """评估 MCP transport 恢复策略的成功率、回报、成本和步骤数。"""

    environment = MCPTransportRecoveryEnv(tasks, seed=seed)
    episodes = [_rollout_transport_task(environment, task, policy) for task in tasks]
    recoverable = [item for item in episodes if item["recoverable"]]
    return {
        "schema_version": "1.0",
        "environment_version": TRANSPORT_ENVIRONMENT_VERSION,
        "seed": seed,
        "task_count": len(tasks),
        "metrics": {
            "success_rate": _mean_flag(item["success"] for item in episodes),
            "recoverable_success_rate": _mean_flag(item["success"] for item in recoverable),
            "average_return": round(mean(item["total_reward"] for item in episodes), 6),
            "average_steps": round(mean(item["step_count"] for item in episodes), 6),
            "average_action_cost": round(mean(item["cumulative_cost"] for item in episodes), 6),
            "average_transport_elapsed_ms": round(mean(item["transport_elapsed_ms"] for item in episodes), 6),
            "invalid_action_rate": _mean_flag(item["terminal_reason"] == "invalid_action" for item in episodes),
        },
        "episodes": episodes,
    }


def _build_transport_task(base, trace: MCPTransportTrace, scenario: MCPTransportScenario) -> MCPTransportRecoveryTask:
    elapsed_ms = trace.elapsed_ms
    retry_cost = round(0.03 + min(math.log1p(elapsed_ms) / 30.0, 0.22), 6)
    rebuild_cost = round(retry_cost + 0.14, 6)
    features = {
        **base.features,
        "transport_timeout": float(scenario in {"timeout", "persistent_timeout"}),
        "transport_disconnected": float(scenario == "disconnect"),
        "transport_response_schema_valid": float(trace.response_schema_valid),
        "transport_elapsed_ms": elapsed_ms,
        "retry_cost_estimate": retry_cost,
        "rebuild_cost_estimate": rebuild_cost,
    }
    return MCPTransportRecoveryTask(
        task_id=f"{base.template_id}-{base.split}-mcp-{scenario}-{trace.trace_id}",
        template_id=base.template_id,
        split=base.split,
        scenario=scenario,
        recoverable=scenario != "persistent_timeout",
        transport_trace=trace,
        features=features,
        clean_verification=base.clean_verification,
        contract_error_verification=_contract_error_verification(),
    )


def _transport_failure_verification() -> dict[str, Any]:
    return {
        "passed": False,
        "errors": ["MCP transport 未返回可验证结果。"],
        "mathematical": {
            "verifiable": False,
            "passed": False,
            "feasible": None,
            "objective_consistent": None,
            "violations": [],
        },
    }


def _contract_error_verification() -> dict[str, Any]:
    return {
        "passed": False,
        "errors": ["MCP 返回不符合 SolveEnvelope 合同。"],
        "mathematical": {
            "verifiable": True,
            "passed": False,
            "feasible": False,
            "objective_consistent": False,
            "violations": ["MCP structuredContent 缺少版本化结果字段。"],
        },
    }


def _rollout_transport_task(environment, task, policy) -> dict[str, Any]:
    state, info = environment.reset(task_id=task.task_id)
    transitions = []
    total_reward = 0.0
    terminated = False
    truncated = False
    while not (terminated or truncated):
        action = policy(state)
        next_state, reward, terminated, truncated, info = environment.step(action)
        transitions.append(
            {
                "t": len(transitions),
                "state": state,
                "action": action,
                "reward": reward,
                "next_state": next_state,
                "done": terminated or truncated,
            }
        )
        total_reward += reward
        state = next_state
    return {
        "task_id": task.task_id,
        "template_id": task.template_id,
        "split": task.split,
        "recoverable": task.recoverable,
        "success": bool(info.get("success")),
        "terminal_reason": info.get("terminal_reason"),
        "total_reward": round(total_reward, 6),
        "step_count": len(transitions),
        "cumulative_cost": float(info.get("cumulative_cost", 0.0)),
        "transport_elapsed_ms": task.transport_trace.elapsed_ms,
        "transitions": transitions,
    }


def _mean_flag(values: Iterable[bool]) -> float:
    items = [float(value) for value in values]
    return round(mean(items), 6) if items else 0.0
