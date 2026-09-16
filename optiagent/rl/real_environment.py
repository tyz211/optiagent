from __future__ import annotations

from collections.abc import Callable, Iterable
from copy import deepcopy
import math
import random
from statistics import mean
from time import perf_counter
from typing import Any, Literal

from pydantic import BaseModel, Field

from optiagent.agent_policy import PolicyAction, decide_after_verification
from optiagent.mcp_contracts import ProblemEnvelope, SolveEnvelope, SourceReference
from optiagent.optimization_gateway import build_problem_envelope, solve_problem_envelope
from optiagent.rl.benchmark import BenchmarkSplit, TEMPLATE_IDS
from optiagent.rl.e2e_benchmark import EndToEndBenchmarkInstance, build_smoke_instances
from optiagent.rl.environment import ACTION_IDS
from optiagent.solution_verifier import verify_solution


REAL_ENVIRONMENT_VERSION = "gateway-recovery-v1.1"
RealRecoveryScenario = Literal[
    "direct_success",
    "transient_solver_failure",
    "tampered_objective",
    "repairable_model_error",
    "persistent_failure",
]
REAL_RECOVERY_SCENARIOS: tuple[RealRecoveryScenario, ...] = (
    "direct_success",
    "transient_solver_failure",
    "tampered_objective",
    "repairable_model_error",
    "persistent_failure",
)


class RealRecoveryTask(BaseModel):
    """由真实 Gateway/Solver/Verifier 结果构造的恢复任务。"""

    schema_version: str = "1.0"
    task_id: str
    template_id: str
    split: BenchmarkSplit
    scenario: RealRecoveryScenario
    recoverable: bool
    solver_name: str
    objective_value: float | None = None
    solver_latency_ms: float = Field(ge=0.0)
    features: dict[str, float]
    clean_verification: dict[str, Any]
    tampered_verification: dict[str, Any]
    model_error_verification: dict[str, Any]


def collect_real_recovery_tasks(
    *,
    seed: int = 42,
    template_ids: Iterable[str] = TEMPLATE_IDS,
    splits: tuple[BenchmarkSplit, ...] = ("train", "validation", "test"),
    scenarios: tuple[RealRecoveryScenario, ...] = REAL_RECOVERY_SCENARIOS,
    time_limit: int = 10,
) -> list[RealRecoveryTask]:
    """运行真实小规模优化问题，收集可复现的恢复训练任务。"""

    selected_templates = tuple(template_ids)
    unknown = set(selected_templates) - set(TEMPLATE_IDS)
    if unknown:
        raise ValueError(f"未知优化模板：{sorted(unknown)}")
    tasks: list[RealRecoveryTask] = []
    for split_index, split in enumerate(splits):
        for scenario_index, scenario in enumerate(scenarios):
            instance_seed = seed + split_index * 1000 + scenario_index * 100
            instances = {item.template_id: item for item in build_smoke_instances(seed=instance_seed)}
            for template_id in selected_templates:
                task = _collect_task(
                    instances[template_id],
                    split=split,
                    scenario=scenario,
                    time_limit=time_limit,
                )
                tasks.append(task)
    return tasks


class RealGatewayRecoveryEnv:
    """在真实求解轨迹上模拟结果损坏、求解失败和模型错误的恢复环境。"""

    def __init__(
        self,
        tasks: list[RealRecoveryTask],
        *,
        seed: int = 42,
        max_model_attempts: int = 2,
        max_solver_attempts: int = 2,
        max_steps: int = 4,
        cost_budget: float = 0.65,
    ) -> None:
        if not tasks:
            raise ValueError("真实恢复环境至少需要一个任务。")
        self.tasks = list(tasks)
        self.max_model_attempts = max_model_attempts
        self.max_solver_attempts = max_solver_attempts
        self.max_steps = max_steps
        self.cost_budget = cost_budget
        self._randomizer = random.Random(seed)
        self._task: RealRecoveryTask | None = None
        self._verification: dict[str, Any] = {}
        self._model_attempt = 1
        self._solver_attempt = 1
        self._step_count = 0
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
        """选择一个真实任务，并返回不包含故障标签的初始观察。"""

        if seed is not None:
            self._randomizer.seed(seed)
        if task_id is None:
            self._task = self._randomizer.choice(self.tasks)
        else:
            self._task = next((item for item in self.tasks if item.task_id == task_id), None)
            if self._task is None:
                raise KeyError(f"未找到真实恢复任务：{task_id}")
        self._verification = self._initial_verification(self._task)
        self._model_attempt = 1
        self._solver_attempt = 1
        self._step_count = 0
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
        """执行恢复动作，并按真实耗时估计扣除动作成本。"""

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
            "environment_version": REAL_ENVIRONMENT_VERSION,
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
            "solver_latency_ms": self._task.solver_latency_ms,
        }

    def _action_cost(self, action: PolicyAction) -> float:
        if self._task is None:
            return 0.0
        key = "retry_cost_estimate" if action == "retry_solver" else "rebuild_cost_estimate"
        return float(self._task.features[key])

    @staticmethod
    def _initial_verification(task: RealRecoveryTask) -> dict[str, Any]:
        if task.scenario == "direct_success":
            return deepcopy(task.clean_verification)
        if task.scenario == "tampered_objective":
            return deepcopy(task.tampered_verification)
        if task.scenario == "repairable_model_error":
            return deepcopy(task.model_error_verification)
        return _unsolved_verification()

    @staticmethod
    def _after_retry(task: RealRecoveryTask) -> dict[str, Any]:
        if task.scenario in {"transient_solver_failure", "tampered_objective"}:
            return deepcopy(task.clean_verification)
        if task.scenario == "repairable_model_error":
            return deepcopy(task.model_error_verification)
        return _unsolved_verification()

    @staticmethod
    def _after_rebuild(task: RealRecoveryTask) -> dict[str, Any]:
        if task.scenario in {"tampered_objective", "repairable_model_error"}:
            return deepcopy(task.clean_verification)
        return _unsolved_verification()

    @staticmethod
    def _normalize_action(action: int | PolicyAction) -> PolicyAction:
        if isinstance(action, int):
            if action < 0 or action >= len(ACTION_IDS):
                raise ValueError(f"动作编号超出范围：{action}")
            return ACTION_IDS[action]
        if action not in ACTION_IDS:
            raise ValueError(f"未知动作：{action}")
        return action


def cost_aware_teacher_policy(observation: dict[str, Any]) -> PolicyAction:
    """区分结果传输损坏与模型不可行，选择最低成本的正确恢复动作。"""

    decision = decide_after_verification(
        observation["verification"],
        model_attempt=int(observation["model_attempt"]),
        solver_attempt=int(observation["solver_attempt"]),
    )
    enabled = dict(zip(decision.candidate_ids, decision.action_mask, strict=True))
    mathematical = observation["verification"].get("mathematical") or {}
    if enabled.get("accept_solution"):
        return "accept_solution"
    if (
        mathematical.get("verifiable") is True
        and mathematical.get("feasible") is True
        and mathematical.get("objective_consistent") is False
        and enabled.get("retry_solver")
    ):
        return "retry_solver"
    if mathematical.get("feasible") is False and enabled.get("rebuild_model"):
        return "rebuild_model"
    if enabled.get("retry_solver"):
        return "retry_solver"
    return "terminate"


def evaluate_real_policy(
    tasks: list[RealRecoveryTask],
    policy: Callable[[dict[str, Any]], PolicyAction],
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """在真实恢复任务上评估成功率、累计奖励、动作成本和步骤数。"""

    environment = RealGatewayRecoveryEnv(tasks, seed=seed)
    episodes = [_rollout_real_task(environment, task, policy) for task in tasks]
    recoverable = [item for item in episodes if item["recoverable"]]
    return {
        "schema_version": "1.0",
        "environment_version": REAL_ENVIRONMENT_VERSION,
        "seed": seed,
        "policy": getattr(policy, "__name__", policy.__class__.__name__),
        "task_count": len(tasks),
        "metrics": {
            "success_rate": _mean_flag(item["success"] for item in episodes),
            "recoverable_success_rate": _mean_flag(item["success"] for item in recoverable),
            "average_return": round(mean(item["total_reward"] for item in episodes), 6),
            "average_steps": round(mean(item["step_count"] for item in episodes), 6),
            "average_action_cost": round(mean(item["cumulative_cost"] for item in episodes), 6),
            "average_solver_latency_ms": round(mean(item["solver_latency_ms"] for item in episodes), 6),
            "invalid_action_rate": _mean_flag(item["terminal_reason"] == "invalid_action" for item in episodes),
        },
        "episodes": episodes,
    }


def _collect_task(
    instance: EndToEndBenchmarkInstance,
    *,
    split: BenchmarkSplit,
    scenario: RealRecoveryScenario,
    time_limit: int,
) -> RealRecoveryTask:
    problem, solved, elapsed_ms = _solve_reference(instance, time_limit=time_limit)
    clean_report = solved.solution_verification
    if clean_report is None or not clean_report.passed:
        raise RuntimeError(f"真实参考求解未通过验证：{instance.task_id}")
    tampered = solved.model_copy(
        update={
            "objective_value": float(solved.objective_value or 0.0) + max(1.0, abs(float(solved.objective_value or 0.0)) * 0.25),
            "solution_verification": None,
        }
    )
    # 同时清空决策和汇总指标，避免 TSP 等模板从 metrics 恢复出原始正确解。
    model_error = solved.model_copy(update={"decisions": [], "metrics": {}, "solution_verification": None})
    tampered_report = verify_solution(problem, tampered)
    model_error_report = verify_solution(problem, model_error)
    retry_cost = round(0.04 + min(math.log1p(elapsed_ms) / 25.0, 0.16), 6)
    rebuild_cost = round(retry_cost + 0.12, 6)
    return RealRecoveryTask(
        task_id=f"{instance.template_id}-{split}-{scenario}-{instance.task_id}",
        template_id=instance.template_id,
        split=split,
        scenario=scenario,
        recoverable=scenario != "persistent_failure",
        solver_name=solved.solver_name,
        objective_value=solved.objective_value,
        solver_latency_ms=elapsed_ms,
        features={
            "instance_size": float(_instance_size(instance.data)),
            "data_density": 1.0,
            "difficulty": round(min(1.0, 0.2 + math.log1p(elapsed_ms) / 8.0), 6),
            "solver_latency_ms": elapsed_ms,
            "retry_cost_estimate": retry_cost,
            "rebuild_cost_estimate": rebuild_cost,
        },
        clean_verification=_wrap_verification(clean_report),
        tampered_verification=_wrap_verification(tampered_report),
        model_error_verification=_wrap_verification(model_error_report),
    )


def _solve_reference(
    instance: EndToEndBenchmarkInstance,
    *,
    time_limit: int,
) -> tuple[ProblemEnvelope, SolveEnvelope, float]:
    problem = build_problem_envelope(
        instance.template_id,
        deepcopy(instance.data),
        question=instance.question,
        sources=[SourceReference(kind="inline", name=f"real-recovery:{instance.task_id}")],
        metadata={"benchmark": REAL_ENVIRONMENT_VERSION},
    )
    started = perf_counter()
    solved = solve_problem_envelope(problem, time_limit=time_limit)
    elapsed_ms = round((perf_counter() - started) * 1000.0, 6)
    return problem, solved, elapsed_ms


def _wrap_verification(report) -> dict[str, Any]:
    mathematical = report.model_dump(mode="json")
    return {
        "passed": bool(report.passed),
        "errors": [] if report.passed else ["数学 SolutionVerifier 未通过。"],
        "mathematical": mathematical,
    }


def _unsolved_verification() -> dict[str, Any]:
    return {
        "passed": False,
        "errors": ["真实求解链路未返回可验证结果。"],
        "mathematical": {
            "verifiable": False,
            "passed": False,
            "feasible": None,
            "objective_consistent": None,
            "violations": [],
        },
    }


def _instance_size(value: Any) -> int:
    if isinstance(value, dict):
        return sum(_instance_size(item) for item in value.values())
    if isinstance(value, list):
        return max(1, sum(_instance_size(item) for item in value))
    return 1


def _rollout_real_task(
    environment: RealGatewayRecoveryEnv,
    task: RealRecoveryTask,
    policy: Callable[[dict[str, Any]], PolicyAction],
) -> dict[str, Any]:
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
        "schema_version": "1.0",
        "episode_id": f"real-gateway-{task.task_id}",
        "task_id": task.task_id,
        "template_id": task.template_id,
        "split": task.split,
        "recoverable": task.recoverable,
        "success": bool(info.get("success")),
        "terminal_reason": info.get("terminal_reason"),
        "total_reward": round(total_reward, 6),
        "step_count": len(transitions),
        "cumulative_cost": float(info.get("cumulative_cost", 0.0)),
        "solver_latency_ms": task.solver_latency_ms,
        "transitions": transitions,
    }


def _mean_flag(values: Iterable[bool]) -> float:
    items = [float(value) for value in values]
    return round(mean(items), 6) if items else 0.0
