from __future__ import annotations

from collections.abc import Callable
import json
from pathlib import Path
from statistics import mean
from typing import Any

from optiagent.agent_policy import PolicyAction, decide_after_verification
from optiagent.rl.benchmark import BenchmarkTask
from optiagent.rl.environment import OptimizationAgentEnv


Policy = Callable[[dict[str, Any]], PolicyAction]


def recovery_baseline_policy(observation: dict[str, Any]) -> PolicyAction:
    """使用生产环境相同的确定性 Recovery Policy。"""

    return decide_after_verification(
        observation["verification"],
        model_attempt=int(observation["model_attempt"]),
        solver_attempt=int(observation["solver_attempt"]),
    ).selected_action


def evaluate_policy(
    tasks: list[BenchmarkTask],
    policy: Policy = recovery_baseline_policy,
    *,
    seed: int = 42,
) -> dict[str, Any]:
    """在指定任务集上 rollout policy，返回 episode 与聚合指标。"""

    environment = OptimizationAgentEnv(tasks, seed=seed)
    episodes = [_rollout_task(environment, task, policy) for task in tasks]
    recoverable = [item for item in episodes if item["task"]["scenario"] != "persistent_failure"]
    return {
        "schema_version": "1.0",
        "seed": seed,
        "policy": getattr(policy, "__name__", policy.__class__.__name__),
        "task_count": len(tasks),
        "metrics": {
            "success_rate": round(mean(float(item["success"]) for item in episodes), 6),
            "recoverable_success_rate": (
                round(mean(float(item["success"]) for item in recoverable), 6) if recoverable else 0.0
            ),
            "average_return": round(mean(float(item["total_reward"]) for item in episodes), 6),
            "average_steps": round(mean(float(item["step_count"]) for item in episodes), 6),
            "invalid_action_rate": round(
                mean(float(item["terminal_reason"] == "invalid_action") for item in episodes),
                6,
            ),
        },
        "episodes": episodes,
    }


def export_rollouts_jsonl(result: dict[str, Any], path: str | Path) -> Path:
    """将每个 episode 写为一行 JSON，便于流式训练与版本管理。"""

    output_path = Path(path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lines = [json.dumps(item, ensure_ascii=False) for item in result.get("episodes", [])]
    output_path.write_text("\n".join(lines) + ("\n" if lines else ""), encoding="utf-8")
    return output_path


def _rollout_task(environment: OptimizationAgentEnv, task: BenchmarkTask, policy: Policy) -> dict[str, Any]:
    """执行单任务 episode，保留标准 state/action/reward/next_state。"""

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
                "candidate_actions": state["candidate_actions"],
                "action_mask": state["action_mask"],
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
        "episode_id": f"benchmark-{task.task_id}",
        "task": task.model_dump(mode="json"),
        "recoverable": task.scenario != "persistent_failure",
        "success": bool(info.get("success")),
        "terminal_reason": info.get("terminal_reason"),
        "total_reward": round(total_reward, 6),
        "step_count": len(transitions),
        "transitions": transitions,
    }
