from __future__ import annotations

from copy import deepcopy
from functools import lru_cache
import hashlib
import json
import math
import os
from pathlib import Path
from typing import Any

from optiagent.agent_policy import PolicyDecision, decide_after_verification
from optiagent.instance_identity import instance_fingerprint


def workflow_observation(state: dict[str, Any]) -> dict[str, Any]:
    """将真实工作流映射到恢复训练状态，只读取当前已经发生的观测。"""

    model_attempt = int(state.get("model_attempt", 0))
    solver_attempt = int(state.get("solver_attempt", 0))
    baseline = decide_after_verification(
        state.get("verification") or {}, model_attempt=model_attempt, solver_attempt=solver_attempt,
    )
    decisions = state.get("policy_decisions") or []
    solver_traces = [item for item in state.get("trace_nodes", []) if item.get("node_id") == "solver"]
    latency = float(solver_traces[0].get("elapsed_ms", 0.0)) if solver_traces else 0.0
    latency = max(0.0, latency)
    # 使用训练环境相同的归一化成本单位，数值不是实际货币费用。
    retry_cost = round(0.04 + min(math.log1p(latency) / 25.0, 0.16), 6)
    rebuild_cost = round(retry_cost + 0.12, 6)
    incurred = []
    for decision in decisions:
        previous = decision.get("metadata", {}).get("observation", {})
        action = decision.get("selected_action")
        key = {"retry_solver": "retry_cost_estimate", "rebuild_model": "rebuild_cost_estimate"}.get(action)
        incurred.append(float(previous.get("features", {}).get(key, 0.0)) if key else 0.0)
    cumulative = sum(incurred)
    # 内联 JSON 能精确计数；其他数据源暂记为未知，而不猜测实例规模。
    data = None
    question = str(state.get("question") or "")
    try:
        data = json.loads(question[question.index("{"):question.rindex("}") + 1])
    except ValueError:
        pass
    size = _instance_size(data) if isinstance(data, dict) else 0
    template_id = (state.get("problem_spec") or {}).get("template_id", "")
    data_context = state.get("data_context") or {}
    fingerprint = data_context.get("instance_fingerprint") if template_id == "facility_location" else None
    inline_is_used = data_context.get("source_type") == "inline_json" or (
        template_id != "facility_location" and state.get("requested_dataset_id") is None
        and not data_context.get("uploaded_file_count")
    )
    if not fingerprint and inline_is_used and template_id and isinstance(data, dict) and data:
        fingerprint = instance_fingerprint(template_id, data)
    return {
        "schema_version": "1.0",
        "template_id": template_id,
        "instance_fingerprint": fingerprint,
        "verification": deepcopy(state.get("verification") or {}),
        "features": {
            "instance_size": float(size), "data_density": 1.0 if size else 0.0,
            "difficulty": round(min(1.0, 0.2 + math.log1p(latency) / 8.0), 6),
            "solver_latency_ms": latency,
            "retry_cost_estimate": retry_cost, "rebuild_cost_estimate": rebuild_cost,
        },
        "model_attempt": model_attempt, "solver_attempt": solver_attempt,
        "remaining_model_attempts": max(0, 2 - model_attempt),
        "remaining_solver_attempts": max(0, 2 - solver_attempt),
        "step_count": len(decisions),
        "last_action": decisions[-1]["selected_action"] if decisions else None,
        "last_action_cost": incurred[-1] if incurred else 0.0,
        "cumulative_cost": cumulative, "cost_budget": 0.65,
        "remaining_cost_budget": max(0.0, 0.65 - cumulative),
        "candidate_actions": baseline.candidate_ids, "action_mask": baseline.action_mask,
    }


def _instance_size(value: Any) -> int:
    """与真实恢复数据集一致，以叶子字段数表示结构规模。"""

    if isinstance(value, dict):
        return sum(_instance_size(item) for item in value.values())
    if isinstance(value, list):
        return max(1, sum(_instance_size(item) for item in value))
    return 1


class RecoveryPolicyRuntime:
    """统一规则和 DQN 推理；用规则合同约束网络输出并记录降级原因。"""

    def __init__(self, agent: Any = None, *, checkpoint_sha256: str | None = None) -> None:
        self.agent = agent
        self.checkpoint_sha256 = checkpoint_sha256
        self.name = "masked_double_dqn" if agent is not None else "deterministic_recovery_baseline"
        self.version = checkpoint_sha256 or "1.0"

    def decide(self, state: dict[str, Any]) -> PolicyDecision:
        """每个决策只调用一次网络，非法输出或推理异常回退到合法规则动作。"""

        observation = workflow_observation(state)
        decision = decide_after_verification(
            observation["verification"], model_attempt=observation["model_attempt"],
            solver_attempt=observation["solver_attempt"],
        )
        decision.metadata.update({"observation": observation, "requested_policy": self.name})
        if self.agent is None:
            return decision
        if observation['template_id'] == 'linear_program':
            # 旧检查点只训练过六种业务模板，新类型使用规则而不伪称已泛化。
            decision.metadata['fallback_reason'] = 'unsupported_template'
            return decision
        decision.metadata["checkpoint_sha256"] = self.checkpoint_sha256
        checkpoint_metadata = getattr(self.agent, "checkpoint_metadata", {})
        if isinstance(checkpoint_metadata, dict):
            decision.metadata["training_stage"] = (
                checkpoint_metadata.get("selection", {}).get("selected_stage") or checkpoint_metadata.get("stage")
            )
        try:
            selected = self.agent.policy(deepcopy(observation))
            enabled = dict(zip(decision.candidate_ids, decision.action_mask, strict=True))
            if not isinstance(selected, str) or not enabled.get(selected, False):
                decision.metadata["fallback_reason"] = "invalid_or_masked_action"
                return decision
        except Exception as exc:
            # 不写原始异常文本，防止运行环境信息进入轨迹。
            decision.metadata["fallback_reason"] = "inference_error"
            decision.metadata["error_type"] = type(exc).__name__
            return decision
        decision.selected_action = selected
        decision.policy_name = self.name
        decision.policy_version = self.version
        decision.reason = "学习策略在当前合法动作中选择下一步。"
        return decision


def load_recovery_runtime(checkpoint: str | Path | None = None) -> RecoveryPolicyRuntime:
    """配置可来自服务环境；空字符串显式选择规则，不加载可选 PyTorch。"""

    selected = os.environ.get("OPTIAGENT_RECOVERY_CHECKPOINT", "") if checkpoint is None else str(checkpoint)
    if not selected:
        return RecoveryPolicyRuntime()
    path = Path(selected).expanduser().resolve()
    stat = path.stat()
    return _load_checkpoint(str(path), stat.st_mtime_ns, stat.st_size)


@lru_cache(maxsize=4)
def _load_checkpoint(path: str, modified_ns: int, size: int) -> RecoveryPolicyRuntime:
    """按文件版本缓存推理对象；加载错误直接暴露，避免悄悄使用错误模型。"""

    from optiagent.rl.dqn import MaskedDoubleDQNAgent

    agent = MaskedDoubleDQNAgent.load(path, device="cpu")
    if agent.state_encoder.version not in {"1.0", "2.0"}:
        raise ValueError("主工作流尚无完整 transport 观测，请使用 v1/v2 checkpoint。")
    agent.policy_network.eval()
    digest = hashlib.sha256(Path(path).read_bytes()).hexdigest()
    return RecoveryPolicyRuntime(agent, checkpoint_sha256=digest)
