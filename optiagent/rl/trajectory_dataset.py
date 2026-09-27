from __future__ import annotations

from collections import Counter
from collections.abc import Iterable, Iterator
from contextlib import closing
import hashlib
import json
import math
from pathlib import Path
import re
import sqlite3
from typing import Any

from optiagent.agent_policy import decide_after_verification
from optiagent.rl.environment import ACTION_IDS


DATASET_VERSION = "workflow-recovery-offline-v1"
REWARD_VERSION = "workflow-sparse-terminal-v1"
_FINGERPRINT = re.compile(r"[0-9a-f]{64}\Z")
_NUMERIC_KEYS = ("model_attempt", "solver_attempt", "remaining_model_attempts", "remaining_solver_attempts",
                 "step_count", "last_action_cost", "cumulative_cost", "cost_budget", "remaining_cost_budget")
_FEATURE_KEYS = ("instance_size", "data_density", "difficulty", "solver_latency_ms", "retry_cost_estimate", "rebuild_cost_estimate")


def read_database_episodes(path: Path, *, user_id: int | None) -> Iterator[dict]:
    """在只读事务内读取指定用户全部轨迹，不迁移数据库，也不受 Web 列表的 500 条限制。"""

    from api.database import build_training_episode

    # sqlite 的事务上下文不会关闭连接，使用 closing 显式释放只读快照。
    with closing(sqlite3.connect(path.resolve().as_uri() + "?mode=ro", uri=True)) as connection:
        connection.row_factory = sqlite3.Row
        connection.execute("BEGIN")
        clause, parameters = ("user_id IS NULL", ()) if user_id is None else ("user_id = ?", (user_id,))
        for row in connection.execute(f"SELECT * FROM agent_episodes WHERE {clause} ORDER BY episode_id", parameters):
            episode = dict(row)
            for key in ("reward_json", "result_json"):
                episode[key.removesuffix("_json")] = json.loads(episode.pop(key) or "{}")
            episode["steps"] = []
            for step_row in connection.execute("SELECT * FROM agent_steps WHERE episode_id = ? ORDER BY sequence, attempt, id", (episode["episode_id"],)):
                step = dict(step_row)
                for key in ("state_json", "action_json", "observation_json", "reward_json"):
                    step[key.removesuffix("_json")] = json.loads(step.pop(key) or "{}")
                episode["steps"].append(step)
            yield build_training_episode(episode)


def split_for_instance(fingerprint: str, seed: int) -> str:
    """内容分组采用稳定哈希；增加新 episode 不会挪动已有实例所在集合。"""

    bucket = int(hashlib.sha256(f"{seed}:{fingerprint}".encode()).hexdigest(), 16) % 100
    return "train" if bucket < 70 else "validation" if bucket < 85 else "test"


def export_trajectory_dataset(episodes: Iterable[dict], output: Path, *, seed: int = 42,
                              source: str = "live_workflow") -> dict:
    """校验、脱敏并分组导出；不完整轨迹进入隔离清单，成功与失败样本均保留。"""

    if source not in {"live_workflow", "controlled_workflow_benchmark"}:
        raise ValueError("必须明确指定轨迹来源。")
    output.mkdir(parents=True, exist_ok=False)
    accepted, rejected = [], []
    seen = set()
    groups = {name: set() for name in ("train", "validation", "test")}
    outcomes: Counter = Counter()
    for episode in episodes:
        reference = hashlib.sha256(str(episode.get("episode_id", "")).encode()).hexdigest()
        try:
            if not episode.get("episode_id") or reference in seen:
                raise ValueError("duplicate_or_missing_episode_id")
            seen.add(reference)
            if episode.get("trajectory_source") != source:
                raise ValueError("source_mismatch")
            sample = _validated_episode(episode)
            sample.update(episode_ref=reference, trajectory_source=source)
            sample["split"] = split_for_instance(sample["instance_fingerprint"], seed)
            groups[sample["split"]].add(sample["instance_fingerprint"])
            outcomes[sample["outcome"]] += 1
            accepted.append(sample)
        except (ValueError, KeyError, TypeError, IndexError) as exc:
            # 仅允许预定义原因码，原始输入和异常文本不进入隔离文件。
            reason = str(exc) if isinstance(exc, ValueError) and str(exc) in _REASONS else "invalid_contract"
            rejected.append({"episode_ref": reference, "reason": reason})
    files = {}
    for split in groups:
        path = output / f"{split}.jsonl"
        _write_jsonl(path, [row for row in accepted if row["split"] == split])
        files[path.name] = _file_metadata(path)
    _write_jsonl(output / "quarantine.jsonl", rejected)
    files["quarantine.jsonl"] = _file_metadata(output / "quarantine.jsonl")
    manifest = {
        "schema_version": DATASET_VERSION, "reward_version": REWARD_VERSION, "state_encoder_version": "2.0",
        "trajectory_source": source, "split_seed": seed, "split_method": "instance_sha256_70_15_15",
        "accepted_episodes": len(accepted), "rejected_episodes": len(rejected),
        "transition_count": sum(len(row["transitions"]) for row in accepted),
        "outcomes": dict(outcomes), "rejection_reasons": dict(Counter(row["reason"] for row in rejected)),
        "unique_instances_by_split": {key: len(value) for key, value in groups.items()},
        "cross_split_overlap_count": 0, "all_splits_nonempty": all(groups.values()),
        "files": files,
        "limitations": ["按输入内容分组，不等于变量重命名同构去重。", "保留应用稀疏终局奖励，未重算为成本感知环境奖励。",
                        "未记录行为概率，不能直接用于重要性采样估计。"],
    }
    (output / "manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2), encoding="utf-8")
    return manifest


_REASONS = {"duplicate_or_missing_episode_id", "source_mismatch", "incomplete_episode", "missing_instance_fingerprint",
            "missing_policy_observation", "no_recovery_decisions", "invalid_action_mask", "broken_transition_chain",
            "invalid_reward", "invalid_contract"}


def _validated_episode(episode: dict) -> dict:
    """逐步校验动作、预算、终止、链条和 reward，防止用补零数据掩盖缺失状态。"""

    if episode.get("status") not in {"completed", "failed"}:
        raise ValueError("incomplete_episode")
    fingerprint = episode.get("task", {}).get("instance_fingerprint")
    if not isinstance(fingerprint, str) or not _FINGERPRINT.fullmatch(fingerprint):
        raise ValueError("missing_instance_fingerprint")
    rows = episode.get("decision_transitions") or []
    if not rows:
        raise ValueError("no_recovery_decisions")
    if episode["status"] == "completed" and rows[-1].get("action") not in {"accept_solution", "terminate"}:
        raise ValueError("broken_transition_chain")
    transitions = []
    for index, row in enumerate(rows):
        state = row.get("policy_observation")
        if not isinstance(state, dict):
            raise ValueError("missing_policy_observation")
        if state.get("instance_fingerprint") != fingerprint or state.get("template_id") != episode["task"]["template_id"]:
            raise ValueError("invalid_contract")
        clean = _safe_observation(state)
        expected = decide_after_verification(clean["verification"], model_attempt=clean["model_attempt"], solver_attempt=clean["solver_attempt"])
        if (row.get("candidate_actions") != list(ACTION_IDS) or row.get("action_mask") != expected.action_mask
                or clean["action_mask"] != expected.action_mask or row.get("action") not in ACTION_IDS):
            raise ValueError("invalid_action_mask")
        if not expected.action_mask[ACTION_IDS.index(row["action"])]:
            raise ValueError("invalid_action_mask")
        done = index == len(rows) - 1
        if row.get("done") is not done or row.get("status") != "completed":
            raise ValueError("broken_transition_chain")
        next_state = row.get("next_policy_observation")
        if done and next_state is not None:
            raise ValueError("broken_transition_chain")
        if not done and (next_state != rows[index + 1].get("policy_observation") or next_state is None):
            raise ValueError("broken_transition_chain")
        if not done:
            if row["action"] not in {"retry_solver", "rebuild_model"}:
                raise ValueError("broken_transition_chain")
            expected_model = clean["model_attempt"] + int(row["action"] == "rebuild_model")
            if (next_state.get("last_action") != row["action"] or next_state.get("step_count") != clean["step_count"] + 1
                    or next_state.get("solver_attempt") != clean["solver_attempt"] + 1
                    or next_state.get("model_attempt") != expected_model):
                raise ValueError("broken_transition_chain")
        reward = _finite(row.get("reward"))
        if reward != (_finite(episode.get("total_reward")) if done else 0.0):
            raise ValueError("invalid_reward")
        # 导出状态白名单与编码器一致，不保留问题原文、文件路径或错误描述。
        policy = row.get("policy") or {}
        name = policy.get("name")
        version = str(policy.get("version") or "")
        transitions.append({"state": clean, "action": row["action"], "reward": reward,
                            "next_state": _safe_observation(next_state) if next_state is not None else None, "done": done,
                            "policy": {"name": name if name in {"masked_double_dqn", "deterministic_recovery_baseline"} else "unknown",
                                       "version": version if _FINGERPRINT.fullmatch(version) or re.fullmatch(r"\d+\.\d+", version) else "unknown"}})
    outcome = "execution_failed" if episode["status"] == "failed" else "terminated_failure" if rows[-1]["action"] == "terminate" else (
        "recovered_success" if any(not row["state"]["verification"]["passed"] for row in transitions) else "verified_success")
    return {"schema_version": DATASET_VERSION, "reward_version": REWARD_VERSION, "instance_fingerprint": fingerprint,
            "template_id": transitions[0]["state"]["template_id"], "outcome": outcome, "transitions": transitions}


def _safe_observation(state: dict) -> dict:
    """严格复制数值/布尔字段；文本错误仅保留数量以兼容既有状态编码。"""

    from optiagent.rl.benchmark import TEMPLATE_IDS

    if state.get("template_id") not in TEMPLATE_IDS or state.get("candidate_actions") != list(ACTION_IDS):
        raise ValueError("invalid_contract")
    mask = state.get("action_mask")
    if not isinstance(mask, list) or len(mask) != len(ACTION_IDS) or not all(type(value) is bool for value in mask):
        raise ValueError("invalid_action_mask")
    verification = state["verification"]
    math_report = verification.get("mathematical") or {}
    if type(verification.get("passed")) is not bool:
        raise ValueError("invalid_contract")
    for key in ("verifiable", "feasible", "objective_consistent", "passed"):
        if math_report.get(key) is not None and type(math_report[key]) is not bool:
            raise ValueError("invalid_contract")
    clean = {key: _finite(state[key]) for key in _NUMERIC_KEYS}
    if any(value < 0 for value in clean.values()):
        raise ValueError("invalid_contract")
    for key in ("model_attempt", "solver_attempt", "step_count", "remaining_model_attempts", "remaining_solver_attempts"):
        if not clean[key].is_integer():
            raise ValueError("invalid_contract")
        clean[key] = int(clean[key])
    if clean["remaining_model_attempts"] != max(0, 2 - clean["model_attempt"]) or clean["remaining_solver_attempts"] != max(0, 2 - clean["solver_attempt"]):
        raise ValueError("invalid_contract")
    if clean["model_attempt"] > 2 or clean["solver_attempt"] > 2:
        raise ValueError("invalid_contract")
    last = state.get("last_action")
    if last is not None and last not in ACTION_IDS:
        raise ValueError("invalid_contract")
    clean.update(template_id=state["template_id"], last_action=last, candidate_actions=list(ACTION_IDS), action_mask=mask,
                 features={key: _finite(state["features"][key]) for key in _FEATURE_KEYS},
                 verification={"passed": verification["passed"], "errors": [None] * min(len(verification.get("errors") or []), 5),
                               "mathematical": {**{key: math_report.get(key) for key in ("verifiable", "feasible", "objective_consistent", "passed")},
                                                "violations": [None] * min(len(math_report.get("violations") or []), 5)}})
    return clean


def _finite(value: Any) -> float:
    """拒绝字符串、布尔值和非有限数值，避免静默类型转换污染训练。"""

    if isinstance(value, bool) or not isinstance(value, (int, float)) or not math.isfinite(value):
        raise ValueError("invalid_contract")
    return float(value)


def _write_jsonl(path: Path, rows: list[dict]) -> None:
    """每行输出一个经过白名单检查的对象。"""

    with path.open("x", encoding="utf-8") as stream:
        for row in rows:
            stream.write(json.dumps(row, ensure_ascii=False, sort_keys=True, allow_nan=False) + "\n")


def _file_metadata(path: Path) -> dict:
    """保存文件摘要，便于消费端核验数据版本。"""

    return {"sha256": hashlib.sha256(path.read_bytes()).hexdigest(), "bytes": path.stat().st_size}
