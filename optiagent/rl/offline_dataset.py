from __future__ import annotations

from collections import Counter
from dataclasses import dataclass
import hashlib
import json
from pathlib import Path
import re

from optiagent.rl.trajectory_dataset import DATASET_VERSION, REWARD_VERSION, _validated_episode, split_for_instance


@dataclass(frozen=True)
class OfflineDataset:
    """通过文件摘要、状态合同和内容分组校验的固定离线数据集。"""

    splits: dict[str, list[dict]]
    manifest: dict
    manifest_sha256: str


def load_offline_dataset(directory: str | Path) -> OfflineDataset:
    """加载时重新核验文件和逐条合同；任何异常均拒绝整次训练。"""

    root = Path(directory)
    manifest_bytes = (root / "manifest.json").read_bytes()
    manifest = json.loads(manifest_bytes)
    if (manifest.get("schema_version") != DATASET_VERSION or manifest.get("reward_version") != REWARD_VERSION
            or manifest.get("state_encoder_version") != "2.0"
            or manifest.get("split_method") != "instance_sha256_70_15_15"
            or type(manifest.get("split_seed")) is not int
            or manifest.get("trajectory_source") not in {"live_workflow", "controlled_workflow_benchmark"}):
        raise ValueError("离线数据集版本、来源或分组合同不兼容。")
    splits, groups, seen = {}, {}, set()
    outcomes: Counter = Counter()
    transitions = 0
    for name in ("train", "validation", "test", "quarantine"):
        filename = f"{name}.jsonl"
        content = (root / filename).read_bytes()
        metadata = manifest.get("files", {}).get(filename, {})
        if metadata.get("bytes") != len(content) or metadata.get("sha256") != hashlib.sha256(content).hexdigest():
            raise ValueError(f"数据文件摘要不匹配：{filename}")
        rows = [json.loads(line) for line in content.splitlines() if line.strip()]
        if name == "quarantine":
            if len(rows) != manifest.get("rejected_episodes"):
                raise ValueError("隔离记录数量不匹配。")
            continue
        if not rows:
            raise ValueError(f"离线训练要求非空的 {name} 集合。")
        groups[name] = set()
        for row in rows:
            if (row.get("schema_version") != DATASET_VERSION or row.get("reward_version") != REWARD_VERSION
                    or row.get("trajectory_source") != manifest["trajectory_source"] or row.get("split") != name):
                raise ValueError("episode 版本、来源或划分不匹配。")
            reference = row.get("episode_ref")
            if not isinstance(reference, str) or not re.fullmatch(r"[0-9a-f]{64}", reference) or reference in seen:
                raise ValueError("episode 引用非法或重复。")
            seen.add(reference)
            if split_for_instance(row["instance_fingerprint"], manifest["split_seed"]) != name:
                raise ValueError("实例分组与已声明的内容哈希划分不一致。")
            checked = validate_exported_episode(row)
            groups[name].add(row["instance_fingerprint"])
            outcomes[checked["outcome"]] += 1
            transitions += len(checked["transitions"])
        splits[name] = rows
    if any(groups[a] & groups[b] for a, b in (("train", "validation"), ("train", "test"), ("validation", "test"))):
        raise ValueError("存在跨集合重复实例。")
    if (manifest.get("unique_instances_by_split") != {key: len(value) for key, value in groups.items()}
            or manifest.get("accepted_episodes") != len(seen) or manifest.get("transition_count") != transitions
            or manifest.get("outcomes") != dict(outcomes) or manifest.get("cross_split_overlap_count") != 0
            or manifest.get("all_splits_nonempty") is not True):
        raise ValueError("manifest 汇总与实际数据不一致。")
    return OfflineDataset(splits, manifest, hashlib.sha256(manifest_bytes).hexdigest())


def validate_exported_episode(row: dict) -> dict:
    """复用导出端校验规则，还原最小合同后核对脱敏状态与终局分类。"""

    transitions = row.get("transitions")
    if not isinstance(transitions, list) or not transitions:
        raise ValueError("episode 缺少 transition。")
    restored = []
    for item in transitions:
        state = {**item["state"], "instance_fingerprint": row["instance_fingerprint"]}
        next_state = ({**item["next_state"], "instance_fingerprint": row["instance_fingerprint"]}
                      if item["next_state"] is not None else None)
        restored.append({"policy_observation": state, "next_policy_observation": next_state,
                         "candidate_actions": state["candidate_actions"], "action_mask": state["action_mask"],
                         "action": item["action"], "reward": item["reward"], "done": item["done"],
                         "policy": item["policy"], "status": "completed"})
    checked = _validated_episode({"status": "failed" if row.get("outcome") == "execution_failed" else "completed",
                                  "task": {"template_id": row["template_id"], "instance_fingerprint": row["instance_fingerprint"]},
                                  "total_reward": transitions[-1]["reward"], "decision_transitions": restored})
    if checked["transitions"] != transitions or checked["outcome"] != row.get("outcome"):
        raise ValueError("离线状态、策略版本或终局分类与可验证合同不一致。")
    return checked


def transition_reward(episode: dict, transition: dict, mode: str) -> float:
    """显式选择原始稀疏奖励或经验证的终局奖惩减动作成本，不修改源数据。"""

    if mode == "sparse":
        return float(transition["reward"])
    if mode != "verified_cost":
        raise ValueError("未知离线奖励模式。")
    # 模型奖励以完成验证为依据；失败终止统一为负，防止响应格式分掩盖失败。
    terminal = (1.0 if episode["outcome"] in {"verified_success", "recovered_success"} else -1.0) if transition["done"] else 0.0
    key = {"retry_solver": "retry_cost_estimate", "rebuild_model": "rebuild_cost_estimate"}.get(transition["action"])
    cost = float(transition["state"]["features"][key]) if key else 0.0
    if cost < 0:
        raise ValueError("恢复动作成本不能为负数。")
    return terminal - cost
