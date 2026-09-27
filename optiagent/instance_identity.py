from __future__ import annotations

import hashlib
import json
import math
from typing import Any


FINGERPRINT_VERSION = "instance-content-v1"


def instance_fingerprint(template_id: str, data: dict[str, Any]) -> str:
    """按模板和实例内容生成指纹；忽略字典顺序、表行顺序与等值数值表示。"""

    if not template_id or not isinstance(data, dict) or not data:
        raise ValueError("实例指纹需要明确模板和非空数据。")
    payload = {"version": FINGERPRINT_VERSION, "template_id": template_id, "data": _canonical(data)}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, ensure_ascii=False, separators=(",", ":"), allow_nan=False).encode()).hexdigest()


def _canonical(value: Any) -> Any:
    """矩阵及标量序列保留顺序，带字段的表行按内容排序；不做变量重命名同构判断。"""

    if isinstance(value, dict):
        if not all(isinstance(key, str) for key in value):
            raise ValueError("实例字段名必须是字符串。")
        return {key: _canonical(item) for key, item in value.items()}
    if isinstance(value, list):
        items = [_canonical(item) for item in value]
        if items and all(isinstance(item, dict) for item in items):
            items.sort(key=lambda item: json.dumps(item, sort_keys=True, ensure_ascii=False))
        return items
    if isinstance(value, float):
        if not math.isfinite(value):
            raise ValueError("实例包含非有限数值。")
        return int(value) if value.is_integer() else value
    if value is None or isinstance(value, (str, bool, int)):
        return value
    raise ValueError(f"实例包含不支持的数据类型：{type(value).__name__}")


def audit_instance_splits(tasks: list[Any]) -> dict[str, Any]:
    """校验指纹与原始数据一致，并拒绝同一实例跨集合出现。"""

    owners: dict[str, str] = {}
    split_groups = {split: set() for split in ("train", "validation", "test")}
    for task in tasks:
        if task.split not in split_groups:
            raise ValueError("未知的数据集划分。")
        fingerprint = getattr(task, "instance_fingerprint", None)
        data = getattr(task, "instance_data", None)
        if not fingerprint or not data:
            raise ValueError("训练任务缺少可验证的实例内容；请重新采集旧版本数据。")
        if fingerprint != instance_fingerprint(task.template_id, data):
            raise ValueError("实例内容与指纹不一致。")
        previous = owners.setdefault(fingerprint, task.split)
        if previous != task.split:
            raise ValueError("发现跨集合重复实例，拒绝训练。")
        split_groups[task.split].add(fingerprint)
    return {"fingerprint_version": FINGERPRINT_VERSION, "unique_instance_count": len(owners),
            "unique_instances_by_split": {key: len(value) for key, value in split_groups.items()},
            "cross_split_overlap_count": 0}
