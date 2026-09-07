from __future__ import annotations

import random
from typing import Literal

from pydantic import BaseModel, Field


BenchmarkSplit = Literal["train", "validation", "test"]
FaultScenario = Literal[
    "direct_success",
    "transient_solver_failure",
    "repairable_model_error",
    "persistent_failure",
]

TEMPLATE_IDS = (
    "knapsack",
    "assignment",
    "tsp",
    "job_shop_scheduling",
    "production_mix",
    "facility_location",
)
FAULT_SCENARIOS: tuple[FaultScenario, ...] = (
    "direct_success",
    "transient_solver_failure",
    "repairable_model_error",
    "persistent_failure",
)


class BenchmarkTask(BaseModel):
    """用于高层恢复策略的可复现微型 benchmark 任务。"""

    schema_version: str = "1.0"
    task_id: str
    template_id: str
    split: BenchmarkSplit
    scenario: FaultScenario
    difficulty: float = Field(ge=0.0, le=1.0)
    features: dict[str, float]


def generate_benchmark(*, seed: int = 42, variants_per_template: int = 12) -> list[BenchmarkTask]:
    """
    生成按模板分层且划分固定的 policy benchmark。

    这一版专注于恢复决策，不声称替代真实求解器 benchmark。
    """

    if variants_per_template < 3:
        raise ValueError("variants_per_template 至少为 3，以保留 train/validation/test 划分。")
    randomizer = random.Random(seed)
    tasks: list[BenchmarkTask] = []
    for template_id in TEMPLATE_IDS:
        scenarios = [FAULT_SCENARIOS[index % len(FAULT_SCENARIOS)] for index in range(variants_per_template)]
        randomizer.shuffle(scenarios)
        for variant, scenario in enumerate(scenarios):
            difficulty = round(randomizer.uniform(0.15, 0.95), 4)
            tasks.append(
                BenchmarkTask(
                    task_id=f"{template_id}-{variant:03d}-s{seed}",
                    template_id=template_id,
                    split=_split_for_variant(variant, variants_per_template),
                    scenario=scenario,
                    difficulty=difficulty,
                    features=_task_features(template_id, difficulty, randomizer),
                )
            )
    return tasks


def _split_for_variant(variant: int, total: int) -> BenchmarkSplit:
    """每个问题模板内独立分层，避免某个 split 缺失模板。"""

    train_end = min(total - 2, max(1, int(total * 0.7)))
    validation_end = min(total - 1, max(train_end + 1, int(total * 0.85)))
    if variant < train_end:
        return "train"
    if variant < validation_end:
        return "validation"
    return "test"


def _task_features(template_id: str, difficulty: float, randomizer: random.Random) -> dict[str, float]:
    """生成不泄露场景答案的通用数值特征。"""

    base_sizes = {
        "knapsack": 40,
        "assignment": 25,
        "tsp": 18,
        "job_shop_scheduling": 32,
        "production_mix": 20,
        "facility_location": 28,
    }
    base = base_sizes[template_id]
    return {
        "instance_size": float(max(2, round(base * (0.5 + difficulty)))),
        "data_density": round(randomizer.uniform(0.55, 1.0), 4),
        "difficulty": difficulty,
    }
