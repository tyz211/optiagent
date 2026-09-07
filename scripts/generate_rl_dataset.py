from __future__ import annotations

import argparse
import json
from pathlib import Path

from optiagent.rl import evaluate_policy, export_rollouts_jsonl, generate_benchmark


def main() -> None:
    """生成确定性 policy baseline 的 RL benchmark 轨迹。"""

    parser = argparse.ArgumentParser(description="生成 OptiAgent RL benchmark 轨迹")
    parser.add_argument("--output", default="data/rl/baseline.jsonl", help="JSONL 输出路径")
    parser.add_argument("--seed", type=int, default=42, help="随机种子")
    parser.add_argument("--variants-per-template", type=int, default=12, help="每个模板的任务数")
    parser.add_argument("--split", choices=["train", "validation", "test", "all"], default="all")
    args = parser.parse_args()

    tasks = generate_benchmark(seed=args.seed, variants_per_template=args.variants_per_template)
    if args.split != "all":
        tasks = [item for item in tasks if item.split == args.split]
    result = evaluate_policy(tasks, seed=args.seed)
    output_path = export_rollouts_jsonl(result, Path(args.output))
    # CLI 只输出简短可机器读取的摘要，详细轨迹保存在 JSONL。
    print(json.dumps({"output": str(output_path), **result["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
