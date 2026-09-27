from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# 直接执行入口时定位仓库模块，不依赖额外设置 PYTHONPATH。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """导出真实工作流或受控评测轨迹；两类来源始终写入不同数据集。"""

    from optiagent.rl.trajectory_dataset import export_trajectory_dataset, read_database_episodes

    parser = argparse.ArgumentParser(description="按实例隔离并脱敏导出恢复决策 JSONL")
    inputs = parser.add_mutually_exclusive_group(required=True)
    inputs.add_argument("--db", type=Path)
    inputs.add_argument("--evaluation-report", type=Path)
    parser.add_argument("--user-id", type=int, default=None, help="数据库来源的用户范围；默认仅匿名用户")
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()
    if args.db:
        episodes = read_database_episodes(args.db, user_id=args.user_id)
        source = "live_workflow"
    else:
        report = json.loads(args.evaluation_report.read_text(encoding="utf-8"))
        if report.get("status") != "completed":
            parser.error("评测报告必须完整完成。")
        episodes = [row["trajectory"] for row in report["cases"]]
        source = "controlled_workflow_benchmark"
    manifest = export_trajectory_dataset(episodes, args.output_dir, seed=args.seed, source=source)
    print(json.dumps({"output_dir": str(args.output_dir), **manifest}, ensure_ascii=False))


if __name__ == "__main__":
    main()
