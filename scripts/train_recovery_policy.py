from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# 直接执行 scripts 下的文件时，将项目根目录加入模块搜索路径。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """训练并保存 OptiAgent 的第一个可学习 Recovery Policy。"""

    try:
        from optiagent.rl.benchmark import generate_benchmark
        from optiagent.rl.dqn import DQNConfig
        from optiagent.rl.experiment_log import TrainingRunRecorder
        from optiagent.rl.training import train_recovery_policy
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit("未安装 PyTorch，请先执行 pip install -r requirements-rl.txt。") from exc
        raise

    parser = argparse.ArgumentParser(description="训练 OptiAgent Masked Double DQN Recovery Policy")
    parser.add_argument("--seed", type=int, default=42)
    parser.add_argument("--episodes", type=int, default=800)
    parser.add_argument("--bc-epochs", type=int, default=120)
    parser.add_argument("--variants-per-template", type=int, default=12)
    parser.add_argument("--output-root", default="artifacts/rl/runs")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()

    config = DQNConfig(seed=args.seed, train_episodes=args.episodes, bc_epochs=args.bc_epochs)
    recorder = TrainingRunRecorder(
        args.output_root,
        seed=args.seed,
        algorithm="behavior_cloning+action_masked_double_dqn",
        parameters={
            "seed": args.seed,
            "episodes": args.episodes,
            "bc_epochs": args.bc_epochs,
            "variants_per_template": args.variants_per_template,
            "llm_used_for_training": False,
        },
        run_id=args.run_id,
    )
    try:
        tasks = generate_benchmark(seed=args.seed, variants_per_template=args.variants_per_template)
        training = train_recovery_policy(tasks, config)
        checkpoint_path = training.agent.save(
            recorder.paths.checkpoint,
            metadata={
                "run_id": recorder.paths.run_id,
                "algorithm": training.report["algorithm"],
                "evaluation": training.report["evaluation"],
                "selection": training.report["training"]["selection"],
            },
        )
        report_path = recorder.paths.report
        report_path.write_text(json.dumps(training.report, ensure_ascii=False, indent=2), encoding="utf-8")
        comparisons = training.save_comparisons(recorder.paths.run_directory, run_id=recorder.paths.run_id)
        record = recorder.complete(training.report, extra_artifacts=comparisons)
    except BaseException as exc:
        recorder.fail(exc)
        raise

    test_metrics = record["metrics"]["test"]["learned_policy"]
    print(
        json.dumps(
            {
                "run_id": recorder.paths.run_id,
                "manifest": str(recorder.paths.manifest),
                "checkpoint": str(checkpoint_path),
                "report": str(report_path),
                "test_metrics": test_metrics,
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
