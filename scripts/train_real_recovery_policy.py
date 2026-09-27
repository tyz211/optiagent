from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys


# 允许从项目根目录直接运行本训练入口。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """收集真实 Gateway 轨迹，训练并记录成本感知 Recovery Policy。"""

    try:
        from optiagent.rl.dqn import DQNConfig
        from optiagent.rl.experiment_log import TrainingRunRecorder
        from optiagent.rl.real_environment import REAL_ENVIRONMENT_VERSION, collect_real_recovery_tasks
        from optiagent.rl.real_training import train_real_recovery_policy
    except ModuleNotFoundError as exc:
        if exc.name == "torch":
            raise SystemExit("未安装 PyTorch，请先执行 pip install -r requirements-rl.txt。") from exc
        raise

    parser = argparse.ArgumentParser(description="训练真实 Gateway 成本感知 Recovery Policy")
    parser.add_argument("--seed", type=int, default=47)
    parser.add_argument("--episodes", type=int, default=1200)
    parser.add_argument("--bc-epochs", type=int, default=160)
    parser.add_argument("--time-limit", type=int, default=10)
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--validation-interval", type=int, default=100)
    parser.add_argument("--instances-per-split", type=int, default=4)
    parser.add_argument("--output-root", default="artifacts/rl/runs")
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("--threads 必须为正整数。")
    import torch

    # 小型网络用单线程避免线程调度开销，同时明确记录运行设置。
    torch.set_num_threads(args.threads)

    config = DQNConfig(
        seed=args.seed,
        train_episodes=args.episodes,
        bc_epochs=args.bc_epochs,
        epsilon_decay_steps=max(800, args.episodes),
        validation_interval=args.validation_interval,
    )
    recorder = TrainingRunRecorder(
        args.output_root,
        seed=args.seed,
        algorithm="behavior_cloning+action_masked_double_dqn+real_gateway_cost",
        parameters={
            "seed": args.seed,
            "episodes": args.episodes,
            "bc_epochs": args.bc_epochs,
            "time_limit": args.time_limit,
            "threads": args.threads,
            "validation_interval": args.validation_interval,
            "instances_per_split": args.instances_per_split,
            "environment_version": REAL_ENVIRONMENT_VERSION,
            "llm_used_for_training": False,
        },
        run_id=args.run_id,
    )
    task_path = recorder.paths.run_directory / "real_gateway_tasks.json"
    try:
        tasks = collect_real_recovery_tasks(seed=args.seed, time_limit=args.time_limit, instances_per_split=args.instances_per_split)
        task_path.write_text(
            json.dumps([item.model_dump(mode="json") for item in tasks], ensure_ascii=False, indent=2),
            encoding="utf-8",
        )
        training = train_real_recovery_policy(tasks, config)
        checkpoint_path = training.agent.save(
            recorder.paths.checkpoint,
            metadata={
                "run_id": recorder.paths.run_id,
                "algorithm": training.report["algorithm"],
                "environment_version": REAL_ENVIRONMENT_VERSION,
                "evaluation": training.report["evaluation"],
                "selection": training.report["training"]["selection"],
            },
        )
        report_path = recorder.paths.report
        report_path.write_text(json.dumps(training.report, ensure_ascii=False, indent=2), encoding="utf-8")
        comparisons = training.save_comparisons(recorder.paths.run_directory, run_id=recorder.paths.run_id)
        record = recorder.complete(training.report, extra_artifacts={"real_gateway_tasks": task_path, **comparisons})
    except BaseException as exc:
        recorder.fail(exc)
        raise

    print(
        json.dumps(
            {
                "run_id": recorder.paths.run_id,
                "manifest": str(recorder.paths.manifest),
                "checkpoint": str(checkpoint_path),
                "report": str(report_path),
                "dataset": training.report["dataset"],
                "test_metrics": record["metrics"]["test"]["learned_policy"],
            },
            ensure_ascii=False,
        )
    )


if __name__ == "__main__":
    main()
