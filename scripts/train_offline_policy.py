from __future__ import annotations

from dataclasses import asdict
import argparse
import json
from pathlib import Path
import sys


# 允许从项目根目录直接执行离线训练入口。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def main() -> None:
    """读取固定数据集完成 BC/CQL 对照，记录模型、输入摘要和奖励版本。"""

    from optiagent.rl.experiment_log import TrainingRunRecorder
    from optiagent.rl.offline_dataset import load_offline_dataset
    from optiagent.rl.offline_training import OfflineConfig, train_offline_policy
    import torch

    parser = argparse.ArgumentParser(description="训练离线 BC + Masked CQL 恢复策略")
    parser.add_argument("--dataset", type=Path, required=True)
    parser.add_argument("--seed", type=int, default=57)
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--bc-epochs", type=int, default=160)
    parser.add_argument("--cql-alpha", type=float, default=0.1)
    parser.add_argument("--reward-mode", choices=("sparse", "verified_cost"), default="verified_cost")
    parser.add_argument("--threads", type=int, default=1)
    parser.add_argument("--output-root", type=Path, default=Path("artifacts/rl/offline_runs"))
    parser.add_argument("--run-id", default=None)
    args = parser.parse_args()
    if args.threads < 1:
        parser.error("线程数必须为正数。")
    torch.set_num_threads(args.threads)
    config = OfflineConfig(seed=args.seed, gradient_steps=args.steps, bc_epochs=args.bc_epochs,
                           cql_alpha=args.cql_alpha, reward_mode=args.reward_mode)
    recorder = TrainingRunRecorder(args.output_root, seed=args.seed, algorithm="behavior_cloning+masked_discrete_cql_double_dqn",
                                   parameters={**asdict(config), "threads": args.threads, "llm_used_for_training": False}, run_id=args.run_id)
    try:
        dataset = load_offline_dataset(args.dataset)
        training = train_offline_policy(dataset, config)
        metadata = {"run_id": recorder.paths.run_id, "algorithm": training.report["algorithm"],
                    "selection": training.report["training"]["selection"], "reward": training.report["reward"],
                    "dataset_manifest_sha256": dataset.manifest_sha256, "requires_runtime_evaluation": True}
        training.agent.save(recorder.paths.checkpoint, metadata=metadata)
        bc_path = training.bc_agent.save(recorder.paths.run_directory / "bc_policy.pt", metadata={**metadata, "selection": {"selected_stage": "bc"}})
        final_path = training.final_agent.save(recorder.paths.run_directory / "cql_final_policy.pt", metadata={**metadata, "selection": {"selected_stage": "offline_cql"}})
        manifest_copy = recorder.paths.run_directory / "dataset_manifest.json"
        manifest_copy.write_text(json.dumps(dataset.manifest, ensure_ascii=False, indent=2), encoding="utf-8")
        recorder.paths.report.write_text(json.dumps(training.report, ensure_ascii=False, indent=2), encoding="utf-8")
        recorder.complete(training.report, extra_artifacts={"bc_checkpoint": bc_path, "cql_final_checkpoint": final_path,
                                                          "dataset_manifest": manifest_copy})
    except BaseException as exc:
        recorder.fail(exc)
        raise
    print(json.dumps({"run_id": recorder.paths.run_id, "checkpoint": str(recorder.paths.checkpoint),
                      "report": str(recorder.paths.report), "selection": training.report["training"]["selection"],
                      "test_diagnostics": training.report["evaluation"]["learned_policy"]["test"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
