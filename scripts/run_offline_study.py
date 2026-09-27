from __future__ import annotations

import argparse
from collections import Counter
from concurrent.futures import ProcessPoolExecutor
import hashlib
import json
import multiprocessing
import os
from pathlib import Path
import subprocess
import sys

# 统一编排固定数据、训练与外部评测，所有输出写到新的实验目录。
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from optiagent.instance_identity import instance_fingerprint
from optiagent.rl.e2e_benchmark import EndToEndBenchmarkInstance
from optiagent.rl.offline_dataset import load_offline_dataset
from optiagent.rl.study import audit_dataset_against_plan, build_study_plan, paired_comparison, summarize_study, workflow_instance_data
from optiagent.rl.trajectory_dataset import export_trajectory_dataset
from scripts.evaluate_workflow_policy import evaluate


def write_json(path: Path, value: dict) -> None:
    """拒绝覆盖已有结果，保留每次实验的原始证据。"""

    with path.open("x", encoding="utf-8") as stream:
        json.dump(value, stream, ensure_ascii=False, indent=2, allow_nan=False)


def digest(path: Path) -> str:
    """计算文件内容摘要，校验阶段间输入没有发生变化。"""

    return hashlib.sha256(path.read_bytes()).hexdigest()


def progress(stage: str, **fields) -> None:
    """输出精简进度，详细数据仍写到实验目录。"""

    print(json.dumps({"stage": stage, **fields}, ensure_ascii=False), flush=True)


def source_hashes() -> dict:
    """冻结会影响采集、训练及评测的项目源码，包含未提交的本地改动。"""

    paths = sorted({*ROOT.glob("optiagent/**/*.py"), *ROOT.glob("api/**/*.py"), *ROOT.glob("scripts/*.py")})
    return {str(path.relative_to(ROOT)): digest(path) for path in paths}


def evaluate_batch(arguments: tuple) -> dict:
    """各子进程拥有独立数据库和图补丁，不在线程之间共享可变全局状态。"""

    seed, rows, policies, retain, profile = arguments
    # 限制每个离线求解进程的线程数，避免并行进程争抢全部 CPU。
    os.environ["OPTIAGENT_SOLVER_THREADS"] = "1"
    import torch
    torch.set_num_threads(1)
    return evaluate("", seed, instances=[EndToEndBenchmarkInstance(**row["instance"]) for row in rows],
                    policy_checkpoints=policies, retain_trajectories=retain, benchmark_profile=profile)


def batches(rows: list, seed: int, policies: dict, retain: bool, workers: int, profile: str = "feedback_v1"):
    """按固定顺序返回批结果；进程并行只改变执行耗时，不改变实例归属。"""

    arguments = [(seed, rows[start:start + 10], policies, retain, profile) for start in range(0, len(rows), 10)]
    if workers == 1:
        yield from map(evaluate_batch, arguments)
    else:
        # spawn 避免继承已有求解器或 PyTorch 的线程锁。
        with ProcessPoolExecutor(max_workers=workers, mp_context=multiprocessing.get_context("spawn")) as pool:
            yield from pool.map(evaluate_batch, arguments)


def prepare(args) -> None:
    """冻结内容清单后分批采集；行为模型仅用于生成固定离线数据。"""

    excluded = set()
    behavior = {}
    if args.behavior_checkpoint:
        if not args.behavior_tasks:
            raise ValueError("提供行为模型时必须同时提供来源实例文件，供外部测试去重。")
        tasks = json.loads(args.behavior_tasks.read_text())
        if isinstance(tasks, dict) and tasks.get("schema_version") == "recovery-study-v1":
            # 前轮冻结清单包含其内部、外部及已知来源实例，全部排除而非只按任务编号隔离。
            excluded.update(tasks["excluded_fingerprints"])
            tasks = [{"template_id": row["instance"]["template_id"], "instance_data": row["instance"]["data"],
                      "instance_fingerprint": row["fingerprint"]} for row in tasks["instances"]]
        for task in tasks:
            fingerprint = instance_fingerprint(task["template_id"], task["instance_data"])
            if fingerprint != task["instance_fingerprint"]:
                raise ValueError("行为模型来源数据的指纹与内容不符。")
            excluded.add(fingerprint)
            excluded.add(instance_fingerprint(task["template_id"], workflow_instance_data(task["template_id"], task["instance_data"])))
        behavior = {"checkpoint": str(args.behavior_checkpoint.resolve()), "checkpoint_sha256": digest(args.behavior_checkpoint),
                    "tasks": str(args.behavior_tasks.resolve()), "tasks_sha256": digest(args.behavior_tasks)}
    elif args.behavior_tasks:
        raise ValueError("来源实例文件需要搭配行为模型。")
    if len(set(args.seeds)) != len(args.seeds) or not args.seeds:
        raise ValueError("训练种子必须唯一且非空。")
    plan = build_study_plan(seed=args.instance_seed, split_seed=args.split_seed, train_count=args.train_count,
                            validation_count=args.holdout_count, test_count=args.holdout_count,
                            external_count=args.external_count, excluded=excluded)
    args.output.mkdir(parents=True, exist_ok=False)
    config = {"seeds": args.seeds, "steps": args.steps, "bc_epochs": args.bc_epochs, "behavior": behavior, "workers": args.workers,
              "solver_threads_per_worker": 1, "benchmark_profile": args.benchmark_profile,
              "repair_gate": "five seeds: recovery >= rule, recovery > previous, no safety failures; original cost gate reported separately",
              "source_sha256": source_hashes(), "git_head": subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=ROOT, text=True).strip(),
              "git_dirty": bool(subprocess.check_output(["git", "status", "--porcelain"], cwd=ROOT, text=True)),
              "selection": "validation logged_return_mse; no external test selection"}
    write_json(args.output / "plan.json", plan)
    write_json(args.output / "config.json", config)
    policies = {"rule_policy": ""}
    if behavior:
        policies["behavior_policy"] = behavior["checkpoint"]
    rows = [row for row in plan["instances"] if row["group"] != "external"]
    def episodes():
        # 每批只保留少量原始轨迹，导出端仅保存脱敏的策略状态。
        for index, report in enumerate(batches(rows, args.instance_seed, policies, True, args.workers, args.benchmark_profile)):
            start = index * 10
            batch = rows[start:start + 10]
            expected = {row["instance"]["task_id"]: row["fingerprint"] for row in batch}
            for case in report["cases"]:
                if expected[case["task_id"]] != case["instance_fingerprint"]:
                    raise ValueError("工作流中的实例内容指纹与预生成实例不一致。")
                yield case["trajectory"]
            progress("collect", instances_completed=min(start + 10, len(rows)), total_instances=len(rows))
    manifest = export_trajectory_dataset(episodes(), args.output / "dataset", seed=args.split_seed, source="controlled_workflow_benchmark")
    # 在数据摘要冻结前登记故障合同，防止把两种不同机制的数据混称为同一实验。
    manifest["benchmark_profile"] = args.benchmark_profile
    (args.output / "dataset/manifest.json").write_text(json.dumps(manifest, ensure_ascii=False, indent=2))
    dataset = load_offline_dataset(args.output / "dataset")
    coverage = audit_dataset_against_plan(dataset, plan)
    expected_episodes = len(rows) * 5 * len(policies)
    if manifest["accepted_episodes"] != expected_episodes:
        raise ValueError("采集 episode 数量与冻结配额不一致。")
    write_json(args.output / "prepared.json", {"coverage": coverage, "episodes": expected_episodes,
               "plan_sha256": digest(args.output / "plan.json"), "config_sha256": digest(args.output / "config.json"),
               "dataset_manifest_sha256": dataset.manifest_sha256})
    progress("prepared", episodes=expected_episodes, coverage=coverage)


def verify_inputs(output: Path) -> tuple[dict, dict]:
    """每个后续阶段重新检查冻结配置、源码和数据，不默默重划分或换模型。"""

    prepared = json.loads((output / "prepared.json").read_text())
    for name in ("plan", "config"):
        if digest(output / f"{name}.json") != prepared[f"{name}_sha256"]:
            raise ValueError("冻结的实验计划或配置被修改。")
    config = json.loads((output / "config.json").read_text())
    plan = json.loads((output / "plan.json").read_text())
    if config["source_sha256"] != source_hashes():
        raise ValueError("实验源码发生变化，请创建新的实验目录。")
    dataset = load_offline_dataset(output / "dataset")
    if dataset.manifest_sha256 != prepared["dataset_manifest_sha256"]:
        raise ValueError("阶段间数据集被修改。")
    audit_dataset_against_plan(dataset, plan)
    behavior = config["behavior"]
    if behavior and any(digest(Path(behavior[name])) != behavior[f"{name}_sha256"] for name in ("checkpoint", "tasks")):
        raise ValueError("冻结的行为模型或来源实例文件被修改。")
    return config, plan


def train(output: Path) -> None:
    """逐种子调用已有训练入口，沿用其记录、失败处理与检查点合同。"""

    config, _ = verify_inputs(output)
    checkpoints = {}
    for seed in config["seeds"]:
        command = [sys.executable, str(ROOT / "scripts/train_offline_policy.py"), "--dataset", str(output / "dataset"),
                   "--seed", str(seed), "--steps", str(config["steps"]), "--bc-epochs", str(config["bc_epochs"]),
                   "--output-root", str(output / "runs"), "--run-id", f"seed-{seed}", "--threads", "1"]
        with (output / f"train-{seed}.log").open("x") as stream:
            subprocess.run(command, cwd=ROOT, stdout=stream, stderr=subprocess.STDOUT, check=True)
        directory = output / "runs" / f"seed-{seed}"
        checkpoints[str(seed)] = {name: digest(directory / name) for name in ("recovery_policy.pt", "bc_policy.pt", "cql_final_policy.pt")}
        report = json.loads((directory / "training_report.json").read_text())
        progress("trained", seed=seed, selection=report["training"]["selection"])
    write_json(output / "trained.json", {"checkpoint_sha256": checkpoints})


def evaluate_study(output: Path) -> None:
    """所有模型训练完成后才读取冻结的外部任务结果，逐实例记录配对差异。"""

    config, plan = verify_inputs(output)
    trained = json.loads((output / "trained.json").read_text())
    rows = [row for row in plan["instances"] if row["group"] == "external"]
    expected = {row["fingerprint"] for row in rows}
    internal = {row["fingerprint"] for row in plan["instances"] if row["group"] != "external"}
    if len(expected) != len(rows) or expected & (internal | set(plan["excluded_fingerprints"])):
        raise ValueError("外部实例重复或与已知训练数据重叠。")
    runs = []
    for seed in config["seeds"]:
        directory = output / "runs" / f"seed-{seed}"
        files = {"bc_policy": "bc_policy.pt", "cql_final_policy": "cql_final_policy.pt", "selected_policy": "recovery_policy.pt"}
        if any(digest(directory / name) != value for name, value in trained["checkpoint_sha256"][str(seed)].items()):
            raise ValueError("训练完成后检查点被修改。")
        policies = {"rule_policy": "", **{key: str(directory / value) for key, value in files.items()}}
        if config["benchmark_profile"] == "data_repair_v1" and config["behavior"]:
            policies["previous_policy"] = config["behavior"]["checkpoint"]
        # 不同种子循环轮换执行次序，仍将时间指标限定为诊断，不作收益结论。
        names = list(policies)
        offset = config["seeds"].index(seed) % len(names)
        names = names[offset:] + names[:offset]
        cases, metrics = [], {}
        for name in names:
            policy_cases = []
            for report in batches(rows, plan["seed"], {name: policies[name]}, False, config["workers"], config["benchmark_profile"]):
                policy_cases.extend(report["cases"])
            # 所有批次同样大小，以案例数加权汇总；计数类指标直接相加。
            from statistics import mean
            report = {"cases": policy_cases, "metrics": {name: {
                "case_count": len(policy_cases), "success_rate": mean(case["success"] for case in policy_cases),
                "recoverable_success_rate": mean(case["success"] for case in policy_cases if case["recoverable"]),
                "invalid_case_count": sum(not case["valid_actions"] for case in policy_cases),
                "fallback_count": sum(case["fallbacks"] for case in policy_cases),
                "unverified_accept_count": sum(case["unverified_accept"] for case in policy_cases),
                **{f"average_{metric}": mean(case[metric] for case in policy_cases)
                   for metric in ("solver_calls", "model_calls", "estimated_recovery_cost", "elapsed_seconds")},
            }}}
            if {case["instance_fingerprint"] for case in report["cases"]} != expected:
                raise ValueError("外部执行实例与冻结计划不一致。")
            cases.extend(report["cases"])
            metrics.update(report["metrics"])
            progress("evaluated", seed=seed, policy=name, metrics=metrics[name])
        by_policy = {name: [case for case in cases if case["policy"] == name] for name in policies}
        comparisons = {name: paired_comparison(by_policy[name], by_policy["selected_policy"]) for name in ("rule_policy", "bc_policy")}
        run = {"seed": seed, "metrics": metrics, "paired_selected_vs": comparisons}
        write_json(output / f"evaluation-{seed}.json", {**run, "checkpoint_sha256": trained["checkpoint_sha256"][str(seed)], "cases": cases})
        runs.append(run)
    summary = {"schema_version": "recovery-study-v1", "runs": runs, **summarize_study(runs),
               "benchmark_profile": config["benchmark_profile"], "external_instances": len(rows), "external_template_counts": dict(Counter(row["instance"]["template_id"] for row in rows)),
               "known_instance_overlap": 0, "plan_sha256": digest(output / "plan.json"),
               "dataset_manifest_sha256": digest(output / "dataset/manifest.json"),
               "evaluation_sha256": {str(seed): digest(output / f"evaluation-{seed}.json") for seed in config["seeds"]}}
    if config["benchmark_profile"] == "data_repair_v1" and config["behavior"]:
        # 修复实验先考察正确恢复；保留原成本门槛，不能用新增判据掩盖成本未改善。
        gates = []
        for run in runs:
            selected = run["metrics"]["selected_policy"]
            passed = (selected["recoverable_success_rate"] >= run["metrics"]["rule_policy"]["recoverable_success_rate"]
                      and selected["recoverable_success_rate"] > run["metrics"]["previous_policy"]["recoverable_success_rate"]
                      and not any(selected[key] for key in ("invalid_case_count", "fallback_count", "unverified_accept_count")))
            gates.append({"seed": run["seed"], "passed": passed})
        summary["repair_reliability_gates"] = gates
        summary["repair_reliability_gate_passed"] = len(runs) >= 5 and all(row["passed"] for row in gates)
    write_json(output / "summary.json", summary)
    progress("completed", summary=str(output / "summary.json"), gate=summary["controlled_benchmark_gate_passed"])


def main() -> None:
    """阶段可分别调用；同阶段已有结果时拒绝覆盖。"""

    parser = argparse.ArgumentParser(description="扩大实例覆盖并执行五种子离线恢复策略研究")
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--phase", choices=("all", "prepare", "train", "evaluate"), default="all")
    parser.add_argument("--instance-seed", type=int, default=6100)
    parser.add_argument("--split-seed", type=int, default=61)
    parser.add_argument("--train-count", type=int, default=40)
    parser.add_argument("--holdout-count", type=int, default=20)
    parser.add_argument("--external-count", type=int, default=20)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--steps", type=int, default=1500)
    parser.add_argument("--bc-epochs", type=int, default=160)
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--benchmark-profile", choices=("feedback_v1", "data_repair_v1"), default="feedback_v1")
    parser.add_argument("--behavior-checkpoint", type=Path)
    parser.add_argument("--behavior-tasks", type=Path)
    args = parser.parse_args()
    if args.workers < 1 or args.steps < 0 or args.bc_epochs < 0:
        parser.error("并发进程数必须为正数，训练步数不可为负。")
    args.output = args.output.resolve()
    stage = args.phase
    record_failure = not (args.phase in ("all", "prepare") and args.output.exists())
    try:
        if args.phase in ("all", "prepare"):
            stage = "prepare"
            prepare(args)
        if args.phase in ("all", "train"):
            stage = "train"
            train(args.output)
        if args.phase in ("all", "evaluate"):
            stage = "evaluate"
            evaluate_study(args.output)
    except BaseException as exc:
        # 只记录异常类型和阶段，不把任意原始错误写进实验元数据。
        if record_failure and args.output.exists() and not (args.output / "failure.json").exists() and not (args.output / "summary.json").exists():
            write_json(args.output / "failure.json", {"status": "failed", "stage": stage, "error_type": type(exc).__name__})
        raise


if __name__ == "__main__":
    main()
