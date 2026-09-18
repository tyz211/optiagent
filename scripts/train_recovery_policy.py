"""Run a reproducible local RL experiment; never updates the production policy."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import random
from statistics import mean, stdev
import sys

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from optiagent.rl import OptimizationAgentEnv, evaluate_policy, export_rollouts_jsonl, generate_benchmark
from optiagent.rl.environment import ACTION_IDS
from optiagent.rl.q_learning import QLearningPolicy, allowed_actions


def write_json(path, value):
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding="utf-8")


def train(train_tasks, validation_tasks, *, seed, episodes, interval, output, resume=None):
    if episodes < 1 or interval < 1:
        raise ValueError("Positive episode and interval counts required")
    if not train_tasks or any(t.split != "train" for t in train_tasks):
        raise ValueError("Only training tasks may update Q values")
    if not validation_tasks or any(t.split != "validation" for t in validation_tasks):
        raise ValueError("Checkpoint selection requires a validation split")
    if {t.task_id for t in train_tasks} & {t.task_id for t in validation_tasks}:
        raise ValueError("Overlapping train and validation IDs")
    parent = json.loads(Path(resume).read_text(encoding="utf-8")) if resume else None
    policy = QLearningPolicy.load(resume) if resume else QLearningPolicy()
    parent_episodes = int(parent["metadata"]["episode"]) if parent else 0
    initial_updates = policy.updates
    lineage = {"parent_checkpoint": str(Path(resume).resolve()) if resume else None,
               "parent_sha256": hashlib.sha256(Path(resume).read_bytes()).hexdigest() if resume else None,
               "parent_episodes": parent_episodes, "parent_updates": initial_updates,
               "resume_mode": "Q-table warm start; new seeded RNG and exploration schedule" if resume else "from scratch"}
    output.mkdir(parents=True, exist_ok=False)
    # Old checkpoints did not store RNG state. Continue Q values, not the exact random stream.
    stream_seed = seed + parent_episodes
    env = OptimizationAgentEnv(train_tasks, seed=stream_seed)
    rng = random.Random(stream_seed)
    best_return = -float("inf")
    selected_episode = 0
    history = []
    rewards = []
    if resume:
        initial_metrics = evaluate_policy(validation_tasks, policy)["metrics"]
        best_return = initial_metrics["average_return"]
        policy.save(output / "best_policy.json", metadata={
            **lineage, "seed": seed, "episode": parent_episodes,
            "selection": "validation average_return", "environment": "synthetic recovery v1",
            "production_ready": False,
        })
        history.append({"episode": 0, "cumulative_episode": parent_episodes,
                        "validation": initial_metrics, "stage": "before_continuation"})
    for episode in range(1, episodes + 1):
        start_epsilon = 0.2 if resume else 1.0
        epsilon = start_epsilon - (start_epsilon - 0.05) * min(1.0, (episode - 1) / max(1, episodes * 0.8))
        state, _ = env.reset()
        done = False
        total = 0.0
        while not done:
            action = policy.explore(state, epsilon, rng)
            next_state, reward, terminated, truncated, _ = env.step(action)
            # v1 truncation is a penalized task budget exhaustion, not a time-limit wrapper.
            done = terminated or truncated
            policy.update(state, action, reward, next_state, done)
            total += reward
            state = next_state
        rewards.append(total)
        if episode % interval == 0 or episode == episodes:
            metrics = evaluate_policy(validation_tasks, policy)["metrics"]
            history.append({"episode": episode, "cumulative_episode": parent_episodes + episode, "epsilon": epsilon,
                            "exploration_return": mean(rewards), "validation": metrics})
            rewards.clear()
            # On equal validation return retain the latest trained checkpoint.
            if metrics["average_return"] >= best_return:
                best_return = metrics["average_return"]
                selected_episode = episode
                policy.save(output / "best_policy.json", metadata={
                    **lineage, "seed": seed, "episode": parent_episodes + episode, "selection": "validation average_return; latest on ties",
                    "environment": "synthetic recovery v1", "production_ready": False,
                })
    policy.save(output / "last_policy.json", metadata={
        **lineage, "seed": seed, "episode": parent_episodes + episodes,
        "environment": "synthetic recovery v1", "production_ready": False,
    })
    write_json(output / "continuation.json", {
        **lineage, "additional_episodes": episodes, "cumulative_episodes": parent_episodes + episodes,
        "additional_updates": policy.updates - initial_updates, "cumulative_updates": policy.updates,
        "stream_seed": stream_seed,
    })
    write_json(output / "learning_curve.json", history)
    return QLearningPolicy.load(output / "best_policy.json"), selected_episode, policy.updates


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, default=ROOT / "artifacts/rl/recovery-qlearning-v1")
    parser.add_argument("--episodes", type=int, default=5000)
    parser.add_argument("--eval-every", type=int, default=250)
    parser.add_argument("--seeds", type=int, nargs="+", default=[11, 22, 33, 44, 55])
    parser.add_argument("--resume-from", type=Path, help="Previous run directory; load each seed's last_policy.json")
    parser.add_argument("--benchmark-seed", type=int, default=42)
    parser.add_argument("--test-seed-start", type=int, default=1000)
    args = parser.parse_args()
    if args.episodes < 1 or args.eval_every < 1 or len(set(args.seeds)) != len(args.seeds):
        parser.error("Positive episode/interval counts and distinct seeds required")
    test_seeds = list(range(args.test_seed_start, args.test_seed_start + 10))
    if args.benchmark_seed in test_seeds:
        parser.error("Benchmark seed must be disjoint from fresh test seeds")
    if args.resume_from:
        parent_manifest = json.loads((args.resume_from / "manifest.json").read_text(encoding="utf-8"))
        for name in ("optiagent/agent_policy.py", "optiagent/rl/environment.py", "optiagent/rl/benchmark.py", "optiagent/rl/q_learning.py"):
            actual = hashlib.sha256((ROOT / name).read_bytes()).hexdigest()
            if parent_manifest["source_sha256"].get(name) != actual:
                parser.error(f"Resume environment/encoder differs from parent: {name}")
        for seed in args.seeds:
            checkpoint = args.resume_from / f"seed-{seed}" / "last_policy.json"
            QLearningPolicy.load(checkpoint)
            metadata = json.loads(checkpoint.read_text(encoding="utf-8"))["metadata"]
            if metadata.get("seed") != seed or int(metadata.get("episode", -1)) < 0:
                parser.error(f"Invalid parent metadata: {checkpoint}")
    # Refuse overwrites so a new experiment cannot silently erase an earlier result.
    args.output.mkdir(parents=True, exist_ok=False)
    tasks = generate_benchmark(seed=args.benchmark_seed)
    splits = {name: [t for t in tasks if t.split == name] for name in ("train", "validation", "test")}
    # Fresh task IDs and features, but the SAME four synthetic transition mechanisms.
    holdout = [task.model_copy(update={"split": "test"})
               for seed in test_seeds for task in generate_benchmark(seed=seed)]
    assert not {t.task_id for t in tasks} & {t.task_id for t in holdout}
    write_json(args.output / "tasks.json", {
        **{name: [t.model_dump() for t in group] for name, group in splits.items()},
        "fresh_seed_test": [t.model_dump() for t in holdout],
    })
    sources = [ROOT / "optiagent/agent_policy.py", *sorted((ROOT / "optiagent/rl").glob("*.py")), Path(__file__)]
    write_json(args.output / "manifest.json", {
        "created_utc": datetime.now(timezone.utc).isoformat(), "python": sys.version,
        "algorithm": "masked tabular Q-learning", "alpha": 0.15, "gamma": 1.0,
        "episodes_per_seed": args.episodes, "training_seeds": args.seeds,
        "epsilon": f"linear {0.2 if args.resume_from else 1.0} -> 0.05 during first 80% of additional episodes",
        "benchmark_seed": args.benchmark_seed, "fresh_test_seeds": test_seeds,
        "resume_from": str(args.resume_from.resolve()) if args.resume_from else None,
        "base_github_commit": "8f2f19d07b444e70752abd5e0ac545983a9088fd",
        "split_counts": {name: len(group) for name, group in splits.items()},
        "scenario_counts": {name: dict(Counter(t.scenario for t in group)) for name, group in splits.items()},
        "source_sha256": {str(p.relative_to(ROOT)).replace('\\', '/'): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources},
        "limitations": ["Synthetic transitions; no real MCP or solver execution.",
                        "Task-ID disjointness does not establish out-of-distribution generalization.",
                        "The legal-action mask comes from existing handcrafted rules.",
                        "No LLM weights or production policies are changed."],
    })
    runs = []
    for seed in args.seeds:
        directory = args.output / f"seed-{seed}"
        resume = args.resume_from / f"seed-{seed}" / "last_policy.json" if args.resume_from else None
        learned, selected_episode, updates = train(
            splits["train"], splits["validation"], seed=seed, episodes=args.episodes,
            interval=args.eval_every, output=directory, resume=resume)
        evaluations = {}
        for name, group in (("test", splits["test"]), ("fresh_seed_test", holdout)):
            rng = random.Random(seed)
            def random_policy(observation):
                return ACTION_IDS[rng.choice(allowed_actions(observation))]
            result = evaluate_policy(group, learned)
            export_rollouts_jsonl(result, directory / f"{name}_rollouts.jsonl")
            evaluations[name] = {
                "learned": result["metrics"],
                "untrained": evaluate_policy(group, QLearningPolicy())["metrics"],
                "random_legal": evaluate_policy(group, random_policy)["metrics"],
                "rule": evaluate_policy(group)["metrics"],
            }
            if resume:
                evaluations[name]["parent"] = evaluate_policy(group, QLearningPolicy.load(resume))["metrics"]
        continuation = json.loads((directory / "continuation.json").read_text(encoding="utf-8"))
        run = {"seed": seed, "selected_episode": selected_episode, "updates": updates,
               "continuation": continuation, "evaluations": evaluations}
        write_json(directory / "evaluation.json", run)
        runs.append(run)
        print(json.dumps({"seed": seed, "updates": updates, **evaluations["fresh_seed_test"]["learned"]}), flush=True)
    aggregate = {}
    policy_labels = [("untrained", "未训练贪心"), ("random_legal", "合法动作随机")]
    if args.resume_from:
        policy_labels.append(("parent", "续训前检查点"))
    policy_labels.extend([("learned", "训练后 Q-learning"), ("rule", "原规则")])
    for policy_name, _ in policy_labels:
        aggregate[policy_name] = {}
        for metric in runs[0]["evaluations"]["fresh_seed_test"][policy_name]:
            values = [r["evaluations"]["fresh_seed_test"][policy_name][metric] for r in runs]
            aggregate[policy_name][metric] = {"mean": mean(values), "sample_std": stdev(values) if len(values) > 1 else 0.0}
    write_json(args.output / "summary.json", {"runs": runs, "fresh_seed_test_aggregate": aggregate})
    lines = ["# 恢复策略强化学习训练报告", "",
             f"本轮完成 {len(runs)} 个随机种子，每个新增 {args.episodes} 个 episode。算法：表格型 Q-learning。",
             f"新增 Q 更新：{sum(r['continuation']['additional_updates'] for r in runs)}；累计 Q 更新：{sum(r['updates'] for r in runs)}。",
             f"每个种子的累计 episode：{[r['continuation']['cumulative_episodes'] for r in runs]}。",
             f"训练 / 验证 / 测试任务数：{len(splits['train'])} / {len(splits['validation'])} / {len(splits['test'])}。",
             "仅训练集更新 Q 值，仅验证集选择检查点。另在 720 个新种子模拟任务上冻结评测。", "",
             "| 策略 | 成功率（均值 ± 标准差） | 可恢复成功率 | 平均回报 | 非法动作率 |",
             "| --- | --- | --- | --- | --- |"]
    for name, label in policy_labels:
        m = aggregate[name]
        lines.append(f"| {label} | {m['success_rate']['mean']:.2%} ± {m['success_rate']['sample_std']:.2%} | {m['recoverable_success_rate']['mean']:.2%} | {m['average_return']['mean']:.4f} | {m['invalid_action_rate']['mean']:.2%} |")
    lines += ["", "边界：以上为预设故障转移的模拟实验，不调用真实 MCP / 求解器。",
              "续训加载旧 Q 表和更新计数；旧检查点未保存 RNG，因此启动新的确定性随机流和探索调度，不声称精确恢复中断位置。",
              "新种子只改变任务编号、场景排列和特征，不能证明跨真实任务的泛化。",
              "规则策略已达到当前环境上限，训练收益应与未训练策略比较，不能宣称超过规则或提升真实业务能力。",
              "合法动作由原规则提供 mask；零非法动作主要归因于该约束。",
              "大模型权重与线上工作流未修改。后续需要真实修复动作、真实执行反馈和更有区分力的基准。", "",
              "检查点：每个 seed 目录的 best_policy.json（按验证集选择）与 last_policy.json。",
              "复现记录：manifest.json、tasks.json、learning_curve.json、evaluation.json 和测试轨迹。"]
    (args.output / "REPORT.md").write_text("\n".join(lines) + "\n", encoding="utf-8")


if __name__ == "__main__":
    main()
