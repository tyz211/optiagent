from __future__ import annotations

import argparse
from copy import deepcopy
import json
from pathlib import Path
from statistics import mean
import sys
import tempfile
from time import perf_counter
from unittest.mock import patch


# 支持直接从项目根目录运行；所有会话写入临时库。
PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))


def evaluate(checkpoint: str, seed: int, instances_per_template: int = 1, *,
             instances: list | None = None, policy_checkpoints: dict[str, str] | None = None,
             retain_trajectories: bool = True, benchmark_profile: str = "feedback_v1",
             selected_scenarios: tuple[str, ...] | None = None, event_callback=None, evidence_callback=None) -> dict:
    """真实主流程求解后注入受控反馈，检查规则与已保存模型的恢复回路。"""

    import api.database as database
    from api.services import agent_workflow as workflow
    from optiagent.recovery_runtime import load_recovery_runtime
    from optiagent.rl.e2e_benchmark import build_smoke_instances
    from optiagent.rl.instances import generate_recovery_instance
    from optiagent.data import SupplyChainData, normalize_data
    import pandas as pd

    if benchmark_profile not in {"feedback_v1", "data_repair_v1"}:
        raise ValueError("未知评测合同。")

    if instances_per_template < 1:
        raise ValueError("每个模板至少需要一个实例。")
    if instances is None:
        instances = []
        for example in build_smoke_instances(seed=seed):
            for index in range(instances_per_template):
                instance = generate_recovery_instance(example.template_id, seed=seed, split="test", index=index)
                # 保留应用可识别的自然语言目标，实际实例内容使用新的确定性生成器。
                instances.append(instance.model_copy(update={"question": example.question}))
    if not instances:
        raise ValueError("评测实例不能为空。")
    policies = policy_checkpoints if policy_checkpoints is not None else {"rule_policy": "", "learned_policy": checkpoint}
    if not policies:
        raise ValueError("至少需要一种评测策略。")
    # 在执行前检查所有模型，避免部分评测完成后才发现检查点不可用。
    checkpoint_hashes = {name: load_recovery_runtime(path).checkpoint_sha256 for name, path in policies.items()}
    scenarios = ("clean", "transient_failure", "tampered_objective", "model_error", "persistent_failure")
    if selected_scenarios is not None:
        if not selected_scenarios or len(set(selected_scenarios)) != len(selected_scenarios) or set(selected_scenarios) - set(scenarios):
            raise ValueError("场景选择必须合法、非空且唯一。")
        scenarios = selected_scenarios
    records = []
    for policy_name, policy_checkpoint in policies.items():
        for instance in instances:
            for scenario in scenarios:
                started = perf_counter()
                # 环境故障标签只由测试驱动器使用，不进入网络状态。
                def solver(state: dict) -> dict:
                    update = workflow._solver_node(state)
                    result = deepcopy(update["result"])
                    if not (result.get("solution_verification") or {}).get("passed"):
                        raise RuntimeError(f"干净参考求解失败：{instance.task_id}")
                    first = int(state.get("solver_attempt", 0)) == 0
                    if scenario == "persistent_failure" or (scenario == "transient_failure" and first):
                        result.update(status="ERROR", objective_value=None, solution_verification=None)
                    elif scenario == "tampered_objective" and first:
                        result["solution_verification"].update(passed=False, objective_consistent=False)
                    elif scenario == "model_error" and int(state.get("model_attempt", 0)) < 2:
                        result["solution_verification"].update(passed=False, feasible=False)
                    update["result"] = result
                    return update

                harness = None
                workers = {"solver": solver}
                if benchmark_profile == "data_repair_v1":
                    from optiagent.rl.data_repair import DataRepairHarness
                    harness = DataRepairHarness(instance, scenario)
                    harness.event_callback = evidence_callback
                    workers = {"modeler": harness.modeler, "solver": harness.solver}
                question = "请进行求解。" + instance.question + "\n数据：\n" + json.dumps(instance.data, ensure_ascii=False)
                with tempfile.TemporaryDirectory() as directory:
                    with patch.object(database, "DB_PATH", Path(directory) / "evaluation.sqlite3"):
                        database.init_db()
                        dataset_id = None
                        if instance.template_id == "facility_location":
                            # 仓库选址主入口消费已登记的三表数据，使用与真实上传相同的规范化合同。
                            data = normalize_data(SupplyChainData(
                                warehouses=pd.DataFrame(instance.data["warehouses"]),
                                customers=pd.DataFrame(instance.data["customers"]),
                                costs=pd.DataFrame(instance.data["costs"]),
                            ))
                            dataset_id = database.save_dataset("评测仓库数据", data)
                        graph = workflow._build_agent_graph(workers)
                        with patch.object(workflow, "AGENT_GRAPH", graph):
                            result = workflow.run_agent_workflow(
                                question=question, requested_dataset_id=dataset_id, mcp_config="", user_id=None,
                                conversation_id=None, recovery_checkpoint=policy_checkpoint,
                                trajectory_source="controlled_workflow_benchmark",
                                event_callback=event_callback,
                            )
                        trajectory = database.get_training_episode(result["agent_episode_id"], user_id=None)
                decisions = result["agent_policy"]["decisions"]
                actions = [item["selected_action"] for item in decisions]
                valid = all(item["action_mask"][item["candidate_ids"].index(item["selected_action"])] for item in decisions)
                records.append({
                    "policy": policy_name, "template_id": instance.template_id, "scenario": scenario, "task_id": instance.task_id,
                    "recoverable": scenario != "persistent_failure",
                    "success": bool(actions and actions[-1] == "accept_solution" and result["workflow_verification"]["passed"]),
                    "actions": actions, "valid_actions": valid,
                    "solver_calls": sum(node["node_id"] == "solver" for node in result["agent_graph"]["nodes"]),
                    "model_calls": sum(node["node_id"] == "modeler" for node in result["agent_graph"]["nodes"]),
                    "fallbacks": sum(bool(item["metadata"].get("fallback_reason")) for item in decisions),
                    "instance_fingerprint": trajectory["task"]["instance_fingerprint"],
                    "benchmark_profile": benchmark_profile,
                    **({"repair_events": harness.events} if harness is not None else {}),
                    "unverified_accept": bool(actions and actions[-1] == "accept_solution" and not result["workflow_verification"]["passed"]),
                    # 成本单位来自运行时估计；墙钟耗时包含建库、图执行和轨迹导出。
                    "estimated_recovery_cost": sum(float(item["metadata"]["observation"]["features"].get(
                        {"retry_solver": "retry_cost_estimate", "rebuild_model": "rebuild_cost_estimate"}.get(item["selected_action"], ""), 0))
                        for item in decisions),
                    "elapsed_seconds": perf_counter() - started,
                    **({"trajectory": trajectory} if retain_trajectories else {}),
                })
    metrics = {}
    for policy_name in policies:
        rows = [row for row in records if row["policy"] == policy_name]
        metrics[policy_name] = {
            "case_count": len(rows),
            "success_rate": mean(row["success"] for row in rows),
            "recoverable_success_rate": mean(row["success"] for row in rows if row["recoverable"]) if any(row["recoverable"] for row in rows) else None,
            "invalid_case_count": sum(not row["valid_actions"] for row in rows),
            "fallback_count": sum(row["fallbacks"] for row in rows),
            "average_solver_calls": mean(row["solver_calls"] for row in rows),
            "average_model_calls": mean(row["model_calls"] for row in rows),
            "unverified_accept_count": sum(row["unverified_accept"] for row in rows),
            "average_estimated_recovery_cost": mean(row["estimated_recovery_cost"] for row in rows),
            "average_elapsed_seconds": mean(row["elapsed_seconds"] for row in rows),
        }
    return {"schema_version": "1.1", "seed": seed, "instance_count": len(instances),
            "checkpoint_sha256": checkpoint_hashes.get("learned_policy"), "checkpoint_hashes": checkpoint_hashes,
            "scope": "真实 LangGraph/Solver + 受控验证反馈；集成回归，不代表未见 OR 实例泛化。",
            "benchmark_profile": benchmark_profile,
            "llm_used": False, "metrics": metrics, "cases": records}


def main() -> None:
    """生成不可覆盖的评测报告，同时保留可导出的真实主流程决策轨迹。"""

    parser = argparse.ArgumentParser(description="评测学习策略接入主 LangGraph 后的恢复行为")
    parser.add_argument("--checkpoint", required=True)
    parser.add_argument("--seed", type=int, default=54)
    parser.add_argument("--instances-per-template", type=int, default=1)
    parser.add_argument("--output", required=True)
    args = parser.parse_args()
    output = Path(args.output)
    output.parent.mkdir(parents=True, exist_ok=True)
    # 执行前占用目标路径，避免完成高成本评测后才发现报告重名。
    with output.open("x", encoding="utf-8") as stream:
        try:
            report = evaluate(args.checkpoint, args.seed, args.instances_per_template)
        except BaseException as exc:
            # 失败也写明状态，不留下看似正常的空报告或敏感异常内容。
            json.dump({"status": "failed", "seed": args.seed, "error_type": type(exc).__name__}, stream)
            raise
        json.dump({"status": "completed", **report}, stream, ensure_ascii=False, indent=2)
    print(json.dumps({"report": str(output), "metrics": report["metrics"]}, ensure_ascii=False))


if __name__ == "__main__":
    main()
