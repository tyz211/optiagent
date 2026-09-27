from __future__ import annotations

from collections import Counter, defaultdict
from statistics import mean, stdev

from optiagent.instance_identity import instance_fingerprint
from optiagent.rl.e2e_benchmark import build_smoke_instances
from optiagent.rl.instances import generate_recovery_instance
from optiagent.rl.trajectory_dataset import split_for_instance


SCENARIOS = {"clean", "transient_failure", "tampered_objective", "model_error", "persistent_failure"}


def workflow_instance_data(template_id: str, data: dict) -> dict:
    """选址数据先补齐真实入库默认列，按实际执行内容分组，避免同义输入漏检。"""

    if template_id != "facility_location":
        return data
    import pandas as pd
    from optiagent.data import SupplyChainData, normalize_data

    normalized = normalize_data(SupplyChainData(**{key: pd.DataFrame(data[key]) for key in ("warehouses", "customers", "costs")}))
    return {key: getattr(normalized, key).to_dict(orient="records") for key in ("warehouses", "customers", "costs")}


def build_study_plan(*, seed: int, split_seed: int, train_count: int = 40,
                     validation_count: int = 20, test_count: int = 20,
                     external_count: int = 20, excluded: set[str] | None = None) -> dict:
    """在求解与训练之前按模板/哈希集合收满配额，避免按评测效果筛选实例。"""

    quotas = {"train": train_count, "validation": validation_count, "test": test_count}
    if min(*quotas.values(), external_count) < 1:
        raise ValueError("各集合每模板实例数必须为正数。")
    used = set(excluded or ())
    rows = []
    for example in build_smoke_instances(seed=seed):
        counts: Counter = Counter()
        # 拒绝采样只使用内容哈希，不运行求解器，也不读取策略结果。
        for index in range(1000 * sum(quotas.values())):
            instance = generate_recovery_instance(example.template_id, seed=seed, split="train", index=index)
            instance = instance.model_copy(update={"data": workflow_instance_data(instance.template_id, instance.data)})
            fingerprint = instance_fingerprint(instance.template_id, instance.data)
            split = split_for_instance(fingerprint, split_seed)
            if fingerprint in used or counts[split] >= quotas[split]:
                continue
            used.add(fingerprint)
            counts[split] += 1
            rows.append({"group": split, "fingerprint": fingerprint,
                         "instance": instance.model_copy(update={"question": example.question, "split": split}).model_dump()})
            if all(counts[key] == value for key, value in quotas.items()):
                break
        else:
            raise ValueError("无法在候选上限内满足实例配额。")
        for index in range(1000 * external_count):
            instance = generate_recovery_instance(example.template_id, seed=seed, split="test", index=index)
            instance = instance.model_copy(update={"data": workflow_instance_data(instance.template_id, instance.data)})
            fingerprint = instance_fingerprint(instance.template_id, instance.data)
            if fingerprint in used:
                continue
            used.add(fingerprint)
            counts["external"] += 1
            rows.append({"group": "external", "fingerprint": fingerprint,
                         "instance": instance.model_copy(update={"question": example.question}).model_dump()})
            if counts["external"] == external_count:
                break
        else:
            raise ValueError("无法满足外部测试实例配额。")
    return {"schema_version": "recovery-study-v1", "seed": seed, "split_seed": split_seed,
            "quotas_per_template": {**quotas, "external": external_count}, "instances": rows,
            "excluded_fingerprints": sorted(excluded or ()),
            "sampling": "按固定内容哈希划分后满足模板配额；未经结果筛选，集合比例不再为70/15/15。"}


def audit_dataset_against_plan(dataset, plan: dict) -> dict:
    """检查真实工作流导出的指纹、模板覆盖和预先冻结的集合完全一致。"""

    coverage = {}
    for split in ("train", "validation", "test"):
        expected = {(row["instance"]["template_id"], row["fingerprint"]) for row in plan["instances"] if row["group"] == split}
        actual = {(row["template_id"], row["instance_fingerprint"]) for row in dataset.splits[split]}
        if expected != actual:
            raise ValueError(f"实际数据与冻结计划不一致：{split}")
        coverage[split] = dict(Counter(template for template, _ in actual))
    if dataset.manifest["rejected_episodes"]:
        raise ValueError("实验采集存在被隔离的轨迹，须先解决合同错误。")
    return coverage


def paired_comparison(reference: list[dict], candidate: list[dict]) -> dict:
    """按内容指纹配对，先在实例内平均场景差值，再汇总独立实例。"""

    def index(rows):
        indexed = {}
        for row in rows:
            key = (row["instance_fingerprint"], row["scenario"])
            if key in indexed or row["scenario"] not in SCENARIOS:
                raise ValueError("配对评测包含重复或未知场景。")
            indexed[key] = row
        groups = defaultdict(set)
        for fingerprint, scenario in indexed:
            groups[fingerprint].add(scenario)
        if not groups or any(scenarios != SCENARIOS for scenarios in groups.values()):
            raise ValueError("每个独立实例必须包含完整五种场景。")
        return indexed

    baseline, learned = index(reference), index(candidate)
    if baseline.keys() != learned.keys():
        raise ValueError("配对策略的实例/场景不一致。")
    fields = ("success", "model_calls", "solver_calls", "estimated_recovery_cost", "elapsed_seconds")
    result = {}
    for field in fields:
        groups = defaultdict(list)
        for key in baseline:
            groups[key[0]].append(float(learned[key][field]) - float(baseline[key][field]))
        values = [mean(deltas) for deltas in groups.values()]
        result[field] = {"mean_delta": mean(values), "instance_sample_std": stdev(values) if len(values) > 1 else 0.0,
                         "improved_instances": sum(v > 0 if field == "success" else v < 0 for v in values),
                         "worsened_instances": sum(v < 0 if field == "success" else v > 0 for v in values)}
    return {"instance_count": len(groups), "case_count": len(baseline), "candidate_minus_reference": result}


def summarize_study(runs: list[dict]) -> dict:
    """报告种子间方差；验收不因均值掩盖某次成功率下降或安全回退。"""

    if not runs or len({run["seed"] for run in runs}) != len(runs):
        raise ValueError("需要非空且种子唯一的实验记录。")
    aggregate = {}
    for policy in ("rule_policy", "bc_policy", "cql_final_policy", "selected_policy"):
        aggregate[policy] = {}
        for metric in runs[0]["metrics"][policy]:
            values = [run["metrics"][policy][metric] for run in runs]
            aggregate[policy][metric] = {"mean": mean(values), "seed_sample_std": stdev(values) if len(values) > 1 else 0.0}
    passed = []
    for run in runs:
        metrics = run["metrics"]
        selected, rule, bc = (metrics[name] for name in ("selected_policy", "rule_policy", "bc_policy"))
        safety = not any(selected[key] for key in ("invalid_case_count", "fallback_count", "unverified_accept_count"))
        success = selected["recoverable_success_rate"] >= max(rule["recoverable_success_rate"], bc["recoverable_success_rate"])
        # 成本验收使用调用次数，墙钟时间仅作环境相关诊断；两类调用均不得增加。
        cheaper = all(selected["average_solver_calls"] <= other["average_solver_calls"]
                      and selected["average_model_calls"] < other["average_model_calls"] for other in (rule, bc))
        passed.append({"seed": run["seed"], "safety_passed": safety, "success_passed": success,
                       "call_cost_improved_vs_rule_and_bc": cheaper, "passed": safety and success and cheaper})
    return {"aggregate": aggregate, "per_seed_gates": passed,
            "controlled_benchmark_gate_passed": len(runs) >= 5 and all(row["passed"] for row in passed),
            "production_ready": False,
            "limitations": ["真实求解与受控反馈，不代表真实线上故障。", "所有训练种子共享测试实例，种子间方差不是独立任务置信区间。",
                            "按实例汇总配对差异，不把五种场景当作五个独立实例。", "墙钟时间受运行顺序、机器负载与求解器影响，不用于验收。"]}
