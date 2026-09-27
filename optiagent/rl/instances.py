from __future__ import annotations

import hashlib
import random

from optiagent.rl.e2e_benchmark import EndToEndBenchmarkInstance


def generate_recovery_instance(template_id: str, *, seed: int, split: str, index: int) -> EndToEndBenchmarkInstance:
    """按稳定坐标生成小规模可行实例，改变规模、成本和约束，避免烟雾样本循环复用。"""

    identity = f"recovery-instances-v2:{seed}:{split}:{template_id}:{index}"
    rng = random.Random(int.from_bytes(hashlib.sha256(identity.encode()).digest(), "big"))
    n = rng.randint(3, 7)
    if template_id == "knapsack":
        items = [{"item": f"I{i}", "value": rng.randint(5, 90), "weight": rng.randint(1, 15)} for i in range(n)]
        data = {"items": items, "capacity": max(max(row["weight"] for row in items), sum(row["weight"] for row in items) // 2)}
    elif template_id == "assignment":
        resources, tasks = [f"R{i}" for i in range(n)], [f"T{i}" for i in range(n)]
        data = {"resources": resources, "tasks": tasks,
                "costs": [{"resource": r, "task": t, "cost": rng.randint(1, 99)} for r in resources for t in tasks]}
    elif template_id == "tsp":
        matrix = [[0] * n for _ in range(n)]
        for i in range(n):
            for j in range(i):
                matrix[i][j] = matrix[j][i] = rng.randint(1, 99)
        data = {"nodes": [f"N{i}" for i in range(n)], "distance_matrix": matrix}
    elif template_id == "job_shop_scheduling":
        machines = [f"M{i}" for i in range(rng.randint(2, 4))]
        rows = []
        for job in range(n):
            order = rng.sample(machines, len(machines))
            rows.extend({"job": f"J{job}", "machine": machine, "duration": rng.randint(1, 15), "order": step}
                        for step, machine in enumerate(order, 1))
        data = {"tasks": rows}
    elif template_id == "production_mix":
        products = [{"product": f"P{i}", "profit": rng.randint(5, 90), "labor": rng.randint(1, 8),
                     "machine": rng.randint(1, 8), "max_qty": rng.randint(3, 12)} for i in range(n)]
        data = {"products": products, "capacities": {"labor": rng.randint(20, 70), "machine": rng.randint(20, 70)}, "integer": True}
    elif template_id == "facility_location":
        customers = [{"customer": f"C{i}", "demand": rng.randint(2, 12)} for i in range(n)]
        demand = sum(row["demand"] for row in customers)
        warehouses = [{"warehouse": f"W{i}", "region": "测试区域", "capacity": demand,
                       "fixed_cost": rng.randint(10, 99)} for i in range(rng.randint(2, 4))]
        data = {"warehouses": warehouses, "customers": customers,
                "costs": [{"warehouse": w["warehouse"], "customer": c["customer"], "cost": rng.randint(1, 20)}
                          for w in warehouses for c in customers]}
    else:
        raise ValueError(f"未知实例模板：{template_id}")
    return EndToEndBenchmarkInstance(task_id=identity, template_id=template_id, split=split,
                                    question=f"请对 {template_id} 实例进行优化求解。", data=data)
