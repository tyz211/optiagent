from __future__ import annotations

from copy import deepcopy

from optiagent.instance_identity import instance_fingerprint
from optiagent.optimization_gateway import build_problem_envelope, solve_problem_envelope
from optiagent.solution_verifier import verify_solution


PROFILE = "data_repair_v1"


def corrupt_mapping(template_id: str, data: dict) -> dict:
    """模拟单位/系数映射错误，真正改变求解输入；原始权威数据保持不变。"""

    altered = deepcopy(data)
    fields = {
        "knapsack": ("items", "value"), "assignment": ("costs", "cost"),
        "job_shop_scheduling": ("tasks", "duration"), "production_mix": ("products", "profit"),
        "facility_location": ("costs", "cost"),
    }
    if template_id == "tsp":
        altered["distance_matrix"] = [[2 * value for value in row] for row in altered["distance_matrix"]]
    elif template_id in fields:
        table, field = fields[template_id]
        for row in altered[table]:
            row[field] *= 2
        if template_id == "facility_location":
            for row in altered["warehouses"]:
                row["fixed_cost"] *= 2
    else:
        raise ValueError("不支持的数据修复模板。")
    return altered


class DataRepairHarness:
    """隔离评测中的确定性映射修复器；场景标签不进入策略状态或参数更新。"""

    def __init__(self, instance, scenario: str):
        if scenario not in {"clean", "transient_failure", "tampered_objective", "model_error", "persistent_failure"}:
            raise ValueError("未知修复场景。")
        self.instance = instance
        self.scenario = scenario
        self.source = deepcopy(instance.data)
        self.corrupted = corrupt_mapping(instance.template_id, self.source)
        self.bound = deepcopy(self.source)
        self.events = []
        self.event_callback = None
        self.original = build_problem_envelope(instance.template_id, self.source, question=instance.question)

    def fingerprint(self, data: dict) -> str:
        """记录输入实际变化，用于证明重建动作确实替换了映射。"""
        return instance_fingerprint(self.instance.template_id, data)

    def _record(self, event: dict) -> None:
        """证据同时保留在结果中，并可由独立演示进程实时推送。"""
        self.events.append(event)
        if self.event_callback is not None:
            self.event_callback(deepcopy(event))

    def modeler(self, state: dict) -> dict:
        """重建时从已知权威源重新绑定字段；不使用故障答案训练 LLM。"""
        from api.services import agent_workflow as workflow
        update = workflow._modeler_node(state)
        before = self.fingerprint(self.bound)
        rebuilding = bool(state.get("recovery_feedback"))
        if self.scenario == "persistent_failure" or (self.scenario == "model_error" and not rebuilding):
            self.bound = deepcopy(self.corrupted)
        else:
            self.bound = deepcopy(self.source)
        self._record({"event": "rebind", "rebuilding": rebuilding, "before": before,
                            "after": self.fingerprint(self.bound), "source": self.fingerprint(self.source)})
        return update

    def solver(self, state: dict) -> dict:
        """对当前映射实际求解，然后用独立原始问题复算；不直接改 passed 标记。"""
        first = int(state.get("solver_attempt", 0)) == 0
        if self.scenario == "transient_failure" and first:
            # 受控传输失败仍保留为对照，明确记录没有运行底层求解器。
            self._record({"event": "transport_failure"})
            return {"result": {"status": "ERROR", "answer": "受控传输失败", "structured_answer": {},
                               "objective_value": None, "solution_verification": None}}
        data = self.corrupted if self.scenario == "tampered_objective" and first else self.bound
        problem = build_problem_envelope(self.instance.template_id, data, question=self.instance.question)
        solved = solve_problem_envelope(problem, time_limit=10)
        if not (solved.solution_verification and solved.solution_verification.passed):
            raise RuntimeError(f"当前映射下的参考求解失败：{self.instance.task_id}")
        verification = verify_solution(self.original, solved)
        changed = self.fingerprint(data) != self.fingerprint(self.source)
        if changed and verification.passed:
            raise RuntimeError("错误映射没有被独立原问题验证器检测到。")
        self._record({"event": "solve", "input": self.fingerprint(data), "source": self.fingerprint(self.source),
                            "self_verified": True, "source_verified": verification.passed,
                            "reported_objective": solved.objective_value, "recomputed_objective": verification.recomputed_objective})
        return {"result": {"status": solved.status, "answer": solved.summary, "structured_answer": {},
                           "objective_value": solved.objective_value, "solution_verification": verification.model_dump(mode="json")}}
