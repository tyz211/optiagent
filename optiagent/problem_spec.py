from __future__ import annotations

from dataclasses import asdict, dataclass, field
from typing import Any

from optiagent.data import SupplyChainData


@dataclass(frozen=True)
class DataRequirement:
    table: str
    columns: list[str]
    description: str


@dataclass(frozen=True)
class ProblemSpec:
    problem_type: str
    display_name: str
    objective: str
    sets: list[str]
    parameters: list[str]
    decision_variables: list[str]
    constraints: list[str]
    recommended_solver: str
    solver_reason: str
    data_requirements: list[DataRequirement]
    output_schema: list[str]
    template_id: str
    confidence: float = 0.0
    assumptions: list[str] = field(default_factory=list)
    missing_data: list[str] = field(default_factory=list)
    notes: list[str] = field(default_factory=list)

    def to_dict(self) -> dict[str, Any]:
        payload = asdict(self)
        payload["data_requirements"] = [asdict(item) for item in self.data_requirements]
        return payload


def infer_problem_spec(question: str, data: SupplyChainData | None = None) -> ProblemSpec:
    from optiagent.templates.registry import get_template, rank_templates

    ranked = rank_templates(question, data)
    template = ranked[0] if ranked else get_template("facility_location")
    spec = template.build_spec(question=question, data=data)
    return spec


def spec_summary(spec: ProblemSpec) -> list[str]:
    return [
        f"识别问题：{spec.display_name}（{spec.problem_type}），置信度 {spec.confidence:.0%}。",
        f"目标：{spec.objective}",
        f"推荐求解器：{spec.recommended_solver}，原因：{spec.solver_reason}",
        f"核心变量：{'；'.join(spec.decision_variables[:4])}",
        f"关键约束：{'；'.join(spec.constraints[:4])}",
    ]
