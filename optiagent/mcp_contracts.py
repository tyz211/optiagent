from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field

from optiagent.problem_spec import ProblemSpec


SCHEMA_VERSION = "1.0"


class DataRequirementModel(BaseModel):
    """MCP 边界上的数据需求。"""

    table: str
    columns: list[str]
    description: str


class ProblemSpecModel(BaseModel):
    """可被 MCP 工具稳定传输的问题定义。"""

    problem_type: str
    display_name: str
    objective: str
    sets: list[str] = Field(default_factory=list)
    parameters: list[str] = Field(default_factory=list)
    decision_variables: list[str] = Field(default_factory=list)
    constraints: list[str] = Field(default_factory=list)
    recommended_solver: str
    solver_reason: str
    data_requirements: list[DataRequirementModel] = Field(default_factory=list)
    output_schema: list[str] = Field(default_factory=list)
    template_id: str
    confidence: float = 0.0
    assumptions: list[str] = Field(default_factory=list)
    missing_data: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)

    @classmethod
    def from_problem_spec(cls, spec: ProblemSpec) -> "ProblemSpecModel":
        """将项目内部 dataclass 转成 MCP 合同模型。"""

        return cls.model_validate(spec.to_dict())


class SourceReference(BaseModel):
    """记录问题数据的来源，便于结果审计。"""

    kind: Literal["file", "database", "api", "inline", "knowledge"]
    name: str
    uri: str | None = None
    details: dict[str, Any] = Field(default_factory=dict)


class ValidationReport(BaseModel):
    """统一的数据和模型校验结果。"""

    valid: bool = True
    errors: list[str] = Field(default_factory=list)
    warnings: list[str] = Field(default_factory=list)
    checks: dict[str, Any] = Field(default_factory=dict)


class ProblemEnvelope(BaseModel):
    """Document/Data MCP 与 Solver MCP 之间的唯一公共输入。"""

    schema_version: str = SCHEMA_VERSION
    problem_spec: ProblemSpecModel
    data: dict[str, Any] = Field(default_factory=dict)
    sources: list[SourceReference] = Field(default_factory=list)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    metadata: dict[str, Any] = Field(default_factory=dict)


class SolverCapability(BaseModel):
    """求解器能力描述。"""

    template_id: str
    display_name: str
    solver_name: str
    exact: bool
    input_keys: list[str]
    notes: list[str] = Field(default_factory=list)


class SolveEnvelope(BaseModel):
    """Solver MCP 的统一输出。"""

    schema_version: str = SCHEMA_VERSION
    template_id: str
    status: str
    objective_value: float | None = None
    objective_label: str = "目标值"
    solver_name: str = "Unknown"
    summary: str
    decisions: list[dict[str, Any]] = Field(default_factory=list)
    metrics: dict[str, Any] = Field(default_factory=dict)
    warnings: list[str] = Field(default_factory=list)
    validation: ValidationReport = Field(default_factory=ValidationReport)
    provenance: list[SourceReference] = Field(default_factory=list)
