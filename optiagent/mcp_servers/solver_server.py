from __future__ import annotations

from mcp.server.fastmcp import FastMCP

from optiagent.mcp_contracts import (
    ProblemEnvelope,
    SolutionVerificationReport,
    SolveEnvelope,
    SolverCapability,
    ValidationReport,
)
from optiagent.mcp_servers.common import run_server
from optiagent.optimization_gateway import solve_problem_envelope, validate_problem_envelope
from optiagent.solution_verifier import verify_solution
from optiagent.template_extensions import list_template_extensions


mcp = FastMCP(
    "OptiAgent Solver MCP",
    instructions=(
        "只接收已结构化的 ProblemEnvelope，先校验再调用已注册求解器。"
        "不从自然语言猜测或伪造优化参数。"
    ),
    json_response=True,
)


@mcp.tool(title="列出求解能力")
def solver_list_capabilities() -> list[SolverCapability]:
    """列出当前 Solver MCP 已注册的模板、求解器与输入字段。"""

    # 展示和求解读取同一份完整扩展，不再维护仓库选址特例或字段映射。
    return [
        SolverCapability(
            template_id=extension.template.template_id,
            display_name=extension.template.display_name,
            solver_name=extension.solver_name,
            exact=extension.capability.exact,
            input_keys=list(extension.capability.input_keys),
            notes=["实际最优性以返回的 status 和 optimality_proven 为准。",
                   "支持：" + "、".join(extension.capability.supported),
                   "暂不支持：" + "、".join(extension.capability.unsupported)],
        )
        for extension in list_template_extensions()
    ]

@mcp.tool(title="校验求解请求")
def solver_validate_problem(problem: ProblemEnvelope) -> ValidationReport:
    """在运行求解器前重新校验 ProblemEnvelope 的版本和数据。"""

    return validate_problem_envelope(problem)


@mcp.tool(title="执行优化求解")
def solver_solve_problem(problem: ProblemEnvelope, time_limit: int | None = None) -> SolveEnvelope:
    """根据 ProblemEnvelope.template_id 选择已注册求解器，返回统一结果。"""

    return solve_problem_envelope(problem, time_limit=time_limit)


@mcp.tool(title="独立验证优化解")
def solver_verify_solution(
    problem: ProblemEnvelope,
    solution: SolveEnvelope,
) -> SolutionVerificationReport:
    """根据原始问题数据复算约束和目标值，不信任求解器汇总字段。"""

    return verify_solution(problem, solution)


if __name__ == "__main__":
    run_server(mcp, default_port=8103)
