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
from optiagent.solver_registry import list_generic_solvers


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

    capabilities = [
        SolverCapability(
            template_id="facility_location",
            display_name="仓库选址与客户分配",
            solver_name="Gurobi MILP",
            exact=True,
            input_keys=["warehouses", "customers", "costs"],
            notes=["需要可用的 Gurobi 许可证。"],
        )
    ]
    input_keys = {
        "knapsack": ["items", "capacity"],
        "assignment": ["resources", "tasks", "costs"],
        "tsp": ["distances 或 distance_matrix"],
        "job_shop_scheduling": ["tasks"],
        "production_mix": ["products", "capacities"],
    }
    exact = {
        "knapsack": True,
        "assignment": True,
        "tsp": False,
        "job_shop_scheduling": False,
        "production_mix": True,
    }
    for adapter in list_generic_solvers():
        capabilities.append(
            SolverCapability(
                template_id=adapter.template_id,
                display_name=adapter.display_name,
                solver_name=adapter.solver_name,
                exact=exact.get(adapter.template_id, False),
                input_keys=input_keys.get(adapter.template_id, []),
                notes=["实际最优性以返回的 status 和 optimality_proven 为准。"],
            )
        )
    return capabilities


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
