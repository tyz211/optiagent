"""集中组装内置模板，求解模块不再在导入时自行注册。"""

from __future__ import annotations

from optiagent.template_extensions import TemplateExtension
from optiagent.templates.capabilities import BUILTIN_CAPABILITIES
from optiagent.templates.definitions import BUILTIN_TEMPLATES
from optiagent.templates import validators


def _solve_facility(problem, report, time_limit):
    """保留仓库选址的专有结果合同，延迟引用避免 Gateway 导入环。"""
    from optiagent.optimization_gateway import _solve_facility_envelope
    return _solve_facility_envelope(problem, report, time_limit)


def builtin_extensions() -> list[TemplateExtension]:
    """只在统一注册表初始化时组装八个现有模板。"""
    from optiagent import generic_solvers as solvers
    from optiagent import solution_verifier as verifiers
    from optiagent.linear_solver import extract_linear_data, solve_linear_model, verify_linear_decisions
    from optiagent.transportation import extract_transportation_data, solve_transportation, verify_transportation
    from optiagent.solver_registry import GenericSolverAdapter

    bindings = {
        "facility_location": (validators.validate_facility_location, verifiers._verify_facility_location, None),
        "knapsack": (validators.validate_knapsack, verifiers._verify_knapsack,
                     GenericSolverAdapter("knapsack", "0-1 背包选择问题", "Gurobi", solvers.solve_knapsack, solvers._extract_knapsack_data)),
        "assignment": (validators.validate_assignment, verifiers._verify_assignment,
                       GenericSolverAdapter("assignment", "指派匹配问题", "Gurobi", solvers.solve_assignment, solvers._extract_assignment_data)),
        "tsp": (validators.validate_tsp, verifiers._verify_tsp,
                GenericSolverAdapter("tsp", "旅行商路径问题", "Exact DP / Gurobi MILP / 2-opt", solvers.solve_tsp, solvers._extract_tsp_data)),
        "job_shop_scheduling": (validators.validate_job_shop_scheduling, verifiers._verify_job_shop,
                                GenericSolverAdapter("job_shop_scheduling", "作业车间调度问题", "Gurobi MILP / List scheduling",
                                                     solvers.solve_job_shop_scheduling, solvers._extract_job_shop_data)),
        "production_mix": (validators.validate_production_mix, verifiers._verify_production_mix,
                           GenericSolverAdapter("production_mix", "产品组合与生产计划问题", "Gurobi",
                                                solvers.solve_production_mix, solvers._extract_production_mix_data)),
        "linear_program": (validators.validate_linear_program, verify_linear_decisions,
                           GenericSolverAdapter("linear_program", "文本线性规划 / 整数规划", "Gurobi LP/MILP", solve_linear_model, extract_linear_data)),
        "transportation": (validators.validate_transportation, verify_transportation,
                           GenericSolverAdapter("transportation", "运输分配", "Gurobi LP", solve_transportation, extract_transportation_data)),
    }
    extensions = []
    for template in BUILTIN_TEMPLATES:
        validator, verifier, adapter = bindings[template.template_id]
        extensions.append(TemplateExtension(
            template=template, capability=BUILTIN_CAPABILITIES[template.template_id],
            validate_data=validator, verify_decisions=verifier, generic_solver=adapter,
            envelope_solver=_solve_facility if adapter is None else None,
            envelope_solver_name="Gurobi MILP" if adapter is None else "",
        ))
    return extensions
