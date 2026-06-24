from __future__ import annotations

from dataclasses import dataclass
import os


@dataclass(frozen=True)
class SolverConfig:
    time_limit: int = 60
    mip_gap: float = 0.01
    threads: int = 0
    tsp_exact_limit: int = 18
    tsp_milp_limit: int = 80
    tsp_local_search_limit: int = 1000
    job_shop_milp_limit: int = 240
    cp_sat_time_limit: int = 60


def get_solver_config() -> SolverConfig:
    return SolverConfig(
        time_limit=_env_int("OPTIAGENT_TIME_LIMIT", 60),
        mip_gap=_env_float("OPTIAGENT_MIP_GAP", 0.01),
        threads=_env_int("OPTIAGENT_SOLVER_THREADS", 0),
        tsp_exact_limit=_env_int("OPTIAGENT_TSP_EXACT_LIMIT", 18),
        tsp_milp_limit=_env_int("OPTIAGENT_TSP_MILP_LIMIT", 80),
        tsp_local_search_limit=_env_int("OPTIAGENT_TSP_LOCAL_SEARCH_LIMIT", 1000),
        job_shop_milp_limit=_env_int("OPTIAGENT_JOB_SHOP_MILP_LIMIT", 240),
        cp_sat_time_limit=_env_int("OPTIAGENT_CP_SAT_TIME_LIMIT", 60),
    )


def configure_gurobi_model(model, config: SolverConfig | None = None) -> None:
    cfg = config or get_solver_config()
    model.Params.OutputFlag = 0
    model.Params.TimeLimit = cfg.time_limit
    model.Params.MIPGap = cfg.mip_gap
    if cfg.threads > 0:
        model.Params.Threads = cfg.threads


def quality_status(status: str, relative_gap: float | None = None) -> str:
    if status == "OPTIMAL":
        return "OPTIMAL"
    if status in {"TIME_LIMIT", "SUBOPTIMAL", "NODE_LIMIT", "ITERATION_LIMIT", "SOLUTION_LIMIT", "INTERRUPTED"} and relative_gap is not None and relative_gap <= get_solver_config().mip_gap:
        return "NEAR_OPTIMAL"
    return status


def _env_int(name: str, default: int) -> int:
    try:
        return int(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default


def _env_float(name: str, default: float) -> float:
    try:
        return float(os.environ.get(name, default))
    except (TypeError, ValueError):
        return default
