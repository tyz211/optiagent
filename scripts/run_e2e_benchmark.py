from __future__ import annotations

import argparse
import json
from pathlib import Path

from optiagent.rl.e2e_benchmark import run_end_to_end_benchmark


def main() -> None:
    """运行真实 Gateway/Solver/Verifier 烟雾测试矩阵。"""

    parser = argparse.ArgumentParser(description="运行 OptiAgent 端到端 benchmark")
    parser.add_argument("--seed", type=int, default=42, help="实例生成随机种子")
    parser.add_argument("--time-limit", type=int, default=10, help="单次求解时间上限")
    parser.add_argument("--output", default="", help="可选的完整 JSON 报告路径")
    args = parser.parse_args()

    result = run_end_to_end_benchmark(seed=args.seed, time_limit=args.time_limit)
    if args.output:
        output_path = Path(args.output)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        output_path.write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({"case_count": result["case_count"], **result["metrics"]}, ensure_ascii=False))
    # 用非零退出码让 CI 能够拦截任何未检出的故障。
    if result["metrics"]["pass_rate"] < 1.0:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
