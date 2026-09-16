# End-to-End Optimization Agent Benchmark 测试计划

## 目标

这套测试用于回答一个核心问题：Agent 的 reward 和恢复决策是否建立在真实优化环境反馈上，而不是仅仅学习模拟标签。

第一版烟雾测试使用六类小规模真实问题，所有干净样本都经过：

```text
ProblemEnvelope
  -> MCP Optimization Gateway
  -> Data Validation
  -> Registered Solver
  -> SolveEnvelope
  -> Independent SolutionVerifier
```

## 阶段计划

### Phase 1：合同与数学验证（已开始）

- 六类模板各一个真实求解实例；
- 干净数据必须得到可验证解；
- 删除必需 Schema 后必须在求解前拦截；
- 篡改 Solver 目标值后必须由 SolutionVerifier 检出。

### Phase 2：真实环境故障

- MCP transport 连接失败、超时和非法返回；
- Solver 超时、不可行、无许可证和后备求解器切换；
- 建模模板误选、约束遗漏和数据单位错误；
- 记录 Policy 是否在预算内选择正确恢复动作。

### Phase 3：规模化 Benchmark

- 每类模板 50–100 个独立实例；
- 每个实例生成自然语言、Schema 和工具故障变体；
- 按实例和表述模板双重隔离 train/validation/test；
- 比较 Fixed、Random Valid、Rule、Behavior Cloning 和 Offline RL。

## 首轮测试矩阵

| 故障 | 样本数 | 期望行为 |
| --- | ---: | --- |
| `clean` | 6 | Solver 成功且 Verifier 通过 |
| `missing_schema` | 6 | `INVALID_DATA`，不进入有效求解 |
| `tampered_objective` | 6 | Verifier 独立复算后拒绝 |

验收指标：

- `clean_success_rate = 1.0`；
- `fault_detection_rate = 1.0`；
- 全部 18 个 case 通过。

## 执行方式

```bash
PYTHONPATH=. python scripts/run_e2e_benchmark.py \
  --seed 42 \
  --time-limit 10 \
  --output data/benchmarks/e2e-smoke.json
```

脚本在任何 case 失败时使用非零退出码，可以直接接入 CI。

## 当前边界

首轮已经真正调用优化求解器和 Verifier，但 Schema 故障和目标值篡改仍是确定性注入。它暂时不包含网络层 MCP 异常、LLM 生成的建模错误或大规模求解器超时。
