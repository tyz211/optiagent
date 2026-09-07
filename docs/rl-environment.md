# OptiAgent RL Environment v1.0

## 目标

`OptimizationAgentEnv` 将 Verifier 之后的 Agent 恢复问题抽象成可复现的序列决策环境。这一阶段学习的是高层策略，不是 Gurobi、CP-SAT 或启发式算法内部的搜索。

环境不强制依赖 Gymnasium，但提供兼容的返回形式：

```python
observation, info = env.reset(task_id=task_id)
next_observation, reward, terminated, truncated, info = env.step(action)
```

## 动作空间

动作编号是版本化训练合同的一部分：

| 编号 | 动作 | 含义 |
| ---: | --- | --- |
| 0 | `accept_solution` | 接受已通过 Verifier 的解 |
| 1 | `retry_solver` | 使用剩余求解预算重试 |
| 2 | `rebuild_model` | 根据数学违反反馈重新建模并求解 |
| 3 | `terminate` | 终止恢复并如实保留失败结果 |

每个 observation 都包含 `candidate_actions` 和等长 `action_mask`。选择被 mask 禁止的动作会立即终止 episode 并得到 `-1.0` 奖励。

## 状态

当前 observation 包含：

- 任务 ID、问题模板和 train/validation/test 划分；
- 实例规模、数据密度和难度等数值特征；
- Verifier 的通过状态、可验证性、可行性和约束违反；
- 建模/求解尝试次数与剩余预算；
- 上一个动作、当前步数和 action mask。

环境默认从第一次 Solver + Verifier 结束后开始，因此初始 `model_attempt=1`、`solver_attempt=1`。

## 当前环境参数

| 参数 | 默认值 |
| --- | ---: |
| `max_model_attempts` | 2 |
| `max_solver_attempts` | 2 |
| `max_steps` | 4 |
| 恢复动作步惩罚 | -0.05 |
| 接受已验证解 | +1.00 |
| 尚有恢复机会时提前终止 | -0.60 |
| 恢复预算耗尽后终止 | -0.20 |
| 步数预算耗尽 | -0.75 |
| 非法/被 mask 动作 | -1.00 |

这些是环境超参数，不是已训练的模型权重。

## Benchmark v1.0

`generate_benchmark()` 默认为六类问题模板各生成 12 个任务，共 72 个 episode：

- `direct_success`：应直接接受；
- `transient_solver_failure`：重试 Solver 后恢复；
- `repairable_model_error`：重建模后恢复；
- `persistent_failure`：恢复预算耗尽后正确终止。

每个模板内都独立划分 train/validation/test。随机种子控制场景顺序和数值特征，同一种子会生成相同数据。

默认规则 baseline 在 seed 42 的 72 个任务上得到：

- 全部任务成功率：`0.75`；
- 可恢复任务成功率：`1.00`；
- 非法动作率：`0.00`；
- 平均 return：`0.6625`；
- 平均步数：`1.75`。

`persistent_failure` 不应被计为可恢复成功；在这类任务上正确行为是终止而不是伪造可行解。

## 生成 JSONL

```bash
PYTHONPATH=. python scripts/generate_rl_dataset.py \
  --output data/rl/baseline.jsonl \
  --seed 42 \
  --variants-per-template 12 \
  --split all
```

每行是一个可回放 episode，包含 `state`、`candidate_actions`、`action_mask`、`action`、`reward`、`next_state` 和 `done`。

## 研究边界

当前 benchmark 是高层恢复策略的可控微型环境，故障转移是确定性注入，尚未真正调用 MCP 和求解器。它用于验证状态/动作/奖励合同和训练代码，不能代替最终的端到端 OR benchmark。下一版需将故障注入 MCP transport、数据映射和真实 Solver 执行层。
