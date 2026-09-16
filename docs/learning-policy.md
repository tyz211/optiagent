# Learnable Recovery Policy

## 学习目标

当前参数化策略专注于 Verifier 之后的高层恢复决策：

$$
\pi_\theta(a_t \mid s_t, m_t)
$$

$s_t$ 是结构化状态，$m_t$ 是合法动作 mask。学习目标是在不绕过数学 Verifier 的前提下，最大化期望累积奖励：

$$
\max_\theta\;\mathbb{E}_{\pi_\theta}
\left[\sum_{t=0}^{T}\gamma^t r_t\right]
$$

学习对象不是 Gurobi 内部搜索，而是从 `accept_solution`、`retry_solver`、`rebuild_model` 和 `terminate` 中选择下一步。

## 为什么选择 BC + Action-Masked Double DQN

当前问题具有四个特征：

- 动作空间小且离散；
- 大量动作在特定状态下不合法；
- 真实 Solver/MCP 交互比普通游戏环境昂贵；
- 已经有可解释的规则策略可作为教师。

因此使用两阶段策略：

1. **Behavior Cloning 预热**：先学习规则教师的合法恢复行为，避免随机冷启动。
2. **Action-Masked Double DQN**：通过 replay buffer 和环境 reward 继续改进，用 target network 降低 Q 值过估计。

直接对 LLM token 做 PPO 暂不合适：成本高、credit assignment 难，且当前可验证数据量不足。

## 状态编码

`RecoveryStateEncoder v1.0` 生成 29 维向量：

- 六类问题模板 one-hot；
- Verifier 通过、可验证、可行性和目标一致性；
- 错误与约束违反数量；
- 实例规模、数据密度和难度；
- 建模/求解次数、剩余预算和当前步数；
- 上一个动作 one-hot。

编码器不读取 benchmark `scenario` 标签，避免向模型泄露正确答案。

## 训练目标

BC 使用被 action mask 约束的交叉熵：

$$
\mathcal{L}_{BC}=-\log \pi_\theta(a_t^{teacher}\mid s_t,m_t)
$$

Double DQN 的 target 为：

$$
y_t=r_t+\gamma(1-d_t)
Q_{\theta^-}\left(s_{t+1},
\arg\max_{a\in A(m_{t+1})}Q_\theta(s_{t+1},a)\right)
$$

只有 mask 允许的动作参与 exploration 和 target argmax。

## 默认超参数

| 参数 | 值 |
| --- | ---: |
| hidden dimension | 64 |
| learning rate | 0.001 |
| discount factor $\gamma$ | 0.95 |
| batch size | 32 |
| replay capacity | 10,000 |
| warmup transitions | 64 |
| target sync interval | 100 |
| epsilon | 0.80 → 0.05 |
| epsilon decay steps | 800 |
| gradient clip norm | 5.0 |
| BC epochs | 120 |
| BC learning rate | 0.002 |
| DQN train episodes | 800 |

checkpoint 同时保存状态编码器版本、动作顺序、超参数、policy/target network 和 optimizer 状态。不兼容的合同会被拒绝加载。

## 第一轮训练结果

seed 42、72 个分层 benchmark 任务、默认超参数：

- BC 教师决策：85 条；
- 环境交互：1,315 步；
- DQN 梯度更新：1,252 次；
- BC loss：`0.38268 → 0.000092`；
- learned policy test recoverable success：`1.00`；
- learned policy test average return：`0.75833`；
- random-valid test average return：`0.11250`；
- learned policy 非法动作率：`0.00`。

学习策略目前与规则教师持平，尚未证明能超越规则。现有微型环境的最优决策边界太简单，这是下一阶段必须通过真实 MCP 故障、动态成本和更大恢复空间解决的问题。

## 运行

```bash
pip install -r requirements-rl.txt
python scripts/train_recovery_policy.py \
  --seed 42 \
  --episodes 800 \
  --bc-epochs 120 \
  --output-root artifacts/rl/runs
```

每次执行都会生成新的 `run_id`，且不会覆盖历史训练：

```text
artifacts/rl/runs/
├── training_runs.jsonl
├── latest.json
└── <run_id>/
    ├── manifest.json
    ├── recovery_policy.pt
    └── training_report.json
```

`manifest.json` 记录状态、开始/结束时间、训练时长、超参数、Git 版本、运行环境、核心指标、文件大小和 SHA-256。失败训练也会以 `failed` 状态追加到 `training_runs.jsonl`。任何名称类似 API Key、Token 或 Secret 的字段都会从记录中移除；当前 Recovery Policy 训练不使用 LLM 生成梯度，所以会明确记录 `llm_used_for_training=false`。

checkpoint 和实验记录默认不提交 Git，避免二进制文件和机器相关运行产物污染仓库。

## 第二轮训练记录

第二轮采用与第一轮相同的训练规模，以 `seed=43` 做独立复现：

- run ID：`20260909T135846+0800_seed43_7f0b6c66`；
- 状态：`completed`；
- test success rate：`0.833333`；
- test recoverable success rate：`1.0`；
- test average return：`0.775`；
- test average steps：`1.5`；
- invalid action rate：`0.0`。

这一轮仍与规则教师持平，但保持了全部可恢复故障成功、零非法动作，并在该随机种子的测试任务上取得更短的平均恢复路径。该差异不能单独视为算法改进，需要通过多随机种子均值和置信区间判断稳定性。

## 五随机种子复现实验

使用完全相同的超参数完成 `seed=42,43,44,45,46` 五次独立训练，每次训练均保存独立 manifest、checkpoint 和完整报告：

| 指标 | Learned Policy 均值 ± 标准差 | Random Valid 均值 ± 标准差 |
| --- | ---: | ---: |
| test success rate | `0.8167 ± 0.0697` | `0.5333 ± 0.1118` |
| recoverable success rate | `1.0000 ± 0.0000` | `0.6539 ± 0.1326` |
| average return | `0.7417 ± 0.0876` | `0.3083 ± 0.1883` |
| average steps | `1.7667 ± 0.1807` | `1.5000 ± 0.1179` |
| invalid action rate | `0.0000 ± 0.0000` | `0.0000 ± 0.0000` |

Learned Policy 在全部五次实验中恢复了所有可恢复任务，并明显优于随机合法动作基线；但 Learned Policy 与 Rule Policy 的五项指标完全相同。这说明当前网络已经稳定复现教师恢复规则，同时也说明环境中的决策边界过于简单。后续训练应优先引入真实 MCP 故障、求解时延、调用成本和解质量差异，而不是继续增加当前微型环境的 episode 数量。

## 真实 Gateway 成本感知训练

`gateway-recovery-v1.1` 使用六类问题的真实 Gateway、求解器和数学 Verifier 输出作为参考轨迹，然后在这些真实结果上可控注入五类恢复场景：直接成功、瞬时求解失败、目标值传输损坏、可修复模型错误和持续失败。

本轮正式训练参数与规模：

- run ID：`20260909T144458+0800_seed47_d4134b18`；
- 真实参考任务：90 个，train/validation/test 各 30 个；
- 状态编码：`CostAwareRecoveryStateEncoder v2.0`，35 维；
- Behavior Cloning：160 epochs，54 条教师决策；
- Double DQN：1,200 episodes，2,080 次环境交互，2,017 次梯度更新；
- BC loss：`0.412268 -> 0.000270`；
- DQN final loss：`0.025987`。

独立测试集结果：

| 策略 | 可恢复成功率 | 平均回报 | 平均动作成本 | 非法动作率 |
| --- | ---: | ---: | ---: | ---: |
| Learned Policy | `1.0` | `0.677304` | `0.082696` | `0.0` |
| Cost-aware Teacher | `1.0` | `0.677304` | `0.082696` | `0.0` |
| Rule Policy | `1.0` | `0.653304` | `0.106696` | `0.0` |
| Random Valid | `0.666667` | `0.317381` | `0.055952` | `0.0` |

Learned Policy 已完全复现成本感知教师，并在不降低恢复成功率的前提下，相比现有 Rule Policy 将平均恢复成本降低约 22.5%。关键学习行为是：当解可行但目标值复算不一致时优先低成本 `retry_solver`；当模型可行性验证失败时使用 `rebuild_model`。

这里的“真实”指参考解、Solver 状态、Verifier 结果与延迟来自实际执行；故障转移仍是在这些真实轨迹上受控注入，并不等同于已经采集真实网络断连、MCP transport 超时或生产 LLM 错误。下一阶段应将这些外部故障作为真实 episode 写入 trajectory store。

## 真实 MCP transport 故障训练

`mcp-transport-recovery-v1` 将上一阶段的受控状态注入推进到协议层。采集器会实际启动隔离的 MCP stdio 子进程，建立 `ClientSession`，调用故障探针工具并记录四类安全观测：正常返回、超时、子进程断连和非法 structured content。训练任务还加入 `persistent_timeout`，用于学习在重试预算耗尽后终止，而不是伪造成功。

本轮正式训练参数与规模：

- run ID：`20260909T153158+0800_seed52_929552c6`；
- 任务：90 个，覆盖 6 类优化模板和 5 类 transport 场景，train/validation/test 各 30 个；
- 实际 MCP transport trace：12 条，每个 split 独立采集 4 类；
- 状态编码：`TransportAwareRecoveryStateEncoder v3.0`，40 维；
- Behavior Cloning：180 epochs，54 条教师决策；
- Double DQN：1,500 episodes，2,543 次环境交互，2,480 次梯度更新；
- BC loss：`0.321330 -> 0.000023`；
- DQN average/final loss：`0.062376 / 0.020344`。

独立测试集结果：

| 策略 | 全部任务成功率 | 可恢复成功率 | 平均回报 | 平均动作成本 | 非法动作率 |
| --- | ---: | ---: | ---: | ---: | ---: |
| Learned Policy | `0.800000` | `1.000000` | `0.553086` | `0.206914` | `0.0` |
| Transport Teacher | `0.800000` | `1.000000` | `0.553086` | `0.206914` | `0.0` |
| Rule Policy | `0.800000` | `1.000000` | `0.553086` | `0.206914` | `0.0` |
| Random Valid | `0.433333` | `0.541667` | `0.067328` | `0.106005` | `0.0` |

Learned Policy 对正常返回选择 `accept_solution`，对瞬时超时和断连选择 `retry_solver`，对非法返回选择 `rebuild_model`，对持续超时先重试并在预算耗尽后终止。它已经稳定复现教师与规则策略，并明显优于随机合法动作基线；本轮结果证明的是协议故障识别和恢复闭环已经可学习，而不是证明 DQN 超越了规则策略。

当前 trace 来自受控的本地 stdio 故障服务器，不等同于远程 Streamable HTTP 或真实生产网络。训练过程中不调用 LLM API，`llm_used_for_training=false`；记录只保存错误类型与耗时，不保存 API Key、Token、原始异常文本或模型请求内容。

运行命令：

```bash
python scripts/train_mcp_transport_policy.py \
  --seed 52 \
  --episodes 1500 \
  --bc-epochs 180 \
  --timeout-ms 80 \
  --delay-ms 240
```

## 下一个研究问题

下一步不是继续堆叠相同 episode，而是采集远程 Streamable HTTP MCP、LLM 建模错误和主 LangGraph 的真实轨迹，让不同动作在成功率、解质量、延迟与调用成本之间产生更复杂的 trade-off，再在未见实例上比较 Rule、LLM ReAct、BC、DQN 与 Offline RL。开源模型的 SFT/GRPO 属于独立的模型 policy 训练层，二者关系见 [双主线总体路线](two-track-roadmap.md)。
