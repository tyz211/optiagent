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

## 2026-09-17：主流程接入与验证集选模

`optiagent/recovery_runtime.py` 为主 LangGraph 提供统一恢复推理入口。默认继续使用规则策略；配置 `OPTIAGENT_RECOVERY_CHECKPOINT` 后加载本地 v1/v2（29/35 维）checkpoint，并固定使用贪心推理。网络必须遵守由确定性 Verifier 和尝试预算生成的 action mask；推理异常或非法输出会回退规则，并在决策 metadata 中记录原因。不存在或版本不兼容的文件在加载阶段明确报错。

40 维 transport checkpoint 仍用于独立环境。主工作流尚未提供完整连接超时、断连、协议合同和失败计数观测，因此暂时拒绝将这些 checkpoint 用于主流程。

每次网络决策仅执行一次。节点完成时，将实际动作与节点结果一起持久化，不再为记录轨迹额外调用规则策略。训练视图增加 `policy_observation`、`next_policy_observation` 与逐步 policy 版本；输入只包含当前可见的验证反馈、尝试预算、动作历史和成本估计。终止后的下一 observation 为 `null`，离线训练消费者必须结合 `done` 处理。

主流程成本使用与真实恢复环境一致的归一化公式，并以首次 Solver 节点耗时估算后续动作成本；它不是货币费用，且节点耗时包含应用组织响应的开销，与训练集中纯 Gateway 耗时存在差异。当前只从内联 JSON 精确计数实例规模，其他数据源记为未知；扩大上线范围前需补齐上传数据特征。

### 训练与模型选择

训练只在 train split 上收集教师样本和经验回放，每 100 episodes 在独立 validation 环境做贪心评估。以平均回报选模，同分保留较早 checkpoint；test split 不参与选择。保存三个版本：

- `recovery_policy.pt`：验证集选出的版本；
- `bc_policy.pt`：BC 预热结束时冻结的版本；
- `dqn_final_policy.pt`：最后一轮 DQN 版本。

报告单独列出三者指标，`training.selection` 包含评估历史、选中轮次与阶段。训练总更新次数和所选 checkpoint 的更新次数分别记录，不能混淆。旧 checkpoint 可继续加载；这些文件用于推理与复核，不是包含完整回放池的精确续训快照。

```bash
# 执行训练，保留全部模型对照和验证历史。
.venv/bin/python scripts/train_real_recovery_policy.py \
  --seed 53 --episodes 1200 --bc-epochs 160 \
  --threads 1 --validation-interval 100

# 替换为训练输出路径，评测真实主流程；输出文件不能重名。
.venv/bin/python scripts/evaluate_workflow_policy.py \
  --checkpoint artifacts/rl/runs/<run_id>/recovery_policy.pt \
  --seed 54 --output artifacts/rl/workflow_evaluations/<evaluation_id>.json

# 启用主流程学习策略；未配置此项时使用规则。
export OPTIAGENT_RECOVERY_CHECKPOINT="/absolute/path/to/recovery_policy.pt"
./start.sh
```

评测脚本每个案例使用临时数据库，真实执行 Requirement、Planner、Modeler、Solver 和 Verifier 等节点，并对求解后的验证反馈注入受控故障。仓库选址通过规范化三表数据走实际应用入口。生成的报告保留每个案例的训练视图、实际动作和模型摘要，不读取开发环境中的对话或 LLM 配置。

### 本轮结果

正式记录：`20260917T143825+0800_seed53_c4238979`。90 个任务，train/validation/test 各 30；BC 160 epochs、54 条教师决策；DQN 1,200 episodes、2,028 次交互、1,965 次更新。

| 测试策略 | 可恢复成功率 | 平均回报 | 平均动作成本 | 非法动作率 |
| --- | ---: | ---: | ---: | ---: |
| 验证集所选模型（BC） | 1.0 | 0.665474 | 0.094525 | 0.0 |
| BC | 1.0 | 0.665474 | 0.094525 | 0.0 |
| 末轮 DQN | 1.0 | 0.665474 | 0.094525 | 0.0 |
| Rule | 1.0 | 0.641475 | 0.118526 | 0.0 |
| Random Valid | 0.541667 | 0.097320 | 0.049347 | 0.0 |

验证集最终选择 BC（episode 0），因为 DQN 没有取得更高回报。BC 与末轮 DQN 均复现成本感知教师，不能将其相对 Rule 的优势解释为本轮 RL 更新带来的收益。随机策略成本更低伴随着更低的任务成功率。

对应主流程报告：`artifacts/rl/workflow_evaluations/20260917_seed54_final.json`。规则与学习策略各完成 30 个案例，均达到可恢复成功率 1.0、全部任务成功率 0.8，持续失败案例正确终止；无非法动作或推理回退。平均 Solver 调用均为 1.8 次，平均 Modeler 调用从 Rule 的 1.4 次降为学习策略的 1.2 次。

这些数据验证恢复策略与应用集成。当前烟雾实例使用少量数值变体，部分模板在不同 split 中仍可能出现相同的 OR 实例内容；不同 task ID 或 seed 不等于实例内容隔离。本轮不能作为严格未见实例泛化证据。下一步优先补充按规范化实例内容去重的训练集划分、真实用户修正和失败轨迹，再扩展 offline RL，避免仅增加重复 episode。

## 2026-09-17：实例内容隔离与轨迹数据集

真实恢复环境升级为 `gateway-recovery-v2.0`。六类模板不再复用少量烟雾数值变体，而是通过稳定的 `(seed, split, template, index)` 坐标生成不同规模、成本、容量与工序的实例。同一个实例仅做一次参考求解，然后派生五种恢复场景，共享参考结果与耗时，并留在同一个 split。

每个任务保存 `instance_data` 和 `instance_fingerprint`。训练前重新计算内容指纹，拒绝旧版缺少实例内容的数据、指纹与内容不一致的数据和跨 split 重复实例。MCP transport 训练也继承这一检查。指纹规范化字典顺序、表行顺序与等值数字表示，但不声称完成数学同构或所有语义重复检查。

```bash
# 每个模板、每个集合使用四个实例：6 × 3 × 4 = 72 个实例，派生 360 个故障任务。
.venv/bin/python scripts/train_real_recovery_policy.py \
  --seed 55 --episodes 1200 --bc-epochs 160 \
  --instances-per-split 4 --threads 1

# 扩展主流程采集：24 个实例 × 5 种场景 × 2 个策略 = 240 个 episode。
.venv/bin/python scripts/evaluate_workflow_policy.py \
  --checkpoint artifacts/rl/runs/<run_id>/recovery_policy.pt \
  --seed 56 --instances-per-template 4 \
  --output artifacts/rl/workflow_evaluations/expanded.json

# 按实际实例内容分组导出，保持实际执行动作与失败终止样本。
.venv/bin/python scripts/export_workflow_dataset.py \
  --evaluation-report artifacts/rl/workflow_evaluations/expanded.json \
  --output-dir artifacts/rl/datasets/controlled_expanded --seed 56
```

本轮训练记录：`20260917T173217+0800_seed55_ea4ab06f`。

- 训练/验证/测试各 24 个不同内容实例、120 个故障任务，跨集合指纹重叠为 0；
- BC 使用 216 条教师决策；DQN 完成 1,200 episodes、2,014 次交互和 1,951 次更新；
- 所选模型测试可恢复成功率 1.0、全部任务成功率 0.8、平均回报 0.655060、非法动作率 0；
- Rule 平均回报 0.631060，随机合法策略平均回报 0.225734；
- 验证选模仍保留 BC，末轮 DQN 与 BC 在测试上持平。移除显式内容重复后，当前控制场景仍未显示 DQN 超越教师的收益。

扩展主流程报告：`artifacts/rl/workflow_evaluations/20260917_seed56_expanded.json`。规则与学习策略各 120 个案例，可恢复成功率均为 1.0；平均建模次数分别为 1.4 与 1.2，均无非法动作或推理回退。

导出清单：`artifacts/rl/datasets/20260917_controlled_expanded/manifest.json`。共 240 个 episode、432 条决策，其中直接成功 48、恢复成功 144、失败终止 48；无被隔离记录。24 个实例经稳定哈希分为 train/validation/test 的 21/2/1 个实例，三个集合非空且内容隔离。由于验证和测试实例仍少，这批数据用于验证离线数据管道，不能单独作为稳定的离线策略评估基准。

真实工作流已增加来源、结果分类和内容指纹，数据库导出支持只读用户范围。当前实际检查的匿名范围内没有历史 episode；受控评测样本始终标记为受控来源，不能冒充生产失败轨迹。测试另验证了“策略选择重试后工具抛异常”的轨迹能保留实际动作及终局负奖励。

下一步：扩大真实用户失败与修正轨迹覆盖，补齐上传数据的内容标识，建立生产稀疏终局奖励与成本奖励的明确转换，再实现与 BC 对照的离线训练入口。当前新增的是可信数据合同与导出流程，并未完成生产轨迹 offline RL 训练。

## 2026-09-17：固定轨迹上的 BC + Masked CQL

新增 `scripts/train_offline_policy.py`，从已导出的 JSONL 直接训练，不在训练期间调用 Gateway、Solver、LLM 或主流程环境。消费端重新核验文件摘要、episode 合同、实际实例分组及 manifest 汇总；任一划分为空、文件被修改、动作非法或状态链不完整都会拒绝训练。

算法采用 BC 预热，再加入离散 Conservative Q-Learning 的保守项。CQL 在 Bellman 误差之外加入 Q 值正则，以缓解离线数据分布与所学策略之间的差异，参考 [Kumar 等人的 CQL 论文](https://arxiv.org/abs/2006.04779)。本项目采用固定系数、合法动作 mask、Double DQN 目标和 Huber TD 损失：

$$
L = L_{TD} + \alpha\,\mathbb{E}_{(s,a)\sim D}
\left[\log\sum_{a'\in A_{legal}(s)}\exp Q(s,a') - Q(s,a)\right].
$$

非法动作不参与保守项或下一状态 argmax；终止样本不读取下一状态网络。单个状态只有一个合法动作时，保守项为零。这里是针对当前四动作环境的工程实现，不宣称继承论文在其他条件下的全部理论保证。

### 奖励与评估边界

默认 `--reward-mode verified_cost` 使用新版本 `workflow-verified-cost-v1`：验证成功终局 +1、失败终局 -1，恢复动作另外扣除记录在状态中的成本估计。非终局基础奖励为 0。可用 `--reward-mode sparse` 保留原始稀疏奖励；源数据不会被覆盖。

两种模式都保留失败轨迹。BC 默认拟合全部训练集已记录动作，因此是行为策略对照，并不假定每条记录都是最优示范。数据集 manifest 摘要和训练奖励版本写入报告及 checkpoint，便于复核。

验证集选模使用 `logged_return_mse`，即 Q 网络对日志实际动作的估值与该条轨迹折扣回报之间的误差。测试集仅在训练和选模完成后用于报告。这个指标是诊断性代理，不是新策略价值估计；BC 分类分数原本也没有 Q 值校准，因此误差下降不能证明策略改进。报告另提供全部状态和多合法动作状态的行为一致率，避免大量强制动作掩盖差异。

```bash
# 固定数据集训练：默认只做离线采样，不产生新环境交互。
.venv/bin/python scripts/train_offline_policy.py \
  --dataset artifacts/rl/datasets/20260917_controlled_expanded \
  --seed 57 --steps 1500 --bc-epochs 160 --cql-alpha 0.1 \
  --reward-mode verified_cost

# 在新的主流程实例上分别测试验证选中模型和 BC 模型。
.venv/bin/python scripts/evaluate_workflow_policy.py \
  --checkpoint artifacts/rl/offline_runs/<run_id>/recovery_policy.pt \
  --seed 58 --instances-per-template 4 \
  --output artifacts/rl/workflow_evaluations/offline_cql.json
.venv/bin/python scripts/evaluate_workflow_policy.py \
  --checkpoint artifacts/rl/offline_runs/<run_id>/bc_policy.pt \
  --seed 58 --instances-per-template 4 \
  --output artifacts/rl/workflow_evaluations/offline_bc.json
```

训练产物位于 `artifacts/rl/offline_runs/<run_id>/`，包含 `recovery_policy.pt`、`bc_policy.pt`、`cql_final_policy.pt`、训练报告、输入数据 manifest 副本与运行 manifest。模型兼容现有 35 维推理接口，自动标记仍需主流程评测；训练不会自动修改 Web 服务使用的 checkpoint。这些文件不包含完整续训数据加载器状态，不作为精确续训快照。

### 第一轮离线结果

运行：`20260917T190810+0800_seed57_57ae19bb`。固定数据来自受控工作流，共 240 个 episode、432 条 transition，划分为 train/validation/test 的 210/20/10 个 episode，分别对应 21/2/1 个实例。BC 160 epochs，CQL 更新 1,500 次，训练期间环境交互次数为 0；验证集选中第 200 次更新的 CQL checkpoint。

测试日志上，选中模型的全部动作一致率约 94.44%，多合法动作状态一致率 87.5%，日志回报 MSE 为 0.212900，非法动作率为 0。以上不是成功率或部署收益。

随后在 seed 58 的 24 个新实例、120 个故障案例上分别运行 BC 和 CQL。已核验这 24 个实例与离线数据三个集合的内容指纹重叠为 0，且两个模型评测使用相同实例：

| 主流程指标 | BC | 离线 CQL |
| --- | ---: | ---: |
| 全部任务成功率 | 0.8 | 0.8 |
| 可恢复任务成功率 | 1.0 | 1.0 |
| 平均 Solver 调用 | 1.8 | 1.8 |
| 平均 Modeler 调用 | 1.4 | 1.233333 |
| 非法动作 / 异常回退 | 0 / 0 | 0 / 0 |

比较报告：`artifacts/rl/workflow_evaluations/20260917_offline_comparison.json`，包含两个原始报告的文件摘要。CQL 在这轮受控测试中保持恢复率，并比本轮 BC 减少约 11.9% 的建模调用；这是单种子、小规模结果，未证明生产收益或统计显著性。离线验证和测试实例尤其少，下一阶段需扩大覆盖并做多随机种子复现。

已验证文件篡改检测、伪造 action/跨集合记录拒绝、奖励转换、非法动作的保守项梯度、终局不自举、Double DQN 目标选择，以及修改测试奖励不会改变训练权重或选模。真实用户数据的离线训练仍需等待足够的真实失败与修正样本，本轮不把受控轨迹标记为生产数据。
