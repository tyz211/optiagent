# OptiAgent 双主线总体路线

## 1. 项目总目标

项目未来由两个相互连接、训练对象不同的部分组成：

1. **Agent System**：负责规划、状态管理、MCP 工具路由、求解控制、结果验证和失败恢复；
2. **Open Model Post-Training**：在云端部署开源语言模型，并使用监督微调（SFT）与 Group Relative Policy Optimization（GRPO）提升模型的运筹建模、结构化工具调用和基于反馈修复能力。

统一研究问题可以表述为：

$$
\boxed{\text{Learning Verifiable Agent and Language Model Policies for Automated Optimization Modeling and Solving}}
$$

原研究问题 **Learning an Agent Policy for Automated Optimization Modeling and Solving** 仍然是第一部分的核心；第二部分补充为：

$$
\boxed{\text{Post-Training Open-Source Language Models for Verifiable Optimization Modeling and Tool Use}}
$$

## 2. 总体架构

```text
                         用户任务 / 数据文件
                                  │
                                  ▼
┌──────────────────────────────────────────────────────────────┐
│ Part A · Agent System                                        │
│ Planner → Data Agent → Modeler → Tool Router → Solver        │
│                         ▲                    │                │
│                         └── Policy ← Verifier ┘                │
│                             │                                 │
│              accept / retry / rebuild / terminate            │
└─────────────────────────────┬────────────────────────────────┘
                              │ OpenAI-compatible API
                              ▼
┌──────────────────────────────────────────────────────────────┐
│ Part B · Open Model Service                                  │
│ Base Open Model → SFT Model → GRPO Model → Versioned Serving │
│        理解 / ProblemSpec / Tool Call / Model Repair          │
└─────────────────────────────┬────────────────────────────────┘
                              │
                              ▼
               Document MCP / Data MCP / Solver MCP
                              │
                              ▼
                    OR Solver + Solution Verifier
                              │
                  可行性 / 目标质量 / 成本 / 延迟
                              │
                              ▼
                    Trajectory & Reward Store
                              │
                 ┌────────────┴────────────┐
                 ▼                         ▼
          Agent Policy 训练          SFT / GRPO 数据
```

## 3. Part A：Agent System

### 3.1 职责

- 保存完整任务状态与执行预算；
- 选择澄清、数据解析、知识检索、建模、求解和验证动作；
- 通过 MCP 调用 Document、Data 和 Solver 服务；
- 根据 Verifier、延迟、成本和故障信号进行重试或修复；
- 将所有状态、动作、工具返回和奖励保存为可回放 trajectory。

### 3.2 学习对象

Agent policy 记为：

$$
\pi_{agent}(a_t \mid s_t, m_t)
$$

其中 $a_t$ 是有限的高层动作，$m_t$ 是合法动作 mask。当前实现的 BC + Action-Masked Double DQN 属于这一层，学习的是 `accept_solution`、`retry_solver`、`rebuild_model` 和 `terminate` 等控制决策，不直接生成语言 token。

### 3.3 当前进度

- LangGraph 七节点执行链与有界恢复回路；
- Document/Data/Solver MCP 与统一 Gateway；
- 数学 Solution Verifier 和确定性 reward；
- episode/step trajectory store；
- 29/35/40 维版本化状态编码；
- BC + Masked Double DQN；
- 真实 Solver 参考结果和真实 MCP stdio 故障训练；
- 独立 train/validation/test 与不可覆盖训练记录。
- 29/35 维 checkpoint 已接入主 LangGraph，实际动作与观测可导出；训练支持验证集选模、独立 BC/末轮 DQN 对照和权重保存。
- 真实恢复训练集已按实例内容指纹隔离；工作流支持来源/结果标记、只读 JSONL 导出和数据质量隔离清单。
- 已提供固定轨迹的 BC + Masked CQL 离线训练入口、版本化奖励转换与独立日志诊断；生产用户轨迹覆盖仍待扩大。

### 3.4 下一步

- 补齐主流程 transport 观测并接入 40 维策略，扩展上传数据的规模特征与动态预算；
- 扩展 `ask_clarification`、`select_tool`、`select_solver` 和 `revise_model` 等层次化动作；
- 采集远程 Streamable HTTP MCP、LLM 建模错误和真实用户修正轨迹；
- 建立 rule、LLM ReAct、BC、DQN 和 offline RL 的统一推理接口。

## 4. Part B：Open Model Post-Training

### 4.1 职责

开源模型负责需要语言与代码生成能力的部分：

- 理解自然语言目标、约束和数据语义；
- 生成版本化 `ProblemSpec`；
- 生成符合 Schema 的 MCP tool call；
- 根据 Solver/Verifier 错误修复模型；
- 生成忠于真实求解结果的解释。

模型不替代 OR Solver，也不能自行宣称可行或最优。所有数学结论仍由 Solver 和 Verifier 确认。

### 4.2 SFT 目标

SFT 首先学习高质量示范：

```text
自然语言 + 数据摘要 + 工具历史
  → 结构化计划
  → ProblemSpec
  → MCP Tool Call
  → 基于错误反馈的修复动作
```

训练样本主要来自人工校正、规则系统、强教师模型和经过 Verifier 筛选的成功轨迹。第一阶段优先使用参数高效微调，保留基础模型能力并降低个人项目的云端训练成本。

### 4.3 GRPO 目标

完成 SFT 后，再对同一任务采样一组候选输出，通过可执行环境计算组内相对奖励。模型策略记为：

$$
\pi_{LM}(y \mid x, c)
$$

其中 $y$ 是模型生成的 token 或结构化动作，$x$ 是任务输入，$c$ 是 Agent 提供的状态上下文。GRPO 优化的不是当前 40 维 DQN policy，而是开源语言模型本身。

优先使用无需单独训练 reward model 的可验证奖励：

$$
R_{model} = w_s R_{schema} + w_e R_{executable}
  + w_f R_{feasible} + w_q R_{quality} + w_v R_{verified}
  - w_c C_{tool} - w_l C_{latency} - w_h P_{hallucination}
$$

- `R_schema`：`ProblemSpec` 与 tool call 是否满足合同；
- `R_executable`：生成模型能否被 Gateway 和 Solver 执行；
- `R_feasible`：求解结果是否满足约束；
- `R_quality`：目标值相对最优值或强基线的质量；
- `R_verified`：结论能否被独立复算；
- `C_tool` / `C_latency`：工具调用和延迟成本；
- `P_hallucination`：虚构字段、结果或最优性声明。

### 4.4 云端服务边界

模型训练和模型推理解耦：

- **Training Job**：读取版本化数据集，执行 SFT/GRPO，输出 adapter、checkpoint 和训练报告；
- **Model Registry**：保存 base/SFT/GRPO 版本、数据版本、超参数和评测结果；
- **Inference Service**：通过 OpenAI-compatible API 暴露模型；
- **Agent Client**：只依赖 `base_url`、`model` 和运行时密钥，不绑定具体云厂商或推理框架。

密钥只用于在线推理，不进入 trajectory、训练集、checkpoint 或 Git 仓库。

## 5. 两种 RL 必须明确区分

| 维度 | Agent Policy RL | Open Model GRPO |
| --- | --- | --- |
| 学习对象 | 小型高层决策网络 | 开源语言模型 |
| 输入 | 结构化状态与 action mask | prompt、上下文和历史轨迹 |
| 输出 | 有限离散动作 | token、ProblemSpec 或 tool call |
| 当前算法 | BC + Masked Double DQN | 计划使用 SFT + GRPO |
| 主要奖励 | 恢复成功、动作成本、延迟 | Schema、可执行性、可行性、解质量 |
| 主要作用 | 控制流程和预算 | 提升理解、建模与修复能力 |

早期不建议同时更新两种 policy。先固定 Agent 训练模型，再固定模型训练 Agent，可以避免环境与行为策略同时变化造成的非平稳问题。

## 6. 数据闭环

每条训练数据都应能追溯：

```text
task_id
  → dataset_version
  → agent_policy_version
  → model_version
  → prompt/messages
  → generated ProblemSpec / tool calls
  → MCP observations
  → solver result
  → verifier report
  → scalar reward + reward components
  → human correction（可选）
```

数据形成两个视图：

- **Agent transition view**：`state → high-level action → next_state → reward`；
- **Model rollout view**：`messages → generated tokens/structure → verifier reward`。

同一次运行可以同时产生这两个视图，但必须记录各自的 policy/model 版本，防止训练数据来源混淆。

## 7. 统一评测设计

### Agent System 指标

- task success / recovery success；
- 非法动作率、工具调用次数与重复调用率；
- MCP/Solver 成本、端到端延迟与预算违约率；
- 在超时、断连、脏数据和不可行模型下的恢复率。

### Open Model 指标

- `ProblemSpec` Schema 通过率；
- tool call 合法率与参数正确率；
- model execution rate；
- solution feasibility、objective consistency 和 objective gap；
- 修复成功率与无依据结论率。

### 端到端消融

至少比较下列组合：

| Agent | Model | 目的 |
| --- | --- | --- |
| Rule | Base Model | 最低学习基线 |
| Learned Agent | Base Model | 测量 Agent policy 收益 |
| Rule | SFT Model | 测量监督微调收益 |
| Rule | SFT + GRPO Model | 测量可验证 RL 收益 |
| Learned Agent | SFT + GRPO Model | 最终系统效果 |

## 8. 分阶段路线

### M0：当前基础

完成 Agent runtime、MCP、Solver/Verifier、trajectory、Recovery Policy 和可复现实验记录。

### M1：模型训练数据合同

扩展 trajectory schema，记录脱敏 messages、模型输出、token 级生成元数据、model version 和细粒度 reward components，并生成 SFT/GRPO 两种数据视图。

### M2：开源模型 SFT

在云端部署基础模型，建立 base-model benchmark；用经过 Verifier 筛选的数据完成第一轮参数高效 SFT，并记录数据、模型和训练版本。

### M3：Verifier-guided GRPO

在隔离 benchmark 中为每个 prompt 采样多个候选，通过 Schema、Solver 和 Verifier 计算奖励；先做离线评测，再将通过门槛的模型部署为新版本。

### M4：Agent 与模型集成

让主 LangGraph 可选择 base/SFT/GRPO 模型，把模型错误反馈写回 trajectory，并将 learned Agent policy 接入受约束的工具路由和恢复节点。

### M5：联合研究评测

完成双因素消融、跨问题类型泛化、故障恢复、成本/延迟和奖励投机测试，形成可复现报告。

## 9. 建议的仓库边界

在真正开始模型后训练时，建议保留单仓库，但按职责拆分：

```text
optiagent/              Agent runtime、MCP、Gateway、Solver、Verifier
model_training/
  datasets/             SFT/GRPO 数据构建与版本检查
  sft/                  监督微调配置和训练入口
  grpo/                 rollout、奖励函数和 GRPO 训练入口
  evaluation/           base/SFT/GRPO 独立评测
deployment/
  model_service/        云端推理服务配置
artifacts/
  rl/runs/              Agent policy 实验
  model_runs/           SFT/GRPO manifest、指标与 checkpoint 引用
```

模型权重与大型数据集不直接提交 Git；仓库只保存配置、数据 manifest、可复现脚本、指标和远程 artifact 引用。

## 10. 项目边界与对外表述

当前可以表述为：

> OptiAgent 是一个面向自动运筹建模与求解的、MCP 原生且数学可验证的 Agent 研究原型，已经实现高层恢复策略学习，并计划通过 SFT 与 GRPO 对云端开源模型进行后训练。

在完成模型数据集、SFT checkpoint 和独立评测之前，不应表述为“已经完成开源模型后训练”；在 Agent policy 与后训练模型都通过端到端未见任务测试之前，也不应表述为“联合训练系统”。
