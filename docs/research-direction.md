# Research Direction: Learning Agent and Model Policies for Automated Optimization Modeling and Solving

## 1. 核心研究问题

$$
\boxed{\text{Learning an Agent Policy for Automated Optimization Modeling and Solving}}
$$

给定自然语言任务、一个或多个异构数据源、可调用的建模与求解工具，以及有限的时间和调用预算，如何学习一个 Agent policy，使系统能够自主完成：

1. 判断信息是否足够，必要时请求澄清；
2. 选择数据解析、知识检索、建模和求解工具；
3. 构造可执行的优化问题表示；
4. 选择合适的求解器、算法与参数；
5. 验证解的可行性、目标值和结论；
6. 在失败时定位错误并修复，而不是直接停止或产生不可验证答案。

这一定义把项目从“LLM 包装的求解器”转变为“在可验证优化环境中进行序列决策的 Agent”。

项目未来分为两条主线：

- **Agent System**：保留原研究问题，学习规划、MCP 路由、求解控制和失败恢复；
- **Open Model Post-Training**：在云端部署开源语言模型，通过 SFT 与 GRPO 学习 `ProblemSpec`、tool call、建模与修复。

两条主线共享 trajectory、Solver 和 Solution Verifier，但不混淆训练对象。Agent policy 输出有限高层动作，语言模型 policy 输出 token 或结构化建模结果。完整设计见 [双主线总体路线](two-track-roadmap.md)。

## 2. 研究假设

项目的核心假设是：相比一次性提示或固定工作流，一个利用求解器反馈、验证器反馈和历史轨迹学习得到的策略，能够在未见问题、脏数据和工具失败条件下取得更高的端到端成功率，并以更少的无效工具调用完成任务。

这里的学习对象不是 Gurobi、CP-SAT 或启发式算法内部的搜索过程，而是更高层的 Agent policy：下一步该澄清、检索、建模、调用哪个工具、调整什么参数、修复哪一部分，或者何时终止。

## 3. 序列决策形式化

将一次优化任务表示为部分可观测决策过程。策略为：

$$
\pi_\theta(a_t \mid s_t)
$$

其中状态 $s_t$ 不直接等同于完整环境，而是 Agent 可见的结构化上下文：

- 用户问题与对话历史；
- 文件、表结构、缺失值和字段语义；
- 当前 `ProblemSpec` 与建模草案；
- 已调用工具、参数、成本、延迟与返回值；
- 求解状态、MIP gap、约束违反和验证报告；
- 剩余 token、时间和工具调用预算。

动作 $a_t$ 来自一个受约束的层次化动作空间：

| 动作族 | 示例动作 |
| --- | --- |
| 信息获取 | `ask_clarification`、`inspect_file`、`retrieve_modeling_knowledge` |
| 建模 | `select_template`、`build_problem_spec`、`revise_constraints` |
| 工具路由 | `call_document_mcp`、`call_data_mcp`、`call_solver_mcp` |
| 求解控制 | `select_solver`、`set_time_limit`、`set_mip_gap`、`retry_with_fallback` |
| 验证与修复 | `verify_solution`、`repair_schema`、`repair_model` |
| 终止 | `return_result`、`report_missing_information` |

MCP 工具和优化求解器构成环境。它们返回结构化 observation，LangGraph 管理状态转移，Verifier 提供可计算反馈。

## 4. 奖励设计

奖励不能只看“有没有输出答案”，应由多个可验证信号组成：

$$
R = w_s R_{success} + w_f R_{feasible} + w_q R_{quality}
    + w_v R_{verified} - w_c C_{tool} - w_l C_{latency}
    - w_i P_{invalid} - w_h P_{hallucination}
$$

- `R_success`：任务是否得到有效终局结果。
- `R_feasible`：返回决策是否满足全部约束。
- `R_quality`：目标值相对已知最优值、best-known solution 或基线的归一化质量。
- `R_verified`：目标值与关键结论能否被独立复算。
- `C_tool`：LLM、MCP 和求解器调用成本。
- `C_latency`：端到端时间与超时惩罚。
- `P_invalid`：错误 Schema、非法参数、重复或无意义工具调用。
- `P_hallucination`：声称最优但无法证明、虚构字段或未引用真实求解结果。

奖励应按问题类型归一化，避免最小化与最大化、不同目标量纲或不同实例规模之间不可比较。训练时优先使用确定性验证信号，语言风格评分只作为辅助项。

## 5. Policy 学习路线

### 阶段 A：可观测基线

先运行规则 policy 与 LLM planner，完整记录 $(s_t,a_t,o_t)$。当前七个职责节点、Verifier 后条件边和有界恢复回路已经提供可执行的 policy baseline。

### 阶段 B：Trajectory Dataset

把每次运行保存为版本化 episode：任务、状态摘要、候选动作、实际动作、工具 observation、验证结果、成本和终局 reward。加入成功、失败、恢复成功和人工修正案例。

### 阶段 C：Behavior Cloning

从规则、强 LLM 和人工修正产生的高质量轨迹学习初始 policy。当前已实现基于 29 维结构化状态和 action mask 的 BC 预热。

### 阶段 D：Offline RL / Preference Optimization

利用历史轨迹中的 solver/verifier reward 学习比行为策略更好的工具路由和失败恢复策略。离线训练适合个人项目：实验可复现、成本可控，也避免在线探索频繁产生无效求解。
当前已在可控微型环境上实现 Action-Masked Double DQN，使用真实 Gateway/Solver/Verifier 输出构建成本感知恢复任务，并通过真实 MCP stdio 会话采集超时、子进程断连和非法结构返回。生产 trajectory store 的 offline RL、远程 Streamable HTTP 故障和真实 LLM 建模失败学习尚未完成。

### 阶段 E：受约束的在线改进

只在沙箱基准任务中尝试 contextual bandit 或受约束 RL。动作必须通过工具 Schema 校验，终局结果必须经过 Solution Verifier，不允许策略绕过确定性验证。

## 6. 评测问题

研究评测至少回答以下问题：

- 学习策略是否比固定规则和纯 LLM ReAct 更容易得到可执行模型？
- 在列名变化、缺失数据、矛盾约束和工具故障下，是否能主动恢复？
- 是否能减少无效 MCP 调用、总延迟和求解成本？
- 能否在未见过的实例规模、问题表述或组合任务上泛化？
- reward 提升是否对应真实可行率和目标质量提升，而非奖励投机？

建议基线：

| Policy | 作用 |
| --- | --- |
| Fixed Workflow | 衡量状态图本身的稳定上限与局限 |
| Rule Router | 当前无 LLM 的确定性基线 |
| LLM ReAct / Tool Calling | 衡量通用 LLM 的零样本决策能力 |
| Behavior Cloning | 衡量高质量轨迹监督的收益 |
| Offline RL Policy | 衡量 reward 驱动的策略改进 |

主要指标包括 task success、model execution rate、solution feasibility、objective gap、verification pass rate、recovery rate、tool calls、latency 和 token/solver cost。

## 7. Benchmark 设计

基准任务不应只包含干净的标准实例，还应系统地产生 Agentic 扰动：

- 自然语言改写、隐含目标和需要澄清的歧义；
- CSV 列名变化、单位变化、缺失值、重复行和多文件错配；
- 不可行约束、求解超时、求解器不可用和 MCP 临时失败；
- 需要“解析 → 建模 → 求解 → 验证 → 修复”的长程任务；
- 背包、指派、TSP、调度、生产计划和仓库选址之间的跨任务泛化。

数据集划分应按问题实例与表述模板双重隔离，防止仅记住 prompt 或固定 Schema。

## 8. 与当前系统的对应关系

| 研究组件 | 当前实现 | 下一步 |
| --- | --- | --- |
| Policy execution | LangGraph 条件状态图、四动作 Recovery Policy、有界重试、独立参数化 policy | 将学习策略接入主 LangGraph，并支持动态预算 |
| Environment | MCP + Gateway、真实 stdio transport 故障，以及可独立 `reset/step` 的 RL 环境 | 扩展远程 HTTP、LLM 和生产网络故障 |
| State | `AgentWorkflowState`、Verifier/成本/transport 反馈与 29/35/40 维版本化编码 | 加入 LLM 置信度、历史摘要和跨步工具上下文 |
| Trajectory | episode/step store、候选动作、mask、`next_state` 与 decision transitions | 数据集版本、质量筛选和批量文件导出 |
| Reward | 六类数学 Solution Verifier、响应合同与确定性终局 reward | 加入成本、延迟与跨实例归一化 reward |
| Baselines | Rule、Random Valid、BC + Action-Masked Double DQN | ReAct、真实轨迹 Offline RL 与统一推理接口 |
| Evaluation | 72 任务 Recovery Benchmark、六模板真实 Solver 测试和 90 任务 MCP transport 测试 | 大规模 OR 实例、远程 MCP、指标面板和消融实验 |

## 9. 分阶段交付

1. **Agent Runtime**：LangGraph 状态、真实节点事件、轨迹可视化与持久化。
2. **Verifiable Environment**：独立复算约束和目标，并区分可行性验证与求解器最优性证明。
3. **Trajectory Infrastructure**：episode store、失败分类、回放与 transition 数据导出。
4. **Policy Baselines**：规则、LLM tool calling、ReAct 和 behavior cloning。
5. **RL Experiment**：offline RL 或 contextual bandit，用于高层工具路由和恢复。
6. **Research Evaluation**：跨任务基准、扰动测试、消融实验和可复现实验报告。

## 10. 预期研究贡献

- 一个面向运筹建模与求解的、MCP 原生的 Agent 环境；
- 一个包含工具调用、求解器反馈、验证信号和失败恢复的 trajectory 数据集；
- 一套以数学可验证性为核心，而非只依赖 LLM judge 的 reward 与评测方法；
- 一个比较固定工作流、通用 LLM policy、模仿学习与 RL policy 的可复现实验框架；
- 对“何时应该学习 Agent policy、何时应该保留确定性优化组件”的工程与研究结论。
- 一套使用 SFT 与 verifier-guided GRPO 后训练开源模型，并与外部 Agent policy 组合评测的方法。

## 11. 项目定位边界

OptiAgent 不把 RL 作为装饰性标签。项目现在已经具备 episode schema、reward、BC + Masked Double DQN、对照基线、独立 test split 和真实 MCP stdio 故障轨迹，因此可以声明已实现第一个 **transport-aware learned recovery policy**。但在学习策略接入主 LangGraph、覆盖 LLM 建模动作并通过更大规模未见 OR 实例之前，不宣称已实现“通用 RL Optimization Agent”。
