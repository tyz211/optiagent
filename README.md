# OptiAgent

> 一个面向运筹优化场景的 Agentic 系统：学习如何规划建模、选择工具、调用求解器并验证结果。

> A local-first optimization agent for operations research workflows, combining natural language understanding, structured modeling, RAG, solver execution, and explainable results.

![Python](https://img.shields.io/badge/Python-3.14%20tested-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Gurobi](https://img.shields.io/badge/Gurobi-Optimizer-E87722)
![RAG](https://img.shields.io/badge/RAG-Knowledge%20Augmented-4A90E2)
![LangChain](https://img.shields.io/badge/LangChain-Agent%20Tools-1C3C3C)
![SQLite](https://img.shields.io/badge/SQLite-Local%20Storage-003B57?logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

[English](README_EN.md) · [快速开始](#快速开始) · [复现与验收](docs/reproducible-release.md)

当前已支持七类业务模板与显式 LP/MILP、独立验算和多轮数据修订。配置模型后优先使用 LLM 主控，API Key 按账户持久化；主控具备上下文预算、显式记忆、原文数值修改和方案事实引用。模板与 Agent 工具已采用统一注册入口。2026-10-07 完整回归 241 项通过；真实模型任务效果、任意自然语言约束编译和非线性模型仍待验证。详见 [主控运行说明](docs/llm-agent-runtime.md) 和 [实施计划](docs/llm-agent-implementation-plan.md)。

该项目尝试把 **自然语言理解、结构化建模、RAG、工具路由、求解器执行、结果验证与策略学习** 串成一条完整闭环，让用户可以像和分析助手对话一样提出优化问题，并得到可审计、可解释、可执行的求解结果。

项目的长期研究问题是：

$$
\boxed{\text{Learning an Agent Policy for Automated Optimization Modeling and Solving}}
$$

重点是学习一个可评估、可改进的 Agent policy，使系统能在澄清、检索、建模、工具选择、求解、验证和修复之间作出序列决策。完整定义见 [研究方向文档](docs/research-direction.md)。

## 当前使用入口

运行 `./start.sh`，在主对话页载入背包或数学模型示例。运输分配支持中文供需描述、内联 JSON、数值修改和禁运，详见 [运输示例](examples/transportation_sample.md)；显式数学模型输入见 [LP/MILP 示例](examples/linear_program_sample.md)。

运行 `.venv/bin/python scripts/run_llm_agent_demo.py --followup` 可在临时数据库查看求解、原文修改和事实解释三轮演示；模型决策使用模拟响应，本地求解与验算真实执行。

## 目录

- [当前使用入口](#当前使用入口)
- [快速开始](#快速开始)
- [验证命令](#验证命令)
- [Who Is This For](#who-is-this-for)
- [典型使用场景](#典型使用场景)
- [项目的优势](#项目的优势)
- [效果图](#效果图)
- [架构图](#架构图)
- [研究目标](#研究目标)
- [当前能力](#当前能力)
- [已支持的可执行问题](#已支持的可执行问题)
- [系统如何工作](#系统如何工作)
- [CSV 处理策略](#csv-处理策略)
- [工具体系](#工具体系)
- [Examples](#examples)
- [Roadmap](#roadmap)
- [Community](#community)
- [项目结构](#项目结构)
- [数据示例](#数据示例)
- [训练 Recovery Policy](#训练-recovery-policy)

## Who Is This For

- 想做“运筹优化 Agent / OR Copilot / 求解助手”的开发者
- 需要把 LLM、RAG、结构化数据和 MILP/启发式求解串起来的研究者
- 想把自然语言前端接到优化后端的课程项目、毕业设计或原型项目作者
- 希望了解“从前端上传数据到后端建模求解”完整链路的同学

## 典型使用场景

- 用自然语言描述仓库选址、生产计划、作业车间调度、指派、背包、TSP 等问题
- 上传业务 CSV，让系统自动识别数据角色、完成字段标准化并调用求解器
- 借助本地 RAG 为建模、字段要求、求解策略提供解释依据
- 将该仓库作为运筹 Agent 的最小可行原型，继续扩展新模板、新工具、新求解器

## 项目的优势

- 支持自然语言驱动的运筹优化求解，而不要求用户先写数学模型。
- 支持多类问题模板。
- 支持上传 CSV，并自动识别数据角色、标准化字段、校验可行性。
- 使用本地 RAG 提供建模依据、数据 Schema、求解策略和代码模板。
- 配置 LLM 后可进行更柔性的路由与工具调度；未配置时也能本地规则兜底。
- 结果展示问题类型、RAG 依据、Agent 步骤、决策表和风险提示。

## 效果图

![OptiAgent 界面效果图](assets/ui-demo.png)

## 架构图

![通用运筹优化 Agent 技术架构图](assets/architecture.png)

当前执行层使用 LangGraph 编排 `Requirement Analyst → Planner → Data Agent → Modeler → Solver → Verifier → Policy → Explainer`。Requirement Analyst 会跨轮累积目标、约束、数据和未决问题；Policy 可根据验证反馈返回 Modeler/Solver；MCP 层拆分为 Document / Data / Solver 三个可独立运行的服务。详细设计见 [Agent 工作流](docs/agent-workflow.md)、[MCP 架构文档](docs/mcp-architecture.md) 与 [多轮 Agent 路线](docs/multi-turn-agent-roadmap.md)。

## 研究目标

OptiAgent 的长期方向由两条相互闭环的主线组成：第一部分是当前正在开发的 Agent System，学习规划、MCP 工具路由、求解控制和失败恢复；第二部分以本地部署开源小模型为目标，先完善模板、结构化需求解析与评测，再按错误分布安排 SFT 和 GRPO。模型训练地点与本地推理部署解耦。Solver 与 Verifier 为两条主线提供统一、可执行且可复算的环境反馈，需求语义正确性另行评估。

两部分的训练对象不同：当前 BC + Masked Double DQN 学习结构化状态上的高层 Agent 动作；未来 GRPO 学习的是开源语言模型的 token/结构化输出策略。项目早期将交替固定其中一侧进行训练，避免 Agent 与模型同时更新造成非平稳训练。

当前已实现 BC + Masked Double DQN 训练，并将 29/35 维 checkpoint 接入主 LangGraph 的恢复节点，支持动作约束、异常回退与实际决策轨迹导出。40 维 transport-aware 策略仍在独立环境评测，待补齐主流程连接观测后接入。训练使用 validation 选择 checkpoint，单独保留 BC、末轮 DQN 与测试指标；这些小规模受控实验不代表已学会通用优化建模。

离线恢复策略另外提供 BC + Masked CQL。较早的受控实验减少了建模调用；加入持续映射错误后，新策略的可恢复成功率达到规则基线，但没有证明成本优于规则。学习动作目前主要是接受、重试、重建和终止；修复演示中的字段重新绑定由确定性修复器执行。

完整的双主线架构、数据闭环、训练边界与里程碑见 [双主线总体路线](docs/two-track-roadmap.md)。

## 当前能力

### 1. 问题理解与建模

- 将自然语言问题转换为 `ProblemSpec`
- 在求解前将多轮对话合并为结构化 `RequirementBrief`，记录已确认事实、假设和待澄清项
- 当问题类型、目标、约束或数据不完整时主动追问，并阻止误调 Solver
- 输出目标函数、变量、约束、数据要求和推荐求解器
- 支持 LLM 路由与本地规则路由双模式
- 通过 LangGraph 保存八个职责节点及需求澄清、有界恢复回路的执行轨迹
- 记录 Policy 候选动作、action mask、实际动作和恢复原因
- 将每次运行记录为版本化 episode，并提供离线 RL/行为克隆 transition 导出
- 通过六类真实求解实例测试 Gateway、Solver、Schema Validation 和 SolutionVerifier

### 2. RAG 知识增强

- 检索 `optiagent/knowledge_base.md`
- 检索 `optiagent/or_knowledge_base.md`
- 为每次求解提供：
  - 建模知识
  - 数据 Schema
  - 代码模板
  - 求解策略

### 3. 工具调用与求解

- 根据问题模板自动调用对应求解工具
- 对求解结果做最优性/可行性标记
- 对六类模板独立复算约束和目标值，并生成可审计的确定性 reward
- 对真实数据类问题支持 Web Research 证据检索
- 内置 Document MCP、Data MCP 和 Solver MCP，并支持外部 MCP 服务接入
- MCP 服务部分故障时可按服务降级，不会因单个连接失败丢失全部工具

### 4. 数据与对话管理

- 支持 CSV 上传、完整内容保存和预览
- 支持按会话隔离上传文件、结构化数据集与运行记录
- 支持按会话持久化多轮需求状态，补充信息后可继承上一轮目标和约束

### 5. 结果展示

- 结构化结论
- 流式回答输出
- 指标卡片
- 决策表
- 风险提示
- RAG 命中文档
- Agent 工具调用轨迹

## 已支持的可执行问题

| 模板 | `template_id` | 数据入口 | 求解方式 | 结果状态 |
| --- | --- | --- | --- | --- |
| 单商品运输分配 | `transportation` | 受限中文、内联 JSON；Data MCP 可读 JSON 文件 | Gurobi LP + 独立业务验算 | `OPTIMAL`、`FEASIBLE` 或明确失败状态 |
| 显式线性 / 整数规划 | `linear_program` | 数学文本、LaTeX 或规范模型 JSON，无需上传表格 | Gurobi LP/MILP + 独立验算 | `OPTIMAL`、`FEASIBLE` 或明确失败状态 |
| 仓库选址与客户分配 | `facility_location` | 三个 CSV：`warehouses/customers/costs` | Gurobi MILP | `OPTIMAL`、`NEAR_OPTIMAL` 或 Gurobi 状态 |
| 0-1 背包 | `knapsack` | JSON 或 CSV：`item/value/weight` | Gurobi IP | `OPTIMAL` 或 `NEAR_OPTIMAL` |
| 指派匹配 | `assignment` | JSON 或 CSV：`resource/task/cost` | Gurobi MILP | `OPTIMAL` 或 `NEAR_OPTIMAL` |
| 旅行商路径 | `tsp` | JSON 或 CSV：`from/to/distance`；或坐标 CSV：`City/X/Y` | 精确枚举 / Held-Karp / Gurobi MILP / 多起点 2-opt 近似 | `OPTIMAL`、`NEAR_OPTIMAL` 或 `FEASIBLE` |
| 作业车间调度 | `job_shop_scheduling` | JSON 或 CSV：`job/machine/duration/order` | Gurobi MILP / 列表调度兜底 | `OPTIMAL`、`NEAR_OPTIMAL` 或 `FEASIBLE` |
| 产品组合与生产计划 | `production_mix` | JSON 或 CSV：`product/profit/资源列 + capacities` | Gurobi LP/MILP | `OPTIMAL` 或 `NEAR_OPTIMAL` |

说明：运输分配首版支持供给上限、需求恰好满足与显式禁运；VRP/VRPTW 仍仅作为 RAG 建模参考，尚非可执行模板。

### 求解质量策略

- Gurobi 类模型统一使用时间限制和 MIPGap 策略，默认尽量证明 `OPTIMAL`。
- 当大规模 MILP 在时间限制内未完全证明最优但 gap 达到阈值时，系统标记为 `NEAR_OPTIMAL`，并在 Agent 工作过程里说明 gap 与最优性状态。
- TSP 小规模使用精确枚举或 Held-Karp 动态规划；中等规模优先使用 Gurobi MILP 争取证明 `OPTIMAL` 或 `NEAR_OPTIMAL`；更大规模使用多起点最近邻构造加 2-opt 局部搜索，返回高质量可行解并明确未证明全局最优。
- 作业车间调度优先使用 Gurobi MILP 证明最优或接近最优；规模过大或精确求解器不可用时回退到列表调度启发式，并标记为 `FEASIBLE`。
- 结果页会优先展示目标值、关键成本、MIP Gap 和最优性证明状态；Agent 工具调用过程作为可展开审计信息。
- 可通过环境变量调整默认策略：`OPTIAGENT_TIME_LIMIT`、`OPTIAGENT_MIP_GAP`、`OPTIAGENT_SOLVER_THREADS`、`OPTIAGENT_TSP_EXACT_LIMIT`、`OPTIAGENT_TSP_MILP_LIMIT`、`OPTIAGENT_TSP_LOCAL_SEARCH_LIMIT`、`OPTIAGENT_JOB_SHOP_MILP_LIMIT`。

### 流式输出

- 前端默认调用 `/api/ask/stream`，通过 `fetch + ReadableStream` 接收 SSE 事件。
- 后端会推送 `status`、`agent_step`、`answer_delta` 和 `final`：用户可以实时看到八个职责节点的开始、完成或失败状态，最终再渲染完整结构化卡片、决策表和 Agent 轨迹。
- `/api/ask` 保留为非流式兼容接口。

### Trajectory 与训练数据

- `GET /api/agent/episodes`：列出 episode 摘要。
- `GET /api/agent/episodes/{episode_id}`：读取完整审计轨迹。
- `GET /api/agent/episodes/{episode_id}/training`：导出单个 transition 序列。
- `GET /api/agent/training-data`：批量导出完成或失败的训练 episode。
- `OptimizationAgentEnv`：独立于 Web API 的 `reset/step` 恢复策略环境。
- `scripts/generate_rl_dataset.py`：生成可复现 JSONL rollout 数据。

环境定义、动作编号、奖励和 benchmark 指标见 [RL Environment 文档](docs/rl-environment.md)。
真实求解链路的故障矩阵与扩展计划见 [E2E Benchmark 测试计划](docs/e2e-test-plan.md)。
可学习策略的算法选择、状态编码、损失函数和首轮结果见 [Learnable Policy 文档](docs/learning-policy.md)。

## 系统如何工作

```text
用户问题 / 上传数据
  -> LangGraph Agent Policy
     -> Requirement Analyst -> 信息不足：针对性追问
     -> Planner
     -> Data Agent
     -> Modeler
     -> Solver -> MCP Gateway -> Document / Data / Solver MCP
     -> Verifier
     -> Policy -> accept / retry solver / rebuild model / terminate
     -> Explainer
  -> 轨迹与结构化结果持久化
```

对于仓库选址等供应链问题，系统支持：

- 基准场景求解
- `what-if` 修改
- 成本变化解释
- 启用仓库与客户分配展示

## CSV 处理策略

上传 CSV 后，系统不会直接把文件“丢给模型猜”。它会先做结构化处理：

1. 读取 CSV，并兼容 `utf-8-sig / utf-8 / gb18030 / gbk`
2. 根据列名语义和文件内容识别数据角色
3. 保存完整 CSV 和预览到当前会话
4. 对仓库选址三张表执行标准化和校验
5. 如果数据完整，生成结构化数据集并激活求解链路
6. 提问时再由 Agent 按模板解析和调用求解器


## 工具体系

项目内置的核心工具包括：

- `problem_spec_tool`
- `rag_context_pack_tool`
- `generic_optimizer_tool`
- `gurobi_facility_location_tool`
- `rag_search_tool`
- `web_search_tool`
- `data_profile_tool`
- `city_reference_tool`

这些工具主要定义在 [optiagent/langchain_agents.py](optiagent/langchain_agents.py) 中，负责连接自然语言理解、知识检索、数据分析与求解执行。

内置 MCP 另外暴露 11 个可发现工具，覆盖文档读取、知识检索、数据画像、问题数据构建、数据校验、求解器能力发现、统一求解和独立解验证。

## 快速开始

以下为新克隆仓库的规则模式，在 Python 3.14 / macOS arm64 上验收。六类模板的完整求解验收需要当前机器具备可用的 Gurobi 许可；Python、依赖安装包与许可证不包含在仓库中。

```bash
# 已有本地项目时跳过克隆与切换目录。
git clone https://github.com/tyz211/optiagent.git
cd optiagent

# 创建独立环境并安装固定版本，无需激活环境或安装 PyTorch。
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-demo.lock
.venv/bin/python scripts/verify_demo.py --profile core
./start.sh
```

打开终端显示的地址（默认 `http://127.0.0.1:8000`，端口占用时自动顺延）。主页面新建对话后点击「载入背包演示数据」并发送；`/repair-demo` 可体验规则恢复。没有配置权重时，学习策略选项会禁用。

常用启动方式：

```bash
# 指定端口；默认单进程、不启用热重载。
PORT=8010 ./start.sh

# 开发时显式启用热重载。
RELOAD=1 ./start.sh

# 手动启动也使用项目环境中的解释器。
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

开发环境可使用 `requirements.txt` 的宽版本范围，但不代表已通过固定版本验收。完整学习策略演示的权重来源与安装步骤见 [可复现交付说明](docs/reproducible-release.md)。

## Examples

可直接体验的示例放在 [examples/README.md](examples/README.md)：

- 仓库选址：使用 `data/facility_location_*.csv`
- 指派问题：使用 [examples/assignment_sample.json](examples/assignment_sample.json)
- 作业车间调度：使用 [examples/job_shop_scheduling_sample.json](examples/job_shop_scheduling_sample.json)
- 产品组合：使用 [examples/production_mix_sample.json](examples/production_mix_sample.json)

如果你是第一次了解这个项目，建议先从 `facility_location` 或 `assignment` 开始，最容易看到完整的上传、建模、求解与结果展示链路。


## Community

- 仓库变更记录见 [CHANGELOG.md](CHANGELOG.md)
- 贡献方式见 [CONTRIBUTING.md](CONTRIBUTING.md)
- 如果你也在做 OR Agent、Optimization Copilot、Decision Intelligence 或 Solver + LLM 结合的方向，欢迎基于这个仓库继续扩展

## 运行依赖

- 交付验收：Python 3.14 / macOS arm64；其他环境需另行验证
- FastAPI / Uvicorn
- pandas / numpy
- gurobipy
- requests
- langchain / langchain-openai / langchain-mcp-adapters / MCP Python SDK
- openpyxl / xlrd / pypdf / python-docx（Document/Data MCP 文件解析）
- SQLite

如果本机没有有效 Gurobi license，依赖它的模板无法完成求解；TSP 与调度的部分路径有独立算法或启发式回退。这不等同于六类模板的完整验收通过。

## 项目结构

```text
api/
  main.py                  FastAPI 路由、上传、配置入口
  database.py              SQLite 持久化
  services/agent_workflow.py  LangGraph 条件状态图、恢复回路与运行事件
  services/llm_controller.py  LLM 决策、预算、审计和终止处理
  services/agent_tool_registry.py  Agent 工具定义与执行分派
  services/agent_tool_catalog.py   内置工具注册与前置条件
  services/agent_tools.py     本地工具处理函数与本轮执行上下文
  services/agent_decision.py  共享行动合同与工具参数模型
  services/ask_service.py  提问编排、RAG、数据解析、工具调用响应

optiagent/
  mcp_contracts.py         ProblemEnvelope / SolveEnvelope 版本化合同
  mcp_validation.py        Data / Solver MCP 共享的确定性数据校验
  solution_verifier.py     模板决策的约束与目标值独立复算
  template_extensions.py   模板说明、能力、校验、求解和验算的统一注册入口
  mcp_client.py            内置与外部 MCP 发现、前缀和降级
  optimization_gateway.py  本地与 MCP 共用的唯一合同化求解入口
  mcp_servers/             Document / Data / Solver MCP 服务
  problem_spec.py          ProblemSpec 数据结构
  templates/registry.py    模板查询与自动识别的兼容入口
  templates/definitions.py 内置模板元数据与 ProblemSpec 构造
  templates/builtins.py    内置完整模板扩展的组装
  templates/validators.py  各模板的数据校验实现
  solver_registry.py       通用求解适配器查询的兼容入口
  generic_solvers.py       背包、指派、TSP、调度、产品组合求解器
  solver.py                仓库选址 Gurobi MILP
  rag.py                   本地 Markdown RAG 检索
  langchain_agents.py      LangChain Supervisor Agent 与 MCP 工具加载
  scenario.py              what-if 场景修改与结果解释
  llm.py                   OpenAI-compatible Chat Completions 调用
  data.py                  数据规范化和校验
  web_research.py          网页检索

web/
  index.html
  app.js
  styles.css

data/
  01_knapsack_data.csv
  tsp.csv
  china_city_reference.csv
```

本地文件管理：`__pycache__/`、`.pytest_cache/` 和 `.DS_Store` 是可再生缓存，可清理；`.venv/` 是运行环境，`.idea/` 是本地 IDE 配置。`data/optiagent.sqlite3` 保存会话与配置，`artifacts/rl/` 保存训练数据、权重和实验记录，均应按实际用途保留。2026-09-27 交付版的原始 core/full 验收报告统一保存在 `artifacts/releases/demo-2026.09.27/validation/`，已清理 `artifacts/validation/` 下逐文件校验一致的同版副本；其他验收记录继续保留。忽略文件不等于无用文件。

## 数据示例

### 0-1 背包 CSV

```csv
物品编号,重量,价值
1,2,3
2,3,4
3,4,8
4,5,8
5,9,10
6,7,6
```

提问：

```text
有一个容量为 15 的背包，每个物品最多选一次，求最大价值。
```

### 指派 JSON

```json
{
  "resources": ["员工A", "员工B"],
  "tasks": ["早班", "晚班"],
  "costs": [
    {"resource": "员工A", "task": "早班", "cost": 3},
    {"resource": "员工A", "task": "晚班", "cost": 8},
    {"resource": "员工B", "task": "早班", "cost": 5},
    {"resource": "员工B", "task": "晚班", "cost": 4}
  ]
}
```

### TSP JSON

```json
{
  "distances": [
    {"from": "A", "to": "B", "distance": 4},
    {"from": "A", "to": "C", "distance": 2},
    {"from": "B", "to": "C", "distance": 5},
    {"from": "B", "to": "D", "distance": 10},
    {"from": "C", "to": "D", "distance": 3},
    {"from": "A", "to": "D", "distance": 7}
  ]
}
```

### 作业车间调度 JSON

```json
{
  "tasks": [
    {"job": "J1", "machine": "M1", "duration": 3, "order": 1},
    {"job": "J1", "machine": "M2", "duration": 2, "order": 2},
    {"job": "J2", "machine": "M2", "duration": 2, "order": 1},
    {"job": "J2", "machine": "M1", "duration": 4, "order": 2}
  ]
}
```

### 产品组合 JSON

```json
{
  "products": [
    {"product": "A", "profit": 30, "labor": 2, "material": 1},
    {"product": "B", "profit": 40, "labor": 1, "material": 3}
  ],
  "capacities": {
    "labor": 100,
    "material": 90
  }
}
```

### 仓库选址 CSV

`warehouses.csv`

```csv
warehouse,region,capacity,fixed_cost,min_open_ratio,force_open,force_closed
Shanghai,华东,1200,3600,0.20,0,0
Beijing,华北,900,2600,0.15,0,0
```

`customers.csv`

```csv
customer,demand
Hangzhou,420
Nanjing,360
```

`costs.csv`

```csv
warehouse,customer,cost
Shanghai,Hangzhou,2.1
Shanghai,Nanjing,2.5
Beijing,Hangzhou,4.6
Beijing,Nanjing,4.2
```

## LLM 与 MCP 配置

登录后保存模型配置，API Key 会按账户长期保存在项目数据库中。刷新或重新登录会恢复模型设置，并显示密钥已保存；编辑时密钥留空保留旧值，输入新值才替换。配置后主对话默认优先由 LLM 驱动整个 Agent 的行动规划，求解与验算仍由工具执行；需要旧工作流时可显式指定 `agent_mode=legacy`。

页面中可填写 OpenAI-compatible Chat Completions 配置：

- Base URL
- 模型名
- API Key
- Temperature

MCP 配置留空时，启用 LLM 的 Agent 会使用当前 Python 环境自动发现内置 Document / Data / Solver MCP。合并外部服务的配置示例：

```json
{
  "include_builtin": true,
  "servers": {
    "external_data": {
      "transport": "streamable_http",
      "url": "https://example.com/mcp"
    }
  }
}
```

旧版 MultiServerMCPClient 裸 JSON 配置仍可使用，并会与内置服务合并。未配置 LLM 时，系统仍可运行本地 ProblemSpec、RAG、数据解析和求解器调用链路，但不会触发 LLM 的 MCP 工具路由。

## 验证命令

```bash
# 新克隆仓库的基础验收，不依赖 PyTorch 或权重。
.venv/bin/python scripts/verify_demo.py --profile core

# 仅在安装 requirements-rl.lock 且两份固定权重齐全后执行。
.venv/bin/python scripts/verify_demo.py --profile full
```

两种模式按实际安装范围选择一个。完整模式要求模型摘要一致并执行全部测试；报告写入新的 `artifacts/validation/` 子目录，不覆盖已有记录或用户数据库。

## 生成 RL baseline 轨迹

```bash
# 从项目根目录生成可重复的基线轨迹。
PYTHONPATH=. .venv/bin/python scripts/generate_rl_dataset.py --output data/rl/baseline.jsonl
```

## 训练 Recovery Policy

### 表格型 Q-learning 基线与续训

GitHub 上游的表格型 Q-learning 已作为独立基线整合，避免与神经网络训练入口冲突：

```bash
# 从头训练表格策略，输出目录必须尚不存在。
.venv/bin/python scripts/train_tabular_recovery_policy.py --output artifacts/rl/tabular-v1
# 延续 Q 表和更新计数；使用新的确定性随机流，不等同于精确恢复中断现场。
.venv/bin/python scripts/train_tabular_recovery_policy.py \
  --resume-from artifacts/rl/tabular-v1 --output artifacts/rl/tabular-v2 --episodes 10000
```

此入口只训练合成环境中的 Q 表，不更新 LLM 权重，也不自动接入线上恢复节点。
历史训练结果见 [上游训练报告](docs/rl-training-results.md)。[09-19 状态记录](docs/project-status-2026-09-19.md) 保留当时的分支与实验情况；其中本地/远端分叉描述是历史状态。

### 神经网络恢复策略

扩大实例覆盖与五种子独立评测的编排入口为 `scripts/run_offline_study.py`。
它在采集前冻结实例清单、模板配额、源码摘要和训练配置，随后采集轨迹、训练并在独立实例上评测。
默认使用 240/120/120 个训练/验证/测试实例，以及 120 个外部测试实例；详细命令、指标与边界见
[离线多种子研究流程](docs/offline-study.md)。

后续已完成实际系数映射修复实验：错误输入真正进入求解器，
再依据原始数据独立验算；持续错误必须重建映射后才能恢复。
训练计划、复现命令与交互回放生成方法见 [修复 Demo 训练计划](docs/repair-demo-training.md)。

现在也可启动现有 Web 服务，进入 `/repair-demo` 或主对话页的「实时修复演示」。页面支持六类问题、五种场景，以及规则 / 上一轮模型 / 新修复模型的现场对比，实时显示映射变化、求解与独立验算，并可下载结果。每次运行使用独立进程和临时数据库；没有 LLM 调用，不写入普通对话历史。模型缺失会明确提示，可选择规则策略体验。启动方法、检查点配置与演示步骤见 [实时修复 Demo](docs/live-repair-demo.md)。

已完成五种子、120 个外部实例的受控评测：验证选中 CQL 的平均建模调用为 1.205 次，
相比规则减少 13.90%、相比 BC 减少 6.76%，可恢复成功率保持 100%。
收益集中在预设的目标不一致恢复场景，不代表通用模型修复或真实线上成本收益。
数据规模、逐种子结果和验收边界见 [完整实验报告](docs/offline-study-results.md)。

后续数据修复实验使用 60 个新的外部实例、五个训练种子：旧模型可恢复成功率为 79.17%，新 CQL 策略达到 100%，与规则持平；两者平均建模调用均为 1.600 次，原成本改善门槛未通过。该实验通过的是修复可靠性门槛，不能把上一轮 13.90% 的调用改善继续套用到它上面。[修复实验结果](docs/repair-demo-training.md)

```bash
# 神经网络训练需可选 RL 依赖；训练输出不覆盖本版固定权重。
.venv/bin/python -m pip install -r requirements-rl.lock
.venv/bin/python scripts/train_recovery_policy.py
```

每次训练都会创建独立的 `run_id`，并在 `artifacts/rl/runs/` 中保存 manifest、checkpoint、完整报告和追加式历史索引；成功和失败运行都不会覆盖旧记录。

训练报告现在包含独立 `bc_policy`、`dqn_final_policy` 和验证集选出的 `learned_policy`。同分保留更早的 checkpoint，因此 `recovery_policy.pt` 可能来自 BC 阶段；请查看 `training.selection.selected_stage`，不要把全部收益归因于 DQN。`bc_policy.pt` 和 `dqn_final_policy.pt` 同时保存以便复核。

将验证过的 v1/v2 checkpoint 用于 Web 主流程：

```bash
# 替换为本次训练输出的实际路径，再启动服务。
export OPTIAGENT_RECOVERY_CHECKPOINT="/absolute/path/to/recovery_policy.pt"
./start.sh
```

未配置时使用规则策略。Python 调用 `run_agent_workflow(..., recovery_checkpoint="")` 可显式选择规则；配置不存在或不兼容的 checkpoint 会报错，运行时推理异常或非法动作则回退规则并记入轨迹。更完整的训练与主流程评测命令见 [学习策略文档](docs/learning-policy.md)。

当前真实恢复数据集已升级为 `gateway-recovery-v2.0`：按实例内容指纹隔离 train/validation/test，同一实例的不同故障场景始终属于同一集合。训练入口通过 `--instances-per-split` 控制每类问题在每个集合中的独立实例数，默认 4，对应 72 个实例、360 个故障任务。缺少实例内容、指纹不匹配或跨集合重复的数据会被拒绝训练。

生产轨迹可按用户范围只读导出，输出独立 JSONL、文件摘要和隔离清单：

```bash
# 默认仅导出匿名用户；登录用户用 --user-id 指定范围，目标目录必须不存在。
.venv/bin/python scripts/export_workflow_dataset.py \
  --db data/optiagent.sqlite3 \
  --output-dir artifacts/rl/datasets/my_live_dataset --seed 56
```

导出文件移除问题原文与错误文本，保留状态编码所需数值、实际动作和稀疏终局奖励。真实运行与受控评测分别标记，不能混为生产失败样本；详细质量门槛见 [轨迹数据合同](docs/trajectory-data.md)。

离线训练入口读取经过校验的固定数据集，执行 BC 预热和带合法动作约束的离散 CQL，不运行环境交互：

```bash
# 默认将奖励转换为验证成功 +1 / 失败 -1，再扣除已记录的恢复动作成本。
.venv/bin/python scripts/train_offline_policy.py \
  --dataset artifacts/rl/datasets/my_dataset \
  --seed 57 --steps 1500 --bc-epochs 160 --cql-alpha 0.1 \
  --reward-mode verified_cost
```

也可用 `--reward-mode sparse` 保留原始稀疏奖励。输出包含 BC、末轮 CQL、验证选中模型、奖励版本、数据摘要及日志诊断；这些诊断不代表新策略成功率。验证和测试集合必须非空，部署前仍需运行独立主流程评测。使用方法与边界见 [学习策略文档](docs/learning-policy.md)。

训练真实 Gateway 成本感知策略：

```bash
# 采集真实求解结果并训练成本感知策略。
.venv/bin/python scripts/train_real_recovery_policy.py --seed 47 --episodes 1200
```

训练真实 MCP stdio 故障恢复策略：

```bash
# 在隔离 MCP 子进程中采集受控故障。
.venv/bin/python scripts/train_mcp_transport_policy.py \
  --seed 52 \
  --episodes 1500 \
  --bc-epochs 180
```

该入口会启动隔离的 MCP 子进程，采集正常、超时、断连和非法结构返回，并把安全摘要、任务集、checkpoint 与评测报告写入独立 run 目录。当前训练不调用 LLM API，也不会把 API Key、Token 或原始错误文本写入训练产物。

## 运行真实端到端 Benchmark

```bash
# 从项目根目录验证六类模板及异常检测。
PYTHONPATH=. .venv/bin/python scripts/run_e2e_benchmark.py --output data/benchmarks/e2e-smoke.json
```

## Roadmap

- 近期优先推进本地小模型部署、可执行模板扩展和结构化需求解析，实施顺序与验收标准见 [本地小模型 Agent 推进计划](docs/local-small-model-roadmap.md)。
- 完善多轮需求 Agent：支持显式修改/删除约束、方案确认、what-if 分支和会话摘要压缩。
- 建立对话 policy 评测集，测量澄清轮数、需求覆盖率、无效工具调用率与最终求解成功率。
- 扩充已有 trajectory 数据集，采集真实用户需求变更和失败—修复记录。
- 在已有 Solution Verifier 上增加新的问题类型、错误机制与验证覆盖。
- 将已实现的 Rule、Random Valid 和 BC + Masked Double DQN 扩展到远程 Streamable HTTP MCP 与生产轨迹。
- 学习高层工具路由与失败恢复策略，先不直接学习求解器内部搜索。
- 加入预算约束下的 solver portfolio routing，联合优化正确率、解质量、延迟和调用成本。
- 扩展 VRP/VRPTW、网络流、员工排班与鲁棒优化任务。
