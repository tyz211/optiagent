# OptiAgent

> 一个面向运筹优化场景的本地 Agent 系统：支持自然语言提问、CSV/JSON 数据输入、RAG 建模知识检索、工具调用与优化求解。

OptiAgent 不是一个只会“聊天”的大模型壳子，也不是一个只能跑单一模型的运筹脚本。  
它尝试把 **自然语言理解、结构化建模、RAG、工具路由、求解器执行、结果解释** 串成一条完整闭环，让用户可以像和分析助手对话一样提出优化问题，并得到可审计、可解释、可执行的求解结果。

## 为什么这个项目值得看

- 支持自然语言驱动的运筹优化求解，而不要求用户先写数学模型。
- 支持多类问题模板，而不仅仅是单一仓库选址。
- 支持上传 CSV，并自动识别数据角色、标准化字段、校验可行性。
- 使用本地 RAG 提供建模依据、数据 Schema、求解策略和代码模板。
- 配置 LLM 后可进行更柔性的路由与工具调度；未配置时也能本地规则兜底。
- 结果不仅给出答案，还展示问题类型、RAG 依据、Agent 步骤、决策表和风险提示。

## 架构图

![通用运筹优化 Agent 技术架构图](assets/architecture.png)

## 项目定位

当前项目更准确的定位是：

**模板化、可扩展、RAG 增强的通用运筹优化 Agent 原型**

它已经具备完整系统闭环：

- 自然语言识别问题类型
- 生成标准化 `ProblemSpec`
- 检索本地运筹知识库
- 调用对应求解器或工具
- 返回结构化结果与解释
- 支持会话、文件、数据集隔离

但它还不是“任意运筹问题都能自动建模并可靠求解”的开放域优化编译器。对没有活跃求解器的场景，系统会给出建模建议、数据缺口和扩展方向，而不是伪造答案。

## 当前能力

### 1. 问题理解与建模

- 将自然语言问题转换为 `ProblemSpec`
- 输出目标函数、变量、约束、数据要求和推荐求解器
- 支持 LLM 路由与本地规则路由双模式

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
- 对真实数据类问题支持 Web Research 证据检索
- 支持 MCP 外部工具接入

### 4. 数据与对话管理

- 支持 CSV 上传、完整内容保存和预览
- 支持按会话隔离上传文件、结构化数据集与运行记录
- 支持多轮追问，不同问题文件不会相互污染

### 5. 结果展示

- 结构化结论
- 指标卡片
- 决策表
- 风险提示
- RAG 命中文档
- Agent 工具调用轨迹

## 已支持的可执行问题

| 模板 | `template_id` | 数据入口 | 求解方式 | 结果状态 |
| --- | --- | --- | --- | --- |
| 仓库选址与客户分配 | `facility_location` | 三个 CSV：`warehouses/customers/costs` | Gurobi MILP | `OPTIMAL` 或 Gurobi 状态 |
| 0-1 背包 | `knapsack` | JSON 或 CSV：`item/value/weight` | Gurobi IP | `OPTIMAL` |
| 指派匹配 | `assignment` | JSON 或 CSV：`resource/task/cost` | Gurobi MILP | `OPTIMAL` |
| 旅行商路径 | `tsp` | JSON 或 CSV：`from/to/distance`；或坐标 CSV：`City/X/Y` | 精确枚举 / Held-Karp / 最近邻启发式 | `OPTIMAL` 或 `FEASIBLE` |
| 作业车间调度 | `job_shop_scheduling` | JSON 或 CSV：`job/machine/duration/order` | 列表调度启发式 | `FEASIBLE` |
| 产品组合与生产计划 | `production_mix` | JSON 或 CSV：`product/profit/资源列 + capacities` | Gurobi LP/MILP | `OPTIMAL` |

说明：运输分配、VRP/VRPTW 等内容目前保留在 RAG 知识库中作为建模参考，还不是活跃自动求解模板。

## 系统如何工作

```text
用户问题 / 上传数据
  -> LLM 路由器 或 本地规则路由器
  -> ProblemSpec 结构化建模
  -> RAG 检索
  -> 数据解析与校验
  -> 求解器执行
  -> 最优性检查
  -> 结构化结果输出
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

这意味着：

- 文件名不标准不一定有问题，系统更依赖列名和字段结构
- 上传文件属于任务数据输入，不属于 RAG 知识库内容
- 上传阶段不调用 LLM 直接求解

## RAG 与 MCP 在项目中的角色

### RAG

RAG 用于补充“知识上下文”，而不是替代求解器。

它负责：

- 帮助问题识别与模板选择
- 给出每类问题的数据 Schema
- 给出变量、目标、约束解释
- 给出 Gurobi / 启发式 / 精确算法的求解策略说明

### MCP

MCP 用于扩展“外部工具能力”。

它允许将外部工具与内置工具一起交给 Agent 调用，而不是把所有能力写死在项目内部。当前项目已经预留了 MCP JSON 配置入口，可通过 `langchain-mcp-adapters` 加载外部工具。

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

## 快速开始

推荐使用启动脚本：

```bash
./start.sh
```

指定端口：

```bash
PORT=8010 ./start.sh
```

首次运行：

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

手动启动：

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

浏览器打开：

```text
http://127.0.0.1:8000
```

## 运行依赖

- Python 3.11+
- FastAPI / Uvicorn
- pandas / numpy
- gurobipy
- requests
- langchain / langchain-openai / langchain-mcp-adapters
- SQLite

如果本机没有有效 Gurobi license，相关模板会返回不可用状态，不会退化成“LLM 猜答案”。

## 项目结构

```text
api/
  main.py                  FastAPI 路由、上传、配置入口
  database.py              SQLite 持久化
  services/ask_service.py  提问编排、RAG、数据解析、工具调用响应

optiagent/
  problem_spec.py          ProblemSpec 数据结构
  templates/registry.py    问题模板与自动识别
  solver_registry.py       通用求解器注册表
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

页面中可填写 OpenAI-compatible Chat Completions 配置：

- Base URL
- 模型名
- API Key
- Temperature

MCP 配置示例：

```json
{
  "math": {
    "command": "python",
    "args": ["server.py"],
    "transport": "stdio"
  }
}
```

未配置 LLM 时，系统仍可运行本地 ProblemSpec、RAG、数据解析和求解器调用链路。

## 验证命令

```bash
python3 -m compileall api optiagent
node --check web/app.js
```

## 已知边界

- 当前是模板化通用 Agent，不是开放域自动建模编译器
- 登录是轻量 session token 方案，不是完整权限系统
- LLM API Key 以明文保存在本地 SQLite，适合本机实验，不适合直接用于多用户生产部署
- 作业车间调度当前返回 `FEASIBLE`，不承诺全局最优
- 超大规模 TSP 会回退为启发式可行解
- VRP/VRPTW、复杂运输网络、多资源调度仍处于后续扩展阶段

## 后续方向

- 增加 VRP / VRPTW 求解适配器
- 增加 infeasible 诊断与 IIS 解释
- 增加更多通用模板和更强的数据映射能力
- 增加更严格的密钥管理和用户隔离
- 接入向量数据库或企业知识库
- 增加 benchmark，评估识别准确率、求解成功率、求解时间和解释质量

## 适合的使用场景

- 运筹优化课程项目 / 毕业设计
- 智能决策系统原型
- 供应链优化演示平台
- LLM + OR Agent 研究型项目
- 个人简历或作品集展示项目>>>>>>> origin/main
