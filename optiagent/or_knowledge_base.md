## 通用 ProblemSpec 建模流程
类别：modeling
通用运筹优化 Agent 应先把自然语言问题转为 ProblemSpec，再决定是否求解。ProblemSpec 至少包含问题类型、目标、集合、参数、决策变量、约束、推荐求解器、数据要求和输出结构。这样可以避免 LLM 直接生成不稳定模型，也便于对不同问题复用模板。

## 仓库选址 MILP 建模模板
类别：modeling
仓库选址适合建成 MILP：连续变量 x[w,c] 表示仓库到客户的发货量，二进制变量 y[w] 表示仓库是否启用。目标是最小化运输成本与固定启用成本。关键约束包括客户需求满足、仓库容量联动、未启用仓库不能发货、强制启用或关闭。

## 运输分配 LP 建模模板
类别：modeling
运输问题不考虑固定启用成本时通常是 LP。决策变量 x[s,d] 表示供给点 s 向需求点 d 的运输量，目标最小化单位运输成本乘以运输量。约束包括需求满足、供给能力和非负运输量。它适合做仓库选址的 LP 对照或已知仓库网络下的日常调拨。

## 0-1 背包 IP 建模模板
类别：modeling
背包问题用于在容量、预算、工时等限制下选择一组对象。二进制变量 z[i] 表示是否选择物品 i，目标最大化价值或收益，约束是总资源消耗不超过容量。常见业务场景包括项目组合选择、营销活动选择、采购清单筛选。

## 指派匹配 MILP 建模模板
类别：modeling
指派问题用于把资源分配给任务，例如人员排班、机器任务匹配、订单到拣货员分配。二进制变量 a[r,t] 表示资源 r 是否承担任务 t，目标最小化成本或最大化收益。约束包括每个任务被覆盖、资源容量限制、技能资格限制和公平性限制。

## 车辆路径 VRP 建模模板
类别：modeling
车辆路径问题关注车辆访问客户节点的顺序，常见约束包括车辆容量、时间窗、服务时长、路线连续性和回仓要求。对于 VRP/VRPTW，OR-Tools Routing 通常比手写 MILP 更适合作为第一选择；若要做严谨全局最优证明，可再考虑 MILP 或分解算法。

## 旅行商 TSP 建模模板
类别：modeling
旅行商问题要求从起点出发访问每个节点一次并回到起点，目标最小化总距离、时间或成本。典型模型包含节点集合、距离矩阵和路线决策，关键约束是每个节点恰好进入一次、恰好离开一次并消除子回路。小规模 TSP 可以精确枚举或使用 MILP，路径类业务通常也可使用 OR-Tools Routing。

## 作业车间调度 CP-SAT 建模模板
类别：modeling
作业车间调度问题需要安排每个作业的多道工序。每道工序有指定机器、加工时长和先后顺序，目标通常是最小化最大完工时间 makespan。核心约束包括同一作业内工序先后关系、同一机器上工序不能重叠，以及开始时间、结束时间和加工时长之间的一致性。CP-SAT 的 interval variable 与 NoOverlap 适合表达这类问题。

## 产品组合 MILP 建模模板
类别：modeling
产品组合或生产计划问题用于在原料、工时、预算、设备能力等资源约束下决定各产品产量，使利润或收益最大。决策变量 q[p] 表示产品 p 的产量，可为连续变量或整数变量；目标最大化 sum profit[p] * q[p]。关键约束是每类资源总消耗不超过容量，并可加入最小产量、最大需求、是否投产等 0-1 变量。

## 仓库选址数据 Schema
类别：schema
仓库选址需要 warehouses、customers、costs 三类数据。warehouses 至少包含 warehouse、capacity、fixed_cost；customers 至少包含 customer、demand；costs 至少包含 warehouse、customer、cost，并且需要覆盖每个仓库和客户的组合。可选字段包括 region、min_open_ratio、force_open、force_closed。

## 背包数据 Schema
类别：schema
背包问题需要 items 表，至少包含 item、value、weight。若业务是预算约束，可以把 weight 替换或扩展为 cost、budget_usage、labor_hours 等资源消耗字段。若有多资源约束，需要为每种资源建立一个容量参数。

## 指派数据 Schema
类别：schema
指派问题通常需要 resources、tasks、costs 三类数据。resources 表描述员工、机器或车辆等资源；tasks 表描述待分配任务；costs 表描述 resource 到 task 的匹配成本或收益。若存在技能资格，需要 eligibility 或 skill 字段。

## VRP 数据 Schema
类别：schema
VRP 通常需要 locations、vehicles、distance_matrix 或 time_matrix。locations 包含节点、经纬度、需求、服务时长和时间窗；vehicles 包含车辆、容量、起终点；矩阵数据包含节点之间的距离或行驶时间。

## TSP 数据 Schema
类别：schema
TSP 通常需要 distances 表或 distance_matrix。distances 表至少包含 from、to、distance 三列，表示节点之间的距离、行驶时间或成本。若提供的是无向距离，系统可以补齐反向边；若矩阵不完整，需要提示缺失边或使用大惩罚值避免选择不可达路径。

## 调度数据 Schema
类别：schema
作业车间调度需要 tasks 表，至少包含 job、machine、duration，可选包含 order。job 表示作业或订单，machine 表示加工机器，duration 表示加工时长，order 表示同一作业内的工序顺序。若没有 order，可按上传顺序作为默认工序顺序。

## 产品组合数据 Schema
类别：schema
产品组合问题需要 products 和 capacities。products 至少包含 product、profit 以及一个或多个资源消耗列，例如 labor、material、machine_hours；capacities 给出每个资源的可用上限。可选字段包括 min_qty、max_qty 和 integer，用于表达最低产量、需求上限或整数产量。

## Gurobi 代码模板策略
类别：template
Gurobi 模板适合 LP、MILP、QP 等数学规划。推荐代码结构是：准备集合和参数、创建 Model、添加变量、设置目标、添加约束、设置 TimeLimit 和 OutputFlag、求解、读取变量值、生成结构化结果。对于不可行模型，应保留计算 IIS 的扩展点。

## OR-Tools 代码模板策略
类别：template
OR-Tools 适合 CP-SAT、Routing 和组合优化。排班、指派、逻辑约束较多的问题可优先 CP-SAT；车辆路径与时间窗配送可优先 RoutingModel。模板应包含数据管理器、约束注册、搜索参数、求解状态和路线或排班结果解析。

## TSP 求解策略
类别：solver
TSP 小规模实例可以使用精确枚举、动态规划或 MILP 来证明最优；中大规模实例通常使用最近邻、2-opt、局部搜索或 OR-Tools Routing 等启发式。Agent 输出时应说明是否为精确最优、可行启发式，还是受 time limit 影响的近似解。

## 调度求解策略
类别：solver
作业车间调度可使用 CP-SAT、MILP 或启发式列表调度。CP-SAT 能表达机器互斥和工序先后约束并尝试证明最优；启发式调度可以快速生成可行方案，但不保证全局最优。Agent 应在结果中标注 OPTIMAL 或 FEASIBLE 的区别。

## 产品组合求解策略
类别：solver
产品组合如果变量连续就是 LP，如果要求整数产量、投产启停或批量约束就是 MILP。Gurobi 适合求解这类资源约束利润最大化问题。结果解释应包括最优产量、总利润、资源使用量和剩余容量。

## 求解器选择经验
类别：solver
LP 或连续运输分配可用 SciPy HiGHS、Gurobi 或 OR-Tools 线性求解；带固定成本、启停、批量约束的 MILP 优先 Gurobi；路径规划、时间窗配送优先 OR-Tools Routing；逻辑约束多、排班类问题优先 OR-Tools CP-SAT；大规模企业模型可用 Gurobi 并设置 time limit、MIPGap、warm start。

## 不可行模型诊断策略
类别：solver
如果模型不可行，Agent 应先检查数据完整性、供需平衡、容量是否足够、强制启用与强制关闭是否冲突、上下界是否矛盾。Gurobi 可通过 IIS 找出冲突约束。业务解释中应避免只说 infeasible，而要指出最可能的约束冲突和可修复动作。

## 多 Agent 协作职责
类别：modeling
通用 OR Agent 可拆成需求解析 Agent、数据校验 Agent、建模 Agent、求解 Agent、结果解释 Agent 和风控诊断 Agent。Supervisor 负责根据问题路由到 RAG、数据库、模板库、求解器或 MCP 工具。RAG 应贯穿问题分类、Schema 检查、模板选择、求解器选择和结果解释。
