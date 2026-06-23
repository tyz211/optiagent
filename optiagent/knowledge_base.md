## 仓库选址业务规则
仓库选址问题需要同时考虑运输成本和仓库固定启用成本。一个仓库即使运输距离较近，也可能因为固定成本较高而不被启用；反过来，固定成本较低的仓库可能承担更多中转任务。若业务要求某仓必须保留，可以使用 force_open；若某仓停业、检修或受灾，可以使用 force_closed。

## MILP 模型说明
仓库选址模型使用连续运输变量 x[i,j] 和二进制启用变量 y[i]。目标函数最小化 sum(cost[i,j] * x[i,j]) + sum(fixed_cost[i] * y[i])。客户需求约束要求每个客户都被满足；容量联动约束要求 sum_j x[i,j] <= capacity[i] * y[i]，从而保证未启用仓库不能发货。

## Gurobi 求解器说明
Gurobi 适合求解 LP、MILP、QP 等数学规划问题，尤其适合仓库选址、产能规划、库存调拨、排产排程等带有整数决策的大规模优化场景。它能处理二进制变量、固定成本、强制启停、时间限制和求解状态，并且在工业界认可度很高。

## OR-Tools 求解器说明
OR-Tools 是 Google 开源优化工具包，适合车辆路径 VRP、带时间窗路径 VRPTW、排班、约束规划和中小规模整数优化。若项目进入配送路线层，例如车辆从仓库出发访问多个客户并满足载重与时间窗，应优先考虑 OR-Tools Routing 或 CP-SAT。

## RAG 在本项目中的作用
RAG 负责把业务规则、模型定义、求解器选择经验和解释模板检索出来，辅助 Agent 给出有依据的解释。它不替代优化求解器，而是在解析问题、说明约束、解释结果和建议下一步动作时提供可追溯的知识上下文。

## 多 Agent 协作方向
多 Agent 可以拆成需求解析 Agent、数据校验 Agent、建模 Agent、求解 Agent、结果分析 Agent 和风控解释 Agent。解析 Agent 把自然语言转为结构化场景；建模 Agent 选择 LP、MILP 或 VRP；求解 Agent 调用 Gurobi 或 OR-Tools；分析 Agent 对比方案；风控 Agent 检查不可行、容量瓶颈和业务规则冲突。

## 真实城市参考数据
本项目已在 data/china_city_reference.csv 保存公开中国城市参考数据，来源为 GitHub 仓库 xiaofanliang/intercity_connectivity 的 CN_Public.csv。该数据包含 275 个中国城市的 cityId、中文名、英文名、经纬度、户籍人口、常住人口、GDP、移动连接度和公共连接度等字段。原仓库 README 说明数据整合自中国城市统计年鉴、中国城市建设统计年鉴、腾讯位置数据和公开交通网络数据，并使用 MIT License。当前应用将其作为 RAG 背景知识，不作为前端默认业务数据。

来源链接：https://github.com/xiaofanliang/intercity_connectivity
原始 CSV：https://raw.githubusercontent.com/xiaofanliang/intercity_connectivity/main/Data/CN_Public.csv
