# OptiAgent Trajectory 数据合同

## 目标

Trajectory store 将一次 Agent 运行从“日志文本”提升为可以审计、回放和训练的结构化 episode。它记录 policy 在每个状态下选择了什么动作、环境返回了什么 observation，以及最终得到什么 reward。

```text
episode
  ├─ step 0: state -> Planner action -> observation
  ├─ step 1: state -> Data action -> observation
  ├─ step 2: state -> Modeler action -> observation
  ├─ step 3: state -> Solver action -> observation
  ├─ step 4: state -> Verifier action -> observation/reward
  ├─ step 5: state + candidates/mask -> Policy action -> route
  └─ step 6: state -> Explainer action -> terminal result
```

## SQLite 表

### `agent_episodes`

每次 `run_agent_workflow()` 创建一条记录：

| 字段 | 含义 |
| --- | --- |
| `episode_id` | 与数据库自增 ID 解耦的稳定 UUID |
| `user_id` / `conversation_id` | 与现有会话隔离规则一致 |
| `run_id` | 成功生成回答后关联原有 `runs` 记录 |
| `policy_name` / `policy_version` | 当前行为策略标识 |
| `status` | `running`、`completed` 或 `failed` |
| `template_id` | Modeler 识别的问题模板 |
| `total_reward` / `reward_json` | 可复算的终局奖励及分量 |
| `result_json` | 完整终局结果 |
| `error` | 失败类型与信息 |

### `agent_steps`

每个节点、每次尝试对应一条记录：

| 字段 | 含义 |
| --- | --- |
| `sequence` / `node_id` / `attempt` | 节点顺序、身份和重试次数 |
| `state_json` | 动作发生前 policy 可见的状态 |
| `action_json` | policy 选择的动作类型、描述和参数 |
| `observation_json` | 节点或工具执行后的结构化观察 |
| `reward_json` | 节点奖励；当前主要记录 Verifier 或失败奖励 |
| `status` | `running`、`completed` 或 `failed` |
| `elapsed_ms` | 节点执行耗时 |
| `error` | 失败节点的异常摘要 |

唯一键 `(episode_id, node_id, attempt)` 为后续条件回路与重试保留空间。

## State snapshot

当前 state snapshot 使用 `schema_version=1.0`，只保留 policy 决策需要的信息：

- 用户问题、当前节点与已完成节点；
- 数据集、上传文件数量、模板偏好和求解意图；
- `data_context` 与 `ProblemSpec`；
- 求解状态、目标值和验证状态；
- 当前 episode ID。

快照不会保存 LLM API key，也不会保存完整 `LLMConfig`。Planner observation 可以保存结构化计划，但密钥只留在运行时对象中。

## Action 与 Observation

基础节点动作空间为：

```text
plan
inspect_data
build_problem_spec
call_solver
verify_solution
route
return_result
```

Policy 节点使用 `accept_solution`、`retry_solver`、`rebuild_model` 和 `terminate` 四个受约束动作。轨迹同时记录 `candidate_ids`、`action_mask`、`selected_action` 与选择原因。每类重试默认最多两次，预算耗尽后必须终止，避免无界循环。

## 失败轨迹

失败运行不会被丢弃：

- 已完成节点保持 `completed`；
- 异常节点记录 `failed`、异常类型和 observation；
- episode 标记为 `failed`；
- 确定性终局 reward 为 `-1.0`。

这类负样本对学习失败恢复策略与训练 reward model 都很重要。

## 查询与训练视图

| API | 输出 |
| --- | --- |
| `GET /api/agent/episodes` | 当前用户的 episode 摘要，可按 `conversation_id` 筛选 |
| `GET /api/agent/episodes/{episode_id}` | 完整 episode 和有序 steps |
| `GET /api/agent/episodes/{episode_id}/training` | 单个标准 transition 序列 |
| `GET /api/agent/training-data` | 完成或失败 episode 的批量训练视图 |

训练 transition 结构为：

```json
{
  "t": 3,
  "node_id": "solver",
  "state": {},
  "action": {"type": "call_solver"},
  "observation": {},
  "next_state": {},
  "reward": 0.0,
  "done": false,
  "status": "completed"
}
```

完整 `transitions` 保留所有环境节点。`decision_transitions` 只保留真正存在候选动作的 Policy 决策，并把终局 reward 归到最后一个决策上，可直接用于行为克隆和离线 RL。

## 生命周期

```text
create_agent_episode(running)
  -> start_agent_step(running)
  -> finish_agent_step(completed/failed)
  -> ...
  -> complete_agent_episode(completed/failed)
```

删除对话或清空运行记录时，相应 episode 与 steps 会同步删除，避免孤立训练数据。

## 当前边界

- 当前候选动作、mask 和条件路由确定性 Recovery Policy 产生，尚无模型 logits 或行为概率。
- 已支持有界建模/求解重试并记录 `attempt`，但尚未加入工具故障注入和动态预算。
- Web 批量接口仍返回 JSON；RL benchmark 脚本已可导出 JSONL，尚未提供 Parquet 和数据集版本清单。
- 当前 reward 是确定性稀疏终局奖励，尚未加入 token、MCP 调用次数、延迟和求解器成本。
