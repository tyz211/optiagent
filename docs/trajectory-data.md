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

## 未来的模型后训练视图

当前 transition 主要服务高层 Agent policy。为了支持云端开源模型的 SFT 与 GRPO，后续需要从同一 episode 生成第二种 model rollout view：

```json
{
  "task_id": "stable-task-id",
  "dataset_version": "dataset-version",
  "agent_policy_version": "agent-policy-version",
  "model_version": "base-or-checkpoint-version",
  "messages": [],
  "generated_output": {},
  "tool_observations": [],
  "verifier_report": {},
  "reward": 0.0,
  "reward_components": {},
  "human_correction": null
}
```

SFT 视图只保留经过 Schema、Solver、Verifier 或人工确认的高质量目标输出；GRPO 视图需要保留同一 prompt 的多个候选及其可复算奖励。原始 API Key、授权头和其他运行时密钥不得进入任一训练视图。

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
- Web 批量接口返回 JSON；独立导出脚本已提供按实例分组的 JSONL、数据版本清单与文件摘要，尚未提供 Parquet。
- 当前 reward 是确定性稀疏终局奖励，尚未加入 token、MCP 调用次数、延迟和求解器成本。

### 2026-09-17 恢复策略补充

主流程恢复节点已支持加载 v1/v2 学习策略。`agent_steps.action` 在推理完成后写入实际动作，执行开始时只保存 `pending` 标记，避免将预先计算的规则动作误当作网络动作。未完成的决策节点不进入 `decision_transitions`。

每条 decision transition 额外提供 `policy_observation`、`next_policy_observation` 和实际 `policy` 名称/版本。对应状态也保存在 `state.recovery_observation`，包含固定顺序的候选动作和 mask；决策 metadata 记录 checkpoint SHA-256、训练阶段及异常回退原因。历史记录缺少这些字段时返回 `null`，不能默认将缺失状态当作零向量进行离线训练。

主流程奖励仍为原有确定性终局奖励；状态中的成本估计用于在线推理，不意味着当前轨迹奖励已经与成本感知训练环境完全对齐。开展生产轨迹 offline RL 前需要显式版本化奖励合同并处理动作成本与终止状态。

### 实例分组与离线导出（2026-09-17 第二阶段）

新轨迹记录 `trajectory_source`，真实应用默认为 `live_workflow`，受控评测必须显式使用 `controlled_workflow_benchmark`；旧记录没有来源时标记为 `unknown_legacy`，不自动升级为真实样本。训练视图同时提供 `outcome` 和失败验证次数，区分直接成功、恢复成功、失败终止和执行异常。episode 状态为 `completed` 只表示图执行结束，不代表求解成功。

实例指纹版本为 `instance-content-v1`。指纹来自模板与实际数据内容，统一字典键顺序、表行顺序和 `1`/`1.0` 等值表示；矩阵与标量序列保留顺序。主流程支持内联 JSON 和已登记的仓库三表数据。尚未取得可靠实例内容的上传文件轨迹不能凭文件名、数据库 ID 或问题文本猜测分组。

`scripts/export_workflow_dataset.py` 在 SQLite 只读快照中导出指定用户范围内的全部 episode，不迁移数据库，也不受 Web 列表的 500 条上限限制。默认只读取匿名用户；登录用户需显式指定 `--user-id`。受控评测通过 `--evaluation-report` 输入，输出独立来源的数据集。

```bash
# 输出目录必须不存在，避免覆盖历史数据集。
.venv/bin/python scripts/export_workflow_dataset.py \
  --db data/optiagent.sqlite3 --user-id 1 \
  --output-dir artifacts/rl/datasets/live_v1 --seed 56

# 从完整的受控评测报告导出，不将这些轨迹标记为生产样本。
.venv/bin/python scripts/export_workflow_dataset.py \
  --evaluation-report artifacts/rl/workflow_evaluations/example.json \
  --output-dir artifacts/rl/datasets/controlled_v1 --seed 56
```

输出包括 `train.jsonl`、`validation.jsonl`、`test.jsonl`、`quarantine.jsonl` 和 `manifest.json`。数据合同为 `workflow-recovery-offline-v1`，对应 35 维状态编码与 `workflow-sparse-terminal-v1` 奖励。每条 JSONL 是一个完整 episode，含实际动作、状态、下一状态、done、行为策略版本及逐步奖励；终止后的下一状态为 `null`。

质量检查拒绝未结束 episode、缺失内容指纹、缺失 observation、非法 mask、非有限数值、不连续状态链和错误终局奖励。已执行 `retry_solver` 后工具抛异常的 episode 会保留该动作、`done=true` 和终局负奖励；首次工具失败前尚未产生恢复动作的记录只能进入隔离清单，不能伪造 transition。

导出采用字段白名单：不包含问题原文、文件路径、原始异常文本、密钥或完整工具返回；错误与违反约束列表仅保留数量。自动测试确认导出前后的编码向量相同。隔离文件仅保存 episode 哈希引用与预定义原因码。

分组由 `SHA256(seed + instance_fingerprint)` 固定映射到 70%/15%/15% 区间。增加新 episode 或改变读取顺序不会移动已有实例，同一实例的重复运行、不同策略和故障场景始终属于同一集合。比例是期望值，小数据集可能出现空集合；manifest 用 `all_splits_nonempty` 明确报告，不会为了凑比例拆散实例。

当前导出合同保留应用原有稀疏终局奖励，尚未将其变换为 Gateway 训练中的动作成本奖励；未记录行为概率，也不提供离线策略价值估计。实例内容指纹不能识别所有变量重命名同构、默认字段等价和语义重复。这些边界需要在开展生产轨迹 offline RL 前继续处理。

### 离线训练消费合同

`offline_dataset.load_offline_dataset()` 会重新核验四个 JSONL 文件的字节数和 SHA-256、版本与来源、逐条状态合同、episode 唯一性、内容哈希分组、跨集合重叠和 manifest 计数。train/validation/test 任一为空都会拒绝训练。文件摘要用于检查与 manifest 的一致性，不是可信签名；只有来源可靠的 manifest 才能作为数据追溯依据。

消费端采用两种显式奖励模式，始终不修改源数据：

- `sparse`：保持 `workflow-sparse-terminal-v1` 的逐步奖励；
- `verified_cost`：使用 `workflow-verified-cost-v1`，非终局的基础奖励为 0，验证成功终局为 +1，失败终局为 -1，再对该步 `retry_solver` 或 `rebuild_model` 扣除状态中记录的估计成本。即使恢复动作后工具抛错，也保留该动作成本和失败终局惩罚。

后者避免将失败响应的格式奖励解释成任务成功。它是新的训练目标，不等同于 Gateway 环境的完整奖励：例如提前终止与预算耗尽均为 -1，不使用 Gateway 中分别定义的惩罚。训练报告和 checkpoint 同时保存源奖励版本与训练奖励版本。

终止 transition 的 `next_state=null` 只在张量存储时使用占位值；TD 更新只对非终止样本读取下一状态网络和动作 mask。验证集选模依据日志动作的折扣回报拟合误差，测试集仅在最终报告中使用；此误差不是新策略价值估计。当前模型没有通过行为概率或反事实数据估计离线部署收益，仍需独立执行测试。
