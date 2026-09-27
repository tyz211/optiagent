# OptiAgent

> An agentic operations-research system for learning how to model, route tools, solve, verify, and recover.

![Python](https://img.shields.io/badge/Python-3.14%20tested-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Gurobi](https://img.shields.io/badge/Gurobi-Optimizer-E87722)
![RAG](https://img.shields.io/badge/RAG-Knowledge%20Augmented-4A90E2)
![LangChain](https://img.shields.io/badge/LangChain-Agent%20Tools-1C3C3C)
![SQLite](https://img.shields.io/badge/SQLite-Local%20Storage-003B57?logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

[中文](README.md) · [Quick start](#quick-start) · [Reproduction and acceptance](docs/reproducible-release.md)

**Current stage: an interactive local research demo.** Six template families, independent mathematical verification, multi-turn data revisions and controlled recovery-policy experiments are implemented. General constraint compilation, real-business validation and LLM SFT/GRPO remain future work. The default demo needs no LLM API key.

OptiAgent is evolving into a two-track research platform. The first track learns an Agent policy for planning, MCP tool routing, solving, verification, and recovery. The second track will deploy an open-source language model in the cloud and post-train it with SFT and GRPO for optimization modeling, structured tool use, and verifier-guided repair. Solvers and deterministic verifiers provide the shared executable feedback loop.

## Why This Repo

- It demonstrates how an OR agent can move beyond chat into actual optimization execution.
- It combines CSV/JSON ingestion, template routing, RAG, and optimization solvers in one local-first system.
- It is suitable both as a demo project and as a starting point for people building solver copilots or decision intelligence assistants.

## Who Is This For

- Developers building OR agents, optimization copilots, or decision-support assistants
- Researchers exploring LLM + optimization + RAG pipelines
- Students working on capstone projects, graduation projects, or prototypes involving natural language to optimization workflows
- Engineers who want to understand the full data flow from frontend upload to backend solving

## Demo

The local delivery baseline is **demo-2026.09.27**, validated with Python 3.14 on macOS arm64. Local acceptance passed 97 core tests, 131 full tests, seven dialogue turns and HTTP/SSE repair checks. These are local results, not a GitHub CI status; other platforms have not been validated.

| Distribution | Contents | Acceptance |
| --- | --- | --- |
| Git clone | Source, examples, dependency locks and tests; no weights | `requirements-demo.lock` and `--profile core` |
| Full local archive | Source snapshot and two fixed recovery checkpoints | `requirements-rl.lock` and `--profile full` |

The archive and model weights are not committed to Git, and there is currently no public model download URL. A fresh clone can run the rule-based demo directly. Full acceptance requires the separately supplied fixed checkpoints; newly trained weights are a new experiment and will not match the frozen hashes. See the [delivery guide](docs/reproducible-release.md) for provenance and setup.

UI demo:

![OptiAgent UI Demo](assets/ui-demo.png)

Architecture:

![OptiAgent Architecture](assets/architecture.png)

## Current Capabilities

- Natural language to `ProblemSpec`
- Local-rule routing with optional LLM-based routing
- Local markdown-based RAG over optimization knowledge
- CSV upload, schema inference, normalization, and validation
- Structured execution for optimization templates
- Independent constraint/objective verification with deterministic reward signals
- Built-in Document, Data, and Solver MCP servers with versioned problem/result contracts
- A conditional LangGraph workflow with Requirement Analyst, Planner, Data Agent, Modeler, Solver, Verifier, Policy, and Explainer nodes
- Persistent multi-turn requirements, versioned inline data, save-without-solve, resume, undo and verified plan comparison
- Natural-language capacity changes for knapsack; complete JSON replacement for five inline templates
- A typed recovery policy with candidate actions, action masks, and bounded modeling/solver retries
- Live `agent_step` events and streaming answer output through `/api/ask/stream`
- Versioned episode/step storage with state-action-observation and offline-training exports
- A real Gateway/Solver/Verifier smoke benchmark covering six templates, schema faults, and tampered objectives
- A learnable recovery policy using behavior-cloning warm-up and action-masked Double DQN
- Fixed-dataset BC/CQL with content-isolated splits and five-seed controlled evaluations
- A live repair page with explicit rule/previous/learned policy selection and isolated solver subprocesses
- A transport-aware policy trained on real MCP stdio timeout, disconnect, and invalid-structure traces
- Session-isolated file handling and SQLite persistence

## Supported Executable Templates

| Template | `template_id` | Input | Solver |
| --- | --- | --- | --- |
| Facility location and customer assignment | `facility_location` | Three CSV files: warehouses / customers / costs | Gurobi MILP |
| 0-1 knapsack | `knapsack` | JSON or CSV | Gurobi IP |
| Assignment | `assignment` | JSON or CSV | Gurobi MILP |
| Traveling salesman problem | `tsp` | JSON or CSV distance data, or coordinate CSV | Exact enumeration / Held-Karp / Gurobi MILP / local search |
| Job shop scheduling | `job_shop_scheduling` | JSON or CSV | Gurobi MILP / heuristic fallback |
| Production mix planning | `production_mix` | JSON or CSV | Gurobi LP/MILP |

## How It Works

```text
User question / uploaded data
  -> LangGraph Agent Policy
     -> Requirement Analyst -> clarify or proceed
     -> Planner -> Data Agent -> Modeler
     -> Solver -> MCP Gateway -> Document / Data / Solver MCP
     -> Verifier -> Policy -> accept / retry / rebuild / terminate
     -> Explainer
  -> Persisted trajectory and structured result
```

## Repository Highlights

- [README.md](README.md): Chinese-first main project documentation
- [examples/README.md](examples/README.md): quick-start examples for visitors
- [CHANGELOG.md](CHANGELOG.md): notable project changes
- [CONTRIBUTING.md](CONTRIBUTING.md): contribution guidance
- [docs/mcp-architecture.md](docs/mcp-architecture.md): MCP contracts, tools, configuration, and security boundaries
- [docs/agent-workflow.md](docs/agent-workflow.md): observable LangGraph execution and SSE event contract
- [docs/research-direction.md](docs/research-direction.md): research question, policy formulation, reward design, and experimental roadmap
- [docs/trajectory-data.md](docs/trajectory-data.md): episode schema, lifecycle, and training export contract
- [docs/rl-environment.md](docs/rl-environment.md): action space, rewards, fault scenarios, and reproducible rollouts
- [docs/e2e-test-plan.md](docs/e2e-test-plan.md): real solver test matrix, acceptance metrics, and fault-injection roadmap
- [docs/learning-policy.md](docs/learning-policy.md): state encoder, BC/Double-DQN objective, hyperparameters, and initial results
- [docs/two-track-roadmap.md](docs/two-track-roadmap.md): Agent-policy and open-model post-training architecture, data loop, evaluation, and milestones

## Quick Start

For a fresh clone, start with the core profile. The validated environment is Python 3.14 / macOS arm64. All six templates require a working Gurobi license for full solver acceptance; the interpreter, dependency wheels and license are not bundled in Git.

```bash
# 已有项目时跳过克隆和切换目录。
git clone https://github.com/tyz211/optiagent.git
cd optiagent

# 创建独立环境；规则模式不需要 PyTorch 或模型权重。
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-demo.lock
.venv/bin/python scripts/verify_demo.py --profile core
./start.sh
```

Open the address printed by the startup script, usually `http://127.0.0.1:8000`; occupied ports are skipped automatically. Create a conversation and load the knapsack demo data. `/repair-demo` supports rule-based recovery without weights; unavailable learned-policy options are disabled.

The startup script defaults to one server process without reload. Use `PORT=8010 ./start.sh` to request a port or `RELOAD=1 ./start.sh` for development. The unpinned `requirements.txt` remains a development option, not a claim of frozen-environment acceptance.

The acceptance command verifies locked dependencies, the source manifest when present, regression tests, seven dialogue turns and HTTP/SSE flows. It uses temporary databases. Full mode also verifies the two checkpoint hashes and executes all neural-policy tests. Reports are written to new directories under `artifacts/validation/`.

The current product is a bounded local demo. Arbitrary constraint editing, general model-code repair, learned dialogue policies, production authentication and LLM SFT/GRPO are not completed. Controlled recovery experiments do not establish real-business gains.

Manual backend launch:

```bash
# 显式使用项目环境中的解释器。
.venv/bin/python -m uvicorn api.main:app --host 127.0.0.1 --port 8000
```

## Recovery Policy Training

Tabular Q-learning and continuation use a separate entry point from neural training:

```bash
# 从头训练及续训均使用新的输出目录。
.venv/bin/python scripts/train_tabular_recovery_policy.py --output artifacts/rl/tabular-v1
.venv/bin/python scripts/train_tabular_recovery_policy.py \
  --resume-from artifacts/rl/tabular-v1 --output artifacts/rl/tabular-v2 --episodes 10000
```

The [historical tabular report](docs/rl-training-results.md) describes synthetic-environment results that match the rule baseline. It does not establish real-task generalization.

Train the neural Recovery Policy:

```bash
# 神经网络训练使用可选 RL 依赖，不覆盖固定交付权重。
.venv/bin/python -m pip install -r requirements-rl.lock
.venv/bin/python scripts/train_recovery_policy.py
```

Every attempt receives a unique run ID and writes an immutable manifest, checkpoint, report, and append-only history under `artifacts/rl/runs/`. Failed runs are recorded as well.

Train the cost-aware policy on verified results collected from the real Gateway/Solver/Verifier path:

```bash
# 根据真实求解参考结果训练成本感知策略。
.venv/bin/python scripts/train_real_recovery_policy.py --seed 47 --episodes 1200
```

Train the transport-aware recovery policy on controlled, real MCP stdio failures:

```bash
# 从隔离 MCP 子进程采集受控故障。
.venv/bin/python scripts/train_mcp_transport_policy.py \
  --seed 52 \
  --episodes 1500 \
  --bc-epochs 180
```

This run launches isolated MCP subprocesses, captures normal, timeout, disconnect, and invalid structured responses, and stores a redacted dataset, checkpoint, and evaluation report in an immutable run directory. It does not call the LLM API during training.

## Controlled Study Results

- The first five-seed offline study used 120 independent external instances. Validation-selected CQL reduced average modeling calls from 1.400 to 1.205 (13.90%) while retaining 100% recoverable success. The gain was concentrated in the predefined transient objective-mismatch scenario. [Study report](docs/offline-study-results.md)
- A later data-repair study used 60 new external instances. The previous policy achieved 79.17% recoverable success; the new CQL policy reached 100%, matching the rule baseline. Both the new policy and rules averaged 1.600 modeling calls. Reliability passed; the cost-improvement gate did not. [Repair report](docs/repair-demo-training.md)

The policies choose among accept, retry, rebuild and terminate. The repair demo uses a deterministic mapping repairer; these results do not show general autonomous model repair, LLM training or real-business cost savings. The 29/35-dimensional policies can run in the main workflow; the 40-dimensional transport-aware policy remains in its separate evaluation environment.

## Example Inputs

- Facility location: `data/facility_location_warehouses.csv`, `data/facility_location_customers.csv`, `data/facility_location_costs.csv`
- Assignment: [examples/assignment_sample.json](examples/assignment_sample.json)
- Job shop scheduling: [examples/job_shop_scheduling_sample.json](examples/job_shop_scheduling_sample.json)
- Production mix: [examples/production_mix_sample.json](examples/production_mix_sample.json)

## Roadmap

- Extend multi-turn edits, named scenario branches and dialogue-policy evaluation
- Expand the existing trajectory dataset with real user corrections and failure-repair records
- Extend the existing verifier to new templates and failure mechanisms
- Compare recovery policies on richer faults and independent real tasks, including actual latency and cost
- Learn tool routing, failure recovery, and budget-aware solver portfolio selection
- Establish model-training data contracts before starting open-model SFT and verifier-guided GRPO
- Extend the benchmark to VRP/VRPTW, network flow, staff scheduling, and robust optimization

## Community

If you are building OR agents, solver copilots, or optimization-aware decision systems, this repo is meant to be a practical base for extension and experimentation.
