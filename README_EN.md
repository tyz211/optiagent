# OptiAgent

> An agentic operations-research system for learning how to model, route tools, solve, verify, and recover.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Gurobi](https://img.shields.io/badge/Gurobi-Optimizer-E87722)
![RAG](https://img.shields.io/badge/RAG-Knowledge%20Augmented-4A90E2)
![LangChain](https://img.shields.io/badge/LangChain-Agent%20Tools-1C3C3C)
![SQLite](https://img.shields.io/badge/SQLite-Local%20Storage-003B57?logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

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

The frozen local delivery is **demo-2026.09.27**, with Python 3.14 on macOS arm64 as its validated baseline. The full archive carries the source snapshot and two hash-checked recovery checkpoints. Install `requirements-rl.lock` and run `.venv/bin/python scripts/verify_demo.py --profile full`. For rule-only operation, use `requirements-demo.lock` and `--profile core`; PyTorch and an LLM API key are not required.

See the [reproducible delivery guide](docs/reproducible-release.md) for archive generation, checksums, checkpoint provenance and acceptance details. Weights are supplied in the local full archive, not in Git; no public download endpoint is assumed. Other operating systems and Python versions are outside this delivery's validated scope.

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

```bash
./start.sh
```

or

```bash
PORT=8010 ./start.sh
```

First-time setup:

```bash
# 完整学习策略演示改用 requirements-rl.lock 与 --profile full。
python3.14 -m venv .venv
.venv/bin/python -m pip install -r requirements-demo.lock
.venv/bin/python scripts/verify_demo.py --profile core
```

The startup script defaults to one server process without reload. Use `RELOAD=1 ./start.sh` for development. The unpinned `requirements.txt` remains a development option, not a claim of frozen-environment acceptance.

The acceptance command verifies locked dependencies, the source manifest when present, regression tests, seven dialogue turns and HTTP/SSE flows. It uses temporary databases. Full mode also verifies the two checkpoint hashes and executes all neural-policy tests. Reports are written to new directories under `artifacts/validation/`.

The current product is a bounded local demo. Arbitrary constraint editing, general model-code repair, learned dialogue policies, production authentication and LLM SFT/GRPO are not completed. Controlled recovery experiments do not establish real-business gains.

Manual backend launch:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

Train the learnable Recovery Policy:

```bash
pip install -r requirements-rl.txt
python scripts/train_recovery_policy.py
```

Every attempt receives a unique run ID and writes an immutable manifest, checkpoint, report, and append-only history under `artifacts/rl/runs/`. Failed runs are recorded as well.

Train the cost-aware policy on verified results collected from the real Gateway/Solver/Verifier path:

```bash
python scripts/train_real_recovery_policy.py --seed 47 --episodes 1200
```

Train the transport-aware recovery policy on controlled, real MCP stdio failures:

```bash
python scripts/train_mcp_transport_policy.py \
  --seed 52 \
  --episodes 1500 \
  --bc-epochs 180
```

This run launches isolated MCP subprocesses, captures normal, timeout, disconnect, and invalid structured responses, and stores a redacted dataset, checkpoint, and evaluation report in an immutable run directory. It does not call the LLM API during training.

## Example Inputs

- Facility location: `data/facility_location_warehouses.csv`, `data/facility_location_customers.csv`, `data/facility_location_costs.csv`
- Assignment: [examples/assignment_sample.json](examples/assignment_sample.json)
- Job shop scheduling: [examples/job_shop_scheduling_sample.json](examples/job_shop_scheduling_sample.json)
- Production mix: [examples/production_mix_sample.json](examples/production_mix_sample.json)

## Roadmap

- Build an optimization-agent trajectory dataset from observable graph runs
- Add deterministic solution verification and verifiable reward signals
- Compare rule, LLM-planner, behavior-cloning, and offline-RL policies
- Learn tool routing, failure recovery, and budget-aware solver portfolio selection
- Extend the benchmark to VRP/VRPTW, network flow, staff scheduling, and robust optimization

## Community

If you are building OR agents, solver copilots, or optimization-aware decision systems, this repo is meant to be a practical base for extension and experimentation.
