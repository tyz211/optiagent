# OptiAgent

> A local-first optimization agent for operations research workflows, combining natural language understanding, structured modeling, RAG, solver execution, and explainable results.

![Python](https://img.shields.io/badge/Python-3.11%2B-3776AB?logo=python&logoColor=white)
![FastAPI](https://img.shields.io/badge/FastAPI-API-009688?logo=fastapi&logoColor=white)
![Gurobi](https://img.shields.io/badge/Gurobi-Optimizer-E87722)
![RAG](https://img.shields.io/badge/RAG-Knowledge%20Augmented-4A90E2)
![LangChain](https://img.shields.io/badge/LangChain-Agent%20Tools-1C3C3C)
![SQLite](https://img.shields.io/badge/SQLite-Local%20Storage-003B57?logo=sqlite&logoColor=white)
![License](https://img.shields.io/badge/License-MIT-green)

OptiAgent is a practical prototype for building optimization-focused agents. It connects natural language input, structured problem specification, local RAG, data parsing, solver execution, and explainable result rendering into one end-to-end workflow.

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

UI demo:

![OptiAgent UI Demo](/Users/tianyuanzhe/运筹优化/assets/ui-demo.png)

Architecture:

![OptiAgent Architecture](/Users/tianyuanzhe/运筹优化/assets/architecture.png)

## Current Capabilities

- Natural language to `ProblemSpec`
- Local-rule routing with optional LLM-based routing
- Local markdown-based RAG over optimization knowledge
- CSV upload, schema inference, normalization, and validation
- Structured execution for optimization templates
- Streaming answer output through `/api/ask/stream`
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
  -> LLM router or local rule router
  -> ProblemSpec generation
  -> RAG retrieval
  -> Data parsing and validation
  -> Solver execution
  -> Optimality / feasibility checks
  -> Structured answer rendering
```

## Repository Highlights

- [README.md](/Users/tianyuanzhe/运筹优化/README.md): Chinese-first main project documentation
- [examples/README.md](/Users/tianyuanzhe/运筹优化/examples/README.md): quick-start examples for visitors
- [CHANGELOG.md](/Users/tianyuanzhe/运筹优化/CHANGELOG.md): notable project changes
- [CONTRIBUTING.md](/Users/tianyuanzhe/运筹优化/CONTRIBUTING.md): contribution guidance

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
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

Manual backend launch:

```bash
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

Open:

```text
http://127.0.0.1:8000
```

## Example Inputs

- Facility location: `data/facility_location_warehouses.csv`, `data/facility_location_customers.csv`, `data/facility_location_costs.csv`
- Assignment: [examples/assignment_sample.json](/Users/tianyuanzhe/运筹优化/examples/assignment_sample.json)
- Job shop scheduling: [examples/job_shop_scheduling_sample.json](/Users/tianyuanzhe/运筹优化/examples/job_shop_scheduling_sample.json)
- Production mix: [examples/production_mix_sample.json](/Users/tianyuanzhe/运筹优化/examples/production_mix_sample.json)

## Roadmap

- Add more active templates such as VRP, VRPTW, network flow, staff scheduling, and robust optimization
- Improve LLM planning and clarification for ambiguous user questions
- Strengthen multi-file schema alignment and data correction
- Support more pluggable solvers and finer-grained solver routing
- Add benchmarks, curated cases, and evaluation scripts

## Community

If you are building OR agents, solver copilots, or optimization-aware decision systems, this repo is meant to be a practical base for extension and experimentation.
