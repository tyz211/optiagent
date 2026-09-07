# Changelog

All notable changes to this project will be documented in this file.

## 2026-09

### Added

- Added an observable LangGraph workflow with Planner, Data Agent, Modeler, Solver, Verifier, Policy, and Explainer nodes.
- Added real-time `agent_step` SSE events, persisted execution traces, and an in-product seven-stage execution rail.
- Added the research direction **Learning an Agent Policy for Automated Optimization Modeling and Solving**, including a policy/reward formulation and RL roadmap.
- Added independent mathematical solution verification for all six executable templates and deterministic terminal reward signals.
- Added the `solver_verify_solution` MCP tool and automatic verification reports on every solver response.
- Added versioned `agent_episodes` and `agent_steps` trajectory storage with state/action/observation snapshots.
- Added replay, single-episode training views, batch training-data APIs, and persisted failure episodes with negative rewards.
- Added a typed Recovery Policy with constrained candidate actions, action masks, bounded Modeler/Solver loops, and dynamic LangGraph routing.
- Added policy-only decision transitions with `next_state` and terminal reward attribution for behavior cloning and offline RL.
- Added a framework-independent `reset/step` RL environment, four reproducible fault scenarios, stratified benchmark generation, baseline evaluation, and JSONL rollout export.
- Added standalone Document, Data, and Solver MCP servers with stdio and Streamable HTTP transports.
- Added versioned `ProblemEnvelope` and `SolveEnvelope` contracts with source provenance and validation reports.
- Added built-in MCP discovery, per-server degradation, prefixed tool names, and a read-only `/api/mcp` manifest.
- Added MCP contract and protocol tests covering document retrieval, path isolation, data conversion, and solver execution.

### Improved

- Improved local intent routing so natural wording such as “解决” enters the optimization path.
- Improved the LangChain supervisor so MCP discovery happens only when LLM tool routing is active.
- Migrated local-rule, uploaded-file, inline-JSON, and facility-location solver paths to the shared MCP Optimization Gateway.
- Removed duplicate local solving for structured datasets when LLM routing is disabled.
- Improved local file safety by restricting Document/Data MCP reads to explicitly allowed roots.
- Updated the API and documentation for built-in and external MCP configuration.

## 2026-06

### Added

- Added streaming ask pipeline with `/api/ask/stream` and frontend SSE rendering.
- Added richer README with architecture image, UI demo, bilingual positioning, and GitHub-facing sections.
- Added example assets and usage guidance for common optimization templates.
- Added local-only learning docs workflow and ignored `docs/` from version control.

### Improved

- Improved frontend answer rendering, status feedback, and structured result display.
- Improved repository positioning for OR Agent, RAG + solver, and optimization copilot audiences.
- Improved data upload flow and template-oriented problem solving experience.

## Earlier Milestones

### Initial Prototype

- Built FastAPI backend and static frontend.
- Added local SQLite persistence and file/session isolation.
- Added facility location workflow with Gurobi-based solving.
- Added generic optimization templates for knapsack, assignment, TSP, job shop scheduling, and production mix.
- Added local RAG over markdown knowledge bases.
