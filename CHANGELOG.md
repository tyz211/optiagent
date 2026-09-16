# Changelog

All notable changes to this project will be documented in this file.

## 2026-09

### Added

- Added a persistent multi-turn `RequirementBrief`, a Requirement Analyst graph node, clarification routing, and a solver-readiness guard.
- Added a conversation-scoped requirements API and frontend requirement cards for confirmed constraints, missing information, and follow-up questions.
- Added an observable LangGraph workflow with Planner, Data Agent, Modeler, Solver, Verifier, Policy, and Explainer nodes.
- Added real-time `agent_step` SSE events, persisted execution traces, and an in-product seven-stage execution rail.
- Added the research direction **Learning an Agent Policy for Automated Optimization Modeling and Solving**, including a policy/reward formulation and RL roadmap.
- Added a two-track roadmap separating the learnable Agent System from cloud open-model SFT/GRPO post-training, with a shared solver/verifier trajectory loop and unified ablations.
- Added independent mathematical solution verification for all six executable templates and deterministic terminal reward signals.
- Added the `solver_verify_solution` MCP tool and automatic verification reports on every solver response.
- Added versioned `agent_episodes` and `agent_steps` trajectory storage with state/action/observation snapshots.
- Added replay, single-episode training views, batch training-data APIs, and persisted failure episodes with negative rewards.
- Added a typed Recovery Policy with constrained candidate actions, action masks, bounded Modeler/Solver loops, and dynamic LangGraph routing.
- Added policy-only decision transitions with `next_state` and terminal reward attribution for behavior cloning and offline RL.
- Added a framework-independent `reset/step` RL environment, four reproducible fault scenarios, stratified benchmark generation, baseline evaluation, and JSONL rollout export.
- Added an end-to-end smoke benchmark that executes all six templates through the real Gateway, solvers, and verifier, with schema-loss and objective-tampering fault detection.
- Added a 29-dimensional versioned state encoder, behavior-cloning warm-up, action-masked Dueling Double DQN, replay training, checkpoint validation, and held-out policy evaluation.
- Added immutable per-run RL experiment directories, append-only training history, source/runtime metadata, artifact hashes, failed-run records, and credential redaction.
- Added a real Gateway recovery dataset covering six optimization templates, five recovery scenarios, measured solver latency, and isolated train/validation/test splits.
- Added a 35-dimensional cost-aware state encoder and a learned policy that distinguishes low-cost solver retry from model rebuild while preserving v1 checkpoint loading.
- Added a controlled MCP fault-probe server and real stdio client traces for normal responses, timeouts, process disconnects, and invalid structured content.
- Added a 40-dimensional transport-aware state encoder, MCP recovery environment, BC + masked Double DQN trainer, held-out evaluation, and immutable transport-training datasets.
- Added standalone Document, Data, and Solver MCP servers with stdio and Streamable HTTP transports.
- Added versioned `ProblemEnvelope` and `SolveEnvelope` contracts with source provenance and validation reports.
- Added built-in MCP discovery, per-server degradation, prefixed tool names, and a read-only `/api/mcp` manifest.
- Added MCP contract and protocol tests covering document retrieval, path isolation, data conversion, and solver execution.

### Improved

- Removed the obsolete hidden frontend result renderer and its duplicated table, metric, and model DOM.
- Consolidated repeated response persistence, LLM configuration/JSON parsing, and the three recovery-policy training loops.
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
