# Changelog

All notable changes to this project will be documented in this file.

## 2026-09

### Added

- Added standalone Document, Data, and Solver MCP servers with stdio and Streamable HTTP transports.
- Added versioned `ProblemEnvelope` and `SolveEnvelope` contracts with source provenance and validation reports.
- Added built-in MCP discovery, per-server degradation, prefixed tool names, and a read-only `/api/mcp` manifest.
- Added MCP contract and protocol tests covering document retrieval, path isolation, data conversion, and solver execution.

### Improved

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
