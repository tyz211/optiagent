# Contributing

Thank you for your interest in OptiAgent.

This repository is a practical prototype for people exploring operations research agents, solver copilots, and LLM + optimization workflows. Small focused contributions are welcome.

## Good First Contribution Areas

- Add a new optimization template and register it in the routing flow
- Improve CSV schema recognition and field normalization
- Add benchmark cases and reproducible example datasets
- Improve result explanations, audit trails, and failure messages
- Extend solver support or fallback heuristics
- Improve English documentation for international readers

## Local Development

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## Suggested Validation

```bash
python3 -m compileall api optiagent
node --check web/app.js
```

## Contribution Style

- Keep changes small and intentional
- Prefer clear data contracts over implicit magic
- Preserve local-first behavior when possible
- Document any new template, tool, or solver entry clearly
- If a feature depends on external services, keep a local fallback path whenever reasonable

## Pull Request Notes

- Explain what problem the change solves
- Mention affected templates, tools, or APIs
- Include a small runnable example when adding a new optimization capability
- Add README updates when the user-facing workflow changes

## Discussion Topics That Fit This Repo

- OR Agent architecture
- RAG for optimization modeling knowledge
- Natural language to structured optimization specs
- Solver routing and explainability
- CSV/JSON ingestion for optimization systems
