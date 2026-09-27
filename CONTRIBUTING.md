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
python3.14 -m venv .venv
source .venv/bin/activate
pip install -r requirements-demo.lock
uvicorn api.main:app --reload --host 127.0.0.1 --port 8000
```

## Suggested Validation

```bash
# 基础模式不要求神经网络依赖或检查点。
python scripts/verify_demo.py --profile core

# 完整交付验收需 requirements-rl.lock 及两份固定权重。
python scripts/verify_demo.py --profile full
```

Choose the profile matching the installed dependencies. See [reproducible delivery](docs/reproducible-release.md) for the frozen environment, archive builder, model provenance and output reports. Generated code should include Chinese comments.

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
