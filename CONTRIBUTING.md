# Contributing to agentrecall

Thanks for your interest! agentrecall aims to stay **small, dependency-light, and honest**.
Contributions that keep the core to the Python stdlib and push everything else into optional
extras are the most likely to be merged.

## Development setup

```bash
git clone https://github.com/shaxzodbek-uzb/agentrecall
cd agentrecall
python -m venv .venv && source .venv/bin/activate
pip install -e ".[dev]"        # core + pytest + ruff
pip install -e ".[all]"        # add semantic + mcp to run those tests too
```

## Before you open a PR

```bash
ruff check .
ruff format .
pytest
```

- **The core stays dependency-free.** Anything beyond the stdlib goes under
  `[project.optional-dependencies]` and is imported lazily inside the function that needs it.
- **FTS-only tests must pass with no extras installed.** Guard semantic/MCP tests with
  `pytest.importorskip(...)`.
- **No network at import or collection time.** Model downloads only happen inside an
  explicitly-marked semantic test.
- Match the existing style; `SPEC.md` is the source of truth for public API names and
  behavior — update it in the same PR if you change the surface.

## Good first contributions

- Adapters for popular agent frameworks (LangGraph, pydantic-ai, CrewAI) under `examples/`.
- Additional `Embedder` implementations (OpenAI, Ollama, fastembed) as opt-in extras.
- Benchmarks of recall quality vs. corpus size.

## Reporting bugs

Open an issue with the SQLite version (`python -c "import sqlite3; print(sqlite3.sqlite_version)"`),
your Python version, and a minimal reproduction.

By contributing you agree your work is licensed under the project's MIT license.
