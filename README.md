# Product AI Agent

Deterministic FastAPI backend for product listing generation from an uploaded image and title.

## Architecture

- `app/api`: FastAPI routes
- `app/orchestrator`: pipeline orchestration
- `app/agents`: vision, core, and marketplace agents
- `app/services`: service boundaries for image processing and future model integrations
- `app/schemas`: strict request and response contracts
- `tests`: agent, pipeline, and API coverage

## Run

```bash
uvicorn app.main:app --reload
```

To use Ollama-backed text agents:

```bash
ollama serve
uv run uvicorn app.main:app --reload --env-file .env
```

Set these env vars:

```bash
OLLAMA_ENABLED=true
OLLAMA_BASE_URL=http://127.0.0.1:11434
OLLAMA_MODEL=qwen3:8b
```

## Test

```bash
pytest
```

Each request writes JSON files to `output/<run_id>/`.
