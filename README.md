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

## Test

```bash
pytest
```
