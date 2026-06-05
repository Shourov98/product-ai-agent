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

To use OpenAI-backed structured generation for higher-quality product data:

```bash
OPENAI_ENABLED=true
OPENAI_API_KEY=your_openai_api_key_here
OPENAI_MODEL=gpt-5
OPENAI_IMAGE_MODEL=gpt-image-1
```

If both OpenAI and Ollama are configured, the pipeline will try OpenAI first and fall back to Ollama, then to deterministic local generation.

## Test

```bash
pytest
```

Each request writes JSON files to `output/<run_id>/`.

## Image Pipeline

The backend now also produces marketplace image variants for:

- Amazon
- eBay
- TikTok

Each run saves:

- original source upload
- transparent cutout when OpenAI image editing is enabled
- marketplace-specific image variants
- validation metadata and output paths in the API response

When OpenAI image editing is disabled or unavailable, the backend still saves source-image fallbacks and returns validation metadata explaining that fallback path.
