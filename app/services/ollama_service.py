from __future__ import annotations

import json
from dataclasses import dataclass
from typing import Any
from urllib.error import HTTPError, URLError
from urllib.request import Request, urlopen


class OllamaServiceError(RuntimeError):
    pass


@dataclass(slots=True)
class OllamaResult:
    parsed: dict[str, Any]
    raw: dict[str, Any]


class OllamaService:
    def __init__(self, base_url: str, model: str, enabled: bool = True) -> None:
        self.base_url = base_url.rstrip("/")
        self.model = model
        self.enabled = enabled

    async def generate_json(self, *, prompt: str) -> OllamaResult:
        if not self.enabled:
            raise OllamaServiceError("Ollama is disabled.")

        payload = {
            "model": self.model,
            "prompt": prompt,
            "stream": False,
            "format": "json",
        }
        request = Request(
            f"{self.base_url}/api/generate",
            data=json.dumps(payload).encode("utf-8"),
            headers={"Content-Type": "application/json"},
            method="POST",
        )

        try:
            with urlopen(request, timeout=120) as response:
                raw_payload = json.loads(response.read().decode("utf-8"))
        except HTTPError as exc:
            detail = exc.read().decode("utf-8", errors="ignore")
            raise OllamaServiceError(f"Ollama HTTP error: {exc.code} {detail}") from exc
        except URLError as exc:
            raise OllamaServiceError(f"Ollama connection error: {exc.reason}") from exc
        except TimeoutError as exc:
            raise OllamaServiceError("Ollama request timed out.") from exc

        content = raw_payload.get("response", "").strip()
        if not content:
            raise OllamaServiceError("Ollama returned an empty response.")

        try:
            parsed = json.loads(content)
        except json.JSONDecodeError as exc:
            raise OllamaServiceError(f"Ollama returned invalid JSON: {content}") from exc

        return OllamaResult(parsed=parsed, raw=raw_payload)
