from __future__ import annotations

import json
from typing import Any

import httpx


class GeminiServiceError(RuntimeError):
    pass


class GeminiService:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        api_base_url: str,
        enabled: bool = False,
    ) -> None:
        self.enabled = enabled and bool(api_key)
        self.api_key = api_key
        self.model = model
        self.api_base_url = api_base_url.rstrip("/")

    async def generate_structured_output(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        use_google_search: bool = False,
        schema: dict[str, Any] | None = None,
    ) -> dict[str, Any]:
        if not self.enabled or not self.api_key:
            raise GeminiServiceError("Gemini is disabled.")

        user_text = "\n".join(f"{key}: {self._format_value(value)}" for key, value in user_payload.items())
        payload: dict[str, Any] = {
            "systemInstruction": {
                "parts": [{"text": system_prompt}],
            },
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": user_text}],
                }
            ],
            "generationConfig": {
                "temperature": 0.2,
                "responseMimeType": "application/json",
            },
        }
        if schema is not None:
            payload["generationConfig"]["responseSchema"] = schema
        if use_google_search:
            payload["tools"] = [{"google_search": {}}]

        url = f"{self.api_base_url}/models/{self.model}:generateContent"
        params = {"key": self.api_key}
        async with httpx.AsyncClient(timeout=45.0) as client:
            response = await client.post(url, params=params, json=payload)
        if response.status_code >= 400:
            raise GeminiServiceError(
                f"Gemini request failed with status {response.status_code}: {response.text[:240]}"
            )

        data = response.json()
        try:
            parts = data["candidates"][0]["content"]["parts"]
        except (KeyError, IndexError, TypeError) as exc:
            raise GeminiServiceError("Gemini returned an unexpected response shape.") from exc

        text = "".join(str(part.get("text", "")) for part in parts).strip()
        if not text:
            raise GeminiServiceError("Gemini returned an empty response.")

        parsed = self._parse_json_text(text)
        if not isinstance(parsed, dict):
            raise GeminiServiceError("Gemini returned a non-object JSON payload.")
        return parsed

    @staticmethod
    def _format_value(value: Any) -> str:
        if isinstance(value, (dict, list, tuple)):
            return json.dumps(value, ensure_ascii=False)
        return str(value)

    @staticmethod
    def _parse_json_text(text: str) -> dict[str, Any]:
        candidates = [text.strip()]
        stripped = text.strip()
        if stripped.startswith("```"):
            lines = stripped.splitlines()
            if len(lines) >= 3:
                candidates.append("\n".join(lines[1:-1]).strip())

        start = stripped.find("{")
        end = stripped.rfind("}")
        if start != -1 and end != -1 and end > start:
            candidates.append(stripped[start : end + 1].strip())

        for candidate in candidates:
            try:
                parsed = json.loads(candidate)
            except json.JSONDecodeError:
                continue
            if isinstance(parsed, dict):
                return parsed

        raise GeminiServiceError("Gemini returned invalid JSON.")
