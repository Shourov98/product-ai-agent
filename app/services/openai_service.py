from __future__ import annotations

from typing import Any

from openai import AsyncOpenAI


class OpenAIServiceError(RuntimeError):
    pass

class OpenAIService:
    def __init__(
        self,
        *,
        api_key: str | None,
        model: str,
        image_model: str,
        enabled: bool = False,
    ) -> None:
        self.enabled = enabled and bool(api_key)
        self.model = model
        self.image_model = image_model
        self.client = AsyncOpenAI(api_key=api_key) if self.enabled else None

    async def generate_structured_output(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
        schema_name: str,
        schema: dict[str, Any],
    ) -> dict[str, Any]:
        if not self.enabled or self.client is None:
            raise OpenAIServiceError("OpenAI is disabled.")

        import json

        def _format_value(value: Any) -> str:
            if isinstance(value, (dict, list, tuple)):
                return json.dumps(value, ensure_ascii=False)
            return str(value)

        user_lines = [f"{key}: {_format_value(value)}" for key, value in user_payload.items()]

        try:
            response = await self.client.responses.create(
                model=self.model,
                input=[
                    {"role": "system", "content": system_prompt},
                    {"role": "user", "content": "\n".join(user_lines)},
                ],
                text={
                    "format": {
                        "type": "json_schema",
                        "name": schema_name,
                        "schema": schema,
                        "strict": False,
                    }
                },
            )
        except Exception as exc:  # pragma: no cover - network/provider failures
            raise OpenAIServiceError(str(exc)) from exc

        output_text = (response.output_text or "").strip()
        if not output_text:
            raise OpenAIServiceError("OpenAI returned an empty structured response.")

        try:
            parsed = json.loads(output_text)
        except Exception as exc:
            raise OpenAIServiceError("OpenAI returned invalid JSON.") from exc

        if not isinstance(parsed, dict):
            raise OpenAIServiceError("OpenAI returned a non-object JSON payload.")

        return parsed

    async def edit_image(
        self,
        *,
        prompt: str,
        image_bytes: bytes,
        filename: str,
        mime_type: str,
        size: str,
        background: str,
    ) -> bytes:
        if not self.enabled or self.client is None:
            raise OpenAIServiceError("OpenAI is disabled.")

        try:
            response = await self.client.images.edit(
                model=self.image_model,
                prompt=prompt,
                size=size,
                background=background,
                image=(filename, image_bytes, mime_type),
            )
        except Exception as exc:  # pragma: no cover - network/provider failures
            raise OpenAIServiceError(str(exc)) from exc

        data = getattr(response, "data", None) or []
        if not data:
            raise OpenAIServiceError("OpenAI image edit returned no data.")

        image_item = data[0]
        base64_payload = getattr(image_item, "b64_json", None) or getattr(image_item, "base64", None)
        if not base64_payload:
            raise OpenAIServiceError("OpenAI image edit returned no binary payload.")

        import base64

        return base64.b64decode(base64_payload)

    async def analyze_image(
        self,
        *,
        image_bytes: bytes,
        mime_type: str,
    ) -> dict[str, Any]:
        if not self.enabled or self.client is None:
            raise OpenAIServiceError("OpenAI is disabled.")

        import base64
        base64_image = base64.b64encode(image_bytes).decode("utf-8")
        try:
            response = await self.client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "text",
                                "text": (
                                    "Analyze this product image. Identify the exact product name/model, its category, "
                                    "and all visual/physical attributes (brand, color, material, style, model, etc.).\n"
                                    "Return only JSON with the following fields:\n"
                                    "- title: A clean, descriptive title of the product (e.g. 'Sony PlayStation 5 Console' or 'Nike Air Max Running Shoes').\n"
                                    "- product_type: The canonical category/type of the product.\n"
                                    "- confidence: A float between 0.0 and 1.0 representing your confidence.\n"
                                    "- attributes: A dictionary of key-value pairs representing all detected visual properties."
                                )
                            },
                            {
                                "type": "image_url",
                                "image_url": {
                                    "url": f"data:{mime_type};base64,{base64_image}"
                                }
                            }
                        ]
                    }
                ],
                response_format={"type": "json_object"}
            )
        except Exception as exc:
            raise OpenAIServiceError(str(exc)) from exc

        output_text = (response.choices[0].message.content or "").strip()
        if not output_text:
            raise OpenAIServiceError("OpenAI returned an empty response.")

        try:
            import json
            parsed = json.loads(output_text)
            if not isinstance(parsed, dict):
                raise OpenAIServiceError("OpenAI returned a non-object JSON payload.")
            return parsed
        except Exception as exc:
            raise OpenAIServiceError("OpenAI returned invalid JSON.") from exc
