from __future__ import annotations

import re

from app.schemas.response import CoreProductResponse, TikTokResponse
from app.services.openai_service import OpenAIService, OpenAIServiceError
from app.services.ollama_service import OllamaService, OllamaServiceError
from app.utils.product_text import unique_strings


class TikTokAgent:
    _SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "social_description", "hashtags"],
        "properties": {
            "title": {"type": "string"},
            "social_description": {"type": "string"},
            "hashtags": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 8},
        },
    }

    def __init__(
        self,
        ollama_service: OllamaService | None = None,
        openai_service: OpenAIService | None = None,
    ) -> None:
        self.ollama_service = ollama_service
        self.openai_service = openai_service

    async def process(self, core_data: CoreProductResponse) -> TikTokResponse:
        fallback = TikTokResponse(
            title=self._build_title(core_data),
            social_description=(
                f"{core_data.product_summary} Highlights: {', '.join(core_data.features[:3])}."
            ),
            hashtags=self._build_hashtags(core_data),
        )
        if self.openai_service is not None:
            try:
                data = await self.openai_service.generate_structured_output(
                    system_prompt=(
                        "You are a TikTok Shop copywriter. Produce short-form commerce copy that feels native, concise, and conversion-oriented. "
                        "Do not mention AI, confidence, or internal processing."
                        "Do not include any text inside image."
                    ),
                    user_payload={"core_product": core_data.model_dump()},
                    schema_name="tiktok_listing",
                    schema=self._SCHEMA,
                )
                return self._from_data(data, fallback)
            except OpenAIServiceError:
                pass

        if self.ollama_service is None:
            return fallback

        prompt = (
            "You are a senior TikTok listing agent. Return only valid JSON with keys title, "
            "social_description, hashtags.\n"
            "Write for social commerce, not internal tooling. Do not mention AI or confidence.\n"
            f"Core product: {core_data.model_dump()}\n"
        )
        try:
            result = await self.ollama_service.generate_json(prompt=prompt)
        except OllamaServiceError:
            return fallback

        return self._from_data(result.parsed, fallback)

    def _from_data(self, data: dict[str, object], fallback: TikTokResponse) -> TikTokResponse:
        hashtags = data.get("hashtags")
        return TikTokResponse(
            title=str(data.get("title", fallback.title)),
            social_description=str(data.get("social_description", fallback.social_description)),
            hashtags=hashtags if isinstance(hashtags, list) and all(isinstance(item, str) for item in hashtags) else fallback.hashtags,
        )

    @staticmethod
    def _build_title(core_data: CoreProductResponse) -> str:
        color = core_data.attributes.get("color")
        if color:
            return f"{color.title()} {core_data.normalized_title}"
        return core_data.normalized_title

    @staticmethod
    def _build_hashtags(core_data: CoreProductResponse) -> list[str]:
        raw_tags = [
            core_data.product_type,
            core_data.category,
            *core_data.attributes.values(),
        ]
        hashtags = []
        for value in raw_tags:
            compact = re.sub(r"[^a-zA-Z0-9]+", "", value.title())
            if compact:
                hashtags.append(f"#{compact}")
        hashtags.append("#TikTokMadeMeBuyIt")
        return unique_strings(hashtags, limit=6)
