from __future__ import annotations

import re

from app.schemas.response import CoreProductResponse, TikTokResponse
from app.services.ollama_service import OllamaService, OllamaServiceError


class TikTokAgent:
    def __init__(self, ollama_service: OllamaService | None = None) -> None:
        self.ollama_service = ollama_service

    async def process(self, core_data: CoreProductResponse) -> TikTokResponse:
        fallback = TikTokResponse(
            title=self._build_title(core_data),
            social_description=(
                f"{core_data.normalized_title} packaged for short-form commerce. "
                f"Highlights: {', '.join(core_data.features[:3])}."
            ),
            hashtags=self._build_hashtags(core_data),
        )
        if self.ollama_service is None:
            return fallback

        prompt = (
            "You are a TikTok listing agent. Return only valid JSON with keys title, "
            "social_description, hashtags.\n"
            f"Core product: {core_data.model_dump()}\n"
        )
        try:
            result = await self.ollama_service.generate_json(prompt=prompt)
        except OllamaServiceError:
            return fallback

        data = result.parsed
        hashtags = data.get("hashtags")
        return TikTokResponse(
            title=str(data.get("title", fallback.title)),
            social_description=str(
                data.get("social_description", fallback.social_description)
            ),
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
        return hashtags[:6]
