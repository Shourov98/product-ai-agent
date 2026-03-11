from __future__ import annotations

import re

from app.schemas.response import CoreProductResponse, TikTokResponse


class TikTokAgent:
    async def process(self, core_data: CoreProductResponse) -> TikTokResponse:
        title = self._build_title(core_data)
        hashtags = self._build_hashtags(core_data)
        return TikTokResponse(
            title=title,
            social_description=(
                f"{core_data.normalized_title} packaged for short-form commerce. "
                f"Highlights: {', '.join(core_data.features[:3])}."
            ),
            hashtags=hashtags,
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
