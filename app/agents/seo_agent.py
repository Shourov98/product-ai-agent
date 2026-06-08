from __future__ import annotations

from app.schemas.response import CoreProductResponse, MarketResearchBundleResponse, SeoInsightsResponse
from app.services.openai_service import OpenAIService, OpenAIServiceError
from app.utils.product_text import title_keywords, unique_strings


class SeoAgent:
    _SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["primary_keywords", "secondary_keywords", "title_terms", "marketplace_keywords"],
        "properties": {
            "primary_keywords": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 8},
            "secondary_keywords": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 12},
            "title_terms": {"type": "array", "items": {"type": "string"}, "minItems": 3, "maxItems": 12},
            "marketplace_keywords": {
                "type": "object",
                "additionalProperties": {"type": "array", "items": {"type": "string"}},
            },
        },
    }

    def __init__(self, openai_service: OpenAIService | None = None) -> None:
        self.openai_service = openai_service

    async def process(
        self,
        core_data: CoreProductResponse,
        research: MarketResearchBundleResponse,
    ) -> SeoInsightsResponse:
        fallback = self._build_fallback(core_data, research)
        if self.openai_service is None:
            return fallback

        try:
            data = await self.openai_service.generate_structured_output(
                system_prompt=(
                    "You are a senior marketplace SEO strategist. "
                    "Produce practical keyword clusters from canonical product data and competitor-style research evidence. "
                    "Return only JSON matching the schema."
                ),
                user_payload={
                    "core_product": core_data.model_dump(),
                    "research": research.model_dump(),
                },
                schema_name="seo_insights",
                schema=self._SCHEMA,
            )
            return self._from_data(data, fallback)
        except OpenAIServiceError:
            return fallback

    def _build_fallback(
        self,
        core_data: CoreProductResponse,
        research: MarketResearchBundleResponse,
    ) -> SeoInsightsResponse:
        primary = unique_strings(
            [core_data.normalized_title.lower(), core_data.product_type.lower(), *title_keywords(core_data.normalized_title)],
            limit=6,
        )
        secondary = unique_strings(
            [
                core_data.category.lower(),
                *[value.lower() for value in core_data.attributes.values()],
                *research.amazon.keyword_signals[:3],
                *research.ebay.keyword_signals[:2],
            ],
            limit=10,
        )
        title_terms = unique_strings(primary + secondary, limit=10)
        marketplace_keywords = {
            "amazon": unique_strings(primary + research.amazon.keyword_signals, limit=10),
            "ebay": unique_strings(primary + research.ebay.keyword_signals, limit=10),
            "etsy": unique_strings(primary + research.etsy.keyword_signals, limit=10),
            "tiktok": unique_strings(primary + research.tiktok.keyword_signals, limit=10),
            "shopify": unique_strings(primary + research.shopify.keyword_signals, limit=10),
        }
        return SeoInsightsResponse(
            primary_keywords=primary,
            secondary_keywords=secondary,
            title_terms=title_terms,
            marketplace_keywords=marketplace_keywords,
        )

    def _from_data(self, data: dict[str, object], fallback: SeoInsightsResponse) -> SeoInsightsResponse:
        marketplace_keywords = data.get("marketplace_keywords")
        if isinstance(marketplace_keywords, dict):
            normalized_marketplace_keywords = {
                str(key): unique_strings(
                    [str(item) for item in value if isinstance(item, str)],
                    limit=12,
                )
                for key, value in marketplace_keywords.items()
                if isinstance(value, list)
            }
        else:
            normalized_marketplace_keywords = fallback.marketplace_keywords

        return SeoInsightsResponse(
            primary_keywords=self._coerce_list(data.get("primary_keywords"), fallback.primary_keywords, limit=8),
            secondary_keywords=self._coerce_list(data.get("secondary_keywords"), fallback.secondary_keywords, limit=12),
            title_terms=self._coerce_list(data.get("title_terms"), fallback.title_terms, limit=12),
            marketplace_keywords=normalized_marketplace_keywords,
        )

    @staticmethod
    def _coerce_list(value: object, fallback: list[str], *, limit: int) -> list[str]:
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            keywords = unique_strings(value, limit=limit)
            if keywords:
                return keywords
        return fallback
