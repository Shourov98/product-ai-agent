from __future__ import annotations

from app.schemas.response import (
    AmazonResponse,
    CoreProductResponse,
    EbayResponse,
    EtsyResponse,
    MarketResearchBundleResponse,
    PricingInsightsResponse,
    ProductPipelineResponse,
    SeoInsightsResponse,
    ShopifyResponse,
    TikTokResponse,
)
from app.services.openai_service import OpenAIService, OpenAIServiceError


class ProductOptimizationAgent:
    _SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["core", "amazon", "ebay", "etsy", "tiktok", "shopify"],
        "properties": {
            "core": {
                "type": "object",
                "additionalProperties": False,
                "required": [
                    "normalized_title",
                    "category",
                    "product_type",
                    "product_summary",
                    "features",
                    "attributes",
                    "source_title",
                    "vision_confidence",
                ],
                "properties": {
                    "normalized_title": {"type": "string"},
                    "category": {"type": "string"},
                    "product_type": {"type": "string"},
                    "product_summary": {"type": "string"},
                    "features": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 8},
                    "attributes": {"type": "object", "additionalProperties": {"type": "string"}},
                    "source_title": {"type": "string"},
                    "vision_confidence": {"type": "number"},
                },
            },
            "amazon": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "bullet_points", "description", "backend_search_terms", "structured_attributes"],
                "properties": {
                    "title": {"type": "string"},
                    "bullet_points": {"type": "array", "items": {"type": "string"}, "minItems": 5, "maxItems": 5},
                    "description": {"type": "string"},
                    "backend_search_terms": {"type": "array", "items": {"type": "string"}, "minItems": 6, "maxItems": 20},
                    "structured_attributes": {"type": "object", "additionalProperties": {"type": "string"}},
                },
            },
            "ebay": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "item_specifics", "condition", "listing_notes"],
                "properties": {
                    "title": {"type": "string"},
                    "item_specifics": {"type": "object", "additionalProperties": {"type": "string"}},
                    "condition": {"type": "string"},
                    "listing_notes": {"type": "string"},
                },
            },
            "etsy": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "description", "tags", "materials", "occasion", "seo_keywords"],
                "properties": {
                    "title": {"type": "string"},
                    "description": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 8, "maxItems": 13},
                    "materials": {"type": "array", "items": {"type": "string"}, "minItems": 1, "maxItems": 8},
                    "occasion": {"type": "string"},
                    "seo_keywords": {"type": "array", "items": {"type": "string"}, "minItems": 6, "maxItems": 16},
                },
            },
            "tiktok": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "social_description", "hashtags"],
                "properties": {
                    "title": {"type": "string"},
                    "social_description": {"type": "string"},
                    "hashtags": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 8},
                },
            },
            "shopify": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "body_html", "tags", "product_type", "seo_title", "seo_description"],
                "properties": {
                    "title": {"type": "string"},
                    "body_html": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 12},
                    "product_type": {"type": "string"},
                    "seo_title": {"type": "string"},
                    "seo_description": {"type": "string"},
                },
            },
        },
    }

    def __init__(self, openai_service: OpenAIService | None = None) -> None:
        self.openai_service = openai_service

    async def process(
        self,
        *,
        product: ProductPipelineResponse,
        research: MarketResearchBundleResponse,
        seo: SeoInsightsResponse,
        pricing: PricingInsightsResponse,
        marketplaces: list[str] | None,
        optimize_core: bool,
    ) -> dict[str, object]:
        fallback = {
            "core": product.core.model_dump(),
            "amazon": product.amazon.model_dump(),
            "ebay": product.ebay.model_dump(),
            "etsy": product.etsy.model_dump(),
            "tiktok": product.tiktok.model_dump(),
            "shopify": product.shopify.model_dump(),
        }
        if self.openai_service is None:
            return fallback

        try:
            return await self.openai_service.generate_structured_output(
                system_prompt=(
                    "You are a senior marketplace product optimization agent. "
                    "You receive an already generated product record plus real or heuristic market research, SEO signals, and pricing guidance. "
                    "Perform a second-pass optimization on product data only. Do not mention AI or internal processing. "
                    "Preserve factual accuracy. Do not invent certifications, measurements, or product claims that are not supported. "
                    "Only improve data quality, SEO relevance, attribute clarity, and pricing language. "
                    "If optimize_core is false, keep the core section materially unchanged. "
                    "If marketplaces is provided, only optimize those marketplace sections and keep the others materially unchanged. "
                    "Return only JSON that matches the schema."
                ),
                user_payload={
                    "current_product": product.model_dump(exclude={"images", "intelligence"}),
                    "research": research.model_dump(),
                    "seo": seo.model_dump(),
                    "pricing": pricing.model_dump(),
                    "marketplaces": marketplaces or ["amazon", "ebay", "tiktok", "shopify"],
                    "optimize_core": optimize_core,
                },
                schema_name="product_optimization",
                schema=self._SCHEMA,
            )
        except OpenAIServiceError:
            return fallback

    @staticmethod
    def coerce_core(value: object, fallback: CoreProductResponse) -> CoreProductResponse:
        if not isinstance(value, dict):
            return fallback
        try:
            return CoreProductResponse.model_validate(value)
        except Exception:
            return fallback

    @staticmethod
    def coerce_amazon(value: object, fallback: AmazonResponse) -> AmazonResponse:
        if not isinstance(value, dict):
            return fallback
        try:
            return AmazonResponse.model_validate(value)
        except Exception:
            return fallback

    @staticmethod
    def coerce_ebay(value: object, fallback: EbayResponse) -> EbayResponse:
        if not isinstance(value, dict):
            return fallback
        try:
            return EbayResponse.model_validate(value)
        except Exception:
            return fallback

    @staticmethod
    def coerce_etsy(value: object, fallback: EtsyResponse) -> EtsyResponse:
        if not isinstance(value, dict):
            return fallback
        try:
            return EtsyResponse.model_validate(value)
        except Exception:
            return fallback

    @staticmethod
    def coerce_tiktok(value: object, fallback: TikTokResponse) -> TikTokResponse:
        if not isinstance(value, dict):
            return fallback
        try:
            return TikTokResponse.model_validate(value)
        except Exception:
            return fallback

    @staticmethod
    def coerce_shopify(value: object, fallback: ShopifyResponse) -> ShopifyResponse:
        if not isinstance(value, dict):
            return fallback
        try:
            return ShopifyResponse.model_validate(value)
        except Exception:
            return fallback
