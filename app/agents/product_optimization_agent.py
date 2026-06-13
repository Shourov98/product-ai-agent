from __future__ import annotations

from app.schemas.response import (
    AmazonResponse,
    CoreProductResponse,
    EbayResponse,
    EtsyResponse,
    MarketResearchBundleResponse,
    ProductPipelineResponse,
    SeoInsightsResponse,
    ShopifyResponse,
    TikTokResponse,
)
from app.services.openai_service import OpenAIService, OpenAIServiceError
from app.utils.prompts import PromptRegistry
from app.utils.product_text import sentence_case_summary, title_keywords, unique_strings



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
                    "audit": {"type": "object", "nullable": True},
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
                    "audit": {"type": "object", "nullable": True},
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
                    "audit": {"type": "object", "nullable": True},
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
                    "audit": {"type": "object", "nullable": True},
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
                    "audit": {"type": "object", "nullable": True},
                },
            },
            "shopify": {
                "type": "object",
                "additionalProperties": False,
                "required": ["title", "body_html", "tags", "product_type", "seo_title", "seo_description", "category", "metafields"],
                "properties": {
                    "title": {"type": "string"},
                    "body_html": {"type": "string"},
                    "tags": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 12},
                    "product_type": {"type": "string"},
                    "seo_title": {"type": "string"},
                    "seo_description": {"type": "string"},
                    "category": {"type": "string"},
                    "metafields": {"type": "object", "additionalProperties": {"type": "string"}},
                    "audit": {"type": "object", "nullable": True},
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
        marketplaces: list[str] | None,
        optimize_core: bool,
    ) -> dict[str, object]:
        fallback = self._build_dataset_optimized_fallback(
            product=product,
            research=research,
            seo=seo,
            marketplaces=marketplaces,
            optimize_core=optimize_core,
        )
        if self.openai_service is None:
            return fallback

        try:
            return await self.openai_service.generate_structured_output(
                    system_prompt=PromptRegistry.get_optimization_prompt(),
                user_payload={
                    "current_product": product.model_dump(exclude={"images", "intelligence"}),
                    "research": research.model_dump(),
                    "seo": seo.model_dump(),
                    "marketplaces": marketplaces or ["amazon", "ebay", "tiktok", "shopify"],
                    "optimize_core": optimize_core,
                },
                schema_name="product_optimization",
                schema=self._SCHEMA,
            )
        except OpenAIServiceError:
            return fallback

    def _build_dataset_optimized_fallback(
        self,
        *,
        product: ProductPipelineResponse,
        research: MarketResearchBundleResponse,
        seo: SeoInsightsResponse,
        marketplaces: list[str] | None,
        optimize_core: bool,
    ) -> dict[str, object]:
        selected = set(marketplaces or ["amazon", "ebay", "etsy", "tiktok", "shopify"])
        core = product.core.model_copy(deep=True)
        amazon = product.amazon.model_copy(deep=True)
        ebay = product.ebay.model_copy(deep=True)
        etsy = product.etsy.model_copy(deep=True)
        tiktok = product.tiktok.model_copy(deep=True)
        shopify = product.shopify.model_copy(deep=True)

        if optimize_core:
            core = self._optimize_core(core, research.amazon, seo)
        if "amazon" in selected:
            amazon = self._optimize_amazon(amazon, core, research.amazon, seo)
        if "ebay" in selected:
            ebay = self._optimize_ebay(ebay, core, research.ebay, seo)
        if "etsy" in selected:
            etsy = self._optimize_etsy(etsy, core, research.etsy, seo)
        if "tiktok" in selected:
            tiktok = self._optimize_tiktok(tiktok, core, research.tiktok, seo)
        if "shopify" in selected:
            shopify = self._optimize_shopify(shopify, core, research.shopify, seo)

        return {
            "core": core.model_dump(),
            "amazon": amazon.model_dump(),
            "ebay": ebay.model_dump(),
            "etsy": etsy.model_dump(),
            "tiktok": tiktok.model_dump(),
            "shopify": shopify.model_dump(),
        }

    def _optimize_core(
        self,
        core: CoreProductResponse,
        amazon_research,
        seo: SeoInsightsResponse,
    ) -> CoreProductResponse:
        attributes = dict(core.attributes)
        if amazon_research.similar_listings:
            listing_attrs = amazon_research.similar_listings[0].attributes
            if "category" in listing_attrs and core.category == "General Merchandise":
                category_value = listing_attrs.get("category", "").strip()
                if category_value:
                    core = core.model_copy(update={"category": category_value})
        if "search_terms" not in attributes and seo.primary_keywords:
            attributes["search_terms"] = ", ".join(seo.primary_keywords[:4])

        title_parts = [core.normalized_title]
        for term in seo.title_terms[:2]:
            if term.lower() not in core.normalized_title.lower():
                title_parts.append(term.title())
        normalized_title = " ".join(title_parts)[:200]
        features = unique_strings(
            [
                *core.features,
                *[
                    f"Aligned to Amazon catalog signals around {term}."
                    for term in amazon_research.keyword_signals[:2]
                ],
            ],
            limit=8,
        )
        summary = sentence_case_summary(
            [
                core.product_summary.rstrip("."),
                "Optimized against similar Amazon dataset listings for stronger search alignment",
            ]
        )
        return core.model_copy(
            update={
                "normalized_title": normalized_title,
                "attributes": attributes,
                "features": features,
                "product_summary": summary,
            }
        )

    def _optimize_amazon(
        self,
        amazon: AmazonResponse,
        core: CoreProductResponse,
        research,
        seo: SeoInsightsResponse,
    ) -> AmazonResponse:
        title_parts = [core.normalized_title]
        title_parts.extend(keyword.title() for keyword in seo.marketplace_keywords.get("amazon", [])[:3] if keyword.lower() not in core.normalized_title.lower())
        title = " ".join(unique_strings(title_parts, limit=4))[:200]
        search_terms = unique_strings(
            [*amazon.backend_search_terms, *research.keyword_signals, *seo.marketplace_keywords.get("amazon", [])],
            limit=20,
        )
        bullet_points = unique_strings(
            [
                *amazon.bullet_points,
                *[
                    observation.rstrip(".")
                    for listing in research.similar_listings[:2]
                    for observation in listing.observations[:1]
                ],
            ],
            limit=5,
        )
        while len(bullet_points) < 5:
            bullet_points.append(f"Optimized for Amazon dataset demand patterns around {core.product_type}.")
        description = sentence_case_summary([amazon.description.rstrip("."), "Structured for Amazon search and conversion quality"])
        structured_attributes = dict(amazon.structured_attributes)
        for listing in research.similar_listings[:1]:
            for key, value in listing.attributes.items():
                structured_attributes.setdefault(key, value)
        return amazon.model_copy(
            update={
                "title": title,
                "bullet_points": bullet_points[:5],
                "description": description,
                "backend_search_terms": search_terms,
                "structured_attributes": structured_attributes,
            }
        )

    def _optimize_ebay(self, ebay: EbayResponse, core: CoreProductResponse, research, seo: SeoInsightsResponse) -> EbayResponse:
        title_tokens = unique_strings([core.normalized_title, *seo.marketplace_keywords.get("ebay", [])], limit=3)
        title = " - ".join(title_tokens)[:80]
        specifics = dict(ebay.item_specifics)
        notes = sentence_case_summary([ebay.listing_notes.rstrip("."), "Tuned using Amazon dataset product signals for keyword coverage"])
        return ebay.model_copy(update={"title": title, "item_specifics": specifics, "listing_notes": notes})

    def _optimize_etsy(self, etsy: EtsyResponse, core: CoreProductResponse, research, seo: SeoInsightsResponse) -> EtsyResponse:
        tags = unique_strings([*etsy.tags, *seo.marketplace_keywords.get("etsy", []), *research.keyword_signals], limit=13)
        description = sentence_case_summary([etsy.description.rstrip("."), "Optimized for Etsy search relevance"])
        return etsy.model_copy(update={"tags": tags, "description": description})

    def _optimize_tiktok(self, tiktok: TikTokResponse, core: CoreProductResponse, research, seo: SeoInsightsResponse) -> TikTokResponse:
        hashtags = unique_strings([*tiktok.hashtags, *[f"#{term.title().replace(' ', '')}" for term in seo.marketplace_keywords.get("tiktok", [])[:3]]], limit=8)
        social_description = sentence_case_summary([tiktok.social_description.rstrip("."), "Merchandising angle supports a stronger social commerce posture"])
        return tiktok.model_copy(update={"hashtags": hashtags, "social_description": social_description})

    def _optimize_shopify(self, shopify: ShopifyResponse, core: CoreProductResponse, research, seo: SeoInsightsResponse) -> ShopifyResponse:
        tags = unique_strings([*shopify.tags, *seo.marketplace_keywords.get("shopify", []), *research.keyword_signals[:3]], limit=12)
        seo_title_terms = unique_strings([core.normalized_title, *seo.primary_keywords[:2]], limit=2)
        seo_title = " | ".join(term.title() if term != core.normalized_title else term for term in seo_title_terms)[:70]
        seo_description = sentence_case_summary([shopify.seo_description.rstrip("."), "Optimized for storefront clarity and conversion"])[:180]
        metafields = dict(shopify.metafields)
        metafields.setdefault("color", core.attributes.get("color", "").title() if core.attributes.get("color") else "")
        metafields.setdefault("footwear_material", core.attributes.get("material", "").title() if core.attributes.get("material") else "")
        metafields = {key: value for key, value in metafields.items() if value}
        return shopify.model_copy(
            update={
                "tags": tags,
                "seo_title": seo_title,
                "seo_description": seo_description,
                "category": shopify.category or core.category,
                "metafields": metafields,
            }
        )

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
