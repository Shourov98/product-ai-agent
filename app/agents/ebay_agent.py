from __future__ import annotations

from app.schemas.response import CoreProductResponse, EbayResponse, MarketplacePricingResponse, MarketplaceResearchResponse, SeoInsightsResponse
from app.services.openai_service import OpenAIService, OpenAIServiceError
from app.services.ollama_service import OllamaService, OllamaServiceError


class EbayAgent:
    _SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "item_specifics", "condition", "listing_notes"],
        "properties": {
            "title": {"type": "string"},
            "item_specifics": {"type": "object", "additionalProperties": {"type": "string"}},
            "condition": {"type": "string"},
            "listing_notes": {"type": "string"},
        },
    }

    def __init__(
        self,
        ollama_service: OllamaService | None = None,
        openai_service: OpenAIService | None = None,
    ) -> None:
        self.ollama_service = ollama_service
        self.openai_service = openai_service

    async def process(
        self,
        core_data: CoreProductResponse,
        *,
        research: MarketplaceResearchResponse | None = None,
        seo: SeoInsightsResponse | None = None,
        pricing: MarketplacePricingResponse | None = None,
    ) -> EbayResponse:
        fallback = EbayResponse(
            title=self._build_title(core_data, seo)[:80],
            item_specifics=self._build_item_specifics(core_data, research),
            condition=self._map_condition(core_data),
            listing_notes=self._build_listing_notes(core_data, pricing),
        )
        if self.openai_service is not None:
            try:
                data = await self.openai_service.generate_structured_output(
                    system_prompt=(
                        "You are a senior eBay listing specialist. Return concise, sale-ready listing data. "
                        "Do not mention AI, normalization, or confidence scores."
                    ),
                    user_payload={
                        "core_product": core_data.model_dump(),
                        "research": research.model_dump() if research is not None else None,
                        "seo": seo.model_dump() if seo is not None else None,
                        "pricing": pricing.model_dump() if pricing is not None else None,
                    },
                    schema_name="ebay_listing",
                    schema=self._SCHEMA,
                )
                return self._from_data(data, fallback)
            except OpenAIServiceError:
                pass

        if self.ollama_service is None:
            return fallback

        prompt = (
            "You are a senior eBay listing agent. Return only valid JSON with keys title, "
            "item_specifics, condition, listing_notes.\n"
            "Do not mention AI, confidence, normalization, or internal processing.\n"
            f"Core product: {core_data.model_dump()}\n"
        )
        try:
            result = await self.ollama_service.generate_json(prompt=prompt)
        except OllamaServiceError:
            return fallback

        return self._from_data(result.parsed, fallback)

    def _from_data(self, data: dict[str, object], fallback: EbayResponse) -> EbayResponse:
        specifics = data.get("item_specifics")
        return EbayResponse(
            title=str(data.get("title", fallback.title))[:80],
            item_specifics=specifics if isinstance(specifics, dict) else fallback.item_specifics,
            condition=str(data.get("condition", fallback.condition)),
            listing_notes=str(data.get("listing_notes", fallback.listing_notes)),
        )

    @staticmethod
    def _build_title(core_data: CoreProductResponse, seo: SeoInsightsResponse | None) -> str:
        parts = [
            core_data.normalized_title,
            core_data.attributes.get("color", "").title(),
            seo.marketplace_keywords.get("ebay", [None])[0] if seo is not None else None,
        ]
        return " - ".join(part for part in parts if part)

    @staticmethod
    def _build_item_specifics(
        core_data: CoreProductResponse,
        research: MarketplaceResearchResponse | None,
    ) -> dict[str, str]:
        specifics = {
            "Brand": core_data.attributes.get("brand", "Unbranded"),
            "Type": core_data.product_type.title(),
            **{key.title(): value.title() for key, value in core_data.attributes.items()},
        }
        if research is not None and research.similar_listings:
            for key, value in research.similar_listings[0].attributes.items():
                specifics.setdefault(str(key), str(value))
        return specifics

    @staticmethod
    def _build_listing_notes(core_data: CoreProductResponse, pricing: MarketplacePricingResponse | None) -> str:
        price_sentence = (
            f"Recommended pricing sits near {pricing.recommended:.2f} USD for a {pricing.strategy} listing."
            if pricing is not None
            else ""
        )
        return f"{core_data.product_summary} Suitable for a clean eBay listing with clear item specifics. {price_sentence}".strip()

    @staticmethod
    def _map_condition(core_data: CoreProductResponse) -> str:
        if core_data.vision_confidence >= 0.85:
            return "New"
        return "New other"
