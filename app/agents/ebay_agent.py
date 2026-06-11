from __future__ import annotations

from app.schemas.response import CoreProductResponse, EbayResponse, MarketplaceResearchResponse, SeoInsightsResponse
from app.services.openai_service import OpenAIService, OpenAIServiceError
from app.services.ollama_service import OllamaService, OllamaServiceError
from app.utils.prompts import PromptRegistry
from app.utils.product_text import unique_strings



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
    ) -> EbayResponse:
        fallback = EbayResponse(
            title=self._build_title(core_data, seo)[:80],
            item_specifics=self._build_item_specifics(core_data, research),
            condition=self._map_condition(core_data),
            listing_notes=self._build_listing_notes(core_data),
        )
        if self.openai_service is not None:
            try:
                data = await self.openai_service.generate_structured_output(
                    system_prompt=PromptRegistry.get_copy_prompt("ebay"),
                    user_payload={
                        "core_product": core_data.model_dump(),
                        "research": research.model_dump() if research is not None else None,
                        "seo": seo.model_dump() if seo is not None else None,
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
        from datetime import datetime, UTC
        from app.schemas.response import AgentAuditMetadata
        model_name = self.openai_service.model if self.openai_service and self.openai_service.enabled else "ollama"
        audit_meta = AgentAuditMetadata(
            prompt_version="ebay-copy.v2",
            model_version=model_name,
            timestamp=datetime.now(UTC).isoformat(),
            validation_passed=True,
        )
        return EbayResponse(
            title=str(data.get("title", fallback.title))[:80],
            item_specifics=self._coerce_item_specifics(data.get("item_specifics"), fallback.item_specifics),
            condition=str(data.get("condition", fallback.condition)),
            listing_notes=str(data.get("listing_notes", fallback.listing_notes)),
            audit=audit_meta,
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
    def _coerce_item_specifics(value: object, fallback: dict[str, str]) -> dict[str, str]:
        if not isinstance(value, dict):
            return fallback

        specifics: dict[str, str] = {}
        for key, item in value.items():
            key_text = str(key).strip()
            if not key_text:
                continue

            if isinstance(item, str):
                text = item.strip()
            elif isinstance(item, list):
                text = ", ".join(unique_strings((str(entry) for entry in item), limit=6))
            elif isinstance(item, dict):
                text = ", ".join(
                    f"{str(nested_key).strip()}: {str(nested_value).strip()}"
                    for nested_key, nested_value in item.items()
                    if str(nested_key).strip() and str(nested_value).strip()
                )
            else:
                text = str(item).strip()

            if text:
                specifics[key_text] = text

        return specifics or fallback

    @staticmethod
    def _build_listing_notes(core_data: CoreProductResponse) -> str:
        return f"{core_data.product_summary} Suitable for a clean eBay listing with clear item specifics.".strip()

    @staticmethod
    def _map_condition(core_data: CoreProductResponse) -> str:
        if core_data.vision_confidence >= 0.85:
            return "New"
        return "New other"
