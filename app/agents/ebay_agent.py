from __future__ import annotations

from app.schemas.response import CoreProductResponse, EbayResponse
from app.services.ollama_service import OllamaService, OllamaServiceError


class EbayAgent:
    def __init__(self, ollama_service: OllamaService | None = None) -> None:
        self.ollama_service = ollama_service

    async def process(self, core_data: CoreProductResponse) -> EbayResponse:
        fallback = EbayResponse(
            title=self._build_title(core_data)[:80],
            item_specifics={
                "Brand": "Unbranded",
                "Type": core_data.product_type.title(),
                **{key.title(): value.title() for key, value in core_data.attributes.items()},
            },
            condition=self._map_condition(core_data),
            listing_notes=(
                f"Normalized from source title '{core_data.source_title}' "
                f"with confidence {core_data.vision_confidence:.2f}."
            ),
        )
        if self.ollama_service is None:
            return fallback

        prompt = (
            "You are an eBay listing agent. Return only valid JSON with keys title, "
            "item_specifics, condition, listing_notes.\n"
            f"Core product: {core_data.model_dump()}\n"
        )
        try:
            result = await self.ollama_service.generate_json(prompt=prompt)
        except OllamaServiceError:
            return fallback

        data = result.parsed
        specifics = data.get("item_specifics")
        return EbayResponse(
            title=str(data.get("title", fallback.title))[:80],
            item_specifics=specifics if isinstance(specifics, dict) else fallback.item_specifics,
            condition=str(data.get("condition", fallback.condition)),
            listing_notes=str(data.get("listing_notes", fallback.listing_notes)),
        )

    @staticmethod
    def _build_title(core_data: CoreProductResponse) -> str:
        parts = [
            core_data.normalized_title,
            core_data.attributes.get("color", "").title(),
            core_data.product_type.title(),
        ]
        return " - ".join(part for part in parts if part)

    @staticmethod
    def _map_condition(core_data: CoreProductResponse) -> str:
        if core_data.vision_confidence >= 0.85:
            return "New"
        return "New other"
