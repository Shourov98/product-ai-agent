from __future__ import annotations

from app.schemas.response import CoreProductResponse, EbayResponse


class EbayAgent:
    async def process(self, core_data: CoreProductResponse) -> EbayResponse:
        title = self._build_title(core_data)
        return EbayResponse(
            title=title[:80],
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
