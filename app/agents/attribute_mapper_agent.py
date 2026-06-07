from __future__ import annotations

from app.schemas.response import CoreProductResponse, VisionResponse
from app.services.openai_service import OpenAIService, OpenAIServiceError


class AttributeMapperAgent:
    _SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["attributes", "category", "product_type"],
        "properties": {
            "attributes": {"type": "object", "additionalProperties": {"type": "string"}},
            "category": {"type": "string"},
            "product_type": {"type": "string"},
        },
    }

    _ALIASES = {
        "colour": "color",
        "fabric": "material",
        "finish type": "finish",
        "size name": "size",
    }

    def __init__(self, openai_service: OpenAIService | None = None) -> None:
        self.openai_service = openai_service

    async def process(
        self,
        core_data: CoreProductResponse,
        vision_data: VisionResponse,
    ) -> CoreProductResponse:
        fallback = self._build_fallback(core_data, vision_data)
        if self.openai_service is None:
            return fallback

        try:
            data = await self.openai_service.generate_structured_output(
                system_prompt=(
                    "You are a senior ecommerce attribute normalization specialist. "
                    "Normalize raw product attributes into consistent marketplace-ready fields without inventing unsupported facts. "
                    "Return only JSON matching the schema."
                ),
                user_payload={
                    "core_product": core_data.model_dump(),
                    "vision_data": vision_data.model_dump(),
                },
                schema_name="canonical_attributes",
                schema=self._SCHEMA,
            )
            return self._from_data(data, fallback)
        except OpenAIServiceError:
            return fallback

    def _build_fallback(self, core_data: CoreProductResponse, vision_data: VisionResponse) -> CoreProductResponse:
        attributes = {self._ALIASES.get(key.lower(), key.lower()): value.strip() for key, value in core_data.attributes.items()}
        if "color" not in attributes and vision_data.image_analysis.dominant_palette:
            attributes["color"] = vision_data.image_analysis.dominant_palette[0]
        if "style" not in attributes:
            brightness_to_style = {"light": "clean", "balanced": "balanced", "dark": "bold"}
            attributes["style"] = brightness_to_style.get(vision_data.image_analysis.brightness, "balanced")
        return core_data.model_copy(update={"attributes": attributes})

    def _from_data(self, data: dict[str, object], fallback: CoreProductResponse) -> CoreProductResponse:
        attributes = data.get("attributes")
        normalized_attributes = fallback.attributes
        if isinstance(attributes, dict):
            normalized_attributes = {
                self._ALIASES.get(str(key).strip().lower(), str(key).strip().lower()): str(value).strip()
                for key, value in attributes.items()
                if str(key).strip() and str(value).strip()
            }

        return fallback.model_copy(
            update={
                "attributes": normalized_attributes,
                "category": str(data.get("category", fallback.category)).strip() or fallback.category,
                "product_type": str(data.get("product_type", fallback.product_type)).strip() or fallback.product_type,
            }
        )
