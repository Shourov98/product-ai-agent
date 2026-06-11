from __future__ import annotations

from typing import TYPE_CHECKING
if TYPE_CHECKING:
    from app.services.openai_service import OpenAIService

from app.schemas.response import ExtractedAttribute, ImageAnalysis, VisionResponse
from app.services.image_service import ImagePayload, ImageService


class VisionAgent:
    _PRODUCT_KEYWORDS = {
        "shoe": "running shoes",
        "sneaker": "running shoes",
        "hoodie": "hoodie",
        "chair": "office chair",
        "bottle": "water bottle",
        "lamp": "table lamp",
        "bag": "travel bag",
        "watch": "watch",
    }
    _MATERIAL_KEYWORDS = {
        "leather": "leather",
        "mesh": "mesh",
        "cotton": "cotton",
        "wood": "wood",
        "metal": "metal",
        "plastic": "plastic",
        "glass": "glass",
    }
    _STYLE_KEYWORDS = {
        "sport": "sport",
        "classic": "classic",
        "modern": "modern",
        "casual": "casual",
        "luxury": "luxury",
        "minimal": "minimal",
    }

    def __init__(
        self,
        image_service: ImageService | None = None,
        openai_service: OpenAIService | None = None,
    ) -> None:
        self.image_service = image_service or ImageService()
        self.openai_service = openai_service

    async def process(self, image: ImagePayload) -> VisionResponse:
        palette = self.image_service.extract_palette(image)
        brightness = self.image_service.describe_brightness(image)
        background_removed = self.image_service.remove_background(image)

        # Try to use OpenAI Vision API
        if self.openai_service is not None and self.openai_service.enabled:
            try:
                res = await self.openai_service.analyze_image(
                    image_bytes=image.data,
                    mime_type=image.content_type,
                )
                product_type = str(res.get("product_type", "general product"))
                confidence = float(res.get("confidence", 0.9))
                raw_attrs = res.get("attributes") or {}

                attributes = []
                for name, value in raw_attrs.items():
                    attributes.append(
                        ExtractedAttribute(name=str(name), value=str(value), confidence=0.95)
                    )
                # Ensure style and color are present
                if "color" not in raw_attrs and palette:
                    attributes.append(ExtractedAttribute(name="color", value=palette[0], confidence=0.88))
                if "style" not in raw_attrs:
                    attributes.append(ExtractedAttribute(name="style", value=self._style_from_brightness(brightness), confidence=0.62))

                return VisionResponse(
                    product_type=product_type,
                    confidence=confidence,
                    attributes=attributes,
                    image_analysis=ImageAnalysis(
                        filename=image.filename,
                        content_type=image.content_type,
                        size_bytes=image.size_bytes,
                        brightness=brightness,
                        dominant_palette=palette,
                        background_removed=background_removed,
                    ),
                )
            except Exception:
                pass

        # Fallback to local regex-based logic
        lowered_name = image.filename.lower()
        product_type = self._detect_product_type(lowered_name)
        attributes = [
            ExtractedAttribute(name="color", value=color, confidence=0.88)
            for color in palette
        ]

        material = self._find_keyword(lowered_name, self._MATERIAL_KEYWORDS)
        if material:
            attributes.append(
                ExtractedAttribute(name="material", value=material, confidence=0.84)
            )

        style = self._find_keyword(lowered_name, self._STYLE_KEYWORDS)
        if style:
            attributes.append(
                ExtractedAttribute(name="style", value=style, confidence=0.8)
            )
        else:
            attributes.append(
                ExtractedAttribute(name="style", value=self._style_from_brightness(brightness), confidence=0.62)
            )

        return VisionResponse(
            product_type=product_type,
            confidence=0.9 if product_type != "general product" else 0.55,
            attributes=attributes,
            image_analysis=ImageAnalysis(
                filename=image.filename,
                content_type=image.content_type,
                size_bytes=image.size_bytes,
                brightness=brightness,
                dominant_palette=palette,
                background_removed=background_removed,
            ),
        )

    def _detect_product_type(self, lowered_name: str) -> str:
        for keyword, product_type in self._PRODUCT_KEYWORDS.items():
            if keyword in lowered_name:
                return product_type
        return "general product"

    @staticmethod
    def _find_keyword(lowered_name: str, keyword_map: dict[str, str]) -> str | None:
        for keyword, value in keyword_map.items():
            if keyword in lowered_name:
                return value
        return None

    @staticmethod
    def _style_from_brightness(brightness: str) -> str:
        if brightness == "dark":
            return "bold"
        if brightness == "light":
            return "clean"
        return "balanced"
