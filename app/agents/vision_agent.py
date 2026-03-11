from __future__ import annotations

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

    def __init__(self, image_service: ImageService | None = None) -> None:
        self.image_service = image_service or ImageService()

    async def process(self, image: ImagePayload) -> VisionResponse:
        lowered_name = image.filename.lower()
        product_type = self._detect_product_type(lowered_name)
        palette = self.image_service.extract_palette(image)
        brightness = self.image_service.describe_brightness(image)
        background_removed = self.image_service.remove_background(image)

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
