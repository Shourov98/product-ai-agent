from __future__ import annotations

import re

from app.schemas.response import CoreProductResponse, VisionResponse


class CoreAgent:
    async def process(self, title: str, vision_data: VisionResponse) -> CoreProductResponse:
        normalized_title = self._normalize_title(title)
        attributes = self._merge_attributes(vision_data)
        category = self._build_category(vision_data.product_type)
        features = self._build_features(normalized_title, attributes, vision_data)
        summary = self._build_summary(normalized_title, category, attributes)

        return CoreProductResponse(
            normalized_title=normalized_title,
            category=category,
            product_type=vision_data.product_type,
            product_summary=summary,
            features=features,
            attributes=attributes,
            source_title=title,
            vision_confidence=vision_data.confidence,
        )

    @staticmethod
    def _normalize_title(title: str) -> str:
        cleaned = re.sub(r"\s+", " ", title.strip())
        return cleaned.title()

    @staticmethod
    def _merge_attributes(vision_data: VisionResponse) -> dict[str, str]:
        merged: dict[str, str] = {}
        for attribute in vision_data.attributes:
            merged.setdefault(attribute.name, attribute.value)
        if vision_data.image_analysis.dominant_palette and "color" not in merged:
            merged["color"] = vision_data.image_analysis.dominant_palette[0]
        return merged

    @staticmethod
    def _build_category(product_type: str) -> str:
        if product_type == "running shoes":
            return "Footwear"
        if product_type == "hoodie":
            return "Apparel"
        if product_type == "office chair":
            return "Furniture"
        if product_type == "water bottle":
            return "Hydration"
        return "General Merchandise"

    @staticmethod
    def _build_features(
        normalized_title: str,
        attributes: dict[str, str],
        vision_data: VisionResponse,
    ) -> list[str]:
        features = [f"Product type: {vision_data.product_type}"]
        if "material" in attributes:
            features.append(f"Material: {attributes['material']}")
        if "color" in attributes:
            features.append(f"Primary color: {attributes['color']}")
        features.append(f"Visual style: {attributes.get('style', 'standard')}")
        if normalized_title.lower() not in vision_data.product_type:
            features.append(f"Source title preserved: {normalized_title}")
        return features

    @staticmethod
    def _build_summary(
        normalized_title: str,
        category: str,
        attributes: dict[str, str],
    ) -> str:
        descriptive_bits = [value for key, value in attributes.items() if key in {"color", "material", "style"}]
        descriptor = ", ".join(descriptive_bits) if descriptive_bits else "general use"
        return f"{normalized_title} is a {category.lower()} product with {descriptor} characteristics."
