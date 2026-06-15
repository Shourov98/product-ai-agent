from __future__ import annotations

from app.schemas.response import CoreProductResponse, VisionResponse
from app.services.gemini_service import GeminiService, GeminiServiceError
from app.services.openai_service import OpenAIService, OpenAIServiceError
from app.utils.product_text import build_category, infer_product_type, normalize_title, sentence_case_summary, title_keywords, unique_strings


class CoreAgent:
    _SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "normalized_title",
            "category",
            "product_type",
            "product_summary",
            "features",
            "attributes",
        ],
        "properties": {
            "normalized_title": {"type": "string"},
            "category": {"type": "string"},
            "product_type": {"type": "string"},
            "product_summary": {"type": "string"},
            "features": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 8},
            "attributes": {"type": "object", "additionalProperties": {"type": "string"}},
        },
    }

    def __init__(
        self,
        openai_service: OpenAIService | None = None,
        gemini_service: GeminiService | None = None,
    ) -> None:
        self.openai_service = openai_service
        self.gemini_service = gemini_service

    async def process(self, title: str, vision_data: VisionResponse) -> CoreProductResponse:
        fallback = self._build_fallback(title, vision_data)
        provider = "fallback"
        if self.gemini_service is not None:
            try:
                data = await self.gemini_service.generate_structured_output(
                    system_prompt=self._build_openai_system_prompt(),
                    user_payload=self._build_user_payload(title, vision_data),
                    schema=self._SCHEMA,
                )
                provider = self.gemini_service.model
                return self._response_from_data(data, fallback, title, vision_data, provider)
            except GeminiServiceError:
                pass
        if self.openai_service is not None:
            try:
                data = await self.openai_service.generate_structured_output(
                    system_prompt=self._build_openai_system_prompt(),
                    user_payload=self._build_user_payload(title, vision_data),
                    schema_name="core_product",
                    schema=self._SCHEMA,
                )
                provider = self.openai_service.model
                return self._response_from_data(data, fallback, title, vision_data, provider)
            except OpenAIServiceError:
                pass

        return fallback

    def _response_from_data(
        self,
        data: dict[str, object],
        fallback: CoreProductResponse,
        title: str,
        vision_data: VisionResponse,
        model_name: str,
    ) -> CoreProductResponse:
        from datetime import datetime, UTC
        from app.schemas.response import AgentAuditMetadata
        audit_meta = AgentAuditMetadata(
            prompt_version="core.v4",
            model_version=model_name,
            timestamp=datetime.now(UTC).isoformat(),
            validation_passed=True,
        )
        return CoreProductResponse(
            normalized_title=str(data.get("normalized_title", fallback.normalized_title)),
            category=str(data.get("category", fallback.category)),
            product_type=str(data.get("product_type", fallback.product_type)),
            product_summary=str(data.get("product_summary", fallback.product_summary)),
            features=self._ensure_string_list(data.get("features"), fallback.features),
            attributes=self._ensure_string_dict(data.get("attributes"), fallback.attributes),
            source_title=title,
            vision_confidence=vision_data.confidence,
            audit=audit_meta,
        )


    def _build_fallback(self, title: str, vision_data: VisionResponse) -> CoreProductResponse:
        normalized_title = normalize_title(title)
        attributes = self._merge_attributes(vision_data)
        product_type = infer_product_type(title, vision_data.product_type, vision_data.image_analysis.filename)
        category = build_category(product_type)
        features = self._build_features(normalized_title, attributes, vision_data)
        summary = self._build_summary(normalized_title, category, attributes, product_type)

        return CoreProductResponse(
            normalized_title=normalized_title,
            category=category,
            product_type=product_type,
            product_summary=summary,
            features=features,
            attributes=attributes,
            source_title=title,
            vision_confidence=vision_data.confidence,
        )

    @staticmethod
    def _build_openai_system_prompt() -> str:
        from app.utils.prompts import PromptRegistry
        return PromptRegistry.get_core_prompt()

    @staticmethod
    def _build_user_payload(title: str, vision_data: VisionResponse) -> dict[str, object]:
        return {
            "source_title": title,
            "vision_product_type": vision_data.product_type,
            "vision_confidence": vision_data.confidence,
            "vision_attributes": [item.model_dump() for item in vision_data.attributes],
            "image_analysis": vision_data.image_analysis.model_dump(),
        }

    @staticmethod
    def _merge_attributes(vision_data: VisionResponse) -> dict[str, str]:
        merged: dict[str, str] = {}
        for attribute in vision_data.attributes:
            merged.setdefault(attribute.name, attribute.value)
        if vision_data.image_analysis.dominant_palette and "color" not in merged:
            merged["color"] = vision_data.image_analysis.dominant_palette[0]
        return merged

    @staticmethod
    def _build_features(
        normalized_title: str,
        attributes: dict[str, str],
        vision_data: VisionResponse,
    ) -> list[str]:
        features = []
        material = attributes.get("material")
        color = attributes.get("color")
        style = attributes.get("style")
        if material:
            features.append(f"Crafted with a {material} finish for everyday durability.")
        if color:
            features.append(f"Presented in a {color} colorway for a clear merchandising identity.")
        features.append(f"Designed as a {vision_data.product_type} with a versatile, easy-to-list profile.")
        if style:
            features.append(f"Visual styling leans {style}, making it suitable for modern marketplace presentation.")
        keywords = title_keywords(normalized_title)
        if keywords:
            features.append(f"Search-relevant title terms include {' '.join(keywords[:3])}.")
        return unique_strings(features, limit=6)

    @staticmethod
    def _build_summary(
        normalized_title: str,
        category: str,
        attributes: dict[str, str],
        product_type: str,
    ) -> str:
        descriptive_bits = [value for key, value in attributes.items() if key in {"color", "material", "style"}]
        descriptor = ", ".join(descriptive_bits[:3])
        parts = [
            f"{normalized_title} is positioned within {category.lower()}",
            f"built around a {product_type} use case",
            f"with {descriptor} cues" if descriptor else "with a broadly merchandisable presentation",
        ]
        return sentence_case_summary(parts)

    @staticmethod
    def _ensure_string_list(value: object, fallback: list[str]) -> list[str]:
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            normalized = unique_strings(value, limit=8)
            if normalized:
                return normalized
        return fallback

    @staticmethod
    def _ensure_string_dict(value: object, fallback: dict[str, str]) -> dict[str, str]:
        if isinstance(value, dict) and all(
            isinstance(key, str) and isinstance(item, str) for key, item in value.items()
        ):
            return {str(key).strip().lower(): str(item).strip() for key, item in value.items() if str(key).strip() and str(item).strip()}
        return fallback
