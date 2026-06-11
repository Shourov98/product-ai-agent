from __future__ import annotations

from app.schemas.response import CoreProductResponse, EtsyResponse, MarketplaceResearchResponse, SeoInsightsResponse
from app.services.openai_service import OpenAIService, OpenAIServiceError
from app.services.ollama_service import OllamaService, OllamaServiceError
from app.utils.prompts import PromptRegistry
from app.utils.product_text import title_keywords, unique_strings



class EtsyAgent:
    _SCHEMA = {
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
    ) -> EtsyResponse:
        fallback = EtsyResponse(
            title=self._build_title(core_data, seo),
            description=self._build_description(core_data),
            tags=self._build_tags(core_data, research, seo),
            materials=self._build_materials(core_data, research),
            occasion=self._build_occasion(core_data),
            seo_keywords=self._build_keywords(core_data, research, seo),
        )
        if self.openai_service is not None:
            try:
                data = await self.openai_service.generate_structured_output(
                    system_prompt=PromptRegistry.get_copy_prompt("etsy"),
                    user_payload={
                        "core_product": core_data.model_dump(),
                        "research": research.model_dump() if research is not None else None,
                        "seo": seo.model_dump() if seo is not None else None,
                    },
                    schema_name="etsy_listing",
                    schema=self._SCHEMA,
                )
                return self._from_data(data, fallback)
            except OpenAIServiceError:
                pass

        if self.ollama_service is None:
            return fallback

        prompt = (
            "You are a senior Etsy listing agent. Return only valid JSON with keys "
            "title, description, tags, materials, occasion, seo_keywords.\n"
            "Do not mention AI, confidence, or internal metadata.\n"
            f"Core product: {core_data.model_dump()}\n"
        )
        try:
            result = await self.ollama_service.generate_json(prompt=prompt)
        except OllamaServiceError:
            return fallback
        return self._from_data(result.parsed, fallback)

    def _from_data(self, data: dict[str, object], fallback: EtsyResponse) -> EtsyResponse:
        from datetime import datetime, UTC
        from app.schemas.response import AgentAuditMetadata
        model_name = self.openai_service.model if self.openai_service and self.openai_service.enabled else "ollama"
        audit_meta = AgentAuditMetadata(
            prompt_version="etsy-copy.v2",
            model_version=model_name,
            timestamp=datetime.now(UTC).isoformat(),
            validation_passed=True,
        )
        return EtsyResponse(
            title=str(data.get("title", fallback.title))[:140],
            description=str(data.get("description", fallback.description)),
            tags=self._coerce_list(data.get("tags"), fallback.tags, limit=13),
            materials=self._coerce_list(data.get("materials"), fallback.materials, limit=8),
            occasion=str(data.get("occasion", fallback.occasion))[:120],
            seo_keywords=self._coerce_list(data.get("seo_keywords"), fallback.seo_keywords, limit=16),
            audit=audit_meta,
        )


    @staticmethod
    def _build_title(core_data: CoreProductResponse, seo: SeoInsightsResponse | None) -> str:
        color = core_data.attributes.get("color")
        material = core_data.attributes.get("material")
        keyword = seo.marketplace_keywords.get("etsy", [None])[0] if seo is not None else None
        parts = [color.title() if color else None, material.title() if material else None, core_data.normalized_title, keyword.title() if keyword else None]
        return " ".join(part for part in parts if part)[:140]

    @staticmethod
    def _build_description(core_data: CoreProductResponse) -> str:
        return f"{core_data.product_summary} Features: {' '.join(core_data.features[:4])}.".strip()

    @staticmethod
    def _build_tags(
        core_data: CoreProductResponse,
        research: MarketplaceResearchResponse | None,
        seo: SeoInsightsResponse | None,
    ) -> list[str]:
        raw_tags = [
            core_data.normalized_title,
            core_data.product_type,
            core_data.category,
            *core_data.attributes.values(),
            *title_keywords(core_data.normalized_title),
        ]
        if research is not None:
            raw_tags.extend(research.keyword_signals[:5])
        if seo is not None:
            raw_tags.extend(seo.marketplace_keywords.get("etsy", []))
        return unique_strings(raw_tags, limit=13)

    @staticmethod
    def _build_materials(core_data: CoreProductResponse, research: MarketplaceResearchResponse | None) -> list[str]:
        materials = []
        material = core_data.attributes.get("material")
        if material:
            materials.append(material)
        if research is not None and research.similar_listings:
            research_material = research.similar_listings[0].attributes.get("material")
            if research_material:
                materials.append(research_material)
        return unique_strings(materials or ["mixed material"], limit=8)

    @staticmethod
    def _build_occasion(core_data: CoreProductResponse) -> str:
        category = core_data.category.lower()
        if "gift" in category or "fashion" in category:
            return "gift"
        if "drink" in category:
            return "everyday use"
        return "general occasion"

    @staticmethod
    def _build_keywords(
        core_data: CoreProductResponse,
        research: MarketplaceResearchResponse | None,
        seo: SeoInsightsResponse | None,
    ) -> list[str]:
        keywords = [
            core_data.normalized_title.lower(),
            core_data.product_type.lower(),
            *title_keywords(core_data.normalized_title),
            *[value.lower() for value in core_data.attributes.values()],
        ]
        if research is not None:
            keywords.extend(research.keyword_signals)
        if seo is not None:
            keywords.extend(seo.marketplace_keywords.get("etsy", []))
        return unique_strings(keywords, limit=16)

    @staticmethod
    def _coerce_list(value: object, fallback: list[str], *, limit: int) -> list[str]:
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            coerced = unique_strings(value, limit=limit)
            if coerced:
                return coerced
        return fallback
