from __future__ import annotations

from app.schemas.response import AmazonResponse, CoreProductResponse, MarketplaceResearchResponse, SeoInsightsResponse
from app.services.gemini_service import GeminiService, GeminiServiceError
from app.services.openai_service import OpenAIService, OpenAIServiceError
from app.utils.prompts import PromptRegistry

from app.utils.product_text import title_keywords, unique_strings


class AmazonAgent:
    _SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "bullet_points", "description", "backend_search_terms", "structured_attributes"],
        "properties": {
            "title": {"type": "string"},
            "bullet_points": {"type": "array", "items": {"type": "string"}, "minItems": 5, "maxItems": 5},
            "description": {"type": "string"},
            "backend_search_terms": {"type": "array", "items": {"type": "string"}, "minItems": 6, "maxItems": 20},
            "structured_attributes": {"type": "object", "additionalProperties": {"type": "string"}},
        },
    }

    def __init__(
        self,
        openai_service: OpenAIService | None = None,
        gemini_service: GeminiService | None = None,
    ) -> None:
        self.openai_service = openai_service
        self.gemini_service = gemini_service

    async def process(
        self,
        core_data: CoreProductResponse,
        *,
        research: MarketplaceResearchResponse | None = None,
        seo: SeoInsightsResponse | None = None,
    ) -> AmazonResponse:
        fallback = AmazonResponse(
            title=self._build_title(core_data, seo),
            bullet_points=self._build_bullet_points(core_data),
            description=self._build_description(core_data),
            backend_search_terms=self._build_search_terms(core_data, research, seo),
            structured_attributes=self._build_structured_attributes(core_data, research),
        )
        if self.gemini_service is not None:
            try:
                data = await self.gemini_service.generate_structured_output(
                    system_prompt=PromptRegistry.get_copy_prompt("amazon"),
                    user_payload={
                        "core_product": core_data.model_dump(),
                        "research": research.model_dump() if research is not None else None,
                        "seo": seo.model_dump() if seo is not None else None,
                    },
                    use_google_search=True,
                    schema=self._SCHEMA,
                )
                return self._from_data(data, fallback, self.gemini_service.model)
            except GeminiServiceError:
                pass
        if self.openai_service is not None:
            try:
                data = await self.openai_service.generate_structured_output(
                    system_prompt=PromptRegistry.get_copy_prompt("amazon"),
                    user_payload={
                        "core_product": core_data.model_dump(),
                        "research": research.model_dump() if research is not None else None,
                        "seo": seo.model_dump() if seo is not None else None,
                    },
                    schema_name="amazon_listing",
                    schema=self._SCHEMA,
                )
                return self._from_data(data, fallback, self.openai_service.model)
            except OpenAIServiceError:
                pass

        return fallback

    def _from_data(self, data: dict[str, object], fallback: AmazonResponse, model_name: str) -> AmazonResponse:
        from datetime import datetime, UTC
        from app.schemas.response import AgentAuditMetadata
        audit_meta = AgentAuditMetadata(
            prompt_version="amazon-copy.v2",
            model_version=model_name,
            timestamp=datetime.now(UTC).isoformat(),
            validation_passed=True,
        )
        return AmazonResponse(
            title=str(data.get("title", fallback.title))[:200],
            bullet_points=self._coerce_list(data.get("bullet_points"), fallback.bullet_points),
            description=str(data.get("description", fallback.description)),
            backend_search_terms=self._coerce_list(data.get("backend_search_terms"), fallback.backend_search_terms),
            structured_attributes=self._coerce_dict(data.get("structured_attributes"), fallback.structured_attributes),
            audit=audit_meta,
        )


    @staticmethod
    def _build_title(core_data: CoreProductResponse, seo: SeoInsightsResponse | None) -> str:
        title_parts = [
            core_data.normalized_title,
            core_data.attributes.get("color"),
            core_data.attributes.get("material"),
            seo.primary_keywords[0].title() if seo is not None and seo.primary_keywords else None,
        ]
        compact = [str(part).title() if part != core_data.normalized_title else part for part in title_parts if part]
        return " ".join(compact[:4])[:200]

    @staticmethod
    def _build_bullet_points(core_data: CoreProductResponse) -> list[str]:
        bullets = list(core_data.features[:5])
        while len(bullets) < 5:
            bullets.append(f"Built for {core_data.category.lower()} merchandising and multi-channel catalog consistency.")
        return bullets[:5]

    @staticmethod
    def _build_description(core_data: CoreProductResponse) -> str:
        features = " ".join(core_data.features[:4])
        return f"{core_data.product_summary} Key highlights: {features}".strip()

    @staticmethod
    def _build_search_terms(
        core_data: CoreProductResponse,
        research: MarketplaceResearchResponse | None,
        seo: SeoInsightsResponse | None,
    ) -> list[str]:
        raw_terms = [
            core_data.normalized_title.lower(),
            core_data.product_type.lower(),
            core_data.category.lower(),
            *[value.lower() for value in core_data.attributes.values()],
            *title_keywords(core_data.normalized_title),
        ]
        if research is not None:
            raw_terms.extend(research.keyword_signals)
        if seo is not None:
            raw_terms.extend(seo.marketplace_keywords.get("amazon", []))
        return unique_strings(raw_terms, limit=20)

    @staticmethod
    def _build_structured_attributes(
        core_data: CoreProductResponse,
        research: MarketplaceResearchResponse | None,
    ) -> dict[str, str]:
        attributes = dict(core_data.attributes)
        if research is not None and research.similar_listings:
            for key, value in research.similar_listings[0].attributes.items():
                attributes.setdefault(key, value)
        return attributes

    @staticmethod
    def _coerce_list(value: object, fallback: list[str]) -> list[str]:
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            coerced = unique_strings(value, limit=20)
            if coerced:
                return coerced
        return fallback

    @staticmethod
    def _coerce_dict(value: object, fallback: dict[str, str]) -> dict[str, str]:
        if isinstance(value, dict) and all(
            isinstance(key, str) and isinstance(item, str) for key, item in value.items()
        ):
            return {str(key).strip(): str(item).strip() for key, item in value.items() if str(key).strip() and str(item).strip()}
        return fallback
