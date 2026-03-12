from __future__ import annotations

from app.schemas.response import AmazonResponse, CoreProductResponse
from app.services.ollama_service import OllamaService, OllamaServiceError


class AmazonAgent:
    def __init__(self, ollama_service: OllamaService | None = None) -> None:
        self.ollama_service = ollama_service

    async def process(self, core_data: CoreProductResponse) -> AmazonResponse:
        fallback = AmazonResponse(
            title=self._build_title(core_data),
            bullet_points=self._build_bullet_points(core_data),
            description=self._build_description(core_data),
            backend_search_terms=self._build_search_terms(core_data),
            structured_attributes=core_data.attributes,
        )
        if self.ollama_service is None:
            return fallback

        prompt = (
            "You are an Amazon listing agent. Return only valid JSON with keys title, "
            "bullet_points, description, backend_search_terms, structured_attributes.\n"
            f"Core product: {core_data.model_dump()}\n"
        )
        try:
            result = await self.ollama_service.generate_json(prompt=prompt)
        except OllamaServiceError:
            return fallback

        data = result.parsed
        return AmazonResponse(
            title=str(data.get("title", fallback.title)),
            bullet_points=self._coerce_list(data.get("bullet_points"), fallback.bullet_points),
            description=str(data.get("description", fallback.description)),
            backend_search_terms=self._coerce_list(
                data.get("backend_search_terms"), fallback.backend_search_terms
            ),
            structured_attributes=self._coerce_dict(
                data.get("structured_attributes"), fallback.structured_attributes
            ),
        )

    @staticmethod
    def _build_title(core_data: CoreProductResponse) -> str:
        title_parts = [
            core_data.normalized_title,
            core_data.attributes.get("color"),
            core_data.attributes.get("material"),
            core_data.category,
        ]
        compact = [part for part in title_parts if part]
        return " | ".join(compact[:4])

    @staticmethod
    def _build_bullet_points(core_data: CoreProductResponse) -> list[str]:
        bullet_points = [
            core_data.product_summary,
            f"Category focus: {core_data.category}",
            f"Visual confidence score: {core_data.vision_confidence:.2f}",
        ]
        bullet_points.extend(core_data.features[:2])
        return bullet_points[:5]

    @staticmethod
    def _build_description(core_data: CoreProductResponse) -> str:
        features = "; ".join(core_data.features)
        return (
            f"{core_data.normalized_title} is prepared for Amazon with a normalized "
            f"{core_data.category.lower()} schema. Key facts: {features}."
        )

    @staticmethod
    def _build_search_terms(core_data: CoreProductResponse) -> list[str]:
        search_terms = {
            core_data.normalized_title.lower(),
            core_data.product_type,
            core_data.category.lower(),
            *[value.lower() for value in core_data.attributes.values()],
        }
        return sorted(search_terms)

    @staticmethod
    def _coerce_list(value: object, fallback: list[str]) -> list[str]:
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            return value
        return fallback

    @staticmethod
    def _coerce_dict(value: object, fallback: dict[str, str]) -> dict[str, str]:
        if isinstance(value, dict) and all(
            isinstance(key, str) and isinstance(item, str) for key, item in value.items()
        ):
            return value
        return fallback
