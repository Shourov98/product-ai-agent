from __future__ import annotations

from app.schemas.response import AmazonResponse, CoreProductResponse


class AmazonAgent:
    async def process(self, core_data: CoreProductResponse) -> AmazonResponse:
        title = self._build_title(core_data)
        return AmazonResponse(
            title=title,
            bullet_points=self._build_bullet_points(core_data),
            description=self._build_description(core_data),
            backend_search_terms=self._build_search_terms(core_data),
            structured_attributes=core_data.attributes,
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
