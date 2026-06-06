from __future__ import annotations

from app.schemas.response import CoreProductResponse, ShopifyResponse
from app.services.openai_service import OpenAIService, OpenAIServiceError
from app.services.ollama_service import OllamaService, OllamaServiceError
from app.utils.product_text import title_keywords, unique_strings


class ShopifyAgent:
    _SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "body_html", "tags", "product_type", "seo_title", "seo_description"],
        "properties": {
            "title": {"type": "string"},
            "body_html": {"type": "string"},
            "tags": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 12},
            "product_type": {"type": "string"},
            "seo_title": {"type": "string"},
            "seo_description": {"type": "string"},
        },
    }

    def __init__(
        self,
        ollama_service: OllamaService | None = None,
        openai_service: OpenAIService | None = None,
    ) -> None:
        self.ollama_service = ollama_service
        self.openai_service = openai_service

    async def process(self, core_data: CoreProductResponse) -> ShopifyResponse:
        fallback = ShopifyResponse(
            title=self._build_title(core_data),
            body_html=self._build_body_html(core_data),
            tags=self._build_tags(core_data),
            product_type=core_data.product_type.title(),
            seo_title=self._build_seo_title(core_data),
            seo_description=self._build_seo_description(core_data),
        )

        if self.openai_service is not None:
            try:
                data = await self.openai_service.generate_structured_output(
                    system_prompt=(
                        "You are a senior Shopify merchandising copywriter. "
                        "Produce storefront-ready product content that feels polished, concise, and conversion-oriented. "
                        "Do not mention AI, confidence, normalization, or internal processing."
                    ),
                    user_payload={"core_product": core_data.model_dump()},
                    schema_name="shopify_listing",
                    schema=self._SCHEMA,
                )
                return self._from_data(data, fallback)
            except OpenAIServiceError:
                pass

        if self.ollama_service is None:
            return fallback

        prompt = (
            "You are a senior Shopify product copy agent. Return only valid JSON with keys "
            "title, body_html, tags, product_type, seo_title, seo_description.\n"
            "Write polished storefront copy and avoid internal metadata.\n"
            f"Core product: {core_data.model_dump()}\n"
        )
        try:
            result = await self.ollama_service.generate_json(prompt=prompt)
        except OllamaServiceError:
            return fallback

        return self._from_data(result.parsed, fallback)

    def _from_data(self, data: dict[str, object], fallback: ShopifyResponse) -> ShopifyResponse:
        return ShopifyResponse(
            title=str(data.get("title", fallback.title))[:160],
            body_html=str(data.get("body_html", fallback.body_html)),
            tags=self._coerce_list(data.get("tags"), fallback.tags),
            product_type=str(data.get("product_type", fallback.product_type)),
            seo_title=str(data.get("seo_title", fallback.seo_title))[:70],
            seo_description=str(data.get("seo_description", fallback.seo_description))[:180],
        )

    @staticmethod
    def _build_title(core_data: CoreProductResponse) -> str:
        color = core_data.attributes.get("color")
        material = core_data.attributes.get("material")
        parts = [core_data.normalized_title]
        if color:
            parts.append(color.title())
        if material:
            parts.append(material.title())
        return " | ".join(parts[:3])[:160]

    @staticmethod
    def _build_body_html(core_data: CoreProductResponse) -> str:
        bullets = "".join(f"<li>{feature}</li>" for feature in core_data.features[:5])
        return f"<p>{core_data.product_summary}</p><ul>{bullets}</ul>"

    @staticmethod
    def _build_tags(core_data: CoreProductResponse) -> list[str]:
        raw_tags = [
            core_data.product_type,
            core_data.category,
            *core_data.attributes.values(),
            *title_keywords(core_data.normalized_title),
        ]
        return unique_strings(raw_tags, limit=12)

    @staticmethod
    def _build_seo_title(core_data: CoreProductResponse) -> str:
        return f"{core_data.normalized_title} | {core_data.category}"[:70]

    @staticmethod
    def _build_seo_description(core_data: CoreProductResponse) -> str:
        return f"{core_data.product_summary} Explore features, materials, and product details."[:180]

    @staticmethod
    def _coerce_list(value: object, fallback: list[str]) -> list[str]:
        if isinstance(value, list) and all(isinstance(item, str) for item in value):
            coerced = unique_strings(value, limit=12)
            if coerced:
                return coerced
        return fallback
