from __future__ import annotations

import re

from app.schemas.response import CoreProductResponse, MarketplacePricingResponse, MarketplaceResearchResponse, SeoInsightsResponse, TikTokResponse
from app.services.openai_service import OpenAIService, OpenAIServiceError
from app.services.ollama_service import OllamaService, OllamaServiceError
from app.utils.product_text import unique_strings


class TikTokAgent:
    _SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": ["title", "social_description", "hashtags"],
        "properties": {
            "title": {"type": "string"},
            "social_description": {"type": "string"},
            "hashtags": {"type": "array", "items": {"type": "string"}, "minItems": 4, "maxItems": 8},
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
        pricing: MarketplacePricingResponse | None = None,
    ) -> TikTokResponse:
        fallback = TikTokResponse(
            title=self._build_title(core_data, seo),
            social_description=self._build_social_description(core_data, pricing),
            hashtags=self._build_hashtags(core_data, research, seo),
        )
        if self.openai_service is not None:
            try:
                data = await self.openai_service.generate_structured_output(
                    system_prompt=(
                        "You are a senior TikTok Shop copywriter. "
                        "Create TikTok-native product copy using a strong hook, clear product payoff, realistic use-case, and a light commerce CTA. "
                        "Keep the product facts accurate. Do not invent claims, certifications, performance outcomes, or social proof. "
                        "The social_description must read like a polished viral-style caption for TikTok Shop, not a generic product summary. "
                        "Prefer concise mobile-friendly rhythm, natural language, and buyer-intent phrasing. "
                        "Hashtags must be SEO-aware and commerce-relevant: mix broad category tags, niche intent tags, and product-specific tags. "
                        "Avoid spammy tag stuffing, repeated tags, and vague filler. "
                        "Do not mention AI, confidence, or internal processing. "
                        "Do not include any text inside image."
                    ),
                    user_payload={
                        "core_product": core_data.model_dump(),
                        "research": research.model_dump() if research is not None else None,
                        "seo": seo.model_dump() if seo is not None else None,
                        "pricing": pricing.model_dump() if pricing is not None else None,
                    },
                    schema_name="tiktok_listing",
                    schema=self._SCHEMA,
                )
                return self._from_data(data, fallback)
            except OpenAIServiceError:
                pass

        if self.ollama_service is None:
            return fallback

        prompt = (
            "You are a senior TikTok listing agent. Return only valid JSON with keys title, "
            "social_description, hashtags.\n"
            "Write for social commerce, not internal tooling. Do not mention AI or confidence.\n"
            "Use a stop-scroll hook, product payoff, use-case, and soft CTA in social_description.\n"
            "Generate SEO-aware hashtags for TikTok Shop using broad, niche, and product-intent tags.\n"
            f"Core product: {core_data.model_dump()}\n"
        )
        try:
            result = await self.ollama_service.generate_json(prompt=prompt)
        except OllamaServiceError:
            return fallback

        return self._from_data(result.parsed, fallback)

    def _from_data(self, data: dict[str, object], fallback: TikTokResponse) -> TikTokResponse:
        title = self._clean_inline_text(str(data.get("title", fallback.title)))
        social_description = self._clean_inline_text(str(data.get("social_description", fallback.social_description)))
        hashtags = data.get("hashtags")
        normalized_hashtags = (
            self._normalize_hashtags(hashtags)
            if isinstance(hashtags, list) and all(isinstance(item, str) for item in hashtags)
            else fallback.hashtags
        )
        return TikTokResponse(
            title=title or fallback.title,
            social_description=social_description or fallback.social_description,
            hashtags=normalized_hashtags,
        )

    def _build_title(self, core_data: CoreProductResponse, seo: SeoInsightsResponse | None) -> str:
        color = core_data.attributes.get("color")
        hook = seo.marketplace_keywords.get("tiktok", [None])[0] if seo is not None else None
        if color:
            title = " ".join(part for part in [color.title(), core_data.normalized_title, hook] if part).strip()
            return self._truncate_text(title, 90)
        title = " ".join(part for part in [core_data.normalized_title, hook] if part).strip()
        return self._truncate_text(title, 90)

    def _build_social_description(self, core_data: CoreProductResponse, pricing: MarketplacePricingResponse | None) -> str:
        hook = self._build_hook(core_data)
        payoff = self._build_payoff(core_data)
        use_case = self._build_use_case(core_data)
        cta = self._build_soft_cta(core_data)
        price_hint = (
            f" Price point lands around ${pricing.recommended:.2f}."
            if pricing is not None and pricing.recommended > 0
            else ""
        )
        caption = " ".join(part for part in [hook, payoff, use_case, cta] if part).strip()
        return self._truncate_text(f"{caption}{price_hint}".strip(), 220)

    def _build_hashtags(
        self,
        core_data: CoreProductResponse,
        research: MarketplaceResearchResponse | None,
        seo: SeoInsightsResponse | None,
    ) -> list[str]:
        broad_tags = [
            core_data.product_type,
            core_data.category,
            "tiktok shop finds",
        ]
        niche_tags = list(core_data.attributes.values())
        product_intent_tags = [
            core_data.normalized_title,
            *core_data.features[:2],
        ]
        if research is not None:
            niche_tags.extend(research.keyword_signals[:3])
        if seo is not None:
            product_intent_tags.extend(seo.marketplace_keywords.get("tiktok", []))

        hashtags = [
            *self._hashtags_from_values(broad_tags),
            *self._hashtags_from_values(niche_tags),
            *self._hashtags_from_values(product_intent_tags),
            "#TikTokShop",
            "#TikTokMadeMeBuyIt",
        ]
        return self._normalize_hashtags(hashtags)

    @staticmethod
    def _truncate_text(value: str, limit: int) -> str:
        cleaned = re.sub(r"\s+", " ", value).strip()
        if len(cleaned) <= limit:
            return cleaned
        truncated = cleaned[:limit].rstrip()
        last_space = truncated.rfind(" ")
        if last_space > max(limit // 2, 24):
            truncated = truncated[:last_space]
        return truncated.rstrip(" ,.;:") + "..."

    @staticmethod
    def _clean_inline_text(value: str) -> str:
        return re.sub(r"\s+", " ", value).strip()

    @staticmethod
    def _build_hook(core_data: CoreProductResponse) -> str:
        product = core_data.product_type or core_data.normalized_title
        color = core_data.attributes.get("color", "").strip()
        style = core_data.attributes.get("style", "").strip()
        hook_parts = ["POV:"]
        if color and style:
            hook_parts.append(f"you found the {color.lower()} {style.lower()} {product.lower()} your setup was missing.")
        elif color:
            hook_parts.append(f"you found the {color.lower()} {product.lower()} your setup was missing.")
        else:
            hook_parts.append(f"you found the {product.lower()} that actually fits your everyday setup.")
        return " ".join(hook_parts)

    @staticmethod
    def _build_payoff(core_data: CoreProductResponse) -> str:
        feature_line = ", ".join(feature.strip().rstrip(".") for feature in core_data.features[:2] if feature.strip())
        if feature_line:
            return f"Clean look, clear use-case, and {feature_line.lower()}."
        return core_data.product_summary.strip().rstrip(".") + "."

    @staticmethod
    def _build_use_case(core_data: CoreProductResponse) -> str:
        category = core_data.category.strip().lower()
        product_type = core_data.product_type.strip().lower()
        if category:
            return f"Built for {category} routines and easy everyday use."
        if product_type:
            return f"Made for simple everyday {product_type} use."
        return "Made for simple everyday use."

    @staticmethod
    def _build_soft_cta(core_data: CoreProductResponse) -> str:
        product_type = core_data.product_type.strip().lower() or "pick"
        return f"See why this {product_type} keeps getting added to carts."

    @staticmethod
    def _hashtags_from_values(values: list[str]) -> list[str]:
        hashtags: list[str] = []
        for value in values:
            compact = re.sub(r"[^a-zA-Z0-9]+", "", value.title())
            if compact and len(compact) > 2:
                hashtags.append(f"#{compact}")
        return hashtags

    @staticmethod
    def _normalize_hashtags(values: list[str]) -> list[str]:
        normalized: list[str] = []
        seen: set[str] = set()
        for value in values:
            if not isinstance(value, str):
                continue
            compact = re.sub(r"[^a-zA-Z0-9#]+", "", value.strip())
            if not compact:
                continue
            if not compact.startswith("#"):
                compact = f"#{compact.lstrip('#')}"
            key = compact.lower()
            if key in seen:
                continue
            seen.add(key)
            normalized.append(compact)
        return unique_strings(normalized, limit=8)
