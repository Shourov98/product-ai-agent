from __future__ import annotations

import asyncio
from statistics import mean

from app.config import Settings
from app.schemas.response import (
    CoreProductResponse,
    MarketResearchBundleResponse,
    MarketplaceResearchResponse,
    ResearchEvidenceResponse,
)
from app.services.ebay_market_research_service import EbayMarketResearchService, EbayMarketResearchServiceError
from app.utils.product_text import title_keywords, unique_strings


class MarketResearchService:
    _MARKETPLACE_PREFIXES = {
        "amazon": ("best selling", "top rated", "high intent"),
        "ebay": ("buy it now", "item specifics", "value listing"),
        "etsy": ("handmade style", "giftable", "search rich"),
        "tiktok": ("viral find", "creator favorite", "trend driven"),
        "shopify": ("premium storefront", "direct to consumer", "brand led"),
    }

    def __init__(self, settings: Settings) -> None:
        self.settings = settings
        self.ebay_live = EbayMarketResearchService(
            enabled=settings.market_research_realtime_enabled and settings.ebay_research_enabled,
            client_id=settings.ebay_client_id,
            client_secret=settings.ebay_client_secret,
            marketplace_id=settings.ebay_marketplace_id,
            api_base_url=settings.ebay_api_base_url,
            identity_base_url=settings.ebay_identity_base_url,
            search_limit=settings.ebay_search_limit,
        )

    async def build_research_bundle(self, core_data: CoreProductResponse) -> MarketResearchBundleResponse:
        amazon, ebay, etsy, tiktok, shopify = await asyncio.gather(
            self._safe_build_marketplace_research("amazon", core_data),
            self._safe_build_marketplace_research("ebay", core_data),
            self._safe_build_marketplace_research("etsy", core_data),
            self._safe_build_marketplace_research("tiktok", core_data),
            self._safe_build_marketplace_research("shopify", core_data),
        )
        return MarketResearchBundleResponse(
            amazon=amazon,
            ebay=ebay,
            etsy=etsy,
            tiktok=tiktok,
            shopify=shopify,
        )

    async def _safe_build_marketplace_research(
        self,
        marketplace: str,
        core_data: CoreProductResponse,
    ) -> MarketplaceResearchResponse:
        try:
            if marketplace == "amazon":
                return await self._build_amazon_marketplace_research(core_data)
            if marketplace == "ebay":
                return await self._build_ebay_marketplace_research(core_data)
            return await self._build_search_augmented_marketplace_research(marketplace, core_data)
        except Exception:
            return self._build_marketplace_research(marketplace, core_data)

    async def _build_amazon_marketplace_research(self, core_data: CoreProductResponse) -> MarketplaceResearchResponse:
        return self._build_marketplace_research("amazon", core_data)

    async def _build_ebay_marketplace_research(self, core_data: CoreProductResponse) -> MarketplaceResearchResponse:
        fallback = self._build_marketplace_research("ebay", core_data)
        queries = fallback.search_queries
        try:
            live = await self.ebay_live.search(queries)
        except EbayMarketResearchServiceError as exc:
            live = fallback.model_copy(
                update={
                    "source_mode": "heuristic_with_live_error",
                    "keyword_signals": unique_strings(fallback.keyword_signals + [f"live_error:{exc}"], limit=12),
                }
            )
        if live is not None:
            live = live.model_copy(update={"source_mode": "live_api"})
        return live or fallback

    async def _build_search_augmented_marketplace_research(
        self,
        marketplace: str,
        core_data: CoreProductResponse,
    ) -> MarketplaceResearchResponse:
        return self._build_marketplace_research(marketplace, core_data)

    def _build_marketplace_research(
        self,
        marketplace: str,
        core_data: CoreProductResponse,
    ) -> MarketplaceResearchResponse:
        search_queries = self._build_queries(marketplace, core_data)
        keyword_signals = self._build_keyword_signals(marketplace, core_data)
        base_price, spread = self._estimate_price_profile(core_data, marketplace, keyword_signals)
        price_points = self._build_price_points(base_price, spread, marketplace)
        listings = self._build_similar_listings(marketplace, core_data, price_points)
        numeric_prices = [listing.price for listing in listings if listing.price is not None]

        return MarketplaceResearchResponse(
            marketplace=marketplace,
            source_mode="heuristic_market_research",
            search_queries=search_queries,
            keyword_signals=keyword_signals,
            price_min=min(numeric_prices) if numeric_prices else None,
            price_max=max(numeric_prices) if numeric_prices else None,
            price_avg=round(mean(numeric_prices), 2) if numeric_prices else None,
            regular_price_avg=round(mean(numeric_prices), 2) if numeric_prices else None,
            sale_price_avg=None,
            discount_percent_avg=None,
            similar_listings=listings,
        )

    def _build_queries(self, marketplace: str, core_data: CoreProductResponse) -> list[str]:
        brand = core_data.attributes.get("brand")
        model = core_data.attributes.get("model") or core_data.attributes.get("model_number")
        color = core_data.attributes.get("color")
        material = core_data.attributes.get("material")
        style = core_data.attributes.get("style")
        size = core_data.attributes.get("size") or core_data.attributes.get("capacity")
        title_terms = title_keywords(core_data.normalized_title or core_data.source_title)
        summary_terms = title_keywords(core_data.product_summary)
        visual_identity = " ".join(
            part for part in [brand, model, color, material, style, core_data.product_type] if part
        )
        category_identity = " ".join(
            part for part in [brand, core_data.category, core_data.product_type, color, material] if part
        )
        candidates = [
            core_data.normalized_title,
            core_data.source_title,
            visual_identity,
            category_identity,
            " ".join(part for part in [brand, model, core_data.product_type] if part),
            " ".join(part for part in [color, material, core_data.product_type] if part),
            " ".join(part for part in [style, color, core_data.product_type] if part),
            " ".join(part for part in [core_data.category, size, core_data.product_type] if part),
            " ".join(title_terms[:6]),
            " ".join(summary_terms[:6]),
        ]
        return unique_strings([item.strip() for item in candidates if item and item.strip()], limit=4)

    def _build_keyword_signals(self, marketplace: str, core_data: CoreProductResponse) -> list[str]:
        title_terms = title_keywords(core_data.normalized_title)
        attributes = [value.lower() for value in core_data.attributes.values()]
        prefixes = list(self._MARKETPLACE_PREFIXES[marketplace])
        if marketplace == "amazon":
            prefixes.extend(["durable", "giftable", "everyday use"])
        elif marketplace == "ebay":
            prefixes.extend(["fast shipping", "clear condition", "spec driven"])
        elif marketplace == "etsy":
            prefixes.extend(["handmade appeal", "gift intent", "artisan style"])
        elif marketplace == "tiktok":
            prefixes.extend(["scroll stopping", "shareable", "creator style"])
        else:
            prefixes.extend(["premium", "brand story", "storefront ready"])
        return unique_strings(prefixes + title_terms + attributes, limit=12)

    def _build_similar_listings(
        self,
        marketplace: str,
        core_data: CoreProductResponse,
        price_points: list[float],
    ) -> list[ResearchEvidenceResponse]:
        color = core_data.attributes.get("color")
        material = core_data.attributes.get("material")
        size = core_data.attributes.get("size") or core_data.attributes.get("capacity")
        descriptors = [value.title() for value in [color, material, size] if value]
        descriptor_text = " ".join(descriptors[:2]).strip()
        title_root = f"{descriptor_text} {core_data.normalized_title}".strip()

        evidence = []
        for index, price in enumerate(price_points, start=1):
            evidence.append(
                ResearchEvidenceResponse(
                    source=marketplace,
                    title=self._similar_title_for_marketplace(marketplace, title_root, index),
                    price=price,
                    relevance_score=max(0.72, 0.96 - (index * 0.06)),
                    attributes=self._marketplace_attributes(marketplace, core_data),
                    observations=self._observations_for_marketplace(marketplace, core_data, index),
                )
            )
        return evidence

    @staticmethod
    def _similar_title_for_marketplace(marketplace: str, title_root: str, index: int) -> str:
        if marketplace == "amazon":
            return f"{title_root} for Everyday Performance Variation {index}".strip()
        if marketplace == "ebay":
            return f"{title_root} | Clean Listing Option {index}".strip()
        if marketplace == "etsy":
            return f"{title_root} Gift Ready Listing {index}".strip()
        if marketplace == "tiktok":
            return f"{title_root} Trend Pick {index}".strip()
        return f"{title_root} Storefront Feature {index}".strip()

    @staticmethod
    def _marketplace_attributes(marketplace: str, core_data: CoreProductResponse) -> dict[str, str]:
        attributes = dict(core_data.attributes)
        if marketplace == "amazon":
            attributes.setdefault("category", core_data.category)
        if marketplace == "ebay":
            attributes.setdefault("Condition", "New")
        if marketplace == "etsy":
            attributes.setdefault("occasion", "gift")
        if marketplace == "tiktok":
            attributes.setdefault("hook", "visual commerce")
        if marketplace == "shopify":
            attributes.setdefault("merchandising", "storefront hero")
        return {str(key): str(value) for key, value in attributes.items()}

    @staticmethod
    def _observations_for_marketplace(
        marketplace: str,
        core_data: CoreProductResponse,
        index: int,
    ) -> list[str]:
        observations = [
            f"Similar {marketplace} listings emphasize {core_data.product_type}.",
            f"Top listing set {index} highlights visible product traits before lifestyle messaging.",
        ]
        if "color" in core_data.attributes:
            observations.append("Color is consistently surfaced in listing titles and filters.")
        if marketplace == "amazon":
            observations.append("Feature-led bullets and searchable modifiers are common.")
        elif marketplace == "ebay":
            observations.append("Structured item specifics drive discoverability.")
        elif marketplace == "etsy":
            observations.append("Keyword-rich descriptive phrasing and gift intent are common.")
        elif marketplace == "tiktok":
            observations.append("Short hooks and visual-first phrasing outperform longer copy.")
        else:
            observations.append("Brand tone and polished merchandising language are more prominent.")
        return observations

    @staticmethod
    def _estimate_price_profile(
        core_data: CoreProductResponse,
        marketplace: str,
        keyword_signals: list[str],
    ) -> tuple[float, float]:
        identity = " ".join(
            [
                core_data.normalized_title,
                core_data.source_title,
                core_data.category,
                core_data.product_type,
                " ".join(core_data.features),
                " ".join(f"{key}:{value}" for key, value in core_data.attributes.items()),
            ]
        ).lower()

        if any(keyword in identity for keyword in ("ps5", "playstation 5", "playstation5", "sony playstation 5", "sony ps5", "ps 5")):
            return 499.99, 0.16
        if any(keyword in identity for keyword in ("xbox series x", "xbox series s", "xbox", "gaming console", "video game console", "game console", "console")):
            if "series s" in identity:
                return 299.99, 0.17
            return (399.99 if "console" in identity and "gaming" not in identity else 499.99), 0.16
        if any(keyword in identity for keyword in ("nintendo switch", "switch oled", "switch lite")):
            if "lite" in identity:
                return 199.99, 0.18
            return 299.99, 0.18

        baseline = 14.99
        category = core_data.category.lower()
        product_type = core_data.product_type.lower()
        attributes = core_data.attributes

        if "drink" in category or "bottle" in product_type:
            baseline = 24.99
        elif "footwear" in category or "shoe" in product_type:
            baseline = 69.99
        elif "electronics" in category or "case" in product_type:
            baseline = 39.99
        elif "fashion" in category:
            baseline = 34.99

        if "material" in attributes:
            baseline += 4.0
        if "size" in attributes or "capacity" in attributes:
            baseline += 2.5
        if "brand" in attributes:
            baseline += 3.0

        signal_boost = min(0.22, 0.01 * len(keyword_signals))
        spread = max(0.12, min(0.34, 0.14 + (len(core_data.features) * 0.015) + (len(attributes) * 0.01) + signal_boost))
        marketplace_bias = {
            "amazon": 1.0,
            "ebay": 0.98,
            "etsy": 1.04,
            "tiktok": 0.96,
            "shopify": 1.06,
        }[marketplace]
        return round(baseline * marketplace_bias, 2), spread

    @staticmethod
    def _build_price_points(base_price: float, spread: float, marketplace: str) -> list[float]:
        market_bias = {
            "amazon": 1.0,
            "ebay": 0.99,
            "etsy": 1.02,
            "tiktok": 0.97,
            "shopify": 1.05,
        }[marketplace]
        center = max(0.01, base_price * market_bias)
        lower = max(0.01, center * (1 - spread * 1.25))
        lower_mid = max(0.01, center * (1 - spread * 0.45))
        upper_mid = max(lower_mid, center * (1 + spread * 0.35))
        upper = max(upper_mid, center * (1 + spread * 1.15))
        return [round(lower, 2), round(lower_mid, 2), round(center, 2), round(upper_mid, 2), round(upper, 2)]
