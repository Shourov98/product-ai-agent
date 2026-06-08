from __future__ import annotations

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

    _MARKETPLACE_PRICE_MULTIPLIERS = {
        "amazon": (0.94, 1.0, 1.08),
        "ebay": (0.9, 0.97, 1.04),
        "etsy": (0.96, 1.05, 1.16),
        "tiktok": (0.88, 0.95, 1.02),
        "shopify": (0.98, 1.08, 1.18),
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
        amazon = self._build_marketplace_research("amazon", core_data)
        ebay = await self._build_ebay_marketplace_research(core_data)
        etsy = self._build_marketplace_research("etsy", core_data)
        tiktok = self._build_marketplace_research("tiktok", core_data)
        shopify = self._build_marketplace_research("shopify", core_data)
        return MarketResearchBundleResponse(
            amazon=amazon,
            ebay=ebay,
            etsy=etsy,
            tiktok=tiktok,
            shopify=shopify,
        )

    async def _build_ebay_marketplace_research(self, core_data: CoreProductResponse) -> MarketplaceResearchResponse:
        fallback = self._build_marketplace_research("ebay", core_data)
        queries = fallback.search_queries
        try:
            live = await self.ebay_live.search(queries)
        except EbayMarketResearchServiceError as exc:
            return fallback.model_copy(
                update={
                    "source_mode": "heuristic_with_live_error",
                    "keyword_signals": unique_strings(fallback.keyword_signals + [f"live_error:{exc}"], limit=12),
                }
            )
        if live is not None:
            return live.model_copy(update={"source_mode": "live_api"})
        return fallback

    def _build_marketplace_research(
        self,
        marketplace: str,
        core_data: CoreProductResponse,
    ) -> MarketplaceResearchResponse:
        search_queries = self._build_queries(marketplace, core_data)
        keyword_signals = self._build_keyword_signals(marketplace, core_data)
        base_price = self._estimate_base_price(core_data)
        price_points = [round(base_price * multiplier, 2) for multiplier in self._MARKETPLACE_PRICE_MULTIPLIERS[marketplace]]
        listings = self._build_similar_listings(marketplace, core_data, price_points)
        numeric_prices = [listing.price for listing in listings if listing.price is not None]

        return MarketplaceResearchResponse(
            marketplace=marketplace,
            source_mode="heuristic",
            search_queries=search_queries,
            keyword_signals=keyword_signals,
            price_min=min(numeric_prices) if numeric_prices else None,
            price_max=max(numeric_prices) if numeric_prices else None,
            price_avg=round(mean(numeric_prices), 2) if numeric_prices else None,
            similar_listings=listings,
        )

    def _build_queries(self, marketplace: str, core_data: CoreProductResponse) -> list[str]:
        color = core_data.attributes.get("color")
        material = core_data.attributes.get("material")
        size = core_data.attributes.get("size") or core_data.attributes.get("capacity")
        candidates = [
            core_data.normalized_title,
            " ".join(part for part in [color, material, core_data.product_type] if part),
            " ".join(part for part in [core_data.category, size, core_data.product_type] if part),
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
    def _estimate_base_price(core_data: CoreProductResponse) -> float:
        baseline = 14.99
        category = core_data.category.lower()
        product_type = core_data.product_type.lower()
        attributes = core_data.attributes

        if "drink" in category or "bottle" in product_type:
            baseline = 24.99
        elif "footwear" in category or "shoe" in product_type:
            baseline = 69.99
        elif "electronics" in category or "case" in product_type:
            baseline = 19.99
        elif "fashion" in category:
            baseline = 34.99

        if "material" in attributes:
            baseline += 4.0
        if "size" in attributes or "capacity" in attributes:
            baseline += 2.5
        if "brand" in attributes:
            baseline += 3.0

        return baseline
