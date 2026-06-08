from __future__ import annotations

from app.schemas.response import (
    MarketResearchBundleResponse,
    MarketplacePricingResponse,
    PricingInsightsResponse,
)


class PricingService:
    _STRATEGIES = {
        "amazon": ("balanced", 1.0),
        "ebay": ("competitive", 0.97),
        "etsy": ("value-crafted", 1.05),
        "tiktok": ("aggressive", 0.94),
        "shopify": ("premium", 1.08),
    }

    def build_pricing(self, research: MarketResearchBundleResponse) -> PricingInsightsResponse:
        return PricingInsightsResponse(
            amazon=self._build_marketplace_pricing(research.amazon),
            ebay=self._build_marketplace_pricing(research.ebay),
            etsy=self._build_marketplace_pricing(research.etsy),
            tiktok=self._build_marketplace_pricing(research.tiktok),
            shopify=self._build_marketplace_pricing(research.shopify),
        )

    def _build_marketplace_pricing(self, research) -> MarketplacePricingResponse:
        average = research.price_avg or 19.99
        minimum = research.price_min or average * 0.9
        maximum = research.price_max or average * 1.1
        strategy, multiplier = self._STRATEGIES.get(research.marketplace, ("balanced", 1.0))
        recommended = round(average * multiplier, 2)
        floor = round(min(minimum, recommended * 0.92), 2)
        ceiling = round(max(maximum, recommended * 1.08), 2)
        confidence = 0.72 if len(research.similar_listings) >= 3 else 0.58
        reasons = [
            f"Recommendation anchored to average similar-listing price of {average:.2f} USD.",
            f"{research.marketplace.title()} strategy set to {strategy} based on marketplace selling behavior.",
        ]
        if research.keyword_signals:
            reasons.append(f"Keyword cluster includes {', '.join(research.keyword_signals[:3])}.")

        return MarketplacePricingResponse(
            marketplace=research.marketplace,
            recommended=recommended,
            floor=max(0.0, floor),
            ceiling=max(recommended, ceiling),
            strategy=strategy,
            confidence=confidence,
            reasons=reasons,
        )
