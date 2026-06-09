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
        regular_average = research.regular_price_avg or average
        sale_average = research.sale_price_avg
        discount_average = research.discount_percent_avg
        minimum = research.price_min or average * 0.9
        maximum = research.price_max or average * 1.1
        strategy, multiplier = self._STRATEGIES.get(research.marketplace, ("balanced", 1.0))
        recommended = round(average * multiplier, 2)
        discounted_recommended = None
        if sale_average is not None:
            discounted_recommended = round(sale_average * multiplier, 2)
        elif discount_average is not None and discount_average > 0:
            discounted_recommended = round(recommended * (1 - (discount_average / 100)), 2)
        floor = round(min(minimum, recommended * 0.92), 2)
        ceiling = round(max(maximum, recommended * 1.08), 2)
        confidence = 0.72 if len(research.similar_listings) >= 3 else 0.58
        reasons = [
            f"Recommendation anchored to average similar-listing price of {average:.2f} USD.",
            f"{research.marketplace.title()} strategy set to {strategy} based on marketplace selling behavior.",
        ]
        if sale_average is not None:
            reasons.append(f"Observed sale-price average is {sale_average:.2f} USD.")
        if discount_average is not None:
            reasons.append(f"Observed average discount pressure is {discount_average:.2f}%.")
        if research.keyword_signals:
            reasons.append(f"Keyword cluster includes {', '.join(research.keyword_signals[:3])}.")

        summary_parts = [
            f"{research.marketplace.title()} {strategy} pricing recommends ${recommended:.2f}",
            f"with promo at ${discounted_recommended:.2f}" if discounted_recommended is not None else None,
            f"against market average ${average:.2f}",
            f"and average discount {discount_average:.1f}%" if discount_average is not None else None,
        ]
        summary = " ".join(part for part in summary_parts if part) + "."

        return MarketplacePricingResponse(
            marketplace=research.marketplace,
            recommended=recommended,
            discounted_recommended=discounted_recommended,
            floor=max(0.0, floor),
            ceiling=max(recommended, ceiling),
            market_average=average,
            regular_price_average=regular_average,
            sale_price_average=sale_average,
            discount_percent_average=discount_average,
            strategy=strategy,
            confidence=confidence,
            summary=summary,
            reasons=reasons,
        )
