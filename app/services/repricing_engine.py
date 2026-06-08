from __future__ import annotations

import json
from dataclasses import asdict
from datetime import UTC, datetime
from typing import Any

from app.providers.base import CompetitorSnapshot, PriceDataProvider, ProductData
from app.schemas.repricing import AIDecision, BulkRepricingResponse, RepricingResult


class RepricingEngine:
    def __init__(self, provider: PriceDataProvider, openai_client: Any | None) -> None:
        self.provider = provider
        self.openai_client = openai_client

    async def run(self, asin: str, strategy: str, dry_run: bool) -> RepricingResult:
        product = await self.provider.get_product(asin)
        snapshot = await self.provider.get_competitor_snapshot(asin)
        decision = await self._ai_decide(product, snapshot, strategy)
        safe_price = self._apply_guardrails(decision.recommended_price, product, decision)
        guardrails_applied = safe_price != decision.recommended_price
        action = self._action_for_prices(product.our_price, safe_price)
        decision = decision.model_copy(
            update={
                "recommended_price": safe_price,
                "action": action,
                "margin_at_new_price": self._margin_at_price(safe_price, product.cost_price),
                "guardrails_applied": guardrails_applied,
            }
        )

        update_result = None
        if not dry_run and safe_price != product.our_price:
            update_result = await self.provider.update_price(asin, safe_price)

        price_delta = round(safe_price - product.our_price, 2)
        return RepricingResult(
            asin=asin,
            product_name=product.title,
            category=product.category_name,
            marketplace=product.marketplace,
            timestamp=datetime.now(UTC).isoformat(),
            is_best_seller=product.is_best_seller,
            demand_signal=snapshot.demand_signal,
            stars=product.stars,
            reviews=product.reviews,
            pricing={
                "old_price": product.our_price,
                "new_price": safe_price,
                "price_delta": price_delta,
                "price_changed": safe_price != product.our_price,
                "list_price": product.list_price,
                "cost_price": product.cost_price,
            },
            competitor_snapshot=asdict(snapshot),
            ai_decision=decision,
            price_update_result=asdict(update_result) if update_result is not None else None,
            data_source=snapshot.source,
        )

    async def _ai_decide(self, product: ProductData, snapshot: CompetitorSnapshot, strategy: str) -> AIDecision:
        if self.openai_client is None:
            return self._fallback_decision(product, snapshot, strategy)

        prompt = self._build_prompt(product, snapshot, strategy)
        try:
            response = await self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "Return valid JSON only."},
                    {"role": "user", "content": prompt},
                ],
                temperature=0.2,
                response_format={"type": "json_object"},
                max_tokens=200,
            )
            content = response.choices[0].message.content or "{}"
            payload = json.loads(content)
            return AIDecision.model_validate(payload)
        except (AttributeError, IndexError, KeyError, TypeError, ValueError, json.JSONDecodeError, Exception):
            return self._fallback_decision(product, snapshot, strategy)

    def _apply_guardrails(self, price: float, product: ProductData, decision: AIDecision) -> float:
        del decision
        safe_price = price
        safe_price = max(safe_price, product.cost_price * 1.15)
        safe_price = max(safe_price, product.min_price)
        safe_price = min(safe_price, product.max_price)
        safe_price = max(safe_price, product.our_price * 0.80)
        safe_price = min(safe_price, product.our_price * 1.20)
        return round(safe_price, 2)

    async def run_bulk(self, asins: list[str], strategy: str, dry_run: bool) -> BulkRepricingResponse:
        results: list[RepricingResult] = []
        errors: list[dict[str, str]] = []
        data_source = "unknown"
        for asin in asins:
            try:
                result = await self.run(asin, strategy, dry_run)
                data_source = result.data_source
                results.append(result)
            except Exception as exc:
                errors.append({"asin": asin, "error": str(exc)})

        return BulkRepricingResponse(
            total_analyzed=len(results),
            price_lowered=sum(1 for item in results if item.ai_decision.action == "lower"),
            price_raised=sum(1 for item in results if item.ai_decision.action == "raise"),
            price_held=sum(1 for item in results if item.ai_decision.action == "hold"),
            total_potential_savings=round(
                sum(abs(float(item.pricing["price_delta"])) for item in results if item.ai_decision.action == "lower"),
                2,
            ),
            results=results,
            errors=errors,
            data_source=data_source,
        )

    def _fallback_decision(self, product: ProductData, snapshot: CompetitorSnapshot, strategy: str) -> AIDecision:
        selected_strategy = strategy if strategy != "auto" else self._select_strategy(product, snapshot)
        recommended = product.our_price
        reason = "Current price is already aligned with marketplace signals."

        if selected_strategy == "buy_box_win" and product.our_price > snapshot.buy_box_price:
            recommended = snapshot.buy_box_price - 0.25
            reason = "Price slightly below the buy box to improve competitiveness."
        elif selected_strategy == "profit_maximize":
            recommended = min(snapshot.average_price * 1.08, product.max_price)
            reason = "Demand and market average support a higher profit-focused price."
        elif selected_strategy == "scarcity" and product.current_stock < 15 and snapshot.demand_signal == "high":
            recommended = product.our_price * 1.12
            reason = "Low stock and high demand support a scarcity price increase."
        elif selected_strategy == "velocity" and snapshot.demand_signal == "low":
            recommended = min(snapshot.lowest_price, product.our_price * 0.92)
            reason = "Low demand supports a price reduction to increase sales velocity."
        elif product.our_price > snapshot.average_price * 1.05 and snapshot.demand_signal != "high":
            recommended = snapshot.average_price * 0.98
            selected_strategy = "buy_box_win"
            reason = "Current price is above category average without high-demand support."
        elif snapshot.demand_signal == "high" and product.current_stock < 20:
            recommended = product.our_price * 1.08
            selected_strategy = "scarcity"
            reason = "High demand and limited stock justify a controlled price increase."

        recommended = round(max(0.01, recommended), 2)
        action = self._action_for_prices(product.our_price, recommended)
        return AIDecision(
            recommended_price=recommended,
            action=action,
            strategy_used=selected_strategy,
            reason=reason,
            confidence=0.72,
            margin_at_new_price=self._margin_at_price(recommended, product.cost_price),
        )

    @staticmethod
    def _select_strategy(product: ProductData, snapshot: CompetitorSnapshot) -> str:
        if product.current_stock < 15 and snapshot.demand_signal == "high":
            return "scarcity"
        if snapshot.demand_signal == "low" and product.units_sold_7d < 5:
            return "velocity"
        if product.our_price > snapshot.average_price * 1.05:
            return "buy_box_win"
        if snapshot.demand_signal == "high" and product.is_best_seller:
            return "profit_maximize"
        return "auto"

    @staticmethod
    def _action_for_prices(old_price: float, new_price: float) -> str:
        if new_price < old_price:
            return "lower"
        if new_price > old_price:
            return "raise"
        return "hold"

    @staticmethod
    def _margin_at_price(price: float, cost_price: float) -> float:
        if price <= 0:
            return 0.0
        return round(((price - cost_price) / price) * 100, 2)

    @staticmethod
    def _build_prompt(product: ProductData, snapshot: CompetitorSnapshot, strategy: str) -> str:
        return f"""
You are an expert Amazon pricing strategist.

Product: {product.title}
Category: {product.category_name}
Marketplace: {product.marketplace}
Best Seller: {product.is_best_seller}
Rating: {product.stars}/5 ({product.reviews} reviews)

Our pricing:
- Current price: ${product.our_price}
- Cost price: ${product.cost_price} (min 15% margin required)
- Price floor: ${product.min_price}
- Price ceiling: ${product.max_price}

Competitor intelligence:
- Buy Box price: ${snapshot.buy_box_price}
- Lowest competitor: ${snapshot.lowest_price}
- Category average: ${snapshot.average_price}
- Category median: ${snapshot.median_price}
- Active competitors: {snapshot.competitor_count}
- Our position: {snapshot.our_position}

Demand intelligence:
- Units sold last 30 days: {product.units_sold_30d}
- Units sold last 7 days: {product.units_sold_7d}
- Current stock level: {product.current_stock}
- Demand signal: {snapshot.demand_signal}

Requested strategy: {strategy}

Strategy definitions:
- auto: choose best strategy based on all signals
- buy_box_win: price $0.01-$0.50 below buy box
- profit_maximize: highest price market will bear
- scarcity: stock < 15 AND demand = high -> raise price
- velocity: demand = low AND sales slow -> lower to drive volume

Rules you must never break:
1. Price >= cost_price * 1.15
2. Price must be within [min_price, max_price]
3. Single change limit: max 20% up or down from current
4. Already at/below buy box + demand not high -> hold

Output valid JSON only:
{{
  "recommended_price": <float 2dp>,
  "action": "<lower|raise|hold>",
  "strategy_used": "<name>",
  "reason": "<one sentence>",
  "confidence": <0.0-1.0>,
  "margin_at_new_price": <percentage float>
}}
"""
