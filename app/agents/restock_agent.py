from __future__ import annotations

from app.schemas.forecasting import ForecastResponse, StockForecast


class RestockAgent:
    def generate_recommendation(self, forecast: StockForecast) -> ForecastResponse:
        if forecast.reorder_urgency == "critical":
            recommendation = (
                f"URGENT: {forecast.product_id} is projected to stock out in "
                f"{forecast.days_until_stockout} days. Reorder "
                f"{forecast.recommended_reorder_qty} units immediately."
            )
        elif forecast.reorder_urgency == "soon":
            recommendation = (
                f"Restock soon: {forecast.product_id} is projected to stock out in "
                f"{forecast.days_until_stockout} days. Recommended reorder quantity is "
                f"{forecast.recommended_reorder_qty} units."
            )
        else:
            recommendation = (
                f"Stock is healthy: {forecast.product_id} has about "
                f"{forecast.days_until_stockout} days before stockout. Plan a future reorder of "
                f"{forecast.recommended_reorder_qty} units when inventory approaches the reorder window."
            )

        return ForecastResponse(
            product_id=forecast.product_id,
            forecast=forecast,
            recommendation=recommendation,
        )
