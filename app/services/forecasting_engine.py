from __future__ import annotations

import logging
import math
from datetime import date, timedelta
from statistics import mean, pstdev
from typing import Any

from app.schemas.forecasting import ForecastRequest, StockForecast


logging.getLogger("prophet").setLevel(logging.WARNING)
logging.getLogger("cmdstanpy").setLevel(logging.WARNING)


class ForecastingEngine:
    def forecast(self, request: ForecastRequest) -> StockForecast:
        sorted_history = sorted(request.sales_history, key=lambda item: item.date)
        if len(sorted_history) >= 14:
            daily_forecast, confidence = self._prophet_or_rolling_forecast(sorted_history)
        else:
            daily_forecast, confidence = self._rolling_average_forecast(sorted_history, confidence_cap=0.4)

        daily_demand = max(mean(daily_forecast), 0.0)
        days_until_stockout = self._days_until_stockout(request.current_stock, daily_demand)
        predicted_stockout_date = date.today() + timedelta(days=days_until_stockout)
        reorder_urgency = self._urgency(
            days_until_stockout=days_until_stockout,
            lead_time_days=request.lead_time_days,
            safety_stock_days=request.safety_stock_days,
        )
        weekly_demand_forecast = [
            max(0, int(round(sum(daily_forecast[index : index + 7]))))
            for index in range(0, 28, 7)
        ]
        recommended_reorder_qty = self._recommended_reorder_qty(
            weekly_demand_forecast=weekly_demand_forecast,
            daily_demand=daily_demand,
            lead_time_days=request.lead_time_days,
            safety_stock_days=request.safety_stock_days,
            current_stock=request.current_stock,
        )

        return StockForecast(
            product_id=request.product_id,
            days_until_stockout=days_until_stockout,
            predicted_stockout_date=predicted_stockout_date,
            recommended_reorder_qty=recommended_reorder_qty,
            reorder_urgency=reorder_urgency,
            weekly_demand_forecast=weekly_demand_forecast,
            confidence_score=round(max(0.0, min(confidence, 1.0)), 2),
        )

    def _prophet_or_rolling_forecast(self, sales_history: list[Any]) -> tuple[list[float], float]:
        try:
            from prophet import Prophet
            import pandas as pd
        except ImportError:
            return self._rolling_average_forecast(sales_history, confidence_cap=0.65)

        try:
            frame = pd.DataFrame(
                {
                    "ds": [record.date.isoformat() for record in sales_history],
                    "y": [record.units_sold for record in sales_history],
                }
            )
            model = Prophet(
                daily_seasonality=False,
                weekly_seasonality=True,
                yearly_seasonality=False,
            )
            model.fit(frame)
            future = model.make_future_dataframe(periods=28)
            forecast = model.predict(future).tail(28)
            daily = [max(0.0, float(value)) for value in forecast["yhat"].tolist()]
            confidence = self._confidence_from_history([record.units_sold for record in sales_history], base=0.72)
            return daily, confidence
        except (ValueError, RuntimeError, TypeError, KeyError, AttributeError):
            return self._rolling_average_forecast(sales_history, confidence_cap=0.65)

    def _rolling_average_forecast(
        self,
        sales_history: list[Any],
        *,
        confidence_cap: float,
    ) -> tuple[list[float], float]:
        values = [float(record.units_sold) for record in sales_history]
        window = values[-7:] if len(values) >= 7 else values
        daily_average = mean(window) if window else 0.0
        daily = [max(0.0, daily_average) for _ in range(28)]
        confidence = min(confidence_cap, self._confidence_from_history(values, base=confidence_cap))
        return daily, confidence

    @staticmethod
    def _days_until_stockout(current_stock: int, daily_demand: float) -> int:
        if daily_demand <= 0:
            return 365
        return max(0, int(math.floor(current_stock / daily_demand)))

    @staticmethod
    def _urgency(*, days_until_stockout: int, lead_time_days: int, safety_stock_days: int) -> str:
        if days_until_stockout <= lead_time_days:
            return "critical"
        if days_until_stockout <= lead_time_days + safety_stock_days:
            return "soon"
        return "healthy"

    @staticmethod
    def _recommended_reorder_qty(
        *,
        weekly_demand_forecast: list[int],
        daily_demand: float,
        lead_time_days: int,
        safety_stock_days: int,
        current_stock: int,
    ) -> int:
        next_30_day_demand = sum(weekly_demand_forecast)
        buffer_demand = daily_demand * (lead_time_days + safety_stock_days)
        needed = next_30_day_demand + buffer_demand - current_stock
        return max(0, int(math.ceil(needed)))

    @staticmethod
    def _confidence_from_history(values: list[float], *, base: float) -> float:
        if len(values) < 2:
            return min(base, 0.25)
        avg = mean(values)
        if avg <= 0:
            return min(base, 0.3)
        variability = pstdev(values) / avg
        penalty = min(0.45, variability * 0.25)
        history_bonus = min(0.18, len(values) / 120)
        return max(0.1, base + history_bonus - penalty)
