from __future__ import annotations

from datetime import date, timedelta

from app.agents.restock_agent import RestockAgent
from app.schemas.forecasting import ForecastRequest, ForecastResponse, SalesRecord
from app.services.csv_parser import CSVParser
from app.services.forecasting_engine import ForecastingEngine


def _history(product_id: str, days: int, units_per_day: int) -> list[SalesRecord]:
    start = date.today() - timedelta(days=days)
    return [
        SalesRecord(
            date=start + timedelta(days=index),
            units_sold=units_per_day,
            product_id=product_id,
        )
        for index in range(days)
    ]


def test_critical_urgency() -> None:
    request = ForecastRequest(
        product_id="sku-1",
        sales_history=_history("sku-1", 7, 3),
        current_stock=5,
    )

    forecast = ForecastingEngine().forecast(request)

    assert forecast.reorder_urgency == "critical"


def test_healthy_stock() -> None:
    request = ForecastRequest(
        product_id="sku-2",
        sales_history=_history("sku-2", 7, 2),
        current_stock=200,
    )

    forecast = ForecastingEngine().forecast(request)

    assert forecast.reorder_urgency == "healthy"


def test_simple_fallback() -> None:
    request = ForecastRequest(
        product_id="sku-3",
        sales_history=_history("sku-3", 7, 4),
        current_stock=50,
    )

    forecast = ForecastingEngine().forecast(request)

    assert forecast.confidence_score <= 0.4
    assert forecast.weekly_demand_forecast == [28, 28, 28, 28]


def test_csv_parser_standard_columns() -> None:
    payload = b"date,units_sold,product_id\n2026-01-01,3,sku-1\n2026-01-02,4,sku-1\n"

    parsed = CSVParser().parse(payload)

    assert parsed.row_count == 2
    assert len(parsed.parsed_records) == 2
    assert parsed.product_ids_found == ["sku-1"]


def test_csv_parser_flexible_columns() -> None:
    payload = b"ds,qty,SKU\n2026-01-01,6,sku-flex\n"

    parsed = CSVParser().parse(payload)

    assert parsed.row_count == 1
    assert parsed.parsed_records[0].units_sold == 6
    assert parsed.parsed_records[0].product_id == "sku-flex"


def test_recommendation_critical_message() -> None:
    request = ForecastRequest(
        product_id="sku-4",
        sales_history=_history("sku-4", 7, 3),
        current_stock=5,
    )
    forecast = ForecastingEngine().forecast(request)

    response = RestockAgent().generate_recommendation(forecast)

    assert "URGENT" in response.recommendation
    assert str(forecast.days_until_stockout) in response.recommendation
    assert str(forecast.recommended_reorder_qty) in response.recommendation


def test_full_pipeline() -> None:
    request = ForecastRequest(
        product_id="sku-5",
        sales_history=_history("sku-5", 7, 2),
        current_stock=20,
    )

    forecast = ForecastingEngine().forecast(request)
    response = RestockAgent().generate_recommendation(forecast)

    assert isinstance(response, ForecastResponse)
    assert response.product_id == "sku-5"
    assert response.forecast.product_id == "sku-5"
    assert response.forecast.predicted_stockout_date
    assert len(response.forecast.weekly_demand_forecast) == 4
    assert response.recommendation
