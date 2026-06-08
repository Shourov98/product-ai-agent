from __future__ import annotations

from datetime import date
from typing import Literal

from pydantic import BaseModel, Field, field_validator, model_validator


class SalesRecord(BaseModel):
    date: date
    units_sold: int = Field(ge=0)
    product_id: str = Field(min_length=1, max_length=200)

    @field_validator("product_id")
    @classmethod
    def normalize_product_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("product_id cannot be empty.")
        return normalized


class ForecastRequest(BaseModel):
    product_id: str = Field(min_length=1, max_length=200)
    sales_history: list[SalesRecord] = Field(min_length=1)
    current_stock: int = Field(ge=0)
    lead_time_days: int = Field(default=7, ge=1)
    safety_stock_days: int = Field(default=5, ge=0)

    @field_validator("product_id")
    @classmethod
    def normalize_product_id(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("product_id cannot be empty.")
        return normalized

    @model_validator(mode="after")
    def ensure_matching_product_history(self) -> "ForecastRequest":
        mismatched = [
            record.product_id
            for record in self.sales_history
            if record.product_id != self.product_id
        ]
        if mismatched:
            raise ValueError("sales_history records must match product_id.")
        return self


class StockForecast(BaseModel):
    product_id: str
    days_until_stockout: int
    predicted_stockout_date: date
    recommended_reorder_qty: int = Field(ge=0)
    reorder_urgency: Literal["critical", "soon", "healthy"]
    weekly_demand_forecast: list[int] = Field(min_length=4, max_length=4)
    confidence_score: float = Field(ge=0.0, le=1.0)


class ForecastResponse(BaseModel):
    product_id: str
    forecast: StockForecast
    recommendation: str


class CSVUploadResponse(BaseModel):
    parsed_records: list[SalesRecord]
    product_ids_found: list[str]
    row_count: int = Field(ge=0)
