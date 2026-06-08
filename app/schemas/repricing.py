from __future__ import annotations

from typing import Any, Literal

from pydantic import BaseModel, Field, field_validator


StrategyLiteral = Literal["auto", "buy_box_win", "profit_maximize", "scarcity", "velocity"]
ActionLiteral = Literal["lower", "raise", "hold"]


class RepricingRequest(BaseModel):
    asin: str = Field(min_length=1, max_length=32)
    strategy: StrategyLiteral = "auto"
    dry_run: bool = True

    @field_validator("asin")
    @classmethod
    def normalize_asin(cls, value: str) -> str:
        normalized = value.strip()
        if not normalized:
            raise ValueError("asin cannot be empty.")
        return normalized


class PricingBoundaries(BaseModel):
    min_price: float = Field(gt=0)
    max_price: float = Field(gt=0)
    min_margin_pct: float = Field(default=0.15, ge=0)
    max_drop_pct: float = Field(default=0.20, ge=0)
    max_raise_pct: float = Field(default=0.20, ge=0)


class AIDecision(BaseModel):
    recommended_price: float = Field(gt=0)
    action: ActionLiteral
    strategy_used: str
    reason: str
    confidence: float = Field(ge=0.0, le=1.0)
    margin_at_new_price: float
    guardrails_applied: bool = False


class RepricingResult(BaseModel):
    asin: str
    product_name: str
    category: str
    marketplace: str
    timestamp: str
    is_best_seller: bool
    demand_signal: str
    stars: float
    reviews: int
    pricing: dict[str, Any]
    competitor_snapshot: dict[str, Any]
    ai_decision: AIDecision
    price_update_result: dict[str, Any] | None = None
    data_source: str


class BulkRepricingRequest(BaseModel):
    asins: list[str] = Field(min_length=1, max_length=10)
    strategy: StrategyLiteral = "auto"
    dry_run: bool = True

    @field_validator("asins")
    @classmethod
    def normalize_asins(cls, value: list[str]) -> list[str]:
        normalized = [item.strip() for item in value if item.strip()]
        if not normalized:
            raise ValueError("asins cannot be empty.")
        return normalized


class BulkRepricingResponse(BaseModel):
    total_analyzed: int = Field(ge=0)
    price_lowered: int = Field(ge=0)
    price_raised: int = Field(ge=0)
    price_held: int = Field(ge=0)
    total_potential_savings: float
    results: list[RepricingResult]
    errors: list[dict[str, str]]
    data_source: str


class DemoResponse(BulkRepricingResponse):
    demo_scenarios: list[str]


class ProductRepricingRequest(BaseModel):
    strategy: StrategyLiteral = "auto"
    dry_run: bool = True


class ProductMatchResponse(BaseModel):
    asin: str
    title: str
    category: str
    confidence: float = Field(ge=0.0, le=1.0)
    source: str


class ProductRepricingResponse(BaseModel):
    product_id: str
    matched_product: ProductMatchResponse
    repricing: RepricingResult
