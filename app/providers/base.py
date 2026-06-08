from __future__ import annotations

from abc import ABC, abstractmethod
from dataclasses import dataclass


@dataclass(slots=True)
class ProductData:
    asin: str
    title: str
    our_price: float
    list_price: float
    cost_price: float
    current_stock: int
    units_sold_7d: int
    units_sold_30d: int
    min_price: float
    max_price: float
    is_best_seller: bool
    stars: float
    reviews: int
    category_id: str
    category_name: str
    currency: str = "USD"
    marketplace: str = "amazon"


@dataclass(slots=True)
class CompetitorSnapshot:
    asin: str
    buy_box_price: float
    lowest_price: float
    average_price: float
    median_price: float
    competitive_floor: float
    competitor_count: int
    demand_signal: str
    our_position: str
    source: str
    fetched_at: str


@dataclass(slots=True)
class PriceUpdateResult:
    asin: str
    success: bool
    old_price: float
    new_price: float
    marketplace: str
    error: str | None = None
    simulated: bool = True


@dataclass(slots=True)
class ProviderHealth:
    provider_name: str
    is_healthy: bool
    latency_ms: float
    error: str | None = None


class PriceDataProvider(ABC):
    @abstractmethod
    async def get_product(self, asin: str) -> ProductData:
        raise NotImplementedError

    @abstractmethod
    async def get_competitor_snapshot(self, asin: str) -> CompetitorSnapshot:
        raise NotImplementedError

    @abstractmethod
    async def update_price(self, asin: str, new_price: float) -> PriceUpdateResult:
        raise NotImplementedError

    @abstractmethod
    async def list_products(self, limit: int, category: str | None) -> list[ProductData]:
        raise NotImplementedError

    @abstractmethod
    async def list_categories(self) -> list[str]:
        raise NotImplementedError

    @abstractmethod
    async def health_check(self) -> ProviderHealth:
        raise NotImplementedError
