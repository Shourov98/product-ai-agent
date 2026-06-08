from __future__ import annotations

from dataclasses import asdict
from typing import Any, Callable

from fastapi import APIRouter, HTTPException, Query, status

from app.config import get_settings
from app.core.exceptions import ProductNotFoundError, ProviderUnavailableError
from app.providers.base import PriceDataProvider, ProductData
from app.providers.factory import get_provider
from app.providers.kaggle_provider import KaggleProvider
from app.schemas.repricing import (
    BulkRepricingRequest,
    BulkRepricingResponse,
    DemoResponse,
    RepricingRequest,
    RepricingResult,
)
from app.services.repricing_engine import RepricingEngine


router = APIRouter(prefix="/repricing", tags=["Repricing"])


def get_openai_client() -> Any | None:
    settings = get_settings()
    if not settings.openai_enabled or not settings.openai_api_key:
        return None
    try:
        from openai import AsyncOpenAI
    except ModuleNotFoundError:
        return None
    return AsyncOpenAI(api_key=settings.openai_api_key)


def get_engine() -> RepricingEngine:
    provider = get_provider()
    openai_client = get_openai_client()
    return RepricingEngine(provider, openai_client)


@router.get("/health", status_code=status.HTTP_200_OK)
async def repricing_health() -> dict[str, Any]:
    provider = get_provider()
    health = await provider.health_check()
    data_source = health.provider_name
    message = (
        "Using Kaggle demo provider. Ready for real API: implement AmazonSPAPIProvider and set SP-API env vars."
        if data_source == "kaggle_dataset"
        else "Using production provider."
    )
    return {
        "provider": health.provider_name,
        "is_healthy": health.is_healthy,
        "latency_ms": health.latency_ms,
        "data_source": data_source,
        "message": message,
        "detail": health.error,
    }


@router.get("/categories", status_code=status.HTTP_200_OK)
async def list_repricing_categories() -> dict[str, list[str]]:
    try:
        return {"categories": await get_provider().list_categories()}
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.get("/products", status_code=status.HTTP_200_OK)
async def list_repricing_products(
    limit: int = Query(default=20, ge=1, le=200),
    category: str | None = Query(default=None),
) -> dict[str, Any]:
    try:
        products = await get_provider().list_products(limit=limit, category=category)
    except ProviderUnavailableError as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc
    return {"products": [asdict(product) for product in products], "count": len(products)}


@router.post(
    "/analyze",
    response_model=RepricingResult,
    status_code=status.HTTP_200_OK,
)
async def analyze_repricing(payload: RepricingRequest) -> RepricingResult:
    try:
        return await get_engine().run(payload.asin, payload.strategy, payload.dry_run)
    except ProductNotFoundError as exc:
        raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail=str(exc)) from exc
    except (ProviderUnavailableError, RuntimeError, ValueError, NotImplementedError) as exc:
        raise HTTPException(status_code=status.HTTP_500_INTERNAL_SERVER_ERROR, detail=str(exc)) from exc


@router.post(
    "/analyze-bulk",
    response_model=BulkRepricingResponse,
    status_code=status.HTTP_200_OK,
)
async def analyze_repricing_bulk(payload: BulkRepricingRequest) -> BulkRepricingResponse:
    return await get_engine().run_bulk(payload.asins, payload.strategy, payload.dry_run)


@router.get(
    "/demo",
    response_model=DemoResponse,
    status_code=status.HTTP_200_OK,
)
async def repricing_demo() -> DemoResponse:
    provider = get_provider()
    engine = get_engine()
    products = await provider.list_products(limit=5000, category=None)
    scenarios = await _select_demo_scenarios(provider, products)
    scenario_labels = [item[0] for item in scenarios]
    response = await engine.run_bulk([item[1].asin for item in scenarios], strategy="auto", dry_run=True)
    return DemoResponse(
        total_analyzed=response.total_analyzed,
        price_lowered=response.price_lowered,
        price_raised=response.price_raised,
        price_held=response.price_held,
        total_potential_savings=response.total_potential_savings,
        results=response.results,
        errors=response.errors,
        data_source=response.data_source,
        demo_scenarios=scenario_labels,
    )


async def _select_demo_scenarios(
    provider: PriceDataProvider,
    products: list[ProductData],
) -> list[tuple[str, ProductData]]:
    selected: list[tuple[str, ProductData]] = []
    used_asins: set[str] = set()

    def add(label: str, product: ProductData | None) -> None:
        if product is None:
            return
        if product.asin in used_asins:
            return
        selected.append((label, product))
        used_asins.add(product.asin)

    add(
        "Best Seller + High Demand",
        _first(products, lambda item: item.is_best_seller and item.units_sold_30d > 1000),
    )
    add(
        "Best Seller + Slow Sales",
        _first(products, lambda item: item.is_best_seller and item.units_sold_30d < 50),
    )
    add(
        "High Demand, Not Best Seller",
        _first(products, lambda item: not item.is_best_seller and item.units_sold_30d > 500),
    )
    add(
        "Slow Mover",
        _first(products, lambda item: not item.is_best_seller and item.units_sold_30d < 20),
    )

    mid_range = None
    for product in products[:300]:
        snapshot = await provider.get_competitor_snapshot(product.asin)
        if snapshot.our_position == "at_avg":
            mid_range = product
            break
    add("Mid-Range Competitive", mid_range)

    for product in products:
        if len(selected) >= 5:
            break
        add(f"Fallback Scenario {len(selected) + 1}", product)

    return selected[:5]


def _first(products: list[ProductData], predicate: Callable[[ProductData], bool]) -> ProductData | None:
    for product in products:
        if predicate(product):
            return product
    return None
