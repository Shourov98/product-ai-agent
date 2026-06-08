from __future__ import annotations

import asyncio
import json
from dataclasses import replace
from pathlib import Path

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app
from app.providers.factory import get_provider, reset_provider_cache
from app.providers.kaggle_provider import KaggleProvider
from app.schemas.repricing import AIDecision
from app.services.repricing_engine import RepricingEngine


PRODUCT_ROWS = [
    ("B000000001", "Best Seller High Demand", 4.8, 1200, 100.0, 130.0, 10, True, 2000),
    ("B000000002", "Best Seller Slow Sales", 4.5, 500, 80.0, 95.0, 10, True, 20),
    ("B000000003", "High Demand Regular", 4.2, 300, 60.0, 75.0, 20, False, 700),
    ("B000000004", "Slow Mover", 3.9, 50, 40.0, 55.0, 20, False, 10),
    ("B000000005", "Mid Range Competitive", 4.0, 250, 70.0, 90.0, 10, False, 250),
    ("B000000006", "Above Average Product", 4.7, 100, 150.0, 180.0, 10, False, 30),
    ("B000000007", "Below Average Product", 3.8, 40, 30.0, 40.0, 20, False, 5),
    ("B000000008", "Kitchen Item", 4.1, 70, 35.0, 45.0, 30, False, 100),
    ("B000000009", "Kitchen Premium", 4.6, 210, 55.0, 65.0, 30, False, 400),
    ("B000000010", "Kitchen Budget", 4.0, 20, 22.0, 30.0, 30, False, 15),
    ("B000000011", "Sports Item 1", 4.3, 80, 25.0, 35.0, 40, False, 60),
    ("B000000012", "Sports Item 2", 4.4, 90, 28.0, 38.0, 40, False, 75),
    ("B000000013", "Sports Item 3", 4.1, 35, 31.0, 39.0, 40, False, 18),
    ("B000000014", "Office Item 1", 4.2, 95, 45.0, 60.0, 50, False, 85),
    ("B000000015", "Office Item 2", 4.9, 500, 65.0, 85.0, 50, True, 1500),
    ("B000000016", "Office Item 3", 3.7, 12, 20.0, 28.0, 50, False, 0),
    ("B000000017", "Home Item 1", 4.5, 140, 75.0, 100.0, 60, False, 300),
    ("B000000018", "Home Item 2", 4.6, 160, 85.0, 110.0, 60, False, 450),
    ("B000000019", "Home Item 3", 4.1, 65, 95.0, 125.0, 60, False, 12),
    ("B000000020", "Home Item 4", 3.9, 25, 105.0, 140.0, 60, False, 8),
]


@pytest.fixture
def kaggle_files(tmp_path: Path) -> tuple[Path, Path]:
    products = tmp_path / "amazon_products.csv"
    categories = tmp_path / "amazon_categories.csv"
    categories.write_text(
        "id,category_name\n10,Luggage\n20,Electronics\n30,Kitchen\n40,Sports\n50,Office\n60,Home\n",
        encoding="utf-8",
    )
    lines = ["asin,title,stars,reviews,price,listPrice,category_id,isBestSeller,boughtInLastMonth"]
    for row in PRODUCT_ROWS:
        asin, title, stars, reviews, price, list_price, category_id, best_seller, bought = row
        lines.append(
            f"{asin},{title},{stars},{reviews},{price},{list_price},{category_id},{best_seller},{bought}"
        )
    products.write_text("\n".join(lines), encoding="utf-8")
    return products, categories


@pytest.fixture
def kaggle_provider(kaggle_files: tuple[Path, Path]) -> KaggleProvider:
    products, categories = kaggle_files
    return KaggleProvider(products_csv=str(products), categories_csv=str(categories))


class _MockMessage:
    content = json.dumps(
        {
            "recommended_price": 90.0,
            "action": "lower",
            "strategy_used": "buy_box_win",
            "reason": "Mocked competitive adjustment.",
            "confidence": 0.8,
            "margin_at_new_price": 45.0,
        }
    )


class _MockChoice:
    message = _MockMessage()


class _MockCompletions:
    async def create(self, **kwargs):
        return type("MockResponse", (), {"choices": [_MockChoice()]})()


class _MockChat:
    completions = _MockCompletions()


class _MockOpenAI:
    chat = _MockChat()


@pytest.fixture
def mock_openai() -> _MockOpenAI:
    return _MockOpenAI()


def test_provider_factory_returns_kaggle_without_env(
    monkeypatch: pytest.MonkeyPatch,
    kaggle_files: tuple[Path, Path],
) -> None:
    products, categories = kaggle_files
    monkeypatch.delenv("AMAZON_CLIENT_ID", raising=False)
    monkeypatch.delenv("AMAZON_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("AMAZON_REFRESH_TOKEN", raising=False)
    monkeypatch.setenv("KAGGLE_PRODUCTS_CSV", str(products))
    monkeypatch.setenv("KAGGLE_CATEGORIES_CSV", str(categories))
    reset_provider_cache()

    provider = get_provider()

    assert isinstance(provider, KaggleProvider)
    reset_provider_cache()


def test_kaggle_provider_get_product_known_asin(kaggle_provider: KaggleProvider) -> None:
    product = asyncio.run(kaggle_provider.get_product("B000000001"))

    assert product.asin == "B000000001"
    assert product.title
    assert product.cost_price == pytest.approx(product.our_price * 0.55)
    assert product.min_price < product.our_price < product.max_price


def test_kaggle_provider_stock_is_deterministic(kaggle_provider: KaggleProvider) -> None:
    first = asyncio.run(kaggle_provider.get_product("B000000001"))
    second = asyncio.run(kaggle_provider.get_product("B000000001"))

    assert first.current_stock == second.current_stock


def test_competitor_snapshot_logical_consistency(kaggle_provider: KaggleProvider) -> None:
    snapshot = asyncio.run(kaggle_provider.get_competitor_snapshot("B000000001"))

    assert snapshot.lowest_price <= snapshot.average_price
    assert snapshot.competitive_floor <= snapshot.average_price
    assert snapshot.demand_signal in ["high", "medium", "low"]


def test_guardrails_enforce_minimum_margin(kaggle_provider: KaggleProvider) -> None:
    engine = RepricingEngine(kaggle_provider, None)
    product = asyncio.run(kaggle_provider.get_product("B000000001"))
    decision = AIDecision(
        recommended_price=product.cost_price * 1.05,
        action="lower",
        strategy_used="test",
        reason="test",
        confidence=0.5,
        margin_at_new_price=5,
    )

    safe_price = engine._apply_guardrails(decision.recommended_price, product, decision)

    assert safe_price >= round(product.cost_price * 1.15, 2)


def test_guardrails_enforce_floor(kaggle_provider: KaggleProvider) -> None:
    engine = RepricingEngine(kaggle_provider, None)
    base_product = asyncio.run(kaggle_provider.get_product("B000000001"))
    product = replace(base_product, min_price=85.0)
    decision = AIDecision(
        recommended_price=product.min_price - 10,
        action="lower",
        strategy_used="test",
        reason="test",
        confidence=0.5,
        margin_at_new_price=5,
    )

    safe_price = engine._apply_guardrails(decision.recommended_price, product, decision)

    assert safe_price == product.min_price


def test_guardrails_max_drop_20pct(kaggle_provider: KaggleProvider) -> None:
    engine = RepricingEngine(kaggle_provider, None)
    product = asyncio.run(kaggle_provider.get_product("B000000001"))
    decision = AIDecision(
        recommended_price=product.our_price * 0.50,
        action="lower",
        strategy_used="test",
        reason="test",
        confidence=0.5,
        margin_at_new_price=5,
    )

    safe_price = engine._apply_guardrails(decision.recommended_price, product, decision)

    assert safe_price >= round(product.our_price * 0.80, 2)


def test_full_pipeline_dry_run(kaggle_provider: KaggleProvider, mock_openai: _MockOpenAI) -> None:
    engine = RepricingEngine(kaggle_provider, mock_openai)

    result = asyncio.run(engine.run("B000000001", "auto", True))

    assert result.price_update_result is None
    assert result.asin == "B000000001"
    assert result.product_name
    assert result.competitor_snapshot
    assert result.data_source == "kaggle_dataset"


def test_full_pipeline_apply_price(kaggle_provider: KaggleProvider, mock_openai: _MockOpenAI) -> None:
    engine = RepricingEngine(kaggle_provider, mock_openai)

    result = asyncio.run(engine.run("B000000001", "auto", False))

    assert result.price_update_result is not None
    assert result.price_update_result["simulated"] is True
    assert result.price_update_result["success"] is True


def test_bulk_continues_on_single_failure(kaggle_provider: KaggleProvider, mock_openai: _MockOpenAI) -> None:
    engine = RepricingEngine(kaggle_provider, mock_openai)

    response = asyncio.run(engine.run_bulk(["B000000001", "INVALID", "B000000002"], "auto", True))

    assert len(response.errors) == 1
    assert len(response.results) == 2


def test_demo_returns_5_scenarios(
    monkeypatch: pytest.MonkeyPatch,
    kaggle_files: tuple[Path, Path],
) -> None:
    products, categories = kaggle_files
    monkeypatch.setenv("KAGGLE_PRODUCTS_CSV", str(products))
    monkeypatch.setenv("KAGGLE_CATEGORIES_CSV", str(categories))
    monkeypatch.setenv("OPENAI_ENABLED", "false")
    monkeypatch.delenv("AMAZON_CLIENT_ID", raising=False)
    monkeypatch.delenv("AMAZON_CLIENT_SECRET", raising=False)
    monkeypatch.delenv("AMAZON_REFRESH_TOKEN", raising=False)
    reset_provider_cache()

    async def run_request():
        async with AsyncClient(transport=ASGITransport(app=app), base_url="http://testserver") as client:
            return await client.get("/repricing/demo")

    response = asyncio.run(run_request())

    assert response.status_code == 200
    body = response.json()
    assert len(body["results"]) == 5
    assert len(body["demo_scenarios"]) == 5
    reset_provider_cache()
