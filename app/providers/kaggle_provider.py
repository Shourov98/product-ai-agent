from __future__ import annotations

import asyncio
import logging
import os
import time
from dataclasses import asdict
from datetime import UTC, datetime
from hashlib import md5
from pathlib import Path
from typing import Any

import pandas as pd

from app.core.exceptions import ProductNotFoundError, ProviderUnavailableError
from app.providers.base import (
    CompetitorSnapshot,
    PriceDataProvider,
    PriceUpdateResult,
    ProductData,
    ProviderHealth,
)


logger = logging.getLogger(__name__)
logging.getLogger("pandas").setLevel(logging.WARNING)


class KaggleProvider(PriceDataProvider):
    def __init__(
        self,
        *,
        products_csv: str | None = None,
        categories_csv: str | None = None,
    ) -> None:
        self.products_csv = Path(products_csv or os.getenv("KAGGLE_PRODUCTS_CSV") or self._default_products_csv())
        self.categories_csv = Path(categories_csv or os.getenv("KAGGLE_CATEGORIES_CSV") or self._default_categories_csv())
        self.products = pd.DataFrame()
        self.categories = pd.DataFrame()
        self.category_stats: dict[str, dict[str, float]] = {}
        self._load_data()

    async def get_product(self, asin: str) -> ProductData:
        return await asyncio.to_thread(self._get_product_sync, asin)

    async def get_competitor_snapshot(self, asin: str) -> CompetitorSnapshot:
        return await asyncio.to_thread(self._get_competitor_snapshot_sync, asin)

    async def update_price(self, asin: str, new_price: float) -> PriceUpdateResult:
        product = await self.get_product(asin)
        logger.info("DEMO: Would update %s to $%.2f", asin, new_price)
        return PriceUpdateResult(
            asin=asin,
            success=True,
            old_price=product.our_price,
            new_price=round(new_price, 2),
            marketplace=product.marketplace,
            simulated=True,
        )

    async def list_products(self, limit: int, category: str | None) -> list[ProductData]:
        return await asyncio.to_thread(self._list_products_sync, limit, category)

    async def list_categories(self) -> list[str]:
        return await asyncio.to_thread(self._list_categories_sync)

    async def health_check(self) -> ProviderHealth:
        start = time.perf_counter()
        healthy = not self.products.empty and not self.categories.empty
        error = None if healthy else "Kaggle provider dataframes are not loaded."
        if healthy:
            error = f"Loaded {len(self.products)} products across {len(self.category_stats)} categories."
        return ProviderHealth(
            provider_name="kaggle_dataset",
            is_healthy=healthy,
            latency_ms=round((time.perf_counter() - start) * 1000, 2),
            error=error,
        )

    def _load_data(self) -> None:
        if not self.products_csv.exists():
            raise ProviderUnavailableError(f"Products CSV not found: {self.products_csv}")
        if not self.categories_csv.exists():
            raise ProviderUnavailableError(f"Categories CSV not found: {self.categories_csv}")

        product_columns = [
            "asin",
            "title",
            "stars",
            "reviews",
            "price",
            "listPrice",
            "category_id",
            "isBestSeller",
            "boughtInLastMonth",
        ]
        self.products = pd.read_csv(self.products_csv, usecols=product_columns)
        self.categories = pd.read_csv(self.categories_csv)

        self.products["price"] = pd.to_numeric(self.products["price"], errors="coerce")
        self.products["listPrice"] = pd.to_numeric(self.products["listPrice"], errors="coerce").fillna(0.0)
        self.products["stars"] = pd.to_numeric(self.products["stars"], errors="coerce").fillna(0.0)
        self.products["reviews"] = pd.to_numeric(self.products["reviews"], errors="coerce").fillna(0).astype(int)
        self.products["boughtInLastMonth"] = (
            pd.to_numeric(self.products["boughtInLastMonth"], errors="coerce").fillna(0).astype(int)
        )
        self.products["category_id"] = self.products["category_id"].astype(str)
        self.categories["id"] = self.categories["id"].astype(str)
        self.products = self.products.dropna(subset=["asin", "title", "price", "category_id"])
        self.products = self.products[self.products["price"] > 0].copy()
        self.products = self.products.merge(
            self.categories.rename(columns={"id": "category_id"}),
            on="category_id",
            how="left",
        )
        self.products["category_name"] = self.products["category_name"].fillna("Uncategorized")
        self.products["isBestSeller"] = self.products["isBestSeller"].apply(self._coerce_bool)
        self.products = self.products.sort_values("asin").reset_index(drop=True)
        self.category_stats = self._build_category_stats()

    def _build_category_stats(self) -> dict[str, dict[str, float]]:
        stats: dict[str, dict[str, float]] = {}
        grouped = self.products.groupby("category_id")["price"]
        for category_id, prices in grouped:
            stats[str(category_id)] = {
                "mean": round(float(prices.mean()), 2),
                "median": round(float(prices.median()), 2),
                "min": round(float(prices.min()), 2),
                "p25": round(float(prices.quantile(0.25)), 2),
                "count": float(prices.count()),
            }
        return stats

    def _get_product_sync(self, asin: str) -> ProductData:
        row = self._find_row(asin)
        return self._product_from_row(row)

    def _get_competitor_snapshot_sync(self, asin: str) -> CompetitorSnapshot:
        product = self._get_product_sync(asin)
        stats = self.category_stats.get(product.category_id)
        if stats is None:
            stats = {
                "mean": product.our_price,
                "median": product.our_price,
                "min": product.our_price,
                "p25": product.our_price,
                "count": 1.0,
            }

        average_price = float(stats["mean"])
        demand_signal = self._demand_signal(product.units_sold_30d)
        if product.our_price < average_price * 0.95:
            our_position = "below_avg"
        elif product.our_price > average_price * 1.05:
            our_position = "above_avg"
        else:
            our_position = "at_avg"

        return CompetitorSnapshot(
            asin=asin,
            buy_box_price=round(float(stats["median"]), 2),
            lowest_price=round(float(stats["min"]), 2),
            average_price=round(average_price, 2),
            median_price=round(float(stats["median"]), 2),
            competitive_floor=round(float(stats["p25"]), 2),
            competitor_count=max(1, int(stats["count"])),
            demand_signal=demand_signal,
            our_position=our_position,
            source="kaggle_dataset",
            fetched_at=datetime.now(UTC).isoformat(),
        )

    def _list_products_sync(self, limit: int, category: str | None) -> list[ProductData]:
        frame = self.products
        if category:
            lowered = category.lower()
            frame = frame[frame["category_name"].astype(str).str.lower() == lowered]
        frame = frame.sort_values("asin").head(max(0, limit))
        return [self._product_from_row(row) for _, row in frame.iterrows()]

    def _list_categories_sync(self) -> list[str]:
        return sorted(
            str(category)
            for category in self.products["category_name"].dropna().unique().tolist()
            if str(category).strip()
        )

    def _find_row(self, asin: str) -> pd.Series:
        normalized = asin.strip()
        matches = self.products[self.products["asin"] == normalized]
        if matches.empty:
            raise ProductNotFoundError(f"Product not found for ASIN {asin}.")
        return matches.iloc[0]

    @staticmethod
    def _product_from_row(row: pd.Series) -> ProductData:
        asin = str(row["asin"])
        our_price = round(float(row["price"]), 2)
        list_price = round(float(row.get("listPrice", 0.0) or 0.0), 2)
        bought_last_month = int(row.get("boughtInLastMonth", 0) or 0)
        seed = int(md5(asin.encode("utf-8")).hexdigest()[:8], 16) % 1000
        return ProductData(
            asin=asin,
            title=str(row["title"]),
            our_price=our_price,
            list_price=list_price,
            cost_price=round(our_price * 0.55, 2),
            current_stock=5 + (seed % 145),
            units_sold_7d=bought_last_month // 4,
            units_sold_30d=bought_last_month,
            min_price=round(our_price * 0.75, 2),
            max_price=round(our_price * 1.35, 2),
            is_best_seller=bool(row.get("isBestSeller", False)),
            stars=round(float(row.get("stars", 0.0) or 0.0), 2),
            reviews=int(row.get("reviews", 0) or 0),
            category_id=str(row.get("category_id", "")),
            category_name=str(row.get("category_name", "Uncategorized")),
        )

    @staticmethod
    def _demand_signal(units_sold_30d: int) -> str:
        if units_sold_30d > 1000:
            return "high"
        if units_sold_30d > 200:
            return "medium"
        return "low"

    @staticmethod
    def _coerce_bool(value: Any) -> bool:
        if isinstance(value, bool):
            return value
        return str(value).strip().lower() in {"true", "1", "yes", "y"}

    @staticmethod
    def _default_products_csv() -> str:
        local = Path("Datasets") / "Amazon Products Dataset 2023 (1.4M Products)" / "amazon_products.csv"
        if local.exists():
            return str(local)
        return "data/amazon_products.csv"

    @staticmethod
    def _default_categories_csv() -> str:
        local = Path("Datasets") / "Amazon Products Dataset 2023 (1.4M Products)" / "amazon_categories.csv"
        if local.exists():
            return str(local)
        return "data/amazon_categories.csv"

    @staticmethod
    def snapshot_to_dict(snapshot: CompetitorSnapshot) -> dict[str, Any]:
        return asdict(snapshot)
