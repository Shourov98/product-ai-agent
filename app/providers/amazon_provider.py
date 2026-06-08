from __future__ import annotations

import os

from app.providers.base import (
    CompetitorSnapshot,
    PriceDataProvider,
    PriceUpdateResult,
    ProductData,
    ProviderHealth,
)


class AmazonSPAPIProvider(PriceDataProvider):
    def __init__(self) -> None:
        self.client_id = os.getenv("AMAZON_CLIENT_ID")
        self.client_secret = os.getenv("AMAZON_CLIENT_SECRET")
        self.refresh_token = os.getenv("AMAZON_REFRESH_TOKEN")
        self.marketplace_id = os.getenv("AMAZON_MARKETPLACE_ID", "ATVPDKIKX0DR")

    @classmethod
    def is_configured(cls) -> bool:
        return all(
            [
                os.getenv("AMAZON_CLIENT_ID"),
                os.getenv("AMAZON_CLIENT_SECRET"),
                os.getenv("AMAZON_REFRESH_TOKEN"),
            ]
        )

    async def get_product(self, asin: str) -> ProductData:
        # SP-API: GET /catalog/2022-04-01/items/{asin}
        # SP-API: GET /products/pricing/v0/price?Asins={asin}
        raise NotImplementedError(
            "AmazonSPAPIProvider.get_product is not implemented yet. "
            "Configure SP-API auth, then map Catalog Items and Pricing responses to ProductData."
        )

    async def get_competitor_snapshot(self, asin: str) -> CompetitorSnapshot:
        # SP-API: GET /products/pricing/v0/competitivePrice
        # SP-API: GET /products/pricing/v0/listings/{asin}/offers
        raise NotImplementedError(
            "AmazonSPAPIProvider.get_competitor_snapshot is not implemented yet. "
            "Map SP-API competitive pricing and offer data to CompetitorSnapshot."
        )

    async def update_price(self, asin: str, new_price: float) -> PriceUpdateResult:
        # SP-API: PUT /listings/2021-08-01/items/{sellerId}/{sku}
        # SP-API Feeds: POST /feeds/2021-06-30/feeds for bulk price updates.
        raise NotImplementedError(
            "AmazonSPAPIProvider.update_price is not implemented yet. "
            "Use Listings Items API for single-SKU updates or Feeds API for bulk."
        )

    async def list_products(self, limit: int, category: str | None) -> list[ProductData]:
        # SP-API: GET /catalog/2022-04-01/items?keywords=...
        raise NotImplementedError(
            "AmazonSPAPIProvider.list_products is not implemented yet. "
            "Use Catalog Items search or seller listings inventory as the product source."
        )

    async def list_categories(self) -> list[str]:
        raise NotImplementedError(
            "AmazonSPAPIProvider.list_categories is not implemented yet. "
            "Derive categories from seller catalog/listings data."
        )

    async def health_check(self) -> ProviderHealth:
        raise NotImplementedError(
            "AmazonSPAPIProvider.health_check is not implemented yet. "
            "Validate auth token refresh and marketplace access."
        )
