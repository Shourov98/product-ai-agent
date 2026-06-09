from __future__ import annotations

import base64
import time
from statistics import mean

import httpx

from app.schemas.response import MarketplaceResearchResponse, ResearchEvidenceResponse


class EbayMarketResearchServiceError(RuntimeError):
    pass


class EbayMarketResearchService:
    def __init__(
        self,
        *,
        enabled: bool,
        client_id: str | None,
        client_secret: str | None,
        marketplace_id: str,
        api_base_url: str,
        identity_base_url: str,
        search_limit: int,
    ) -> None:
        self.enabled = enabled and bool(client_id) and bool(client_secret)
        self.client_id = client_id
        self.client_secret = client_secret
        self.marketplace_id = marketplace_id
        self.api_base_url = api_base_url.rstrip("/")
        self.identity_base_url = identity_base_url.rstrip("/")
        self.search_limit = max(1, min(search_limit, 20))
        self._access_token: str | None = None
        self._access_token_expires_at: float = 0.0

    async def search(self, queries: list[str]) -> MarketplaceResearchResponse | None:
        if not self.enabled:
            return None

        query = next((candidate.strip() for candidate in queries if candidate.strip()), "")
        if not query:
            return None

        token = await self._get_access_token()
        headers = {
            "Authorization": f"Bearer {token}",
            "X-EBAY-C-MARKETPLACE-ID": self.marketplace_id,
            "Accept": "application/json",
        }
        params = {
            "q": query,
            "limit": str(self.search_limit),
            "fieldgroups": "ASPECT_REFINEMENTS",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.get(
                f"{self.api_base_url}/buy/browse/v1/item_summary/search",
                headers=headers,
                params=params,
            )
        if response.status_code >= 400:
            raise EbayMarketResearchServiceError(
                f"eBay Browse API search failed with status {response.status_code}: {response.text[:240]}"
            )

        payload = response.json()
        listings = self._parse_item_summaries(payload.get("itemSummaries", []))
        prices = [listing.price for listing in listings if listing.price is not None]
        keyword_signals = self._parse_refinement_signals(payload)
        return MarketplaceResearchResponse(
            marketplace="ebay",
            search_queries=queries,
            keyword_signals=keyword_signals,
            price_min=min(prices) if prices else None,
            price_max=max(prices) if prices else None,
            price_avg=round(mean(prices), 2) if prices else None,
            regular_price_avg=round(mean(prices), 2) if prices else None,
            sale_price_avg=None,
            discount_percent_avg=None,
            similar_listings=listings,
        )

    async def _get_access_token(self) -> str:
        now = time.time()
        if self._access_token is not None and now < self._access_token_expires_at - 60:
            return self._access_token

        if not self.client_id or not self.client_secret:
            raise EbayMarketResearchServiceError("Missing eBay client credentials.")

        credentials = f"{self.client_id}:{self.client_secret}".encode("utf-8")
        encoded = base64.b64encode(credentials).decode("ascii")
        headers = {
            "Authorization": f"Basic {encoded}",
            "Content-Type": "application/x-www-form-urlencoded",
        }
        data = {
            "grant_type": "client_credentials",
            "scope": "https://api.ebay.com/oauth/api_scope",
        }

        async with httpx.AsyncClient(timeout=30.0) as client:
            response = await client.post(
                f"{self.identity_base_url}/identity/v1/oauth2/token",
                headers=headers,
                data=data,
            )
        if response.status_code >= 400:
            raise EbayMarketResearchServiceError(
                f"eBay OAuth token request failed with status {response.status_code}: {response.text[:240]}"
            )

        payload = response.json()
        access_token = str(payload.get("access_token", "")).strip()
        expires_in = int(payload.get("expires_in", 7200))
        if not access_token:
            raise EbayMarketResearchServiceError("eBay OAuth token response did not include access_token.")
        self._access_token = access_token
        self._access_token_expires_at = now + expires_in
        return access_token

    def _parse_item_summaries(self, items: list[dict]) -> list[ResearchEvidenceResponse]:
        parsed: list[ResearchEvidenceResponse] = []
        for index, item in enumerate(items, start=1):
            title = str(item.get("title", "")).strip()
            if not title:
                continue
            price_block = item.get("price") or {}
            price_raw = price_block.get("value")
            try:
                price = float(price_raw) if price_raw is not None else None
            except (TypeError, ValueError):
                price = None

            attributes = {}
            for aspect in item.get("localizedAspects", []) or []:
                key = str(aspect.get("name", "")).strip()
                value = str(aspect.get("value", "")).strip()
                if key and value:
                    attributes[key] = value

            category_names = [category.get("categoryName") for category in item.get("categories", []) if category.get("categoryName")]
            observations = []
            if category_names:
                observations.append(f"Listed under {' > '.join(category_names[:3])}.")
            if item.get("buyingOptions"):
                observations.append(f"Buying options: {', '.join(item['buyingOptions'])}.")
            if item.get("condition"):
                observations.append(f"Condition: {item['condition']}.")

            parsed.append(
                ResearchEvidenceResponse(
                    source="ebay",
                    title=title,
                    url=str(item.get("itemWebUrl") or item.get("itemAffiliateWebUrl") or "").strip() or None,
                    price=price,
                    currency=str(price_block.get("currency", "USD")),
                    relevance_score=max(0.6, 0.98 - (index * 0.05)),
                    attributes=attributes,
                    observations=observations,
                )
            )
        return parsed

    @staticmethod
    def _parse_refinement_signals(payload: dict) -> list[str]:
        signals: list[str] = []
        for refinement in payload.get("refinement", {}).get("aspectDistributions", []) or []:
            aspect = str(refinement.get("localizedAspectName", "")).strip()
            if aspect:
                signals.append(aspect.lower())
            values = refinement.get("aspectValueDistributions", []) or []
            for item in values[:3]:
                value = str(item.get("localizedAspectValue", "")).strip()
                if value:
                    signals.append(value.lower())
        deduped: list[str] = []
        seen: set[str] = set()
        for signal in signals:
            if signal in seen:
                continue
            seen.add(signal)
            deduped.append(signal)
        return deduped[:12]
