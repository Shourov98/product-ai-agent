from __future__ import annotations

import asyncio
import html
import json
import re
from statistics import mean
from urllib.parse import parse_qs, quote_plus, unquote, urlparse

import httpx

from app.schemas.response import MarketplaceResearchResponse, ResearchEvidenceResponse
from app.utils.product_text import title_keywords, unique_strings


class SearchPriceResearchService:
    _SEARCH_BASE_URL = "https://html.duckduckgo.com/html/"
    _RESULT_LINK_RE = re.compile(r'<a[^>]+class="[^"]*result__a[^"]*"[^>]+href="([^"]+)"', re.IGNORECASE)
    _JSON_LD_RE = re.compile(
        r'<script[^>]+type=["\']application/ld\+json["\'][^>]*>(.*?)</script>',
        re.IGNORECASE | re.DOTALL,
    )
    _META_TAG_RE = re.compile(
        r'<meta[^>]+(?:property|name|itemprop)=["\']([^"\']+)["\'][^>]+content=["\']([^"\']+)["\']',
        re.IGNORECASE,
    )
    _TITLE_RE = re.compile(r"<title[^>]*>(.*?)</title>", re.IGNORECASE | re.DOTALL)
    _PRICE_RE = re.compile(r"(?<!\d)(?:USD|\$|£|EUR|€)\s?(\d{1,5}(?:[.,]\d{2})?)", re.IGNORECASE)

    _META_CURRENT_KEYS = {
        "product:price:amount",
        "og:price:amount",
        "twitter:data1",
        "price",
    }
    _META_LIST_KEYS = {
        "product:list_price:amount",
        "product:original_price:amount",
        "og:list_price:amount",
        "list_price",
        "compare_at_price",
    }
    _META_CURRENCY_KEYS = {
        "product:price:currency",
        "og:price:currency",
        "pricecurrency",
        "currency",
    }

    def __init__(self, *, enabled: bool, timeout_seconds: int, result_limit: int) -> None:
        self.enabled = enabled
        self.timeout_seconds = timeout_seconds
        self.result_limit = max(1, min(result_limit, 8))

    async def search(self, marketplace: str, queries: list[str], attributes: dict[str, str]) -> MarketplaceResearchResponse | None:
        if not self.enabled:
            return None

        query = next((item.strip() for item in queries if item.strip()), "")
        if not query:
            return None

        search_query = f"{marketplace} {query} price sale"
        async with httpx.AsyncClient(
            timeout=float(self.timeout_seconds),
            headers={"User-Agent": "Mozilla/5.0 ProductAIAgent/1.0"},
            follow_redirects=True,
        ) as client:
            search_html = await self._fetch_search_results(client, search_query)
            urls = self._extract_search_urls(search_html)[: self.result_limit]
            if not urls:
                return None

            pages = await asyncio.gather(
                *(self._fetch_candidate_page(client, marketplace, url, attributes) for url in urls),
                return_exceptions=True,
            )

        evidence = [item for item in pages if isinstance(item, ResearchEvidenceResponse)]
        if not evidence:
            return None

        effective_prices = [item.price for item in evidence if item.price is not None]
        regular_prices = [item.list_price for item in evidence if item.list_price is not None]
        sale_prices = [item.sale_price for item in evidence if item.sale_price is not None]
        discount_percents = [item.discount_percent for item in evidence if item.discount_percent is not None]
        keyword_signals = self._keyword_signals(marketplace, query, evidence, attributes)

        return MarketplaceResearchResponse(
            marketplace=marketplace,
            source_mode="search_engine",
            search_queries=queries,
            keyword_signals=keyword_signals,
            price_min=min(effective_prices) if effective_prices else None,
            price_max=max(effective_prices) if effective_prices else None,
            price_avg=round(mean(effective_prices), 2) if effective_prices else None,
            regular_price_avg=round(mean(regular_prices), 2) if regular_prices else None,
            sale_price_avg=round(mean(sale_prices), 2) if sale_prices else None,
            discount_percent_avg=round(mean(discount_percents), 2) if discount_percents else None,
            similar_listings=evidence,
        )

    async def _fetch_search_results(self, client: httpx.AsyncClient, query: str) -> str:
        response = await client.get(f"{self._SEARCH_BASE_URL}?q={quote_plus(query)}")
        response.raise_for_status()
        return response.text

    def _extract_search_urls(self, payload: str) -> list[str]:
        urls: list[str] = []
        for raw in self._RESULT_LINK_RE.findall(payload):
            resolved = html.unescape(raw)
            if "duckduckgo.com/l/?" in resolved:
                parsed = urlparse(resolved)
                redirect = parse_qs(parsed.query).get("uddg", [])
                if redirect:
                    resolved = unquote(redirect[0])
            if resolved.startswith("//"):
                resolved = f"https:{resolved}"
            if not resolved.startswith("http"):
                continue
            if resolved not in urls:
                urls.append(resolved)
        return urls

    async def _fetch_candidate_page(
        self,
        client: httpx.AsyncClient,
        marketplace: str,
        url: str,
        attributes: dict[str, str],
    ) -> ResearchEvidenceResponse | None:
        try:
            response = await client.get(url)
            response.raise_for_status()
        except Exception:
            return None

        payload = response.text
        title = self._extract_title(payload) or urlparse(url).netloc
        price_info = self._extract_price_info(payload)
        current_price = price_info["current_price"]
        list_price = price_info["list_price"]
        sale_price = price_info["sale_price"]
        currency = price_info["currency"] or "USD"
        if current_price is None and sale_price is None and list_price is None:
            return None

        effective_price = sale_price or current_price or list_price
        regular_price = list_price or current_price
        discount_percent = None
        if effective_price is not None and regular_price is not None and regular_price > effective_price:
            discount_percent = round(((regular_price - effective_price) / regular_price) * 100, 2)

        relevance = self._estimate_relevance(title, marketplace, attributes)
        if relevance < 0.45:
            return None

        observations = [f"Source domain: {urlparse(url).netloc}."]
        if regular_price is not None and effective_price is not None and regular_price > effective_price:
            observations.append(
                f"Observed discounted price {effective_price:.2f} from regular price {regular_price:.2f}."
            )

        return ResearchEvidenceResponse(
            source=f"{marketplace}_search",
            title=title[:240],
            url=url,
            price=round(effective_price, 2) if effective_price is not None else None,
            list_price=round(regular_price, 2) if regular_price is not None else None,
            sale_price=round(sale_price, 2) if sale_price is not None else None,
            discount_percent=discount_percent,
            currency=currency,
            relevance_score=relevance,
            attributes={"domain": urlparse(url).netloc},
            observations=observations,
        )

    def _extract_title(self, payload: str) -> str | None:
        match = self._TITLE_RE.search(payload)
        if not match:
            return None
        return html.unescape(re.sub(r"\s+", " ", match.group(1))).strip()

    def _extract_price_info(self, payload: str) -> dict[str, float | str | None]:
        current_price = None
        list_price = None
        sale_price = None
        currency = None

        for key, value in self._META_TAG_RE.findall(payload):
            key_normalized = key.strip().lower()
            numeric = self._parse_price(value)
            if key_normalized in self._META_CURRENT_KEYS and numeric is not None and current_price is None:
                current_price = numeric
            if key_normalized in self._META_LIST_KEYS and numeric is not None and list_price is None:
                list_price = numeric
            if key_normalized in self._META_CURRENCY_KEYS and currency is None:
                currency = value.strip().upper()

        for script_payload in self._JSON_LD_RE.findall(payload):
            extracted = self._extract_from_json_ld(script_payload)
            current_price = current_price or extracted["current_price"]
            list_price = list_price or extracted["list_price"]
            sale_price = sale_price or extracted["sale_price"]
            currency = currency or extracted["currency"]

        if current_price is None:
            fallback_prices = [self._parse_price(match) for match in self._PRICE_RE.findall(payload)]
            fallback_prices = [price for price in fallback_prices if price is not None]
            if fallback_prices:
                current_price = fallback_prices[0]
                if len(fallback_prices) > 1 and fallback_prices[1] > current_price:
                    list_price = list_price or fallback_prices[1]

        if sale_price is None and list_price is not None and current_price is not None and list_price > current_price:
            sale_price = current_price

        return {
            "current_price": current_price,
            "list_price": list_price,
            "sale_price": sale_price,
            "currency": currency,
        }

    def _extract_from_json_ld(self, raw_payload: str) -> dict[str, float | str | None]:
        try:
            data = json.loads(html.unescape(raw_payload.strip()))
        except Exception:
            return {"current_price": None, "list_price": None, "sale_price": None, "currency": None}

        prices: dict[str, float | str | None] = {"current_price": None, "list_price": None, "sale_price": None, "currency": None}
        for node in self._walk_json_ld(data):
            if not isinstance(node, dict):
                continue
            node_type = str(node.get("@type", "")).lower()
            if "offer" in node_type:
                price = self._parse_price(node.get("price"))
                low_price = self._parse_price(node.get("lowPrice"))
                high_price = self._parse_price(node.get("highPrice"))
                candidate = price or low_price or high_price
                if candidate is not None:
                    prices["current_price"] = prices["current_price"] or candidate
                currency = node.get("priceCurrency")
                if currency and prices["currency"] is None:
                    prices["currency"] = str(currency).upper()
                price_spec = node.get("priceSpecification")
                self._merge_price_spec(prices, price_spec)
            elif "product" in node_type and node.get("offers"):
                self._merge_price_spec(prices, node.get("offers"))
        return prices

    def _merge_price_spec(self, prices: dict[str, float | str | None], spec) -> None:
        specs = spec if isinstance(spec, list) else [spec]
        for item in specs:
            if not isinstance(item, dict):
                continue
            price_type = str(item.get("priceType", "")).lower()
            candidate = self._parse_price(item.get("price")) or self._parse_price(item.get("minPrice"))
            if candidate is None:
                continue
            if "listprice" in price_type or "strikethrough" in price_type:
                prices["list_price"] = prices["list_price"] or candidate
            elif "sale" in price_type:
                prices["sale_price"] = prices["sale_price"] or candidate
            else:
                prices["current_price"] = prices["current_price"] or candidate
            currency = item.get("priceCurrency")
            if currency and prices["currency"] is None:
                prices["currency"] = str(currency).upper()

    def _walk_json_ld(self, node):
        if isinstance(node, list):
            for item in node:
                yield from self._walk_json_ld(item)
        elif isinstance(node, dict):
            yield node
            for value in node.values():
                yield from self._walk_json_ld(value)

    @staticmethod
    def _parse_price(value) -> float | None:
        if value is None:
            return None
        if isinstance(value, (int, float)):
            return float(value)
        text = str(value).strip()
        match = re.search(r"(\d{1,5}(?:[.,]\d{2})?)", text)
        if not match:
            return None
        try:
            return float(match.group(1).replace(",", "."))
        except ValueError:
            return None

    @staticmethod
    def _estimate_relevance(title: str, marketplace: str, attributes: dict[str, str]) -> float:
        haystack = title.lower()
        terms = [marketplace.lower(), *title_keywords(title), *[value.lower() for value in attributes.values()]]
        matches = sum(1 for term in unique_strings(terms, limit=8) if term and term in haystack)
        return max(0.4, min(0.95, 0.42 + (matches * 0.08)))

    @staticmethod
    def _keyword_signals(
        marketplace: str,
        query: str,
        evidence: list[ResearchEvidenceResponse],
        attributes: dict[str, str],
    ) -> list[str]:
        listing_terms: list[str] = []
        for item in evidence:
            listing_terms.extend(title_keywords(item.title))
            domain = item.attributes.get("domain")
            if domain:
                listing_terms.append(domain.split(".")[0])
        return unique_strings(
            [marketplace.lower(), *title_keywords(query), *[value.lower() for value in attributes.values()], *listing_terms],
            limit=12,
        )
