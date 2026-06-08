from __future__ import annotations

import re
from functools import lru_cache
from statistics import mean

from app.core.exceptions import ProviderUnavailableError
from app.providers.kaggle_provider import KaggleProvider
from app.schemas.response import MarketplaceResearchResponse, ResearchEvidenceResponse
from app.utils.product_text import title_keywords, unique_strings


class AmazonDatasetResearchService:
    def __init__(self, *, enabled: bool = True) -> None:
        self.enabled = enabled

    def search(self, *, title: str, product_type: str, category: str, attributes: dict[str, str]) -> MarketplaceResearchResponse | None:
        if not self.enabled:
            return None

        try:
            provider = _get_kaggle_provider()
        except ProviderUnavailableError:
            return None

        frame = provider.products
        title_terms = title_keywords(title)
        attribute_terms = [value.lower() for value in attributes.values() if len(value.strip()) >= 3]
        query_terms = unique_strings([product_type.lower(), category.lower(), *title_terms, *attribute_terms], limit=6)
        if not query_terms:
            return None

        if "title_lower" not in frame.columns:
            frame["title_lower"] = frame["title"].astype(str).str.lower()
        if "category_lower" not in frame.columns:
            frame["category_lower"] = frame["category_name"].astype(str).str.lower()

        safe_terms = [re.escape(term) for term in query_terms[:4] if term]
        if not safe_terms:
            return None
        pattern = "|".join(safe_terms)
        candidates = frame[frame["title_lower"].str.contains(pattern, regex=True, na=False)].copy()
        if category:
            category_lower = category.lower()
            category_matches = candidates[candidates["category_lower"] == category_lower]
            if not category_matches.empty:
                candidates = category_matches

        if candidates.empty:
            return None

        candidates["match_score"] = candidates.apply(
            lambda row: self._score_row(
                title_lower=str(row["title_lower"]),
                category_lower=str(row["category_lower"]),
                terms=query_terms,
                category=category.lower(),
                is_best_seller=bool(row.get("isBestSeller", False)),
                reviews=int(row.get("reviews", 0) or 0),
                bought_in_last_month=int(row.get("boughtInLastMonth", 0) or 0),
            ),
            axis=1,
        )
        top = candidates.sort_values(
            by=["match_score", "boughtInLastMonth", "reviews", "stars"],
            ascending=[False, False, False, False],
        ).head(5)

        listings = [
            ResearchEvidenceResponse(
                source="amazon_dataset",
                title=str(row["title"]),
                price=round(float(row["price"]), 2),
                currency="USD",
                relevance_score=max(0.5, min(0.99, float(row["match_score"]) / 10.0)),
                attributes={
                    "category": str(row.get("category_name", "")),
                    "stars": str(round(float(row.get("stars", 0.0) or 0.0), 2)),
                    "reviews": str(int(row.get("reviews", 0) or 0)),
                    "best_seller": "true" if bool(row.get("isBestSeller", False)) else "false",
                },
                observations=self._build_observations(row),
            )
            for _, row in top.iterrows()
        ]
        if not listings:
            return None

        prices = [listing.price for listing in listings if listing.price is not None]
        keyword_signals = self._keyword_signals_from_listings(query_terms, listings)
        return MarketplaceResearchResponse(
            marketplace="amazon",
            source_mode="dataset",
            search_queries=[title, " ".join(query_terms[:3]), f"{category} {product_type}".strip()],
            keyword_signals=keyword_signals,
            price_min=min(prices) if prices else None,
            price_max=max(prices) if prices else None,
            price_avg=round(mean(prices), 2) if prices else None,
            similar_listings=listings,
        )

    @staticmethod
    def _score_row(
        *,
        title_lower: str,
        category_lower: str,
        terms: list[str],
        category: str,
        is_best_seller: bool,
        reviews: int,
        bought_in_last_month: int,
    ) -> float:
        keyword_matches = sum(1 for term in terms if term and term in title_lower)
        category_bonus = 2.0 if category and category == category_lower else 0.0
        bestseller_bonus = 1.5 if is_best_seller else 0.0
        review_bonus = min(2.0, reviews / 5000)
        velocity_bonus = min(2.0, bought_in_last_month / 2000)
        return keyword_matches + category_bonus + bestseller_bonus + review_bonus + velocity_bonus

    @staticmethod
    def _build_observations(row) -> list[str]:
        observations = [
            f"Category: {row.get('category_name', 'Unknown')}.",
            f"Stars: {round(float(row.get('stars', 0.0) or 0.0), 2)} from {int(row.get('reviews', 0) or 0)} reviews.",
            f"Bought in last month: {int(row.get('boughtInLastMonth', 0) or 0)}.",
        ]
        if bool(row.get("isBestSeller", False)):
            observations.append("This listing is marked as a best seller in the dataset.")
        return observations

    @staticmethod
    def _keyword_signals_from_listings(terms: list[str], listings: list[ResearchEvidenceResponse]) -> list[str]:
        listing_terms: list[str] = []
        for listing in listings:
            listing_terms.extend(title_keywords(listing.title))
        return unique_strings([*terms, *listing_terms], limit=12)


@lru_cache(maxsize=1)
def _get_kaggle_provider() -> KaggleProvider:
    return KaggleProvider()
