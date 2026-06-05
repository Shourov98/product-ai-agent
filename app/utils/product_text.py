from __future__ import annotations

import re
from typing import Iterable


STOPWORDS = {
    "a",
    "an",
    "and",
    "as",
    "at",
    "by",
    "for",
    "from",
    "in",
    "of",
    "on",
    "or",
    "the",
    "to",
    "with",
}

CATEGORY_MAP = {
    "running shoes": "Athletic Footwear",
    "hoodie": "Apparel & Sweatshirts",
    "office chair": "Office Furniture",
    "water bottle": "Drinkware & Hydration",
    "table lamp": "Home Lighting",
    "travel bag": "Travel Bags & Luggage",
    "watch": "Fashion Accessories",
}

PRODUCT_TYPE_ALIASES = {
    "shoe": "running shoes",
    "shoes": "running shoes",
    "sneaker": "running shoes",
    "sneakers": "running shoes",
    "hoodie": "hoodie",
    "chair": "office chair",
    "bottle": "water bottle",
    "lamp": "table lamp",
    "bag": "travel bag",
    "watch": "watch",
}


def normalize_title(title: str) -> str:
    cleaned = re.sub(r"[_\-]+", " ", title.strip())
    cleaned = re.sub(r"\s+", " ", cleaned)

    words = []
    for token in cleaned.split(" "):
        if token.isupper() and len(token) > 1:
            words.append(token)
            continue
        if re.search(r"\d", token):
            words.append(token.upper() if token.isalpha() else token)
            continue
        words.append(token.capitalize())
    return " ".join(words)


def infer_product_type(*texts: str) -> str:
    haystack = " ".join(texts).lower()
    for keyword, product_type in PRODUCT_TYPE_ALIASES.items():
        if keyword in haystack:
            return product_type
    return "general product"


def build_category(product_type: str) -> str:
    return CATEGORY_MAP.get(product_type, "General Merchandise")


def title_keywords(title: str) -> list[str]:
    parts = re.findall(r"[a-zA-Z0-9]+", title.lower())
    deduped = []
    seen = set()
    for part in parts:
        if len(part) < 3 or part in STOPWORDS or part in seen:
            continue
        seen.add(part)
        deduped.append(part)
    return deduped


def unique_strings(items: Iterable[str], *, limit: int | None = None) -> list[str]:
    result = []
    seen = set()
    for item in items:
        normalized = item.strip()
        if not normalized:
            continue
        key = normalized.lower()
        if key in seen:
            continue
        seen.add(key)
        result.append(normalized)
        if limit is not None and len(result) >= limit:
            break
    return result


def sentence_case_summary(parts: list[str]) -> str:
    filtered = [part.strip() for part in parts if part.strip()]
    if not filtered:
        return ""
    text = ". ".join(filtered)
    if not text.endswith("."):
        text += "."
    return text
