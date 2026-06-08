from __future__ import annotations

import logging
from functools import lru_cache

from app.providers.amazon_provider import AmazonSPAPIProvider
from app.providers.base import PriceDataProvider
from app.providers.kaggle_provider import KaggleProvider


logger = logging.getLogger(__name__)


@lru_cache(maxsize=1)
def get_provider() -> PriceDataProvider:
    if AmazonSPAPIProvider.is_configured():
        return AmazonSPAPIProvider()
    logger.info("Amazon SP-API not configured. Using Kaggle dataset.")
    return KaggleProvider()


def reset_provider_cache() -> None:
    get_provider.cache_clear()
