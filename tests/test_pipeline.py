from __future__ import annotations

import asyncio

from app.orchestrator.pipeline import ProductPipeline
from app.services.image_service import ImagePayload


def test_pipeline_returns_all_marketplace_outputs() -> None:
    pipeline = ProductPipeline()
    payload = ImagePayload(
        filename="red-leather-bag.jpeg",
        content_type="image/jpeg",
        data=bytes([120, 130, 150, 140]) * 40,
    )

    result = asyncio.run(pipeline.run(payload, "Urban Carry Bag"))

    assert result.core.category == "General Merchandise"
    assert result.amazon.title
    assert result.tiktok.hashtags
    assert result.ebay.condition in {"New", "New other"}
