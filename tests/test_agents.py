from __future__ import annotations

import asyncio

from app.agents.amazon_agent import AmazonAgent
from app.agents.core_agent import CoreAgent
from app.agents.vision_agent import VisionAgent
from app.services.image_service import ImagePayload


def test_vision_agent_extracts_expected_metadata() -> None:
    agent = VisionAgent()
    payload = ImagePayload(
        filename="black-mesh-running-shoe.jpg",
        content_type="image/jpeg",
        data=bytes([12, 15, 18, 25]) * 50,
    )

    result = asyncio.run(agent.process(payload))

    assert result.product_type == "running shoes"
    assert result.image_analysis.background_removed is True
    assert any(attribute.value == "black" for attribute in result.attributes)
    assert any(attribute.value == "mesh" for attribute in result.attributes)


def test_amazon_agent_generates_listing_fields() -> None:
    core_agent = CoreAgent()
    vision_agent = VisionAgent()
    payload = ImagePayload(
        filename="blue-cotton-hoodie.png",
        content_type="image/png",
        data=bytes([180, 190, 210, 200]) * 25,
    )

    vision = asyncio.run(vision_agent.process(payload))
    core = asyncio.run(core_agent.process("Ultra Comfort Hoodie", vision))
    listing = asyncio.run(AmazonAgent().process(core))

    assert listing.title
    assert isinstance(listing.bullet_points, list)
    assert "color" in listing.structured_attributes
