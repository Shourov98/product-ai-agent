from __future__ import annotations

import asyncio
from io import BytesIO

from app.agents.amazon_agent import AmazonAgent
from app.agents.ebay_agent import EbayAgent
from app.agents.core_agent import CoreAgent
from app.agents.vision_agent import VisionAgent
from app.schemas.response import EbayResponse
from app.services.image_service import ImagePayload
from PIL import Image


def _build_test_png(color: tuple[int, int, int]) -> bytes:
    image = Image.new("RGB", (64, 64), color)
    buffer = BytesIO()
    image.save(buffer, format="PNG")
    return buffer.getvalue()


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


def test_ebay_agent_coerces_list_specifics_to_strings() -> None:
    agent = EbayAgent()
    fallback = EbayResponse(
        title="Fallback Title",
        item_specifics={"Brand": "Example"},
        condition="New",
        listing_notes="Fallback notes",
    )

    result = agent._from_data(
        {
            "title": "Custom Title",
            "item_specifics": {
                "Features": ["Ultra-High-Speed SSD", "Backward Compatibility"],
                "Condition": "New",
            },
            "condition": "New",
            "listing_notes": "Optimized notes",
        },
        fallback,
    )

    assert result.item_specifics["Features"] == "Ultra-High-Speed SSD, Backward Compatibility"
    assert result.item_specifics["Condition"] == "New"


def test_vision_agent_can_identify_compound_blue_family() -> None:
    agent = VisionAgent()
    payload = ImagePayload(
        filename="joan-tran-unsplash.jpg",
        content_type="image/png",
        data=_build_test_png((28, 52, 92)),
    )

    result = asyncio.run(agent.process(payload))
    colors = [attribute.value for attribute in result.attributes if attribute.name == "color"]

    assert any(color in {"navy blue", "slate blue", "blue"} for color in colors)


def test_vision_agent_can_identify_compound_green_family() -> None:
    agent = VisionAgent()
    payload = ImagePayload(
        filename="forest-product.jpg",
        content_type="image/png",
        data=_build_test_png((47, 92, 63)),
    )

    result = asyncio.run(agent.process(payload))
    colors = [attribute.value for attribute in result.attributes if attribute.name == "color"]

    assert any(color in {"forest green", "sage green", "green"} for color in colors)
