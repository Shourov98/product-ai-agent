from __future__ import annotations

import asyncio

from app.agents.amazon_agent import AmazonAgent
from app.agents.core_agent import CoreAgent
from app.agents.ebay_agent import EbayAgent
from app.agents.tiktok_agent import TikTokAgent
from app.agents.vision_agent import VisionAgent
from app.schemas.response import ProductPipelineResponse
from app.services.image_service import ImagePayload


class ProductPipeline:
    def __init__(
        self,
        *,
        vision: VisionAgent | None = None,
        core: CoreAgent | None = None,
        amazon: AmazonAgent | None = None,
        tiktok: TikTokAgent | None = None,
        ebay: EbayAgent | None = None,
    ) -> None:
        self.vision = vision or VisionAgent()
        self.core = core or CoreAgent()
        self.amazon = amazon or AmazonAgent()
        self.tiktok = tiktok or TikTokAgent()
        self.ebay = ebay or EbayAgent()

    async def run(self, image: ImagePayload, title: str) -> ProductPipelineResponse:
        vision_data = await self.vision.process(image)
        core_data = await self.core.process(title, vision_data)

        amazon_data, tiktok_data, ebay_data = await asyncio.gather(
            self.amazon.process(core_data),
            self.tiktok.process(core_data),
            self.ebay.process(core_data),
        )

        return ProductPipelineResponse(
            core=core_data,
            amazon=amazon_data,
            tiktok=tiktok_data,
            ebay=ebay_data,
        )
