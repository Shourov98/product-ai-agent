from __future__ import annotations

import asyncio

from app.agents.amazon_agent import AmazonAgent
from app.agents.core_agent import CoreAgent
from app.agents.ebay_agent import EbayAgent
from app.agents.image_agent import ImageAgent
from app.agents.tiktok_agent import TikTokAgent
from app.agents.vision_agent import VisionAgent
from app.config import get_settings
from app.schemas.response import ProductPipelineResponse
from app.services.image_service import ImagePayload
from app.services.openai_service import OpenAIService
from app.services.ollama_service import OllamaService
from app.services.output_service import OutputService


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
        settings = get_settings()
        self.output_service = OutputService(settings.output_dir)
        self.ollama_service = OllamaService(
            base_url=settings.ollama_base_url,
            model=settings.ollama_model,
            enabled=settings.ollama_enabled,
        )
        self.openai_service = OpenAIService(
            api_key=settings.openai_api_key,
            model=settings.openai_model,
            image_model=settings.openai_image_model,
            enabled=settings.openai_enabled,
        )
        self.vision = vision or VisionAgent()
        self.core = core or CoreAgent(self.ollama_service, self.openai_service)
        self.amazon = amazon or AmazonAgent(self.ollama_service, self.openai_service)
        self.tiktok = tiktok or TikTokAgent(self.ollama_service, self.openai_service)
        self.ebay = ebay or EbayAgent(self.ollama_service, self.openai_service)
        self.images = ImageAgent(self.openai_service)

    async def run(self, image: ImagePayload, title: str) -> ProductPipelineResponse:
        run_dir = self.output_service.create_run_dir()
        vision_data = await self.vision.process(image)
        self.output_service.save_json(run_dir, "vision", vision_data.model_dump())
        core_data = await self.core.process(title, vision_data)
        self.output_service.save_json(run_dir, "core", core_data.model_dump())

        amazon_data, tiktok_data, ebay_data = await asyncio.gather(
            self.amazon.process(core_data),
            self.tiktok.process(core_data),
            self.ebay.process(core_data),
        )
        self.output_service.save_json(run_dir, "amazon", amazon_data.model_dump())
        self.output_service.save_json(run_dir, "tiktok", tiktok_data.model_dump())
        self.output_service.save_json(run_dir, "ebay", ebay_data.model_dump())
        image_data = await self.images.process(
            image=image,
            core_data=core_data,
            amazon_data=amazon_data,
            ebay_data=ebay_data,
            tiktok_data=tiktok_data,
            run_dir=run_dir,
            output_service=self.output_service,
        )
        self.output_service.save_json(run_dir, "images", image_data.model_dump())

        response = ProductPipelineResponse(
            core=core_data,
            amazon=amazon_data,
            tiktok=tiktok_data,
            ebay=ebay_data,
            images=image_data,
        )
        self.output_service.save_json(run_dir, "final", response.model_dump())
        return response
