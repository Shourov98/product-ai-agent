from __future__ import annotations

import asyncio
from dataclasses import dataclass
from pathlib import Path

from app.agents.amazon_agent import AmazonAgent
from app.agents.attribute_mapper_agent import AttributeMapperAgent
from app.agents.core_agent import CoreAgent
from app.agents.ebay_agent import EbayAgent
from app.agents.etsy_agent import EtsyAgent
from app.agents.image_agent import ImageAgent
from app.agents.seo_agent import SeoAgent
from app.agents.shopify_agent import ShopifyAgent
from app.agents.tiktok_agent import TikTokAgent
from app.agents.vision_agent import VisionAgent
from app.config import get_settings
from app.schemas.response import ProductPipelineResponse
from app.services.image_service import ImagePayload
from app.services.gemini_service import GeminiService
from app.services.s3_service import S3Service
from app.services.openai_service import OpenAIService
from app.services.ollama_service import OllamaService
from app.services.market_research_service import MarketResearchService
from app.services.output_service import OutputService
from app.services.validation_service import ValidationService


@dataclass(slots=True)
class PipelineRunResult:
    run_dir: Path
    response: ProductPipelineResponse


class ProductPipeline:
    def __init__(
        self,
        *,
        vision: VisionAgent | None = None,
        core: CoreAgent | None = None,
        attribute_mapper: AttributeMapperAgent | None = None,
        amazon: AmazonAgent | None = None,
        tiktok: TikTokAgent | None = None,
        ebay: EbayAgent | None = None,
        etsy: EtsyAgent | None = None,
        shopify: ShopifyAgent | None = None,
        seo: SeoAgent | None = None,
    ) -> None:
        settings = get_settings()
        self.output_service = OutputService(
            settings.output_dir,
            s3_service=S3Service(
                region=settings.aws_region,
                bucket_name=settings.aws_s3_bucket,
                access_key_id=settings.aws_access_key_id,
                secret_access_key=settings.aws_secret_access_key,
                prefix=settings.aws_s3_prefix,
            ),
            local_output_enabled=settings.local_output_enabled,
        )
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
        self.gemini_service = GeminiService(
            api_key=settings.gemini_api_key,
            model=settings.gemini_model,
            api_base_url=settings.gemini_api_base_url,
            enabled=settings.gemini_enabled,
        )
        self.vision = vision or VisionAgent(openai_service=self.openai_service)
        self.core = core or CoreAgent(self.ollama_service, self.openai_service)
        self.attribute_mapper = attribute_mapper or AttributeMapperAgent(self.openai_service)
        self.amazon = amazon or AmazonAgent(self.ollama_service, self.openai_service)
        self.tiktok = tiktok or TikTokAgent(self.ollama_service, self.openai_service)
        self.ebay = ebay or EbayAgent(self.ollama_service, self.openai_service)
        self.etsy = etsy or EtsyAgent(self.ollama_service, self.openai_service)
        self.shopify = shopify or ShopifyAgent(self.ollama_service, self.openai_service)
        self.seo = seo or SeoAgent(self.openai_service)
        self.images = ImageAgent(self.openai_service)
        self.research = MarketResearchService(settings)
        self.validation = ValidationService()

    async def run(self, image: ImagePayload, title: str) -> ProductPipelineResponse:
        result = await self.run_with_context(image, title)
        return result.response

    async def run_with_context(self, image: ImagePayload, title: str) -> PipelineRunResult:
        run_dir = self.output_service.create_run_dir()
        vision_data = await self.vision.process(image)
        self.output_service.save_json(run_dir, "vision", vision_data.model_dump())
        core_data = await self.core.process(title, vision_data)
        core_data = await self.attribute_mapper.process(core_data, vision_data)
        self.output_service.save_json(run_dir, "core", core_data.model_dump())
        research_data = await self.research.build_research_bundle(core_data)
        self.output_service.save_json(run_dir, "research", research_data.model_dump())
        seo_data = await self.seo.process(core_data, research_data)
        self.output_service.save_json(run_dir, "seo", seo_data.model_dump())

        amazon_data, tiktok_data, ebay_data, etsy_data, shopify_data = await asyncio.gather(
            self.amazon.process(core_data, research=research_data.amazon, seo=seo_data),
            self.tiktok.process(core_data, research=research_data.tiktok, seo=seo_data),
            self.ebay.process(core_data, research=research_data.ebay, seo=seo_data),
            self.etsy.process(core_data, research=research_data.etsy, seo=seo_data),
            self.shopify.process(core_data, research=research_data.shopify, seo=seo_data),
        )
        self.output_service.save_json(run_dir, "amazon", amazon_data.model_dump())
        self.output_service.save_json(run_dir, "tiktok", tiktok_data.model_dump())
        self.output_service.save_json(run_dir, "ebay", ebay_data.model_dump())
        self.output_service.save_json(run_dir, "etsy", etsy_data.model_dump())
        self.output_service.save_json(run_dir, "shopify", shopify_data.model_dump())
        image_data = await self.images.process(
            image=image,
            core_data=core_data,
            amazon_data=amazon_data,
            ebay_data=ebay_data,
            etsy_data=etsy_data,
            tiktok_data=tiktok_data,
            shopify_data=shopify_data,
            run_dir=run_dir,
            output_service=self.output_service,
        )
        self.output_service.save_json(run_dir, "images", image_data.model_dump())
        validation_data = self.validation.validate_pipeline(
            core=core_data,
            amazon=amazon_data,
            ebay=ebay_data,
            etsy=etsy_data,
            tiktok=tiktok_data,
            shopify=shopify_data,
            images=image_data,
        )
        self.output_service.save_json(run_dir, "validation", validation_data.model_dump())

        response = ProductPipelineResponse(
            core=core_data,
            amazon=amazon_data,
            etsy=etsy_data,
            tiktok=tiktok_data,
            ebay=ebay_data,
            shopify=shopify_data,
            images=image_data,
            intelligence={
                "research": research_data,
                "seo": seo_data,
                "validation": validation_data,
            },
        )
        self.output_service.save_json(run_dir, "final", response.model_dump())
        return PipelineRunResult(run_dir=run_dir, response=response)
