from __future__ import annotations

from app.config import get_settings
from app.services.image_service import ImagePayload
from app.services.gemini_service import GeminiService
from app.services.openai_service import OpenAIService
from app.services.market_research_service import MarketResearchService
from app.services.output_service import OutputService
from app.services.validation_service import ValidationService
from app.services.s3_service import S3Service
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
from app.schemas.response import ProductPipelineResponse

from app.orchestrator.agent_graph import PipelineRunResult, ProductAgentGraph


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
        self.core = core or CoreAgent(self.openai_service, self.gemini_service)
        self.attribute_mapper = attribute_mapper or AttributeMapperAgent(self.openai_service, self.gemini_service)
        self.amazon = amazon or AmazonAgent(self.openai_service, self.gemini_service)
        self.tiktok = tiktok or TikTokAgent(self.openai_service, self.gemini_service)
        self.ebay = ebay or EbayAgent(self.openai_service, self.gemini_service)
        self.etsy = etsy or EtsyAgent(self.openai_service, self.gemini_service)
        self.shopify = shopify or ShopifyAgent(self.openai_service, self.gemini_service)
        self.seo = seo or SeoAgent(self.openai_service, self.gemini_service)
        self.images = ImageAgent(self.openai_service)
        self.research = MarketResearchService(settings)
        self.validation = ValidationService()
        self.graph = ProductAgentGraph(
            vision=self.vision,
            core=self.core,
            attribute_mapper=self.attribute_mapper,
            amazon=self.amazon,
            tiktok=self.tiktok,
            ebay=self.ebay,
            etsy=self.etsy,
            shopify=self.shopify,
            seo=self.seo,
            images=self.images,
            research=self.research,
            validation=self.validation,
            output_service=self.output_service,
        )

    async def run(self, image: ImagePayload, title: str) -> ProductPipelineResponse:
        result = await self.run_with_context(image, title)
        return result.response

    async def run_with_context(self, image: ImagePayload, title: str) -> PipelineRunResult:
        return await self.graph.execute(image, title)
