from __future__ import annotations

from dataclasses import dataclass
from datetime import UTC, datetime
from pathlib import Path
from typing import Any, TypedDict

try:
    from langgraph.graph import END, START, StateGraph
except ModuleNotFoundError:  # pragma: no cover - exercised in environments without langgraph
    END = START = None
    StateGraph = None

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
from app.schemas.response import (
    AmazonResponse,
    CoreProductResponse,
    EbayResponse,
    EtsyResponse,
    GeneratedImagesResponse,
    IntelligenceLayerResponse,
    MarketResearchBundleResponse,
    PipelineValidationResponse,
    ProductPipelineResponse,
    SeoInsightsResponse,
    ShopifyResponse,
    TikTokResponse,
)
from app.services.image_service import ImagePayload
from app.services.market_research_service import MarketResearchService
from app.services.output_service import OutputService
from app.services.validation_service import ValidationService


@dataclass(slots=True)
class PipelineRunResult:
    run_dir: Path
    response: ProductPipelineResponse


class ProductAgentState(TypedDict, total=False):
    image: ImagePayload
    title: str
    run_dir: Path
    vision_data: Any
    core_data: CoreProductResponse
    research_data: MarketResearchBundleResponse
    seo_data: SeoInsightsResponse
    amazon_data: AmazonResponse
    tiktok_data: TikTokResponse
    ebay_data: EbayResponse
    etsy_data: EtsyResponse
    shopify_data: ShopifyResponse
    image_data: GeneratedImagesResponse
    validation_data: PipelineValidationResponse


class ProductAgentGraph:
    def __init__(
        self,
        *,
        vision: VisionAgent,
        core: CoreAgent,
        attribute_mapper: AttributeMapperAgent,
        amazon: AmazonAgent,
        tiktok: TikTokAgent,
        ebay: EbayAgent,
        etsy: EtsyAgent,
        shopify: ShopifyAgent,
        seo: SeoAgent,
        images: ImageAgent,
        research: MarketResearchService,
        validation: ValidationService,
        output_service: OutputService,
    ) -> None:
        self.vision = vision
        self.core = core
        self.attribute_mapper = attribute_mapper
        self.amazon = amazon
        self.tiktok = tiktok
        self.ebay = ebay
        self.etsy = etsy
        self.shopify = shopify
        self.seo = seo
        self.images = images
        self.research = research
        self.validation = validation
        self.output_service = output_service
        self.app = self._build_graph()

    async def execute(self, image: ImagePayload, title: str) -> PipelineRunResult:
        run_dir = self.output_service.create_run_dir()
        state: ProductAgentState = {
            "image": image,
            "title": title,
            "run_dir": run_dir,
        }
        result = await self.app.ainvoke(state)
        return self._finalize(result)

    def _build_graph(self):
        if StateGraph is None:
            return _SequentialProductAgentApp(self)
        graph = StateGraph(ProductAgentState)
        graph.add_node("vision", self._vision_node)
        graph.add_node("core", self._core_node)
        graph.add_node("research", self._research_node)
        graph.add_node("seo", self._seo_node)
        graph.add_node("marketplace", self._marketplace_node)
        graph.add_node("images", self._image_node)
        graph.add_node("validation", self._validation_node)

        graph.add_edge(START, "vision")
        graph.add_edge("vision", "core")
        graph.add_edge("core", "research")
        graph.add_edge("research", "seo")
        graph.add_edge("seo", "marketplace")
        graph.add_edge("marketplace", "images")
        graph.add_edge("images", "validation")
        graph.add_edge("validation", END)
        return graph.compile(name="product_agent_graph")

    async def _vision_node(self, state: ProductAgentState) -> dict[str, Any]:
        image = self._require(state.get("image"), "image")
        vision_data = await self.vision.process(image)
        self.output_service.save_json(state["run_dir"], "vision", vision_data.model_dump())
        return {"vision_data": vision_data}

    async def _core_node(self, state: ProductAgentState) -> dict[str, Any]:
        title = self._require(state.get("title"), "title")
        vision_data = self._require(state.get("vision_data"), "vision_data")
        core_data = await self.core.process(title, vision_data)
        core_data = await self.attribute_mapper.process(core_data, vision_data)
        self.output_service.save_json(state["run_dir"], "core", core_data.model_dump())
        return {"core_data": core_data}

    async def _research_node(self, state: ProductAgentState) -> dict[str, Any]:
        core_data = self._require(state.get("core_data"), "core_data")
        research_data = await self.research.build_research_bundle(core_data)
        self.output_service.save_json(state["run_dir"], "research", research_data.model_dump())
        return {"research_data": research_data}

    async def _seo_node(self, state: ProductAgentState) -> dict[str, Any]:
        core_data = self._require(state.get("core_data"), "core_data")
        research_data = self._require(state.get("research_data"), "research_data")
        seo_data = await self.seo.process(core_data, research_data)
        self.output_service.save_json(state["run_dir"], "seo", seo_data.model_dump())
        return {"seo_data": seo_data}

    async def _marketplace_node(self, state: ProductAgentState) -> dict[str, Any]:
        core_data = self._require(state.get("core_data"), "core_data")
        research_data = self._require(state.get("research_data"), "research_data")
        seo_data = self._require(state.get("seo_data"), "seo_data")
        amazon_data, tiktok_data, ebay_data, etsy_data, shopify_data = await self._run_marketplace_agents(
            core_data=core_data,
            research_data=research_data,
            seo_data=seo_data,
        )
        self.output_service.save_json(state["run_dir"], "amazon", amazon_data.model_dump())
        self.output_service.save_json(state["run_dir"], "tiktok", tiktok_data.model_dump())
        self.output_service.save_json(state["run_dir"], "ebay", ebay_data.model_dump())
        self.output_service.save_json(state["run_dir"], "etsy", etsy_data.model_dump())
        self.output_service.save_json(state["run_dir"], "shopify", shopify_data.model_dump())
        return {
            "amazon_data": amazon_data,
            "tiktok_data": tiktok_data,
            "ebay_data": ebay_data,
            "etsy_data": etsy_data,
            "shopify_data": shopify_data,
        }

    async def _image_node(self, state: ProductAgentState) -> dict[str, Any]:
        image_data = await self.images.process(
            image=self._require(state.get("image"), "image"),
            core_data=self._require(state.get("core_data"), "core_data"),
            amazon_data=self._require(state.get("amazon_data"), "amazon_data"),
            ebay_data=self._require(state.get("ebay_data"), "ebay_data"),
            etsy_data=self._require(state.get("etsy_data"), "etsy_data"),
            tiktok_data=self._require(state.get("tiktok_data"), "tiktok_data"),
            shopify_data=self._require(state.get("shopify_data"), "shopify_data"),
            run_dir=state["run_dir"],
            output_service=self.output_service,
        )
        self.output_service.save_json(state["run_dir"], "images", image_data.model_dump())
        return {"image_data": image_data}

    async def _validation_node(self, state: ProductAgentState) -> dict[str, Any]:
        validation_data = self.validation.validate_pipeline(
            core=self._require(state.get("core_data"), "core_data"),
            amazon=self._require(state.get("amazon_data"), "amazon_data"),
            ebay=self._require(state.get("ebay_data"), "ebay_data"),
            etsy=self._require(state.get("etsy_data"), "etsy_data"),
            tiktok=self._require(state.get("tiktok_data"), "tiktok_data"),
            shopify=self._require(state.get("shopify_data"), "shopify_data"),
            images=self._require(state.get("image_data"), "image_data"),
        )
        self.output_service.save_json(state["run_dir"], "validation", validation_data.model_dump())
        return {"validation_data": validation_data}

    async def _run_marketplace_agents(
        self,
        *,
        core_data: CoreProductResponse,
        research_data: MarketResearchBundleResponse,
        seo_data: SeoInsightsResponse,
    ) -> tuple[AmazonResponse, TikTokResponse, EbayResponse, EtsyResponse, ShopifyResponse]:
        return await self._gather_marketplace_outputs(core_data, research_data, seo_data)

    async def _gather_marketplace_outputs(
        self,
        core_data: CoreProductResponse,
        research_data: MarketResearchBundleResponse,
        seo_data: SeoInsightsResponse,
    ) -> tuple[AmazonResponse, TikTokResponse, EbayResponse, EtsyResponse, ShopifyResponse]:
        import asyncio

        return await asyncio.gather(
            self.amazon.process(core_data, research=research_data.amazon, seo=seo_data),
            self.tiktok.process(core_data, research=research_data.tiktok, seo=seo_data),
            self.ebay.process(core_data, research=research_data.ebay, seo=seo_data),
            self.etsy.process(core_data, research=research_data.etsy, seo=seo_data),
            self.shopify.process(core_data, research=research_data.shopify, seo=seo_data),
        )

    def _finalize(self, state: ProductAgentState) -> PipelineRunResult:
        response = ProductPipelineResponse(
            core=self._require(state.get("core_data"), "core_data"),
            amazon=self._require(state.get("amazon_data"), "amazon_data"),
            etsy=self._require(state.get("etsy_data"), "etsy_data"),
            tiktok=self._require(state.get("tiktok_data"), "tiktok_data"),
            ebay=self._require(state.get("ebay_data"), "ebay_data"),
            shopify=self._require(state.get("shopify_data"), "shopify_data"),
            images=self._require(state.get("image_data"), "image_data"),
            intelligence=IntelligenceLayerResponse(
                research=self._require(state.get("research_data"), "research_data"),
                seo=self._require(state.get("seo_data"), "seo_data"),
                validation=self._require(state.get("validation_data"), "validation_data"),
            ),
        )
        self.output_service.save_json(state["run_dir"], "final", response.model_dump())
        return PipelineRunResult(run_dir=state["run_dir"], response=response)

    @staticmethod
    def _require(value: object | None, node: str):
        if value is None:
            raise RuntimeError(f"Graph node '{node}' did not produce a value.")
        return value


class _SequentialProductAgentApp:
    """Fallback runner used when langgraph is not installed."""

    def __init__(self, graph: ProductAgentGraph) -> None:
        self.graph = graph

    async def ainvoke(self, state: ProductAgentState) -> ProductAgentState:
        next_state = dict(state)
        next_state.update(await self.graph._vision_node(next_state))
        next_state.update(await self.graph._core_node(next_state))
        next_state.update(await self.graph._research_node(next_state))
        next_state.update(await self.graph._seo_node(next_state))
        next_state.update(await self.graph._marketplace_node(next_state))
        next_state.update(await self.graph._image_node(next_state))
        next_state.update(await self.graph._validation_node(next_state))
        return next_state
