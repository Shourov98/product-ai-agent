from __future__ import annotations

import asyncio
import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
from statistics import mean, median
from urllib.request import urlopen
from uuid import uuid4

from fastapi import HTTPException, status

from app.auth import AuthenticatedUser
from app.agents.amazon_agent import AmazonAgent
from app.agents.ebay_agent import EbayAgent
from app.agents.etsy_agent import EtsyAgent
from app.agents.image_agent import ImageAgent
from app.agents.product_optimization_agent import ProductOptimizationAgent
from app.agents.shopify_agent import ShopifyAgent
from app.agents.tiktok_agent import TikTokAgent
from app.config import get_settings
from app.orchestrator.pipeline import ProductPipeline
from app.schemas.request import (
    ProductOptimizationRequest,
    ProductUpdateRequest,
    PublishTargetAnalysisRequest,
    VariantCreateRequest,
)
from app.schemas.response import (
    AmazonResponse,
    CoreProductResponse,
    EbayResponse,
    EtsyResponse,
    GeneratedImagesResponse,
    ImageValidationResponse,
    ImageVariantResponse,
    IntelligenceLayerResponse,
    MarketplaceLiteral,
    MarketplaceResearchResponse,
    MarketResearchBundleResponse,
    MarketplaceVariantsResponse,
    PaginatedProductListResponse,
    PublishTargetAnalysisJobResponse,
    PublishTargetAnalysisResponse,
    PipelineValidationResponse,
    ProductListItemResponse,
    ProductPipelineResponse,
    ProductRecordResponse,
    ProductVariantResponse,
    SectionValidationResponse,
    SeoInsightsResponse,
    SuggestedPriceRangeResponse,
    ShopifyResponse,
    TikTokResponse,
)
from app.services.cloudinary_service import CloudinaryService
from app.services.image_service import ImagePayload
from app.services.publish_target_job_store import PublishTargetJobStore
from app.services.mongo_product_store import MongoProductStore
from app.services.output_service import OutputService
from app.services.product_store import ProductStore
from app.utils.prompts import PromptRegistry
from app.utils.product_text import title_keywords, unique_strings


class ProductService:
    _PUBLISH_TARGET_SCHEMA = {
        "type": "object",
        "additionalProperties": False,
        "required": [
            "marketplace",
            "vendor",
            "default_sku",
            "default_price",
            "publish_description",
            "suggested_price_range",
            "market_signal",
            "analysis_summary",
        ],
        "properties": {
            "marketplace": {"type": "string"},
            "vendor": {"type": "string"},
            "default_sku": {"type": "string"},
            "default_price": {"type": "string"},
            "publish_description": {"type": "string"},
            "suggested_price_range": {
                "type": "object",
                "nullable": True,
                "additionalProperties": False,
                "required": ["minimum", "maximum", "recommended", "currency", "source"],
                "properties": {
                    "minimum": {"type": "number"},
                    "maximum": {"type": "number"},
                    "recommended": {"type": "number"},
                    "currency": {"type": "string"},
                    "source": {"type": "string"},
                },
            },
            "market_signal": {"type": "string"},
            "analysis_summary": {"type": "string"},
        },
    }

    _shared_settings = None
    _shared_pipeline = None
    _shared_store = None
    _shared_output_service = None
    _shared_image_agent = None
    _shared_amazon_agent = None
    _shared_ebay_agent = None
    _shared_etsy_agent = None
    _shared_tiktok_agent = None
    _shared_shopify_agent = None
    _shared_optimization_agent = None

    def __init__(self) -> None:
        settings = self._shared_settings or get_settings()
        self.__class__._shared_settings = settings
        self.settings = settings
        self.pipeline = self._shared_pipeline or ProductPipeline()
        self.__class__._shared_pipeline = self.pipeline
        self.store = self._shared_store or (
            MongoProductStore(
                mongodb_uri=settings.mongodb_uri,
                db_name=settings.mongodb_db_name,
                products_collection=settings.mongodb_products_collection,
                imports_collection=settings.mongodb_imports_collection,
            )
            if settings.mongodb_uri and settings.mongodb_enabled
            else ProductStore(settings.output_dir)
        )
        self.__class__._shared_store = self.store
        self.output_service = self._shared_output_service or OutputService(
            settings.output_dir,
            cloudinary_service=CloudinaryService(
                cloud_name=settings.cloudinary_cloud_name,
                api_key=settings.cloudinary_api_key,
                api_secret=settings.cloudinary_api_secret,
                folder=settings.cloudinary_folder,
                secure=settings.cloudinary_secure,
            ),
            local_output_enabled=settings.local_output_enabled,
        )
        self.__class__._shared_output_service = self.output_service
        self.image_agent = self._shared_image_agent or ImageAgent(self.pipeline.openai_service)
        self.__class__._shared_image_agent = self.image_agent
        self.amazon_agent = self._shared_amazon_agent or AmazonAgent(self.pipeline.ollama_service, self.pipeline.openai_service)
        self.__class__._shared_amazon_agent = self.amazon_agent
        self.ebay_agent = self._shared_ebay_agent or EbayAgent(self.pipeline.ollama_service, self.pipeline.openai_service)
        self.__class__._shared_ebay_agent = self.ebay_agent
        self.etsy_agent = self._shared_etsy_agent or EtsyAgent(self.pipeline.ollama_service, self.pipeline.openai_service)
        self.__class__._shared_etsy_agent = self.etsy_agent
        self.tiktok_agent = self._shared_tiktok_agent or TikTokAgent(self.pipeline.ollama_service, self.pipeline.openai_service)
        self.__class__._shared_tiktok_agent = self.tiktok_agent
        self.shopify_agent = self._shared_shopify_agent or ShopifyAgent(self.pipeline.ollama_service, self.pipeline.openai_service)
        self.__class__._shared_shopify_agent = self.shopify_agent
        self.optimization_agent = self._shared_optimization_agent or ProductOptimizationAgent(self.pipeline.openai_service)
        self.__class__._shared_optimization_agent = self.optimization_agent

    @classmethod
    def reset_shared_state(cls) -> None:
        store = cls._shared_store
        close = getattr(store, "close", None)
        if callable(close):
            close()

        cls._shared_settings = None
        cls._shared_pipeline = None
        cls._shared_store = None
        cls._shared_output_service = None
        cls._shared_image_agent = None
        cls._shared_amazon_agent = None
        cls._shared_ebay_agent = None
        cls._shared_etsy_agent = None
        cls._shared_tiktok_agent = None
        cls._shared_shopify_agent = None
        cls._shared_optimization_agent = None

    async def generate_text_only(
        self,
        image: ImagePayload,
        title: str,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        result = await self._build_generated_product(
            image=image,
            title=title,
            image_marketplaces=[],
        )
        record = ProductRecordResponse(
            id=self._new_id(),
            status="draft",
            created_at=self._timestamp(),
            updated_at=self._timestamp(),
            run_id=result["run_id"],
            product=result["product"],
            variants=MarketplaceVariantsResponse(),
        )
        self.store.save(record, user_id=current_user.user_id if current_user is not None else None)
        return record

    async def generate_marketplace_draft(
        self,
        image: ImagePayload,
        title: str,
        marketplace: MarketplaceLiteral,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        result = await self._build_generated_product(
            image=image,
            title=title,
            image_marketplaces=[marketplace],
        )
        record = ProductRecordResponse(
            id=self._new_id(),
            status="draft",
            created_at=self._timestamp(),
            updated_at=self._timestamp(),
            run_id=result["run_id"],
            product=result["product"],
            variants=MarketplaceVariantsResponse(),
        )
        self.store.save(record, user_id=current_user.user_id if current_user is not None else None)
        return record

    def create_product_from_pipeline(
        self,
        *,
        product: ProductPipelineResponse,
        current_user: AuthenticatedUser | None = None,
        status: str = "draft",
    ) -> ProductRecordResponse:
        record = ProductRecordResponse(
            id=self._new_id(),
            status=status,  # type: ignore[arg-type]
            created_at=self._timestamp(),
            updated_at=self._timestamp(),
            run_id=self.output_service.create_run_dir().name,
            product=product,
            variants=MarketplaceVariantsResponse(),
        )
        self.store.save(record, user_id=current_user.user_id if current_user is not None else None)
        return record

    def create_imported_draft(
        self,
        *,
        title: str,
        sku: str,
        brand: str,
        category: str,
        product_type: str,
        description: str,
        price: str,
        quantity: str,
        color: str,
        size: str,
        material: str,
        image_url: str,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        product = self.build_imported_product(
            title=title,
            sku=sku,
            brand=brand,
            category=category,
            product_type=product_type,
            description=description,
            price=price,
            quantity=quantity,
            color=color,
            size=size,
            material=material,
            image_url=image_url,
        )
        run_id = self.output_service.create_run_dir().name
        record = ProductRecordResponse(
            id=self._new_id(),
            status="draft",
            created_at=self._timestamp(),
            updated_at=self._timestamp(),
            run_id=run_id,
            product=product,
            variants=MarketplaceVariantsResponse(),
        )
        self.store.save(record, user_id=current_user.user_id if current_user is not None else None)
        return record

    def build_imported_product(
        self,
        *,
        title: str,
        sku: str,
        brand: str,
        category: str,
        product_type: str,
        description: str,
        price: str,
        quantity: str,
        color: str,
        size: str,
        material: str,
        image_url: str,
    ) -> ProductPipelineResponse:
        normalized_title = title or "Untitled Imported Product"
        attributes = {
            key: value
            for key, value in {
                "sku": sku,
                "brand": brand,
                "price": price,
                "quantity": quantity,
                "color": color,
                "size": size,
                "material": material,
            }.items()
            if value.strip()
        }
        core = CoreProductResponse(
            normalized_title=normalized_title,
            category=category or "General Merchandise",
            product_type=product_type or "general product",
            product_summary=description or "",
            features=[],
            attributes=attributes,
            source_title=title or normalized_title,
            vision_confidence=0.0,
        )
        amazon = AmazonResponse(
            title=normalized_title,
            bullet_points=[],
            description=description or "",
            backend_search_terms=[],
            structured_attributes={key.title(): value for key, value in attributes.items()},
        )
        ebay = EbayResponse(
            title=normalized_title[:80],
            item_specifics={key.title(): value for key, value in attributes.items()},
            condition="New",
            listing_notes=description or "",
        )
        etsy = EtsyResponse(
            title=normalized_title[:140],
            description=description or "",
            tags=[],
            materials=[material] if material else [],
            occasion="",
            seo_keywords=[],
        )
        tiktok = TikTokResponse(
            title=normalized_title,
            social_description=description or "",
            hashtags=[],
        )
        shopify = ShopifyResponse(
            title=normalized_title,
            body_html=f"<p>{description}</p>" if description else "",
            tags=[],
            product_type=product_type or "",
            seo_title=normalized_title[:70],
            seo_description=(description or "")[:180],
            category=category or "",
            metafields={
                key: value
                for key, value in {
                    "color": color.title() if color else "",
                    "footwear_material": material.title() if material else "",
                }.items()
                if value
            },
        )
        return ProductPipelineResponse(
            core=core,
            amazon=amazon,
            ebay=ebay,
            etsy=etsy,
            tiktok=tiktok,
            shopify=shopify,
            images=self._empty_generated_images(image_url=image_url),
            intelligence=self._empty_intelligence(),
        )

    def list_products(self, current_user: AuthenticatedUser | None = None) -> list[ProductListItemResponse]:
        return self.store.list(user_id=current_user.user_id if current_user is not None else None)

    def list_products_paginated(
        self,
        *,
        page: int,
        page_size: int,
        current_user: AuthenticatedUser | None = None,
    ) -> PaginatedProductListResponse:
        return self.store.list_paginated(
            page=page,
            page_size=page_size,
            user_id=current_user.user_id if current_user is not None else None,
        )

    def get_product(self, product_id: str, current_user: AuthenticatedUser | None = None) -> ProductRecordResponse:
        record = self.store.get(product_id, user_id=current_user.user_id if current_user is not None else None)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found.")
        return record

    def update_product(
        self,
        product_id: str,
        payload: ProductUpdateRequest,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        record = self.get_product(product_id, current_user=current_user)
        product = record.product

        if payload.core is not None:
            product = product.model_copy(
                update={
                    "core": product.core.model_copy(update=payload.core.model_dump(exclude_unset=True))
                }
            )
        if payload.amazon is not None:
            product = product.model_copy(
                update={
                    "amazon": product.amazon.model_copy(update=payload.amazon.model_dump(exclude_unset=True))
                }
            )
        if payload.tiktok is not None:
            product = product.model_copy(
                update={
                    "tiktok": product.tiktok.model_copy(update=payload.tiktok.model_dump(exclude_unset=True))
                }
            )
        if payload.ebay is not None:
            product = product.model_copy(
                update={
                    "ebay": product.ebay.model_copy(update=payload.ebay.model_dump(exclude_unset=True))
                }
            )
        if payload.etsy is not None:
            product = product.model_copy(
                update={
                    "etsy": product.etsy.model_copy(update=payload.etsy.model_dump(exclude_unset=True))
                }
            )
        if payload.shopify is not None:
            product = product.model_copy(
                update={
                    "shopify": product.shopify.model_copy(update=payload.shopify.model_dump(exclude_unset=True))
                }
            )

        updated = record.model_copy(
            update={
                "product": product,
                "updated_at": self._timestamp(),
            }
        )
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    async def upload_source_image(
        self,
        product_id: str,
        image: ImagePayload,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        record = self.get_product(product_id, current_user=current_user)
        product_dir = self.store.get_product_dir(record.id)
        source_asset = self.image_agent._save_source(image=image, run_dir=product_dir, output_service=self.output_service)
        cutout_asset = await self.image_agent._build_cutout(
            image=image,
            core_data=record.product.core,
            run_dir=product_dir,
            output_service=self.output_service,
        )
        images = self._pending_generated_images(
            source_asset=source_asset,
            transparent_cutout=cutout_asset,
        )
        validation = self.pipeline.validation.validate_pipeline(
            core=record.product.core,
            amazon=record.product.amazon,
            ebay=record.product.ebay,
            etsy=record.product.etsy,
            tiktok=record.product.tiktok,
            shopify=record.product.shopify,
            images=images,
        )
        updated_product = record.product.model_copy(
            update={
                "images": images,
                "intelligence": record.product.intelligence.model_copy(update={"validation": validation}),
            }
        )
        updated = record.model_copy(update={"product": updated_product, "updated_at": self._timestamp()})
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    def delete_product(
        self,
        product_id: str,
        current_user: AuthenticatedUser | None = None,
    ) -> None:
        record = self.get_product(product_id, current_user=current_user)
        user_id = current_user.user_id if current_user is not None else None
        deleted = False
        if hasattr(self.store, "delete"):
            deleted = bool(self.store.delete(product_id, user_id=user_id))  # type: ignore[attr-defined]
        if not deleted:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Product not found.")
        del record

    async def optimize_product(
        self,
        product_id: str,
        payload: ProductOptimizationRequest,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        record = self.get_product(product_id, current_user=current_user)
        current_product = record.product
        research = await self.pipeline.research.build_research_bundle(current_product.core)
        seo = await self.pipeline.seo.process(current_product.core, research)

        optimized = await self.optimization_agent.process(
            product=current_product,
            research=research,
            seo=seo,
            marketplaces=payload.marketplaces,
            optimize_core=payload.optimize_core,
        )
        optimized_core = self.optimization_agent.coerce_core(optimized.get("core"), current_product.core)
        optimized_amazon = self.optimization_agent.coerce_amazon(optimized.get("amazon"), current_product.amazon)
        optimized_ebay = self.optimization_agent.coerce_ebay(optimized.get("ebay"), current_product.ebay)
        optimized_etsy = self.optimization_agent.coerce_etsy(optimized.get("etsy"), current_product.etsy)
        optimized_tiktok = self.optimization_agent.coerce_tiktok(optimized.get("tiktok"), current_product.tiktok)
        optimized_shopify = self.optimization_agent.coerce_shopify(optimized.get("shopify"), current_product.shopify)

        validation = self.pipeline.validation.validate_pipeline(
            core=optimized_core,
            amazon=optimized_amazon,
            ebay=optimized_ebay,
            etsy=optimized_etsy,
            tiktok=optimized_tiktok,
            shopify=optimized_shopify,
            images=current_product.images,
        )
        optimized_product = ProductPipelineResponse(
            core=optimized_core,
            amazon=optimized_amazon,
            ebay=optimized_ebay,
            etsy=optimized_etsy,
            tiktok=optimized_tiktok,
            shopify=optimized_shopify,
            images=current_product.images,
            intelligence={
                "research": research,
                "seo": seo,
                "validation": validation,
            },
        )
        updated = record.model_copy(
            update={
                "product": optimized_product,
                "updated_at": self._timestamp(),
            }
        )
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    async def optimize_marketplace(
        self,
        product_id: str,
        marketplace: MarketplaceLiteral,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        payload = ProductOptimizationRequest(marketplaces=[marketplace], optimize_core=False)
        return await self.optimize_product(product_id, payload, current_user=current_user)

    async def regenerate_marketplace(
        self,
        product_id: str,
        marketplace: MarketplaceLiteral,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        return await self.generate_product_images(
            product_id,
            marketplaces=[marketplace],
            current_user=current_user,
        )

    async def generate_product_images(
        self,
        product_id: str,
        *,
        marketplaces: list[MarketplaceLiteral],
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        record = self.get_product(product_id, current_user=current_user)
        source_image = self._load_source_image(record)
        core = record.product.core
        research = await self.pipeline.research.build_research_bundle(core)
        seo = await self.pipeline.seo.process(core, research)
        amazon = record.product.amazon
        ebay = record.product.ebay
        etsy = record.product.etsy
        tiktok = record.product.tiktok
        shopify = record.product.shopify
        text_content = await self._generate_marketplace_content(core=core, research=research, seo=seo)
        if "amazon" in marketplaces:
            amazon = text_content["amazon"]
        if "ebay" in marketplaces:
            ebay = text_content["ebay"]
        if "etsy" in marketplaces:
            etsy = text_content["etsy"]
        if "tiktok" in marketplaces:
            tiktok = text_content["tiktok"]
        if "shopify" in marketplaces:
            shopify = text_content["shopify"]

        product_dir = self.store.get_product_dir(record.id)
        generated_images = await self._generate_marketplace_images(
            source=source_image,
            existing_images=record.product.images,
            core_data=core,
            amazon_data=amazon,
            ebay_data=ebay,
            etsy_data=etsy,
            tiktok_data=tiktok,
            shopify_data=shopify,
            marketplaces=marketplaces,
            run_dir=product_dir,
        )
        latest_record = self.get_product(product_id, current_user=current_user)
        latest_product = latest_record.product
        images = latest_product.images.model_copy(
            update={marketplace: getattr(generated_images, marketplace) for marketplace in marketplaces}
        )
        merged_amazon = amazon if "amazon" in marketplaces else latest_product.amazon
        merged_ebay = ebay if "ebay" in marketplaces else latest_product.ebay
        merged_etsy = etsy if "etsy" in marketplaces else latest_product.etsy
        merged_tiktok = tiktok if "tiktok" in marketplaces else latest_product.tiktok
        merged_shopify = shopify if "shopify" in marketplaces else latest_product.shopify
        validation = self.pipeline.validation.validate_pipeline(
            core=latest_product.core,
            amazon=merged_amazon,
            ebay=merged_ebay,
            etsy=merged_etsy,
            tiktok=merged_tiktok,
            shopify=merged_shopify,
            images=images,
        )
        product = ProductPipelineResponse(
            core=latest_product.core,
            amazon=merged_amazon,
            ebay=merged_ebay,
            etsy=merged_etsy,
            tiktok=merged_tiktok,
            shopify=merged_shopify,
            images=images,
            intelligence={
                "research": research,
                "seo": seo,
                "validation": validation,
            },
        )
        updated = latest_record.model_copy(update={"product": product, "updated_at": self._timestamp()})
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    async def _build_generated_product(
        self,
        *,
        image: ImagePayload,
        title: str,
        image_marketplaces: list[MarketplaceLiteral],
    ) -> dict[str, object]:
        run_dir = self.output_service.create_run_dir()
        vision_data = await self.pipeline.vision.process(image)
        self.output_service.save_json(run_dir, "vision", vision_data.model_dump())
        core_data = await self.pipeline.core.process(title, vision_data)
        core_data = await self.pipeline.attribute_mapper.process(core_data, vision_data)
        self.output_service.save_json(run_dir, "core", core_data.model_dump())
        research = await self.pipeline.research.build_research_bundle(core_data)
        self.output_service.save_json(run_dir, "research", research.model_dump())
        seo = await self.pipeline.seo.process(core_data, research)
        self.output_service.save_json(run_dir, "seo", seo.model_dump())

        text_content = await self._generate_marketplace_content(core=core_data, research=research, seo=seo)
        for marketplace, payload in text_content.items():
            self.output_service.save_json(run_dir, marketplace, payload.model_dump())

        source_asset = self.image_agent._save_source(image=image, run_dir=run_dir, output_service=self.output_service)
        cutout_asset = await self.image_agent._build_cutout(
            image=image,
            core_data=core_data,
            run_dir=run_dir,
            output_service=self.output_service,
        )
        images = self._pending_generated_images(
            source_asset=source_asset,
            transparent_cutout=cutout_asset,
        )
        if image_marketplaces:
            images = await self._generate_marketplace_images(
                source=image,
                existing_images=images,
                core_data=core_data,
                amazon_data=text_content["amazon"],
                ebay_data=text_content["ebay"],
                etsy_data=text_content["etsy"],
                tiktok_data=text_content["tiktok"],
                shopify_data=text_content["shopify"],
                marketplaces=image_marketplaces,
                run_dir=run_dir,
            )
        self.output_service.save_json(run_dir, "images", images.model_dump())
        validation = self.pipeline.validation.validate_pipeline(
            core=core_data,
            amazon=text_content["amazon"],
            ebay=text_content["ebay"],
            etsy=text_content["etsy"],
            tiktok=text_content["tiktok"],
            shopify=text_content["shopify"],
            images=images,
        )
        product = ProductPipelineResponse(
            core=core_data,
            amazon=text_content["amazon"],
            ebay=text_content["ebay"],
            etsy=text_content["etsy"],
            tiktok=text_content["tiktok"],
            shopify=text_content["shopify"],
            images=images,
            intelligence={
                "research": research,
                "seo": seo,
                "validation": validation,
            },
        )
        self.output_service.save_json(run_dir, "validation", validation.model_dump())
        self.output_service.save_json(run_dir, "final", product.model_dump())
        return {
            "run_id": run_dir.name,
            "product": product,
        }

    async def _generate_marketplace_content(
        self,
        *,
        core: CoreProductResponse,
        research: MarketResearchBundleResponse,
        seo: SeoInsightsResponse,
    ) -> dict[str, AmazonResponse | EbayResponse | EtsyResponse | TikTokResponse | ShopifyResponse]:
        amazon, ebay, etsy, tiktok, shopify = await asyncio.gather(
            self.amazon_agent.process(core, research=research.amazon, seo=seo),
            self.ebay_agent.process(core, research=research.ebay, seo=seo),
            self.etsy_agent.process(core, research=research.etsy, seo=seo),
            self.tiktok_agent.process(core, research=research.tiktok, seo=seo),
            self.shopify_agent.process(core, research=research.shopify, seo=seo),
        )
        return {
            "amazon": amazon,
            "ebay": ebay,
            "etsy": etsy,
            "tiktok": tiktok,
            "shopify": shopify,
        }

    async def _generate_marketplace_images(
        self,
        *,
        source: ImagePayload,
        existing_images: GeneratedImagesResponse,
        core_data: CoreProductResponse,
        amazon_data: AmazonResponse,
        ebay_data: EbayResponse,
        etsy_data: EtsyResponse,
        tiktok_data: TikTokResponse,
        shopify_data: ShopifyResponse,
        marketplaces: list[MarketplaceLiteral],
        run_dir: Path,
    ) -> GeneratedImagesResponse:
        async def generate_one(marketplace: MarketplaceLiteral) -> tuple[MarketplaceLiteral, ImageVariantResponse]:
            asset = await self.image_agent.regenerate_marketplace_asset(
                marketplace=marketplace,
                source=source,
                existing_images=existing_images,
                core_data=core_data,
                amazon_data=amazon_data,
                ebay_data=ebay_data,
                etsy_data=etsy_data,
                tiktok_data=tiktok_data,
                shopify_data=shopify_data,
                run_dir=run_dir,
                output_service=self.output_service,
            )
            return marketplace, asset

        unique_marketplaces = list(dict.fromkeys(marketplaces))
        generated = await asyncio.gather(*(generate_one(marketplace) for marketplace in unique_marketplaces))
        return existing_images.model_copy(update={marketplace: asset for marketplace, asset in generated})

    def start_publish_target_analysis(
        self,
        product_id: str,
        marketplace: MarketplaceLiteral,
        payload: PublishTargetAnalysisRequest,
        current_user: AuthenticatedUser | None = None,
    ) -> PublishTargetAnalysisJobResponse:
        record = self.get_product(product_id, current_user=current_user)
        core = self._merge_publish_target_core(record.product.core, payload.product_identity)
        preview_research = self.pipeline.research._build_marketplace_research(marketplace, core)
        preview_result = self._build_publish_target_analysis(
            record,
            marketplace,
            preview_research,
            core=core,
            publish_fields=payload.publish_fields.model_dump(exclude_none=True) if payload.publish_fields is not None else {},
        )
        job = PublishTargetJobStore.create(
            product_id,
            marketplace,
            user_id=current_user.user_id if current_user is not None else None,
            result=preview_result,
        )
        asyncio.create_task(
            self._run_publish_target_analysis_job(
                job_id=job.job_id,
                product_id=product_id,
                marketplace=marketplace,
                payload=payload,
                current_user=current_user,
            )
        )
        return PublishTargetJobStore.to_response(job)

    def get_publish_target_analysis_job(
        self,
        job_id: str,
        current_user: AuthenticatedUser | None = None,
    ) -> PublishTargetAnalysisJobResponse | None:
        job = PublishTargetJobStore.get(job_id)
        if job is None:
            return None
        if current_user is not None and job.user_id not in {None, current_user.user_id}:
            return None
        return PublishTargetJobStore.to_response(job)

    async def _run_publish_target_analysis_job(
        self,
        *,
        job_id: str,
        product_id: str,
        marketplace: MarketplaceLiteral,
        payload: PublishTargetAnalysisRequest,
        current_user: AuthenticatedUser | None,
    ) -> None:
        PublishTargetJobStore.update_running(job_id)
        try:
            result = await self._resolve_publish_target_analysis(
                product_id=product_id,
                marketplace=marketplace,
                payload=payload,
                current_user=current_user,
            )
        except Exception as exc:
            PublishTargetJobStore.update_failed(job_id, str(exc))
            return

        PublishTargetJobStore.update_completed(job_id, result)

    async def _resolve_publish_target_analysis(
        self,
        *,
        product_id: str,
        marketplace: MarketplaceLiteral,
        payload: PublishTargetAnalysisRequest,
        current_user: AuthenticatedUser | None = None,
    ) -> PublishTargetAnalysisResponse:
        record = self.get_product(product_id, current_user=current_user)
        core = self._merge_publish_target_core(record.product.core, payload.product_identity)
        publish_fields = payload.publish_fields.model_dump(exclude_none=True) if payload.publish_fields is not None else {}
        research = await self.pipeline.research.build_research_bundle(core)
        selected_research = self._research_for_marketplace(research, marketplace)
        fallback = self._build_publish_target_analysis(
            record,
            marketplace,
            selected_research,
            core=core,
            publish_fields=publish_fields,
        )

        openai_service = self.pipeline.openai_service
        if openai_service is None or not openai_service.enabled:
            return fallback

        try:
            data = await openai_service.generate_structured_output(
                system_prompt=PromptRegistry.get_publish_target_prompt(),
                user_payload={
                    "marketplace": marketplace,
                    "product_identity": core.model_dump(),
                    "publish_fields": publish_fields,
                    "product": record.product.model_dump(exclude={"images", "intelligence"}),
                    "research": selected_research.model_dump(),
                    "baseline_analysis": fallback.model_dump(),
                },
                schema_name="publish_target_analysis",
                schema=self._PUBLISH_TARGET_SCHEMA,
            )
            return self._from_publish_target_data(data, fallback)
        except Exception:
            return fallback

    def add_size_variant(
        self,
        product_id: str,
        marketplace: MarketplaceLiteral,
        payload: VariantCreateRequest,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        record = self.get_product(product_id, current_user=current_user)
        variant = ProductVariantResponse(
            id=self._new_id(),
            marketplace=marketplace,
            variant_type="size",
            name=payload.name,
            value=payload.value or payload.name,
            image=None,
            created_at=self._timestamp(),
        )
        updated = self._append_variant(record, marketplace, variant)
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    def add_color_variant(
        self,
        product_id: str,
        marketplace: MarketplaceLiteral,
        payload: VariantCreateRequest,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        record = self.get_product(product_id, current_user=current_user)
        source_image = self._load_source_image(record)
        product_dir = self.store.get_product_dir(record.id)
        image_asset = self.image_agent.build_color_variant_asset(
            marketplace=marketplace,
            source=source_image,
            existing_images=record.product.images,
            color_name=payload.value or payload.name,
            title=record.product.core.normalized_title,
            run_dir=product_dir,
            output_service=self.output_service,
        )
        variant = ProductVariantResponse(
            id=self._new_id(),
            marketplace=marketplace,
            variant_type="color",
            name=payload.name,
            value=payload.value or payload.name,
            image=image_asset,
            created_at=self._timestamp(),
        )
        updated = self._append_variant(record, marketplace, variant)
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    def _append_variant(
        self,
        record: ProductRecordResponse,
        marketplace: MarketplaceLiteral,
        variant: ProductVariantResponse,
    ) -> ProductRecordResponse:
        variant_map = {
            "amazon": list(record.variants.amazon),
            "ebay": list(record.variants.ebay),
            "etsy": list(record.variants.etsy),
            "tiktok": list(record.variants.tiktok),
            "shopify": list(record.variants.shopify),
        }
        variant_map[marketplace].append(variant)
        return record.model_copy(
            update={
                "variants": MarketplaceVariantsResponse(**variant_map),
                "updated_at": self._timestamp(),
            }
        )

    @staticmethod
    def _research_for_marketplace(research: MarketResearchBundleResponse, marketplace: MarketplaceLiteral) -> MarketplaceResearchResponse:
        if marketplace == "amazon":
            return research.amazon
        if marketplace == "ebay":
            return research.ebay
        if marketplace == "etsy":
            return research.etsy
        if marketplace == "tiktok":
            return research.tiktok
        return research.shopify

    def _build_publish_target_analysis(
        self,
        record: ProductRecordResponse,
        marketplace: MarketplaceLiteral,
        research: MarketplaceResearchResponse,
        *,
        core: CoreProductResponse | None = None,
        publish_fields: dict[str, object] | None = None,
    ) -> PublishTargetAnalysisResponse:
        core = core or record.product.core
        product_label = self._product_identity_label(core)
        vendor = self._coalesce_publish_field(
            publish_fields.get("vendor") if publish_fields is not None else None,
            self._publish_vendor_for_marketplace(record, marketplace),
        )
        sku = self._coalesce_publish_field(
            publish_fields.get("default_sku") if publish_fields is not None else None,
            self._publish_sku_for_marketplace(record, marketplace),
        )
        description = self._coalesce_publish_field(
            publish_fields.get("publish_description") if publish_fields is not None else None,
            self._publish_description_for_marketplace(record, marketplace),
        )
        suggested = self._suggested_price_for_marketplace(core, marketplace, research)
        if suggested is not None and research.source_mode != "live_api":
            suggested = suggested.model_copy(update={"source": "ai_market_analysis"})
        provided_default_price = self._parse_price(publish_fields.get("default_price")) if publish_fields is not None else None
        default_price = self._default_price_from_band(suggested, research, core, marketplace)
        if default_price is None:
            default_price = provided_default_price
        if default_price is None:
            default_price = self._fallback_default_price(core, marketplace)
        market_signal = self._market_signal_for_marketplace(marketplace, research)
        summary = self._analysis_summary_for_marketplace(product_label, marketplace, research, default_price, core.product_summary)
        return PublishTargetAnalysisResponse(
            marketplace=marketplace,
            vendor=vendor,
            default_sku=sku,
            default_price=f"{default_price:.2f}",
            publish_description=description,
            suggested_price_range=suggested,
            market_signal=market_signal,
            analysis_summary=summary,
        )

    def _from_publish_target_data(
        self,
        data: dict[str, object],
        fallback: PublishTargetAnalysisResponse,
    ) -> PublishTargetAnalysisResponse:
        suggested = data.get("suggested_price_range")
        if isinstance(suggested, dict):
            range_payload: dict[str, object] | None = suggested
        else:
            range_payload = None

        normalized_range = self._normalize_suggested_price_range(range_payload, fallback.suggested_price_range)
        normalized_default_price = self._normalize_default_price(
            data.get("default_price"),
            normalized_range,
            fallback.default_price,
        )

        payload: dict[str, object] = {
            "marketplace": str(data.get("marketplace", fallback.marketplace)),
            "vendor": str(data.get("vendor", fallback.vendor)),
            "default_sku": str(data.get("default_sku", fallback.default_sku)),
            "default_price": f"{normalized_default_price:.2f}",
            "publish_description": str(data.get("publish_description", fallback.publish_description)),
            "suggested_price_range": normalized_range,
            "market_signal": str(data.get("market_signal", fallback.market_signal)),
            "analysis_summary": str(data.get("analysis_summary", fallback.analysis_summary)),
        }
        return PublishTargetAnalysisResponse.model_validate(payload)

    @classmethod
    def _normalize_suggested_price_range(
        cls,
        value: dict[str, object] | None,
        fallback: SuggestedPriceRangeResponse | None,
    ) -> SuggestedPriceRangeResponse | None:
        if fallback is None and not isinstance(value, dict):
            return None

        fallback_minimum = fallback.minimum if fallback is not None else 1.0
        fallback_maximum = fallback.maximum if fallback is not None else max(fallback_minimum, 10.0)
        fallback_recommended = fallback.recommended if fallback is not None else fallback_minimum
        fallback_currency = fallback.currency if fallback is not None else "USD"
        fallback_source = fallback.source if fallback is not None else "market_research"

        if not isinstance(value, dict):
            return fallback

        minimum = cls._parse_price(value.get("minimum")) or fallback_minimum
        maximum = cls._parse_price(value.get("maximum")) or fallback_maximum
        recommended = cls._parse_price(value.get("recommended")) or fallback_recommended

        if maximum < minimum:
            minimum, maximum = maximum, minimum

        if maximum <= 0:
            maximum = fallback_maximum
        if minimum <= 0:
            minimum = min(fallback_minimum, maximum)

        if maximum <= minimum:
            spread = max(fallback_maximum - fallback_minimum, minimum * 0.08, 1.0)
            maximum = round(minimum + spread, 2)

        recommended = min(max(recommended, minimum), maximum)

        return SuggestedPriceRangeResponse(
            minimum=round(minimum, 2),
            maximum=round(maximum, 2),
            recommended=round(recommended, 2),
            currency=str(value.get("currency", fallback_currency) or fallback_currency),
            source=str(value.get("source", fallback_source) or fallback_source),
        )

    @classmethod
    def _normalize_default_price(
        cls,
        value: object,
        suggested_range: SuggestedPriceRangeResponse | None,
        fallback_default_price: str,
    ) -> float:
        fallback_price = cls._parse_price(fallback_default_price) or 1.0
        parsed = cls._parse_price(value) or fallback_price

        if suggested_range is None:
            return round(parsed, 2)

        minimum = suggested_range.minimum
        maximum = suggested_range.maximum
        recommended = suggested_range.recommended

        if parsed < minimum:
            return round(max(minimum, recommended), 2)
        if parsed > maximum:
            return round(min(maximum, recommended), 2)
        return round(parsed, 2)

    @staticmethod
    def _default_price_from_band(
        suggested: SuggestedPriceRangeResponse | None,
        research: MarketplaceResearchResponse,
        core: CoreProductResponse,
        marketplace: MarketplaceLiteral,
    ) -> float | None:
        if suggested is None:
            return None

        minimum = suggested.minimum
        maximum = suggested.maximum
        recommended = suggested.recommended
        if maximum <= minimum:
            return round(recommended, 2)

        identity = " ".join(
            [
                core.normalized_title,
                core.source_title,
                core.category,
                core.product_type,
                core.product_summary,
                " ".join(core.features),
            ]
        ).lower()
        is_high_value_electronics = any(
            keyword in identity
            for keyword in (
                "playstation",
                "ps5",
                "xbox",
                "nintendo switch",
                "console",
                "gaming",
                "electronics",
            )
        )

        anchor_ratio = 0.78 if is_high_value_electronics else 0.66
        if research.source_mode == "live_api":
            anchor_ratio = max(anchor_ratio, 0.76)
        elif marketplace in {"amazon", "shopify"}:
            anchor_ratio = max(anchor_ratio, 0.7)

        strategic_price = minimum + ((maximum - minimum) * anchor_ratio)
        strategic_price = max(strategic_price, recommended)
        return round(min(maximum, strategic_price), 2)

    @staticmethod
    def _publish_vendor_for_marketplace(record: ProductRecordResponse, marketplace: MarketplaceLiteral) -> str:
        core = record.product.core
        vendor = core.attributes.get("brand") or core.attributes.get("vendor")
        if vendor:
            return vendor.strip()
        if marketplace == "shopify":
            return core.category or "CommandCtr"
        return core.normalized_title.split(" ")[0] if core.normalized_title else "CommandCtr"

    @staticmethod
    def _publish_sku_for_marketplace(record: ProductRecordResponse, marketplace: MarketplaceLiteral) -> str:
        core = record.product.core
        sku = core.attributes.get("sku")
        if sku and sku.strip():
            return sku.strip()
        prefix = marketplace[:3].upper()
        seed = re.sub(r"[^A-Za-z0-9]+", "", core.normalized_title or core.source_title or record.id).upper()[:8]
        if not seed:
            seed = record.id[:8].upper()
        return f"{prefix}-{seed}"

    @staticmethod
    def _publish_description_for_marketplace(record: ProductRecordResponse, marketplace: MarketplaceLiteral) -> str:
        product = record.product
        if marketplace == "amazon":
            text = product.amazon.description
        elif marketplace == "ebay":
            text = product.ebay.listing_notes
        elif marketplace == "etsy":
            text = product.etsy.description
        elif marketplace == "tiktok":
            text = product.tiktok.social_description
        else:
            text = ProductService._strip_html(product.shopify.body_html) or product.shopify.seo_description

        text = str(text or "").strip()
        if text:
            return text
        return product.core.product_summary.strip() or product.core.normalized_title.strip()

    @staticmethod
    def _strip_html(value: str) -> str:
        text = re.sub(r"<[^>]+>", " ", value or "")
        return re.sub(r"\s+", " ", text).strip()

    @staticmethod
    def _fallback_default_price(core: CoreProductResponse, marketplace: MarketplaceLiteral) -> float:
        current_price = ProductService._parse_price(core.attributes.get("price"))
        if current_price is None:
            current_price = ProductService._estimate_base_price(core)
        multiplier = {
            "amazon": 1.0,
            "ebay": 0.97,
            "etsy": 1.05,
            "tiktok": 0.95,
            "shopify": 1.08,
        }[marketplace]
        return round(max(0.01, current_price * multiplier), 2)

    @staticmethod
    def _suggested_price_for_marketplace(
        core: CoreProductResponse,
        marketplace: MarketplaceLiteral,
        research: MarketplaceResearchResponse,
    ) -> SuggestedPriceRangeResponse | None:
        listing_prices = [listing.price for listing in research.similar_listings if listing.price is not None]
        if listing_prices:
            observed_prices = sorted(listing_prices)
            observed_min = observed_prices[0]
            observed_max = observed_prices[-1]
            observed_median = median(observed_prices)
            observed_mean = mean(observed_prices)
            spread = observed_max - observed_min
            if spread <= 0:
                spread = max(0.08 * observed_median, 5.0)

            marketplace_bias = {
                "amazon": 1.0,
                "ebay": 0.99,
                "etsy": 1.02,
                "tiktok": 0.97,
                "shopify": 1.04,
            }[marketplace]
            recommended = round(max(0.01, (observed_median + observed_mean) / 2 * marketplace_bias), 2)

            padding = max(spread * 0.18, recommended * 0.06)
            minimum = round(max(0.01, min(observed_min, recommended - padding)), 2)
            maximum = round(max(observed_max, recommended + padding), 2)
        else:
            base_price = ProductService._estimate_base_price(core)
            current_price = ProductService._parse_price(core.attributes.get("price")) or base_price
            spread = max(0.18, min(0.42, 0.18 + (len(core.features) * 0.025) + (len(core.attributes) * 0.015)))
            minimum = round(max(0.01, current_price * (1 - spread)), 2)
            maximum = round(current_price * (1 + spread), 2)
            recommended = round(current_price, 2)

        return SuggestedPriceRangeResponse(
            minimum=round(minimum, 2),
            maximum=round(maximum, 2),
            recommended=recommended,
            currency="USD",
            source=research.source_mode or "market_research",
        )

    @staticmethod
    def _market_signal_for_marketplace(marketplace: MarketplaceLiteral, research: MarketplaceResearchResponse) -> str:
        signals = ", ".join(research.keyword_signals[:4])
        source = research.source_mode.replace("_", " ")
        if signals:
            return f"{marketplace.title()} {source}: {signals}"
        return f"{marketplace.title()} {source}"

    @staticmethod
    def _analysis_summary_for_marketplace(
        product_label: str,
        marketplace: MarketplaceLiteral,
        research: MarketplaceResearchResponse,
        default_price: float,
        product_summary: str = "",
    ) -> str:
        source_label = "live market data" if research.source_mode == "live_api" else "AI market analysis"
        if research.price_min is not None and research.price_max is not None:
            summary_part = f" {product_summary.strip()}" if product_summary.strip() else ""
            return (
                f"{product_label} on {marketplace.title()} supports a default price around ${default_price:.2f}."
                f"{summary_part}"
                f" {source_label} bands from ${research.price_min:.2f} to ${research.price_max:.2f}."
            )
        summary_part = f" {product_summary.strip()}" if product_summary.strip() else ""
        return f"{product_label} on {marketplace.title()} {source_label} recommends a default price around ${default_price:.2f}.{summary_part}"

    @staticmethod
    def _product_identity_label(core: CoreProductResponse) -> str:
        label = core.normalized_title.strip() or core.source_title.strip() or "Product"
        return re.sub(r"\s+", " ", label).strip()

    @staticmethod
    def _merge_publish_target_core(
        core: CoreProductResponse,
        payload: object | None,
    ) -> CoreProductResponse:
        if payload is None:
            return core

        payload_dict = payload.model_dump(exclude_none=True) if hasattr(payload, "model_dump") else {}
        updates: dict[str, object] = {}
        for field in ("normalized_title", "source_title", "category", "product_type", "product_summary", "features", "attributes"):
            value = payload_dict.get(field)
            if value is not None:
                updates[field] = value
        if not updates:
            return core
        return core.model_copy(update=updates)

    @staticmethod
    def _coalesce_publish_field(preferred: object | None, fallback: str) -> str:
        text = str(preferred).strip() if preferred is not None else ""
        return text or fallback

    @staticmethod
    def _parse_price(value: object) -> float | None:
        try:
            parsed = float(str(value).strip())
        except (TypeError, ValueError):
            return None
        if parsed <= 0:
            return None
        return round(parsed, 2)

    @staticmethod
    def _estimate_base_price(core: CoreProductResponse) -> float:
        identity = " ".join(
            [
                core.normalized_title,
                core.source_title,
                core.category,
                core.product_type,
                " ".join(core.features),
                " ".join(f"{key}:{value}" for key, value in core.attributes.items()),
            ]
        ).lower()

        if any(keyword in identity for keyword in ("ps5", "playstation 5", "playstation5", "sony playstation 5", "sony ps5", "ps 5")):
            return 499.99
        if any(keyword in identity for keyword in ("xbox series x", "xbox series s", "xbox", "gaming console", "video game console", "game console", "console")):
            if "series s" in identity:
                return 299.99
            return 399.99 if "console" in identity and "gaming" not in identity else 499.99
        if any(keyword in identity for keyword in ("nintendo switch", "switch oled", "switch lite")):
            if "lite" in identity:
                return 199.99
            return 299.99

        baseline = 14.99
        category = core.category.lower()
        product_type = core.product_type.lower()
        attributes = core.attributes

        if "drink" in category or "bottle" in product_type:
            baseline = 24.99
        elif "footwear" in category or "shoe" in product_type:
            baseline = 69.99
        elif "electronics" in category or "case" in product_type:
            baseline = 39.99
        elif "fashion" in category:
            baseline = 34.99

        if "material" in attributes:
            baseline += 4.0
        if "size" in attributes or "capacity" in attributes:
            baseline += 2.5
        if "brand" in attributes:
            baseline += 3.0

        return baseline

    @staticmethod
    def _new_id() -> str:
        return uuid4().hex

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _empty_generated_images(*, image_url: str) -> GeneratedImagesResponse:
        source_path = image_url.strip()
        return GeneratedImagesResponse(
            source=ProductService._placeholder_image(
                marketplace="source",
                generation_mode="source_passthrough" if source_path else "missing_source",
                prompt="Imported source image reference." if source_path else "No source image was provided during import.",
                absolute_path=source_path,
                expected_background="source",
                mime_type="image/jpeg",
                has_alpha=False,
            ),
            transparent_cutout=ProductService._placeholder_image(
                marketplace="transparent_cutout",
                generation_mode="pending_import_source",
                prompt="Transparent cutout will be generated after source imagery is available.",
                absolute_path="",
                expected_background="transparent",
                mime_type="image/png",
                has_alpha=True,
            ),
            amazon=ProductService._placeholder_image(
                marketplace="amazon",
                generation_mode="pending_import_source",
                prompt="Amazon image not generated yet.",
                absolute_path="",
                expected_background="white",
                mime_type="image/png",
                has_alpha=False,
            ),
            ebay=ProductService._placeholder_image(
                marketplace="ebay",
                generation_mode="pending_import_source",
                prompt="eBay image not generated yet.",
                absolute_path="",
                expected_background="white",
                mime_type="image/png",
                has_alpha=False,
            ),
            etsy=ProductService._placeholder_image(
                marketplace="etsy",
                generation_mode="pending_import_source",
                prompt="Etsy image not generated yet.",
                absolute_path="",
                expected_background="styled",
                mime_type="image/png",
                has_alpha=False,
            ),
            tiktok=ProductService._placeholder_image(
                marketplace="tiktok",
                generation_mode="pending_import_source",
                prompt="TikTok image not generated yet.",
                absolute_path="",
                expected_background="styled",
                mime_type="image/png",
                has_alpha=False,
            ),
            shopify=ProductService._placeholder_image(
                marketplace="shopify",
                generation_mode="pending_import_source",
                prompt="Shopify image not generated yet.",
                absolute_path=source_path,
                expected_background="styled",
                mime_type="image/png",
                has_alpha=False,
            ),
        )

    @staticmethod
    def _pending_generated_images(
        *,
        source_asset: ImageVariantResponse,
        transparent_cutout: ImageVariantResponse | None,
    ) -> GeneratedImagesResponse:
        return GeneratedImagesResponse(
            source=source_asset,
            transparent_cutout=transparent_cutout
            or ProductService._placeholder_image(
                marketplace="transparent_cutout",
                generation_mode="pending_cutout",
                prompt="Transparent cutout is not available yet.",
                absolute_path="",
                expected_background="transparent",
                mime_type="image/png",
                has_alpha=True,
            ),
            amazon=ProductService._placeholder_image(
                marketplace="amazon",
                generation_mode="pending_marketplace_generation",
                prompt="Generate Amazon image from the transparent cutout when needed.",
                absolute_path="",
                expected_background="white",
                mime_type="image/png",
                has_alpha=False,
            ),
            ebay=ProductService._placeholder_image(
                marketplace="ebay",
                generation_mode="pending_marketplace_generation",
                prompt="Generate eBay image from the transparent cutout when needed.",
                absolute_path="",
                expected_background="white",
                mime_type="image/png",
                has_alpha=False,
            ),
            etsy=ProductService._placeholder_image(
                marketplace="etsy",
                generation_mode="pending_marketplace_generation",
                prompt="Generate Etsy image from the transparent cutout when needed.",
                absolute_path="",
                expected_background="opaque",
                mime_type="image/png",
                has_alpha=False,
            ),
            tiktok=ProductService._placeholder_image(
                marketplace="tiktok",
                generation_mode="pending_marketplace_generation",
                prompt="Generate TikTok Shop image from the transparent cutout when needed.",
                absolute_path="",
                expected_background="opaque",
                mime_type="image/png",
                has_alpha=False,
            ),
            shopify=ProductService._placeholder_image(
                marketplace="shopify",
                generation_mode="pending_marketplace_generation",
                prompt="Generate Shopify image from the transparent cutout when needed.",
                absolute_path="",
                expected_background="opaque",
                mime_type="image/png",
                has_alpha=False,
            ),
        )

    @staticmethod
    def _placeholder_image(
        *,
        marketplace: str,
        generation_mode: str,
        prompt: str,
        absolute_path: str,
        expected_background: str,
        mime_type: str,
        has_alpha: bool,
    ) -> ImageVariantResponse:
        return ImageVariantResponse(
            marketplace=marketplace,
            relative_path="",
            absolute_path=absolute_path,
            prompt=prompt,
            generation_mode=generation_mode,
            mime_type=mime_type,
            validation=ImageValidationResponse(
                passed=bool(absolute_path) if marketplace == "source" else False,
                width=None,
                height=None,
                format=None,
                has_alpha=has_alpha,
                file_size_bytes=0,
                expected_width=None,
                expected_height=None,
                expected_background=expected_background,
                errors=[] if absolute_path else ["No image generated yet."],
                mime_type=mime_type,
            ),
        )

    @staticmethod
    def _empty_intelligence() -> IntelligenceLayerResponse:
        def empty_market_research(marketplace: str) -> MarketplaceResearchResponse:
            return MarketplaceResearchResponse(marketplace=marketplace)

        empty_section = SectionValidationResponse(passed=True, issues=[])
        return IntelligenceLayerResponse(
            research=MarketResearchBundleResponse(
                amazon=empty_market_research("amazon"),
                ebay=empty_market_research("ebay"),
                etsy=empty_market_research("etsy"),
                tiktok=empty_market_research("tiktok"),
                shopify=empty_market_research("shopify"),
            ),
            seo=SeoInsightsResponse(),
            validation=PipelineValidationResponse(
                core=empty_section,
                amazon=empty_section,
                ebay=empty_section,
                etsy=empty_section,
                tiktok=empty_section,
                shopify=empty_section,
                images=empty_section,
            ),
        )

    @staticmethod
    def _load_source_image(record: ProductRecordResponse) -> ImagePayload:
        source = record.product.images.source
        if source.absolute_path.startswith("http://") or source.absolute_path.startswith("https://"):
            with urlopen(source.absolute_path, timeout=30) as response:
                payload = response.read()
            filename = Path(source.relative_path or Path(source.absolute_path).name).name
            return ImagePayload(
                filename=filename or "source.bin",
                content_type=source.mime_type,
                data=payload,
            )

        path = Path(source.absolute_path)
        if not path.exists():
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Stored source image is unavailable for regeneration.",
            )
        return ImagePayload(
            filename=path.name,
            content_type=source.mime_type,
            data=path.read_bytes(),
        )
