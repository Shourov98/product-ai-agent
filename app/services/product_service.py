from __future__ import annotations

import re
from datetime import UTC, datetime
from pathlib import Path
from typing import Any
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
from app.providers.factory import get_provider
from app.providers.kaggle_provider import KaggleProvider
from app.schemas.request import ProductOptimizationRequest, ProductUpdateRequest, VariantCreateRequest
from app.schemas.repricing import ProductMatchResponse, ProductRepricingRequest, ProductRepricingResponse
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
    MarketplacePricingResponse,
    MarketplaceResearchResponse,
    MarketResearchBundleResponse,
    MarketplaceVariantsResponse,
    PaginatedProductListResponse,
    PipelineValidationResponse,
    PricingInsightsResponse,
    ProductListItemResponse,
    ProductPipelineResponse,
    ProductRecordResponse,
    ProductVariantResponse,
    SectionValidationResponse,
    SeoInsightsResponse,
    ShopifyResponse,
    TikTokResponse,
)
from app.services.cloudinary_service import CloudinaryService
from app.services.image_service import ImagePayload
from app.services.mongo_product_store import MongoProductStore
from app.services.output_service import OutputService
from app.services.product_store import ProductStore
from app.services.repricing_engine import RepricingEngine
from app.utils.product_text import title_keywords, unique_strings


class ProductService:
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

    async def generate_and_store(
        self,
        image: ImagePayload,
        title: str,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        result = await self.pipeline.run_with_context(image, title)
        record = ProductRecordResponse(
            id=self._new_id(),
            status="draft",
            created_at=self._timestamp(),
            updated_at=self._timestamp(),
            run_id=result.run_dir.name,
            product=result.response,
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
        pricing = self.pipeline.pricing.build_pricing(research)

        optimized = await self.optimization_agent.process(
            product=current_product,
            research=research,
            seo=seo,
            pricing=pricing,
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
                "pricing": pricing,
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
        record = self.get_product(product_id, current_user=current_user)
        source_image = self._load_source_image(record)

        core = record.product.core
        research = await self.pipeline.research.build_research_bundle(core)
        seo = await self.pipeline.seo.process(core, research)
        pricing = self.pipeline.pricing.build_pricing(research)

        amazon = record.product.amazon
        ebay = record.product.ebay
        etsy = record.product.etsy
        tiktok = record.product.tiktok
        shopify = record.product.shopify

        if marketplace == "amazon":
            amazon = await self.amazon_agent.process(core, research=research.amazon, seo=seo, pricing=pricing.amazon)
        elif marketplace == "ebay":
            ebay = await self.ebay_agent.process(core, research=research.ebay, seo=seo, pricing=pricing.ebay)
        elif marketplace == "etsy":
            etsy = await self.etsy_agent.process(core, research=research.etsy, seo=seo, pricing=pricing.etsy)
        elif marketplace == "tiktok":
            tiktok = await self.tiktok_agent.process(core, research=research.tiktok, seo=seo, pricing=pricing.tiktok)
        else:
            shopify = await self.shopify_agent.process(core, research=research.shopify, seo=seo, pricing=pricing.shopify)

        product_dir = self.store.get_product_dir(record.id)
        image_asset = await self.image_agent.regenerate_marketplace_asset(
            marketplace=marketplace,
            source=source_image,
            existing_images=record.product.images,
            core_data=core,
            amazon_data=amazon,
            ebay_data=ebay,
            etsy_data=etsy,
            tiktok_data=tiktok,
            shopify_data=shopify,
            run_dir=product_dir,
            output_service=self.output_service,
        )
        images = record.product.images.model_copy(update={marketplace: image_asset})
        validation = self.pipeline.validation.validate_pipeline(
            core=core,
            amazon=amazon,
            ebay=ebay,
            etsy=etsy,
            tiktok=tiktok,
            shopify=shopify,
            images=images,
        )
        product = ProductPipelineResponse(
            core=core,
            amazon=amazon,
            ebay=ebay,
            etsy=etsy,
            tiktok=tiktok,
            shopify=shopify,
            images=images,
            intelligence={
                "research": research,
                "seo": seo,
                "pricing": pricing,
                "validation": validation,
            },
        )
        updated = record.model_copy(update={"product": product, "updated_at": self._timestamp()})
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    async def analyze_product_repricing(
        self,
        product_id: str,
        payload: ProductRepricingRequest,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRepricingResponse:
        record = self.get_product(product_id, current_user=current_user)
        provider = get_provider()
        matched_product = self._match_repricing_product(record, provider)
        engine = RepricingEngine(provider, self.pipeline.openai_service.client if self.pipeline.openai_service else None)
        repricing = await engine.run(matched_product.asin, payload.strategy, payload.dry_run)
        return ProductRepricingResponse(
            product_id=record.id,
            matched_product=matched_product,
            repricing=repricing,
        )

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

    def _match_repricing_product(
        self,
        record: ProductRecordResponse,
        provider: Any,
    ) -> ProductMatchResponse:
        if not isinstance(provider, KaggleProvider):
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Product-level repricing currently supports the Kaggle-backed Amazon provider only.",
            )

        frame = provider.products
        if frame.empty:
            raise HTTPException(
                status_code=status.HTTP_503_SERVICE_UNAVAILABLE,
                detail="Repricing dataset is unavailable.",
            )

        core = record.product.core
        title = core.normalized_title or core.source_title
        category = core.category.lower()
        product_type = core.product_type.lower()
        attribute_terms = [value.lower() for value in core.attributes.values() if value.strip()]
        query_terms = unique_strings(
            [product_type, category, *title_keywords(title), *attribute_terms],
            limit=8,
        )
        safe_terms = [re.escape(term) for term in query_terms[:5] if term]
        if not safe_terms:
            raise HTTPException(
                status_code=status.HTTP_422_UNPROCESSABLE_ENTITY,
                detail="Product data is not descriptive enough for repricing analysis.",
            )

        if "title_lower" not in frame.columns:
            frame["title_lower"] = frame["title"].astype(str).str.lower()
        if "category_lower" not in frame.columns:
            frame["category_lower"] = frame["category_name"].astype(str).str.lower()

        pattern = "|".join(safe_terms)
        candidates = frame[frame["title_lower"].str.contains(pattern, regex=True, na=False)].copy()
        if category:
            category_matches = candidates[candidates["category_lower"] == category]
            if not category_matches.empty:
                candidates = category_matches
        if candidates.empty and title_keywords(title):
            fallback_terms = [re.escape(term) for term in title_keywords(title)[:3]]
            fallback_pattern = "|".join(fallback_terms)
            candidates = frame[frame["title_lower"].str.contains(fallback_pattern, regex=True, na=False)].copy()
        if candidates.empty:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No comparable marketplace product was found for repricing.",
            )

        candidates["match_score"] = candidates.apply(
            lambda row: self._score_repricing_candidate(
                title_lower=str(row["title_lower"]),
                category_lower=str(row["category_lower"]),
                query_terms=query_terms,
                product_type=product_type,
                category=category,
                attribute_terms=attribute_terms,
                is_best_seller=bool(row.get("isBestSeller", False)),
                stars=float(row.get("stars", 0.0) or 0.0),
                reviews=int(row.get("reviews", 0) or 0),
                bought_in_last_month=int(row.get("boughtInLastMonth", 0) or 0),
            ),
            axis=1,
        )
        top = candidates.sort_values(
            by=["match_score", "boughtInLastMonth", "reviews", "stars"],
            ascending=[False, False, False, False],
        ).iloc[0]
        score = float(top["match_score"])
        if score < 2.0:
            raise HTTPException(
                status_code=status.HTTP_404_NOT_FOUND,
                detail="No confident marketplace match was found for repricing.",
            )

        return ProductMatchResponse(
            asin=str(top["asin"]),
            title=str(top["title"]),
            category=str(top.get("category_name", "Uncategorized")),
            confidence=max(0.55, min(0.99, round(score / 10.0, 2))),
            source="kaggle_dataset",
        )

    @staticmethod
    def _score_repricing_candidate(
        *,
        title_lower: str,
        category_lower: str,
        query_terms: list[str],
        product_type: str,
        category: str,
        attribute_terms: list[str],
        is_best_seller: bool,
        stars: float,
        reviews: int,
        bought_in_last_month: int,
    ) -> float:
        keyword_matches = sum(1.5 for term in query_terms if term and term in title_lower)
        product_type_bonus = 2.0 if product_type and product_type in title_lower else 0.0
        category_bonus = 2.0 if category and category == category_lower else 0.0
        attribute_bonus = sum(0.75 for term in attribute_terms[:4] if term and term in title_lower)
        bestseller_bonus = 1.0 if is_best_seller else 0.0
        rating_bonus = min(1.0, stars / 5.0)
        review_bonus = min(1.5, reviews / 4000)
        velocity_bonus = min(1.5, bought_in_last_month / 1500)
        return (
            keyword_matches
            + product_type_bonus
            + category_bonus
            + attribute_bonus
            + bestseller_bonus
            + rating_bonus
            + review_bonus
            + velocity_bonus
        )

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

        def empty_market_pricing(marketplace: str) -> MarketplacePricingResponse:
            return MarketplacePricingResponse(
                marketplace=marketplace,
                recommended=0.0,
                discounted_recommended=None,
                floor=0.0,
                ceiling=0.0,
                market_average=0.0,
                regular_price_average=None,
                sale_price_average=None,
                discount_percent_average=None,
                strategy="unpriced",
                confidence=0.0,
                summary="No pricing evidence is available yet.",
                reasons=[],
            )

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
            pricing=PricingInsightsResponse(
                amazon=empty_market_pricing("amazon"),
                ebay=empty_market_pricing("ebay"),
                etsy=empty_market_pricing("etsy"),
                tiktok=empty_market_pricing("tiktok"),
                shopify=empty_market_pricing("shopify"),
            ),
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
