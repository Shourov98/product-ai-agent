from __future__ import annotations

from datetime import UTC, datetime
from pathlib import Path
from uuid import uuid4

from fastapi import HTTPException, status

from app.auth import AuthenticatedUser
from app.agents.amazon_agent import AmazonAgent
from app.agents.ebay_agent import EbayAgent
from app.agents.image_agent import ImageAgent
from app.agents.shopify_agent import ShopifyAgent
from app.agents.tiktok_agent import TikTokAgent
from app.config import get_settings
from app.orchestrator.pipeline import ProductPipeline
from app.schemas.request import ProductUpdateRequest, VariantCreateRequest
from app.schemas.response import (
    AmazonResponse,
    CoreProductResponse,
    EbayResponse,
    MarketplaceLiteral,
    MarketplaceVariantsResponse,
    ProductListItemResponse,
    ProductPipelineResponse,
    ProductRecordResponse,
    ProductVariantResponse,
    ShopifyResponse,
    TikTokResponse,
)
from app.services.image_service import ImagePayload
from app.services.mongo_product_store import MongoProductStore
from app.services.output_service import OutputService
from app.services.product_store import ProductStore


class ProductService:
    def __init__(self) -> None:
        settings = get_settings()
        self.settings = settings
        self.pipeline = ProductPipeline()
        self.store = (
            MongoProductStore(
                mongodb_uri=settings.mongodb_uri,
                db_name=settings.mongodb_db_name,
                products_collection=settings.mongodb_products_collection,
            )
            if settings.mongodb_enabled and settings.mongodb_uri
            else ProductStore(settings.output_dir)
        )
        self.output_service = OutputService(settings.output_dir)
        self.image_agent = ImageAgent(self.pipeline.openai_service)
        self.amazon_agent = AmazonAgent(self.pipeline.ollama_service, self.pipeline.openai_service)
        self.ebay_agent = EbayAgent(self.pipeline.ollama_service, self.pipeline.openai_service)
        self.tiktok_agent = TikTokAgent(self.pipeline.ollama_service, self.pipeline.openai_service)
        self.shopify_agent = ShopifyAgent(self.pipeline.ollama_service, self.pipeline.openai_service)

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

    def list_products(self, current_user: AuthenticatedUser | None = None) -> list[ProductListItemResponse]:
        return self.store.list(user_id=current_user.user_id if current_user is not None else None)

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

    async def regenerate_marketplace(
        self,
        product_id: str,
        marketplace: MarketplaceLiteral,
        current_user: AuthenticatedUser | None = None,
    ) -> ProductRecordResponse:
        record = self.get_product(product_id, current_user=current_user)
        source_image = self._load_source_image(record)

        core = record.product.core
        amazon = record.product.amazon
        ebay = record.product.ebay
        tiktok = record.product.tiktok
        shopify = record.product.shopify

        if marketplace == "amazon":
            amazon = await self.amazon_agent.process(core)
        elif marketplace == "ebay":
            ebay = await self.ebay_agent.process(core)
        elif marketplace == "tiktok":
            tiktok = await self.tiktok_agent.process(core)
        else:
            shopify = await self.shopify_agent.process(core)

        product_dir = self.store.get_product_dir(record.id)
        image_asset = await self.image_agent.regenerate_marketplace_asset(
            marketplace=marketplace,
            source=source_image,
            existing_images=record.product.images,
            core_data=core,
            amazon_data=amazon,
            ebay_data=ebay,
            tiktok_data=tiktok,
            shopify_data=shopify,
            run_dir=product_dir,
            output_service=self.output_service,
        )
        images = record.product.images.model_copy(update={marketplace: image_asset})
        product = ProductPipelineResponse(
            core=core,
            amazon=amazon,
            ebay=ebay,
            tiktok=tiktok,
            shopify=shopify,
            images=images,
        )
        updated = record.model_copy(update={"product": product, "updated_at": self._timestamp()})
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

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
    def _new_id() -> str:
        return uuid4().hex

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat()

    @staticmethod
    def _load_source_image(record: ProductRecordResponse) -> ImagePayload:
        source = record.product.images.source
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
