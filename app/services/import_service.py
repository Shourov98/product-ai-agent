from __future__ import annotations

from datetime import UTC, datetime
from urllib.request import urlopen
from uuid import uuid4

from fastapi import HTTPException, status

from app.auth import AuthenticatedUser
from app.schemas.imports import ImportRecordResponse, ImportUploadResponse, ImportedProductRow, UploadImportAsProductResponse
from app.schemas.request import MarketplaceRequestLiteral, ProductOptimizationRequest, ProductUpdateRequest
from app.schemas.response import MarketplaceVariantsResponse, ProductPipelineResponse
from app.services.image_service import ImagePayload
from app.services.import_store import ImportStore
from app.services.mongo_import_store import MongoImportStore
from app.services.product_import_service import ProductImportService
from app.services.product_service import ProductService


class ImportService:
    def __init__(self) -> None:
        self.product_service = ProductService()
        settings = self.product_service.settings
        self.store = (
            MongoImportStore(
                mongodb_uri=settings.mongodb_uri,
                db_name=settings.mongodb_db_name,
                imports_collection=settings.mongodb_imports_collection,
            )
            if settings.mongodb_uri and settings.mongodb_enabled
            else ImportStore(settings.import_store_dir)
        )
        self.parser = ProductImportService()

    def upload_file(
        self,
        *,
        filename: str,
        payload: bytes,
        current_user: AuthenticatedUser | None = None,
    ) -> ImportUploadResponse:
        existing_titles = {item.normalized_title.strip().lower() for item in self.product_service.list_products(current_user=current_user)}
        preview = self.parser.parse_file(filename=filename, payload=payload, existing_titles=existing_titles)
        records = [
            self._save_row_as_import_record(
                filename=filename,
                row=row,
                current_user=current_user,
            )
            for row in preview.rows
        ]
        return ImportUploadResponse(imported_count=len(records), records=records)

    def list_imports(self, current_user: AuthenticatedUser | None = None):
        return self.store.list(user_id=current_user.user_id if current_user is not None else None)

    def get_import(self, record_id: str, current_user: AuthenticatedUser | None = None) -> ImportRecordResponse:
        record = self.store.get(record_id, user_id=current_user.user_id if current_user is not None else None)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Import record not found.")
        return record

    def delete_import(self, record_id: str, current_user: AuthenticatedUser | None = None) -> None:
        self.get_import(record_id, current_user=current_user)
        self.store.delete(record_id, user_id=current_user.user_id if current_user is not None else None)

    async def upload_source_image(
        self,
        record_id: str,
        image: ImagePayload,
        current_user: AuthenticatedUser | None = None,
    ) -> ImportRecordResponse:
        record = self.get_import(record_id, current_user=current_user)
        run_dir = self.product_service.output_service.create_run_dir()
        generated_images = await self.product_service.image_agent.process(
            image=image,
            core_data=record.product.core,
            amazon_data=record.product.amazon,
            ebay_data=record.product.ebay,
            etsy_data=record.product.etsy,
            tiktok_data=record.product.tiktok,
            shopify_data=record.product.shopify,
            run_dir=run_dir,
            output_service=self.product_service.output_service,
        )
        updated_product = record.product.model_copy(update={"images": generated_images})
        updated = self._refresh_record_state(record, updated_product, current_user=current_user)
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    def update_import(
        self,
        record_id: str,
        payload: ProductUpdateRequest,
        current_user: AuthenticatedUser | None = None,
    ) -> ImportRecordResponse:
        record = self.get_import(record_id, current_user=current_user)
        updated_product = self._apply_product_update(record.product, payload)
        updated = self._refresh_record_state(record, updated_product, current_user=current_user)
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    async def optimize_import(
        self,
        record_id: str,
        payload: ProductOptimizationRequest,
        current_user: AuthenticatedUser | None = None,
    ) -> ImportRecordResponse:
        record = self.get_import(record_id, current_user=current_user)
        current_product = record.product
        research = await self.product_service.pipeline.research.build_research_bundle(current_product.core)
        seo = await self.product_service.pipeline.seo.process(current_product.core, research)
        pricing = self.product_service.pipeline.pricing.build_pricing(research)
        optimized = await self.product_service.optimization_agent.process(
            product=current_product,
            research=research,
            seo=seo,
            pricing=pricing,
            marketplaces=payload.marketplaces,
            optimize_core=payload.optimize_core,
        )
        optimized_product = ProductPipelineResponse(
            core=self.product_service.optimization_agent.coerce_core(optimized.get("core"), current_product.core),
            amazon=self.product_service.optimization_agent.coerce_amazon(optimized.get("amazon"), current_product.amazon),
            ebay=self.product_service.optimization_agent.coerce_ebay(optimized.get("ebay"), current_product.ebay),
            etsy=self.product_service.optimization_agent.coerce_etsy(optimized.get("etsy"), current_product.etsy),
            tiktok=self.product_service.optimization_agent.coerce_tiktok(optimized.get("tiktok"), current_product.tiktok),
            shopify=self.product_service.optimization_agent.coerce_shopify(optimized.get("shopify"), current_product.shopify),
            images=current_product.images,
            intelligence={
                "research": research,
                "seo": seo,
                "pricing": pricing,
                "validation": self.product_service.pipeline.validation.validate_pipeline(
                    core=self.product_service.optimization_agent.coerce_core(optimized.get("core"), current_product.core),
                    amazon=self.product_service.optimization_agent.coerce_amazon(optimized.get("amazon"), current_product.amazon),
                    ebay=self.product_service.optimization_agent.coerce_ebay(optimized.get("ebay"), current_product.ebay),
                    etsy=self.product_service.optimization_agent.coerce_etsy(optimized.get("etsy"), current_product.etsy),
                    tiktok=self.product_service.optimization_agent.coerce_tiktok(optimized.get("tiktok"), current_product.tiktok),
                    shopify=self.product_service.optimization_agent.coerce_shopify(optimized.get("shopify"), current_product.shopify),
                    images=current_product.images,
                ),
            },
        )
        updated = self._refresh_record_state(record, optimized_product, current_user=current_user)
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    async def optimize_import_marketplace(
        self,
        record_id: str,
        marketplace: MarketplaceRequestLiteral,
        current_user: AuthenticatedUser | None = None,
    ) -> ImportRecordResponse:
        return await self.optimize_import(
            record_id,
            ProductOptimizationRequest(marketplaces=[marketplace], optimize_core=False),
            current_user=current_user,
        )

    async def regenerate_import_marketplace(
        self,
        record_id: str,
        marketplace: MarketplaceRequestLiteral,
        current_user: AuthenticatedUser | None = None,
    ) -> ImportRecordResponse:
        record = self.get_import(record_id, current_user=current_user)
        source_image = self._load_source_image(record)
        core = record.product.core
        research = await self.product_service.pipeline.research.build_research_bundle(core)
        seo = await self.product_service.pipeline.seo.process(core, research)
        pricing = self.product_service.pipeline.pricing.build_pricing(research)
        amazon = record.product.amazon
        ebay = record.product.ebay
        etsy = record.product.etsy
        tiktok = record.product.tiktok
        shopify = record.product.shopify
        if marketplace == "amazon":
            amazon = await self.product_service.amazon_agent.process(core, research=research.amazon, seo=seo, pricing=pricing.amazon)
        elif marketplace == "ebay":
            ebay = await self.product_service.ebay_agent.process(core, research=research.ebay, seo=seo, pricing=pricing.ebay)
        elif marketplace == "etsy":
            etsy = await self.product_service.etsy_agent.process(core, research=research.etsy, seo=seo, pricing=pricing.etsy)
        elif marketplace == "tiktok":
            tiktok = await self.product_service.tiktok_agent.process(core, research=research.tiktok, seo=seo, pricing=pricing.tiktok)
        else:
            shopify = await self.product_service.shopify_agent.process(core, research=research.shopify, seo=seo, pricing=pricing.shopify)
        run_dir = self.product_service.output_service.create_run_dir()
        image_asset = await self.product_service.image_agent.regenerate_marketplace_asset(
            marketplace=marketplace,
            source=source_image,
            existing_images=record.product.images,
            core_data=core,
            amazon_data=amazon,
            ebay_data=ebay,
            etsy_data=etsy,
            tiktok_data=tiktok,
            shopify_data=shopify,
            run_dir=run_dir,
            output_service=self.product_service.output_service,
        )
        images = record.product.images.model_copy(update={marketplace: image_asset})
        updated_product = ProductPipelineResponse(
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
                "validation": self.product_service.pipeline.validation.validate_pipeline(
                    core=core, amazon=amazon, ebay=ebay, etsy=etsy, tiktok=tiktok, shopify=shopify, images=images
                ),
            },
        )
        updated = self._refresh_record_state(record, updated_product, current_user=current_user)
        self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
        return updated

    def upload_as_product(
        self,
        record_id: str,
        current_user: AuthenticatedUser | None = None,
    ) -> UploadImportAsProductResponse:
        record = self.get_import(record_id, current_user=current_user)
        if record.linked_product_id:
            product_record = self.product_service.get_product(record.linked_product_id, current_user=current_user)
            return UploadImportAsProductResponse(import_record=record, product_record=product_record)
        product_record = self.product_service.create_product_from_pipeline(
            product=record.product,
            current_user=current_user,
            status="draft",
        )
        updated_import = record.model_copy(
            update={
                "status": "uploaded",
                "linked_product_id": product_record.id,
                "updated_at": self._timestamp(),
            }
        )
        self.store.save(updated_import, user_id=current_user.user_id if current_user is not None else None)
        return UploadImportAsProductResponse(import_record=updated_import, product_record=product_record)

    def _save_row_as_import_record(
        self,
        *,
        filename: str,
        row: ImportedProductRow,
        current_user: AuthenticatedUser | None = None,
    ) -> ImportRecordResponse:
        product = self.product_service.build_imported_product(
            title=row.title,
            sku=row.sku,
            brand=row.brand,
            category=row.category,
            product_type=row.product_type,
            description=row.description,
            price=row.price,
            quantity=row.quantity,
            color=row.color,
            size=row.size,
            material=row.material,
            image_url=row.image_url,
        )
        mapped_status = "imported" if row.status == "ready" else "needs_review"
        if row.status == "duplicate":
            mapped_status = "duplicate"
        if row.status == "parse_issue":
            mapped_status = "parse_issue"
        record = ImportRecordResponse(
            id=uuid4().hex,
            status=mapped_status,  # type: ignore[arg-type]
            created_at=self._timestamp(),
            updated_at=self._timestamp(),
            source_filename=filename,
            source_type=row.source_type,
            source_reference=row.source_reference,
            confidence=row.confidence,
            missing_fields=row.missing_fields,
            notes=row.notes,
            linked_product_id=None,
            product=product,
            variants=MarketplaceVariantsResponse(),
        )
        self.store.save(record, user_id=current_user.user_id if current_user is not None else None)
        return record

    @staticmethod
    def _apply_product_update(product: ProductPipelineResponse, payload: ProductUpdateRequest) -> ProductPipelineResponse:
        updated_product = product

        if payload.core is not None:
            updated_product = updated_product.model_copy(
                update={
                    "core": updated_product.core.model_copy(update=payload.core.model_dump(exclude_unset=True))
                }
            )
        if payload.amazon is not None:
            updated_product = updated_product.model_copy(
                update={
                    "amazon": updated_product.amazon.model_copy(update=payload.amazon.model_dump(exclude_unset=True))
                }
            )
        if payload.tiktok is not None:
            updated_product = updated_product.model_copy(
                update={
                    "tiktok": updated_product.tiktok.model_copy(update=payload.tiktok.model_dump(exclude_unset=True))
                }
            )
        if payload.ebay is not None:
            updated_product = updated_product.model_copy(
                update={
                    "ebay": updated_product.ebay.model_copy(update=payload.ebay.model_dump(exclude_unset=True))
                }
            )
        if payload.etsy is not None:
            updated_product = updated_product.model_copy(
                update={
                    "etsy": updated_product.etsy.model_copy(update=payload.etsy.model_dump(exclude_unset=True))
                }
            )
        if payload.shopify is not None:
            updated_product = updated_product.model_copy(
                update={
                    "shopify": updated_product.shopify.model_copy(update=payload.shopify.model_dump(exclude_unset=True))
                }
            )

        return updated_product

    def _refresh_record_state(
        self,
        record: ImportRecordResponse,
        product: ProductPipelineResponse,
        *,
        current_user: AuthenticatedUser | None = None,
    ) -> ImportRecordResponse:
        missing_fields = self._build_missing_fields(product)
        existing_titles = {
            item.normalized_title.strip().lower()
            for item in self.product_service.list_products(current_user=current_user)
            if not record.linked_product_id or item.id != record.linked_product_id
        }
        normalized_title = product.core.normalized_title.strip().lower()
        duplicate = bool(normalized_title) and normalized_title in existing_titles

        next_status = "uploaded" if record.linked_product_id else "imported"
        if not product.core.normalized_title.strip():
            next_status = "parse_issue"
        elif duplicate:
            next_status = "duplicate"
        elif missing_fields:
            next_status = "needs_review"

        notes = [note for note in record.notes if note != "Title matches an existing saved product."]
        if duplicate:
            notes.append("Title matches an existing saved product.")

        return record.model_copy(
            update={
                "status": next_status,
                "missing_fields": missing_fields,
                "notes": notes,
                "product": product,
                "updated_at": self._timestamp(),
            }
        )

    @staticmethod
    def _build_missing_fields(product: ProductPipelineResponse) -> list[str]:
        missing_fields: list[str] = []
        if not product.core.normalized_title.strip():
            missing_fields.append("title")
        if not product.core.attributes.get("price", "").strip():
            missing_fields.append("price")
        if not product.core.attributes.get("quantity", "").strip():
            missing_fields.append("quantity")
        return missing_fields

    @staticmethod
    def _load_source_image(record: ImportRecordResponse) -> ImagePayload:
        source = record.product.images.source
        if source.absolute_path.startswith("http://") or source.absolute_path.startswith("https://"):
            with urlopen(source.absolute_path, timeout=30) as response:
                payload = response.read()
            return ImagePayload(
                filename=source.absolute_path.rstrip("/").split("/")[-1] or "source.bin",
                content_type=source.mime_type,
                data=payload,
            )
        raise HTTPException(status_code=status.HTTP_409_CONFLICT, detail="Imported record does not have a reachable source image URL.")

    @staticmethod
    def _timestamp() -> str:
        return datetime.now(UTC).isoformat()
