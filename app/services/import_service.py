from __future__ import annotations

from datetime import UTC, datetime
from urllib.request import urlopen
from uuid import uuid4

from fastapi import HTTPException, status

from app.auth import AuthenticatedUser
from app.schemas.imports import CatalogConflictProductResponse, DuplicateResolutionResponse, ImportListItemResponse, ImportRecordResponse, ImportUploadResponse, ImportedProductRow, PaginatedImportListResponse, UploadImportAsProductResponse
from app.schemas.request import MarketplaceRequestLiteral, ProductOptimizationRequest, ProductUpdateRequest
from app.schemas.response import MarketplaceVariantsResponse, ProductPipelineResponse
from app.services.image_service import ImagePayload
from app.services.import_store import ImportStore
from app.services.mongo_import_store import MongoImportStore
from app.services.product_import_service import ProductImportService
from app.services.product_service import ProductService


class ImportService:
    _shared_product_service = None
    _shared_store = None
    _shared_parser = None

    def __init__(self) -> None:
        self.product_service = self._shared_product_service or ProductService()
        self.__class__._shared_product_service = self.product_service
        settings = self.product_service.settings
        self.store = self._shared_store or (
            MongoImportStore(
                mongodb_uri=settings.mongodb_uri,
                db_name=settings.mongodb_db_name,
                imports_collection=settings.mongodb_imports_collection,
            )
            if settings.mongodb_uri and settings.mongodb_enabled
            else ImportStore(settings.import_store_dir)
        )
        self.__class__._shared_store = self.store
        self.parser = self._shared_parser or ProductImportService()
        self.__class__._shared_parser = self.parser

    @classmethod
    def reset_shared_state(cls) -> None:
        store = cls._shared_store
        close = getattr(store, "close", None)
        if callable(close):
            close()

        cls._shared_product_service = None
        cls._shared_store = None
        cls._shared_parser = None

    def upload_file(
        self,
        *,
        filename: str,
        payload: bytes,
        current_user: AuthenticatedUser | None = None,
    ) -> ImportUploadResponse:
        existing_products = self.product_service.list_products(current_user=current_user)
        existing_titles = {item.normalized_title.strip().lower() for item in existing_products}
        existing_products_by_title: dict[str, list[str]] = {}
        for item in existing_products:
            key = item.normalized_title.strip().lower()
            if not key:
                continue
            existing_products_by_title.setdefault(key, []).append(item.id)

        preview = self.parser.parse_file(filename=filename, payload=payload, existing_titles=existing_titles)
        existing_records = self.store.list_records(user_id=current_user.user_id if current_user is not None else None)
        primary_by_fingerprint: dict[str, str] = {}
        for record in sorted(existing_records, key=lambda item: (item.created_at, item.id)):
            fingerprint = self._fingerprint_for_record(record)
            if not fingerprint:
                continue
            primary_by_fingerprint.setdefault(fingerprint, record.primary_record_id or record.id)

        records: list[ImportRecordResponse] = []
        for row in preview.rows:
            fingerprint = self._fingerprint_for_row(row)
            primary_record_id = primary_by_fingerprint.get(fingerprint) if fingerprint else None
            records.append(
                self._save_row_as_import_record(
                    filename=filename,
                    row=row,
                    primary_record_id=primary_record_id,
                    catalog_conflict_product_ids=existing_products_by_title.get(row.title.strip().lower(), []),
                    current_user=current_user,
                )
            )
            if fingerprint and primary_record_id is None and records[-1].status != "parse_issue":
                primary_by_fingerprint[fingerprint] = records[-1].id
        return ImportUploadResponse(imported_count=len(records), records=records)

    def list_imports(self, current_user: AuthenticatedUser | None = None) -> list[ImportListItemResponse]:
        return self.store.list(user_id=current_user.user_id if current_user is not None else None)

    def list_imports_paginated(
        self,
        *,
        page: int,
        page_size: int,
        current_user: AuthenticatedUser | None = None,
    ) -> PaginatedImportListResponse:
        return self.store.list_paginated(
            page=page,
            page_size=page_size,
            user_id=current_user.user_id if current_user is not None else None,
        )

    def get_import(self, record_id: str, current_user: AuthenticatedUser | None = None) -> ImportRecordResponse:
        record = self.store.get(record_id, user_id=current_user.user_id if current_user is not None else None)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Import record not found.")
        all_records = self.store.list_records(user_id=current_user.user_id if current_user is not None else None)
        return self._enrich_record(record, all_records)

    def delete_import(self, record_id: str, current_user: AuthenticatedUser | None = None) -> None:
        record = self.get_import(record_id, current_user=current_user)
        all_records = self.store.list_records(user_id=current_user.user_id if current_user is not None else None)
        group_records = self._group_records_for(record, all_records)
        self.store.delete(record_id, user_id=current_user.user_id if current_user is not None else None)
        if record.primary_record_id is None and group_records:
            remaining = [item for item in group_records if item.id != record.id]
            if remaining:
                promoted = min(remaining, key=lambda item: (item.created_at, item.id))
                self._promote_group_primary(promoted.id, remaining, current_user=current_user)

    def get_duplicate_group(self, record_id: str, current_user: AuthenticatedUser | None = None) -> DuplicateResolutionResponse:
        record = self.store.get(record_id, user_id=current_user.user_id if current_user is not None else None)
        if record is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Import record not found.")

        all_records = self.store.list_records(user_id=current_user.user_id if current_user is not None else None)
        group_records = self._group_records_for(record, all_records)
        if len(group_records) <= 1:
            catalog_conflict_ids = self._resolve_catalog_conflict_ids(record, current_user=current_user)
            if catalog_conflict_ids:
                primary = self._enrich_record(record, all_records)
                return DuplicateResolutionResponse(
                    kind="catalog_conflict",
                    primary=primary,
                    catalog_matches=self._catalog_conflict_matches(catalog_conflict_ids, current_user=current_user),
                )
            if record.duplicate_group_key:
                primary = self._enrich_record(record, all_records, include_catalog_fallback=False)
                return DuplicateResolutionResponse(kind="import_group", primary=primary, duplicates=[])
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No duplicate group found for this record.")
        primary = next((item for item in group_records if item.primary_record_id is None), record)
        primary = self._enrich_record(primary, all_records, include_catalog_fallback=False)
        duplicates = [
            self._enrich_record(item, all_records, include_catalog_fallback=False)
            for item in group_records
            if item.id != primary.id
        ]
        return DuplicateResolutionResponse(kind="import_group", primary=primary, duplicates=duplicates)

    def promote_duplicate_to_primary(self, record_id: str, current_user: AuthenticatedUser | None = None) -> DuplicateResolutionResponse:
        record = self.get_import(record_id, current_user=current_user)
        all_records = self.store.list_records(user_id=current_user.user_id if current_user is not None else None)
        group_records = self._group_records_for(record, all_records)
        if len(group_records) <= 1:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="No duplicate group found for this record.")
        return self._promote_group_primary(record.id, group_records, current_user=current_user)

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
        if record.primary_record_id is not None or record.duplicate_count > 0:
            raise HTTPException(
                status_code=status.HTTP_409_CONFLICT,
                detail="Resolve this duplicate group before uploading this import record as a product.",
            )
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
        all_records = self.store.list_records(user_id=current_user.user_id if current_user is not None else None)
        return UploadImportAsProductResponse(import_record=self._enrich_record(updated_import, all_records), product_record=product_record)

    def _save_row_as_import_record(
        self,
        *,
        filename: str,
        row: ImportedProductRow,
        primary_record_id: str | None = None,
        catalog_conflict_product_ids: list[str] | None = None,
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
        duplicate_group_key = self._fingerprint_for_row(row)
        mapped_status = self._status_from_row(row)
        notes = list(row.notes)
        if primary_record_id is not None and row.status != "parse_issue":
            mapped_status = "duplicate"
            notes = [note for note in notes if note != "Duplicate title found in uploaded file." and note != "Duplicate SKU found in uploaded file."]
            notes.append("This import record duplicates an earlier imported record.")
        catalog_conflict_ids = catalog_conflict_product_ids or []
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
            notes=notes,
            duplicate_group_key=duplicate_group_key,
            primary_record_id=primary_record_id,
            duplicate_count=0,
            can_upload_as_product=primary_record_id is None,
            catalog_conflict_product_ids=catalog_conflict_ids,
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
        notes = list(record.notes)
        next_status = self._normal_status_for(product=product, linked_product_id=record.linked_product_id)
        if record.primary_record_id is not None:
            next_status = "duplicate"

        return record.model_copy(
            update={
                "status": next_status,
                "missing_fields": missing_fields,
                "notes": notes,
                "product": product,
                "updated_at": self._timestamp(),
            }
        )

    def _promote_group_primary(
        self,
        primary_record_id: str,
        group_records: list[ImportRecordResponse],
        *,
        current_user: AuthenticatedUser | None = None,
    ) -> DuplicateResolutionResponse:
        primary = next((item for item in group_records if item.id == primary_record_id), None)
        if primary is None:
            raise HTTPException(status_code=status.HTTP_404_NOT_FOUND, detail="Primary duplicate candidate not found.")

        updated_records: list[ImportRecordResponse] = []
        for record in group_records:
            if record.id == primary.id:
                updated = record.model_copy(
                    update={
                        "primary_record_id": None,
                        "status": self._normal_status_for(product=record.product, linked_product_id=record.linked_product_id),
                        "notes": [note for note in record.notes if note != "This import record duplicates an earlier imported record."],
                        "updated_at": self._timestamp(),
                    }
                )
            else:
                notes = [note for note in record.notes if note != "This import record duplicates an earlier imported record."]
                notes.append("This import record duplicates an earlier imported record.")
                updated = record.model_copy(
                    update={
                        "primary_record_id": primary.id,
                        "status": "duplicate",
                        "notes": notes,
                        "updated_at": self._timestamp(),
                    }
                )
            self.store.save(updated, user_id=current_user.user_id if current_user is not None else None)
            updated_records.append(updated)

        primary_updated = next(item for item in updated_records if item.id == primary.id)
        duplicates = [item for item in updated_records if item.id != primary.id]
        return DuplicateResolutionResponse(kind="import_group", primary=primary_updated, duplicates=duplicates)

    def _build_list_items(
        self,
        records: list[ImportRecordResponse],
        *,
        current_user: AuthenticatedUser | None = None,
    ) -> list[ImportListItemResponse]:
        del current_user
        items: list[ImportListItemResponse] = []
        for record in records:
            enriched = self._enrich_record(record, records, include_catalog_fallback=False)
            items.append(
                ImportListItemResponse(
                    id=enriched.id,
                    status=enriched.status,
                    created_at=enriched.created_at,
                    updated_at=enriched.updated_at,
                    normalized_title=enriched.product.core.normalized_title,
                    category=enriched.product.core.category,
                    product_type=enriched.product.core.product_type,
                    preview_image_path=enriched.product.images.source.absolute_path or enriched.product.images.shopify.absolute_path,
                    linked_product_id=enriched.linked_product_id,
                    missing_fields=enriched.missing_fields,
                    notes=enriched.notes,
                    primary_record_id=enriched.primary_record_id,
                    duplicate_count=enriched.duplicate_count,
                    can_upload_as_product=enriched.can_upload_as_product,
                    catalog_conflict_product_ids=enriched.catalog_conflict_product_ids,
                )
            )
        return items

    def _group_records_for(self, record: ImportRecordResponse, records: list[ImportRecordResponse]) -> list[ImportRecordResponse]:
        primary_id = record.primary_record_id or record.id
        group_key = record.duplicate_group_key or self._fingerprint_for_record(record)
        grouped = [
            item
            for item in records
            if (item.id == primary_id or item.primary_record_id == primary_id)
            or (
                group_key
                and (
                    item.duplicate_group_key == group_key
                    or self._fingerprint_for_record(item) == group_key
                )
            )
        ]
        unique = {item.id: item for item in grouped}
        return sorted(unique.values(), key=lambda item: (item.created_at, item.id))

    def _enrich_record(
        self,
        record: ImportRecordResponse,
        records: list[ImportRecordResponse],
        *,
        catalog_product_ids_by_title: dict[str, list[str]] | None = None,
        include_catalog_fallback: bool = True,
    ) -> ImportRecordResponse:
        group_records = self._group_records_for(record, records)
        duplicate_count = max(len(group_records) - 1, 0)
        catalog_conflict_ids = self._resolve_catalog_conflict_ids(
            record,
            catalog_product_ids_by_title=catalog_product_ids_by_title,
            use_fallback_lookup=include_catalog_fallback,
        )
        can_upload_as_product = record.primary_record_id is None and duplicate_count == 0 and not record.linked_product_id and not catalog_conflict_ids
        return record.model_copy(
            update={
                "duplicate_count": duplicate_count,
                "can_upload_as_product": can_upload_as_product,
                "catalog_conflict_product_ids": catalog_conflict_ids,
            }
        )

    def _catalog_conflict_matches(
        self,
        product_ids: list[str],
        *,
        current_user: AuthenticatedUser | None = None,
    ) -> list[CatalogConflictProductResponse]:
        matches: list[CatalogConflictProductResponse] = []
        for product_id in product_ids:
            try:
                record = self.product_service.get_product(product_id, current_user=current_user)
            except HTTPException:
                continue
            matches.append(
                CatalogConflictProductResponse(
                    id=record.id,
                    status=record.status,
                    normalized_title=record.product.core.normalized_title,
                    category=record.product.core.category,
                    product_type=record.product.core.product_type,
                    preview_image_path=record.product.images.source.absolute_path or record.product.images.shopify.absolute_path,
                )
            )
        return matches

    def _resolve_catalog_conflict_ids(
        self,
        record: ImportRecordResponse,
        *,
        current_user: AuthenticatedUser | None = None,
        catalog_product_ids_by_title: dict[str, list[str]] | None = None,
        use_fallback_lookup: bool = True,
    ) -> list[str]:
        if record.catalog_conflict_product_ids:
            return record.catalog_conflict_product_ids

        if not use_fallback_lookup:
            return []

        normalized_title = record.product.core.normalized_title.strip().lower()
        if not normalized_title:
            return []

        if catalog_product_ids_by_title is None:
            catalog_product_ids_by_title = self._catalog_product_ids_by_title(current_user=current_user)

        return catalog_product_ids_by_title.get(normalized_title, [])

    def _catalog_product_ids_by_title(
        self,
        *,
        current_user: AuthenticatedUser | None = None,
    ) -> dict[str, list[str]]:
        catalog: dict[str, list[str]] = {}
        for product in self.product_service.list_products(current_user=current_user):
            key = product.normalized_title.strip().lower()
            if not key:
                continue
            catalog.setdefault(key, []).append(product.id)
        return catalog

    @staticmethod
    def _fingerprint_for_row(row: ImportedProductRow) -> str | None:
        title = row.title.strip().lower()
        sku = row.sku.strip().lower()
        if not title and not sku:
            return None
        return f"{title}::{sku}"

    @staticmethod
    def _fingerprint_for_record(record: ImportRecordResponse) -> str | None:
        title = record.product.core.normalized_title.strip().lower()
        sku = str(record.product.core.attributes.get("sku", "")).strip().lower()
        if not title and not sku:
            return None
        return f"{title}::{sku}"

    @staticmethod
    def _status_from_row(row: ImportedProductRow) -> str:
        if row.status == "parse_issue":
            return "parse_issue"
        if row.status == "ready":
            return "imported"
        return "needs_review"

    @staticmethod
    def _normal_status_for(*, product: ProductPipelineResponse, linked_product_id: str | None) -> str:
        if linked_product_id:
            return "uploaded"
        if not product.core.normalized_title.strip():
            return "parse_issue"
        if ImportService._build_missing_fields(product):
            return "needs_review"
        return "imported"

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
