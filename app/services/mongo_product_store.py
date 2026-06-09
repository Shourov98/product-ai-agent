from __future__ import annotations

from pathlib import Path
from typing import Any
from math import ceil

from pymongo import DESCENDING, MongoClient
from pydantic import ValidationError

from app.schemas.response import PaginatedProductListResponse, PaginationMetaResponse, ProductListItemResponse, ProductRecordResponse


class MongoProductStore:
    _initialized_keys: set[tuple[str, str, str, str | None]] = set()

    def __init__(
        self,
        *,
        mongodb_uri: str,
        db_name: str,
        products_collection: str,
        imports_collection: str | None = None,
    ) -> None:
        self.client = MongoClient(mongodb_uri)
        self.database = self.client[db_name]
        self.collection = self.database[products_collection]
        self.imports_collection = self.database[imports_collection] if imports_collection else None
        init_key = (mongodb_uri, db_name, products_collection, imports_collection)
        if init_key not in self._initialized_keys:
            self.collection.create_index([("id", 1)], unique=True)
            self.collection.create_index([("user_id", 1), ("updated_at", -1)])
            self._initialized_keys.add(init_key)

    def save(self, record: ProductRecordResponse, *, user_id: str | None = None) -> None:
        payload = record.model_dump(mode="json")
        payload["user_id"] = user_id
        self.collection.replace_one({"id": record.id}, payload, upsert=True)

    def get(self, product_id: str, *, user_id: str | None = None) -> ProductRecordResponse | None:
        query: dict[str, Any] = {"id": product_id}
        if user_id is not None:
            query["user_id"] = user_id
        payload = self.collection.find_one(query)
        if payload is None:
            return None
        payload.pop("_id", None)
        payload.pop("user_id", None)
        return ProductRecordResponse.model_validate(payload)

    def list(self, *, user_id: str | None = None) -> list[ProductListItemResponse]:
        query: dict[str, Any] = {}
        if user_id is not None:
            query["user_id"] = user_id

        linked_product_ids, pending_import_fingerprints = self._import_visibility_filters(user_id=user_id)
        records: list[ProductListItemResponse] = []
        for payload in self.collection.find(query).sort("updated_at", DESCENDING):
            payload.pop("_id", None)
            payload.pop("user_id", None)
            try:
                record = ProductRecordResponse.model_validate(payload)
            except ValidationError:
                continue
            if self._should_hide_record(record, linked_product_ids, pending_import_fingerprints):
                continue
            records.append(
                ProductListItemResponse(
                    id=record.id,
                    status=record.status,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    normalized_title=record.product.core.normalized_title,
                    category=record.product.core.category,
                    product_type=record.product.core.product_type,
                    preview_image_path=record.product.images.shopify.absolute_path,
                    default_price=self._default_price_for_record(record),
                )
            )
        return records

    def list_paginated(self, *, page: int, page_size: int, user_id: str | None = None) -> PaginatedProductListResponse:
        records = self.list(user_id=user_id)
        total_items = len(records)
        total_pages = ceil(total_items / page_size) if total_items else 0
        start = (page - 1) * page_size
        end = start + page_size
        return PaginatedProductListResponse(
            items=records[start:end],
            pagination=PaginationMetaResponse(
                page=page,
                page_size=page_size,
                total_items=total_items,
                total_pages=total_pages,
            ),
        )

    def _import_visibility_filters(self, *, user_id: str | None = None) -> tuple[set[str], set[tuple[str, str]]]:
        if self.imports_collection is None:
            return set(), set()

        query: dict[str, Any] = {}
        if user_id is not None:
            query["user_id"] = user_id

        linked_product_ids: set[str] = set()
        pending_import_fingerprints: set[tuple[str, str]] = set()
        for payload in self.imports_collection.find(query, {"linked_product_id": 1, "status": 1, "product.core.normalized_title": 1, "product.core.attributes": 1}):
            linked_product_id = payload.get("linked_product_id")
            if isinstance(linked_product_id, str) and linked_product_id.strip():
                linked_product_ids.add(linked_product_id.strip())

            if payload.get("status") == "uploaded":
                continue

            product = payload.get("product", {})
            core = product.get("core", {})
            attributes = core.get("attributes", {}) or {}
            title = str(core.get("normalized_title", "")).strip().lower()
            sku = str(attributes.get("sku", "")).strip().lower()
            if title:
                pending_import_fingerprints.add((title, sku))

        return linked_product_ids, pending_import_fingerprints

    @staticmethod
    def _should_hide_record(
        record: ProductRecordResponse,
        linked_product_ids: set[str],
        pending_import_fingerprints: set[tuple[str, str]],
    ) -> bool:
        if record.id in linked_product_ids:
            return False

        title = record.product.core.normalized_title.strip().lower()
        sku = str(record.product.core.attributes.get("sku", "")).strip().lower()
        return bool(title) and (title, sku) in pending_import_fingerprints

    def get_product_dir(self, product_id: str) -> Path:
        payload = self.collection.find_one({"id": product_id}, {"run_id": 1})
        run_id = str(payload.get("run_id", product_id)) if payload else product_id
        return Path("/tmp") / run_id

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _default_price_for_record(record: ProductRecordResponse) -> str | None:
        raw_price = record.product.core.attributes.get("price")
        if raw_price is not None:
            price = str(raw_price).strip()
            if price:
                return price

        recommended = record.product.intelligence.pricing.shopify.recommended
        if recommended > 0:
            return f"{recommended:.2f}"

        return None
