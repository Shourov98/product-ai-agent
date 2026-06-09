from __future__ import annotations

from math import ceil
from typing import Any

from pymongo import DESCENDING, MongoClient
from pydantic import ValidationError

from app.schemas.imports import ImportListItemResponse, ImportOverviewResponse, ImportRecordResponse, PaginatedImportListResponse
from app.schemas.response import PaginationMetaResponse


class MongoImportStore:
    _initialized_keys: set[tuple[str, str, str]] = set()
    _LIST_PROJECTION = {
        "_id": 0,
        "id": 1,
        "status": 1,
        "created_at": 1,
        "updated_at": 1,
        "linked_product_id": 1,
        "missing_fields": 1,
        "notes": 1,
        "primary_record_id": 1,
        "duplicate_count": 1,
        "can_upload_as_product": 1,
        "catalog_conflict_product_ids": 1,
        "product.core.normalized_title": 1,
        "product.core.category": 1,
        "product.core.product_type": 1,
        "product.images.source.absolute_path": 1,
        "product.images.shopify.absolute_path": 1,
    }

    def __init__(self, *, mongodb_uri: str, db_name: str, imports_collection: str) -> None:
        self.client = MongoClient(mongodb_uri)
        self.database = self.client[db_name]
        self.collection = self.database[imports_collection]
        init_key = (mongodb_uri, db_name, imports_collection)
        if init_key not in self._initialized_keys:
            self.collection.create_index([("id", 1)], unique=True)
            self.collection.create_index([("user_id", 1), ("updated_at", -1)])
            self._initialized_keys.add(init_key)

    def save(self, record: ImportRecordResponse, *, user_id: str | None = None) -> None:
        payload = record.model_dump(mode="json")
        payload["user_id"] = user_id
        self.collection.replace_one({"id": record.id}, payload, upsert=True)

    def get(self, record_id: str, *, user_id: str | None = None) -> ImportRecordResponse | None:
        query: dict[str, Any] = {"id": record_id}
        if user_id is not None:
            query["user_id"] = user_id
        payload = self.collection.find_one(query)
        if payload is None:
            return None
        payload.pop("_id", None)
        payload.pop("user_id", None)
        return ImportRecordResponse.model_validate(payload)

    def list_records(self, *, user_id: str | None = None) -> list[ImportRecordResponse]:
        query: dict[str, Any] = {}
        if user_id is not None:
            query["user_id"] = user_id
        records: list[ImportRecordResponse] = []
        for payload in self.collection.find(query).sort("updated_at", DESCENDING):
            payload.pop("_id", None)
            payload.pop("user_id", None)
            try:
                records.append(ImportRecordResponse.model_validate(payload))
            except ValidationError:
                continue
        return records

    def list(self, *, user_id: str | None = None) -> list[ImportListItemResponse]:
        query: dict[str, Any] = {}
        if user_id is not None:
            query["user_id"] = user_id
        return [
            self._list_item_from_payload(payload)
            for payload in self.collection.find(query, self._LIST_PROJECTION).sort("updated_at", DESCENDING)
        ]

    def list_paginated(self, *, page: int, page_size: int, user_id: str | None = None) -> PaginatedImportListResponse:
        query: dict[str, Any] = {}
        if user_id is not None:
            query["user_id"] = user_id
        total_items = self.collection.count_documents(query)
        total_pages = ceil(total_items / page_size) if total_items else 0
        start = (page - 1) * page_size
        end = start + page_size
        items = [
            self._list_item_from_payload(payload)
            for payload in self.collection.find(query, self._LIST_PROJECTION).sort("updated_at", DESCENDING).skip(start).limit(page_size)
        ]
        return PaginatedImportListResponse(
            items=items,
            pagination=PaginationMetaResponse(
                page=page,
                page_size=page_size,
                total_items=total_items,
                total_pages=total_pages,
            ),
            summary=self._build_summary(query, total_items=total_items),
        )

    def delete(self, record_id: str, *, user_id: str | None = None) -> None:
        query: dict[str, Any] = {"id": record_id}
        if user_id is not None:
            query["user_id"] = user_id
        self.collection.delete_one(query)

    def close(self) -> None:
        self.client.close()

    @staticmethod
    def _list_item_from_payload(payload: dict[str, Any]) -> ImportListItemResponse:
        product = payload.get("product") or {}
        core = product.get("core") or {}
        images = product.get("images") or {}
        source = images.get("source") or {}
        shopify = images.get("shopify") or {}
        return ImportListItemResponse(
            id=str(payload.get("id", "")),
            status=str(payload.get("status", "imported")),
            created_at=str(payload.get("created_at", "")),
            updated_at=str(payload.get("updated_at", "")),
            normalized_title=str(core.get("normalized_title", "")),
            category=str(core.get("category", "")),
            product_type=str(core.get("product_type", "")),
            preview_image_path=str(source.get("absolute_path") or shopify.get("absolute_path") or ""),
            linked_product_id=payload.get("linked_product_id"),
            missing_fields=list(payload.get("missing_fields") or []),
            notes=list(payload.get("notes") or []),
            primary_record_id=payload.get("primary_record_id"),
            duplicate_count=int(payload.get("duplicate_count") or 0),
            can_upload_as_product=bool(payload.get("can_upload_as_product", True)),
            catalog_conflict_product_ids=list(payload.get("catalog_conflict_product_ids") or []),
        )

    def _build_summary(self, query: dict[str, Any], *, total_items: int) -> ImportOverviewResponse:
        counts_by_status = {
            str(entry.get("_id")): int(entry.get("count", 0))
            for entry in self.collection.aggregate(
                [
                    {"$match": query},
                    {"$group": {"_id": "$status", "count": {"$sum": 1}}},
                ]
            )
        }
        return ImportOverviewResponse(
            total_imported=total_items,
            uploaded_as_product=counts_by_status.get("uploaded", 0),
            needs_review=counts_by_status.get("needs_review", 0) + counts_by_status.get("parse_issue", 0),
            duplicates=counts_by_status.get("duplicate", 0),
        )
