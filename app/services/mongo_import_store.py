from __future__ import annotations

from typing import Any

from pymongo import DESCENDING, MongoClient
from pydantic import ValidationError

from app.schemas.imports import ImportListItemResponse, ImportRecordResponse


class MongoImportStore:
    def __init__(self, *, mongodb_uri: str, db_name: str, imports_collection: str) -> None:
        self.client = MongoClient(mongodb_uri)
        self.database = self.client[db_name]
        self.collection = self.database[imports_collection]
        self.collection.create_index([("id", 1)], unique=True)
        self.collection.create_index([("user_id", 1), ("updated_at", -1)])

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

    def list(self, *, user_id: str | None = None) -> list[ImportListItemResponse]:
        query: dict[str, Any] = {}
        if user_id is not None:
            query["user_id"] = user_id
        records: list[ImportListItemResponse] = []
        for payload in self.collection.find(query).sort("updated_at", DESCENDING):
            payload.pop("_id", None)
            payload.pop("user_id", None)
            try:
                record = ImportRecordResponse.model_validate(payload)
            except ValidationError:
                continue
            records.append(
                ImportListItemResponse(
                    id=record.id,
                    status=record.status,
                    created_at=record.created_at,
                    updated_at=record.updated_at,
                    normalized_title=record.product.core.normalized_title,
                    category=record.product.core.category,
                    product_type=record.product.core.product_type,
                    preview_image_path=record.product.images.source.absolute_path or record.product.images.shopify.absolute_path,
                    linked_product_id=record.linked_product_id,
                    missing_fields=record.missing_fields,
                    notes=record.notes,
                )
            )
        return records

    def delete(self, record_id: str, *, user_id: str | None = None) -> None:
        query: dict[str, Any] = {"id": record_id}
        if user_id is not None:
            query["user_id"] = user_id
        self.collection.delete_one(query)
