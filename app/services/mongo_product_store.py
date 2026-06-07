from __future__ import annotations

from pathlib import Path
from typing import Any

from pymongo import DESCENDING, MongoClient

from app.schemas.response import ProductListItemResponse, ProductRecordResponse


class MongoProductStore:
    def __init__(
        self,
        *,
        mongodb_uri: str,
        db_name: str,
        products_collection: str,
    ) -> None:
        self.client = MongoClient(mongodb_uri)
        self.database = self.client[db_name]
        self.collection = self.database[products_collection]
        self.collection.create_index([("id", 1)], unique=True)
        self.collection.create_index([("user_id", 1), ("updated_at", -1)])

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

        records: list[ProductListItemResponse] = []
        for payload in self.collection.find(query).sort("updated_at", DESCENDING):
            payload.pop("_id", None)
            payload.pop("user_id", None)
            record = ProductRecordResponse.model_validate(payload)
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
                )
            )
        return records

    def get_product_dir(self, product_id: str) -> Path:
        payload = self.collection.find_one({"id": product_id}, {"run_id": 1})
        run_id = str(payload.get("run_id", product_id)) if payload else product_id
        return Path("/tmp") / run_id
