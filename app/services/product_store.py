from __future__ import annotations

import json
from pathlib import Path
from typing import Any
from math import ceil

from pydantic import ValidationError

from app.schemas.response import PaginatedProductListResponse, PaginationMetaResponse, ProductListItemResponse, ProductRecordResponse


class ProductStore:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: ProductRecordResponse, *, user_id: str | None = None) -> None:
        del user_id
        run_dir = (self.base_dir / record.run_id).resolve()
        run_dir.mkdir(parents=True, exist_ok=True)
        path = run_dir / "record.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(record.model_dump(mode="json"), handle, indent=2, ensure_ascii=True)

    def get(self, product_id: str, *, user_id: str | None = None) -> ProductRecordResponse | None:
        del user_id
        path = self._find_record_path(product_id)
        if path is None:
            return None
        return self._load_record(path)

    def list(self, *, user_id: str | None = None) -> list[ProductListItemResponse]:
        del user_id
        records: list[ProductListItemResponse] = []
        for path in sorted(self.base_dir.glob("*/record.json")):
            try:
                record = self._load_record(path)
            except ValidationError:
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
                )
            )
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

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

    def get_product_dir(self, product_id: str) -> Path:
        record_path = self._find_record_path(product_id)
        if record_path is not None:
            return record_path.parent

        legacy_dir = (self.base_dir / "products" / product_id).resolve()
        if legacy_dir.exists():
            return legacy_dir

        raise FileNotFoundError(f"Could not locate product directory for {product_id}.")

    def _find_record_path(self, product_id: str) -> Path | None:
        for path in sorted(self.base_dir.glob("*/record.json")):
            record = self._load_record(path)
            if record.id == product_id:
                return path

        legacy_path = (self.base_dir / "products" / product_id / "record.json").resolve()
        if legacy_path.exists():
            return legacy_path

        return None

    @staticmethod
    def _load_record(path: Path) -> ProductRecordResponse:
        with path.open("r", encoding="utf-8") as handle:
            payload: Any = json.load(handle)
        return ProductRecordResponse.model_validate(payload)
