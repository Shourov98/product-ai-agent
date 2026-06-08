from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from pydantic import ValidationError

from app.schemas.imports import ImportListItemResponse, ImportRecordResponse


class ImportStore:
    def __init__(self, base_dir: str) -> None:
        self.base_dir = Path(base_dir).resolve()
        self.base_dir.mkdir(parents=True, exist_ok=True)

    def save(self, record: ImportRecordResponse, *, user_id: str | None = None) -> None:
        del user_id
        path = self.base_dir / f"{record.id}.json"
        with path.open("w", encoding="utf-8") as handle:
            json.dump(record.model_dump(mode="json"), handle, indent=2, ensure_ascii=True)

    def get(self, record_id: str, *, user_id: str | None = None) -> ImportRecordResponse | None:
        del user_id
        path = self.base_dir / f"{record_id}.json"
        if not path.exists():
            return None
        return self._load(path)

    def list(self, *, user_id: str | None = None) -> list[ImportListItemResponse]:
        del user_id
        records: list[ImportListItemResponse] = []
        for path in sorted(self.base_dir.glob("*.json")):
            try:
                record = self._load(path)
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
        return sorted(records, key=lambda item: item.updated_at, reverse=True)

    def delete(self, record_id: str, *, user_id: str | None = None) -> None:
        del user_id
        path = self.base_dir / f"{record_id}.json"
        if path.exists():
            path.unlink()

    @staticmethod
    def _load(path: Path) -> ImportRecordResponse:
        with path.open("r", encoding="utf-8") as handle:
            payload: Any = json.load(handle)
        return ImportRecordResponse.model_validate(payload)
