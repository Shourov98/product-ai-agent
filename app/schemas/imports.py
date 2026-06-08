from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field

from app.schemas.response import MarketplaceVariantsResponse, PaginationMetaResponse, ProductPipelineResponse, ProductRecordResponse


ImportSourceType = Literal["csv", "excel", "pdf"]
ImportStatusType = Literal["imported", "needs_review", "duplicate", "parse_issue", "uploaded"]


class ImportedProductRow(BaseModel):
    row_id: str
    source_type: ImportSourceType
    source_reference: str
    title: str = ""
    sku: str = ""
    brand: str = ""
    category: str = ""
    product_type: str = ""
    description: str = ""
    price: str = ""
    quantity: str = ""
    color: str = ""
    size: str = ""
    material: str = ""
    image_url: str = ""
    status: Literal["ready", "missing_data", "duplicate", "parse_issue"] = "ready"
    confidence: float = Field(default=0.0, ge=0.0, le=1.0)
    missing_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)


class ImportPreviewResponse(BaseModel):
    filename: str
    source_type: ImportSourceType
    total_rows: int = Field(ge=0)
    ready_rows: int = Field(ge=0)
    duplicate_rows: int = Field(ge=0)
    parse_issue_rows: int = Field(ge=0)
    rows: list[ImportedProductRow] = Field(default_factory=list)


class ImportUploadResponse(BaseModel):
    imported_count: int = Field(ge=0)
    records: list["ImportRecordResponse"] = Field(default_factory=list)


class ImportProductUpdateRequest(BaseModel):
    core: dict | None = None
    amazon: dict | None = None
    ebay: dict | None = None
    etsy: dict | None = None
    tiktok: dict | None = None
    shopify: dict | None = None


class ImportRecordResponse(BaseModel):
    id: str
    status: ImportStatusType
    created_at: str
    updated_at: str
    source_filename: str
    source_type: ImportSourceType
    source_reference: str
    confidence: float = Field(ge=0.0, le=1.0)
    missing_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    duplicate_group_key: str | None = None
    primary_record_id: str | None = None
    duplicate_count: int = Field(default=0, ge=0)
    can_upload_as_product: bool = True
    linked_product_id: str | None = None
    product: ProductPipelineResponse
    variants: MarketplaceVariantsResponse = Field(default_factory=MarketplaceVariantsResponse)


class ImportListItemResponse(BaseModel):
    id: str
    status: ImportStatusType
    created_at: str
    updated_at: str
    normalized_title: str
    category: str
    product_type: str
    preview_image_path: str
    linked_product_id: str | None = None
    missing_fields: list[str] = Field(default_factory=list)
    notes: list[str] = Field(default_factory=list)
    primary_record_id: str | None = None
    duplicate_count: int = Field(default=0, ge=0)
    can_upload_as_product: bool = True


class UploadImportAsProductResponse(BaseModel):
    import_record: ImportRecordResponse
    product_record: ProductRecordResponse


class DuplicateGroupResponse(BaseModel):
    primary: ImportRecordResponse
    duplicates: list[ImportRecordResponse] = Field(default_factory=list)


class PaginatedImportListResponse(BaseModel):
    items: list[ImportListItemResponse] = Field(default_factory=list)
    pagination: PaginationMetaResponse
