from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, Query, UploadFile, status

from app.auth import AuthenticatedUser, get_optional_current_user
from app.config import get_settings
from app.orchestrator.pipeline import ProductPipeline
from app.schemas.imports import DuplicateResolutionResponse, ImportRecordResponse, ImportUploadResponse, PaginatedImportListResponse, UploadImportAsProductResponse
from app.schemas.request import (
    MarketplaceRequestLiteral,
    ProductOptimizationRequest,
    ProductUpdateRequest,
    VariantCreateRequest,
)
from app.schemas.response import (
    PaginatedProductListResponse,
    ProductPipelineResponse,
    ProductPricingSnapshotResponse,
    ProductRecordResponse,
)
from app.services.image_service import ImagePayload
from app.services.import_service import ImportService
from app.services.product_service import ProductService

router = APIRouter()


async def _read_image_payload(image: UploadFile) -> ImagePayload:
    settings = get_settings()
    content = await image.read()

    if not content:
        raise HTTPException(
            status_code=status.HTTP_400_BAD_REQUEST,
            detail="Image upload is empty.",
        )

    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds configured upload limit.",
        )

    return ImagePayload(
        filename=image.filename or "upload.bin",
        content_type=image.content_type or "application/octet-stream",
        data=content,
    )


@router.get("/health")
async def healthcheck() -> dict[str, str]:
    return {"status": "ok"}


@router.post(
    "/generate",
    response_model=ProductPipelineResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_product_listing(
    title: str = Form(...),
    image: UploadFile = File(...),
) -> ProductPipelineResponse:
    payload = await _read_image_payload(image)
    pipeline = ProductPipeline()
    return await pipeline.run(payload, title)


@router.post(
    "/products/generate/text",
    response_model=ProductRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_text_only_product(
    title: str = Form(...),
    image: UploadFile = File(...),
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ProductRecordResponse:
    payload = await _read_image_payload(image)
    service = ProductService()
    return await service.generate_text_only(payload, title, current_user=current_user)


@router.post(
    "/products/generate/{marketplace}",
    response_model=ProductRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_single_marketplace_product(
    marketplace: MarketplaceRequestLiteral,
    title: str = Form(...),
    image: UploadFile = File(...),
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ProductRecordResponse:
    payload = await _read_image_payload(image)
    service = ProductService()
    return await service.generate_marketplace_draft(payload, title, marketplace, current_user=current_user)


@router.post(
    "/imports/products/upload",
    response_model=ImportUploadResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_product_import(
    file: UploadFile = File(...),
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ImportUploadResponse:
    content = await file.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Uploaded import file is empty.")
    service = ImportService()
    return service.upload_file(filename=file.filename or "import.bin", payload=content, current_user=current_user)


@router.get(
    "/imports/products",
    response_model=PaginatedImportListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_product_imports(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> PaginatedImportListResponse:
    service = ImportService()
    return service.list_imports_paginated(page=page, page_size=page_size, current_user=current_user)


@router.get(
    "/imports/products/{record_id}",
    response_model=ImportRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def get_product_import(
    record_id: str,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ImportRecordResponse:
    service = ImportService()
    return service.get_import(record_id, current_user=current_user)


@router.post(
    "/imports/products/{record_id}/source-image",
    response_model=ImportRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_product_import_source_image(
    record_id: str,
    image: UploadFile = File(...),
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ImportRecordResponse:
    settings = get_settings()
    content = await image.read()
    if not content:
        raise HTTPException(status_code=status.HTTP_400_BAD_REQUEST, detail="Image upload is empty.")
    if len(content) > settings.max_upload_size_bytes:
        raise HTTPException(
            status_code=status.HTTP_413_REQUEST_ENTITY_TOO_LARGE,
            detail="Image exceeds configured upload limit.",
        )
    service = ImportService()
    return await service.upload_source_image(
        record_id,
        ImagePayload(
            filename=image.filename or "upload.bin",
            content_type=image.content_type or "application/octet-stream",
            data=content,
        ),
        current_user=current_user,
    )


@router.patch(
    "/imports/products/{record_id}",
    response_model=ImportRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def update_product_import(
    record_id: str,
    payload: ProductUpdateRequest,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ImportRecordResponse:
    service = ImportService()
    return service.update_import(record_id, payload, current_user=current_user)


@router.delete(
    "/imports/products/{record_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_product_import(
    record_id: str,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
):
    service = ImportService()
    service.delete_import(record_id, current_user=current_user)
    return None


@router.post(
    "/imports/products/{record_id}/optimize",
    response_model=ImportRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def optimize_product_import(
    record_id: str,
    payload: ProductOptimizationRequest,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ImportRecordResponse:
    service = ImportService()
    return await service.optimize_import(record_id, payload, current_user=current_user)


@router.post(
    "/imports/products/{record_id}/marketplaces/{marketplace}/optimize",
    response_model=ImportRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def optimize_product_import_marketplace(
    record_id: str,
    marketplace: MarketplaceRequestLiteral,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ImportRecordResponse:
    service = ImportService()
    return await service.optimize_import_marketplace(record_id, marketplace, current_user=current_user)


@router.post(
    "/imports/products/{record_id}/marketplaces/{marketplace}/regenerate",
    response_model=ImportRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def regenerate_product_import_marketplace(
    record_id: str,
    marketplace: MarketplaceRequestLiteral,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ImportRecordResponse:
    service = ImportService()
    return await service.regenerate_import_marketplace(record_id, marketplace, current_user=current_user)


@router.post(
    "/imports/products/{record_id}/upload-as-product",
    response_model=UploadImportAsProductResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_product_import_as_product(
    record_id: str,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> UploadImportAsProductResponse:
    service = ImportService()
    return service.upload_as_product(record_id, current_user=current_user)


@router.get(
    "/imports/products/{record_id}/duplicates",
    response_model=DuplicateResolutionResponse,
    status_code=status.HTTP_200_OK,
)
async def get_product_import_duplicates(
    record_id: str,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> DuplicateResolutionResponse:
    service = ImportService()
    return service.get_duplicate_group(record_id, current_user=current_user)


@router.post(
    "/imports/products/{record_id}/duplicates/promote",
    response_model=DuplicateResolutionResponse,
    status_code=status.HTTP_200_OK,
)
async def promote_product_import_duplicate(
    record_id: str,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> DuplicateResolutionResponse:
    service = ImportService()
    return service.promote_duplicate_to_primary(record_id, current_user=current_user)


@router.post(
    "/imports/products/{record_id}/duplicates/delete-all",
    response_model=DuplicateResolutionResponse,
    status_code=status.HTTP_200_OK,
)
async def delete_all_product_import_duplicates(
    record_id: str,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> DuplicateResolutionResponse:
    service = ImportService()
    return service.delete_all_duplicates(record_id, current_user=current_user)


@router.get(
    "/products",
    response_model=PaginatedProductListResponse,
    status_code=status.HTTP_200_OK,
)
async def list_products(
    page: int = Query(default=1, ge=1),
    page_size: int = Query(default=20, ge=1, le=100),
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> PaginatedProductListResponse:
    service = ProductService()
    return service.list_products_paginated(page=page, page_size=page_size, current_user=current_user)


@router.get(
    "/products/{product_id}",
    response_model=ProductRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def get_product(
    product_id: str,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ProductRecordResponse:
    service = ProductService()
    return service.get_product(product_id, current_user=current_user)


@router.patch(
    "/products/{product_id}",
    response_model=ProductRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def update_product(
    product_id: str,
    payload: ProductUpdateRequest,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ProductRecordResponse:
    service = ProductService()
    return service.update_product(product_id, payload, current_user=current_user)


@router.delete(
    "/products/{product_id}",
    status_code=status.HTTP_204_NO_CONTENT,
)
async def delete_product(
    product_id: str,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
):
    service = ProductService()
    service.delete_product(product_id, current_user=current_user)
    return None


@router.post(
    "/products/{product_id}/source-image",
    response_model=ProductRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def upload_product_source_image(
    product_id: str,
    image: UploadFile = File(...),
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ProductRecordResponse:
    service = ProductService()
    return await service.upload_source_image(
        product_id,
        await _read_image_payload(image),
        current_user=current_user,
    )


@router.post(
    "/products/{product_id}/optimize",
    response_model=ProductRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def optimize_product(
    product_id: str,
    payload: ProductOptimizationRequest,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ProductRecordResponse:
    service = ProductService()
    return await service.optimize_product(product_id, payload, current_user=current_user)


@router.post(
    "/products/{product_id}/marketplaces/{marketplace}/optimize",
    response_model=ProductRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def optimize_marketplace(
    product_id: str,
    marketplace: MarketplaceRequestLiteral,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ProductRecordResponse:
    service = ProductService()
    return await service.optimize_marketplace(product_id, marketplace, current_user=current_user)


@router.get(
    "/products/{product_id}/pricing/snapshot",
    response_model=ProductPricingSnapshotResponse,
    status_code=status.HTTP_200_OK,
)
async def get_product_pricing_snapshot(
    product_id: str,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ProductPricingSnapshotResponse:
    service = ProductService()
    return await service.get_product_pricing_snapshot(product_id, current_user=current_user)


@router.post(
    "/products/{product_id}/marketplaces/{marketplace}/variants/size",
    response_model=ProductRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def add_size_variant(
    product_id: str,
    marketplace: MarketplaceRequestLiteral,
    payload: VariantCreateRequest,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ProductRecordResponse:
    service = ProductService()
    return service.add_size_variant(product_id, marketplace, payload, current_user=current_user)


@router.post(
    "/products/{product_id}/marketplaces/{marketplace}/variants/color",
    response_model=ProductRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def add_color_variant(
    product_id: str,
    marketplace: MarketplaceRequestLiteral,
    payload: VariantCreateRequest,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ProductRecordResponse:
    service = ProductService()
    return service.add_color_variant(product_id, marketplace, payload, current_user=current_user)


@router.post(
    "/products/{product_id}/marketplaces/{marketplace}/generate",
    response_model=ProductRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_marketplace_section(
    product_id: str,
    marketplace: MarketplaceRequestLiteral,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ProductRecordResponse:
    service = ProductService()
    return await service.regenerate_marketplace(product_id, marketplace, current_user=current_user)


@router.post(
    "/products/{product_id}/marketplaces/{marketplace}/regenerate",
    response_model=ProductRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def regenerate_marketplace(
    product_id: str,
    marketplace: MarketplaceRequestLiteral,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ProductRecordResponse:
    service = ProductService()
    return await service.regenerate_marketplace(product_id, marketplace, current_user=current_user)
