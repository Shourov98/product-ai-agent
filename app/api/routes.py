from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, HTTPException, UploadFile, status

from app.auth import AuthenticatedUser, get_optional_current_user
from app.config import get_settings
from app.orchestrator.pipeline import ProductPipeline
from app.schemas.repricing import ProductRepricingRequest, ProductRepricingResponse
from app.schemas.request import MarketplaceRequestLiteral, ProductOptimizationRequest, ProductUpdateRequest, VariantCreateRequest
from app.schemas.response import ProductListItemResponse, ProductPipelineResponse, ProductRecordResponse
from app.services.image_service import ImagePayload
from app.services.product_service import ProductService

router = APIRouter()


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

    payload = ImagePayload(
        filename=image.filename or "upload.bin",
        content_type=image.content_type or "application/octet-stream",
        data=content,
    )

    pipeline = ProductPipeline()
    return await pipeline.run(payload, title)


@router.post(
    "/products/generate",
    response_model=ProductRecordResponse,
    status_code=status.HTTP_200_OK,
)
async def generate_and_create_product(
    title: str = Form(...),
    image: UploadFile = File(...),
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ProductRecordResponse:
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

    payload = ImagePayload(
        filename=image.filename or "upload.bin",
        content_type=image.content_type or "application/octet-stream",
        data=content,
    )
    service = ProductService()
    return await service.generate_and_store(payload, title, current_user=current_user)


@router.get(
    "/products",
    response_model=list[ProductListItemResponse],
    status_code=status.HTTP_200_OK,
)
async def list_products(
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> list[ProductListItemResponse]:
    service = ProductService()
    return service.list_products(current_user=current_user)


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


@router.post(
    "/products/{product_id}/repricing",
    response_model=ProductRepricingResponse,
    status_code=status.HTTP_200_OK,
)
async def analyze_product_repricing(
    product_id: str,
    payload: ProductRepricingRequest,
    current_user: AuthenticatedUser | None = Depends(get_optional_current_user),
) -> ProductRepricingResponse:
    service = ProductService()
    return await service.analyze_product_repricing(product_id, payload, current_user=current_user)


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
