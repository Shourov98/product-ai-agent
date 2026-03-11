from __future__ import annotations

from fastapi import APIRouter, File, Form, HTTPException, UploadFile, status

from app.config import get_settings
from app.orchestrator.pipeline import ProductPipeline
from app.schemas.response import ProductPipelineResponse
from app.services.image_service import ImagePayload

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
