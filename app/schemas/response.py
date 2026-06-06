from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ExtractedAttribute(BaseModel):
    name: str
    value: str
    confidence: float = Field(ge=0.0, le=1.0)


class ImageAnalysis(BaseModel):
    filename: str
    content_type: str
    size_bytes: int = Field(ge=0)
    brightness: Literal["light", "balanced", "dark"]
    dominant_palette: list[str]
    background_removed: bool


class VisionResponse(BaseModel):
    product_type: str
    confidence: float = Field(ge=0.0, le=1.0)
    attributes: list[ExtractedAttribute]
    image_analysis: ImageAnalysis


class CoreProductResponse(BaseModel):
    normalized_title: str
    category: str
    product_type: str
    product_summary: str
    features: list[str]
    attributes: dict[str, str]
    source_title: str
    vision_confidence: float = Field(ge=0.0, le=1.0)


class AmazonResponse(BaseModel):
    title: str
    bullet_points: list[str]
    description: str
    backend_search_terms: list[str]
    structured_attributes: dict[str, str]


class TikTokResponse(BaseModel):
    title: str
    social_description: str
    hashtags: list[str]


class EbayResponse(BaseModel):
    title: str = Field(max_length=80)
    item_specifics: dict[str, str]
    condition: str
    listing_notes: str


class ShopifyResponse(BaseModel):
    title: str
    body_html: str
    tags: list[str]
    product_type: str
    seo_title: str
    seo_description: str


class ImageValidationResponse(BaseModel):
    passed: bool
    width: int | None = None
    height: int | None = None
    format: str | None = None
    has_alpha: bool | None = None
    file_size_bytes: int = Field(ge=0)
    expected_width: int | None = None
    expected_height: int | None = None
    expected_background: str
    errors: list[str]
    mime_type: str


class ImageVariantResponse(BaseModel):
    marketplace: str
    relative_path: str
    absolute_path: str
    prompt: str
    generation_mode: str
    mime_type: str
    validation: ImageValidationResponse


class GeneratedImagesResponse(BaseModel):
    source: ImageVariantResponse
    transparent_cutout: ImageVariantResponse | None = None
    amazon: ImageVariantResponse
    ebay: ImageVariantResponse
    tiktok: ImageVariantResponse
    shopify: ImageVariantResponse


class ProductPipelineResponse(BaseModel):
    core: CoreProductResponse
    amazon: AmazonResponse
    tiktok: TikTokResponse
    ebay: EbayResponse
    shopify: ShopifyResponse
    images: GeneratedImagesResponse


MarketplaceLiteral = Literal["amazon", "ebay", "tiktok", "shopify"]
VariantTypeLiteral = Literal["size", "color"]
ProductStatusLiteral = Literal["draft", "published"]


class ProductVariantResponse(BaseModel):
    id: str
    marketplace: MarketplaceLiteral
    variant_type: VariantTypeLiteral
    name: str
    value: str
    image: ImageVariantResponse | None = None
    created_at: str


class MarketplaceVariantsResponse(BaseModel):
    amazon: list[ProductVariantResponse] = Field(default_factory=list)
    ebay: list[ProductVariantResponse] = Field(default_factory=list)
    tiktok: list[ProductVariantResponse] = Field(default_factory=list)
    shopify: list[ProductVariantResponse] = Field(default_factory=list)


class ProductRecordResponse(BaseModel):
    id: str
    status: ProductStatusLiteral
    created_at: str
    updated_at: str
    run_id: str
    product: ProductPipelineResponse
    variants: MarketplaceVariantsResponse


class ProductListItemResponse(BaseModel):
    id: str
    status: ProductStatusLiteral
    created_at: str
    updated_at: str
    normalized_title: str
    category: str
    product_type: str
    preview_image_path: str
