from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


def _empty_section_validation() -> "SectionValidationResponse":
    return SectionValidationResponse(passed=True, issues=[])


def _empty_research_for(marketplace: str) -> "MarketplaceResearchResponse":
    return MarketplaceResearchResponse(marketplace=marketplace)


def _empty_pricing_for(marketplace: str) -> "MarketplacePricingResponse":
    return MarketplacePricingResponse(
        marketplace=marketplace,
        recommended=0.0,
        floor=0.0,
        ceiling=0.0,
        strategy="unpriced",
        confidence=0.0,
        reasons=[],
    )


def _empty_intelligence() -> "IntelligenceLayerResponse":
    return IntelligenceLayerResponse(
        research=MarketResearchBundleResponse(
            amazon=_empty_research_for("amazon"),
            ebay=_empty_research_for("ebay"),
            etsy=_empty_research_for("etsy"),
            tiktok=_empty_research_for("tiktok"),
            shopify=_empty_research_for("shopify"),
        ),
        seo=SeoInsightsResponse(),
        pricing=PricingInsightsResponse(
            amazon=_empty_pricing_for("amazon"),
            ebay=_empty_pricing_for("ebay"),
            etsy=_empty_pricing_for("etsy"),
            tiktok=_empty_pricing_for("tiktok"),
            shopify=_empty_pricing_for("shopify"),
        ),
        validation=PipelineValidationResponse(
            core=_empty_section_validation(),
            amazon=_empty_section_validation(),
            ebay=_empty_section_validation(),
            etsy=_empty_section_validation(),
            tiktok=_empty_section_validation(),
            shopify=_empty_section_validation(),
            images=_empty_section_validation(),
        ),
    )


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


class EtsyResponse(BaseModel):
    title: str
    description: str
    tags: list[str]
    materials: list[str]
    occasion: str
    seo_keywords: list[str]


class ResearchEvidenceResponse(BaseModel):
    source: str
    title: str
    price: float | None = None
    currency: str = "USD"
    relevance_score: float = Field(ge=0.0, le=1.0)
    attributes: dict[str, str] = Field(default_factory=dict)
    observations: list[str] = Field(default_factory=list)


class MarketplaceResearchResponse(BaseModel):
    marketplace: str
    source_mode: str = "heuristic"
    search_queries: list[str] = Field(default_factory=list)
    keyword_signals: list[str] = Field(default_factory=list)
    price_min: float | None = None
    price_max: float | None = None
    price_avg: float | None = None
    similar_listings: list[ResearchEvidenceResponse] = Field(default_factory=list)


class MarketResearchBundleResponse(BaseModel):
    amazon: MarketplaceResearchResponse
    ebay: MarketplaceResearchResponse
    etsy: MarketplaceResearchResponse
    tiktok: MarketplaceResearchResponse
    shopify: MarketplaceResearchResponse


class SeoInsightsResponse(BaseModel):
    primary_keywords: list[str] = Field(default_factory=list)
    secondary_keywords: list[str] = Field(default_factory=list)
    title_terms: list[str] = Field(default_factory=list)
    marketplace_keywords: dict[str, list[str]] = Field(default_factory=dict)


class MarketplacePricingResponse(BaseModel):
    marketplace: str
    recommended: float = Field(ge=0.0)
    floor: float = Field(ge=0.0)
    ceiling: float = Field(ge=0.0)
    strategy: str
    confidence: float = Field(ge=0.0, le=1.0)
    reasons: list[str] = Field(default_factory=list)


class PricingInsightsResponse(BaseModel):
    amazon: MarketplacePricingResponse
    ebay: MarketplacePricingResponse
    etsy: MarketplacePricingResponse
    tiktok: MarketplacePricingResponse
    shopify: MarketplacePricingResponse


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
    etsy: ImageVariantResponse
    tiktok: ImageVariantResponse
    shopify: ImageVariantResponse


class ValidationIssueResponse(BaseModel):
    level: Literal["warning", "error"]
    field: str
    message: str


class SectionValidationResponse(BaseModel):
    passed: bool
    issues: list[ValidationIssueResponse] = Field(default_factory=list)


class PipelineValidationResponse(BaseModel):
    core: SectionValidationResponse
    amazon: SectionValidationResponse
    ebay: SectionValidationResponse
    etsy: SectionValidationResponse
    tiktok: SectionValidationResponse
    shopify: SectionValidationResponse
    images: SectionValidationResponse


class IntelligenceLayerResponse(BaseModel):
    research: MarketResearchBundleResponse
    seo: SeoInsightsResponse
    pricing: PricingInsightsResponse
    validation: PipelineValidationResponse


class ProductPipelineResponse(BaseModel):
    core: CoreProductResponse
    amazon: AmazonResponse
    etsy: EtsyResponse
    tiktok: TikTokResponse
    ebay: EbayResponse
    shopify: ShopifyResponse
    images: GeneratedImagesResponse
    intelligence: IntelligenceLayerResponse = Field(default_factory=_empty_intelligence)


MarketplaceLiteral = Literal["amazon", "ebay", "etsy", "tiktok", "shopify"]
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
    etsy: list[ProductVariantResponse] = Field(default_factory=list)
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
