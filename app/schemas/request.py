from __future__ import annotations

from typing import Literal

from pydantic import BaseModel, Field


class ProductGenerationRequest(BaseModel):
    title: str = Field(min_length=3, max_length=200)


class ProductOptimizationRequest(BaseModel):
    marketplaces: list[Literal["amazon", "ebay", "etsy", "tiktok", "shopify"]] | None = None
    optimize_core: bool = True


class CoreProductUpdateRequest(BaseModel):
    normalized_title: str | None = Field(default=None, min_length=1, max_length=200)
    category: str | None = Field(default=None, min_length=1, max_length=200)
    product_type: str | None = Field(default=None, min_length=1, max_length=120)
    product_summary: str | None = Field(default=None, min_length=1)
    features: list[str] | None = None
    attributes: dict[str, str] | None = None
    source_title: str | None = Field(default=None, min_length=1, max_length=200)


class AmazonUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    bullet_points: list[str] | None = None
    description: str | None = Field(default=None, min_length=1)
    backend_search_terms: list[str] | None = None
    structured_attributes: dict[str, str] | None = None


class TikTokUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=200)
    social_description: str | None = Field(default=None, min_length=1)
    hashtags: list[str] | None = None


class EbayUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=80)
    item_specifics: dict[str, str] | None = None
    condition: str | None = Field(default=None, min_length=1, max_length=100)
    listing_notes: str | None = Field(default=None, min_length=1)


class ShopifyUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=160)
    body_html: str | None = Field(default=None, min_length=1)
    tags: list[str] | None = None
    product_type: str | None = Field(default=None, min_length=1, max_length=120)
    seo_title: str | None = Field(default=None, min_length=1, max_length=70)
    seo_description: str | None = Field(default=None, min_length=1, max_length=180)
    category: str | None = Field(default=None, min_length=1, max_length=200)
    metafields: dict[str, str] | None = None


class EtsyUpdateRequest(BaseModel):
    title: str | None = Field(default=None, min_length=1, max_length=140)
    description: str | None = Field(default=None, min_length=1)
    tags: list[str] | None = None
    materials: list[str] | None = None
    occasion: str | None = Field(default=None, min_length=1, max_length=120)
    seo_keywords: list[str] | None = None


class ProductUpdateRequest(BaseModel):
    core: CoreProductUpdateRequest | None = None
    amazon: AmazonUpdateRequest | None = None
    tiktok: TikTokUpdateRequest | None = None
    ebay: EbayUpdateRequest | None = None
    etsy: EtsyUpdateRequest | None = None
    shopify: ShopifyUpdateRequest | None = None


class VariantCreateRequest(BaseModel):
    name: str = Field(min_length=1, max_length=120)
    value: str | None = Field(default=None, max_length=120)


MarketplaceRequestLiteral = Literal["amazon", "ebay", "etsy", "tiktok", "shopify"]


class PublishTargetAnalysisRequest(BaseModel):
    marketplace: MarketplaceRequestLiteral
    product_identity: PublishTargetProductIdentityRequest | None = None
    publish_fields: PublishTargetFieldsRequest | None = None


class PublishTargetProductIdentityRequest(BaseModel):
    normalized_title: str | None = Field(default=None, min_length=1, max_length=200)
    source_title: str | None = Field(default=None, min_length=1, max_length=200)
    category: str | None = Field(default=None, min_length=1, max_length=200)
    product_type: str | None = Field(default=None, min_length=1, max_length=120)
    product_summary: str | None = Field(default=None, min_length=1)
    features: list[str] | None = None
    attributes: dict[str, str] | None = None


class PublishTargetFieldsRequest(BaseModel):
    vendor: str | None = Field(default=None, min_length=1, max_length=120)
    default_price: str | None = Field(default=None, min_length=1, max_length=40)
    default_sku: str | None = Field(default=None, min_length=1, max_length=80)
    publish_description: str | None = Field(default=None, min_length=1)
    publish_title: str | None = Field(default=None, min_length=1, max_length=200)
