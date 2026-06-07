from __future__ import annotations

from app.schemas.response import (
    AmazonResponse,
    CoreProductResponse,
    EbayResponse,
    GeneratedImagesResponse,
    PipelineValidationResponse,
    SectionValidationResponse,
    ShopifyResponse,
    TikTokResponse,
    ValidationIssueResponse,
)


class ValidationService:
    def validate_pipeline(
        self,
        *,
        core: CoreProductResponse,
        amazon: AmazonResponse,
        ebay: EbayResponse,
        tiktok: TikTokResponse,
        shopify: ShopifyResponse,
        images: GeneratedImagesResponse,
    ) -> PipelineValidationResponse:
        return PipelineValidationResponse(
            core=self._validate_core(core),
            amazon=self._validate_amazon(amazon),
            ebay=self._validate_ebay(ebay),
            tiktok=self._validate_tiktok(tiktok),
            shopify=self._validate_shopify(shopify),
            images=self._validate_images(images),
        )

    def _validate_core(self, core: CoreProductResponse) -> SectionValidationResponse:
        issues: list[ValidationIssueResponse] = []
        if len(core.normalized_title.strip()) < 4:
            issues.append(self._warning("core.normalized_title", "Normalized title is too short for marketplace quality."))
        if not core.attributes:
            issues.append(self._error("core.attributes", "No structured attributes were generated."))
        if "color" not in core.attributes:
            issues.append(self._warning("core.attributes.color", "Color attribute is missing; image styling and filters may be weak."))
        return self._section(issues)

    def _validate_amazon(self, amazon: AmazonResponse) -> SectionValidationResponse:
        issues: list[ValidationIssueResponse] = []
        if len(amazon.title) > 200:
            issues.append(self._error("amazon.title", "Amazon title exceeds 200 characters."))
        if len(amazon.bullet_points) != 5:
            issues.append(self._warning("amazon.bullet_points", "Amazon bullet count is not exactly five."))
        if len(amazon.backend_search_terms) < 6:
            issues.append(self._warning("amazon.backend_search_terms", "Amazon search terms are thin for SEO coverage."))
        return self._section(issues)

    def _validate_ebay(self, ebay: EbayResponse) -> SectionValidationResponse:
        issues: list[ValidationIssueResponse] = []
        if len(ebay.title) > 80:
            issues.append(self._error("ebay.title", "eBay title exceeds 80 characters."))
        if len(ebay.item_specifics) < 3:
            issues.append(self._warning("ebay.item_specifics", "eBay item specifics are sparse."))
        return self._section(issues)

    def _validate_tiktok(self, tiktok: TikTokResponse) -> SectionValidationResponse:
        issues: list[ValidationIssueResponse] = []
        if len(tiktok.hashtags) < 4:
            issues.append(self._warning("tiktok.hashtags", "TikTok hashtag coverage is low."))
        if len(tiktok.social_description.strip()) < 30:
            issues.append(self._warning("tiktok.social_description", "TikTok social copy is too short to carry selling points."))
        return self._section(issues)

    def _validate_shopify(self, shopify: ShopifyResponse) -> SectionValidationResponse:
        issues: list[ValidationIssueResponse] = []
        if len(shopify.seo_title) > 70:
            issues.append(self._error("shopify.seo_title", "Shopify SEO title exceeds 70 characters."))
        if len(shopify.seo_description) > 180:
            issues.append(self._error("shopify.seo_description", "Shopify SEO description exceeds 180 characters."))
        if len(shopify.tags) < 4:
            issues.append(self._warning("shopify.tags", "Shopify tag coverage is low."))
        return self._section(issues)

    def _validate_images(self, images: GeneratedImagesResponse) -> SectionValidationResponse:
        issues: list[ValidationIssueResponse] = []
        image_groups = {
            "source": images.source,
            "transparent_cutout": images.transparent_cutout,
            "amazon": images.amazon,
            "ebay": images.ebay,
            "tiktok": images.tiktok,
            "shopify": images.shopify,
        }
        for name, asset in image_groups.items():
            if asset is None:
                issues.append(self._warning(f"images.{name}", "Image variant is missing."))
                continue
            for error in asset.validation.errors:
                issues.append(self._warning(f"images.{name}", error))
        return self._section(issues)

    @staticmethod
    def _section(issues: list[ValidationIssueResponse]) -> SectionValidationResponse:
        has_error = any(issue.level == "error" for issue in issues)
        return SectionValidationResponse(passed=not has_error, issues=issues)

    @staticmethod
    def _warning(field: str, message: str) -> ValidationIssueResponse:
        return ValidationIssueResponse(level="warning", field=field, message=message)

    @staticmethod
    def _error(field: str, message: str) -> ValidationIssueResponse:
        return ValidationIssueResponse(level="error", field=field, message=message)
