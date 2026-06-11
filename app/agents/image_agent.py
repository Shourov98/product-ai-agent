from __future__ import annotations

import colorsys
from io import BytesIO
from pathlib import Path
from typing import Literal
from urllib.request import urlopen

from app.schemas.response import (
    AmazonResponse,
    CoreProductResponse,
    EbayResponse,
    EtsyResponse,
    GeneratedImagesResponse,
    ImageVariantResponse,
    ImageValidationResponse,
    ShopifyResponse,
    TikTokResponse,
)
from app.services.image_service import ImagePayload
from app.services.openai_service import OpenAIService, OpenAIServiceError

try:
    from PIL import Image
except ImportError:  # pragma: no cover - dependency missing at runtime
    Image = None

MarketplaceName = Literal["source", "transparent_cutout", "amazon", "ebay", "etsy", "tiktok", "shopify"]


class ImageAgent:
    _PROFILES = {
        "amazon": {
            "size": "1024x1024",
            "background": "white",
            "prompt_prefix": (
                "You are a senior ecommerce product imaging agent.\n\n"
                "The uploaded product is the source of truth and must remain unchanged.\n\n"
                "You may modify only:\n"
                "- background\n"
                "- lighting mood\n"
                "- framing\n"
                "- scene styling\n"
                "- presentation environment\n\n"
                "You must not modify:\n"
                "- product color\n"
                "- product material\n"
                "- product finish\n"
                "- product shape\n"
                "- proportions\n"
                "- logo placement\n"
                "- visible construction details\n"
                "- attached components\n"
                "- product identity\n\n"
                "The generated result must look like the exact same physical product placed into a marketplace-specific presentation.\n\n"
                "No text, no watermark, no extra props unless explicitly allowed by marketplace policy.\n\n"
                "Amazon Specific Rules:\n"
                "Create a production-grade Amazon main image.\n"
                "Use a pure white background.\n"
                "Single product only.\n"
                "No props, no text, no badges, no decorations.\n"
                "Keep the exact product unchanged.\n"
                "Center the product and keep it fully visible.\n\n"
                "Prompt Version: image-amazon.v4"
            ),
        },
        "ebay": {
            "size": "1024x1024",
            "background": "white",
            "prompt_prefix": (
                "You are a senior ecommerce product imaging agent.\n\n"
                "The uploaded product is the source of truth and must remain unchanged.\n\n"
                "You may modify only:\n"
                "- background\n"
                "- lighting mood\n"
                "- framing\n"
                "- scene styling\n"
                "- presentation environment\n\n"
                "You must not modify:\n"
                "- product color\n"
                "- product material\n"
                "- product finish\n"
                "- product shape\n"
                "- proportions\n"
                "- logo placement\n"
                "- visible construction details\n"
                "- attached components\n"
                "- product identity\n\n"
                "The generated result must look like the exact same physical product placed into a marketplace-specific presentation.\n\n"
                "No text, no watermark, no extra props unless explicitly allowed by marketplace policy.\n\n"
                "eBay Specific Rules:\n"
                "Create a clean eBay-ready studio product image.\n"
                "Use a neutral white or very light background.\n"
                "Keep the exact product unchanged.\n"
                "No text overlays, badges, or distracting props.\n\n"
                "Prompt Version: image-ebay.v4"
            ),
        },
        "tiktok": {
            "size": "1024x1536",
            "background": "opaque",
            "prompt_prefix": (
                "You are a senior ecommerce product imaging agent.\n\n"
                "The uploaded product is the source of truth and must remain unchanged.\n\n"
                "You may modify only:\n"
                "- background\n"
                "- lighting mood\n"
                "- framing\n"
                "- scene styling\n"
                "- presentation environment\n\n"
                "You must not modify:\n"
                "- product color\n"
                "- product material\n"
                "- product finish\n"
                "- product shape\n"
                "- proportions\n"
                "- logo placement\n"
                "- visible construction details\n"
                "- attached components\n"
                "- product identity\n\n"
                "The generated result must look like the exact same physical product placed into a marketplace-specific presentation.\n\n"
                "No text, no watermark, no extra props unless explicitly allowed by marketplace policy.\n\n"
                "TikTok Shop Specific Rules:\n"
                "Create a vertical TikTok Shop hero image.\n"
                "Change only the background and scene styling.\n"
                "Keep the exact product unchanged.\n"
                "Use an energetic, premium, scroll-stopping commerce scene.\n"
                "No text or logo overlays.\n\n"
                "Prompt Version: image-tiktok.v4"
            ),
            "composite_scale": 0.76,
            "shadow_opacity": 68,
            "shadow_blur": 48,
            "shadow_offset_y": 34,
        },
        "etsy": {
            "size": "1200x900",
            "background": "opaque",
            "prompt_prefix": (
                "You are a senior ecommerce product imaging agent.\n\n"
                "The uploaded product is the source of truth and must remain unchanged.\n\n"
                "You may modify only:\n"
                "- background\n"
                "- lighting mood\n"
                "- framing\n"
                "- scene styling\n"
                "- presentation environment\n\n"
                "You must not modify:\n"
                "- product color\n"
                "- product material\n"
                "- product finish\n"
                "- product shape\n"
                "- proportions\n"
                "- logo placement\n"
                "- visible construction details\n"
                "- attached components\n"
                "- product identity\n\n"
                "The generated result must look like the exact same physical product placed into a marketplace-specific presentation.\n\n"
                "No text, no watermark, no extra props unless explicitly allowed by marketplace policy.\n\n"
                "Etsy Specific Rules:\n"
                "Create an Etsy-ready editorial product image.\n"
                "Change only the background and visual environment.\n"
                "Keep the exact product unchanged.\n"
                "Use a tasteful handcrafted or lifestyle-inspired backdrop.\n"
                "No text, badges, or logos.\n\n"
                "Prompt Version: image-etsy.v4"
            ),
            "composite_scale": 0.7,
            "shadow_opacity": 56,
            "shadow_blur": 30,
            "shadow_offset_y": 20,
        },
        "shopify": {
            "size": "1024x1024",
            "background": "opaque",
            "prompt_prefix": (
                "You are a senior ecommerce product imaging agent.\n\n"
                "The uploaded product is the source of truth and must remain unchanged.\n\n"
                "You may modify only:\n"
                "- background\n"
                "- lighting mood\n"
                "- framing\n"
                "- scene styling\n"
                "- presentation environment\n\n"
                "You must not modify:\n"
                "- product color\n"
                "- product material\n"
                "- product finish\n"
                "- product shape\n"
                "- proportions\n"
                "- logo placement\n"
                "- visible construction details\n"
                "- attached components\n"
                "- product identity\n\n"
                "The generated result must look like the exact same physical product placed into a marketplace-specific presentation.\n\n"
                "No text, no watermark, no extra props unless explicitly allowed by marketplace policy.\n\n"
                "Shopify Specific Rules:\n"
                "Create a polished Shopify storefront hero image.\n"
                "Change only the background and presentation styling.\n"
                "Keep the exact product unchanged.\n"
                "Use premium ecommerce lighting and a refined brand-style scene.\n"
                "No text, badges, or logos.\n\n"
                "Prompt Version: image-shopify.v4"
            ),
            "composite_scale": 0.72,
            "shadow_opacity": 58,
            "shadow_blur": 34,
            "shadow_offset_y": 22,
        },
    }

    def __init__(self, openai_service: OpenAIService | None = None) -> None:
        self.openai_service = openai_service

    async def process(
        self,
        *,
        image: ImagePayload,
        core_data: CoreProductResponse,
        amazon_data: AmazonResponse,
        ebay_data: EbayResponse,
        etsy_data: EtsyResponse,
        tiktok_data: TikTokResponse,
        shopify_data: ShopifyResponse,
        run_dir: Path,
        output_service,
    ) -> GeneratedImagesResponse:
        source_asset = self._save_source(image=image, run_dir=run_dir, output_service=output_service)
        cutout_asset = await self._build_cutout(
            image=image,
            core_data=core_data,
            run_dir=run_dir,
            output_service=output_service,
        )

        amazon_asset = self._build_white_background_variant(
            marketplace="amazon",
            source=image,
            cutout_path=cutout_asset.absolute_path if cutout_asset else None,
            title=amazon_data.title,
            description=amazon_data.description,
            attributes=amazon_data.structured_attributes,
            run_dir=run_dir,
            output_service=output_service,
        )
        ebay_asset = self._build_white_background_variant(
            marketplace="ebay",
            source=image,
            cutout_path=cutout_asset.absolute_path if cutout_asset else None,
            title=ebay_data.title,
            description=ebay_data.listing_notes,
            attributes=ebay_data.item_specifics,
            run_dir=run_dir,
            output_service=output_service,
        )
        tiktok_asset = await self._build_marketplace_variant(
            marketplace="tiktok",
            source=image,
            base_image_path=cutout_asset.absolute_path if cutout_asset else source_asset.absolute_path,
            title=tiktok_data.title,
            description=tiktok_data.social_description,
            attributes=core_data.attributes,
            run_dir=run_dir,
            output_service=output_service,
        )
        etsy_asset = await self._build_marketplace_variant(
            marketplace="etsy",
            source=image,
            base_image_path=cutout_asset.absolute_path if cutout_asset else source_asset.absolute_path,
            title=etsy_data.title,
            description=etsy_data.description,
            attributes=core_data.attributes,
            run_dir=run_dir,
            output_service=output_service,
        )
        shopify_asset = await self._build_marketplace_variant(
            marketplace="shopify",
            source=image,
            base_image_path=cutout_asset.absolute_path if cutout_asset else source_asset.absolute_path,
            title=shopify_data.title,
            description=shopify_data.seo_description,
            attributes=core_data.attributes,
            run_dir=run_dir,
            output_service=output_service,
        )

        return GeneratedImagesResponse(
            source=source_asset,
            transparent_cutout=cutout_asset,
            amazon=amazon_asset,
            ebay=ebay_asset,
            etsy=etsy_asset,
            tiktok=tiktok_asset,
            shopify=shopify_asset,
        )

    async def regenerate_marketplace_asset(
        self,
        *,
        marketplace: Literal["amazon", "ebay", "etsy", "tiktok", "shopify"],
        source: ImagePayload,
        existing_images: GeneratedImagesResponse,
        core_data: CoreProductResponse,
        amazon_data: AmazonResponse,
        ebay_data: EbayResponse,
        etsy_data: EtsyResponse,
        tiktok_data: TikTokResponse,
        shopify_data: ShopifyResponse,
        run_dir: Path,
        output_service,
    ) -> ImageVariantResponse:
        cutout_path = (
            existing_images.transparent_cutout.absolute_path
            if existing_images.transparent_cutout is not None
            else None
        )
        if marketplace == "amazon":
            return self._build_white_background_variant(
                marketplace="amazon",
                source=source,
                cutout_path=cutout_path,
                title=amazon_data.title,
                description=amazon_data.description,
                attributes=amazon_data.structured_attributes,
                run_dir=run_dir,
                output_service=output_service,
            )
        if marketplace == "ebay":
            return self._build_white_background_variant(
                marketplace="ebay",
                source=source,
                cutout_path=cutout_path,
                title=ebay_data.title,
                description=ebay_data.listing_notes,
                attributes=ebay_data.item_specifics,
                run_dir=run_dir,
                output_service=output_service,
            )
        if marketplace == "etsy":
            base_image_path = cutout_path or existing_images.source.absolute_path
            return await self._build_marketplace_variant(
                marketplace="etsy",
                source=source,
                base_image_path=base_image_path,
                title=etsy_data.title,
                description=etsy_data.description,
                attributes=core_data.attributes,
                run_dir=run_dir,
                output_service=output_service,
            )
        if marketplace == "tiktok":
            base_image_path = cutout_path or existing_images.source.absolute_path
            return await self._build_marketplace_variant(
                marketplace="tiktok",
                source=source,
                base_image_path=base_image_path,
                title=tiktok_data.title,
                description=tiktok_data.social_description,
                attributes=core_data.attributes,
                run_dir=run_dir,
                output_service=output_service,
            )

        base_image_path = cutout_path or existing_images.source.absolute_path
        return await self._build_marketplace_variant(
            marketplace="shopify",
            source=source,
            base_image_path=base_image_path,
            title=shopify_data.title,
            description=shopify_data.seo_description,
            attributes=core_data.attributes,
            run_dir=run_dir,
            output_service=output_service,
        )

    def build_color_variant_asset(
        self,
        *,
        marketplace: Literal["amazon", "ebay", "etsy", "tiktok", "shopify"],
        source: ImagePayload,
        existing_images: GeneratedImagesResponse,
        color_name: str,
        title: str,
        run_dir: Path,
        output_service,
    ) -> ImageVariantResponse:
        profile = self._PROFILES[marketplace]
        expected_width = self._size_to_width(profile["size"])
        expected_height = self._size_to_height(profile["size"])
        relative_path = f"variants/{marketplace}-{self._slugify(color_name)}.png"
        prompt = (
            f"Create a {marketplace} color variant image for {title}. "
            f"Apply the color variant {color_name} to the product while preserving shape, proportions, and product identity."
        )
        cutout_path = (
            existing_images.transparent_cutout.absolute_path
            if existing_images.transparent_cutout is not None
            else existing_images.source.absolute_path
        )

        try:
            background = self._background_for_marketplace(marketplace, expected_width, expected_height, color_name)
            image_bytes = self._compose_color_variant(
                cutout_path=cutout_path,
                width=expected_width,
                height=expected_height,
                color_name=color_name,
                background=background,
            )
            absolute_path = output_service.save_binary(run_dir, relative_path, image_bytes, mime_type="image/png")
            validation = self._validate_bytes(
                image_bytes,
                mime_type="image/png",
                expected_width=expected_width,
                expected_height=expected_height,
                background=profile["background"],
                marketplace=marketplace,
            )
            return ImageVariantResponse(
                marketplace=marketplace,
                relative_path=relative_path,
                absolute_path=str(absolute_path),
                prompt=prompt,
                generation_mode="local_color_variant",
                mime_type="image/png",
                validation=validation,
            )
        except Exception as exc:
            absolute_path = output_service.save_binary(run_dir, relative_path, source.data, mime_type=source.content_type)
            validation = self._validate_bytes(
                source.data,
                mime_type=source.content_type,
                expected_width=expected_width,
                expected_height=expected_height,
                background=profile["background"],
                marketplace=marketplace,
                errors=[f"Color variant composition failed: {exc}"],
            )
            return ImageVariantResponse(
                marketplace=marketplace,
                relative_path=relative_path,
                absolute_path=str(absolute_path),
                prompt=prompt,
                generation_mode="source_passthrough",
                mime_type=source.content_type,
                validation=validation,
            )

    def _save_source(self, *, image: ImagePayload, run_dir: Path, output_service) -> ImageVariantResponse:
        relative_path = f"images/source-{image.filename}"
        absolute_path = output_service.save_binary(run_dir, relative_path, image.data, mime_type=image.content_type)
        validation = self._validate_bytes(
            image.data,
            mime_type=image.content_type,
            expected_width=None,
            expected_height=None,
            background="source",
            marketplace="source",
        )
        return ImageVariantResponse(
            marketplace="source",
            relative_path=relative_path,
            absolute_path=str(absolute_path),
            prompt="Original uploaded image saved for audit and downstream editing.",
            generation_mode="source_passthrough",
            mime_type=image.content_type,
            validation=validation,
        )

    async def _build_cutout(
        self,
        *,
        image: ImagePayload,
        core_data: CoreProductResponse,
        run_dir: Path,
        output_service,
    ) -> ImageVariantResponse | None:
        prompt = (
            f"Remove the background and isolate the product for ecommerce use. "
            f"Preserve the exact product shape, color, and material. "
            "Do not add any text, captions, labels, banners, watermark shapes, underline bars, or decorative graphic elements. "
            "Return only the isolated product on a transparent background. "
            f"Product title: {core_data.normalized_title}. "
            f"Product type: {core_data.product_type}. "
            f"Attributes: {core_data.attributes}."
        )
        relative_path = "images/transparent-cutout.png"

        if self.openai_service is None or not self.openai_service.enabled:
            return None

        try:
            image_bytes = await self.openai_service.edit_image(
                prompt=prompt,
                image_bytes=image.data,
                filename=image.filename,
                mime_type=image.content_type,
                size="1024x1024",
                background="transparent",
            )
            image_bytes = self._cleanup_cutout_artifacts(image_bytes)
            absolute_path = output_service.save_binary(run_dir, relative_path, image_bytes, mime_type="image/png")
            validation = self._validate_bytes(
                image_bytes,
                mime_type="image/png",
                expected_width=1024,
                expected_height=1024,
                background="transparent",
                marketplace="transparent_cutout",
            )
            return ImageVariantResponse(
                marketplace="transparent_cutout",
                relative_path=relative_path,
                absolute_path=str(absolute_path),
                prompt=prompt,
                generation_mode="edited",
                mime_type="image/png",
                validation=validation,
            )
        except OpenAIServiceError:
            return None

    def _build_white_background_variant(
        self,
        *,
        marketplace: Literal["amazon", "ebay"],
        source: ImagePayload,
        cutout_path: str | None,
        title: str,
        description: str,
        attributes: dict[str, str],
        run_dir: Path,
        output_service,
    ) -> ImageVariantResponse:
        profile = self._PROFILES[marketplace]
        prompt = (
            f"{profile['prompt_prefix']} "
            f"Product title: {title}. "
            f"Description: {description}. "
            f"Attributes: {attributes}. "
            "Background-only edit requirement: modify the environment/background only while leaving the product unchanged."
        )
        relative_path = f"images/{marketplace}.png"
        expected_width = self._size_to_width(profile["size"])
        expected_height = self._size_to_height(profile["size"])

        if cutout_path is None:
            absolute_path = output_service.save_binary(run_dir, relative_path, source.data, mime_type=source.content_type)
            validation = self._validate_bytes(
                source.data,
                mime_type=source.content_type,
                expected_width=expected_width,
                expected_height=expected_height,
                background=profile["background"],
                marketplace=marketplace,
                errors=[
                    "Transparent cutout was unavailable; white-background composition could not be performed.",
                    *(
                        ["Amazon output is blocked until a compliant cutout and pure-white composition can be generated."]
                        if marketplace == "amazon"
                        else []
                    ),
                ],
            )
            return ImageVariantResponse(
                marketplace=marketplace,
                relative_path=relative_path,
                absolute_path=str(absolute_path),
                prompt=prompt,
                generation_mode="compliance_blocked_source_passthrough" if marketplace == "amazon" else "source_passthrough",
                mime_type=source.content_type,
                validation=validation,
            )

        try:
            image_bytes = self._compose_cutout_on_background(
                cutout_path=cutout_path,
                width=expected_width,
                height=expected_height,
                background=(255, 255, 255, 255),
            )
            absolute_path = output_service.save_binary(run_dir, relative_path, image_bytes, mime_type="image/png")
            validation = self._validate_bytes(
                image_bytes,
                mime_type="image/png",
                expected_width=expected_width,
                expected_height=expected_height,
                background=profile["background"],
                marketplace=marketplace,
            )
            return ImageVariantResponse(
                marketplace=marketplace,
                relative_path=relative_path,
                absolute_path=str(absolute_path),
                prompt=prompt,
                generation_mode="local_composite_from_cutout",
                mime_type="image/png",
                validation=validation,
            )
        except Exception as exc:
            absolute_path = output_service.save_binary(run_dir, relative_path, source.data, mime_type=source.content_type)
            validation = self._validate_bytes(
                source.data,
                mime_type=source.content_type,
                expected_width=expected_width,
                expected_height=expected_height,
                background=profile["background"],
                marketplace=marketplace,
                errors=[
                    f"White-background composition failed: {exc}",
                    *(
                        ["Amazon output is blocked until a compliant pure-white composition can be generated."]
                        if marketplace == "amazon"
                        else []
                    ),
                ],
            )
            return ImageVariantResponse(
                marketplace=marketplace,
                relative_path=relative_path,
                absolute_path=str(absolute_path),
                prompt=prompt,
                generation_mode="compliance_blocked_source_passthrough" if marketplace == "amazon" else "source_passthrough",
                mime_type=source.content_type,
                validation=validation,
            )

    async def _build_marketplace_variant(
        self,
        *,
        marketplace: Literal["etsy", "tiktok", "shopify"],
        source: ImagePayload,
        base_image_path: str,
        title: str,
        description: str,
        attributes: dict[str, str],
        run_dir: Path,
        output_service,
    ) -> ImageVariantResponse:
        profile = self._PROFILES[marketplace]
        prompt = (
            f"{profile['prompt_prefix']} "
            f"Product title: {title}. "
            f"Description: {description}. "
            f"Attributes: {attributes}. "
            "Background-only edit requirement: modify the environment/background only while leaving the product unchanged."
        )
        relative_path = f"images/{marketplace}.png"
        expected_width = self._size_to_width(profile["size"])
        expected_height = self._size_to_height(profile["size"])

        try:
            image_bytes = self._compose_marketplace_scene(
                cutout_path=base_image_path,
                marketplace=marketplace,
                width=expected_width,
                height=expected_height,
                attributes=attributes,
            )
            absolute_path = output_service.save_binary(run_dir, relative_path, image_bytes, mime_type="image/png")
            validation = self._validate_bytes(
                image_bytes,
                mime_type="image/png",
                expected_width=expected_width,
                expected_height=expected_height,
                background=profile["background"],
                marketplace=marketplace,
            )
            return ImageVariantResponse(
                marketplace=marketplace,
                relative_path=relative_path,
                absolute_path=str(absolute_path),
                prompt=prompt,
                generation_mode="local_composite_from_cutout",
                mime_type="image/png",
                validation=validation,
            )
        except Exception as exc:
            absolute_path = output_service.save_binary(run_dir, relative_path, source.data, mime_type=source.content_type)
            validation = self._validate_bytes(
                source.data,
                mime_type=source.content_type,
                expected_width=expected_width,
                expected_height=expected_height,
                background=profile["background"],
                marketplace=marketplace,
                errors=[f"Marketplace composite failed: {exc}"],
            )
            return ImageVariantResponse(
                marketplace=marketplace,
                relative_path=relative_path,
                absolute_path=str(absolute_path),
                prompt=prompt,
                generation_mode="source_passthrough",
                mime_type=source.content_type,
                validation=validation,
            )

    @staticmethod
    def _size_to_width(size: str) -> int:
        return int(size.split("x", maxsplit=1)[0])

    @staticmethod
    def _size_to_height(size: str) -> int:
        return int(size.split("x", maxsplit=1)[1])

    def _compose_cutout_on_background(
        self,
        *,
        cutout_path: str,
        width: int,
        height: int,
        background: tuple[int, int, int, int],
    ) -> bytes:
        if Image is None:
            raise RuntimeError("Pillow is not installed.")

        cutout_bytes = self._read_bytes_from_reference(cutout_path)
        with Image.open(BytesIO(cutout_bytes)) as cutout:
            cutout_rgba = cutout.convert("RGBA")
            scale = min((width * 0.88) / cutout_rgba.width, (height * 0.88) / cutout_rgba.height)
            resized = cutout_rgba.resize(
                (max(1, int(cutout_rgba.width * scale)), max(1, int(cutout_rgba.height * scale))),
                Image.Resampling.LANCZOS,
            )
            canvas = Image.new("RGBA", (width, height), background)
            offset_x = (width - resized.width) // 2
            offset_y = (height - resized.height) // 2
            canvas.alpha_composite(resized, (offset_x, offset_y))
            rgb_canvas = canvas.convert("RGB")
            buffer = BytesIO()
            rgb_canvas.save(buffer, format="PNG")
            return buffer.getvalue()

    def _compose_color_variant(
        self,
        *,
        cutout_path: str,
        width: int,
        height: int,
        color_name: str,
        background: tuple[int, int, int, int] | tuple[tuple[int, int, int], tuple[int, int, int]],
    ) -> bytes:
        if Image is None:
            raise RuntimeError("Pillow is not installed.")

        tone = self._color_name_to_rgb(color_name)
        cutout_bytes = self._read_bytes_from_reference(cutout_path)
        with Image.open(BytesIO(cutout_bytes)) as cutout:
            cutout_rgba = cutout.convert("RGBA")
            scale = min((width * 0.78) / cutout_rgba.width, (height * 0.78) / cutout_rgba.height)
            resized = cutout_rgba.resize(
                (max(1, int(cutout_rgba.width * scale)), max(1, int(cutout_rgba.height * scale))),
                Image.Resampling.LANCZOS,
            )
            tinted = self._tint_image(resized, tone)
            canvas = self._build_canvas(width, height, background)
            offset_x = (width - tinted.width) // 2
            offset_y = (height - tinted.height) // 2
            canvas.alpha_composite(tinted, (offset_x, offset_y))
            buffer = BytesIO()
            canvas.convert("RGBA").save(buffer, format="PNG")
            return buffer.getvalue()

    def _compose_marketplace_scene(
        self,
        *,
        cutout_path: str,
        marketplace: Literal["tiktok", "shopify"],
        width: int,
        height: int,
        attributes: dict[str, str],
    ) -> bytes:
        if Image is None:
            raise RuntimeError("Pillow is not installed.")

        profile = self._PROFILES[marketplace]
        background = self._styled_background_for_marketplace(marketplace, attributes)
        cutout_bytes = self._read_bytes_from_reference(cutout_path)

        with Image.open(BytesIO(cutout_bytes)) as cutout:
            cutout_rgba = cutout.convert("RGBA")
            scale_ratio = profile["composite_scale"]
            scale = min((width * scale_ratio) / cutout_rgba.width, (height * scale_ratio) / cutout_rgba.height)
            resized = cutout_rgba.resize(
                (max(1, int(cutout_rgba.width * scale)), max(1, int(cutout_rgba.height * scale))),
                Image.Resampling.LANCZOS,
            )
            canvas = self._build_canvas(width, height, background)
            self._draw_spotlight(canvas, marketplace)
            offset_x = (width - resized.width) // 2
            headroom_ratio = 0.15 if marketplace == "tiktok" else 0.14
            offset_y = max(int(height * headroom_ratio), (height - resized.height) // 2 - int(height * 0.03))
            canvas.alpha_composite(resized, (offset_x, offset_y))

            buffer = BytesIO()
            canvas.convert("RGBA").save(buffer, format="PNG")
            return buffer.getvalue()

    def _build_canvas(
        self,
        width: int,
        height: int,
        background: tuple[int, int, int, int] | tuple[tuple[int, int, int], tuple[int, int, int]],
    ):
        if Image is None:
            raise RuntimeError("Pillow is not installed.")

        if isinstance(background[0], int):
            return Image.new("RGBA", (width, height), background)

        top_rgb, bottom_rgb = background
        canvas = Image.new("RGBA", (width, height))
        for y in range(height):
            ratio = y / max(height - 1, 1)
            row = tuple(
                int(top_rgb[index] + (bottom_rgb[index] - top_rgb[index]) * ratio)
                for index in range(3)
            )
            for x in range(width):
                canvas.putpixel((x, y), (*row, 255))
        return canvas

    def _draw_spotlight(self, canvas, marketplace: Literal["etsy", "tiktok", "shopify"]) -> None:
        if Image is None:
            raise RuntimeError("Pillow is not installed.")

        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        width, height = canvas.size
        center_x = width // 2
        center_y = int(height * (0.34 if marketplace == "tiktok" else 0.38))
        radius_x = int(width * 0.34)
        radius_y = int(height * (0.16 if marketplace == "tiktok" else 0.2))
        pixels = overlay.load()

        for y in range(height):
            for x in range(width):
                dx = (x - center_x) / max(radius_x, 1)
                dy = (y - center_y) / max(radius_y, 1)
                distance = dx * dx + dy * dy
                if distance >= 1:
                    continue
                intensity = int((1 - distance) * (36 if marketplace == "tiktok" else 28))
                pixels[x, y] = (255, 255, 255, intensity)

        canvas.alpha_composite(overlay)

    def _draw_ground_shadow(
        self,
        canvas,
        *,
        product_x: int,
        product_y: int,
        product_width: int,
        product_height: int,
        shadow_opacity: int,
        shadow_blur: int,
        shadow_offset_y: int,
    ) -> None:
        if Image is None:
            raise RuntimeError("Pillow is not installed.")

        try:
            from PIL import ImageDraw, ImageFilter
        except ImportError as exc:  # pragma: no cover
            raise RuntimeError("Pillow drawing dependencies are not installed.") from exc

        overlay = Image.new("RGBA", canvas.size, (0, 0, 0, 0))
        width, height = canvas.size
        shadow_width = max(36, int(product_width * 0.54))
        shadow_height = max(14, int(product_height * 0.06))
        shadow = Image.new("RGBA", (shadow_width, shadow_height), (0, 0, 0, 0))
        draw = ImageDraw.Draw(shadow)
        draw.ellipse(
            (0, 0, shadow_width - 1, shadow_height - 1),
            fill=(0, 0, 0, max(18, int(shadow_opacity * 0.72))),
        )
        shadow = shadow.filter(ImageFilter.GaussianBlur(radius=max(10, int(shadow_blur * 0.45))))
        shadow_x = product_x + (product_width - shadow.width) // 2
        shadow_y = min(height - shadow.height - 10, product_y + product_height - max(2, int(product_height * 0.03)) + shadow_offset_y)
        shadow_x = max(0, min(width - shadow.width, shadow_x))
        shadow_y = max(0, min(height - shadow.height, shadow_y))
        overlay.alpha_composite(shadow, (shadow_x, shadow_y))
        canvas.alpha_composite(overlay)

    def _tint_image(self, image, tone: tuple[int, int, int]):
        if Image is None:
            raise RuntimeError("Pillow is not installed.")

        tinted = image.copy()
        pixels = tinted.load()
        for y in range(tinted.height):
            for x in range(tinted.width):
                red, green, blue, alpha = pixels[x, y]
                if alpha == 0:
                    continue
                _, lightness, _ = colorsys.rgb_to_hls(red / 255, green / 255, blue / 255)
                toned = tuple(
                    max(0, min(255, int(channel * (0.45 + lightness * 0.85))))
                    for channel in tone
                )
                pixels[x, y] = (*toned, alpha)
        return tinted

    @staticmethod
    def _slugify(value: str) -> str:
        slug = "".join(char.lower() if char.isalnum() else "-" for char in value.strip())
        while "--" in slug:
            slug = slug.replace("--", "-")
        return slug.strip("-") or "variant"

    @staticmethod
    def _color_name_to_rgb(color_name: str) -> tuple[int, int, int]:
        lowered = color_name.lower()
        if "navy" in lowered:
            return (30, 58, 109)
        if "blue" in lowered:
            return (37, 99, 235)
        if "green" in lowered:
            return (22, 101, 52)
        if "red" in lowered:
            return (220, 38, 38)
        if "black" in lowered:
            return (31, 41, 55)
        if "white" in lowered:
            return (229, 231, 235)
        if "pink" in lowered:
            return (236, 72, 153)
        if "purple" in lowered:
            return (124, 58, 237)
        if "yellow" in lowered:
            return (234, 179, 8)
        if "orange" in lowered:
            return (234, 88, 12)
        return (71, 85, 105)

    @staticmethod
    def _read_bytes_from_reference(reference: str) -> bytes:
        if reference.startswith("http://") or reference.startswith("https://"):
            with urlopen(reference, timeout=30) as response:
                return response.read()
        return Path(reference).read_bytes()

    @staticmethod
    def _filename_from_reference(reference: str) -> str:
        return Path(reference.split("?", maxsplit=1)[0]).name or "image.png"

    @staticmethod
    def _background_for_marketplace(
        marketplace: Literal["amazon", "ebay", "etsy", "tiktok", "shopify"],
        width: int,
        height: int,
        color_name: str,
    ) -> tuple[int, int, int, int] | tuple[tuple[int, int, int], tuple[int, int, int]]:
        del width, height
        if marketplace in {"amazon", "ebay"}:
            return (255, 255, 255, 255)

        base = ImageAgent._color_name_to_rgb(color_name)
        lifted = tuple(min(255, channel + 80) for channel in base)
        return (lifted, (245, 247, 250))

    @staticmethod
    def _styled_background_for_marketplace(
        marketplace: Literal["etsy", "tiktok", "shopify"],
        attributes: dict[str, str],
    ) -> tuple[tuple[int, int, int], tuple[int, int, int]]:
        color_name = attributes.get("color", "")
        base = ImageAgent._color_name_to_rgb(color_name)
        softened = tuple(min(255, int(channel * 0.34) + 150) for channel in base)
        neutral = (246, 248, 252) if marketplace == "shopify" else (244, 247, 251)
        accent = tuple(min(255, int(channel * 0.52) + 92) for channel in base)

        if marketplace == "shopify":
            top = tuple(min(255, int((softened[index] * 0.58) + (neutral[index] * 0.42))) for index in range(3))
            bottom = neutral
            return (top, bottom)

        if marketplace == "etsy":
            top = tuple(min(255, int((softened[index] * 0.42) + 140)) for index in range(3))
            bottom = (248, 244, 238)
            return (top, bottom)

        top = tuple(min(255, int((accent[index] * 0.55) + (neutral[index] * 0.45))) for index in range(3))
        bottom = tuple(min(255, int((softened[index] * 0.28) + (neutral[index] * 0.72))) for index in range(3))
        return (top, bottom)

    def _validate_bytes(
        self,
        payload: bytes,
        *,
        mime_type: str,
        expected_width: int | None,
        expected_height: int | None,
        background: str,
        marketplace: MarketplaceName,
        errors: list[str] | None = None,
    ) -> ImageValidationResponse:
        collected_errors = list(errors or [])
        width = None
        height = None
        image_format = None
        has_alpha = None

        if Image is None:
            collected_errors.append("Pillow is not installed; image validation is incomplete.")
        else:
            try:
                with Image.open(BytesIO(payload)) as image:
                    width, height = image.size
                    image_format = image.format
                    has_alpha = "A" in image.getbands()
            except Exception as exc:
                collected_errors.append(f"Could not parse image bytes: {exc}")

        if expected_width is not None and width is not None and width != expected_width:
            collected_errors.append(f"Expected width {expected_width}, got {width}.")
        if expected_height is not None and height is not None and height != expected_height:
            collected_errors.append(f"Expected height {expected_height}, got {height}.")
        if background == "transparent" and has_alpha is False:
            collected_errors.append("Transparent output expected but no alpha channel detected.")
        if background == "white" and Image is not None:
            try:
                with Image.open(BytesIO(payload)) as image:
                    rgb_image = image.convert("RGB")
                    if marketplace == "amazon":
                        white_issue = self._validate_amazon_white_background(rgb_image)
                    else:
                        white_issue = self._validate_white_background(rgb_image, strict=False)
                    if white_issue is not None:
                        collected_errors.append(white_issue)
            except Exception as exc:
                collected_errors.append(f"Could not verify white background: {exc}")

        if marketplace == "amazon" and Image is not None:
            try:
                with Image.open(BytesIO(payload)) as image:
                    rgb_image = image.convert("RGB")
                    fill_ratio_issue = self._validate_product_fill_ratio(rgb_image, minimum_ratio=0.85)
                    if fill_ratio_issue is not None:
                        collected_errors.append(fill_ratio_issue)
            except Exception as exc:
                collected_errors.append(f"Could not verify Amazon fill ratio: {exc}")

        if marketplace in {"amazon", "ebay"} and Image is not None:
            try:
                with Image.open(BytesIO(payload)) as image:
                    rgb_image = image.convert("RGB")
                    blur_issue = self._detect_blur_issue(
                        rgb_image,
                        marketplace=marketplace,
                    )
                    if blur_issue is not None:
                        collected_errors.append(blur_issue)
            except Exception as exc:
                collected_errors.append(f"Could not inspect image sharpness: {exc}")

        if marketplace in {"etsy", "tiktok", "shopify"} and Image is not None:
            try:
                with Image.open(BytesIO(payload)) as image:
                    rgb_image = image.convert("RGB")
                    banner_issue = self._detect_bottom_text_banner(rgb_image)
                    if banner_issue is not None:
                        collected_errors.append(banner_issue)
                    shadow_bar_issue = self._detect_flat_shadow_bar(rgb_image)
                    if shadow_bar_issue is not None:
                        collected_errors.append(shadow_bar_issue)
            except Exception as exc:
                collected_errors.append(f"Could not inspect styled image artifacts: {exc}")

        if marketplace == "transparent_cutout" and Image is not None:
            try:
                with Image.open(BytesIO(payload)) as image:
                    cutout_issue = self._detect_cutout_text_artifacts(image.convert("RGBA"))
                    if cutout_issue is not None:
                        collected_errors.append(cutout_issue)
            except Exception as exc:
                collected_errors.append(f"Could not inspect transparent cutout artifacts: {exc}")

        return ImageValidationResponse(
            passed=len(collected_errors) == 0,
            width=width,
            height=height,
            format=image_format,
            has_alpha=has_alpha,
            file_size_bytes=len(payload),
            expected_width=expected_width,
            expected_height=expected_height,
            expected_background=background,
            errors=collected_errors,
            mime_type=mime_type,
        )

    def _validate_amazon_white_background(self, image) -> str | None:
        return self._validate_white_background(
            image,
            strict=True,
            min_white_ratio=0.45,
        )

    def _validate_white_background(
        self,
        image,
        *,
        strict: bool,
        min_white_ratio: float = 0.3,
    ) -> str | None:
        if Image is None:
            return None

        width, height = image.size
        white_threshold = 255 if strict else 245
        white_pixels = 0
        product_pixels = 0
        min_x = width
        min_y = height
        max_x = -1
        max_y = -1
        pixels = image.load()

        for y in range(height):
            for x in range(width):
                red, green, blue = pixels[x, y]
                if red >= white_threshold and green >= white_threshold and blue >= white_threshold:
                    white_pixels += 1
                    continue
                product_pixels += 1
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        total_pixels = max(width * height, 1)
        white_ratio = white_pixels / total_pixels
        if white_ratio < min_white_ratio:
            return "Expected white background, but the image does not contain enough white background area."

        if product_pixels == 0:
            return "Could not detect a product against the white background."

        if min_x <= 0 or min_y <= 0 or max_x >= width - 1 or max_y >= height - 1:
            return "Expected visible white margin around the product, but the product or non-white regions touch the frame edge."

        margin = 2 if strict else 1
        for y in range(height):
            for x in range(width):
                if (
                    min_x - margin <= x <= max_x + margin
                    and min_y - margin <= y <= max_y + margin
                ):
                    continue
                red, green, blue = pixels[x, y]
                if strict:
                    if red != 255 or green != 255 or blue != 255:
                        return "Amazon requires a pure white background (RGB 255,255,255) outside the product area."
                else:
                    if min(red, green, blue) < 245:
                        return "Expected a clean white background outside the product area."

        return None

    def _validate_product_fill_ratio(self, image, *, minimum_ratio: float) -> str | None:
        if Image is None:
            return None

        bbox = self._estimate_product_bbox_on_white(image)
        if bbox is None:
            return "Could not estimate product size against the white background."

        min_x, min_y, max_x, max_y = bbox
        width, height = image.size
        product_width = max_x - min_x + 1
        product_height = max_y - min_y + 1
        width_ratio = product_width / max(width, 1)
        height_ratio = product_height / max(height, 1)
        fill_ratio = max(width_ratio, height_ratio)

        if fill_ratio < minimum_ratio:
            return (
                f"Amazon requires the product to fill about {int(minimum_ratio * 100)}% of the frame. "
                f"Detected fill ratio was {fill_ratio:.2f}."
            )
        return None

    def _estimate_product_bbox_on_white(self, image) -> tuple[int, int, int, int] | None:
        if Image is None:
            return None

        width, height = image.size
        pixels = image.load()
        min_x = width
        min_y = height
        max_x = -1
        max_y = -1

        for y in range(height):
            for x in range(width):
                red, green, blue = pixels[x, y]
                if red >= 250 and green >= 250 and blue >= 250:
                    continue
                min_x = min(min_x, x)
                min_y = min(min_y, y)
                max_x = max(max_x, x)
                max_y = max(max_y, y)

        if max_x < 0 or max_y < 0:
            return None
        return (min_x, min_y, max_x, max_y)

    def _detect_blur_issue(self, image, *, marketplace: str) -> str | None:
        if Image is None:
            return None

        grayscale = image.convert("L")
        width, height = grayscale.size
        if width < 48 or height < 48:
            return "Image resolution is too low to verify sharpness confidently."

        step = max(1, min(width, height) // 256)
        total = 0
        count = 0

        for y in range(step, height - step, step):
            for x in range(step, width - step, step):
                center = grayscale.getpixel((x, y))
                left = grayscale.getpixel((x - step, y))
                right = grayscale.getpixel((x + step, y))
                up = grayscale.getpixel((x, y - step))
                down = grayscale.getpixel((x, y + step))
                total += abs((4 * center) - left - right - up - down)
                count += 1

        if count == 0:
            return "Image sharpness could not be evaluated."

        sharpness_score = total / count
        threshold = 14.0 if marketplace == "amazon" else 12.0
        if sharpness_score < threshold:
            return (
                f"{marketplace.title()} image appears blurry or too soft for marketplace compliance. "
                f"Detected sharpness score was {sharpness_score:.2f}."
            )
        return None

    def _detect_bottom_text_banner(self, image) -> str | None:
        if Image is None:
            return None

        width, height = image.size
        if width < 160 or height < 160:
            return None

        start_y = int(height * 0.72)
        end_y = height
        row_dark_ratios: list[float] = []

        for y in range(start_y, end_y):
            dark_pixels = 0
            for x in range(width):
                red, green, blue = image.getpixel((x, y))
                if max(red, green, blue) < 170:
                    dark_pixels += 1
            row_dark_ratios.append(dark_pixels / max(width, 1))

        strong_rows = [ratio for ratio in row_dark_ratios if ratio >= 0.22]
        medium_rows = [ratio for ratio in row_dark_ratios if ratio >= 0.08]
        has_sustained_band = len(strong_rows) >= 6 or len(medium_rows) >= 10
        if not has_sustained_band:
            return None

        bottom_crop = image.crop((0, start_y, width, end_y))
        grayscale = bottom_crop.convert("L")
        horizontal_edges = 0
        for y in range(1, grayscale.height):
            transitions = 0
            prev_value = grayscale.getpixel((0, y))
            for x in range(1, grayscale.width):
                current_value = grayscale.getpixel((x, y))
                if abs(current_value - prev_value) > 35:
                    transitions += 1
                prev_value = current_value
            if transitions > max(18, grayscale.width // 18):
                horizontal_edges += 1

        if horizontal_edges >= 4:
            return (
                "Styled marketplace image appears to contain bottom text/banner artifacts. "
                "Reject captions, labels, underline bars, and title strips; keep background swap only."
            )
        return None

    def _detect_flat_shadow_bar(self, image) -> str | None:
        if Image is None:
            return None

        width, height = image.size
        if width < 160 or height < 160:
            return None

        start_y = int(height * 0.52)
        end_y = int(height * 0.88)
        grayscale = image.convert("L")

        for y in range(start_y, end_y):
            run_start = None
            run_length = 0
            for x in range(width):
                value = grayscale.getpixel((x, y))
                if value < 175:
                    if run_start is None:
                        run_start = x
                    run_length += 1
                else:
                    if run_start is not None:
                        break
            if run_start is None:
                continue

            centered = abs((run_start + (run_length / 2)) - (width / 2)) < width * 0.18
            wide_enough = run_length > width * 0.28
            row_group = 0
            for scan_y in range(y, min(end_y, y + 10)):
                dark_pixels = 0
                for scan_x in range(run_start, min(width, run_start + run_length)):
                    if grayscale.getpixel((scan_x, scan_y)) < 185:
                        dark_pixels += 1
                if dark_pixels / max(run_length, 1) > 0.65:
                    row_group += 1
            thin_group = row_group >= 4 and row_group <= 10
            if centered and wide_enough and thin_group:
                return (
                    "Styled marketplace image appears to contain a flat underline-like shadow bar. "
                    "Use only a soft natural grounding shadow; reject thick horizontal bars beneath the product."
                )

        return None

    def _cleanup_cutout_artifacts(self, payload: bytes) -> bytes:
        if Image is None:
            return payload

        with Image.open(BytesIO(payload)) as image:
            rgba = image.convert("RGBA")
            alpha = rgba.getchannel("A")
            width, height = rgba.size
            visited: set[tuple[int, int]] = set()
            components: list[dict[str, object]] = []
            pixels = alpha.load()

            for y in range(height):
                for x in range(width):
                    if pixels[x, y] == 0 or (x, y) in visited:
                        continue
                    stack = [(x, y)]
                    visited.add((x, y))
                    coords: list[tuple[int, int]] = []
                    min_x = max_x = x
                    min_y = max_y = y
                    while stack:
                        cx, cy = stack.pop()
                        coords.append((cx, cy))
                        min_x = min(min_x, cx)
                        max_x = max(max_x, cx)
                        min_y = min(min_y, cy)
                        max_y = max(max_y, cy)
                        for nx, ny in ((cx - 1, cy), (cx + 1, cy), (cx, cy - 1), (cx, cy + 1)):
                            if nx < 0 or ny < 0 or nx >= width or ny >= height:
                                continue
                            if pixels[nx, ny] == 0 or (nx, ny) in visited:
                                continue
                            visited.add((nx, ny))
                            stack.append((nx, ny))
                    components.append(
                        {
                            "coords": coords,
                            "area": len(coords),
                            "min_x": min_x,
                            "max_x": max_x,
                            "min_y": min_y,
                            "max_y": max_y,
                        }
                    )

            if not components:
                return payload

            components.sort(key=lambda item: int(item["area"]), reverse=True)
            primary = components[0]
            cleaned = rgba.copy()
            cleaned_pixels = cleaned.load()

            for component in components[1:]:
                area = int(component["area"])
                min_y = int(component["min_y"])
                max_y = int(component["max_y"])
                max_x = int(component["max_x"])
                min_x = int(component["min_x"])
                component_height = max_y - min_y + 1
                component_width = max_x - min_x + 1
                is_low = min_y > int(height * 0.58)
                is_small = area < max(5000, int(primary["area"]) * 0.12)
                is_short_banner = component_height < int(height * 0.12) and component_width < int(width * 0.65)
                if is_low and is_small and is_short_banner:
                    for cx, cy in component["coords"]:
                        cleaned_pixels[cx, cy] = (0, 0, 0, 0)

            buffer = BytesIO()
            cleaned.save(buffer, format="PNG")
            return buffer.getvalue()

    def _detect_cutout_text_artifacts(self, image) -> str | None:
        if Image is None:
            return None

        alpha = image.getchannel("A")
        width, height = alpha.size
        if width < 160 or height < 160:
            return None

        dark_alpha_rows = 0
        start_y = int(height * 0.68)
        for y in range(start_y, height):
            opaque_pixels = 0
            for x in range(width):
                if alpha.getpixel((x, y)) > 0:
                    opaque_pixels += 1
            if opaque_pixels / max(width, 1) > 0.1:
                dark_alpha_rows += 1

        if dark_alpha_rows >= 8:
            return (
                "Transparent cutout appears to contain bottom text/banner artifacts. "
                "Only the product should remain in the transparent asset."
            )
        return None
