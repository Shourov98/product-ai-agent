from __future__ import annotations

import colorsys
from io import BytesIO
from pathlib import Path
from typing import Literal

from app.schemas.response import (
    AmazonResponse,
    CoreProductResponse,
    EbayResponse,
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

MarketplaceName = Literal["source", "transparent_cutout", "amazon", "ebay", "tiktok", "shopify"]


class ImageAgent:
    _PROFILES = {
        "amazon": {
            "size": "1024x1024",
            "background": "white",
            "prompt_prefix": (
                "Create a production-grade Amazon main image. Use a pure white background. "
                "Show a single product only. No text, no watermark, no props, no lifestyle scene. "
                "Keep edges crisp and the product centered."
            ),
        },
        "ebay": {
            "size": "1024x1024",
            "background": "white",
            "prompt_prefix": (
                "Create a clean eBay-ready product image. Use a neutral white studio background. "
                "Single product only. No text overlays, no badges, no decorative props. "
                "Keep the product realistic and fully visible."
            ),
        },
        "tiktok": {
            "size": "1024x1536",
            "background": "opaque",
            "prompt_prefix": (
                "Create a vertical TikTok Shop hero image by changing only the background and scene styling. "
                "Do not change the product itself. Preserve the exact product color, material, silhouette, proportions, finish, and visible details. "
                "Do not redesign, recolor, restyle, reshape, or replace the product. "
                "Keep the product as the visual focus. Use a premium styled background with modern lighting, "
                "but do not add text or logos."
            ),
        },
        "shopify": {
            "size": "1536x1536",
            "background": "opaque",
            "prompt_prefix": (
                "Create a polished Shopify storefront hero image by changing only the background and scene styling. "
                "Preserve the exact product color, material, silhouette, proportions, finish, and visible details. "
                "Do not redesign, recolor, restyle, reshape, or replace the product. "
                "Use a premium ecommerce background and tasteful lighting, but do not add text, logos, or badges."
            ),
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
            tiktok=tiktok_asset,
            shopify=shopify_asset,
        )

    async def regenerate_marketplace_asset(
        self,
        *,
        marketplace: Literal["amazon", "ebay", "tiktok", "shopify"],
        source: ImagePayload,
        existing_images: GeneratedImagesResponse,
        core_data: CoreProductResponse,
        amazon_data: AmazonResponse,
        ebay_data: EbayResponse,
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
        marketplace: Literal["amazon", "ebay", "tiktok", "shopify"],
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
            absolute_path = output_service.save_binary(run_dir, relative_path, image_bytes)
            validation = self._validate_bytes(
                image_bytes,
                mime_type="image/png",
                expected_width=expected_width,
                expected_height=expected_height,
                background=profile["background"],
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
            absolute_path = output_service.save_binary(run_dir, relative_path, source.data)
            validation = self._validate_bytes(
                source.data,
                mime_type=source.content_type,
                expected_width=expected_width,
                expected_height=expected_height,
                background=profile["background"],
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
        absolute_path = output_service.save_binary(run_dir, relative_path, image.data)
        validation = self._validate_bytes(
            image.data,
            mime_type=image.content_type,
            expected_width=None,
            expected_height=None,
            background="source",
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
            absolute_path = output_service.save_binary(run_dir, relative_path, image_bytes)
            validation = self._validate_bytes(
                image_bytes,
                mime_type="image/png",
                expected_width=1024,
                expected_height=1024,
                background="transparent",
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
            absolute_path = output_service.save_binary(run_dir, relative_path, source.data)
            validation = self._validate_bytes(
                source.data,
                mime_type=source.content_type,
                expected_width=expected_width,
                expected_height=expected_height,
                background=profile["background"],
                errors=["Transparent cutout was unavailable; white-background composition could not be performed."],
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

        try:
            image_bytes = self._compose_cutout_on_background(
                cutout_path=cutout_path,
                width=expected_width,
                height=expected_height,
                background=(255, 255, 255, 255),
            )
            absolute_path = output_service.save_binary(run_dir, relative_path, image_bytes)
            validation = self._validate_bytes(
                image_bytes,
                mime_type="image/png",
                expected_width=expected_width,
                expected_height=expected_height,
                background=profile["background"],
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
            absolute_path = output_service.save_binary(run_dir, relative_path, source.data)
            validation = self._validate_bytes(
                source.data,
                mime_type=source.content_type,
                expected_width=expected_width,
                expected_height=expected_height,
                background=profile["background"],
                errors=[f"White-background composition failed: {exc}"],
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

    async def _build_marketplace_variant(
        self,
        *,
        marketplace: Literal["tiktok", "shopify"],
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

        if self.openai_service is None or not self.openai_service.enabled:
            absolute_path = output_service.save_binary(run_dir, relative_path, source.data)
            validation = self._validate_bytes(
                source.data,
                mime_type=source.content_type,
                expected_width=self._size_to_width(profile["size"]),
                expected_height=self._size_to_height(profile["size"]),
                background=profile["background"],
                errors=["OpenAI image generation disabled; saved source image as fallback."],
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

        try:
            with open(base_image_path, "rb") as handle:
                base_bytes = handle.read()
            image_bytes = await self.openai_service.edit_image(
                prompt=prompt,
                image_bytes=base_bytes,
                filename=Path(base_image_path).name,
                mime_type="image/png" if base_image_path.endswith(".png") else source.content_type,
                size=profile["size"],
                background=profile["background"],
            )
            absolute_path = output_service.save_binary(run_dir, relative_path, image_bytes)
            validation = self._validate_bytes(
                image_bytes,
                mime_type="image/png",
                expected_width=self._size_to_width(profile["size"]),
                expected_height=self._size_to_height(profile["size"]),
                background=profile["background"],
            )
            return ImageVariantResponse(
                marketplace=marketplace,
                relative_path=relative_path,
                absolute_path=str(absolute_path),
                prompt=prompt,
                generation_mode="edited",
                mime_type="image/png",
                validation=validation,
            )
        except OpenAIServiceError as exc:
            absolute_path = output_service.save_binary(run_dir, relative_path, source.data)
            validation = self._validate_bytes(
                source.data,
                mime_type=source.content_type,
                expected_width=self._size_to_width(profile["size"]),
                expected_height=self._size_to_height(profile["size"]),
                background=profile["background"],
                errors=[f"OpenAI image edit failed: {exc}"],
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

        with Image.open(cutout_path) as cutout:
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
        with Image.open(cutout_path) as cutout:
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
    def _background_for_marketplace(
        marketplace: Literal["amazon", "ebay", "tiktok", "shopify"],
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

    def _validate_bytes(
        self,
        payload: bytes,
        *,
        mime_type: str,
        expected_width: int | None,
        expected_height: int | None,
        background: str,
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
                    sampled = [
                        rgb_image.getpixel((0, 0)),
                        rgb_image.getpixel((rgb_image.width - 1, 0)),
                        rgb_image.getpixel((0, rgb_image.height - 1)),
                        rgb_image.getpixel((rgb_image.width - 1, rgb_image.height - 1)),
                    ]
                    for red, green, blue in sampled:
                        if min(red, green, blue) < 245:
                            collected_errors.append("Expected white background, but corner pixels are not close to white.")
                            break
            except Exception as exc:
                collected_errors.append(f"Could not verify white background: {exc}")

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
