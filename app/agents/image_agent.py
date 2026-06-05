from __future__ import annotations

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
    TikTokResponse,
)
from app.services.image_service import ImagePayload
from app.services.openai_service import OpenAIService, OpenAIServiceError

try:
    from PIL import Image
except ImportError:  # pragma: no cover - dependency missing at runtime
    Image = None

MarketplaceName = Literal["source", "transparent_cutout", "amazon", "ebay", "tiktok"]


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

        return GeneratedImagesResponse(
            source=source_asset,
            transparent_cutout=cutout_asset,
            amazon=amazon_asset,
            ebay=ebay_asset,
            tiktok=tiktok_asset,
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
        marketplace: Literal["tiktok"],
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
            f"Attributes: {attributes}."
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
