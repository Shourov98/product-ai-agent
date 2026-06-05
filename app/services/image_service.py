from __future__ import annotations

import colorsys
from dataclasses import dataclass
from hashlib import md5
from io import BytesIO

try:
    from PIL import Image
except ImportError:  # pragma: no cover - runtime dependency issue
    Image = None


@dataclass(slots=True)
class ImagePayload:
    filename: str
    content_type: str
    data: bytes

    @property
    def size_bytes(self) -> int:
        return len(self.data)


class ImageService:
    """Deterministic helper for image metadata extraction with real pixel analysis."""

    _COLOR_KEYWORDS = {
        "black": "black",
        "white": "white",
        "red": "red",
        "blue": "blue",
        "green": "green",
        "yellow": "yellow",
        "pink": "pink",
        "gray": "gray",
        "grey": "gray",
        "brown": "brown",
        "beige": "beige",
        "orange": "orange",
    }

    def describe_brightness(self, payload: ImagePayload) -> str:
        image = self._load_image(payload)
        if image is not None:
            grayscale = image.convert("L").resize((64, 64))
            pixels = list(grayscale.getdata())
            if pixels:
                average = sum(pixels) / len(pixels)
                if average < 85:
                    return "dark"
                if average > 170:
                    return "light"
                return "balanced"

        if not payload.data:
            return "balanced"

        average = sum(payload.data) / len(payload.data)
        if average < 85:
            return "dark"
        if average > 170:
            return "light"
        return "balanced"

    def extract_palette(self, payload: ImagePayload) -> list[str]:
        image = self._load_image(payload)
        if image is not None:
            palette = self._extract_palette_from_pixels(image)
            if palette:
                return palette

        lowered_name = payload.filename.lower()
        palette = [
            color
            for keyword, color in self._COLOR_KEYWORDS.items()
            if keyword in lowered_name
        ]
        if palette:
            return palette[:3]

        digest = md5(payload.data or payload.filename.encode("utf-8")).hexdigest()
        fallback = ["black", "white", "gray", "blue", "red", "green"]
        index = int(digest[:2], 16) % len(fallback)
        second_index = (index + 2) % len(fallback)
        return [fallback[index], fallback[second_index]]

    def remove_background(self, payload: ImagePayload) -> bool:
        # Simulated background removal marker. In production, this would call
        # a segmentation or editing service and persist the processed asset.
        return payload.size_bytes > 0

    def _load_image(self, payload: ImagePayload):
        if Image is None or not payload.data:
            return None

        try:
            image = Image.open(BytesIO(payload.data))
            return image.convert("RGB")
        except Exception:
            return None

    def _extract_palette_from_pixels(self, image) -> list[str]:
        sample = image.copy()
        sample.thumbnail((160, 160))
        quantized = sample.quantize(colors=6, method=Image.Quantize.MEDIANCUT)
        color_counts = quantized.getcolors()
        if not color_counts:
            return []

        palette = quantized.getpalette()
        ranked: list[tuple[int, str]] = []
        for count, palette_index in sorted(color_counts, reverse=True):
            base = palette_index * 3
            red, green, blue = palette[base : base + 3]
            color_name = self._classify_rgb(red, green, blue)
            if color_name is None:
                continue
            ranked.append((count, color_name))

        filtered = self._filter_background_like_colors(ranked)
        deduped: list[str] = []
        seen = set()
        for _, color_name in filtered:
            if color_name in seen:
                continue
            seen.add(color_name)
            deduped.append(color_name)
            if len(deduped) >= 3:
                break
        return deduped

    @staticmethod
    def _filter_background_like_colors(ranked: list[tuple[int, str]]) -> list[tuple[int, str]]:
        if len(ranked) <= 1:
            return ranked

        non_background = [item for item in ranked if item[1] not in {"white", "ivory", "light gray", "gray"}]
        return non_background or ranked

    def _classify_rgb(self, red: int, green: int, blue: int) -> str | None:
        r_norm, g_norm, b_norm = red / 255, green / 255, blue / 255
        hue, lightness, saturation = colorsys.rgb_to_hls(r_norm, g_norm, b_norm)
        hue_degrees = hue * 360

        if lightness <= 0.12:
            return "black"
        if saturation <= 0.08:
            if lightness >= 0.93:
                return "white"
            if lightness >= 0.82:
                return "ivory"
            if lightness >= 0.68:
                return "light gray"
            if lightness >= 0.38:
                return "gray"
            return "charcoal gray"

        if 10 <= hue_degrees < 35:
            if lightness < 0.42:
                return "rust orange"
            if lightness > 0.72:
                return "peach"
            return "orange"

        if 35 <= hue_degrees < 50:
            if lightness < 0.45:
                return "mustard yellow"
            if lightness > 0.75:
                return "cream"
            return "golden yellow"

        if 50 <= hue_degrees < 75:
            if lightness > 0.72:
                return "beige"
            return "olive green"

        if 75 <= hue_degrees < 160:
            if lightness < 0.36:
                return "forest green"
            if saturation < 0.25:
                return "sage green"
            return "green"

        if 160 <= hue_degrees < 200:
            if lightness < 0.42:
                return "deep teal"
            return "teal"

        if 200 <= hue_degrees < 255:
            if lightness < 0.36:
                return "navy blue"
            if saturation < 0.3:
                return "slate blue"
            return "blue"

        if 255 <= hue_degrees < 290:
            if lightness < 0.42:
                return "indigo"
            return "violet"

        if 290 <= hue_degrees < 345:
            if lightness < 0.42:
                return "plum"
            if lightness > 0.75:
                return "blush pink"
            return "pink"

        if saturation < 0.22 and lightness < 0.45:
            return "brown"
        if lightness < 0.4:
            return "burgundy"
        if lightness > 0.78:
            return "soft pink"
        return "red"
