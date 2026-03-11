from __future__ import annotations

from dataclasses import dataclass
from hashlib import md5


@dataclass(slots=True)
class ImagePayload:
    filename: str
    content_type: str
    data: bytes

    @property
    def size_bytes(self) -> int:
        return len(self.data)


class ImageService:
    """Small deterministic helper for image metadata extraction."""

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
        if not payload.data:
            return "balanced"

        average = sum(payload.data) / len(payload.data)
        if average < 85:
            return "dark"
        if average > 170:
            return "light"
        return "balanced"

    def extract_palette(self, payload: ImagePayload) -> list[str]:
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
