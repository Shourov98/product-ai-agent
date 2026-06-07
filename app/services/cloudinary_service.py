from __future__ import annotations

import base64
from dataclasses import dataclass

import cloudinary
import cloudinary.uploader


@dataclass(slots=True)
class CloudinaryAsset:
    secure_url: str
    public_id: str


class CloudinaryService:
    def __init__(
        self,
        *,
        cloud_name: str | None,
        api_key: str | None,
        api_secret: str | None,
        folder: str,
        secure: bool = True,
    ) -> None:
        self.enabled = bool(cloud_name and api_key and api_secret)
        self.folder = folder.strip("/") if folder else "product-ai-agent"
        if self.enabled:
            cloudinary.config(
                cloud_name=cloud_name,
                api_key=api_key,
                api_secret=api_secret,
                secure=secure,
            )

    def upload_bytes(
        self,
        payload: bytes,
        *,
        public_id: str,
        mime_type: str,
    ) -> CloudinaryAsset:
        if not self.enabled:
            raise RuntimeError("Cloudinary is not configured.")

        encoded = base64.b64encode(payload).decode("ascii")
        result = cloudinary.uploader.upload(
            f"data:{mime_type};base64,{encoded}",
            public_id=f"{self.folder}/{public_id.strip('/')}",
            overwrite=True,
            resource_type="image",
            invalidate=True,
        )
        return CloudinaryAsset(
            secure_url=str(result["secure_url"]),
            public_id=str(result["public_id"]),
        )
