from __future__ import annotations

from dataclasses import dataclass
from urllib.parse import quote

import boto3
from botocore.config import Config


@dataclass(slots=True)
class S3Asset:
    url: str
    key: str


class S3Service:
    def __init__(
        self,
        *,
        region: str | None,
        bucket_name: str | None,
        access_key_id: str | None,
        secret_access_key: str | None,
        prefix: str = "product-ai-agent",
    ) -> None:
        self.enabled = bool(region and bucket_name and access_key_id and secret_access_key)
        self.region = region or ""
        self.bucket_name = bucket_name or ""
        self.prefix = prefix.strip("/") if prefix else "product-ai-agent"
        self.client = None
        if self.enabled:
            self.client = boto3.client(
                "s3",
                region_name=self.region,
                aws_access_key_id=access_key_id,
                aws_secret_access_key=secret_access_key,
                config=Config(signature_version="s3v4"),
            )

    def upload_bytes(
        self,
        payload: bytes,
        *,
        key: str,
        mime_type: str,
    ) -> S3Asset:
        if not self.enabled or self.client is None:
            raise RuntimeError("S3 is not configured.")

        object_key = f"{self.prefix}/{key.strip('/')}"
        self.client.put_object(
            Bucket=self.bucket_name,
            Key=object_key,
            Body=payload,
            ContentType=mime_type,
        )
        encoded_object_key = quote(object_key, safe="/")
        url = f"https://{self.bucket_name}.s3.{self.region}.amazonaws.com/{encoded_object_key}"
        return S3Asset(url=url, key=object_key)
