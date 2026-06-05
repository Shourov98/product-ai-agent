from __future__ import annotations

import os

from pydantic import BaseModel, Field


class Settings(BaseModel):
    app_name: str = "Product AI Agent"
    app_version: str = "0.1.0"
    debug: bool = False
    max_upload_size_bytes: int = Field(default=5 * 1024 * 1024)
    supported_marketplaces: tuple[str, ...] = ("amazon", "tiktok", "ebay")
    output_dir: str = "output"
    ollama_enabled: bool = True
    ollama_base_url: str = "http://127.0.0.1:11434"
    ollama_model: str = "qwen3:8b"
    openai_enabled: bool = False
    openai_api_key: str | None = None
    openai_model: str = "gpt-5"
    openai_image_model: str = "gpt-image-1"


def get_settings() -> Settings:
    return Settings(
        debug=os.getenv("DEBUG", "false").lower() == "true",
        max_upload_size_bytes=int(
            os.getenv("MAX_UPLOAD_SIZE_BYTES", str(5 * 1024 * 1024))
        ),
        output_dir=os.getenv("OUTPUT_DIR", "output"),
        ollama_enabled=os.getenv("OLLAMA_ENABLED", "true").lower() == "true",
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        openai_enabled=os.getenv("OPENAI_ENABLED", "false").lower() == "true",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5"),
        openai_image_model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
    )
