from __future__ import annotations

import os
from pathlib import Path

from pydantic import BaseModel, Field
from dotenv import load_dotenv


load_dotenv(Path(__file__).resolve().parents[1] / ".env")


class Settings(BaseModel):
    app_name: str = "Product AI Agent"
    app_version: str = "0.1.0"
    debug: bool = False
    max_upload_size_bytes: int = Field(default=5 * 1024 * 1024)
    supported_marketplaces: tuple[str, ...] = ("amazon", "tiktok", "ebay", "shopify")
    output_dir: str = "output"
    product_store_dir: str = "output/products"
    auth_enabled: bool = False
    jwt_access_secret: str | None = None
    mongodb_enabled: bool = False
    mongodb_uri: str | None = None
    mongodb_db_name: str = "product_ai_agent"
    mongodb_products_collection: str = "products"
    mongodb_runs_collection: str = "product_runs"
    mongodb_users_collection: str = "users"
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
        product_store_dir=os.getenv("PRODUCT_STORE_DIR", "output/products"),
        auth_enabled=os.getenv("AUTH_ENABLED", "false").lower() == "true",
        jwt_access_secret=os.getenv("JWT_ACCESS_SECRET"),
        mongodb_enabled=os.getenv("MONGODB_ENABLED", "false").lower() == "true",
        mongodb_uri=os.getenv("MONGODB_URI"),
        mongodb_db_name=os.getenv("MONGODB_DB_NAME", "product_ai_agent"),
        mongodb_products_collection=os.getenv("MONGODB_PRODUCTS_COLLECTION", "products"),
        mongodb_runs_collection=os.getenv("MONGODB_RUNS_COLLECTION", "product_runs"),
        mongodb_users_collection=os.getenv("MONGODB_USERS_COLLECTION", "users"),
        ollama_enabled=os.getenv("OLLAMA_ENABLED", "true").lower() == "true",
        ollama_base_url=os.getenv("OLLAMA_BASE_URL", "http://127.0.0.1:11434"),
        ollama_model=os.getenv("OLLAMA_MODEL", "qwen3:8b"),
        openai_enabled=os.getenv("OPENAI_ENABLED", "false").lower() == "true",
        openai_api_key=os.getenv("OPENAI_API_KEY"),
        openai_model=os.getenv("OPENAI_MODEL", "gpt-5"),
        openai_image_model=os.getenv("OPENAI_IMAGE_MODEL", "gpt-image-1"),
    )
