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
    local_output_enabled: bool = False
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
    cloudinary_cloud_name: str | None = None
    cloudinary_api_key: str | None = None
    cloudinary_api_secret: str | None = None
    cloudinary_folder: str = "product-ai-agent"
    cloudinary_secure: bool = True
    market_research_realtime_enabled: bool = False
    ebay_research_enabled: bool = False
    ebay_client_id: str | None = None
    ebay_client_secret: str | None = None
    ebay_marketplace_id: str = "EBAY_US"
    ebay_api_base_url: str = "https://api.ebay.com"
    ebay_identity_base_url: str = "https://api.ebay.com"
    ebay_search_limit: int = 5


def get_settings() -> Settings:
    mongodb_uri = os.getenv("MONGODB_URI")
    mongodb_enabled_default = "true" if mongodb_uri else "false"
    return Settings(
        debug=os.getenv("DEBUG", "false").lower() == "true",
        max_upload_size_bytes=int(
            os.getenv("MAX_UPLOAD_SIZE_BYTES", str(5 * 1024 * 1024))
        ),
        output_dir=os.getenv("OUTPUT_DIR", "output"),
        product_store_dir=os.getenv("PRODUCT_STORE_DIR", "output/products"),
        local_output_enabled=os.getenv("LOCAL_OUTPUT_ENABLED", "false").lower() == "true",
        auth_enabled=os.getenv("AUTH_ENABLED", "false").lower() == "true",
        jwt_access_secret=os.getenv("JWT_ACCESS_SECRET"),
        mongodb_enabled=os.getenv("MONGODB_ENABLED", mongodb_enabled_default).lower() == "true",
        mongodb_uri=mongodb_uri,
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
        cloudinary_cloud_name=os.getenv("CLOUDINARY_CLOUD_NAME"),
        cloudinary_api_key=os.getenv("CLOUDINARY_API_KEY"),
        cloudinary_api_secret=os.getenv("CLOUDINARY_API_SECRET"),
        cloudinary_folder=os.getenv("CLOUDINARY_FOLDER", "product-ai-agent"),
        cloudinary_secure=os.getenv("CLOUDINARY_SECURE", "true").lower() == "true",
        market_research_realtime_enabled=os.getenv("MARKET_RESEARCH_REALTIME_ENABLED", "false").lower() == "true",
        ebay_research_enabled=os.getenv("EBAY_RESEARCH_ENABLED", "false").lower() == "true",
        ebay_client_id=os.getenv("EBAY_CLIENT_ID"),
        ebay_client_secret=os.getenv("EBAY_CLIENT_SECRET"),
        ebay_marketplace_id=os.getenv("EBAY_MARKETPLACE_ID", "EBAY_US"),
        ebay_api_base_url=os.getenv("EBAY_API_BASE_URL", "https://api.ebay.com"),
        ebay_identity_base_url=os.getenv("EBAY_IDENTITY_BASE_URL", "https://api.ebay.com"),
        ebay_search_limit=int(os.getenv("EBAY_SEARCH_LIMIT", "5")),
    )
