import asyncio
import json
import os
from pathlib import Path

import httpx

root = Path(r"C:\Miskat\siyam\product-ai-agent")
os.chdir(root)

from app.config import get_settings
from app.utils.prompts import PromptRegistry


async def main() -> None:
    settings = get_settings()
    user_payload = {
        "marketplace": "shopify",
        "product_identity": "Gigabyte GeForce RTX 5070 Ti Graphics Card",
        "source_title": "Gigabyte GeForce RTX 5070 Ti Graphics Card",
        "category": "Graphics Card",
        "product_type": "GPU",
        "product_summary": "Gigabyte GeForce RTX 5070 Ti triple-fan desktop graphics card for gaming and creative workloads.",
        "features": [],
        "attributes": {
            "brand": "Gigabyte",
            "color": "Black",
            "material": "Metal and plastic",
        },
        "existing_search_queries": [
            "Gigabyte GeForce RTX 5070 Ti Graphics Card",
            "Gigabyte GPU",
        ],
    }
    user_text = "\n".join(f"{key}: {json.dumps(value) if isinstance(value, (dict, list, tuple)) else value}" for key, value in user_payload.items())
    payload = {
        "systemInstruction": {"parts": [{"text": PromptRegistry.get_gemini_pricing_search_prompt()}]},
        "contents": [{"role": "user", "parts": [{"text": user_text}]}],
        "generationConfig": {"temperature": 0.2},
        "tools": [{"google_search": {}}],
    }
    url = f"{settings.gemini_api_base_url.rstrip('/')}/models/{settings.gemini_model}:generateContent"
    params = {"key": settings.gemini_api_key}
    async with httpx.AsyncClient(timeout=45.0) as client:
        response = await client.post(url, params=params, json=payload)
    print(response.status_code)
    print(response.text)


asyncio.run(main())
