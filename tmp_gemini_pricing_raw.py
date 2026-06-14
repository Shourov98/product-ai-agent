import asyncio
import json
import os
import tempfile
from pathlib import Path

root = Path(r"C:\Miskat\siyam\product-ai-agent")
os.chdir(root)
workspace = Path(tempfile.mkdtemp(prefix="gemini-pricing-raw-"))
os.environ["OUTPUT_DIR"] = str(workspace / "output")
os.environ["PRODUCT_STORE_DIR"] = str(workspace / "products")
os.environ["IMPORT_STORE_DIR"] = str(workspace / "imports")
os.environ["MONGODB_ENABLED"] = "false"
os.environ["AUTH_ENABLED"] = "false"

from app.services.product_service import ProductService
from app.utils.prompts import PromptRegistry


async def main() -> None:
    service = ProductService()
    record = service.create_imported_draft(
        title="Gigabyte GeForce RTX 5070 Ti Graphics Card",
        sku="RTX5070TI-GIGA",
        brand="Gigabyte",
        category="Graphics Card",
        product_type="GPU",
        description="Gigabyte GeForce RTX 5070 Ti triple-fan desktop graphics card for gaming and creative workloads.",
        price="",
        quantity="1",
        color="Black",
        size="Standard",
        material="Metal and plastic",
        image_url="https://example.com/gpu.png",
    )
    core = record.product.core
    research = await service.pipeline.research.build_research_bundle(core)
    payload = await service.pipeline.gemini_service.generate_structured_output(
        system_prompt=PromptRegistry.get_gemini_pricing_search_prompt(),
        user_payload={
            "marketplace": "shopify",
            "product_identity": service._product_identity_label(core),
            "source_title": core.source_title,
            "category": core.category,
            "product_type": core.product_type,
            "product_summary": core.product_summary,
            "features": core.features,
            "attributes": core.attributes,
            "existing_search_queries": research.shopify.search_queries,
        },
        use_google_search=True,
    )
    print(json.dumps(payload, indent=2))


asyncio.run(main())
