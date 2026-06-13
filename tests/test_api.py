from __future__ import annotations

import pytest
from httpx import ASGITransport, AsyncClient

from app.main import app


@pytest.mark.anyio
async def test_healthcheck() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.get("/api/health")

    assert response.status_code == 200
    assert response.json() == {"status": "ok"}


@pytest.mark.anyio
async def test_generate_endpoint_returns_structured_response() -> None:
    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/generate",
            data={"title": "Pro Runner Sneaker"},
            files={
                "image": (
                    "black-mesh-sneaker.jpg",
                    b"\x10\x18\x20\x28" * 32,
                    "image/jpeg",
                )
            },
        )

    body = response.json()

    assert response.status_code == 200
    assert body["core"]["product_type"] == "running shoes"
    assert "amazon" in body
    assert "tiktok" in body
    assert "ebay" in body
    assert "shopify" in body
    assert "images" in body
    assert "source" in body["images"]
    assert "amazon" in body["images"]
    assert "shopify" in body["images"]


@pytest.mark.anyio
async def test_product_crud_variant_and_regeneration_flow(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("PRODUCT_STORE_DIR", str(tmp_path / "products"))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        create_response = await client.post(
            "/api/products/generate/text",
            data={"title": "Pro Runner Sneaker"},
            files={
                "image": (
                    "black-mesh-sneaker.jpg",
                    b"\x10\x18\x20\x28" * 32,
                    "image/jpeg",
                )
            },
        )

        assert create_response.status_code == 200
        created = create_response.json()
        product_id = created["id"]

        get_response = await client.get(f"/api/products/{product_id}")
        assert get_response.status_code == 200
        assert get_response.json()["id"] == product_id

        update_response = await client.patch(
            f"/api/products/{product_id}",
            json={
                "core": {"normalized_title": "Updated Runner Sneaker"},
                "shopify": {"seo_title": "Updated SEO Title"},
            },
        )
        assert update_response.status_code == 200
        updated = update_response.json()
        assert updated["product"]["core"]["normalized_title"] == "Updated Runner Sneaker"
        assert updated["product"]["shopify"]["seo_title"] == "Updated SEO Title"

        size_variant_response = await client.post(
            f"/api/products/{product_id}/marketplaces/shopify/variants/size",
            json={"name": "XL"},
        )
        assert size_variant_response.status_code == 200
        size_variant_body = size_variant_response.json()
        assert size_variant_body["variants"]["shopify"][0]["variant_type"] == "size"

        color_variant_response = await client.post(
            f"/api/products/{product_id}/marketplaces/amazon/variants/color",
            json={"name": "Forest Green"},
        )
        assert color_variant_response.status_code == 200
        color_variant_body = color_variant_response.json()
        assert color_variant_body["variants"]["amazon"][0]["variant_type"] == "color"
        assert color_variant_body["variants"]["amazon"][0]["image"]["absolute_path"]

        regenerate_response = await client.post(
            f"/api/products/{product_id}/marketplaces/amazon/regenerate"
        )
        assert regenerate_response.status_code == 200
        regenerated = regenerate_response.json()
        assert regenerated["product"]["amazon"]["title"]

        list_response = await client.get("/api/products")
        assert list_response.status_code == 200
        listed = list_response.json()
        assert any(item["id"] == product_id for item in listed["items"])


@pytest.mark.anyio
async def test_text_only_generation_creates_pending_marketplace_images(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("PRODUCT_STORE_DIR", str(tmp_path / "products"))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/products/generate/text",
            data={"title": "Pro Runner Sneaker"},
            files={
                "image": (
                    "black-mesh-sneaker.jpg",
                    b"\x10\x18\x20\x28" * 32,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["product"]["core"]["normalized_title"]
    assert body["product"]["images"]["source"]["absolute_path"]
    assert body["product"]["images"]["amazon"]["absolute_path"] == ""
    assert body["product"]["images"]["amazon"]["generation_mode"] == "pending_marketplace_generation"


@pytest.mark.anyio
async def test_single_marketplace_generation_only_generates_requested_image(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("PRODUCT_STORE_DIR", str(tmp_path / "products"))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        response = await client.post(
            "/api/products/generate/amazon",
            data={"title": "Pro Runner Sneaker"},
            files={
                "image": (
                    "black-mesh-sneaker.jpg",
                    b"\x10\x18\x20\x28" * 32,
                    "image/jpeg",
                )
            },
        )

    assert response.status_code == 200
    body = response.json()
    assert body["product"]["images"]["amazon"]["absolute_path"]
    assert body["product"]["images"]["ebay"]["absolute_path"] == ""
    assert body["product"]["images"]["shopify"]["absolute_path"] == ""


@pytest.mark.anyio
async def test_marketplace_generate_endpoint_updates_requested_marketplace(monkeypatch, tmp_path) -> None:
    monkeypatch.setenv("OUTPUT_DIR", str(tmp_path / "output"))
    monkeypatch.setenv("PRODUCT_STORE_DIR", str(tmp_path / "products"))

    async with AsyncClient(
        transport=ASGITransport(app=app),
        base_url="http://testserver",
    ) as client:
        create_response = await client.post(
            "/api/products/generate/text",
            data={"title": "Pro Runner Sneaker"},
            files={
                "image": (
                    "black-mesh-sneaker.jpg",
                    b"\x10\x18\x20\x28" * 32,
                    "image/jpeg",
                )
            },
        )
        assert create_response.status_code == 200
        product_id = create_response.json()["id"]

        generate_images_response = await client.post(
            f"/api/products/{product_id}/marketplaces/amazon/generate",
        )

    assert generate_images_response.status_code == 200
    body = generate_images_response.json()
    assert body["product"]["images"]["amazon"]["absolute_path"]
    assert body["product"]["amazon"]["title"]
    assert body["product"]["images"]["ebay"]["absolute_path"] == ""
