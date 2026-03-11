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
