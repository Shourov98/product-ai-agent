from __future__ import annotations

from pydantic import BaseModel, Field


class ProductGenerationRequest(BaseModel):
    title: str = Field(min_length=3, max_length=200)
