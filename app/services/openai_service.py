from __future__ import annotations

from typing import Any


class OpenAIService:
    """
    Placeholder service boundary for future structured generation.

    The current project is intentionally deterministic; this class exists so
    marketplace agents can later swap local heuristics for a model-backed
    implementation without changing orchestration boundaries.
    """

    async def generate_structured_output(
        self,
        *,
        system_prompt: str,
        user_payload: dict[str, Any],
    ) -> dict[str, Any]:
        raise NotImplementedError(
            "Structured generation is not configured in this deterministic build."
        )
