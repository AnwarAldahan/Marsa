"""Vendor-neutral interface for optional structured LLM generation."""

from __future__ import annotations

from typing import Any, Protocol

from pydantic import BaseModel


class StructuredLLMProvider(Protocol):
    """Provider boundary; implementations own credentials, transport, and model selection."""

    def generate_structured(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        response_model: type[BaseModel],
    ) -> BaseModel | dict[str, Any] | str:
        ...
