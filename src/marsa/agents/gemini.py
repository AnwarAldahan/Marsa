"""Gemini implementation of the vendor-neutral structured LLM provider."""

from __future__ import annotations

import json
import os
from pathlib import Path
from typing import Any

import httpx
from dotenv import dotenv_values
from pydantic import BaseModel, SecretStr

from marsa.common.paths import resolve_project_path


def _gemini_response_schema(value: Any) -> Any:
    """Remove JSON Schema annotations unsupported by Gemini responseSchema."""
    if isinstance(value, dict):
        return {
            key: _gemini_response_schema(item)
            for key, item in value.items()
            if key != "additionalProperties"
        }
    if isinstance(value, list):
        return [_gemini_response_schema(item) for item in value]
    return value


class GeminiStructuredProvider:
    """Call Gemini generateContent with JSON-schema constrained output."""

    endpoint = "https://generativelanguage.googleapis.com/v1beta/models/{model}:generateContent"

    def __init__(
        self,
        api_key: SecretStr,
        *,
        model: str = "gemini-3.5-flash-lite",
        timeout_seconds: float = 30.0,
        client: httpx.Client | None = None,
    ) -> None:
        if not api_key.get_secret_value():
            raise ValueError("Gemini API key is required")
        self._api_key = api_key
        self.model = model
        self.timeout_seconds = timeout_seconds
        self._client = client

    @classmethod
    def from_env(cls, env_path: str | Path = ".env") -> GeminiStructuredProvider:
        values = dotenv_values(resolve_project_path(env_path))
        api_key = os.getenv("GEMINI_API_KEY") or values.get("GEMINI_API_KEY")
        model = (
            os.getenv("MARSA_GEMINI_MODEL")
            or values.get("MARSA_GEMINI_MODEL")
            or os.getenv("GEMINI_MODEL")
            or values.get("GEMINI_MODEL")
            or "gemini-3.5-flash-lite"
        )
        if not api_key:
            raise ValueError("GEMINI_API_KEY is not configured")
        return cls(SecretStr(str(api_key)), model=str(model))

    def generate_structured(
        self,
        *,
        system_prompt: str,
        payload: dict[str, Any],
        response_model: type[BaseModel],
    ) -> str:
        request_body = {
            "systemInstruction": {"parts": [{"text": system_prompt}]},
            "contents": [
                {
                    "role": "user",
                    "parts": [{"text": json.dumps(payload, separators=(",", ":"))}],
                }
            ],
            "generationConfig": {
                "temperature": 0.0,
                "responseMimeType": "application/json",
                "responseSchema": _gemini_response_schema(response_model.model_json_schema()),
            },
        }
        headers = {
            "Content-Type": "application/json",
            "x-goog-api-key": self._api_key.get_secret_value(),
        }
        url = self.endpoint.format(model=self.model)

        if self._client is not None:
            response = self._client.post(url, headers=headers, json=request_body)
        else:
            with httpx.Client(timeout=self.timeout_seconds) as client:
                response = client.post(url, headers=headers, json=request_body)
        if response.status_code >= 400:
            raise RuntimeError(f"Gemini API request failed with HTTP {response.status_code}")

        body = response.json()
        try:
            parts = body["candidates"][0]["content"]["parts"]
            text = "".join(part.get("text", "") for part in parts)
        except (KeyError, IndexError, TypeError) as error:
            raise RuntimeError("Gemini API response contained no structured candidate") from error
        if not text:
            raise RuntimeError("Gemini API response contained an empty candidate")
        return text
