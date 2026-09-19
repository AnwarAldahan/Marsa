from __future__ import annotations

import json

import httpx
from pydantic import BaseModel, SecretStr

from marsa.agents.gemini import GeminiStructuredProvider


class ExampleResponse(BaseModel):
    summary: str


def test_gemini_provider_uses_header_and_structured_schema():
    captured = {}

    def handler(request: httpx.Request) -> httpx.Response:
        captured["url"] = str(request.url)
        captured["key"] = request.headers.get("x-goog-api-key")
        captured["body"] = json.loads(request.content)
        return httpx.Response(
            200,
            json={
                "candidates": [
                    {"content": {"parts": [{"text": '{"summary":"safe"}'}]}}
                ]
            },
        )

    client = httpx.Client(transport=httpx.MockTransport(handler))
    provider = GeminiStructuredProvider(SecretStr("test-secret"), client=client)
    output = provider.generate_structured(
        system_prompt="System",
        payload={"value": 1},
        response_model=ExampleResponse,
    )

    assert output == '{"summary":"safe"}'
    assert captured["key"] == "test-secret"
    assert "test-secret" not in captured["url"]
    assert captured["body"]["generationConfig"]["responseMimeType"] == "application/json"
    assert captured["body"]["generationConfig"]["responseSchema"]["required"] == ["summary"]
    assert "additionalProperties" not in captured["body"]["generationConfig"]["responseSchema"]


def test_provider_repr_does_not_expose_secret():
    provider = GeminiStructuredProvider(SecretStr("test-secret"))
    assert "test-secret" not in repr(provider.__dict__)


def test_marsa_model_environment_override_takes_precedence(tmp_path, monkeypatch):
    env_file = tmp_path / ".env"
    env_file.write_text(
        "GEMINI_API_KEY=file-secret\nGEMINI_MODEL=legacy-model\n",
        encoding="utf-8",
    )
    monkeypatch.setenv("MARSA_GEMINI_MODEL", "marsa-model")
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    provider = GeminiStructuredProvider.from_env(env_file)
    assert provider.model == "marsa-model"
