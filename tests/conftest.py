"""Keep automated tests offline even when a developer has Gemini credentials."""
import httpx
import pytest


@pytest.fixture(autouse=True)
def block_live_http(monkeypatch):
    def refuse_request(self, request):
        raise AssertionError("Live HTTP is disabled in tests; use httpx.MockTransport")

    monkeypatch.setattr(httpx.HTTPTransport, "handle_request", refuse_request)
