"""HERE: LLM PROVIDER (Gemini / Claude) - narrative only, never numbers.
Optional LLM provider hook. The MVP runs fully deterministically without it.

To plug in a provider, implement `complete(prompt: str) -> str` and pass it to StrategyAgent.
Never let the LLM change numbers: it only writes the human-readable narrative.
"""
from __future__ import annotations
from typing import Protocol


class LLMProvider(Protocol):
    def complete(self, prompt: str) -> str: ...


class NoLLM:
    def complete(self, prompt: str) -> str:
        return ""
