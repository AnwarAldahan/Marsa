"""GRU wrapper dependency check.

The architecture is intentionally deferred to a torch-backed implementation
when PyTorch is available. In this environment PyTorch is not installed, so the
runner records an explicit failed experiment instead of inventing results.
"""

from __future__ import annotations


def require_torch():
    try:
        import torch  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError("GRU experiment cannot run because `torch` is not importable in this environment.") from exc
    return torch


def train_gru(*args, **kwargs):
    require_torch()
    raise NotImplementedError("Torch is available, but the compact GRU training loop must be enabled for this environment.")
