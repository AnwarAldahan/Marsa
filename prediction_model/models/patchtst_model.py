"""PatchTST wrapper dependency check."""

from __future__ import annotations


def require_patchtst():
    try:
        from transformers import PatchTSTForClassification  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PatchTST experiment cannot run because Hugging Face Transformers "
            "with PatchTSTForClassification is not importable."
        ) from exc
    return PatchTSTForClassification


def train_patchtst(*args, **kwargs):
    require_patchtst()
    raise NotImplementedError("PatchTST training requires torch/transformers runtime support.")
