"""PatchTSMixer wrapper dependency check."""

from __future__ import annotations


def require_patchtsmixer():
    try:
        from transformers import PatchTSMixerForTimeSeriesClassification  # type: ignore
    except Exception as exc:  # pragma: no cover - environment dependent
        raise RuntimeError(
            "PatchTSMixer experiment cannot run because Hugging Face Transformers "
            "with PatchTSMixerForTimeSeriesClassification is not importable."
        ) from exc
    return PatchTSMixerForTimeSeriesClassification


def train_patchtsmixer(*args, **kwargs):
    require_patchtsmixer()
    raise NotImplementedError("PatchTSMixer training requires torch/transformers runtime support.")
