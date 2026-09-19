"""Simple validation-only probability calibration."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from prediction_model.src.evaluation import EPS, brier_score, clip_probabilities, log_loss


def _logit(probabilities: np.ndarray) -> np.ndarray:
    p = clip_probabilities(probabilities)
    return np.log(p / (1.0 - p))


@dataclass
class PlattCalibrator:
    intercept: float = 0.0
    slope: float = 1.0

    def fit(self, probabilities: np.ndarray, y_true: np.ndarray, lr: float = 0.05, epochs: int = 1000) -> "PlattCalibrator":
        x = _logit(probabilities)
        y = np.asarray(y_true, dtype=float)
        if len(np.unique(y)) < 2:
            self.intercept = 0.0
            self.slope = 1.0
            return self
        a = 0.0
        b = 1.0
        for _ in range(epochs):
            z = a + b * x
            pred = 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))
            error = pred - y
            grad_a = float(error.mean())
            grad_b = float((error * x).mean())
            a -= lr * grad_a
            b -= lr * grad_b
        self.intercept = float(a)
        self.slope = float(b)
        return self

    def predict(self, probabilities: np.ndarray) -> np.ndarray:
        z = self.intercept + self.slope * _logit(probabilities)
        return clip_probabilities(1.0 / (1.0 + np.exp(-np.clip(z, -50, 50))))

    def to_dict(self) -> dict:
        return {"method": "platt_sigmoid", "intercept": self.intercept, "slope": self.slope}

    @classmethod
    def from_dict(cls, payload: dict) -> "PlattCalibrator":
        return cls(intercept=float(payload.get("intercept", 0.0)), slope=float(payload.get("slope", 1.0)))


def choose_calibration(
    validation_y: np.ndarray,
    validation_probabilities: np.ndarray,
) -> tuple[str, PlattCalibrator | None, dict[str, float]]:
    raw = clip_probabilities(validation_probabilities)
    raw_metrics = {"brier": brier_score(validation_y, raw), "log_loss": log_loss(validation_y, raw)}
    calibrator = PlattCalibrator().fit(raw, validation_y)
    calibrated = calibrator.predict(raw)
    calibrated_metrics = {
        "brier": brier_score(validation_y, calibrated),
        "log_loss": log_loss(validation_y, calibrated),
    }
    if calibrated_metrics["brier"] <= raw_metrics["brier"] and calibrated_metrics["log_loss"] <= raw_metrics["log_loss"]:
        return "platt_sigmoid", calibrator, {
            "raw_brier": raw_metrics["brier"],
            "raw_log_loss": raw_metrics["log_loss"],
            "calibrated_brier": calibrated_metrics["brier"],
            "calibrated_log_loss": calibrated_metrics["log_loss"],
        }
    return "none_raw", None, {
        "raw_brier": raw_metrics["brier"],
        "raw_log_loss": raw_metrics["log_loss"],
        "calibrated_brier": calibrated_metrics["brier"],
        "calibrated_log_loss": calibrated_metrics["log_loss"],
    }
