"""Pure NumPy logistic regression baseline with class weighting."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
import json
import time

import numpy as np

from prediction_model.src.evaluation import log_loss


def _sigmoid(z: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(z, -50, 50)))


@dataclass
class NumpyLogisticRegression:
    l2: float = 0.01
    learning_rate: float = 0.03
    epochs: int = 600
    random_seed: int = 42
    class_weight: str = "balanced"
    weights: list[float] | None = None
    intercept: float = 0.0
    training_loss: list[float] | None = None

    def fit(self, x: np.ndarray, y: np.ndarray, x_val: np.ndarray | None = None, y_val: np.ndarray | None = None) -> "NumpyLogisticRegression":
        rng = np.random.default_rng(self.random_seed)
        x = np.asarray(x, dtype=float)
        y = np.asarray(y, dtype=float)
        n_samples, n_features = x.shape
        weights = rng.normal(0, 0.01, size=n_features)
        intercept = 0.0

        if self.class_weight == "balanced":
            pos = max(float(y.sum()), 1.0)
            neg = max(float(len(y) - y.sum()), 1.0)
            sample_weight = np.where(y == 1, len(y) / (2 * pos), len(y) / (2 * neg))
        else:
            sample_weight = np.ones_like(y)
        sample_weight = sample_weight / sample_weight.mean()

        best_weights = weights.copy()
        best_intercept = intercept
        best_val = float("inf")
        patience = 50
        stale = 0
        self.training_loss = []

        m_w = np.zeros_like(weights)
        v_w = np.zeros_like(weights)
        m_b = 0.0
        v_b = 0.0
        beta1 = 0.9
        beta2 = 0.999
        eps = 1e-8

        for epoch in range(1, self.epochs + 1):
            logits = x @ weights + intercept
            pred = _sigmoid(logits)
            error = (pred - y) * sample_weight
            grad_w = (x.T @ error) / n_samples + self.l2 * weights
            grad_b = float(error.mean())

            m_w = beta1 * m_w + (1 - beta1) * grad_w
            v_w = beta2 * v_w + (1 - beta2) * (grad_w**2)
            m_b = beta1 * m_b + (1 - beta1) * grad_b
            v_b = beta2 * v_b + (1 - beta2) * (grad_b**2)

            weights -= self.learning_rate * (m_w / (1 - beta1**epoch)) / (np.sqrt(v_w / (1 - beta2**epoch)) + eps)
            intercept -= self.learning_rate * (m_b / (1 - beta1**epoch)) / (np.sqrt(v_b / (1 - beta2**epoch)) + eps)

            train_loss = log_loss(y, pred) + 0.5 * self.l2 * float(np.sum(weights**2))
            self.training_loss.append(float(train_loss))

            if x_val is not None and y_val is not None:
                val_loss = log_loss(y_val, self.predict_proba_from_params(x_val, weights, intercept))
                if val_loss + 1e-5 < best_val:
                    best_val = val_loss
                    best_weights = weights.copy()
                    best_intercept = float(intercept)
                    stale = 0
                else:
                    stale += 1
                    if stale >= patience:
                        break

        self.weights = best_weights.astype(float).tolist()
        self.intercept = float(best_intercept)
        return self

    @staticmethod
    def predict_proba_from_params(x: np.ndarray, weights: np.ndarray, intercept: float) -> np.ndarray:
        return _sigmoid(np.asarray(x, dtype=float) @ weights + intercept)

    def predict_proba(self, x: np.ndarray) -> np.ndarray:
        if self.weights is None:
            raise ValueError("Model has not been fitted.")
        return self.predict_proba_from_params(x, np.asarray(self.weights, dtype=float), self.intercept)

    def to_dict(self) -> dict:
        return {
            "model_type": "NumpyLogisticRegression",
            "l2": self.l2,
            "learning_rate": self.learning_rate,
            "epochs": self.epochs,
            "random_seed": self.random_seed,
            "class_weight": self.class_weight,
            "weights": self.weights,
            "intercept": self.intercept,
            "training_loss": self.training_loss,
        }

    @classmethod
    def from_dict(cls, payload: dict) -> "NumpyLogisticRegression":
        model = cls(
            l2=float(payload["l2"]),
            learning_rate=float(payload["learning_rate"]),
            epochs=int(payload["epochs"]),
            random_seed=int(payload["random_seed"]),
            class_weight=str(payload["class_weight"]),
        )
        model.weights = list(payload["weights"])
        model.intercept = float(payload["intercept"])
        model.training_loss = list(payload.get("training_loss") or [])
        return model

    def save(self, path: Path) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "NumpyLogisticRegression":
        return cls.from_dict(json.loads(path.read_text(encoding="utf-8")))


def tune_and_train_logistic(
    x_train: np.ndarray,
    y_train: np.ndarray,
    x_val: np.ndarray,
    y_val: np.ndarray,
    random_seed: int = 42,
) -> tuple[NumpyLogisticRegression, dict, float]:
    start = time.perf_counter()
    candidates = [
        {"l2": 0.01, "learning_rate": 0.03},
    ]
    best_model: NumpyLogisticRegression | None = None
    best_loss = float("inf")
    best_params = {}
    for params in candidates:
        model = NumpyLogisticRegression(
            l2=params["l2"],
            learning_rate=params["learning_rate"],
            epochs=40,
            random_seed=random_seed,
            class_weight="balanced",
        )
        model.fit(x_train, y_train, x_val, y_val)
        val_loss = log_loss(y_val, model.predict_proba(x_val))
        if val_loss < best_loss:
            best_loss = val_loss
            best_model = model
            best_params = params
    if best_model is None:
        raise RuntimeError("No logistic model was trained.")
    elapsed = time.perf_counter() - start
    best_params = {
        **best_params,
        "epochs": best_model.epochs,
        "class_weight": best_model.class_weight,
        "validation_log_loss": best_loss,
    }
    return best_model, best_params, elapsed
