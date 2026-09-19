import json
from pathlib import Path

import joblib
import numpy as np

from marsa.data.load import load_dataset
from marsa.model.predict import APPROVED_FEATURES, forecast


def test_preserved_artifact_contract_loads_and_predicts_probability():
    path = Path("artifacts/models/marsa_xgboost_c1.joblib")
    metadata = json.loads(path.with_name("marsa_xgboost_c1_metadata.json").read_text())
    artifact = joblib.load(path)
    assert tuple(metadata["features"]) == APPROVED_FEATURES
    values = np.zeros((1, len(APPROVED_FEATURES)))
    transformed = artifact["imputer"].transform(values)
    probability = artifact["model"].predict_proba(transformed)[0, 1]
    assert 0.0 <= probability <= 1.0


def test_runtime_explicitly_refuses_fake_feature_parity():
    result = forecast(load_dataset(), "2025-01-01T00:00:00Z")
    assert result["status"] == "ml_unavailable"
    assert result["probability"] is None
    assert result["is_real_model_prediction"] is False
    assert "median_sog" in result["missing_required_features"]
