"""
=====================================================================
 HERE: PREDICTION MODEL - INFERENCE
=====================================================================
Owner : <name>
Status: TODO - replace this stub with the real implementation.

Contract (do not change the function signature or the JSON keys,
the pipeline and the digital twin depend on them):

forecast(df, timestamp, cfg=None, feature_set="ais_only", horizons=None) -> dict
  output: {"timestamp_utc": str, "target": "waiting_vessel_count", "current_value": float,
           "horizons": {"6h": {"predicted": float, "change_vs_now": float,
                                 "top_drivers": [{"feature", "value", "contribution"}]}, "12h": ..., "24h": ...}}
  note  : top_drivers = SHAP-style contributions; the strategy agent shows them as the "why".
=====================================================================
"""
from __future__ import annotations


def forecast(df, timestamp, cfg=None, feature_set="ais_only", horizons=None) -> dict:
    raise NotImplementedError("Prediction model inference goes here")
