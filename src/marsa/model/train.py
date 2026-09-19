"""
=====================================================================
 HERE: PREDICTION MODEL - TRAINING
=====================================================================
Owner : <name>
Status: TODO - replace this stub with the real implementation.

Contract (do not change the function signature or the JSON keys,
the pipeline and the digital twin depend on them):

main() -> None
  reads : data/processed/merged_port_dataset_2025_v2.csv via marsa.data.load / marsa.data.features
  rules : time split (train <= 2025-09-30, valid Oct, test Nov-Dec); target = waiting_vessel_count at t+h
          (wall-clock shift, see features.add_target); exclude cfg["model"]["excluded_features"];
          always report persistence baseline ("in h hours = now") next to the model.
  writes: models/congestion_<feature_set>_<h>h.* and models/training_report.json
=====================================================================
"""
from __future__ import annotations


def main() -> None:
    raise NotImplementedError("Prediction model training goes here")


if __name__ == "__main__":
    main()
