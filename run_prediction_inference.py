#!/usr/bin/env python3
"""Run Agent 1-compatible multi-horizon Layer 1 inference."""

from __future__ import annotations

from pathlib import Path
import argparse
import json

from prediction_model.src.inference import predict_future_congestion, save_prediction_json


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--data", type=Path, default=Path("data/merged_port_dataset_2025.csv"))
    parser.add_argument("--model-dir", type=Path, default=Path("prediction_model/saved_models"))
    parser.add_argument("--timestamp", type=str, default=None, help="Prediction timestamp in the dataset; defaults to latest.")
    parser.add_argument("--output", type=Path, default=Path("prediction_model/results/latest_prediction.json"))
    args = parser.parse_args()
    payload = predict_future_congestion(args.data, args.model_dir, args.timestamp)
    save_prediction_json(payload, args.output)
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
