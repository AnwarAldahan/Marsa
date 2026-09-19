"""Load the merged dataset and expose a port-state snapshot for one hour."""
from __future__ import annotations
import pandas as pd
from marsa.config import DATA_DIR

DEFAULT_DATASET = DATA_DIR / "processed" / "merged_port_dataset_2025_v2.csv"


def load_dataset(path=DEFAULT_DATASET) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["hour_key"] = pd.to_datetime(df["hour_key"], utc=True)
    return df.sort_values("hour_key").reset_index(drop=True)


def snapshot(df: pd.DataFrame, timestamp) -> dict:
    """Return the row for `timestamp` as a dict. Raises if the hour is not in the timeline."""
    ts = pd.to_datetime(timestamp, utc=True).floor("h")
    row = df.loc[df["hour_key"] == ts]
    if row.empty:
        raise KeyError(f"{ts} is not in the dataset timeline (72 source hours are missing).")
    rec = row.iloc[0].to_dict()
    rec["hour_key"] = ts.isoformat()
    return _clean(rec)


def _clean(rec: dict) -> dict:
    out = {}
    for k, v in rec.items():
        if isinstance(v, float) and pd.isna(v):
            out[k] = None
        elif hasattr(v, "item"):
            out[k] = v.item()
        else:
            out[k] = v
    return out
