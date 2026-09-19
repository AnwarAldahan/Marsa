"""Load the merged dataset and expose an exact authoritative port-state hour."""
from __future__ import annotations

from datetime import timedelta

import pandas as pd

from marsa.config import DATA_DIR

DEFAULT_DATASET = DATA_DIR / "processed" / "merged_port_dataset_2025_v2.csv"


def load_dataset(path=DEFAULT_DATASET) -> pd.DataFrame:
    df = pd.read_csv(path)
    df["hour_key"] = pd.to_datetime(df["hour_key"], utc=True)
    return df.sort_values("hour_key").reset_index(drop=True)


def snapshot(df: pd.DataFrame, timestamp) -> dict:
    """Return one exact 2025 UTC hour without flooring or nearest matching."""
    ts = parse_exact_utc_hour(timestamp)
    row = df.loc[df["hour_key"] == ts]
    if row.empty:
        raise KeyError(f"{ts} is not in the dataset timeline (72 source hours are missing).")
    rec = row.iloc[0].to_dict()
    rec["hour_key"] = ts.isoformat()
    return _clean(rec)


def parse_exact_utc_hour(value) -> pd.Timestamp:
    """Validate the public timestamp contract used by the integrated runtime."""
    try:
        ts = pd.Timestamp(value)
    except (TypeError, ValueError) as error:
        raise ValueError("timestamp_utc must be a valid timestamp") from error
    if ts.tzinfo is None:
        raise ValueError("timestamp_utc must be timezone-aware and explicitly UTC")
    if ts.utcoffset() != timedelta(0):
        raise ValueError("timestamp_utc must use UTC (offset +00:00 or Z)")
    if ts.minute or ts.second or ts.microsecond or ts.nanosecond:
        raise ValueError("timestamp_utc must be aligned to an exact whole hour")
    if ts.year != 2025:
        raise ValueError("timestamp_utc must be within calendar year 2025")
    return ts.tz_convert("UTC")


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
