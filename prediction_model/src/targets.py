"""Future target creation using exact timestamp matching."""

from __future__ import annotations

from dataclasses import dataclass

import pandas as pd

from prediction_model.src.schema import TARGET_COLUMN, TIMESTAMP_COLUMN


@dataclass
class HorizonTargetMetadata:
    horizon_hours: int
    total_possible_samples: int
    usable_samples: int
    removed_missing_future_timestamp: int
    positive_targets: int
    negative_targets: int
    positive_prevalence: float


def add_future_target(df: pd.DataFrame, horizon_hours: int) -> tuple[pd.DataFrame, HorizonTargetMetadata]:
    if TARGET_COLUMN not in df.columns:
        raise ValueError(f"{TARGET_COLUMN} must exist before building future targets.")
    target_col = f"target_{horizon_hours}h"
    target_ts_col = f"target_timestamp_{horizon_hours}h"

    target_lookup = df.set_index(TIMESTAMP_COLUMN)[TARGET_COLUMN]
    out = df.copy()
    out[target_ts_col] = out[TIMESTAMP_COLUMN] + pd.Timedelta(hours=horizon_hours)
    out[target_col] = out[target_ts_col].map(target_lookup)

    usable = out[target_col].notna()
    positive = int((out.loc[usable, target_col] == 1).sum())
    negative = int((out.loc[usable, target_col] == 0).sum())
    meta = HorizonTargetMetadata(
        horizon_hours=horizon_hours,
        total_possible_samples=int(len(out)),
        usable_samples=int(usable.sum()),
        removed_missing_future_timestamp=int((~usable).sum()),
        positive_targets=positive,
        negative_targets=negative,
        positive_prevalence=float(positive / usable.sum()) if usable.sum() else 0.0,
    )
    return out, meta


def add_all_future_targets(df: pd.DataFrame, horizons: tuple[int, ...]) -> tuple[pd.DataFrame, pd.DataFrame]:
    out = df.copy()
    metas = []
    for horizon in horizons:
        out, meta = add_future_target(out, horizon)
        metas.append(meta.__dict__)
    return out, pd.DataFrame(metas)
