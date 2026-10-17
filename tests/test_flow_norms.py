import pandas as pd

from marsa.data.flow_norms import COLUMNS, berth_counts, flow_norms
from marsa.data.load import load_dataset
from marsa.twin.calls import load_calls, scenario_at


def test_percentiles_come_from_the_dataset_columns():
    out, df = flow_norms(), load_dataset()
    for key, col in COLUMNS.items():
        assert out["norms"][key]["p50"] == round(float(df[col].quantile(0.5)), 2)
        assert out["norms"][key]["hours"] == int(df[col].notna().sum())


def test_berth_count_matches_the_twin_rule():
    calls = load_calls()
    hours = pd.DatetimeIndex(pd.to_datetime(["2025-07-10T19:00:00Z", "2025-01-04T19:00:00Z", "2025-05-14T06:00:00Z"], utc=True))
    counts = berth_counts(hours, calls)
    for ts in hours:
        assert counts[ts] == len(scenario_at(ts, 23, 24).berthed)


def test_missing_hours_are_the_hours_absent_from_the_dataset():
    out, df = flow_norms(), load_dataset()
    full = pd.date_range(df["hour_key"].min(), df["hour_key"].max(), freq="h")
    assert len(out["missing_hours"]) == len(full) - len(df)
