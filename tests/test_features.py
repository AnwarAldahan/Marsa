import sys
from pathlib import Path
import pandas as pd
sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from marsa.data.features import add_target, add_lags

def test_target_uses_wall_clock_not_rows():
    hk = pd.to_datetime(["2025-01-01 00:00", "2025-01-01 01:00", "2025-01-01 05:00"], utc=True)  # gap 02-04
    df = pd.DataFrame({"hour_key": hk, "y": [1, 2, 3]})
    out = add_target(df, "y", 1)
    assert out["target_1h"].tolist()[0] == 2 and pd.isna(out["target_1h"].tolist()[1])  # 02:00 missing -> NaN
    lag = add_lags(df, ["y"], lags=(4,))
    assert lag["y_lag4h"].tolist()[2] == 2 and pd.isna(lag["y_lag4h"].tolist()[1])
