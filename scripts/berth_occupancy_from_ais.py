"""Derive hourly LA container-berth occupancy from the raw Kaggle AIS parquet files (2025H1/H2).

Method (documented in docs/DATA_DICTIONARY.md):
  1. cargo vessels only (AIS vessel_type 70-79) and tankers separately (80-89)
  2. stationary: sog < 0.5 kn (same definition as the dataset author)
  3. inside the LA port area: lon < LON_SPLIT and lat > LAT_MIN  (ASSUMPTION, see config)
  4. continuous stays longer than MAX_STAY_DAYS are excluded (laid-up vessels)
  5. count distinct vessels per hour_key; hours with none -> 0
Usage: python scripts/berth_occupancy_from_ais.py data/raw/ais_2025H1.parquet data/raw/ais_2025H2.parquet
"""
import sys
import pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from marsa.config import load_config
cfg = load_config()
LON_SPLIT, LAT_MIN = cfg["port"]["ais_lon_split"], cfg["port"]["ais_lat_min_inner"]
MAX_STAY_DAYS, GAP_HOURS = 7, 6

cols = ["hour_key", "mmsi", "vessel_type", "sog", "latitude", "longitude"]
df = pd.concat([pd.read_parquet(p, columns=cols) for p in sys.argv[1:]], ignore_index=True)
df["hour_key"] = pd.to_datetime(df["hour_key"], utc=True)
df = df[df["hour_key"].dt.year == 2025]
all_hours = pd.DatetimeIndex(sorted(df["hour_key"].unique()))
inner = df[(df["sog"] < 0.5) & (df["longitude"] < LON_SPLIT) & (df["latitude"] > LAT_MIN)]

def hourly(sub, name):
    s = sub[["hour_key", "mmsi"]].drop_duplicates().sort_values(["mmsi", "hour_key"])
    new_stay = s.groupby("mmsi")["hour_key"].diff() > pd.Timedelta(hours=GAP_HOURS)
    s = s.assign(stay=new_stay.groupby(s["mmsi"]).cumsum())
    length = s.groupby(["mmsi", "stay"])["hour_key"].transform(lambda x: (x.max() - x.min()).total_seconds() / 86400)
    s = s[length <= MAX_STAY_DAYS]
    return s.groupby("hour_key")["mmsi"].nunique().reindex(all_hours, fill_value=0).rename(name)

out = pd.concat([hourly(inner[inner.vessel_type.between(70, 79)], "pola_berthed_cargo_vessels"),
                 hourly(inner[inner.vessel_type.between(80, 89)], "pola_berthed_tanker_vessels")], axis=1)
out.rename_axis("hour_key").reset_index().to_csv(R / "data/external/pola_berth_occupancy_2025_v2.csv", index=False)
print(out.describe().round(2))
