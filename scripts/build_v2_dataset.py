"""v1 (pure merge of the 3 agent files) -> v2 = v1 + berth-occupancy columns derived from raw AIS.

Input : data/processed/merged_port_dataset_2025_v1.csv
        data/external/pola_berth_occupancy_2025_v2.csv   (from scripts/berth_occupancy_from_ais.py)
Output: data/processed/merged_port_dataset_2025_v2.csv
No existing column or row is modified; new columns are appended at the end.
"""
import pandas as pd
from pathlib import Path
R = Path(__file__).resolve().parents[1]
v1 = pd.read_csv(R / "data/processed/merged_port_dataset_2025_v1.csv")
berth = pd.read_csv(R / "data/external/pola_berth_occupancy_2025_v2.csv")
k1 = pd.to_datetime(v1["hour_key"], utc=True); k2 = pd.to_datetime(berth["hour_key"], utc=True)
berth = berth.assign(_k=k2).drop(columns="hour_key")
v2 = v1.assign(_k=k1).merge(berth, on="_k", how="left", validate="1:1").drop(columns="_k")
assert len(v2) == len(v1) and v2.iloc[:, :len(v1.columns)].equals(v1)
assert v2["pola_berthed_cargo_vessels"].notna().all()
v2.to_csv(R / "data/processed/merged_port_dataset_2025_v2.csv", index=False)
print("v2 written:", v2.shape, "| new columns:", [c for c in v2.columns if c not in v1.columns])
