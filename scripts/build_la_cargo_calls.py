"""Build the Port of Los Angeles cargo-vessel call table used by the Digital Twin.

Source : raw Kaggle AIS files (HarborMind LA/LB 2023-2025), placed in data/raw/
         ais_2023H2.parquet, ais_2024H1.parquet, ais_2024H2.parquet, ais_2025H1.parquet, ais_2025H2.parquet
Output : data/processed/la_cargo_calls_2023_2025.csv  (one row per stay at an LA berth)

Every value in the output comes from AIS pings. The only rules applied are:
  - cargo vessels = AIS vessel_type 70-79 (NOAA/ITU code, same as the dataset author)
  - waiting       = the dataset's own is_in_waiting_area flag (same as the maritime agent file)
  - at LA berth   = sog < 0.5 kn inside the LA port box from config/port_config.yaml
                    (same rule as scripts/berth_occupancy_from_ais.py; box validated against
                    Marine Exchange daily counts)
  - a stay continues across short signal gaps of up to GAP_HOURS (same value as the berth script);
    hours missing from the whole AIS file (data outages, e.g. 1 and 13 July 2025) are skipped
    when measuring a gap, so an outage does not end every stay at once
  - stays of MIN_STAY_HOURS or less are vessels passing through the LA box, not berth calls:
    the stay-length histogram has 1,264 stays of 1-2 h, then drops to 30 stays of 2-3 h
    (a natural gap in the data; no container ship is worked in under 2 h)
  - a wait belongs to an LA call when the vessel reaches an LA berth within GAP_HOURS after
    leaving the waiting area; waits that end anywhere else (Long Beach, departure) are not LA calls

Usage: python scripts/build_la_cargo_calls.py data/raw/ais_2023H2.parquet ... data/raw/ais_2025H2.parquet
"""
import sys
from pathlib import Path

import duckdb
import pandas as pd

R = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(R / "src"))
from marsa.config import load_config  # noqa: E402

cfg = load_config()
LON_SPLIT, LAT_MIN = cfg["port"]["ais_lon_split"], cfg["port"]["ais_lat_min_inner"]
GAP_HOURS = 6
MIN_STAY_HOURS = 2
OUT = R / "data" / "processed" / "la_cargo_calls_2023_2025.csv"


# One row per vessel per hour, with two flags: in the waiting area / stationary at an LA berth.
def vessel_hours(files: list[str]) -> pd.DataFrame:
    con = duckdb.connect()
    return con.execute(f"""
        select hour_key, mmsi,
               bool_or(coalesce(is_in_waiting_area, false))                       as is_waiting,
               bool_or(sog < 0.5 and longitude < {LON_SPLIT} and latitude > {LAT_MIN}) as is_la_berth
        from read_parquet({files})
        where vessel_type between 70 and 79
        group by 1, 2
    """).df()


# Turn hourly flags into continuous stays; a gap longer than GAP_HOURS (counted in hours that
# exist in the AIS file) starts a new stay.
def stays(hours: pd.DataFrame, flag: str) -> pd.DataFrame:
    timeline = pd.Index(sorted(hours["hour_key"].unique()))
    s = hours.loc[hours[flag], ["mmsi", "hour_key"]].sort_values(["mmsi", "hour_key"])
    s = s.assign(pos=timeline.get_indexer(s["hour_key"]))
    new = s.groupby("mmsi")["pos"].diff() > GAP_HOURS
    s = s.assign(stay=new.groupby(s["mmsi"]).cumsum())
    out = s.groupby(["mmsi", "stay"])["hour_key"].agg(start="min", last="max").reset_index()
    out["end"] = out["last"] + pd.Timedelta(hours=1)
    return out.drop(columns=["stay", "last"])


# Attach to each LA berth stay the waiting-area stay that ended just before it (if any).
def link_waits(berths: pd.DataFrame, waits: pd.DataFrame) -> pd.DataFrame:
    b = berths.sort_values("start").rename(columns={"start": "berth_start", "end": "berth_end"})
    w = waits.sort_values("end").rename(columns={"start": "wait_start", "end": "wait_end"})
    m = pd.merge_asof(b, w, left_on="berth_start", right_on="wait_end", by="mmsi",
                      direction="backward", allow_exact_matches=True,
                      tolerance=pd.Timedelta(hours=GAP_HOURS))
    # a wait that started after the berth stay began cannot be this call's wait
    m.loc[m["wait_start"] > m["berth_start"], ["wait_start", "wait_end"]] = pd.NaT
    m["waited"] = m["wait_start"].notna()
    m["arrival"] = m["wait_start"].where(m["waited"], m["berth_start"])
    m["wait_hours"] = (m["berth_start"] - m["arrival"]).dt.total_seconds() / 3600
    m["berth_hours"] = (m["berth_end"] - m["berth_start"]).dt.total_seconds() / 3600
    return m


# Build the call table and write it next to the other processed datasets.
def main(files: list[str]) -> None:
    hours = vessel_hours(files)
    hours["hour_key"] = pd.to_datetime(hours["hour_key"], utc=True)
    berths = stays(hours, "is_la_berth")
    berths = berths[(berths["end"] - berths["start"]) > pd.Timedelta(hours=MIN_STAY_HOURS)]
    calls = link_waits(berths, stays(hours, "is_waiting"))
    calls = calls.sort_values(["arrival", "mmsi"]).reset_index(drop=True)
    calls.insert(0, "call_id", [f"LA{i:05d}" for i in range(len(calls))])
    cols = ["call_id", "mmsi", "arrival", "waited", "wait_start", "wait_end", "berth_start",
            "berth_end", "wait_hours", "berth_hours"]
    calls[cols].to_csv(OUT, index=False)
    print(f"{len(calls)} LA cargo calls written to {OUT.relative_to(R)}")
    print(calls.groupby(calls["arrival"].dt.year)[["waited"]].agg(["size", "sum"]))
    print(calls.loc[calls["waited"], "wait_hours"].describe().round(1))


if __name__ == "__main__":
    main(sys.argv[1:])
