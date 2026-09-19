"""Build the Events & Weather Agent base dataset from original AIS data."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import duckdb
import pyarrow.parquet as pq

RAW_PATH = Path("archive (8)/ais_2023_2025_clean.parquet")
OUTPUT_PATH = Path("data/processed/events_weather_base.parquet")
SCHEMA_PATH = Path("data/processed/events_weather_base_schema.json")
REPORT_PATH = Path("reports/events_weather_base_report.md")

FEATURES = [
    "hour_key",
    "wind_speed_10m",
    "wave_height",
    "weather_code",
    "hour",
    "hour_sin",
    "hour_cos",
    "day_of_week",
    "month",
]

VALUE_FEATURES = [feature for feature in FEATURES if feature != "hour_key"]

DESCRIPTIONS = {
    "hour_key": "Hourly timestamp key from the original AIS dataset.",
    "wind_speed_10m": "Wind speed at 10 meters from the dataset weather fields.",
    "wave_height": "Wave height from the dataset weather fields.",
    "weather_code": "Numeric weather code from the original dataset.",
    "hour": "Hour-of-day integer from the original dataset.",
    "hour_sin": "Sine encoding of hour-of-day from the original dataset.",
    "hour_cos": "Cosine encoding of hour-of-day from the original dataset.",
    "day_of_week": "Day-of-week integer from the original dataset.",
    "month": "Month integer from the original dataset.",
}

UNITS = {
    "hour_key": None,
    "wind_speed_10m": "unknown; likely m/s but not verified in project documentation",
    "wave_height": "unknown; likely meters but not verified in project documentation",
    "weather_code": "code; interpretation unknown",
    "hour": "hour index",
    "hour_sin": "unitless",
    "hour_cos": "unitless",
    "day_of_week": "day index",
    "month": "month index",
}

PROVENANCE = {
    "hour_key": "OBSERVED",
    "wind_speed_10m": "OBSERVED",
    "wave_height": "OBSERVED",
    "weather_code": "OBSERVED",
    "hour": "DERIVED",
    "hour_sin": "DERIVED",
    "hour_cos": "DERIVED",
    "day_of_week": "DERIVED",
    "month": "DERIVED",
}


def md_table(headers: list[str], rows: list[list[Any]]) -> str:
    lines = [
        "| " + " | ".join(headers) + " |",
        "| " + " | ".join(["---"] * len(headers)) + " |",
    ]
    for row in rows:
        lines.append("| " + " | ".join("" if value is None else str(value) for value in row) + " |")
    return "\n".join(lines)


def main() -> None:
    raw_before = RAW_PATH.stat()
    con = duckdb.connect()
    raw_rel = f"read_parquet('{RAW_PATH.as_posix()}')"

    raw_columns = set(pq.ParquetFile(RAW_PATH).schema_arrow.names)
    missing_columns = [feature for feature in FEATURES if feature not in raw_columns]
    if missing_columns:
        raise RuntimeError(f"Missing required columns in original dataset: {missing_columns}")

    consistency_exprs = []
    for feature in VALUE_FEATURES:
        consistency_exprs.extend(
            [
                f"count(distinct \"{feature}\") AS {feature}_distinct_non_null",
                f"sum(CASE WHEN \"{feature}\" IS NULL THEN 1 ELSE 0 END) AS {feature}_null_count",
                f"count(\"{feature}\") AS {feature}_non_null_count",
            ]
        )
    consistency_sql = f"""
        SELECT hour_key, count(*) AS source_rows, {", ".join(consistency_exprs)}
        FROM {raw_rel}
        GROUP BY hour_key
        ORDER BY hour_key
    """
    consistency = con.execute(consistency_sql).fetchall()
    consistency_cols = [desc[0] for desc in con.description]

    conflicts: list[dict[str, Any]] = []
    for row in consistency:
        record = dict(zip(consistency_cols, row, strict=True))
        for feature in VALUE_FEATURES:
            distinct_non_null = record[f"{feature}_distinct_non_null"]
            null_count = record[f"{feature}_null_count"]
            non_null_count = record[f"{feature}_non_null_count"]
            if distinct_non_null > 1 or (null_count > 0 and non_null_count > 0):
                conflicts.append(
                    {
                        "hour_key": str(record["hour_key"]),
                        "feature": feature,
                        "distinct_non_null": int(distinct_non_null),
                        "null_count": int(null_count),
                        "non_null_count": int(non_null_count),
                    }
                )

    if conflicts:
        conflict_path = Path("reports/events_weather_base_conflicts.json")
        conflict_path.parent.mkdir(parents=True, exist_ok=True)
        conflict_path.write_text(json.dumps(conflicts, indent=2), encoding="utf-8")
        raise RuntimeError(
            "Conflicting weather/time values found within one or more hours. "
            f"Details written to {conflict_path}."
        )

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    select_exprs = ["hour_key"] + [f'min("{feature}") AS "{feature}"' for feature in VALUE_FEATURES]
    con.execute(
        f"""
        COPY (
            SELECT {", ".join(select_exprs)}
            FROM {raw_rel}
            GROUP BY hour_key
            ORDER BY hour_key
        )
        TO '{OUTPUT_PATH.as_posix()}' (FORMAT PARQUET, COMPRESSION ZSTD)
        """
    )

    output_rel = f"read_parquet('{OUTPUT_PATH.as_posix()}')"
    output_schema = pq.ParquetFile(OUTPUT_PATH).schema_arrow

    missing_counts = dict(
        con.execute(
            " UNION ALL ".join(
                [
                    f"SELECT '{feature}' AS feature, "
                    f"sum(CASE WHEN \"{feature}\" IS NULL THEN 1 ELSE 0 END) AS missing_count "
                    f"FROM {output_rel}"
                    for feature in FEATURES
                ]
            )
        ).fetchall()
    )
    unique_counts = dict(
        con.execute(
            " UNION ALL ".join(
                [
                    f"SELECT '{feature}' AS feature, count(distinct \"{feature}\") AS unique_count "
                    f"FROM {output_rel}"
                    for feature in FEATURES
                ]
            )
        ).fetchall()
    )

    schema_entries = []
    for field in output_schema:
        name = field.name
        notes = ""
        if name == "weather_code":
            notes = (
                "Project documentation does not verify this coding system. "
                "Original numeric values are preserved without interpretation."
            )
        elif name in {"wind_speed_10m", "wave_height"}:
            notes = "Unit is not verified in local project documentation."
        schema_entries.append(
            {
                "feature_name": name,
                "description": DESCRIPTIONS[name],
                "dtype": str(field.type),
                "unit": UNITS[name],
                "source": str(RAW_PATH),
                "provenance": PROVENANCE[name],
                "missing_value_count": int(missing_counts[name] or 0),
                "unique_value_count": int(unique_counts[name] or 0),
                "notes": notes,
            }
        )
    SCHEMA_PATH.write_text(json.dumps(schema_entries, indent=2), encoding="utf-8")

    summary = con.execute(
        f"""
        WITH bounds AS (
            SELECT min(hour_key) AS min_ts, max(hour_key) AS max_ts FROM {output_rel}
        ),
        expected AS (
            SELECT hour_ts
            FROM bounds, generate_series(min_ts, max_ts, INTERVAL 1 HOUR) AS t(hour_ts)
        )
        SELECT
            (SELECT count(*) FROM {raw_rel}) AS original_rows,
            (SELECT count(distinct hour_key) FROM {raw_rel}) AS unique_hour_keys,
            (SELECT count(*) FROM {output_rel}) AS final_rows,
            (SELECT min(hour_key) FROM {output_rel}) AS earliest_hour_key,
            (SELECT max(hour_key) FROM {output_rel}) AS latest_hour_key,
            (SELECT count(*) FROM expected e LEFT JOIN {output_rel} o ON e.hour_ts = o.hour_key WHERE o.hour_key IS NULL) AS missing_hourly_timestamps,
            (SELECT count(*) - count(distinct hour_key) FROM {output_rel}) AS duplicate_hour_keys
        """
    ).fetchone()

    stats = con.execute(
        f"""
        SELECT
            min(wind_speed_10m), avg(wind_speed_10m), median(wind_speed_10m), max(wind_speed_10m),
            min(wave_height), avg(wave_height), median(wave_height), max(wave_height)
        FROM {output_rel}
        """
    ).fetchone()
    weather_freq = con.execute(
        f"""
        SELECT weather_code, count(*) AS hours
        FROM {output_rel}
        GROUP BY weather_code
        ORDER BY weather_code NULLS FIRST
        """
    ).fetchall()

    raw_after = RAW_PATH.stat()
    raw_unchanged = (
        raw_before.st_size == raw_after.st_size
        and raw_before.st_mtime_ns == raw_after.st_mtime_ns
    )

    REPORT_PATH.parent.mkdir(parents=True, exist_ok=True)
    report = f"""# Events & Weather Base Dataset Report

Source: `{RAW_PATH}`

Output: `{OUTPUT_PATH}`

Schema: `{SCHEMA_PATH}`

## Validation Summary

{md_table(
    ["Metric", "Value"],
    [
        ["Original AIS row count", summary[0]],
        ["Unique hour_key values", summary[1]],
        ["Final dataset row count", summary[2]],
        ["Earliest hour_key", summary[3]],
        ["Latest hour_key", summary[4]],
        ["Missing hourly timestamps in range", summary[5]],
        ["Duplicate hour_key rows in output", summary[6]],
        ["Conflicting values found within same hour", len(conflicts)],
        ["Original AIS unchanged", raw_unchanged],
    ],
)}

## Missing Values

{md_table(["Feature", "Missing Count"], [[feature, missing_counts[feature]] for feature in FEATURES])}

## Summary Statistics

{md_table(
    ["Feature", "Min", "Mean", "Median", "Max"],
    [
        ["wind_speed_10m", stats[0], stats[1], stats[2], stats[3]],
        ["wave_height", stats[4], stats[5], stats[6], stats[7]],
    ],
)}

## Weather Code Frequency

`weather_code` interpretation is currently unknown in local project documentation. Numeric values are preserved as observed.

{md_table(["weather_code", "hours"], [[code, count] for code, count in weather_freq])}

## Consistency Handling

For each `hour_key`, this build checked each requested weather/time feature for:

- more than one non-null value in the same hour
- mixed null and non-null values in the same hour

No conflicts were found, so each hourly value was safely collapsed to one row per `hour_key`.

Missing weather values were preserved as missing. No interpolation, imputation, synthetic weather, holidays, Ramadan/Eid, or external-event features were added.
"""
    REPORT_PATH.write_text(report, encoding="utf-8")

    print(report)


if __name__ == "__main__":
    main()
