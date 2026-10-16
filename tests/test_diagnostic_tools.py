from __future__ import annotations

import pandas as pd
import pytest
from pydantic import ValidationError

from marsa.agents.tools.diagnostics import DiagnosticArguments, DiagnosticTools

TS = "2025-07-10T19:00:00Z"
ARGS = DiagnosticArguments(window_hours=6)


@pytest.fixture
def sources(tmp_path):
    hours = pd.date_range("2025-07-10T13:00:00Z", periods=8, freq="h")
    cargo = pd.DataFrame({
        "timestamp": hours, "containers_in_yard": [100 + i * 2 for i in range(8)],
        "container_arrivals": [100, 3, 3, 3, 3, 3, 3, 99999],
        "container_departures": [100, 1, 1, 1, 1, 1, 1, 99999],
        "yard_occupancy_percent": [10 + i * .2 for i in range(8)],
        "truck_waiting_time_minutes": list(range(8)),
        "truck_arrivals": [10] * 8, "truck_departures": [8] * 8,
        "gate_throughput": [10] * 8, "future_label": [999] * 8,
    })
    maritime = pd.DataFrame({"hour_key": hours,
                             "waiting_vessel_count": list(range(8)), "vessel_count": [20] * 8})
    cargo_path, maritime_path = tmp_path / "cargo.csv", tmp_path / "maritime.csv"

    def write():
        cargo.to_csv(cargo_path, index=False)
        maritime.to_csv(maritime_path, index=False)
        return DiagnosticTools(cargo_path=cargo_path, maritime_path=maritime_path)

    return cargo, maritime, write


def test_exact_endpoint_changes_and_six_flow_bins(sources):
    _, _, write = sources
    tools = write()
    result = tools.execute("cargo_balance", TS, ARGS)
    assert result.status == "complete"
    assert result.metrics == {"inventory_change": 12, "net_cargo_flow": 12,
                              "yard_occupancy_change": pytest.approx(1.2)}
    assert result.direction == "rising"
    assert result.coverage.observed_hours == result.coverage.expected_hours == 7
    assert result.units["yard_occupancy_change"] == "percentage points"
    delay = tools.execute("cargo_delay_profile", TS, ARGS)
    assert delay.metrics == {"truck_waiting_time_change": 6, "truck_arrivals": 60,
                             "truck_departures": 48, "gate_throughput": 60}
    assert delay.direction == "rising"
    assert "future_label" not in result.source_fields
    assert result.source.geographic_scope == "MIXED_SCOPE_SYNTHETIC"
    assert "SYNTHETIC" in result.source.provenance
    assert result.source.identifier.endswith("cargo.csv")


def test_draining_and_improving_delays(sources):
    cargo, _, write = sources
    cargo["containers_in_yard"] = list(reversed(cargo["containers_in_yard"]))
    cargo["container_arrivals"], cargo["container_departures"] = (
        cargo["container_departures"].copy(), cargo["container_arrivals"].copy())
    cargo["truck_waiting_time_minutes"] = list(reversed(cargo["truck_waiting_time_minutes"]))
    tools = write()
    balance = tools.execute("cargo_balance", TS, ARGS)
    assert balance.metrics["inventory_change"] == -12
    assert balance.metrics["net_cargo_flow"] == -12
    assert balance.direction == "falling"
    assert tools.execute("cargo_delay_profile", TS, ARGS).direction == "falling"


@pytest.mark.parametrize("falling", [False, True])
def test_maritime_direction_and_regional_provenance(sources, falling):
    _, maritime, write = sources
    if falling:
        maritime["waiting_vessel_count"] = list(reversed(maritime["waiting_vessel_count"]))
    result = write().execute("maritime_history", TS, ARGS)
    assert result.status == "complete"
    assert result.metrics["waiting_vessel_count_change"] == (-6 if falling else 6)
    assert result.metrics["waiting_ratio_change"] == pytest.approx(-.3 if falling else .3)
    assert result.direction == ("falling" if falling else "rising")
    assert result.source.geographic_scope == "REGIONAL_LA_LB_UNVERIFIED_MEMBERSHIP"
    assert "unverified" in " ".join(result.source.limitations)


def test_missing_clock_hour_never_replaced(sources):
    cargo, _, write = sources
    cargo.drop(index=3, inplace=True)
    result = write().execute("cargo_balance", TS, ARGS)
    assert result.status == "incomplete"
    assert result.coverage.observed_hours == 6
    assert result.coverage.missing_timestamps == ["2025-07-10T16:00:00+00:00"]
    assert result.metrics["inventory_change"] == 12
    assert result.metrics["net_cargo_flow"] is None
    assert result.direction == "unknown"


def test_zero_distinct_from_null(sources):
    cargo, maritime, write = sources
    cargo.loc[:, "containers_in_yard"] = 0
    cargo.loc[:, "container_arrivals"] = 0
    cargo.loc[:, "container_departures"] = 0
    maritime.loc[:, "waiting_vessel_count"] = 0
    maritime.loc[:, "vessel_count"] = 0
    tools = write()
    balance = tools.execute("cargo_balance", TS, ARGS)
    assert balance.status == "complete"
    assert balance.direction == "stable"
    assert balance.metrics["inventory_change"] == balance.metrics["net_cargo_flow"] == 0
    assert tools.execute("maritime_history", TS, ARGS).metrics["waiting_ratio_end"] == 0
    cargo.loc[6, "containers_in_yard"] = None
    missing = write().execute("cargo_balance", TS, ARGS)
    assert missing.status == "incomplete"
    assert missing.metrics["inventory_change"] is None
    assert missing.current_evidence["containers_in_yard"] is None
    assert missing.metrics["net_cargo_flow"] == 0


@pytest.mark.parametrize("tool", ["cargo_balance", "cargo_delay_profile", "maritime_history"])
def test_future_values_labels_and_row_order_cannot_change_evidence(sources, tool):
    cargo, maritime, write = sources
    before = write().execute(tool, TS, ARGS).model_dump()
    for frame, key in [(cargo, "timestamp"), (maritime, "hour_key")]:
        for field in frame.columns:
            if field != key:
                frame[field] = frame[field].astype(object)
                frame.loc[7, field] = "invalid future observation"
    # The first physical row is now future data; timestamp parsing stays independent.
    cargo.sort_index(ascending=False, inplace=True)
    maritime.sort_index(ascending=False, inplace=True)
    assert write().execute(tool, TS, ARGS).model_dump() == before


@pytest.mark.parametrize("timestamp", ["2025-07-10T19:30:00Z", "2025-07-10T19:00:00",
                                        "2025-07-10T22:00:00+03:00", "2026-01-01T00:00:00Z"])
def test_only_exact_2025_utc_hours_accepted(sources, timestamp):
    with pytest.raises(ValueError):
        sources[2]().execute("cargo_balance", timestamp, ARGS)


@pytest.mark.parametrize("arguments", [{"window_hours": 5}, {"window_hours": 7},
    {"window_hours": "6"}, {"window_hours": 6.0}, {"window_hours": True},
    {"window_hours": 6, "path": "anything"}, {}])
def test_arguments_are_strict_and_bounded(arguments):
    with pytest.raises(ValidationError):
        DiagnosticArguments.model_validate(arguments)


@pytest.mark.parametrize("corruption", ["duplicate", "misaligned", "missing_column"])
def test_invalid_source_reports_error(sources, corruption):
    cargo, _, write = sources
    if corruption == "duplicate":
        cargo.loc[3, "timestamp"] = cargo.loc[2, "timestamp"]
    elif corruption == "misaligned":
        cargo.loc[3, "timestamp"] += pd.Timedelta(minutes=30)
    else:
        cargo.drop(columns="container_arrivals", inplace=True)
    assert write().execute("cargo_balance", TS, ARGS).status == "error"


@pytest.mark.parametrize("bad_count", [21, 1.5, -1, float("inf")])
def test_invalid_maritime_counts_are_missing(sources, bad_count):
    _, maritime, write = sources
    maritime["waiting_vessel_count"] = maritime["waiting_vessel_count"].astype(float)
    maritime.loc[3, "waiting_vessel_count"] = bad_count
    result = write().execute("maritime_history", TS, ARGS)
    assert result.status == "incomplete"
    assert result.direction == "unknown"


def test_year_start_reports_real_missing_history():
    result = DiagnosticTools().execute("cargo_balance", "2025-01-01T00:00:00Z", ARGS)
    assert result.status == "incomplete"
    assert result.coverage.observed_hours == 1
    assert len(result.coverage.missing_timestamps) == 6
    assert result.metrics["inventory_change"] is None


def test_missing_source_is_nonfatal(tmp_path):
    result = DiagnosticTools(cargo_path=tmp_path / "absent.csv").execute("cargo_balance", TS, ARGS)
    assert result.status == "error"
    assert not result.metrics
