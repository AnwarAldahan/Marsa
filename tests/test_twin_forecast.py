"""Forecast mode: the twin uses only what is known at the chosen hour."""
import numpy as np
import pandas as pd
from fastapi.testclient import TestClient

from marsa.api.main import app
from marsa.config import load_config
from marsa.twin.calls import load_calls, scenario_at
from marsa.twin.forecast import remaining_median, to_forecast

TS = pd.Timestamp("2025-07-10T19:00:00Z")


def forecast_at(ts, calls):
    sc = scenario_at(ts, berths=23, horizon_hours=24, calls=calls)
    return to_forecast(sc, ts, 5, calls=calls)


def summary(sc):
    return ([(v.call_id, round(v.berth_h + v.service_h, 6)) for v in sc.berthed],
            [(v.call_id, round(v.arrival_h, 6), round(v.ready_h, 6), round(v.service_h, 6)) for v in sc.vessels])


def test_remaining_median_uses_only_longer_values():
    assert remaining_median(np.array([10.0, 20.0, 40.0]), 15.0) == (15.0, 2)   # median of 5 and 25
    assert remaining_median(np.array([10.0]), 30.0) == (0.0, 0)


def test_forecast_does_not_read_the_future():
    calls = load_calls().copy()
    base = summary(forecast_at(TS, calls))
    # change everything that is only known after TS: later berth ends, berth starts and stay lengths
    future_end = calls["berth_end"] > TS
    calls.loc[future_end, "berth_end"] = calls.loc[future_end, "berth_end"] + pd.Timedelta(hours=300)
    calls.loc[future_end, "berth_hours"] = calls.loc[future_end, "berth_hours"] + 300
    future_start = calls["berth_start"] > TS
    calls.loc[future_start, "berth_start"] = calls.loc[future_start, "berth_start"] + pd.Timedelta(hours=5)
    calls.loc[future_start, "wait_hours"] = calls.loc[future_start, "wait_hours"] + 5
    assert summary(forecast_at(TS, calls)) == base


def test_forecast_arrivals_are_ready_on_arrival_and_keep_known_eta():
    calls = load_calls()
    replay = scenario_at(TS, berths=23, horizon_hours=24, calls=calls)
    fc = forecast_at(TS, calls)
    eta = {v.call_id: v.arrival_h for v in replay.vessels}
    for v in fc.vessels:
        assert v.arrival_h == eta[v.call_id]
        if v.arrival_h > 0:
            assert v.ready_h == v.arrival_h
        assert v.estimate["method"] in {"own_history", "similar_length", "all_earlier_stays"}


def test_pipeline_and_api_accept_forecast_mode():
    with TestClient(app) as client:
        ok = client.post("/api/decision-support/analyze",
                         json={"timestamp_utc": "2025-07-10T19:00:00Z", "twin_mode": "forecast"})
        bad = client.post("/api/twin/simulate",
                          json={"timestamp_utc": "2025-07-10T19:00:00Z", "twin_mode": "future"})
        sim = client.post("/api/twin/simulate",
                          json={"timestamp_utc": "2025-07-10T19:00:00Z", "twin_mode": "forecast",
                                "actions": [{"type": "add_berths", "value": 1}]})
    assert ok.status_code == 200
    body = ok.json()
    assert body["digital_twin"]["mode"] == "forecast"
    assert any(s.get("kind") == "forecast" for s in body["digital_twin"]["sources"])
    assert bad.status_code == 422
    assert sim.status_code == 200 and sim.json()["scenario"]["mode"] == "forecast"


def test_replay_mode_is_unchanged_by_default():
    cfg = load_config()
    from marsa.twin.adapter import build_scenario, scenario_summary
    a, _ = build_scenario(TS, cfg)
    b, _ = build_scenario(TS, cfg, mode="replay")
    assert summary(a) == summary(b) and a.mode == "replay"
    assert "mode" not in scenario_summary(a)
