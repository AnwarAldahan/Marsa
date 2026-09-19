"""HTTP adapter: call a team agent served as an API (FastAPI etc.) instead of the local function.

Example:
    from marsa.agents.remote import RemoteAgent
    ctx = RemoteAgent("http://127.0.0.1:8000/api/agents/events-weather/investigate",
                      payload=lambda snap: {"timestamp_utc": snap["hour_key"]})
    run(ts, agents={"context": ctx.assess})

The response must contain at least: status/level, findings[], possible_bottleneck (or null),
and for the context agent `twin_multipliers` {crane_multiplier, gate_multiplier, arrival_multiplier}.
If the API is unreachable the local deterministic agent is used as fallback.
"""
from __future__ import annotations
import json
import urllib.request


class RemoteAgent:
    def __init__(self, url: str, payload=None, timeout: float = 15.0, fallback=None, headers=None):
        self.url, self.payload, self.timeout, self.fallback = url, payload, timeout, fallback
        self.headers = {"Content-Type": "application/json", **(headers or {})}

    def assess(self, snap: dict, cfg: dict) -> dict:
        body = self.payload(snap) if self.payload else snap
        req = urllib.request.Request(self.url, data=json.dumps(body, default=str).encode(),
                                     headers=self.headers, method="POST")
        try:
            with urllib.request.urlopen(req, timeout=self.timeout) as r:
                out = json.loads(r.read().decode())
        except Exception as e:  # network / 4xx / 5xx -> fallback
            if self.fallback is None:
                raise
            out = self.fallback(snap, cfg)
            out["remote_error"] = str(e)
        out.setdefault("twin_multipliers", {"crane_multiplier": 1.0, "gate_multiplier": 1.0, "arrival_multiplier": 1.0})
        out.setdefault("possible_bottleneck", None)
        out.setdefault("findings", [])
        return out
