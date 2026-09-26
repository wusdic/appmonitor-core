"""Session engine — reconstructs interaction sessions from flow/request cadence.

Without host agents we infer sessions from the gap structure of an entity's
activity: bursts of requests separated by idle gaps > threshold are distinct
sessions. Emits session count, mean requests-per-session and mean think-time —
strong human-vs-automation and workload-shape features.
"""
from __future__ import annotations

from typing import List

from ...core.engine import Context, Engine
from ...models.schema import DerivedMetric, MetricKind


class SessionEngine(Engine):
    name = "derived.session"
    layer = "derived"
    consumes = ["http.requests", "l4.flows"]
    produces = [
        "derived.session_count", "derived.req_per_session",
        "derived.think_time_s_avg", "derived.activity_duty_cycle",
    ]
    description = "Session reconstruction from activity-gap structure: count, depth, think-time."

    def __init__(self, idle_gap_ticks: int = 3, window_points: int = 60, **p):
        super().__init__(**p)
        self.idle_gap = idle_gap_ticks
        self.window_points = window_points

    def run(self, ctx: Context, observations=None) -> int:
        n = 0
        for system in ctx.store.systems():
            for entity in ctx.store.entities(system):
                series = ctx.store.raw_series(system, entity, "http.requests")
                if len(series) < 4:
                    series = ctx.store.raw_series(system, entity, "l4.flows")
                if len(series) < 4:
                    continue
                pts = [(m.ts, float(m.value)) for m in series[-self.window_points:]
                       if isinstance(m.value, (int, float))]
                active = [(ts, v) for ts, v in pts if v > 0]
                total_ticks = len(pts) or 1
                duty = len(active) / total_ticks
                # segment active ticks into sessions by index gaps
                sessions: List[List[float]] = []
                idxs = [i for i, (_, v) in enumerate(pts) if v > 0]
                if idxs:
                    cur = [pts[idxs[0]][1]]
                    for prev, cur_i in zip(idxs, idxs[1:]):
                        if cur_i - prev > self.idle_gap:
                            sessions.append(cur)
                            cur = []
                        cur.append(pts[cur_i][1])
                    sessions.append(cur)
                sess_count = len(sessions)
                req_per = (sum(sum(s) for s in sessions) / sess_count) if sess_count else 0.0
                # think time: mean idle gap length in ticks between active runs
                gaps = [cur_i - prev - 1 for prev, cur_i in zip(idxs, idxs[1:]) if cur_i - prev > 1]
                think = (sum(gaps) / len(gaps) * ctx.window_s) if gaps else 0.0
                for name, value, kind in [
                    ("derived.session_count", float(sess_count), MetricKind.COUNTER),
                    ("derived.req_per_session", req_per, MetricKind.GAUGE),
                    ("derived.think_time_s_avg", think, MetricKind.GAUGE),
                    ("derived.activity_duty_cycle", duty, MetricKind.RATE),
                ]:
                    ctx.store.add_derived(DerivedMetric(
                        name=name, value=value, ts=ctx.now, system=system, entity=entity,
                        window_s=ctx.window_s, kind=kind, inputs=["http.requests"]))
                    n += 1
        return n
