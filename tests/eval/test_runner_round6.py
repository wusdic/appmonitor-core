"""Evaluator round 6: the stale checker under the bounded contract (gate 15),
a re-formed class key's grain series, and the recorder's per-engine thread
CPU (PG4 on one core for resumable runs)."""
from __future__ import annotations

import numpy as np

from app.core.store import MetricStore
from app.eval import pscale as PS
from app.eval.runner import _Recorder, _StaleChecker
from app.models.schema import ORG, DerivedMetric, MetricKind, RawMetric

S, E = "sys", "10.0.0.1"
DT = 900.0
BOUNDED = {"lib3": {"resource_mode": "bounded"}}


def _raw(st, ts, e=E):
    st.add_raw(RawMetric(name="l4.flows", value=1.0, ts=ts, system=S, entity=e))


def test_an_earned_only_series_of_a_demoted_ip_is_not_stale_in_bounded_mode():
    """Bounded pack A seed 0 (blib, round 6): behavior.wh / rhythm / cp.* of an
    IP that P15 demoted end by design (lib/pactive.series_owed); they were
    flagged stale (19 of 24 stale series). Full mode still flags them."""
    t0 = 3000 * 3600.0
    found = {}
    for cfg in (BOUNDED, {}):
        st = MetricStore()
        chk = _StaleChecker(cfg)
        for k in range(20):
            ts = t0 + k * DT
            _raw(st, ts)
            st.add_vec(S, E, "feature.active", ts, [1.0])
            st.add_vec(S, E, "behavior.other", ts, [1.0])
            if k < 8:
                st.add_vec(S, E, "behavior.wh", ts, [1.0])
            earned = [E] if k < 8 else []                # P15 demotes E at k = 8
            st.put_model(ORG, ORG, "model.budget", {"systems": {S: {"earned": earned, "ts": ts}}}, ts=ts)
            chk.check(st, ts, DT)
        found[bool(cfg)] = sorted(n for (_, _, n) in chk.found)
    assert found[True] == []
    assert found[False] == ["behavior.wh"]


def test_a_reformed_class_owes_its_grain_series_two_periods_after_it_returns():
    """A class key has no feature.active, so _due_since read every gap of an
    hourly class series as overdue the moment the class was re-formed by B02
    (bounded pack A seed 0: class:r6 flagged on re-birth)."""
    from app.engines.behavior.lib import m_class
    C = "class:r6"
    st = MetricStore()
    chk = _StaleChecker()
    orig = m_class.all_class_keys
    m_class.all_class_keys = lambda store, s: [C]
    try:
        t0 = 4000 * 3600.0
        alive = [k for k in range(40) if not (4 < k < 24)]           # dissolved for ~5 h
        for k in range(40):
            ts = t0 + k * DT
            _raw(st, ts, "10.0.0.9")
            if k in alive:
                st.add_derived(DerivedMetric(name="behavior.class", value={"n": 3}, ts=ts, system=S,
                                             entity=C, window_s=0, kind=MetricKind.CATEGORICAL))
                if k % 4 == 0 and (k <= 4 or k >= 32):             # hourly H series, owed from k = 32
                    st.add_vec(S, C, "behavior.h_series", ts, [1.0])
            chk.check(st, ts, DT)
        assert not chk.found, chk.found
        # a real stall of the H series after the class is back is still found
        for k in range(40, 56):
            ts = t0 + k * DT
            st.add_derived(DerivedMetric(name="behavior.class", value={"n": 3}, ts=ts, system=S,
                                         entity=C, window_s=0, kind=MetricKind.CATEGORICAL))
            chk.check(st, ts, DT)
        assert [n for (_, _, n) in chk.found] == ["behavior.h_series"]
    finally:
        m_class.all_class_keys = orig


class _Eng:
    def __init__(self, name):
        self.name, self.layer = name, "behavior"

    def safe_run(self, ctx, observations=None):
        x = 0.0
        for i in range(20000):
            x += i * 0.5
        return 1


def test_the_recorder_keeps_thread_cpu_per_engine_and_tick():
    engs = [_Eng("behavior.conformity"), _Eng("behavior.pattern_tree")]
    reg = type("R", (), {"ordered": lambda self: engs})()
    rec = _Recorder(reg, "record")
    ctx = type("C", (), {"now": 0.0})()
    for t in range(3):
        rec.begin_tick()
        for e in engs:
            e.safe_run(ctx)
        rec.end_tick(float(t), 900.0, 1, False, 0.0, 0.0, 0.0)
    tm = rec.timings()
    cpu = np.asarray(tm["engine_cpu_ms"])
    assert cpu.shape == (3, 2) and (cpu > 0).all()
    cols = PS.engine_cpu_columns(tm, (PS.SCORING_ENGINE, PS.LEARNING_ENGINE))
    assert set(cols) == {PS.SCORING_ENGINE, PS.LEARNING_ENGINE} and len(cols[PS.SCORING_ENGINE]) == 3
    assert PS.engine_cpu_columns({"engine_names": ["a"], "engine_ms": [[1.0]]}, ("a",)) is None


def test_pg4_cost_per_event_reads_thread_cpu_when_the_run_carries_it():
    """PG4's CPU/event slopes compared points run under different load; the
    per-engine thread CPU of the recorder is the cost on one core."""
    names = [PS.SCORING_ENGINE, PS.LEARNING_ENGINE]
    tm = {"engine_names": names, "engine_ms": [[10.0, 20.0], [10.0, 20.0]],
          "engine_cpu_ms": [[5.0, 10.0], [5.0, 10.0]]}
    c = PS.pcore_cpu(tm, [10.0, 10.0], names)
    assert c["us_per_event"] == 3000.0 and c["us_per_event_cpu"] == 1500.0
    mk = lambda n, us, cpu: {"pack": f"O-scale-500-{n}", "n_ips": 500, "n_attrs": 40 + n, "n_systems": 6,
                              "mem_bytes": 1e6, "mem_by_key": {"a": 1e6},
                              "cpu": {"us_per_event": us, "us_per_event_cpu": cpu}}
    g = PS.pg4_summary([mk(0, 1000.0, 500.0), mk(300, 4000.0, 1000.0)])
    pts = {p["n_attrs"]: p for p in g["details"]["points"]}
    assert pts[40]["us_per_event"] == 500.0 and pts[40]["us_per_event_basis"] == "thread CPU"
    ch = {x["name"]: x["value"] for x in g["details"]["checks"]}
    assert abs(ch["CPU/event slope vs attributes"] - np.log(2) / np.log(340 / 40)) < 1e-9
