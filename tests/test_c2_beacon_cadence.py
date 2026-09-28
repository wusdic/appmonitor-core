"""lib-4 counter clauses are cadence-invariant (evaluator round 3).

`c2_beacon`'s volume clause `http.requests <= 10` was a per-TICK count: a
30-s health checker makes 30 requests per 900-s tick (no match in the whole
warm-up) and 2 per 60-s tick (a HIGH match on every live tick), so the
Runtime's 900 -> 60 s switch alone turned every sanctioned health checker into
a "C2 beacon" (78 % of their post-switch risk, integration.md §8.2). The same
held for every counter clause in the library: at 60 s `maintenance`, `search`,
`health` and `browse` appeared on control entities that never matched them
at 900 s, and B08 scored the new lib-4 categories as first_seen / JSD drift
(eval pack E: jsd p < 0.01 on 99 % of the machine personas' live ticks).
RuleMatchEngine now reads additive counters per 15-min grain (canonical grain
mode; tick mode keeps the v2 reading, see test_tick_mode_keeps_the_per_tick_reading).
"""
from __future__ import annotations

import pytest

from helpers import T0, add_obs_tick, make_store, run_engine

from app.engines.signature.rule_match import RuleMatchEngine
from app.engines.signature.store import Signature, SignatureStore
from app.models.schema import DerivedMetric
from app.pipeline.build import load_signatures

S, E = "api-gateway", "10.40.9.9"
CANON = {"grain_mode": "canonical"}
T_START = T0 - (T0 % 3600.0)          # an hour boundary


def _requests(period_s: float, dt: float, n_ticks: int, t0: float = T_START):
    """(tick end, requests in (end - dt, end]) of a poller firing at k*period_s."""
    out = []
    for i in range(1, n_ticks + 1):
        end = t0 + i * dt
        lo = end - dt
        out.append((end, max(0, int(end // period_s) - int(lo // period_s))))
    return out


def _probe_engine() -> RuleMatchEngine:
    """A one-clause signature whose confidence is the soft membership of
    http.requests <= 20 (band 20), so the matched evidence carries the value."""
    sigs = SignatureStore()
    sigs.signatures.append(Signature(
        id="probe", label="probe", category="test", severity="info",
        all=[{"metric": "http.requests", "op": "ge", "value": 0.0}]))
    return RuleMatchEngine(sigs, min_confidence=0.0)


def _per_grain_values(period_s: float, dt: float, hours: float = 3.0):
    st, eng = make_store(), _probe_engine()
    vals = []
    for end, c in _requests(period_s, dt, int(hours * 3600 / dt)):
        if c:
            add_obs_tick(st, S, E, end, {"http.requests": float(c)})
        run_engine(eng, st, end, dt=dt, config=CANON)
        ms = [m for m in st.matches(S, E, limit=10 ** 6) if m.ts == end]
        vals.append(ms[0].evidence.get("http.requests ge 0.0") if ms else None)
    return vals


@pytest.mark.parametrize("period_s,expected", [(30.0, 30.0), (300.0, 3.0), (60.0, 15.0)])
def test_counter_is_read_per_15_min_at_60_900_3600(period_s, expected):
    for dt in (60.0, 900.0, 3600.0):
        vals = _per_grain_values(period_s, dt)
        tail = [v for v in vals[len(vals) // 2:] if v is not None]
        assert tail, dt
        assert all(v == pytest.approx(expected, abs=1e-3) for v in tail), (dt, tail[:6])


def test_straddling_tick_after_a_900_to_60_switch_is_prorated():
    st, eng = make_store(), _probe_engine()
    t = T_START
    for _ in range(4):                       # 900-s phase, 30 requests per tick
        t += 900.0
        add_obs_tick(st, S, E, t, {"http.requests": 30.0})
        run_engine(eng, st, t, dt=900.0, config=CANON)
    for _ in range(20):                      # 60-s phase, 2 requests per tick
        t += 60.0
        add_obs_tick(st, S, E, t, {"http.requests": 2.0})
        run_engine(eng, st, t, dt=60.0, config=CANON)
        m = [x for x in st.matches(S, E, limit=10 ** 6) if x.ts == t][0]
        assert m.evidence["http.requests ge 0.0"] == pytest.approx(30.0, abs=1e-3)


def _c2_engine() -> RuleMatchEngine:
    sig, _comp = load_signatures()
    only = SignatureStore()
    only.signatures = [s for s in sig.signatures if s.id == "c2_beacon"]
    assert only.signatures, "c2_beacon missing from data/signatures"
    return RuleMatchEngine(only)


def _poller_matches(period_s: float, dt: float, hours: float = 2.0) -> float:
    """Fraction of HTTP ticks (after the entity's first grain) on which
    c2_beacon matches a steady single-destination poller with small
    responses; the window clauses are held at beacon-like values, as D0
    writes them for such a poller."""
    st, rule = make_store(), _c2_engine()
    n_http = n_match = 0
    for end, c in _requests(period_s, dt, int(hours * 3600 / dt)):
        if not c:
            continue
        add_obs_tick(st, S, E, end, {"http.requests": float(c), "http.resp_bytes_avg": 400.0,
                                     "http.write_ratio": 0.0})
        for name, v in (("derived.timing_regularity", 0.95), ("derived.dest_concentration", 1.0),
                        ("derived.periodicity_score", 0.9)):
            st.add_derived(DerivedMetric(name=name, value=v, ts=end, system=S, entity=E,
                                         window_s=int(dt)))
        before = len(st.matches(S, E, limit=10 ** 6))
        run_engine(rule, st, end, dt=dt, config=CANON)
        # before one grain the value is a rate extrapolated from < 15 min
        # (1 request in the first minute of a 5-min beacon reads as 15)
        if end - T_START > 900.0 + dt:
            n_http += 1
            n_match += len(st.matches(S, E, limit=10 ** 6)) > before
    return n_match / max(1, n_http)


def test_c2_beacon_verdict_does_not_depend_on_the_tick_length():
    # a 30-s health checker: never a beacon by volume, at 60 s as at 900 s
    # (before the fix: every 60-s tick matched, no 900-s tick did)
    assert _poller_matches(30.0, 60.0) == 0.0
    assert _poller_matches(30.0, 900.0) == 0.0
    # a 5-min beacon (T1): low volume at every cadence
    assert _poller_matches(300.0, 60.0) == 1.0
    assert _poller_matches(300.0, 900.0) == 1.0
    assert _poller_matches(300.0, 3600.0) == 1.0


def test_tick_mode_keeps_the_per_tick_reading():
    """grain_mode 'tick' is the frozen v2 reference (tick-mode golden)."""
    st, eng = make_store(), _probe_engine()
    add_obs_tick(st, S, E, T_START + 60.0, {"http.requests": 2.0})
    run_engine(eng, st, T_START + 60.0, dt=60.0)
    m = st.matches(S, E, limit=10)[0]
    assert m.evidence["http.requests ge 0.0"] == 2.0
