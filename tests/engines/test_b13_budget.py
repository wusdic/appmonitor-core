"""B13 BudgetEngine (docs/lib3/engines.md '## B13'): spec unit tests (a), (b)
plus core edge cases. Engines only talk to the store, so `Feed` stands in for
B01 (feature.nat / feature.active) and R2 (act.stream, act.rare_events,
act.objs, act.tokens) and R1 (dns.qname_set), writing exactly the storage
forms of contract B / lib/m_template.

Spec test mapping:
  (a) test_a_day_to_date_fires_by_14h_while_per_tick_z_small
  (b) test_b_object_enumeration_breadth_alarm
  (c): test_b13_budget_peer.py; (d), actors, young entities:
  test_b13_budget_exfil.py; training, cadence, rollback, links, perf:
  test_b13_budget_edges.py.
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from helpers import make_store, run_engine

from app.engines.behavior.budget import (ABS_FLOOR_DEFAULT, AXIS_DETECTOR, MODEL, SERIES,
                                         BudgetEngine, fit_tail, tail_p, window_sums, NHIST,
                                         Q_EVAL)
from app.engines.behavior.lib import emit
from app.engines.behavior.lib.detectors import DETECTOR_INDEX
from app.engines.behavior.lib import features as F
from app.engines.behavior.lib import m_template as MT
from app.models.schema import RawMetric

S = "erp"
E = "10.0.0.1"
T_MID = 1699977600.0          # 2023-11-15 00:00 Asia/Shanghai (a Wednesday)
H = 3600.0
DAY = 86400.0
OBJ = "erp.corp/orders/view/{num}"


# ------------------------------------------------------------------ fixtures
class Feed:
    """B01 + R1 + R2 stand-in for one system."""

    def __init__(self, store, system: str = S) -> None:
        self.store, self.s = store, system

    def tick(self, e: str, now: float, dt: float, up: float = 0.0, down: float = 0.0,
             req: float = 0.0, writes: float = 0.0, active: Optional[bool] = None,
             stream: Optional[Sequence[Tuple[float, float, int]]] = None,
             frac: float = 1.0, rare: Optional[Dict[int, List[List[float]]]] = None,
             objs: Optional[Dict[str, Dict[str, Any]]] = None,
             tokens: Optional[Dict[str, float]] = None,
             qnames: Optional[Dict[str, float]] = None, nat_override=None) -> None:
        st, s = self.store, self.s
        nat = np.zeros(F.FEATURE_DIM, dtype=np.float32)
        nat[F.FEATURE_INDEX["bytes_up"]] = up
        nat[F.FEATURE_INDEX["bytes_down"]] = down
        nat[F.FEATURE_INDEX["http_requests"]] = req
        nat[F.FEATURE_INDEX["http_write_ratio"]] = writes / req if req > 0 else np.nan
        if nat_override:
            for k, v in nat_override.items():
                nat[F.FEATURE_INDEX[k]] = v
        busy = bool(up or down or req or stream or rare or objs or tokens or qnames)
        act = busy if active is None else active
        st.add_vec(s, e, "feature.nat", now, nat, window_s=int(dt))
        st.add_vec(s, e, "feature.active", now, np.array([1.0 if act else 0.0], np.float32),
                   window_s=int(dt))
        st.register_entity(s, e)
        raw: Dict[str, Any] = {}
        if stream:
            arr = np.zeros(len(stream), dtype=MT.STREAM_DTYPE)
            for i, (t, u, did) in enumerate(sorted(stream)):
                arr[i]["ts"], arr[i]["up"], arr[i]["dest_id"] = t, u, did
            arr.flags.writeable = False
            raw["act.stream"], raw["act.stream_frac"] = arr, frac
        if rare:
            raw["act.rare_events"] = rare
        if objs:
            raw["act.objs"] = objs
        if tokens:
            raw["act.tokens"] = tokens
        if qnames:
            raw["dns.qname_set"] = qnames
        for name, v in raw.items():
            st.add_raw(RawMetric(name=name, value=v, ts=now, system=s, entity=e), touch=act)
        st.add_raw(RawMetric(name="act.events", value=float(len(stream or ())), ts=now,
                             system=s, entity=e), touch=False)


def budget_row(st, now, e=E) -> Optional[Dict[str, Any]]:
    for m in reversed(st.derived_series(S, e, SERIES)):
        if m.ts == now:
            return m.value
    return None


def acc(st, now, e=E) -> Dict[str, int]:
    return emit.read_dict(st, S, e, emit.ACC_ALARM, now)


def scores(st, now, e=E) -> Dict[str, float]:
    return emit.read_row(st, S, e, emit.SCORE, now)


def last_alarms(st, e=E) -> List[str]:
    m = st.get_model(S, e, MODEL)
    return list((m["live"]["last"] or {}).get("alarms", []))


def events(st, e=None):
    return st.events(system=S, entity=e, kinds=["budget_exceeded"], limit=1000)


# ===================================================================== (a)
def run_a(seed: int = 3):
    """20 training days of upload at 900 s (daily ~50 MB, per-tick LogNormal
    noise sigma 0.3 truncated at +-1.1 sigma so the per-tick statement holds
    literally), a normal live day, then a day with +30 % on every tick."""
    st, eng = make_store(), BudgetEngine()
    feed = Feed(st)
    rng = np.random.default_rng(seed)
    dt, m = 900.0, 50e6 / 96
    z_attack: List[float] = []
    first: Dict[str, float] = {}
    for d in range(22):
        for k in range(96):
            now = T_MID + d * DAY + (k + 1) * dt
            eps = float(np.clip(rng.normal(), -1.1, 1.1))
            f = 1.3 if d == 21 else 1.0
            up = m * f * math.exp(0.3 * eps)
            feed.tick(E, now, dt, up=up, down=3 * m * math.exp(0.3 * eps), req=40.0,
                      writes=4.0, active=True)
            run_engine(eng, st, now, training=d < 20, dt=dt)
            if d == 20:
                assert not acc(st, now).get("budget_vol"), f"null day alarm at tick {k}"
            if d == 21:
                z_attack.append((math.log(up) - math.log(m)) / 0.3)
                for key in last_alarms(st):
                    first.setdefault(key, (k + 1) * dt / H)       # local hour of day
    return st, eng, z_attack, first


def test_a_day_to_date_fires_by_14h_while_per_tick_z_small():
    st, eng, z, first = run_a()
    assert max(abs(v) for v in z) < 2.0                 # every per-tick |z| < 2
    assert "bytes_up.day" in first, first
    assert first["bytes_up.day"] <= 14.0, first
    ev = events(st, E)
    assert ev, "budget_exceeded expected"
    e0 = ev[-1]                                           # oldest first
    assert e0.extra["quantity"] == "bytes_up" and e0.extra["axis"] == "volume"
    assert e0.extra["fit_source"] == "own" and e0.extra["history_days"] >= 20
    assert "MB" in e0.description                        # natural units
    assert e0.axes == ["volume"] and "budget_vol" in e0.p_by_detector
    m = st.get_model(S, E, MODEL)
    assert m["fit"]["days"][0] >= 20
    # one event per quantity and local day, whatever the number of alarming ticks
    assert len({(x.extra["quantity"], x.dedupe_key) for x in ev}) == len(ev)


# ===================================================================== (b)
def objs_entry(ids: Sequence[int], with_hll: bool = False) -> Dict[str, Any]:
    ids = [str(i) for i in ids]
    e: Dict[str, Any] = {"n": len(ids), "ids": ids[:MT.OBJ_IDS_CAP]}
    if with_hll or len(ids) > MT.OBJ_IDS_CAP:
        from app.engines.behavior.lib.sketch import HyperLogLog
        h = HyperLogLog()
        h.add_many(ids)
        e["hll"] = h.to_bytes()
    return e


def test_b_object_enumeration_breadth_alarm():
    st, eng = make_store(), BudgetEngine()
    feed = Feed(st)
    rng = np.random.default_rng(5)
    dt = 3600.0
    alarm_breadth = alarm_vol = False
    for d in range(21):
        for k in range(24):
            now = T_MID + d * DAY + (k + 1) * dt
            hour = k                                           # the tick covers [k, k+1)
            objs = None
            if 9 <= hour < 18:
                if d == 20 and 9 <= hour < 17:                # 8 h of enumeration
                    n = 300 if hour == 12 else 243
                    base = 10_000_000 + hour * 1000
                    objs = {OBJ: objs_entry(range(base, base + n))}
                else:                                          # ~40 distinct ids a day
                    objs = {OBJ: objs_entry(rng.integers(0, 1_000_000, 5))}
            feed.tick(E, now, dt, up=2e5 if objs else 0.0, down=1e6 if objs else 0.0,
                      req=10.0 if objs else 0.0, objs=objs,
                      tokens={"GET erp.corp/orders/view/{num}|2xx": 5.0} if objs else None)
            run_engine(eng, st, now, training=d < 20, dt=dt)
            a = acc(st, now)
            alarm_breadth |= bool(a.get("budget_breadth"))
            alarm_vol |= bool(a.get("budget_vol"))
            if d == 20 and hour == 16:
                row = budget_row(st, now)
                assert row["objs.8h"][0] == pytest.approx(2001, rel=0.13)   # HLL: 4 sigma
    assert alarm_breadth and not alarm_vol
    ev = [x for x in events(st, E) if x.extra["axis"] == "breadth"]
    assert ev and ev[-1].extra["quantity"] == "objs"
    assert ev[-1].extra["value"] > 500 + ev[-1].extra["usual"]
    assert "objects" in ev[-1].description


# ============================================================== edge cases
def test_empty_store_and_unknown_entities():
    st, eng = make_store(), BudgetEngine()
    assert run_engine(eng, st, T_MID) == 0
    st.register_entity(S, E)                               # no feature.nat ever: not ours yet
    assert run_engine(eng, st, T_MID + 900) == 0
    assert st.get_model(S, E, MODEL) is None


def test_silent_entity_scores_zero_budget_without_alarm():
    st, eng = make_store(), BudgetEngine()
    feed = Feed(st)
    dt = 3600.0
    for i in range(8 * 24):                                # 8 d of silence
        now = T_MID + (i + 1) * dt
        feed.tick(E, now, dt, active=False)
        run_engine(eng, st, now, training=i < 7 * 24, dt=dt)
    row = budget_row(st, now)
    assert row is not None and row["bytes_up.day"][0] == 0.0
    assert row["slots.7d"][0] == 0.0
    sc = scores(st, now)
    assert math.isfinite(sc["budget_vol"])                 # own immature fit after 7 d
    assert acc(st, now) == {"budget_vol": 0, "budget_exfil": 0, "budget_breadth": 0}
    assert not events(st)
    assert len(st.derived_series(S, E, SERIES)) <= 24        # newest points only


def test_nan_input_degrades_only_that_axis_and_is_not_learned():
    st, eng = make_store(), BudgetEngine()
    feed = Feed(st)
    dt = 3600.0
    now = T_MID + dt
    feed.tick(E, now, dt, up=1e6, nat_override={"bytes_up": np.nan})
    run_engine(eng, st, now, dt=dt)
    deg = emit.read_dict(st, S, E, emit.DEGRADED, now)
    assert deg.get("budget_vol", "").startswith("nan_input:bytes_up")
    assert "budget_exfil" not in deg and "budget_breadth" not in deg
    assert math.isnan(emit.read_array(st, S, E, emit.SCORE, now)[DETECTOR_INDEX["budget_vol"]])
    eng2 = BudgetEngine()
    assert eng2._fetch(st, S, E, now) is None              # no rows bound: nothing to learn
    m = st.get_model(S, E, MODEL)
    eng._rows_cur = m["rows"]
    try:
        assert eng._fetch(st, S, E, now) is None            # partial row never learned
    finally:
        eng._rows_cur = None


def test_b01_missing_or_failed_degrades_all_three():
    st, eng = make_store(), BudgetEngine()
    feed = Feed(st)
    dt = 900.0
    feed.tick(E, T_MID + dt, dt, up=1e5)
    run_engine(eng, st, T_MID + dt, dt=dt)
    now = T_MID + 2 * dt                                   # B01 did not write this tick
    run_engine(eng, st, now, dt=dt)
    deg = emit.read_dict(st, S, E, emit.DEGRADED, now)
    assert set(deg) == set(AXIS_DETECTOR.values())
    assert all(v == "stale:feature.nat" for v in deg.values())
    sc = emit.read_row(st, S, E, emit.SCORE, now)
    assert not any(k.startswith("budget") for k in sc)     # NaN = unscored, never p = 1
    # R2 failed: exfil and breadth degraded, volume still scored
    now = T_MID + 3 * dt
    feed.tick(E, now, dt, up=1e5)
    st.put_health("raw.action_token", {"last_error_ts": now})
    run_engine(eng, st, now, dt=dt)
    deg = emit.read_dict(st, S, E, emit.DEGRADED, now)
    assert set(deg) == {"budget_exfil", "budget_breadth"}
    assert deg["budget_exfil"] == "producer_error:raw.action_token"


def test_rerun_same_tick_is_idempotent():
    st, eng = make_store(), BudgetEngine()
    feed = Feed(st)
    dt = 900.0
    now = T_MID + dt
    feed.tick(E, now, dt, up=1e6, objs={OBJ: objs_entry(range(10))})
    run_engine(eng, st, now, dt=dt)
    r1 = dict(budget_row(st, now))
    run_engine(eng, st, now, dt=dt)
    r2 = budget_row(st, now)
    assert r1["bytes_up.1h"] == r2["bytes_up.1h"] and r1["objs.day"] == r2["objs.day"]
    assert r2["objs.day"][0] == 10.0 and r2["bytes_up.1h"][0] == pytest.approx(1e6)


# ======================================================== pure helpers
def test_fit_tail_gaussian_threshold_is_sane():
    rng = np.random.default_rng(0)
    t = fit_tail(rng.normal(size=672), 96)
    u, xi, sig, rate, zq = t[:5]
    assert 1.7 < u < 2.4 and 0.0 <= xi <= 0.3 and sig >= 0.3
    assert 3.5 < zq < 9.0                                   # ~4.3 sigma for q ~ 1.2e-5
    p = tail_p(np.array([zq]), t[None, :5], t[None, 5:])
    assert p[0] == pytest.approx(Q_EVAL, rel=1e-6)
    p = tail_p(np.array([-10.0, 0.0, np.nan]), np.tile(t[:5], (3, 1)), np.tile(t[5:], (3, 1)))
    assert p[0] == 1.0 and 0.4 < p[1] < 0.6 and math.isnan(p[2])
    assert fit_tail(np.zeros(10), 96) is None


def test_window_sums_horizons():
    v = np.ones(NHIST)
    v[30] = np.nan
    W = window_sums(v)
    assert W[0, 2, 5] == 1.0 and W[1, 2, 5] == 8.0
    assert W[2, 2, 5] == 6.0 and W[2, 2, 23] == 24.0
    assert math.isnan(W[1, 0, 3])                           # before the history
    assert math.isnan(W[0, 1, 6]) and math.isnan(W[2, 1, 10])   # hour 30 invalid
    assert W[3, 9, 0] == 168.0 and math.isnan(W[3, 6, 22])


def test_config_floor_override_and_validation():
    st, eng = make_store(), BudgetEngine()
    feed = Feed(st)
    feed.tick(E, T_MID + 900, 900.0, up=1.0)
    with pytest.raises(ValueError):
        run_engine(eng, st, T_MID + 900, dt=900.0, config={"budget_abs_floor": {"objs": -1}})
    assert set(ABS_FLOOR_DEFAULT) == {"bytes_up", "bytes_down", "writes", "slots", "up_novel",
                                      "dns_label", "objs", "templates", "dests"}
