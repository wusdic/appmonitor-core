"""Cadence-invariance property test (spec v2.1, docs/lib3/cadence.md §13).

Part A (fast, in the suite). One local day of traffic for 2 human and 1
machine persona is generated at dt = 60 s WITHOUT aggregation, keeping every
Observation with its real ts, and re-batched into 900-s and 3600-s ticks
(a tick ending at t gets the observations with ts in [t - dt, t), the
generator's own tick semantics). Raw + derived (+ D0 periodicity) + B01 run
in canonical grain mode on the three tick sequences, and the H grain rows
must agree at every common H boundary within the §13.1 tolerances; the Q
rows of the 60-s and 900-s runs at every common Q boundary. A seeded
property form then drives B01 alone with random additive raw counts at
60 / 300 / 900 / 3600 s and misaligned tick offsets and checks the H values
against the direct sums over the event list.

Part B (the whole pipeline, ~minutes) runs only with APPMON_SLOW=1.
"""
from __future__ import annotations

import datetime as _dt
import math
import os

import numpy as np
import pytest

from app.core.engine import Registry
from app.core.store import MetricStore
from app.engines.behavior.feature_vector import FeatureVectorEngine
from app.engines.behavior.lib import features as F
from app.engines.behavior.lib import grains as GR
from app.engines.derived.entropy import EntropyEngine
from app.engines.derived.graph import GraphEngine
from app.engines.derived.periodicity import PeriodicityEngine
from app.engines.derived.ratio import RatioEngine
from app.engines.derived.session import SessionEngine
from app.engines.raw.action_token import ActionTokenEngine
from app.engines.raw.active_probe import ActiveProbeEngine
from app.engines.raw.client_stack import ClientStackEngine
from app.engines.raw.dns import DNSEngine
from app.engines.raw.http import HTTPEngine
from app.engines.raw.l2l3 import L2L3Engine
from app.engines.raw.l4flow import L4FlowEngine
from app.engines.raw.tls import TLSEngine
from app.eval import packs as P
from app.models.schema import AcquisitionMethod, RawMetric
from app.pipeline.generator import TrafficGenerator
from app.pipeline.orchestrator import Pipeline

KEYS = ["erp-prod|10.20.1.11", "erp-prod|10.20.1.12", "erp-prod|10.20.9.9"]
CAL = {"holidays": [], "makeup_workdays": []}
CFG = {"grain_mode": "canonical", "tz": P.SHANGHAI, "calendar": CAL, "strict": True}
IDX = F.FEATURE_INDEX


def _registry() -> Registry:
    r = Registry()
    r.add(L2L3Engine(), L4FlowEngine(), HTTPEngine(), TLSEngine(), DNSEngine(),
          ActiveProbeEngine(), ActionTokenEngine(), ClientStackEngine(), PeriodicityEngine(),
          RatioEngine(), EntropyEngine(), GraphEngine(), SessionEngine(), FeatureVectorEngine())
    return r


def _day_observations():
    tl = P.Timeline(P.SHANGHAI, CAL, _dt.date(2025, 3, 10), P._phases((1440, 60.0)))
    pk = P._pack("inv", tl, KEYS, [], "cadence invariance", backgrounds=False)
    gen = TrafficGenerator(seed=0, pack=pk)
    t0 = gen.vt
    flat = []
    for _ in range(1440):
        flat.extend(gen.step(60.0, live=False, aggregated=False))
    flat.sort(key=lambda o: o.ts)
    return t0, flat


def _run(t0, flat, dt):
    st = MetricStore()
    pl = Pipeline(st, _registry(), window_s=int(dt), config=CFG)
    j = 0
    for k in range(1, int(86400 / dt) + 1):
        end = t0 + k * dt
        batch = []
        while j < len(flat) and flat[j].ts < end:
            batch.append(flat[j])
            j += 1
        pl.run_tick(batch, now=end, training=True, dt=dt)
    return st


@pytest.fixture(scope="module")
def runs():
    t0, flat = _day_observations()
    return {dt: _run(t0, flat, dt) for dt in (60.0, 900.0, 3600.0)}


def _rows(st, key, name):
    s, e = key.split("|")
    ts, M = st.vec_since(s, e, name, 0.0)
    return {float(t): np.asarray(r, dtype=np.float64) for t, r in zip(ts, M)}


def _close(name, a, b):
    """cadence.md §13.1 tolerances for one feature (nat units)."""
    if math.isnan(a) or math.isnan(b):
        return math.isnan(a) and math.isnan(b)
    gc = F.GRAIN_CLASS[name]
    if name == "distinct_templates":
        # R2's Drain-lite templater state depends on the batch interleaving,
        # so a few tokens are masked differently early on (not a grain issue)
        return abs(a - b) <= max(1.0, 0.15 * max(abs(a), abs(b)))
    if name == "think_time":
        return abs(a - b) <= 0.1 * max(abs(a), abs(b))
    if name == "req_per_session":
        return abs(a - b) <= 0.1 * max(abs(a), abs(b), 1.0)
    if gc == "map":
        return abs(a - b) <= 0.02
    if gc == "set":
        return a == b
    return abs(a - b) <= 1e-5 * max(1.0, abs(a), abs(b))


def test_h_rows_equal_at_60_900_3600(runs):
    ref = runs[3600.0]
    n_checked = 0
    for key in KEYS:
        R = _rows(ref, key, "feature.nat.h")
        Rm = _rows(ref, key, "feature.meta.h")
        assert len(R) >= 10, key
        for dt in (60.0, 900.0):
            X = _rows(runs[dt], key, "feature.nat.h")
            Xm = _rows(runs[dt], key, "feature.meta.h")
            common = sorted(set(R) & set(X))
            assert len(common) == len(R), (key, dt)        # the same H boundaries
            s_, e_ = key.split("|")
            for t in common:
                assert Xm[t][1] == pytest.approx(3600.0) and Rm[t][1] == pytest.approx(3600.0)
                assert Xm[t][0] == Rm[t][0]                 # active
                # think_time is compared only when R2 kept every event (§13.1):
                # a sampled tick keeps only gaps <= 30 s as think time
                frac = ref.latest_raw_at(s_, e_, "act.stream_frac", t)
                sampled = frac is not None and float(frac.value) < 1.0 - 1e-9
                for i, name in enumerate(F.FEATURE_NAMES_V2):
                    if name == "think_time" and sampled:
                        continue
                    assert _close(name, float(R[t][i]), float(X[t][i])), \
                        (key, dt, t, name, float(R[t][i]), float(X[t][i]))
                n_checked += 1
    assert n_checked >= 80


def test_q_rows_equal_at_60_and_900(runs):
    for key in KEYS:
        A = _rows(runs[900.0], key, "feature.nat.q")
        B = _rows(runs[60.0], key, "feature.nat.q")
        assert len(A) >= 40
        assert set(A) == set(B)
        for t in A:
            for i, name in enumerate(F.FEATURE_NAMES_V2):
                assert _close(name, float(A[t][i]), float(B[t][i])), (key, t, name)
        assert not _rows(runs[3600.0], key, "feature.nat.q")     # Q not observable at 3600


def test_sketch_h_equal_and_live_rows(runs):
    ref = runs[3600.0]
    for key in KEYS:
        R = _rows(ref, key, "feature.sketch.h")
        X = _rows(runs[900.0], key, "feature.sketch.h")
        # the act.tokens block carries the templater's masking (see above); the
        # other four namespaces are the raw sets merged over the same hour
        blk = slice(16, 80)
        for t in R:
            assert np.allclose(R[t][blk], X[t][blk], atol=1e-6), (key, t)
        # live rows every tick, equal to the decision row on decision ticks
        L = _rows(runs[900.0], key, "feature.live.h")
        N = _rows(runs[900.0], key, "feature.nat.h")
        assert len(L) >= 24                          # every tick, retained 6 h
        for t, row in N.items():
            if t not in L:
                continue
            np.testing.assert_array_equal(np.isnan(L[t]), np.isnan(row))
            assert np.allclose(L[t][~np.isnan(row)], row[~np.isnan(row)])


# ------------------------------------------------------------ property form
def _write_tick(st, s, e, now, vals):
    for name, v in vals.items():
        st.add_raw(RawMetric(name=name, value=float(v), ts=now, system=s, entity=e,
                             method=AcquisitionMethod.PASSIVE_FLOW))


@pytest.mark.parametrize("case", range(20))
def test_property_additive_h_values_equal_direct_sums(case):
    rng = np.random.default_rng(1000 + case)
    base = 1_741_536_000.0                       # an epoch multiple of 3600
    n_ev = int(rng.integers(20, 400))
    span = 3 * 3600.0
    ev_t = np.sort(rng.uniform(0.0, span, n_ev))
    ev_b = rng.lognormal(7.0, 1.5, n_ev)
    ev_rtt = rng.uniform(1.0, 40.0, n_ev)
    dt = float(rng.choice([60.0, 300.0, 900.0, 3600.0]))
    off = float(rng.choice([0.0, 17.0, 433.0])) if dt < 3600.0 else 0.0
    st = MetricStore()
    eng = FeatureVectorEngine()
    from app.core.engine import Context
    k = 1
    while True:
        now = base + off + k * dt
        if now > base + span + 3600.0:
            break
        sel = (ev_t + base >= now - dt) & (ev_t + base < now)
        if sel.any():
            n = float(sel.sum())
            _write_tick(st, "s", "e", now, {"l4.flows": n, "l4.bytes_up": ev_b[sel].sum(),
                                           "l4.bytes_down": 2.0 * ev_b[sel].sum(),
                                           "l4.rtt_ms_avg": float(ev_rtt[sel].mean())})
        else:
            st.register_entity("s", "e")
        eng.run(Context(store=st, now=now, window_s=dt, config=dict(CFG)))
        k += 1
    ts, M = st.vec_since("s", "e", "feature.nat.h", 0.0)
    assert len(ts) >= 3
    for t, row in zip(ts.tolist(), M):
        if t - 3600.0 < base + off:              # the first window is only partly covered
            continue
        sel = (ev_t + base >= t - 3600.0) & (ev_t + base < t)
        n = float(sel.sum())
        assert row[IDX["flows"]] == pytest.approx(n)
        assert row[IDX["bytes_up"]] == pytest.approx(ev_b[sel].sum(), rel=1e-5)
        if n:
            assert row[IDX["rtt"]] == pytest.approx(ev_rtt[sel].mean(), rel=1e-5)
            assert row[IDX["updown_log"]] == pytest.approx(
                math.log((ev_b[sel].sum() + 1.0) / (2 * ev_b[sel].sum() + 1.0)), rel=1e-5)
        else:
            assert math.isnan(row[IDX["rtt"]])
        # an H decision tick holds an epoch multiple of 3600 s
        assert GR.decision(t, dt, "h", GR.CANONICAL)


# ------------------------------------------------------------------ Part B
SLOW = os.environ.get("APPMON_SLOW") == "1"


@pytest.fixture(scope="module")
def part_b():
    from tests_cadence_part_b import run_part_b      # tests/tests_cadence_part_b.py
    return run_part_b()


@pytest.mark.skipif(not SLOW, reason="Part B (whole pipeline, minutes): set APPMON_SLOW=1")
def test_part_b_live_60_equals_900(part_b):
    """cadence.md §13.2 Part B (i), (iii)-(v) as twin-run equalities: the same
    live observations at 60 s and re-batched at 900 s give equal H rows and
    the same incident / identity / B14 accumulator outcome."""
    out = part_b
    assert out["h_row_mismatch"] == 0
    a, b = out[60.0], out[900.0]
    assert a["identity_medium_plus"] == b["identity_medium_plus"]
    assert a["acc_alarms_b14"] == b["acc_alarms_b14"]
    assert abs(a["incidents_medium_plus"] - b["incidents_medium_plus"]) <= 1
    fa, fb = a["far_low"], b["far_low"]
    assert (fa == 0 and fb == 0) or 0.5 <= (fa / fb if fb else math.inf) <= 2.0


@pytest.mark.skipif(not SLOW, reason="Part B (whole pipeline, minutes): set APPMON_SLOW=1")
@pytest.mark.xfail(strict=False, reason=(
    "absolute null levels (\u00a713.2 ii-iv) are not met yet: B24 scores B11 timing "
    "from a cc-900 ring of 42 entries after the Fri-Sun warm-up (< 64), so it blends "
    "in B11's own pm, a G test at the float32 floor on ordinary Monday traffic "
    "(p ~ 2e-32 on every human persona, tick mode too; cadence.md \u00a717 open 1)"))
def test_part_b_null_levels(part_b):
    for dt in (60.0, 900.0):
        r = part_b[dt]
        assert r["single_tick_alarms"] <= 2, (dt, r)
        assert r["incidents_medium_plus"] == 0, (dt, r)
        assert r["identity_medium_plus"] == 0, (dt, r)
        assert r["acc_alarms_b14"] == 0, (dt, r)
