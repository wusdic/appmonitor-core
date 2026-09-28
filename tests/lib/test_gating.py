"""Tests for engines/behavior/lib/gating.py: delayed, trust-gated, reversible
learning (contract H) against a real MetricStore v2.

The toy learner is a decayed (W, S1, S2) accumulator whose update decays the
ROW weight for out-of-order rows, as the module doc requires, so "state ==
offline fold of the journal rows in ascending ts" is the invariant every
directive must preserve (checked exactly, or to 1e-9 relative where released
rows were folded out of order). Targets (docs/lib3/engines.md B03 (d)/(e),
B28 (d), contract H): commits lag now by exactly D; quarantined rows are held;
rollback_to restores the offline fit on rows <= tau within 1e-6; release
commits held rows with w_prov; rebase_from bumps the version and replays;
frozen stops commits; link seeding adds 0.5 A. A seeded random scenario
checks the invariant after every tick through rollbacks, releases and rebases.
"""
from __future__ import annotations

import json
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from app.core.store import MetricStore  # noqa: E402
from app.engines.behavior.lib import gating as G  # noqa: E402

S, E, A = "sysA", "10.0.0.1", "10.0.0.2"
T0 = 1_700_000_000.0
DT = 900.0
HL = 7 * 86400.0
X = "test.x"
INF = math.inf


# ------------------------------------------------------------ toy learner
def t_init():
    return (-INF, 0.0, 0.0, 0.0, 0)          # (t_newest, W, S1, S2, n_rebase)


def t_update(st, row, w):
    t, W, S1, S2, nr = st
    ts, x = row
    if ts >= t:
        g = 2.0 ** (-(ts - t) / HL) if math.isfinite(t) else 0.0
        return (ts, W * g + w, S1 * g + w * x, S2 * g + w * x * x, nr)
    a = w * 2.0 ** (-(t - ts) / HL)          # older row: decay the row, not the state
    return (t, W + a, S1 + a * x, S2 + a * x * x, nr)


def t_fetch(store, s, e, ts):
    v = store.vec_at(s, e, X, ts)
    if v is None or not np.isfinite(v[0]):
        return None
    return (ts, float(v[0]))


def t_merge(own, other, w):
    return (max(own[0], other[0]), own[1] + w * other[1], own[2] + w * other[2],
            own[3] + w * other[3], own[4])


def t_rebase_count(st, tau):
    return st[:4] + (st[4] + 1,)


def t_rebase_reset(st, tau):
    return (st[0], 0.0, 0.0, 0.0, st[4] + 1)


def learner(**kw):
    base = dict(name="toy", init=t_init, update=t_update, fetch=t_fetch,
                dump=lambda st: list(st), load=lambda b: tuple(b), merge=t_merge,
                on_rebase=t_rebase_count)
    base.update(kw)
    return G.GatedLearner(**base)


def fold(store, rows, st=None, e=E):
    """Offline fit: rows [(ts, w)] folded in ascending ts."""
    st = t_init() if st is None else st
    for ts, w in sorted(rows):
        r = t_fetch(store, S, e, ts)
        if r is not None:
            st = t_update(st, r, w)
    return st


def journal_fit(store, gate, e=E):
    return fold(store, [(r.ts, r.w_eff) for r in gate.journal], e=e)


def assert_state(a, b, rtol=1e-9):
    assert a[0] == b[0]
    np.testing.assert_allclose(a[1:4], b[1:4], rtol=rtol, atol=1e-12)


# ---------------------------------------------------------------- the sim
class Sim:
    """One entity, pipeline order per tick: B01 writes data at t, the
    learner steps at now = t, then B28 writes trust / trust_prov /
    quarantine at t (1-element float32 vec rings)."""

    def __init__(self, dt=DT, lrn=None, e=E, store=None):
        self.store = store if store is not None else MetricStore()
        self.dt, self.e = dt, e
        self.L = lrn or learner()
        self.state, self.gate = self.L.init(), G.GateState()
        self.i = -1

    def ts(self, i):
        return T0 + i * self.dt

    @property
    def now(self):
        return self.ts(self.i)

    def data(self, t, x):
        self.store.add_vec(S, self.e, "feature.active", t, [1.0])
        if x is not None:
            self.store.add_vec(S, self.e, X, t, [x])

    def gov(self, t, trust=1.0, prov=None, q=0.0):
        prov = trust if prov is None else prov
        for name, v in (("behavior.trust", trust), ("behavior.trust_prov", prov),
                        ("behavior.quarantine", q)):
            if v is not None:
                self.store.add_vec(S, self.e, name, t, [v])

    def control(self, **kw):
        self.store.put_model(S, self.e, "model.control", dict(kw))

    def tick(self, x=1.0, trust=1.0, prov=None, q=0.0, training=False):
        self.i += 1
        t = self.now
        self.data(t, x)
        self.state, self.gate = self.L.step(self.store, S, self.e, self.state, self.gate, t,
                                            self.dt, training=training)
        self.gov(t, trust, prov, q)
        return t

    def run(self, n, **kw):
        for _ in range(n):
            self.tick(**kw)


def xval(i):
    return float(np.float32(10.0 + (i % 7) - 0.5 * (i % 3)))


# ======================================================== frontier / delay
@pytest.mark.parametrize("dt,d", [(900, 4), (60, 10), (150, 4), (120, 5), (7, 86),
                                  (0.1, 6000), (3600, 4), (599.99, 4)])
def test_commit_delay_ticks(dt, d):
    assert G.commit_delay_ticks(dt) == d
    assert G.commit_frontier(T0, dt) == pytest.approx(T0 - d * dt, abs=1e-9)


def test_commit_delay_custom_and_invalid():
    assert G.commit_delay_ticks(60, 1800) == 30
    assert G.commit_delay_ticks(60, 0) == 4
    for bad in (0, -60, float("nan"), INF):
        with pytest.raises(ValueError):
            G.commit_delay_ticks(bad)


@pytest.mark.parametrize("dt", [900.0, 60.0])
def test_commits_lag_now_by_exactly_D(dt):
    sim = Sim(dt=dt)
    D = G.commit_delay_ticks(dt)
    for i in range(3 * D + 5):
        t = sim.tick(x=xval(i))
        if i < D:
            assert sim.gate.journal == [] and sim.gate.last_ts == -INF
        else:
            assert sim.gate.journal[-1].ts == t - D * dt
            assert sim.gate.last_ts == t - D * dt
            assert len(sim.gate.journal) == i - D + 1
    # the state is exactly the offline fit of rows <= now - D
    rows = [(sim.ts(k), 1.0) for k in range(sim.i - D + 1)]
    assert sim.state == fold(sim.store, rows)


# ============================================================ candidates
def test_commit_candidates_window_and_weights():
    st = MetricStore()
    for i in range(10):
        t = T0 + i * DT
        st.add_vec(S, E, "feature.active", t, [1.0])
        if i != 3:                                     # trust missing at row 3
            st.add_vec(S, E, "behavior.trust", t, [1.7 if i == 1 else 0.25 * (i % 5)])
        if i != 5:
            st.add_vec(S, E, "behavior.trust_prov", t, [float("nan") if i == 2 else -0.3 if i == 4 else 1.0])
    now = T0 + 9 * DT                                  # frontier = row 5
    rows = G.commit_candidates(st, S, E, "toy", T0 + 0 * DT, now, DT)
    assert [r[0] for r in rows] == [T0 + k * DT for k in (1, 2, 3, 4, 5)]
    w = {round((r[0] - T0) / DT): (r[1], r[2]) for r in rows}
    assert w[1][0] == 1.0                              # clipped
    assert w[3][0] == 0.0                              # missing live -> 0 (still returned)
    assert w[2][1] == 0.0 and w[4][1] == 0.0 and w[5][1] == 0.0   # NaN / negative / missing
    assert w[2][0] == 0.5
    tr = G.commit_candidates(st, S, E, "toy", T0, now, DT, training=True)
    assert {round((r[0] - T0) / DT): r[1] for r in tr}[3] == 1.0   # missing in training -> 1
    assert {round((r[0] - T0) / DT): r[2] for r in tr}[5] == 1.0
    assert G.commit_candidates(st, S, E, "toy", T0 + 5 * DT, now, DT) == []
    assert G.commit_candidates(st, S, "nobody", "toy", -INF, now, DT) == []
    assert G.commit_candidates(st, S, E, "toy", float("nan"), now, DT)[0][0] == T0


def test_is_quarantined():
    st = MetricStore()
    now = T0 + 10 * DT
    assert G.is_quarantined(st, S, E, now, DT) is False               # missing -> False
    st.add_vec(S, E, "behavior.quarantine", T0 + 8 * DT, [1.0])
    assert G.is_quarantined(st, S, E, now, DT) is True                # t-1 missing: latest < now
    st.add_vec(S, E, "behavior.quarantine", now - DT, [0.4])
    assert G.is_quarantined(st, S, E, now, DT) is False               # t-1 present, <= 0.5
    st.add_vec(S, E, "behavior.quarantine", now, [1.0])                # ts == now is not t-1
    assert G.is_quarantined(st, S, E, now, DT) is False
    assert G.is_quarantined(st, S, E, now + DT, DT) is True
    st2 = MetricStore()
    st2.add_vec(S, E, "behavior.quarantine", now, [1.0])
    assert G.is_quarantined(st2, S, E, now, DT) is False
    st2.add_vec(S, E, "behavior.quarantine", now + DT, [float("nan")])
    # NaN at t-1 is missing, not 0: the latest defined value (1 at now) holds
    assert G.is_quarantined(st2, S, E, now + 2 * DT, DT) is True
    st3 = MetricStore()
    st3.add_vec(S, E, "behavior.quarantine", now, [0.0])
    st3.add_vec(S, E, "behavior.quarantine", now + DT, [float("nan")])
    assert G.is_quarantined(st3, S, E, now + 2 * DT, DT) is False
    st4 = MetricStore()
    st4.add_vec(S, E, "behavior.quarantine", now, [float("nan")])
    assert G.is_quarantined(st4, S, E, now + DT, DT) is False           # only NaN: missing


def test_nan_quarantine_tick_does_not_release_the_gate():
    """Regression: a degraded governor tick (quarantine = NaN) inside a
    quarantine episode used to read as 'not quarantined' and commit the row."""
    sim = Sim()
    sim.run(20, x=2.0)
    sim.run(5, x=2.0, q=1.0)
    n_j, n_h = len(sim.gate.journal), len(sim.gate.held)
    sim.tick(x=500.0, trust=float("nan"), prov=float("nan"), q=float("nan"))
    sim.tick(x=500.0, q=1.0)                             # sees quarantine(t-1) = NaN
    assert len(sim.gate.journal) == n_j
    assert len(sim.gate.held) == n_h + 2


# ============================================================ hold / trust
def test_quarantined_rows_are_held_and_trust_weights_commits():
    sim = Sim()
    for i in range(30):
        sim.tick(x=xval(i), trust=0.5 if i % 4 == 0 else 1.0)
    j_before = list(sim.gate.journal)                  # rows 0..25
    for i in range(30, 40):                            # quarantine written at t = 30..39
        sim.tick(x=xval(i), trust=0.0, prov=0.8, q=1.0)
    # ticks 31..39 saw quarantine(t-1) = 1 -> rows 27..35 are held
    assert [round((r.ts - T0) / DT) for r in sim.gate.held] == list(range(27, 36))
    # weights are those written by the governor at the row's own ts
    assert [r.w_eff for r in sim.gate.held[:3]] == [1.0, 0.5, 1.0]        # rows 27..29
    assert all(r.w_eff == 0.0 and r.w_prov == pytest.approx(0.8) for r in sim.gate.held[3:])
    assert sim.gate.journal[: len(j_before)] == j_before
    # tick 30 still saw quarantine(29) = 0, so row 26 was the last commit
    assert [r.ts for r in sim.gate.journal[len(j_before):]] == [sim.ts(26)]
    assert sim.gate.last_ts == T0 + 35 * DT
    rows = [(sim.ts(k), 0.5 if k % 4 == 0 else 1.0) for k in range(27)]
    assert sim.state == fold(sim.store, rows)
    # fetch None skips the row: processed, neither committed nor held
    sim2 = Sim()
    for i in range(12):
        sim2.tick(x=None if i == 3 else xval(i))
    assert T0 + 3 * DT not in [r.ts for r in sim2.gate.journal + sim2.gate.held]
    assert sim2.gate.last_ts == T0 + 7 * DT


def test_frozen_stops_commits_and_clears_held():
    sim = Sim()
    sim.run(20, x=3.0)
    sim.run(6, x=3.0, q=1.0)
    assert sim.gate.held
    frozen_state, n_j = sim.state, len(sim.gate.journal)
    sim.control(version=1, frozen=True)
    sim.run(10, x=50.0, q=0.0)
    assert sim.gate.frozen and sim.gate.held == []
    assert sim.state == frozen_state and len(sim.gate.journal) == n_j
    assert sim.gate.last_ts == sim.now - 4 * DT       # rows are still consumed (dropped)
    sim.control(version=1, frozen=False)
    sim.run(3, x=3.0)
    assert not sim.gate.frozen
    assert len(sim.gate.journal) == n_j + 3            # only rows after the freeze
    assert sim.gate.journal[-1].ts == sim.now - 4 * DT


# ================================================================ rollback
@pytest.mark.parametrize("ckpt_every", [G.CKPT_EVERY_S, INF])
def test_rollback_restores_offline_fit_and_holds_later_rows(ckpt_every):
    """B03 (e): 100 clean rows, 20 attack rows with trust 1, rollback_to = t100."""
    sim = Sim(lrn=learner(ckpt_every_s=ckpt_every))
    for i in range(124):                               # B28 quarantines at the last tick
        sim.tick(x=xval(i) if i < 100 else 500.0 + i, trust=1.0, prov=0.9 if i >= 100 else 1.0,
                 q=1.0 if i == 123 else 0.0)
    assert len(sim.gate.journal) == 120                # all 120 rows committed
    tau = sim.ts(99)
    sim.control(version=1, rollback_to=tau)
    sim.tick(x=1.0, q=1.0)
    expect = fold(sim.store, [(sim.ts(k), 1.0) for k in range(100)])
    assert sim.state[0] == expect[0]
    np.testing.assert_allclose(sim.state[1:4], expect[1:4], rtol=1e-12, atol=0)
    assert abs(sim.state[2] / sim.state[1] - expect[2] / expect[1]) < 1e-6
    held = [round((r.ts - T0) / DT) for r in sim.gate.held]
    assert held == list(range(100, 121))                # the 20 attack rows (+ row 120) are held
    assert all(r.w_prov == pytest.approx(0.9) for r in sim.gate.held[:20])
    assert sim.gate.journal[-1].ts == tau
    info = sim.gate.applied["_last_rollback"]
    if ckpt_every == INF:
        assert info["ckpt_ts"] is None and info["replayed"] == 100
        assert sim.store.checkpoint_times(S, E, "toy") == []          # inf: never
    else:
        assert info["ckpt_ts"] is not None and info["ckpt_ts"] <= tau and info["replayed"] < 100
    # applied once: the same control on the next ticks changes nothing
    st, n = sim.state, len(sim.gate.held)
    sim.run(3, x=1.0, q=1.0)
    assert sim.state == st and len(sim.gate.held) == n + 3
    assert sim.gate.applied["rollback_to"] == tau


def test_rollback_rate_limit_and_depth():
    sim = Sim()
    sim.run(59, x=2.0)
    sim.tick(x=2.0, q=1.0)
    sim.control(version=1, rollback_to=sim.ts(40))
    sim.tick(x=2.0, q=1.0)
    assert sim.gate.journal[-1].ts == sim.ts(40)
    first_wall = sim.gate.last_rollback_ts
    sim.control(version=1, rollback_to=sim.ts(30))     # within the hour: deferred
    sim.tick(x=2.0, q=1.0)
    assert sim.gate.journal[-1].ts == sim.ts(40)
    assert sim.gate.applied["rollback_to"] == sim.ts(40)
    sim.run(3, x=2.0, q=1.0)                            # 4 ticks x 900 s = 1 h later
    assert sim.gate.last_rollback_ts == first_wall + 4 * DT
    assert sim.gate.journal[-1].ts == sim.ts(30)
    expect = fold(sim.store, [(sim.ts(k), 1.0) for k in range(31)])
    assert sim.state == expect
    # deeper than 7 d: recorded as applied, never executed
    st = sim.state
    sim.control(version=1, rollback_to=sim.now - 8 * 86400.0)
    sim.run(6, x=2.0, q=1.0)
    assert sim.state == st
    assert sim.gate.applied["_last_rollback"].get("skipped") == "depth"


def test_rollback_noop_when_nothing_after_tau():
    sim = Sim()
    sim.run(20, x=2.0)
    sim.tick(x=2.0, q=1.0)
    st = sim.state
    sim.control(version=1, rollback_to=sim.now)          # newer than the frontier
    sim.tick(x=2.0, q=1.0)
    assert sim.state == st and sim.gate.applied["_last_rollback"].get("noop")


# ============================================================ release / rebase
def test_release_commits_held_rows_with_w_prov():
    sim = Sim()
    sim.run(30, x=4.0)
    for i in range(30, 45):
        sim.tick(x=xval(i), trust=0.0, prov=0.6, q=1.0)
    held_ts = [r.ts for r in sim.gate.held]
    assert len(held_ts) == 14
    t0, t1 = held_ts[3], held_ts[12]                   # rows 30..39 (trust 0, prov 0.6)
    sim.control(version=1, release=[t0, t1])
    sim.tick(x=4.0, trust=1.0, q=0.0)                  # this tick's row 41 is still held
    rel = [r for r in sim.gate.journal if t0 <= r.ts <= t1]
    assert len(rel) == 10
    assert all(r.w_eff == pytest.approx(0.6) and r.w_prov == pytest.approx(0.6) for r in rel)
    assert [r.ts for r in sim.gate.held] == held_ts[:3] + [held_ts[13], sim.ts(41)]
    assert all(not (t0 <= r.ts <= t1) for r in sim.gate.held)
    assert [r.ts for r in sim.gate.journal] == sorted(r.ts for r in sim.gate.journal)
    assert_state(sim.state, journal_fit(sim.store, sim.gate))
    # released rows carry trust_prov, not trust (0)
    no_rel = fold(sim.store, [(r.ts, r.w_eff) for r in sim.gate.journal if not t0 <= r.ts <= t1])
    assert sim.state[1] > no_rel[1]


def test_rollback_after_release_skips_stale_checkpoints():
    """Checkpoints written while rows 26..35 were held lack them; after the
    release a rollback must not restore one of those (barrier) or the
    released rows would be silently lost."""
    sim = Sim()
    sim.run(30, x=5.0)
    for i in range(30, 40):
        sim.tick(x=xval(i), trust=0.0, prov=0.7, q=1.0)
    for i in range(40, 90):                             # quarantine lifted, no release yet
        sim.tick(x=xval(i))
    held = [r.ts for r in sim.gate.held]
    assert len(held) == 10 and held[0] == sim.ts(27)
    stale = [t for t in sim.store.checkpoint_times(S, E, "toy") if held[0] < t <= sim.ts(70)]
    assert stale                                        # the trap exists
    sim.control(version=1, release=[held[0], held[-1]])
    sim.tick()
    assert sim.gate.held == []
    assert_state(sim.state, journal_fit(sim.store, sim.gate))
    sim.run(3)
    sim.tick(q=1.0)
    sim.control(version=1, release=[held[0], held[-1]], rollback_to=sim.ts(70))
    sim.tick(q=1.0)
    expect = fold(sim.store, [(r.ts, r.w_eff) for r in sim.gate.journal])
    assert sim.gate.journal[-1].ts == sim.ts(70)
    rel = [r for r in sim.gate.journal if held[0] <= r.ts <= held[-1]]
    assert len(rel) == 10 and all(r.w_eff == r.w_prov for r in rel)
    assert all(r.w_eff == pytest.approx(0.7) for r in rel if r.ts >= sim.ts(30))
    assert_state(sim.state, expect)
    assert sim.gate.applied["_last_rollback"]["ckpt_ts"] is None or \
        sim.gate.applied["_last_rollback"]["ckpt_ts"] < held[0]


def test_rebase_bumps_version_and_replays_held_rows():
    sim = Sim(lrn=learner(on_rebase=t_rebase_reset))
    sim.run(40, x=3.0)
    for i in range(40, 60):
        sim.tick(x=30.0 + (i % 3), trust=0.0, prov=0.9, q=1.0)
    held = [r.ts for r in sim.gate.held]
    w_prov = {r.ts: r.w_prov for r in sim.gate.held}   # float32(0.9)
    tau = sim.ts(45)
    assert held[0] < tau < held[-1]
    sim.control(version=3, branch=1, rebase_from=tau)
    sim.tick(x=31.0, trust=1.0, q=0.0)
    g = sim.gate
    assert g.version == 3 and g.branch == 1
    assert [r.ts for r in g.held] == [sim.now - 4 * DT]  # only this tick's row (q(t-1) = 1)
    assert sim.state[4] == 1                             # hook ran once
    # the hook resets the stats: only rows >= tau survive, folded with w_prov
    post = [(t, w_prov[t]) for t in held if t >= tau]
    exp = fold(sim.store, post)
    assert_state(sim.state, (sim.state[0],) + exp[1:4])
    assert all(r.w_eff == w_prov[r.ts] for r in g.journal if r.ts in w_prov)
    assert sum(r.ts in w_prov for r in g.journal) == len(held)
    sim.run(3, x=31.0)
    assert sim.state[4] == 1 and sim.gate.version == 3   # applied once
    # a version bump without rebase_from only records the version
    sim.control(version=4, branch=1, rebase_from=tau)
    sim.tick(x=31.0)
    assert sim.gate.version == 4 and sim.state[4] == 1


@pytest.mark.parametrize("ckpt_every", [G.CKPT_EVERY_S, INF])
def test_rollback_after_rebase_reapplies_hook(ckpt_every):
    sim = Sim(lrn=learner(on_rebase=t_rebase_reset, ckpt_every_s=ckpt_every))
    sim.run(40, x=3.0)
    for i in range(40, 60):
        sim.tick(x=30.0 + (i % 3), trust=0.0, prov=0.9, q=1.0)
    tau_r = sim.ts(45)
    sim.control(version=2, rebase_from=tau_r)
    for i in range(60, 100):
        sim.tick(x=30.0 + (i % 3), q=1.0 if i == 99 else 0.0)
    sim.control(version=2, rebase_from=tau_r, rollback_to=sim.ts(80))
    sim.tick(q=1.0)
    rows = [(r.ts, r.w_eff) for r in sim.gate.journal if r.ts >= tau_r]
    exp = fold(sim.store, rows)
    assert sim.gate.journal[-1].ts == sim.ts(80)
    assert sim.state[4] == 1
    assert_state(sim.state, (sim.state[0],) + exp[1:4])


# ============================================================= link seeding
def a_state(store):
    for i in range(10):
        store.add_vec(S, A, X, T0 + i * DT, [7.0])
    return fold(store, [(T0 + i * DT, 1.0) for i in range(10)], e=A)


def test_link_seeding_adds_half_of_A_once():
    sim = Sim()
    sim.run(12, x=2.0)
    sA = a_state(sim.store)
    own = sim.state
    load = {A: sA}.get
    st, g = sim.L.seed_from_link(sim.store, S, E, own, sim.gate, load)
    assert (st, g.link_version) == (own, 0)              # no model.link yet
    sim.store.put_model(S, "__system__", "model.link",
                        {"version": 1, "links": [{"from": A, "to": E, "ts": sim.now}]})
    st, g = sim.L.seed_from_link(sim.store, S, E, own, sim.gate, load)
    assert st[1] == own[1] + 0.5 * sA[1] and st[2] == own[2] + 0.5 * sA[2]
    assert g.link_version == 1
    st2, g2 = sim.L.seed_from_link(sim.store, S, E, st, g, load)
    assert st2 == st                                      # once per link
    sim.store.put_model(S, "__system__", "model.link", {"version": 2, "links": [
        {"from": A, "to": E, "ts": sim.now}, {"from": "10.9.9.9", "to": "10.8.8.8", "ts": sim.now}]})
    st3, g3 = sim.L.seed_from_link(sim.store, S, E, st, g2, load)
    assert st3 == st and g3.link_version == 2
    # retracted links, missing A and merge=None never seed
    sim.store.put_model(S, "__system__", "model.link", {"version": 3, "links": {
        "x": {"from": "10.7.7.7", "to": E, "ts": sim.now, "retracted": True},
        "y": {"from": "10.6.6.6", "to": E, "ts": sim.now}}})
    st4, g4 = sim.L.seed_from_link(sim.store, S, E, st, g3, load)
    assert st4 == st and g4.link_version == 3
    assert {sd["from"]: sd["status"] for sd in g4.applied["_seeds"]}["10.6.6.6"] == "empty"
    nomerge = learner(merge=None)
    st5, g5 = nomerge.seed_from_link(sim.store, S, E, own, G.GateState(), load)
    assert st5 == own and g5.link_version == 0


def seeded_sim(ckpt_every=G.CKPT_EVERY_S):
    sim = Sim(lrn=learner(ckpt_every_s=ckpt_every))
    sA = a_state(sim.store)
    sim.run(43, x=2.0)                                   # committed rows 0..38
    t_link = sim.ts(41)
    sim.store.put_model(S, "__system__", "model.link",
                        {"version": 1, "links": [{"from": A, "to": E, "ts": t_link}]})
    sim.state, sim.gate = sim.L.seed_from_link(sim.store, S, E, sim.state, sim.gate, {A: sA}.get)
    pre = fold(sim.store, [(sim.ts(k), 1.0) for k in range(39)])
    return sim, sA, t_link, pre


def test_rollback_to_link_time_undoes_seed():
    sim, sA, t_link, _ = seeded_sim()
    for i in range(43, 61):
        sim.tick(x=2.0, q=1.0 if i == 60 else 0.0)
    assert any(t > t_link for t in sim.store.checkpoint_times(S, E, "toy"))
    sim.control(version=1, rollback_to=t_link)
    sim.tick(q=1.0)
    assert sim.state == fold(sim.store, [(sim.ts(k), 1.0) for k in range(42)])
    st, g = sim.L.seed_from_link(sim.store, S, E, sim.state, sim.gate, {A: sA}.get)
    assert st == sim.state                                 # undone, not re-seeded
    assert g.applied["_seeds"][0]["status"] == "undone"


@pytest.mark.parametrize("ckpt_every", [G.CKPT_EVERY_S, INF])
def test_later_rollback_keeps_seed(ckpt_every):
    sim, sA, t_link, pre = seeded_sim(ckpt_every)
    for i in range(43, 61):
        sim.tick(x=2.0, q=1.0 if i == 60 else 0.0)
    tau = sim.ts(50)
    sim.control(version=1, rollback_to=tau)
    sim.tick(q=1.0)
    st, g = sim.L.seed_from_link(sim.store, S, E, sim.state, sim.gate, {A: sA}.get)
    if ckpt_every == INF:                                   # lost by restore -> re-seeded
        exp = t_merge(fold(sim.store, [(sim.ts(k), 1.0) for k in range(51)]), sA, 0.5)
        assert st == exp and g.applied["_seeds"][0]["status"] == "applied"
    else:                                                   # restored from a seeded checkpoint
        exp = fold(sim.store, [(sim.ts(k), 1.0) for k in range(39, 51)], st=t_merge(pre, sA, 0.5))
        assert_state(sim.state, exp, rtol=1e-12)
        assert st == sim.state


def test_pending_seed_is_dropped_when_link_retracted():
    sim, sA, t_link, _ = seeded_sim(INF)
    for i in range(43, 61):
        sim.tick(x=2.0, q=1.0 if i == 60 else 0.0)
    sim.control(version=1, rollback_to=sim.ts(50))
    sim.tick(q=1.0)
    assert sim.gate.applied["_seeds"][0]["status"] == "pending"
    sim.store.put_model(S, "__system__", "model.link", {"version": 2, "links": [
        {"from": A, "to": E, "ts": t_link, "retracted": True}]})
    st, g = sim.L.seed_from_link(sim.store, S, E, sim.state, sim.gate, {A: sA}.get)
    assert st == sim.state and g.applied["_seeds"][0]["status"] == "undone"
    assert g.link_version == 2


def test_training_mode_defaults_missing_trust_to_one():
    sim = Sim()
    for i in range(10):
        sim.i += 1
        t = sim.now
        sim.data(t, xval(i))
        sim.state, sim.gate = sim.L.step(sim.store, S, E, sim.state, sim.gate, t, DT,
                                         training=True)        # no governor output at all
    assert [r.w_eff for r in sim.gate.journal] == [1.0] * 6
    assert sim.state == fold(sim.store, [(sim.ts(k), 1.0) for k in range(6)])
    live = Sim()
    for i in range(10):
        live.i += 1
        live.data(live.now, xval(i))
        live.state, live.gate = live.L.step(live.store, S, E, live.state, live.gate,
                                            live.now, DT)
    assert [r.w_eff for r in live.gate.journal] == [0.0] * 6     # fail safe
    assert live.state[1] == 0.0


# ============================================================ checkpoints
class RecStore:
    """Real MetricStore behind a proxy that records every method used."""

    def __init__(self, store):
        self._s = store
        self.calls = []
        self.puts = []

    def __getattr__(self, name):
        fn = getattr(self._s, name)
        self.calls.append(name)
        if name == "put_checkpoint":
            def put(*a, **kw):
                self.puts.append(a[3] if len(a) > 3 else kw["ts"])
                return fn(*a, **kw)
            return put
        return fn


def test_checkpoint_cadence_ts_and_blob_isolation():
    sim = Sim()
    rec = RecStore(sim.store)
    written = []
    for i in range(60):
        sim.i += 1
        t = sim.now
        sim.data(t, xval(i))
        n0 = len(rec.puts)
        sim.state, sim.gate = sim.L.step(rec, S, E, sim.state, sim.gate, t, DT)
        if len(rec.puts) > n0:
            assert rec.puts[-1] == sim.gate.journal[-1].ts          # last committed ts
            written.append(G.commit_frontier(t, DT))
        sim.gov(t)
    assert len(written) >= 12
    assert all(b - a >= G.CKPT_EVERY_S for a, b in zip(written, written[1:]))
    ts0, blob = sim.store.get_checkpoint(S, E, "toy")
    assert tuple(blob) == fold(sim.store, [(r.ts, 1.0) for r in sim.gate.journal if r.ts <= ts0])
    blob.append("mutated")                                             # the learner's copy is
    assert sim.store.get_checkpoint(S, E, "toy")[1] is blob          # the store's; load copies


def test_only_contract_store_calls():
    sim = Sim(lrn=learner(on_rebase=t_rebase_reset))
    rec = RecStore(sim.store)
    sA = a_state(sim.store)
    sim.store.put_model(S, "__system__", "model.link",
                        {"version": 1, "links": [{"from": A, "to": E, "ts": T0 + 30 * DT}]})

    def tick(i, q=0.0, trust=1.0):
        sim.i += 1
        t = sim.now
        sim.data(t, xval(i))
        sim.state, sim.gate = sim.L.step(rec, S, E, sim.state, sim.gate, t, DT)
        sim.state, sim.gate = sim.L.seed_from_link(rec, S, E, sim.state, sim.gate, {A: sA}.get)
        sim.gov(t, trust=trust, prov=0.5, q=q)

    for i in range(50):
        tick(i, q=1.0 if 20 <= i < 30 else 0.0)
    sim.control(version=1, release=[T0, T0 + 24 * DT])
    for i in range(50, 60):
        tick(i)
    sim.control(version=1, release=[T0, T0 + 24 * DT], rollback_to=T0 + 40 * DT)
    for i in range(60, 70):
        tick(i, q=1.0)
    sim.control(version=2, release=[T0, T0 + 24 * DT], rollback_to=T0 + 40 * DT,
                rebase_from=T0 + 45 * DT, allow_drift=0.02)
    for i in range(70, 75):
        tick(i)
    assert sim.gate.version == 2 and sim.gate.allow_drift == 0.02
    allowed = {"vec_since", "vec_at", "get_model", "get_checkpoint", "put_checkpoint"}
    assert set(rec.calls) <= allowed, set(rec.calls) - allowed
    assert {"get_checkpoint", "put_checkpoint", "vec_at", "vec_since", "get_model"} <= set(rec.calls)


def test_prune_journal_and_held():
    sim = Sim(dt=3600.0)
    for i in range(24 * 10):
        sim.tick(x=1.0, q=1.0 if 20 <= i < 26 else 0.0)
    now = sim.now
    assert sim.gate.journal[0].ts >= now - G.JOURNAL_MAX_AGE_S
    assert sim.gate.journal[0].ts < now - G.JOURNAL_MAX_AGE_S + 3600.0
    assert sim.gate.held == []                            # held rows from day 1 aged out


@pytest.mark.parametrize("depth_h", [24, 150])
def test_rollback_after_journal_pruning_flags_incomplete_replay(depth_h):
    """After 12 d the journal holds 8 d. A rollback is exact when the
    restored checkpoint is inside the journal window, and is flagged
    complete=False (never silently) when the store only retains an older
    checkpoint (MetricStore v2 geometric retention can keep a >= 168 h floor
    that is ~290 h old)."""
    sim = Sim(dt=3600.0)
    n = 24 * 12
    for i in range(n):
        sim.tick(x=xval(i), q=1.0 if i == n - 1 else 0.0)
    assert sim.gate.applied["_jfloor"] == sim.gate.journal[0].ts - 3600.0
    tau = sim.now - depth_h * 3600.0
    sim.control(version=1, rollback_to=tau)
    sim.tick(q=1.0)
    info = sim.gate.applied["_last_rollback"]
    k = int(round((tau - T0) / 3600.0))
    exp = fold(sim.store, [(sim.ts(j), 1.0) for j in range(k + 1)])
    exact = bool(np.allclose(sim.state[1:4], exp[1:4], rtol=1e-9))
    assert info["complete"] == exact
    if depth_h == 24:
        assert exact and info["history_gap_s"] == 0.0
    else:
        assert exact or info["history_gap_s"] > 0.0


# ================================================================= errors
def test_replay_error_freezes_and_reraises():
    boom = {"on": False}

    def fetch(store, s, e, ts):
        if boom["on"]:
            raise RuntimeError("replay failed")
        return t_fetch(store, s, e, ts)

    sim = Sim(lrn=learner(fetch=fetch, ckpt_every_s=INF))
    sim.run(30, x=2.0)
    gate = sim.gate
    boom["on"] = True
    sim.control(version=1, rollback_to=sim.ts(10))
    with pytest.raises(RuntimeError):
        sim.L.step(sim.store, S, E, sim.state, gate, sim.now + DT, DT)
    assert gate.frozen is True
    assert "rollback_to" not in gate.applied


# ============================================================ GateState io
def test_gatestate_roundtrip_and_json():
    assert G.GateState.from_dict(None) == G.GateState()
    assert G.GateState.from_dict({}) == G.GateState()
    sim, sA, t_link, _ = seeded_sim()
    sim.run(10, x=2.0, q=1.0)
    sim.control(version=1, rollback_to=sim.ts(40))
    sim.tick(q=1.0)
    g = sim.gate
    d = g.to_dict()
    json.dumps(d, allow_nan=False)                        # JSON-safe (no inf / nan)
    g2 = G.GateState.from_dict(json.loads(json.dumps(d)))
    assert g2 == g
    assert g2.to_dict() == d
    fresh = G.GateState().to_dict()
    assert fresh["last_ts"] is None and fresh["held"] == [] and fresh["frozen"] is False
    assert G.GateState.from_dict(g) is g                   # objects pass through
    # the in-memory round trip shares the immutable rows (no O(journal) work)
    g3 = G.GateState.from_dict(d)
    assert g3 == g and g3.journal is not g.journal and g3.journal[0] is g.journal[0]
    with pytest.raises(AttributeError):
        g.journal[0].w_eff = 0.0
    assert G.GateState.from_dict({"held": [[3.0, 0.5, 1.0], {"ts": 1.0, "w_eff": 1, "w_prov": 1}]}
                                 ).held == [G.CommitRow(1.0, 1.0, 1.0), G.CommitRow(3.0, 0.5, 1.0)]


# ============================================================ pure helpers
def test_control_directives():
    empty = G.control_directives(None)
    assert set(empty) == {"version", "branch", "rebase_from", "rollback_to", "release",
                          "frozen", "allow_drift", "accepted_class_change"}
    assert all(v is None for v in empty.values())
    d = G.control_directives({"version": 3.0, "branch": "2", "rollback_to": np.float64(5.5),
                              "release": [9, 4], "frozen": 0, "allow_drift": None,
                              "rebase_from": float("nan"), "accepted_class_change": "role3"})
    assert d == {"version": 3, "branch": 2, "rebase_from": None, "rollback_to": 5.5,
                 "release": (4.0, 9.0), "frozen": False, "allow_drift": None,
                 "accepted_class_change": "role3"}
    assert G.control_directives({"release": [1.0]})["release"] is None
    assert G.control_directives({"release": {"t0": 1, "t1": 2}})["release"] == (1.0, 2.0)
    assert G.control_directives({"allow_drift": 0.05})["allow_drift"] == 0.05


def test_reference_eligible():
    t = T0
    assert G.reference_eligible(t, 1.0, [])
    assert G.reference_eligible(t, 1.0, None)
    assert not G.reference_eligible(t, 0.99, [])
    assert not G.reference_eligible(t, float("nan"), [])
    assert not G.reference_eligible(t, 1.0, [t + 86400.0])            # boundary inclusive
    assert G.reference_eligible(t, 1.0, [t + 86401.0, t - 90000.0, float("nan")])
    assert not G.reference_eligible(t, 1.0, np.array([t - 3600.0]))
    assert G.reference_eligible(t, 1.0, [t - 3600.0], window_s=600.0)


# ======================================================= seeded scenario
def test_random_scenario_state_always_equals_journal_fit():
    """Random quarantine episodes, trust levels, missing rows, releases,
    rollbacks and rebases: after every tick the state equals the offline fold
    of the journal rows with their recorded weights, and the (commuting)
    counting hook has run exactly once per rebase still in effect."""
    rng = np.random.default_rng(20260927)
    sim = Sim(lrn=learner(on_rebase=t_rebase_count))
    ctl = {"version": 0}
    q = 0.0
    n_rb = n_rel = n_reb = 0
    for i in range(420):
        if rng.random() < 0.06:
            q = 1.0 - q
        trust = float(rng.choice([0.0, 0.5, 1.0], p=[0.2, 0.2, 0.6]))
        x = None if rng.random() < 0.05 else xval(i) + float(rng.integers(0, 4))
        if i > 30 and rng.random() < 0.07:
            kind = rng.choice(["rollback", "release", "rebase"])
            held = sim.gate.held
            if kind == "rollback":
                ctl["rollback_to"] = sim.now - float(rng.integers(5, 60)) * DT
                n_rb += 1
            elif kind == "release" and held:
                a, b = sorted(rng.choice(len(held), 2))
                ctl["release"] = [held[a].ts, held[b].ts]
                n_rel += 1
            elif kind == "rebase" and held:
                ctl["version"] += 1
                ctl["rebase_from"] = held[int(rng.integers(0, len(held)))].ts
                n_reb += 1
            sim.control(**ctl)
        sim.tick(x=x, trust=trust, prov=min(1.0, trust + 0.5), q=q)
        assert_state(sim.state, journal_fit(sim.store, sim.gate)), i
        assert sim.state[4] == len(sim.gate.applied.get("_hooks") or []), i
        ts_all = [r.ts for r in sim.gate.journal] + [r.ts for r in sim.gate.held]
        assert len(ts_all) == len(set(ts_all))                     # disjoint
        assert [r.ts for r in sim.gate.journal] == sorted(r.ts for r in sim.gate.journal)
        assert [r.ts for r in sim.gate.held] == sorted(r.ts for r in sim.gate.held)
    assert n_rb >= 3 and n_rel >= 2 and n_reb >= 1


def test_rebase_after_long_absence_keeps_old_checkpoint_restorable():
    """Regression: a rebase hook recorded while the journal was empty (every
    commit pruned after > 8 d of inactivity) used a (-inf, q] barrier, so a
    later rollback to before the rebase restored init() and wiped the model
    although a valid checkpoint at the last commit existed."""
    dt = 3600.0
    sim = Sim(dt=dt)
    for i in range(40):
        sim.tick(x=xval(i))                              # rows 0..35 committed
    old_rows = [(r.ts, r.w_eff) for r in sim.gate.journal]
    assert sim.store.get_checkpoint(S, E, "toy")[0] == old_rows[-1][0]
    sim.i += 9 * 24                                      # 9 d without clock rows
    for i in range(12):
        sim.tick(x=3.0, trust=0.0, prov=0.8, q=1.0)
    assert sim.gate.journal == [] and sim.gate.applied["_jfloor"] == old_rows[-1][0]
    tau_r = sim.gate.held[0].ts
    sim.control(version=1, rebase_from=tau_r)
    for i in range(6):
        sim.tick(x=3.0, q=1.0 if i == 5 else 0.0)
    assert sim.state[4] == 1 and all(r.ts > tau_r for r in sim.gate.held)
    tau = tau_r - 12 * dt                                # inside the gap, < 7 d ago
    sim.control(version=1, rebase_from=tau_r, rollback_to=tau)
    sim.tick(q=1.0)
    info = sim.gate.applied["_last_rollback"]
    assert info["ckpt_ts"] == old_rows[-1][0] and info["complete"] is True
    assert sim.state == fold(sim.store, old_rows)        # the old model, hook undone
    assert sim.gate.journal == [] and len(sim.gate.held) >= 12


def test_step_never_mutates_the_dict_its_gate_came_from():
    """from_dict shares to_dict's row lists (no O(journal) copy per tick);
    step / apply_control / seed_from_link must therefore never write to them."""
    sim, sA, t_link, _ = seeded_sim()
    sim.run(6, x=2.0, q=1.0)
    d = sim.gate.to_dict()
    snap = json.dumps(d, allow_nan=False)
    g = G.GateState.from_dict(d)
    assert g.journal is d["journal"]                      # shared on purpose
    sim.control(version=1, release=[sim.gate.held[0].ts, sim.gate.held[-1].ts])
    st, g2 = sim.L.step(sim.store, S, E, sim.state, g, sim.now + DT, DT)
    st, g2 = sim.L.seed_from_link(sim.store, S, E, st, g2, {A: sA}.get)
    assert json.dumps(d, allow_nan=False) == snap
    assert g2.journal is not d["journal"] and len(g2.journal) > len(d["journal"])


# --------------------------------------------------- spec v2.1 window trust
def test_window_min_trust_for_grain_learners():
    """commit_candidates(window_s=G): a grain row's weight is the MINIMUM
    trust over (ts - G, ts]; window_s=None keeps the per-row v2 lookup."""
    st = MetricStore()
    dt, G_h = 900.0, 3600.0
    for k in range(12):
        t = T0 + k * dt
        st.add_vec(S, E, "feature.meta.h" if k % 4 == 3 else "clock.none", t, np.array([1.0]))
        st.add_vec(S, E, G.TRUST, t, np.array([0.2 if k == 5 else 1.0], dtype=np.float32))
        st.add_vec(S, E, G.TRUST_PROV, t, np.array([1.0], dtype=np.float32))
    now = T0 + 20 * dt
    rows = G.commit_candidates(st, S, E, "h", -INF, now, dt, clock="feature.meta.h",
                               window_s=G_h)
    assert [r[0] for r in rows] == [T0 + 3 * dt, T0 + 7 * dt, T0 + 11 * dt]
    assert [round(r[1], 6) for r in rows] == [1.0, 0.2, 1.0]      # hour holding tick 5
    v2 = G.commit_candidates(st, S, E, "h", -INF, now, dt, clock="feature.meta.h")
    assert [round(r[1], 6) for r in v2] == [1.0, 1.0, 1.0]
