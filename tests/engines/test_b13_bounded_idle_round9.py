"""Round 9 (bounded-mode promotion owner): bounded mode replays the ticks an
IP spent outside the active / earned sets as the idle rows full mode learns.

PG9 pack A seed 0 (progressive.md 16.16.8, OC8-1): oa-portal 10.30.2.23 was
idle from 03-01 20:00 to 03-03 08:00. Full mode processed it on every tick
(B13 committed 37 zero-valued hourly bins at trust 1); bounded mode dropped it
from the active set after the 24-h linger, so B13's ring kept those hours
empty. On 03-13 the two modes' budget thresholds differed (bytes_down.8h
5.9e8 vs 8.9e8), full mode's budget_vol alarm at 04:00 (p 4.2e-5) was 6.9e-5
in bounded mode, and the incident full mode re-opened LOW was re-opened at
MEDIUM six hours later (gate 3: 8 vs 7 MEDIUM).

Now B13 appends the skipped ticks as implicit zero rows (budget._idle_catchup),
B28 back-fills their trust (governor.backfill_idle) and the gated learner
commits an implicit row only once the governor has covered its tick
(gating.commit_candidates extra_ts): the committed ring equals full mode's.
"""
from __future__ import annotations

import math

import numpy as np

from helpers import make_store, run_engine, set_trust

from test_b13_budget import DAY, E, H, S, T_MID, Feed

from app.engines.behavior.budget import MODEL, BudgetEngine
from app.engines.behavior.governor import GovernorEngine
from app.engines.behavior.lib import gating as G

DT = 3600.0
BOUNDED = {"lib3": {"resource_mode": "bounded"}}
N_TICKS = 5 * 24
GAP = range(2 * 24 + 20, 2 * 24 + 20 + 37)      # 37 idle hours, skipped in bounded mode


def _traffic(i: int, rng: np.random.Generator) -> float:
    h = i % 24
    return 2e5 * math.exp(0.3 * rng.normal()) if 9 <= h < 18 and i not in GAP else 0.0


def _run(bounded: bool, backfill: bool = True, governor_until: int = N_TICKS):
    st, eng = make_store(), BudgetEngine()
    feed = Feed(st)
    rng = np.random.default_rng(4)
    cfg = BOUNDED if bounded else None
    model = {"regime": "normal", "last_q": 0}
    for i in range(N_TICKS):
        now = T_MID + (i + 1) * DT
        up = _traffic(i, rng)
        if bounded and i in GAP:
            continue                       # outside A_t u E_t: no B01 row, no engine, no B28
        feed.tick(E, now, DT, up=up, down=3 * up, req=4.0 if up else 0.0,
                  writes=1.0 if up else 0.0)
        run_engine(eng, st, now, dt=DT, config=cfg)
        if i >= governor_until:
            continue                       # the governor has not run on this tick (yet)
        if bounded and backfill:
            GovernorEngine.backfill_idle(st, S, E, model, now, DT)     # B28 runs after B13
        set_trust(st, S, E, [now], 1.0)
    return st


def _state(st):
    m = st.get_model(S, E, MODEL)
    return m["state"], m


def test_idle_ticks_grid():
    assert G.idle_ticks(None, 10 * H, H) == []
    assert G.idle_ticks(9 * H, 10 * H, H) == []                 # processed on the previous tick
    assert G.idle_ticks(6 * H, 10 * H, H) == [7 * H, 8 * H, 9 * H]
    assert G.idle_ticks(0.0, 10 * DAY, H, max_age_s=2 * H) == [10 * DAY - 2 * H, 10 * DAY - H]
    # an hourly gap resumed at 900 s: the 900-s grid back to the last row
    assert G.idle_ticks(8 * H, 9 * H + 900.0, 900.0) == [8 * H + 900.0 * k for k in range(1, 4)] \
        + [9 * H]


def test_bounded_idle_gap_commits_the_same_ring_as_full_mode():
    full, _ = _state(_run(bounded=False))
    bnd, m = _state(_run(bounded=True))
    for k in ("hour", "sx", "sw", "sc", "ox", "oc"):
        assert np.array_equal(full[k], bnd[k]), k
    gap_hours = int((full["sw"] == 1.0).sum())
    assert gap_hours >= 37
    assert not m["live"].get("implicit")           # every implicit row committed


def test_without_trust_the_implicit_rows_wait():
    """Until the governor has covered the skipped ticks, the implicit rows are
    not committed at the missing-trust weight 0: they wait, and the gate does
    not pass them (B13 runs before B28 in a tick: on the resume tick and, here,
    on later ticks the governor did not reach)."""
    st = _run(bounded=True, governor_until=GAP[-1] + 1)
    s, m = _state(st)
    imp = m["live"].get("implicit") or []
    assert len(imp) == len(GAP)
    assert m["gate"].last_ts < min(imp)
    a_gap = {int(math.floor((t - 0.5 * DT + 8 * H) / H)) for t in imp}
    hit = np.isin(s["hour"], sorted(a_gap))
    assert not (s["sw"][hit] > 0).any() and not (s["oc"][hit] > 0).any()


def test_full_mode_never_replays():
    """Full mode: a tick B13 did not run on stays a gap (B01's feature rows
    exist for every tick in full mode; no implicit rows are made)."""
    st = _run(bounded=False)
    _s, m = _state(st)
    assert not m["live"].get("implicit")


def test_commit_candidates_stop_at_an_undecided_implicit_row():
    st = make_store()
    t = [T_MID + k * DT for k in range(1, 11)]
    for x in t[:3]:
        st.add_vec(S, E, "feature.active", x, np.array([1.0], np.float32), window_s=int(DT))
    set_trust(st, S, E, t[:3], 1.0)
    now = t[-1] + DT
    rows = G.commit_candidates(st, S, E, "x", -math.inf, now, DT, extra_ts=t[3:7])
    assert [r[0] for r in rows] == t[:3]          # implicit rows past the governor's tick wait
    set_trust(st, S, E, t[3:7], 1.0)              # back-filled
    rows = G.commit_candidates(st, S, E, "x", -math.inf, now, DT, extra_ts=t[3:7])
    assert [r[0] for r in rows] == t[:7] and all(r[1] == 1.0 for r in rows)
    # without extra_ts the result is the clock's rows, as before
    rows0 = G.commit_candidates(st, S, E, "x", -math.inf, now, DT)
    assert [r[0] for r in rows0] == t[:3]
