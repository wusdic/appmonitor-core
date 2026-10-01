"""P06 learning hygiene: a row P03 judged a violation never becomes the
pattern's stated hard bound (docs/lib3/progressive.md §6.9.3, §6.10;
content_bounds._ledger, pbounds.clean_range).

Pack O, anomaly A3 (day 18): a 12 KB injection login was scored above the
login node's range but not damped (its event p ~ 4e-3 > 1e-4), so P04 learned
it at full weight and the OA login statement read "全部在 0-12 KB"."""
from __future__ import annotations

import numpy as np

from pcontent_oracle import OracleLearner

from app.core.engine import Context
from app.core.store import MetricStore
from app.engines.behavior.content_bounds import ContentBoundsEngine
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV

CFG = {"progressive": {"enabled": True}, "grain_mode": "tick", "strict": True}
DAY = 86400.0
T0 = 1_788_220_800.0


def _run(flag: bool, flags=None, tail=None):
    store = MetricStore()
    orc = OracleLearner(store, "oa", T0, ["POST /login"], config=CFG)
    eng = ContentBoundsEngine()
    r = np.random.default_rng(0)
    nid = orc.node_for("POST /login")
    for d in range(14):
        T = T0 + d * DAY + 12 * 3600
        bb = EV.BatchBuilder("oa")
        rows = [float(r.uniform(1024, 2048)) for _ in range(12)]
        if d == 10:
            rows.append(12288.0)                           # the injection login
        if tail and d in tail:
            rows.append(tail[d])                           # a legitimate tail login
        for i, v in enumerate(rows):
            bb.add(T - 3600 + i * 7.0, f"192.168.1.{20 + i}", {"http.route": "POST /login",
                                                              "body.len": v}, 1.0)
        b = bb.build(T - 43200, T)
        store.add_batch("oa", EV.EVT_BATCH, T, b)
        rr = np.arange(b.n, dtype=np.int32)
        vt = np.zeros(b.n)
        if flag and d == 10:
            vt[-1] = 4.0                                  # P03: content p <= 1e-3
        cols = {"leaf": EV.Col(rr, np.full(b.n, float(nid))), "vtype": EV.Col(rr, vt),
                "damp": EV.Col(rr, np.ones(b.n))}
        fl = (flags or {}).get(d)
        if fl:                                            # P03's flags on the day's last row
            cols["flags"] = EV.Col(np.asarray([b.n - 1], dtype=np.int32),
                                   np.asarray([fl], dtype=object))
        store.add_batch("oa", EV.PAT_ASSIGN, T, b.aligned(cols, {"kind": 0, "tree_key": "oa"}))
        orc.learn(b)                                      # P04 learns it at full weight
        eng.safe_run(Context(store=store, now=T + 3600, window_s=60, config=dict(CFG)))
    rec = MP.get_model(store, "oa", MP.PBOUNDS)["nodes"][0][nid]["attrs"]["body.len"]
    return rec


def test_flagged_row_is_not_the_stated_maximum():
    rec = _run(True)
    assert rec["range"][1] <= 2048.0 + 1e-6, rec["range"]
    assert rec["n_rng"] >= 100
    raw = _run(False)                                     # nothing flagged: the row is data
    assert raw["range"][1] >= 12288.0 - 1e-6


def test_informational_flags_do_not_exclude_a_legitimate_extreme():
    """A value outside a young node's observed range is flagged 'above_range'
    (and, with the body's padding key, 'length' / 'grammar') at an ordinary
    p-value: that is how an observed range grows (P(next outside) = 2/(n+1)).
    Counting such flags as violations froze the range at its first days'
    extremes (pack O, GA login: the 2.6-2.7 KB and 0.6 KB tail logins of the
    0.5-3 KB truth were all flagged and dropped; the range stayed 1-1.8 KB),
    and flooded the FIFO ledger so that real violations were evicted. Only a
    significant typed p, damping or an injection shape makes a violation."""
    flags = {4: "above_range,grammar,length", 6: "below_range",
             10: "above_range,injection_shape"}          # day 10: the 12 KB injection login
    rec = _run(False, flags=flags, tail={4: 2900.0, 6: 600.0})
    assert abs(rec["range"][1] - 2900.0) < 1e-6, rec["range"]   # tail kept, injection not
    assert abs(rec["range"][0] - 600.0) < 1e-6, rec["range"]


def test_violating_extreme_drops_only_its_own_side_of_the_day():
    """When a day's maximum is a violating row and the exceedance reservoir
    holds no replacement, only that day's maximum leaves the range: its
    minimum is still a clean observation. Dropping the whole day lost the
    clean extremes of every day a damped row topped (pack O HEAD 53cd151, GA
    login node: 7 of 10 days dropped, n_rng = 3, range 1.2-1.5 KB)."""
    from app.engines.behavior.lib import pbounds as PB
    from app.engines.behavior.lib import pnode as PN
    num = PN.NumSummary(log=False)
    r = np.random.default_rng(1)
    day0 = 739500
    excl = {}
    for k in range(5):
        vals = list(r.uniform(1100.0, 1900.0, 6))
        if k == 2:
            vals += [600.0, 2900.0]                      # a clean low login, a violating 2.9 KB one
            excl[day0 + k] = [2900.0]
        for j, v in enumerate(vals):
            num.update(v, T0 + k * DAY + j, 1.0, 1.0, day=day0 + k)
    lo, hi, n, dropped = PB.clean_range(num, day0 + 4, excl, day_of=lambda ts: day0)
    assert lo == 600.0, (lo, hi)                        # the day's clean minimum is kept
    assert hi < 1900.0 + 1e-9, (lo, hi)                 # the violation is not the maximum
    assert dropped == 0
    # n_rng is the smaller side's sample (the max was taken over the 4 other
    # days, 24 rows), so the rank bound 2 / (n_rng + 1) stays conservative
    assert n == 24.0, n


def test_day_below_the_stated_maximum_still_counts_for_n_rng():
    """A day whose maximum was a violating row but lies wholly below the
    range's maximum still belongs to the sample the maximum was taken over:
    n_rng counts its clean rows (only the excluded row leaves the count)."""
    from app.engines.behavior.lib import pbounds as PB
    from app.engines.behavior.lib import pnode as PN
    num = PN.NumSummary(log=False)
    day0 = 739500
    excl = {day0 + 1: [1350.0]}
    for k in range(3):
        vals = [1200.0, 1250.0, 1300.0, 1350.0] if k == 1 else [1100.0, 1500.0, 1900.0, 1400.0]
        for j, v in enumerate(vals):
            num.update(v, T0 + k * DAY + j, 1.0, 1.0, day=day0 + k)
    lo, hi, n, dropped = PB.clean_range(num, day0 + 2, excl, day_of=lambda ts: day0)
    assert (lo, hi) == (1100.0, 1900.0)
    assert n == 11.0, n                                 # 12 rows minus the violating one
