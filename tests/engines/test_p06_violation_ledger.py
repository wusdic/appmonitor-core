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


def _run(flag: bool):
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
        for i, v in enumerate(rows):
            bb.add(T - 3600 + i * 7.0, f"192.168.1.{20 + i}", {"http.route": "POST /login",
                                                              "body.len": v}, 1.0)
        b = bb.build(T - 43200, T)
        store.add_batch("oa", EV.EVT_BATCH, T, b)
        rr = np.arange(b.n, dtype=np.int32)
        vt = np.zeros(b.n)
        if flag and d == 10:
            vt[-1] = 4.0                                  # P03: content p <= 1e-3
        store.add_batch("oa", EV.PAT_ASSIGN, T, b.aligned(
            {"leaf": EV.Col(rr, np.full(b.n, float(nid))), "vtype": EV.Col(rr, vt),
             "damp": EV.Col(rr, np.ones(b.n))}, {"kind": 0, "tree_key": "oa"}))
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
