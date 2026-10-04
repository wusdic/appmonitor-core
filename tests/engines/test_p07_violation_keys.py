"""P07 learning hygiene: rows P03 judged violations do not teach the key set
(docs/lib3/progressive.md §6.9.3, §6.11; content_bounds._ledger 'led_set',
pgrammar.clean_presence).

Pack O seeds 0 and 2 (round 4 run 4, day 21): A10's ~150 comment posts
without viewstate (P03: missing_key, grammar, above_range at p <= 1e-3) were
learned by P04 at full weight on the last day, and portal POST /comment
stated viewstate optional (presence 0.77) against the truth's required
{news, text, viewstate}."""
from __future__ import annotations

import numpy as np

from pcontent_oracle import OracleLearner

from app.core.engine import Context
from app.core.store import MetricStore
from app.engines.behavior.content_bounds import ContentBoundsEngine
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.payload_grammar import PayloadGrammarEngine

CFG = {"progressive": {"enabled": True}, "grain_mode": "tick", "strict": True}
DAY = 86400.0
T0 = 1_788_220_800.0
GOOD = frozenset({"news", "text", "viewstate"})
BAD = frozenset({"news", "text"})


def _run(flag: bool, n_bad: int = 40, w_bad: float = 1.0, damp_bad: float = 1.0):
    store = MetricStore()
    orc = OracleLearner(store, "portal", T0, ["POST /comment"], config=CFG)
    p06, p07 = ContentBoundsEngine(), PayloadGrammarEngine()
    r = np.random.default_rng(0)
    nid = orc.node_for("POST /comment")
    for d in range(14):
        T = T0 + d * DAY + 12 * 3600
        bb = EV.BatchBuilder("portal")
        rows = [(GOOD, float(r.uniform(1500, 3000)), False) for _ in range(12)]
        if d == 13:
            rows += [(BAD, float(r.uniform(6000, 9000)), True) for _ in range(n_bad)]
        for i, (ks, v, bad) in enumerate(rows):
            bb.add(T - 3600 + i * 7.0, f"10.60.0.{10 + i}", {"http.route": "POST /comment",
                                                            "body.keys": ks, "body.len": v},
                   w_bad if bad else 1.0)
        b = bb.build(T - 43200, T)
        store.add_batch("portal", EV.EVT_BATCH, T, b)
        rr = np.arange(b.n, dtype=np.int32)
        vt = np.asarray([4.0 if (flag and bad) else 0.0 for _ks, _v, bad in rows])
        dm = np.asarray([damp_bad if bad else 1.0 for _ks, _v, bad in rows])
        cols = {"leaf": EV.Col(rr, np.full(b.n, float(nid))), "vtype": EV.Col(rr, vt),
                "damp": EV.Col(rr, dm)}
        store.add_batch("portal", EV.PAT_ASSIGN, T, b.aligned(cols, {"kind": 0, "tree_key": "portal"}))
        orc.learn(b, damp=dm)                             # P04: mass w/pi x damp
        ctx = Context(store=store, now=T + 3600, window_s=60, config=dict(CFG))
        p06.safe_run(ctx)
        p07.safe_run(ctx)
    return MP.get_model(store, "portal", MP.PGRAMMAR)["nodes"][0][nid]["attrs"]["body.keys"]


def test_violating_rows_do_not_make_a_required_key_optional():
    rec = _run(True)
    assert "viewstate" in rec["required"], rec
    raw = _run(False)                                     # not judged violations: they are data
    assert "viewstate" in raw["optional"], raw
    # violations most of the decayed presence mass (pack O seed 0, day 21)
    assert "viewstate" in _run(True, n_bad=300)["required"]
    # P00 sampled the burst (w/pi = 10) and P03 damped it to 0.1: P04 learned
    # each row at mass 1 (pack O seed 0: A10's posts, damp 0.1, 1/3 of the mass)
    assert "viewstate" in _run(True, n_bad=60, w_bad=10.0, damp_bad=0.1)["required"]


def test_clean_presence_without_ledger_is_the_presence():
    from app.engines.behavior.lib import pgrammar as PG
    from app.engines.behavior.lib import pnode as PN
    ss = PN.SetSummary()
    for i in range(30):
        ss.update(GOOD if i % 3 else BAD, T0 + i * 60.0, 1.0, 1.0)
    t = T0 + 31 * 60.0
    assert PG.clean_presence(ss, t, None) == ss.presence(t)
    # the ledger names the ten rows without viewstate: the rest always has it
    viol = [(T0 + i * 60.0, 1.0, tuple(sorted(BAD))) for i in range(30) if i % 3 == 0]
    p = PG.clean_presence(ss, t, viol)
    assert abs(p["viewstate"] - 1.0) < 1e-6 and abs(p["news"] - 1.0) < 1e-6
