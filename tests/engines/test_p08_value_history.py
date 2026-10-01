"""P08 learning hygiene: a minority value cannot join a source's binding
without sustained, clean evidence, and a rename still rebinds
(docs/lib3/progressive.md §6.12, §6.9.2; lib/pfd.ValueHistory / classify).

Measured on pack O (oa, seeds 0-1, day 21) before the change: anomaly A2
(192.168.1.21 logging in as rose, rose being 192.168.1.23's own username,
days 17-21) turned 192.168.1.21's binding into the set {jack, rose}
(non-adoption failed), and at the GA login node created by a late split the
D2 rename (10.168.7.121: mike -> mike.w) read as {mike, mike.w} because the
rebinding lived in per-node segment baselines of the parent node only."""
from __future__ import annotations

import random

from pcontent_oracle import OracleLearner

from app.core.engine import Context
from app.core.store import MetricStore
from app.engines.behavior.binding import STATE, BindingEngine
from app.engines.behavior.content_bounds import ContentBoundsEngine
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import pfd as FD
from app.engines.behavior.payload_grammar import PayloadGrammarEngine

CFG = {"progressive": {"enabled": True}, "grain_mode": "tick", "strict": True}
DAY = 86400.0
T0 = 1_788_220_800.0
GA = [("192.168.1.21", "jack"), ("192.168.1.23", "rose"), ("10.168.7.121", "mike")]


# ------------------------------------------------------------------ lib
def _hist(rows):
    h = FD.ValueHistory()
    for x, y, d in rows:
        h.observe(x, y, T0 + d * DAY + 9 * 3600, 700000 + d, True)
    return h


def test_classify_borrowed_value_pending_rename_superseded_shared_kept():
    # A2: jack daily from day 0, rose (bound elsewhere) on days 14-19
    h = _hist([("x", "jack", d) for d in range(20)] + [("x", "rose", d) for d in range(14, 20)])
    now = T0 + 20 * DAY
    assert FD.classify(h.get("x"), ["jack", "rose"], now, ["rose"]) == {"rose": "pending"}
    # the same persistence of a value bound nowhere confirms it (a second user)
    assert FD.classify(h.get("x"), ["jack", "rose"], now, []) == {}
    # ... unless the source had a governor episode since the value appeared
    assert FD.classify(h.get("x"), ["jack", "rose"], now, [], lambda t0: True) == {"rose": "pending"}
    # a young newcomer (2 days) is pending even when bound nowhere
    h2 = _hist([("x", "jack", d) for d in range(20)] + [("x", "amy", d) for d in (18, 19)])
    assert FD.classify(h2.get("x"), ["jack", "amy"], now, []) == {"amy": "pending"}
    # D2: mike for 12 days, then only mike.w (2 a day) -> mike superseded
    h3 = _hist([("x", "mike", d) for d in range(12)] +
               [("x", "mike.w", d) for d in range(12, 15) for _ in range(2)])
    assert FD.classify(h3.get("x"), ["mike", "mike.w"], T0 + 15 * DAY, []) == {"mike": "superseded"}
    # shared terminal: both values from the start -> neither excluded
    h4 = _hist([("x", u, d) for d in range(10) for u in ("amy", "bob")])
    assert FD.classify(h4.get("x"), ["amy", "bob"], T0 + 10 * DAY, ["amy"]) == {}


def test_damped_rows_do_not_confirm_a_newcomer():
    h = FD.ValueHistory()
    for d in range(20):
        h.observe("x", "jack", T0 + d * DAY, 700000 + d, True)
    for d in range(10, 20):
        h.observe("x", "eve", T0 + d * DAY + 60, 700000 + d, True, clean=False)
    assert FD.classify(h.get("x"), ["jack", "eve"], T0 + 20 * DAY, []) == {"eve": "pending"}


def test_history_bounded():
    h = FD.ValueHistory(cap_x=16, cap_y=3)
    for i in range(100):
        for k in range(10):
            h.observe(f"10.0.0.{i}", f"u{k}", T0 + i, 700000, True)
    assert len(h.x) == 16 and all(len(s["v"]) <= 3 for s in h.x.values())


# --------------------------------------------------------------- engine
def _ctx(store, now):
    return Context(store=store, now=now, window_s=60, config=dict(CFG))


def _day(store, orc, engs, rows, d):
    T = T0 + d * DAY + 12 * 3600
    bb = EV.BatchBuilder("oa")
    for i, (ip, u) in enumerate(rows):
        bb.add(T - 3600 + i * 7.0, ip, {"http.route": "POST /login", "body.kv.username": u}, 1.0)
    b = bb.build(T - 43200, T)
    store.add_batch("oa", EV.EVT_BATCH, T, b)
    orc.learn(b)
    for e in engs:
        e.safe_run(_ctx(store, T + 3600))


def _world(d, rnd):
    rows = list(GA)
    # a department of 20 so the dependency is judged on many sources
    rows += [(f"192.168.3.{20 + k}", f"s{k:02d}x") for k in range(20)]
    return rows


def test_borrowed_credential_is_not_adopted_and_rename_still_rebinds():
    store = MetricStore()
    orc = OracleLearner(store, "oa", T0, ["POST /login"], config=CFG)
    engs = [ContentBoundsEngine(), PayloadGrammarEngine(), BindingEngine()]
    rnd = random.Random(3)
    for d in range(21):
        rows = _world(d, rnd)
        if d >= 13:                                   # D2: mike renamed mike.w
            rows = [r if r[0] != "10.168.7.121" else ("10.168.7.121", "mike.w") for r in rows]
        if d >= 15:                                   # A2: .21 also logs in as rose, daily
            rows.append(("192.168.1.21", "rose"))
        _day(store, orc, engs, rows, d)
    nid = orc.node_for("POST /login")
    rec = MP.get_model(store, "oa", MP.PBIND)["nodes"][0][nid]["pairs"]["net.src->body.kv.username"]
    tab = rec["table"]
    e21 = tab["192.168.1.21"]
    assert e21.get("bound") and e21["top"] == "jack" and "set" not in e21, e21
    assert "rose" in e21.get("pending", []), e21
    assert tab["192.168.1.23"].get("bound") and tab["192.168.1.23"]["top"] == "rose"
    e121 = tab["10.168.7.121"]
    assert e121.get("bound") and e121["top"] == "mike.w" and "mike" in e121.get("superseded", [])
    assert e121["rebound"]["from"] == "mike"
    assert any(ev.kind == "binding_changed" and ev.extra.get("new") == "mike.w" for ev in store.events())
    st = MP.get_model(store, "oa", STATE)
    assert set(st["hist"]["net.src->body.kv.username"].x) <= set(st["track"]["net.src->body.kv.username"])


def test_rename_reaches_a_node_created_after_it():
    """A node whose pair sketch was filled later (a split child, M5 copies the
    parent's rows) still sees the rename: the history is per tree."""
    store = MetricStore()
    orc = OracleLearner(store, "oa", T0, ["POST /login", "POST /login2"], config=CFG)
    engs = [BindingEngine()]
    rnd = random.Random(4)
    for d in range(18):
        rows = _world(d, rnd)
        if d >= 12:
            rows = [r if r[0] != "10.168.7.121" else ("10.168.7.121", "mike.w") for r in rows]
            rows = rows + [r for r in rows if r[0] == "10.168.7.121"]
        _day(store, orc, engs, rows, d)
    nid = orc.node_for("POST /login")
    tree = MP.get_ptree(store, "oa").kinds[0]
    node = tree.nodes[nid]
    # a "child" node: a fresh node whose pair sketch holds the parent's rows
    other = tree.nodes[orc.node_for("POST /login2")]
    ps = node.pairs[("net.src", "body.kv.username")]
    cps = other.pair("net.src", "body.kv.username")
    for x, rows in ps.table(T0 + 18 * DAY).items():
        for y, e, g in rows:
            cps.update(x, y, T0 + 18 * DAY, max(g, 1e-9), e)
    BindingEngine().fit_tree(_ctx(store, T0 + 18 * DAY + 7200), "oa", MP.get_ptree(store, "oa"),
                             T0 + 18 * DAY + 7200)
    # the engine instance above is new: reuse the state it shares through the store
    rec = MP.get_model(store, "oa", MP.PBIND)["nodes"][0][other.id]["pairs"]["net.src->body.kv.username"]
    e = rec["table"]["10.168.7.121"]
    assert e.get("bound") and e["top"] == "mike.w", e


def test_rename_is_recognised_without_node_growth():
    """The rename qualifies on a row that adds too little evidence to make a
    busy node dirty (< 10 % / 20 units): the node is refitted because the
    value history of one of its sources changed."""
    store = MetricStore()
    orc = OracleLearner(store, "oa", T0, ["POST /login"], config=CFG)
    engs = [BindingEngine()]
    for d in range(12):
        _day(store, orc, engs, _world(d, None), d)
    for d in range(12, 19):                                  # only the renamed account logs in
        _day(store, orc, engs, [("10.168.7.121", "mike.w")], d)
    nid = orc.node_for("POST /login")
    e = MP.get_model(store, "oa", MP.PBIND)["nodes"][0][nid]["pairs"]["net.src->body.kv.username"][
        "table"]["10.168.7.121"]
    assert e["top"] == "mike.w" and "mike" in e.get("superseded", []), e


def test_rename_keeps_the_binding_while_the_new_segment_is_short():
    """§6.12 rebinding moves y*_x to y' with a fresh confidence segment: the
    successor is bound once the history qualified it (>= 5 clean events over
    >= 2 normal days), although its decayed evidence units may still be < 5
    (pack O seed 0, D2: mike.w had 4.6 units at day 21 and the 综合部 login
    statement lost 10.168.7.121's binding)."""
    from app.engines.behavior.lib import pnode as PN
    ps = PN.PairSketch()
    h = FD.ValueHistory()
    t = T0
    for d in range(10):
        for x, y in GA:
            ps.update(x, y, T0 + d * DAY, 1.0, 1.0)
            h.observe(x, y, T0 + d * DAY, 700000 + d, True)
    for d in range(10, 15):                                  # mike.w, a slightly damped evidence
        ps.update("10.168.7.121", "mike.w", T0 + d * DAY, 0.85, 0.85)
        h.observe("10.168.7.121", "mike.w", T0 + d * DAY, 700000 + d, True)
    t = T0 + 15 * DAY
    ex = {x: FD.classify(h.get(x), [y for y, _, _ in rows], t)
          for x, rows in ps.table(t).items()}
    assert ex["10.168.7.121"] == {"mike": "superseded"}
    rec = FD.fit_pair(ps, t, exclude={x: e for x, e in ex.items() if e})
    e = rec["table"]["10.168.7.121"]
    assert e["n"] < FD.N_BIND and e["top"] == "mike.w" and e.get("bound"), e
    assert rec["fd"]["holds"] and rec["fd"]["bound_share"] <= 1.0 + 1e-9, rec["fd"]
