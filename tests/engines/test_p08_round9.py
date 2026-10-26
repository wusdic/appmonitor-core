"""P08 round 9: a stated binding persists through P05's role churn.

O-red seed 0 (94 attributes, P05's due regime, docs/lib3/progressive.md
16.16.11 item 6): on a Saturday without logins P05 gave body.kv.username the
role 'dropped'. P08 took its payload candidates from P05's roles only, so the
next screen dropped net.src -> body.kv.username, P04's daily pair GC dropped
the GA login node's sketch, P08 deleted the pair's value history, and since
the node's P05 target override no longer held the username the re-screened
pair was never assigned to it again: the five bindings stated on days 12-14
were gone from day 15, and D2's rename dora -> dora.k was never stated.
Fails on the round-8 code."""
from __future__ import annotations

from types import SimpleNamespace as NS

from pcontent_oracle import OracleLearner

from app.core.engine import Context
from app.core.store import MetricStore
from app.engines.behavior.binding import STATE, BindingEngine
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.models.schema import SYSTEM_ENTITY

CFG = {"progressive": {"enabled": True}, "grain_mode": "tick", "strict": True}
DAY = 86400.0
T0 = 1_788_220_800.0
PK = "net.src->body.kv.username"
GA = [("192.168.1.21", "jack"), ("192.168.1.23", "rose"), ("10.168.7.121", "mike")]


def _ctx(store, now):
    return Context(store=store, now=now, window_s=60, config=dict(CFG))


def _gc_like_p04(store, orc):
    """P04's daily _gc_pairs: a node keeps only the pair sketches P08 requests."""
    want = ((MP.get_model(store, "oa", MP.PWANT) or {}).get("pairs") or {}).get("by_kind") or {}
    req = want.get(EV.KIND_TXN, {})
    for nid, nd in orc.tree.nodes.items():
        keep = {tuple(xy) for xy in req.get(nid, ())}
        for k in [k for k in nd.pairs if k not in keep]:
            del nd.pairs[k]


def _day(store, orc, eng, rows, d):
    T = T0 + d * DAY + 12 * 3600
    bb = EV.BatchBuilder("oa")
    for i, (ip, u) in enumerate(rows):
        bb.add(T - 3600 + i * 7.0, ip, {"http.route": "POST /login", "body.kv.username": u,
                                         "body.kv.lang": "zh"}, 1.0)
    b = bb.build(T - 43200, T)
    store.add_batch("oa", EV.EVT_BATCH, T, b)
    orc.learn(b)
    eng.safe_run(_ctx(store, T + 3600))
    _gc_like_p04(store, orc)


def _world(d):
    rows = list(GA)
    if d >= 13:                                     # D2: mike renamed mike.w
        rows = [r if r[0] != "10.168.7.121" else ("10.168.7.121", "mike.w") for r in rows]
    return rows + [(f"192.168.3.{20 + k}", f"s{k:02d}x") for k in range(20)]


def test_a_bound_pair_survives_p05_role_churn_and_the_rename_is_stated():
    store = MetricStore()
    orc = OracleLearner(store, "oa", T0, ["POST /login"], config=CFG)
    eng = BindingEngine()
    nid = orc.node_for("POST /login")
    for d in range(10):
        _day(store, orc, eng, _world(d), d)
    tab = MP.get_model(store, "oa", MP.PBIND)["nodes"][0][nid]["pairs"][PK]["table"]
    assert tab["10.168.7.121"].get("bound") and tab["10.168.7.121"]["top"] == "mike"
    # day 10 on: P05 drops the username (a weekend without logins) and the
    # node's bounded target override no longer holds it
    sel = {"roles": {"body.kv.username": "dropped", "body.kv.lang": "target"},
           "targets_sys": {"0": ["body.kv.lang"]}}
    store.put_model("oa", SYSTEM_ENTITY, MP.ATTRSEL, sel)
    orc.target_filter = lambda a: a == "body.kv.lang"
    for nd in orc.tree.nodes.values():
        nd.targets.pop("body.kv.username", None)
    assert "body.kv.username" in BindingEngine._y_candidates(store, "oa", MP.get_registry(store, "oa"))
    for d in range(10, 21):
        _day(store, orc, eng, _world(d), d)
        st = store.get_model("oa", SYSTEM_ENTITY, STATE)
        assert PK in st["hist"], d
        assert any(pk == PK and bm for (_k, _n, pk), bm in st["bound"].items()), d
    rec = MP.get_model(store, "oa", MP.PBIND)["nodes"][0][nid]["pairs"][PK]
    e = rec["table"]["10.168.7.121"]
    assert e.get("bound") and e["top"] == "mike.w" and "mike" in e.get("superseded", []), e
    assert e["rebound"]["from"] == "mike"
    assert rec["table"]["192.168.1.21"]["top"] == "jack" and rec["table"]["192.168.1.21"].get("bound")


def test_bound_payloads_are_candidates_beyond_the_screening_width(monkeypatch):
    """The payload of a stated binding stays a candidate even when P05 names
    more than Y_MAX better-covered attributes."""
    from app.engines.behavior import binding as B
    names = [f"body.kv.a{i:02d}" for i in range(B.Y_MAX + 5)]
    sel = {"roles": {a: "split" for a in names}}
    monkeypatch.setattr(MP, "get_model", lambda store, key, kind: sel if kind == MP.ATTRSEL else None)
    recs = {a: NS(type="categorical", policy="clear", state="active") for a in names + ["body.kv.username"]}

    class Reg:
        def get(self, a):
            return recs.get(a)

        def names(self):
            return list(recs)

        def coverage(self, a):
            return 0.01 if a == "body.kv.username" else 1.0

    store = MetricStore()
    assert "body.kv.username" not in BindingEngine._y_candidates(store, "oa", Reg())
    store.put_model("oa", SYSTEM_ENTITY, STATE,
                    {"bound": {(0, 3, PK): {"10.0.0.1": "amy"}, (0, 4, "net.src->body.kv.x"): {}}})
    out = BindingEngine._y_candidates(store, "oa", Reg())
    assert "body.kv.username" in out and "body.kv.x" not in out
    assert len(out) == B.Y_MAX


def test_a_bound_pair_is_carried_through_a_screen_out():
    """A re-screen that does not find a bound pair (its stratum thin after a
    weekend, its payload out of the candidates) carries the pair's spec: its
    node keeps its sketch, the value history stays, the bindings stay stated."""
    from app.engines.behavior.lib import pfd as FD
    store = MetricStore()
    orc = OracleLearner(store, "oa", T0, ["POST /login"], config=CFG)
    eng = BindingEngine()
    nid = orc.node_for("POST /login")
    for d in range(10):
        _day(store, orc, eng, _world(d), d)
    st = store.get_model("oa", SYSTEM_ENTITY, STATE)
    assert any(pk == PK and bm for (_k, _n, pk), bm in st["bound"].items())
    st["probe"] = FD.ProbeReservoir()            # nothing left to screen on
    st["screened_t"] = -1e18
    now = T0 + 10 * DAY
    eng.fit_tree(_ctx(store, now), "oa", MP.get_ptree(store, "oa"), now)
    pm = MP.get_model(store, "oa", MP.PBIND)
    spec = [d for d in pm["screen"] if f"{d['x']}->{d['y']}" == PK]
    assert spec and spec[0].get("carried")
    want = MP.get_model(store, "oa", MP.PWANT)["pairs"]["by_kind"][EV.KIND_TXN]
    assert ["net.src", "body.kv.username"] in want[nid]
    assert PK in store.get_model("oa", SYSTEM_ENTITY, STATE)["hist"]
    tab = pm["nodes"][0][nid]["pairs"][PK]["table"]
    assert tab["192.168.1.21"].get("bound") and tab["192.168.1.21"]["top"] == "jack"
