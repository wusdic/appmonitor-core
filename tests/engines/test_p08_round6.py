"""P08, round 6 (content / binding owner): a binding its source's latest
clean values refute at the binding's own stated bound is suspended
('changing') until the change is adopted or the old value returns
(lib/pfd.refuted). Fails on the round-5 code."""
from __future__ import annotations

from pcontent_oracle import OracleLearner
from test_p08_value_history import CFG, DAY, T0, _day, _world

from app.core.store import MetricStore
from app.engines.behavior.binding import BindingEngine
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pfd as FD


def _h(seq, clean=True):
    h = FD.ValueHistory()
    for i, y in enumerate(seq):
        h.observe("x", y, T0 + i * DAY, 700000 + i, True, clean=clean)
    return h


def test_refuted_at_the_bindings_own_bound():
    h = _h(["mike"] * 10 + ["mike.w"])
    # LB 0.999: one clean event of another value is a 1e-3 event under the claim
    r = FD.refuted(h.get("x"), "mike", 0.999)
    assert r is not None and r["run"] == 1 and r["values"] == ["mike.w"]
    # a weaker claim (LB 0.9) needs two
    assert FD.refuted(h.get("x"), "mike", 0.9) is None
    h.observe("x", "mike.w", T0 + 11 * DAY, 700011, True)
    assert FD.refuted(h.get("x"), "mike", 0.9) is not None
    # the old value back: the run is broken
    h.observe("x", "mike", T0 + 12 * DAY, 700012, True)
    assert FD.refuted(h.get("x"), "mike", 0.999) is None


def test_a_borrowed_credential_or_a_damped_row_refutes_nothing():
    # A2: the source also logs in with another source's established value
    h = _h(["jack"] * 10 + ["rose", "rose"])
    assert FD.refuted(h.get("x"), "jack", 0.999, ["rose"]) is None
    # damped (not clean) rows are not evidence
    h2 = _h(["jack"] * 10)
    h2.observe("x", "eve", T0 + 11 * DAY, 700011, True, clean=False)
    assert FD.refuted(h2.get("x"), "jack", 0.999) is None


def test_rename_suspends_the_old_binding_until_it_is_adopted():
    """Pack O seed 0, D2 (day 13: 10.168.7.121's mike -> mike.w, one login a
    workday): the 综合部 login statements kept '10.168.7.121 -> username=mike'
    at LB 0.999 on days 15-18 (confidence 0.96-0.98, held 0 on every check)
    until mike.w had the REBIND_N clean events a rename needs."""
    store = MetricStore()
    orc = OracleLearner(store, "oa", T0, ["POST /login"], config=CFG)
    engs = [BindingEngine()]
    for d in range(16):
        rows = _world(d, None)
        if d >= 13:
            rows = [r if r[0] != "10.168.7.121" else ("10.168.7.121", "mike.w") for r in rows]
        _day(store, orc, engs, rows, d)
    nid = orc.node_for("POST /login")
    tab = MP.get_model(store, "oa", MP.PBIND)["nodes"][0][nid]["pairs"]["net.src->body.kv.username"]["table"]
    e = tab["10.168.7.121"]
    assert not e.get("bound") and e.get("changing", {}).get("values") == ["mike.w"], e
    assert tab["192.168.1.21"].get("bound") and tab["192.168.1.21"]["top"] == "jack"
    # adopted once the history qualifies the rename (>= 5 clean events, >= 2 normal days)
    for d in range(16, 19):
        rows = [r if r[0] != "10.168.7.121" else ("10.168.7.121", "mike.w") for r in _world(d, None)]
        _day(store, orc, engs, rows, d)
    tab = MP.get_model(store, "oa", MP.PBIND)["nodes"][0][nid]["pairs"]["net.src->body.kv.username"]["table"]
    e = tab["10.168.7.121"]
    assert e.get("bound") and e["top"] == "mike.w" and "changing" not in e, e
