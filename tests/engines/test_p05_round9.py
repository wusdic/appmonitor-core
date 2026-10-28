"""P05 round 9 (docs/lib3/progressive.md §16.16.11 item 6, §16.16.12 item 5):
a role holds while no new row carries the attribute (a weekend without
logins is no evidence against the login's fields); the payload of a binding
P08 states keeps the target role; an attribute takes a node's target slot
only when the node predicts it (O-red: independent `meta.z*` filled the
login node's target list)."""
from __future__ import annotations

import numpy as np

from app.engines.behavior.lib import pselect as SEL


def _st(**kw):
    st = {"U_t": 0.0004, "U_tc": 0.0, "U_s_max": 0.09, "CR": 0.15, "CR_p": 0.0, "H": 0.17, "H_p": 5.5,
          "cov": 0.018, "cov_rows": 0.24, "distinct0": 0.1, "CR0": 0.1, "S": 0.996, "n_p": 165, "n": 684,
          "card": {0: 45}, "level": 2, "best_levels": [2], "kind": "categorical"}
    st.update(kw)
    return st


def _roles(prev, stats, seen, bound=(), key_seen=None):
    out = SEL.assign_roles(stats, prev, {}, {}, 0.0, lambda a: (0,), seen=seen, bound=bound,
                           key_seen=key_seen)
    return out, dict(out, roles=out["roles"], low=out["low"], ustat=out["ustat"])


def test_a_role_holds_while_no_row_carries_the_attribute():
    """O-red, Saturday: no login since Friday evening, 87 rows of weekend
    traffic entered the probe and body.kv.username's split utility fell from
    0.09 to 0.008 bits; three hourly runs dropped it and P08's login bindings
    lost their payload (PG5 D2). Without a new row carrying the attribute its
    role, hysteresis and listed statistics hold; once rows carry it again a
    low utility demotes it as before."""
    a = "body.kv.username"
    out, prev = _roles({"roles": {a: "split"}}, {a: _st()}, {a: 100.0}, key_seen=100.0)
    assert out["roles"][a] == "split"
    for h in range(5):                                   # the weekend: other traffic, no login
        out, prev = _roles(prev, {a: _st(U_s_max=0.008, best_levels=[])}, {a: 100.0}, key_seen=200.0 + h)
        assert out["roles"][a] == "split" and out["low"].get(a, 0) == 0
        assert abs(out["ustat"][a]["U_s"] - 0.09) < 1e-12
    for t in (300.0, 400.0, 500.0):                      # new rows, still low: demoted after DEMOTE_RUNS
        out, prev = _roles(prev, {a: _st(U_s_max=0.008, best_levels=[])}, {a: t}, key_seen=t)
    assert out["roles"][a] == "dropped"
    # a key with no new row at all (a night without traffic) is judged as before
    out, prev = _roles({"roles": {a: "split"}}, {a: _st()}, {a: 100.0}, key_seen=100.0)
    for _ in range(3):
        out, prev = _roles(prev, {a: _st(U_s_max=0.008, best_levels=[])}, {a: 100.0}, key_seen=100.0)
    assert out["roles"][a] == "dropped"


def test_the_payload_of_a_stated_binding_keeps_the_target_role():
    a = "body.kv.username"
    pbind = {"nodes": {0: {14: {"status": "fitted", "pairs": {
        "net.src->body.kv.username": {"x": "net.src", "y": a, "table": {"10.77.2.14": {"top": "chen", "bound": True}}},
        "body.kv.username->net.src@1": {"x": a, "y": "net.src@1", "table": {"chen": {"bound": True}}},
        "sess.key->meta.z10": {"x": "sess.key", "y": "meta.z10", "table": {"k": {"bound": False}}}}}}}}
    assert SEL.bound_payloads(pbind) == [a]
    prev = {"roles": {a: "split"}, "low": {a: SEL.DEMOTE_RUNS - 1}}
    out = SEL.assign_roles({a: _st(U_s_max=0.008, best_levels=[])}, prev, {}, {}, 0.0, lambda x: (0,),
                           seen={a: 500.0}, bound=SEL.bound_payloads(pbind))
    assert out["roles"][a] == "target"


def test_independent_noise_does_not_take_a_node_target_slot():
    """O-red: P05 lists the independent `meta.z*` attributes among the node-
    local candidates and in the system target list (their roles came from
    chance gains), and ranked by entropy x coverage they took five to seven of
    the login node's nine target slots; the username lost its slot. The
    attributes a node predicts - their presence or their values differ on the
    node's rows from the others' - rank first: the login form's username
    keeps its slot, a node-local candidate that is the same lottery at every
    node gets none, and a system target the node does not predict fills only
    a slot left free (pack O's mail: every node has the same upload sizes,
    and the band is still a constraint of each node's events)."""
    from app.engines.behavior.lib import phier as PH
    from app.engines.behavior.lib import ptree as PT
    probe = SEL.StratifiedProbe(512)
    rng = np.random.default_rng(0)
    for i in range(400):
        login = i % 8 == 0
        row = {"http.route": "POST h /login" if login else "GET h /docs",
               "net.bytes_up": f"b{int(rng.integers(0, 6))}",
               "meta.z01": f"v{int(rng.integers(0, 10))}", "meta.z02": f"z{int(rng.integers(0, 8))}"}
        if login:
            row["body.kv.username"] = f"u{int(rng.integers(0, 12))}"
        probe.offer("all", row, 1.0, float(i), float(rng.random()))
    tr = PT.Tree(0, 0.0)
    tr.split(tr.root, "http.route", 0, [["POST h /login"]], 0.0)
    login_nid = tr.nodes[tr.root].split.children[0]
    hier = PH.Hierarchies(registry={"meta.z01": {"type": "categorical"}, "net.bytes_up": {"type": "categorical"}})
    # noise as node-local candidates: no slot at all
    got = SEL.node_targets_from_probe(tr, probe, 400.0, hier, ["net.bytes_up"], n_min=8,
                                      local_pool=["meta.z01", "meta.z02", "body.kv.username"])
    assert "body.kv.username" in got[login_nid]
    assert "meta.z01" not in got[login_nid] and "meta.z02" not in got[login_nid], got[login_nid]
    assert "net.bytes_up" in got[login_nid]          # a free slot: the system target keeps it
    # noise in the system list: the username ranks first and keeps its slot
    # when the slots are scarce
    for m_t in (1, 2):
        got = SEL.node_targets_from_probe(tr, probe, 400.0, hier, ["meta.z01", "meta.z02", "net.bytes_up"],
                                          m_t=m_t, n_min=8, local_pool=["body.kv.username"])
        assert got[login_nid][0] == "body.kv.username", got[login_nid]
