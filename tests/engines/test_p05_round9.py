"""P05 round 9 (docs/lib3/progressive.md §16.16.11 item 6, §16.16.12 item 5):
a day on which an attribute never occurs does not count towards demoting its
role (a weekend without logins is no evidence against the login's fields);
the payload of a binding P08 states keeps the target role; an attribute no
node predicts takes only the target slots the others leave (O-red:
independent `meta.z*` filled the login node's target list)."""
from __future__ import annotations

import numpy as np

from app.engines.behavior.lib import pselect as SEL


def _st(**kw):
    st = {"U_t": 0.0004, "U_tc": 0.0, "U_s_max": 0.09, "CR": 0.15, "CR_p": 0.0, "H": 0.17, "H_p": 5.5,
          "cov": 0.018, "cov_rows": 0.24, "distinct0": 0.1, "CR0": 0.1, "S": 0.996, "n_p": 165, "n": 684,
          "card": {0: 45}, "level": 2, "best_levels": [2], "kind": "categorical"}
    st.update(kw)
    return st


def _roles(prev, stats, present, bound=()):
    out = SEL.assign_roles(stats, prev, {}, {}, 0.0, lambda a: (0,), present=present, bound=bound)
    return out, dict(out, roles=out["roles"], low=out["low"], ustat=out["ustat"])


def test_a_day_without_the_attribute_does_not_demote_its_role():
    """O-red, Saturday: no login since Friday evening, 87 rows of weekend
    traffic entered the probe and body.kv.username's split utility fell from
    0.09 to 0.008 bits; three hourly runs dropped it and P08's login bindings
    lost their payload (PG5 D2). On a day the attribute never occurs (registry
    day_pres = 0 on the key's latest day of traffic) a low run is no evidence
    against its role: the hysteresis counter holds. On a day it occurs, a low
    utility demotes it after DEMOTE_RUNS runs as before."""
    a = "body.kv.username"
    out, prev = _roles({"roles": {a: "split"}}, {a: _st()}, {a: True})
    assert out["roles"][a] == "split"
    out, prev = _roles(prev, {a: _st(U_s_max=0.008, best_levels=[])}, {a: True})   # Friday night
    assert out["roles"][a] == "split" and out["low"][a] == 1
    for _ in range(3 * SEL.DEMOTE_RUNS):                 # the weekend: other traffic, no login
        out, prev = _roles(prev, {a: _st(U_s_max=0.008, best_levels=[])}, {a: False})
        assert out["roles"][a] == "split" and out["low"][a] == 1
    # the listed statistics are the current ones (only the role holds)
    assert abs(out["ustat"][a]["U_s"] - 0.008) < 1e-12
    for _ in range(SEL.DEMOTE_RUNS - 1):                 # logins again, still low: demoted
        out, prev = _roles(prev, {a: _st(U_s_max=0.008, best_levels=[])}, {a: True})
    assert out["roles"][a] == "dropped"
    # without the presence map (callers before round 9) runs count as before
    out, prev = _roles({"roles": {a: "split"}}, {a: _st()}, None)
    for _ in range(SEL.DEMOTE_RUNS):
        out, prev = _roles(prev, {a: _st(U_s_max=0.008, best_levels=[])}, None)
    assert out["roles"][a] == "dropped"
    # a promotion is not held back on such a day
    out, prev = _roles({"roles": {a: "dropped"}}, {a: _st()}, {a: False})
    assert out["roles"][a] == "split"


def test_the_payload_of_a_stated_binding_keeps_the_target_role():
    a = "body.kv.username"
    pbind = {"nodes": {0: {14: {"status": "fitted", "pairs": {
        "net.src->body.kv.username": {"x": "net.src", "y": a, "table": {"10.77.2.14": {"top": "chen", "bound": True}}},
        "body.kv.username->net.src@1": {"x": a, "y": "net.src@1", "table": {"chen": {"bound": True}}},
        "sess.key->meta.z10": {"x": "sess.key", "y": "meta.z10", "table": {"k": {"bound": False}}}}}}}}
    assert SEL.bound_payloads(pbind) == [a]
    prev = {"roles": {a: "split"}, "low": {a: SEL.DEMOTE_RUNS - 1}}
    out = SEL.assign_roles({a: _st(U_s_max=0.008, best_levels=[])}, prev, {}, {}, 0.0, lambda x: (0,),
                           present={a: True}, bound=SEL.bound_payloads(pbind))
    assert out["roles"][a] == "target"


def test_independent_noise_does_not_take_the_slot_of_a_node_field():
    """O-red: P05 lists the independent `meta.z*` attributes among the node-
    local candidates and in the system target list (their split roles came
    from chance gains), and ranked by entropy x coverage they took five to
    seven of the GA login node's nine target slots; the username lost its
    slot, P08 never refitted the node's bindings and D2's rename was never
    stated. An action's field - an attribute whose presence the node
    predicts - is not displaced by an attribute that is not one: it replaces
    the lowest-ranked of those, and the list otherwise keeps its ranking and
    order (a re-ordered list makes P04 retarget the node; an attribute that is
    the same lottery at every node still fills a free slot)."""
    from app.engines.behavior.lib import phier as PH
    from app.engines.behavior.lib import ptree as PT
    probe = SEL.StratifiedProbe(512)
    rng = np.random.default_rng(0)
    for i in range(400):
        login = i % 8 == 0
        row = {"http.route": "POST h /login" if login else "GET h /docs",
               "net.bytes_up": f"b{int(rng.integers(0, 6))}",
               "meta.z01": f"v{int(rng.integers(0, 16))}", "meta.z02": f"z{int(rng.integers(0, 12))}"}
        if login:
            row["body.kv.username"] = f"u{int(rng.integers(0, 5))}"
        probe.offer("all", row, 1.0, float(i), float(rng.random()))
    tr = PT.Tree(0, 0.0)
    tr.split(tr.root, "http.route", 0, [["POST h /login"]], 0.0)
    login_nid = tr.nodes[tr.root].split.children[0]
    hier = PH.Hierarchies(registry={"meta.z01": {"type": "categorical"}, "net.bytes_up": {"type": "categorical"}})
    # noise as node-local candidates: the username keeps the scarce slot; a
    # free slot is still filled (only P05's role says what is noise)
    got = SEL.node_targets_from_probe(tr, probe, 400.0, hier, ["net.bytes_up"], m_t=1, n_min=8,
                                      local_pool=["meta.z01", "meta.z02", "body.kv.username"])
    assert got[login_nid] == ["body.kv.username"], got[login_nid]
    got = SEL.node_targets_from_probe(tr, probe, 400.0, hier, ["net.bytes_up"], n_min=8,
                                      local_pool=["meta.z01", "meta.z02", "body.kv.username"])
    assert "net.bytes_up" in got[login_nid]          # a free slot: the size band stays tracked
    # noise in the system list: the username keeps its slot when the slots
    # are scarce; the kept list keeps its entropy x coverage order (a
    # re-ordered list makes P04 retarget the node)
    got = SEL.node_targets_from_probe(tr, probe, 400.0, hier, ["meta.z01", "meta.z02", "net.bytes_up"],
                                      m_t=1, n_min=8, local_pool=["body.kv.username"])
    assert got[login_nid] == ["body.kv.username"], got[login_nid]
    got = SEL.node_targets_from_probe(tr, probe, 400.0, hier, ["meta.z01", "meta.z02", "net.bytes_up"],
                                      m_t=2, n_min=8, local_pool=["body.kv.username"])
    assert got[login_nid] == ["meta.z01", "body.kv.username"], got[login_nid]



def test_only_the_bindings_p08_states_keep_their_payloads_target_role():
    """(round 9, evaluator; views owner's proposal) P08 publishes the bound
    pairs the views state (model.pbind['bound_stated'], FD.binding_stated).
    A pair that has a bound entry but is never stated - an action constant
    such as body.fmt = form, the same value for every bound source - must not
    keep its payload's target role for good (a zero-gain `net.src -> body.fmt`
    pair kept P08 on until P12 switched it off for finance, pack O seed 2).
    Without the list (models before round 9) every bound pair counts."""
    a = "body.kv.username"
    pbind = {"nodes": {0: {14: {"status": "fitted", "pairs": {
        "net.src->body.kv.username": {"x": "net.src", "y": a, "table": {"10.77.2.14": {"top": "chen", "bound": True}}},
        "net.src@3->body.fmt": {"x": "net.src@3", "y": "body.fmt", "table": {"10.77.2": {"top": "form", "bound": True}}}}}}}}
    assert SEL.bound_payloads(pbind) == ["body.fmt", a]
    pbind["bound_stated"] = [[0, 14, "net.src->body.kv.username"]]
    assert SEL.bound_payloads(pbind) == [a]
    pbind["bound_stated"] = []
    assert SEL.bound_payloads(pbind) == []
