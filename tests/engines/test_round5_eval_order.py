"""Evaluator round 5: order-only hash-seed dependences found by the determinism
check (pack O seed 0 under PYTHONHASHSEED 0 and 1: identical scores, but the
incident events of one tick came out in a different order, and P04's
`hold_facets` dicts had a different key order). Both iterated a set of str.
Each test fails on the round-5 code before the fix (set order of >= 6 strings
equals sorted order with probability <= 1/720)."""
from __future__ import annotations

from app.engines.behavior import incident as B27
from app.engines.behavior.lib.pnode import HoldRecord


def test_b27_releases_queued_notifications_in_sorted_system_order():
    eng = B27.IncidentEngine()
    st = B27._StoreState()
    systems = ["portal", "oa", "finance", "mail", "crm", "code", "hr", "erp"]
    tok = {s: [("inc-" + s, "open")] for s in systems[:4]}
    st.pending = {s: {"q-" + s: ["open", 0.0]} for s in systems[3:]}
    seen = []
    eng._flush = lambda store, st_, s, new, now, dt, ce, cs: seen.append(s) or 0
    assert eng._flush_systems(None, st, tok, 100.0, 60.0, 1.0, 1.0) == 0
    assert seen == sorted(systems)


def test_hold_record_facets_are_kept_in_sorted_order():
    h = HoldRecord()
    keys = ["who", "when", "bind:user", "net.bytes#q90", "body.kv#set", "tls.ver#cat"]
    t = 1_000.0
    for k in keys:
        h.add(k, True, 0.9, t)
    h._close()
    assert list(h.TF) == sorted(h.TF)
    assert set(h.TF) == {"who", "when", "bind", "net", "content", "~net"}
