"""P06 / P07 content-target requests (lib/pbounds.request_targets) never spend
a node's request slots on the derived time / session context.

Pack O (round 2, seed 0, portal): P05 ranked ctx.sess_age_s, ctx.sess_pos and
ctx.think_s (split candidates) before the body size; the POST /comment node's
three P06 request slots held them for good, the body size was never a target
there and the statement stated a band from one observation (PG1 portal comment
content missed on seeds 0 and 1). The derived context is P09's and P10's
(windows, workflow delay bands), not content."""
from __future__ import annotations

import numpy as np
from pcontent_oracle import OracleLearner
from test_p06_p08_content import CFG, DAY, T0, _day

from app.core.store import MetricStore
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pbounds as PB


def test_context_attributes_do_not_take_the_content_request_slots():
    store = MetricStore()
    orc = OracleLearner(store, "portal", T0, ["POST /comment", "GET /"], config=CFG,
                        target_filter=lambda a: a == "net.bytes_up")
    r = np.random.default_rng(0)
    for d in range(4):
        _day(store, orc, [], "portal", {
            "POST /comment": [(f"10.60.0.{k}", {"body.len": float(r.uniform(1500, 3000)),
                                                "net.bytes_up": 1800.0,
                                                "ctx.sess_age_s": float(r.uniform(10, 900)),
                                                "ctx.sess_pos": float(r.integers(1, 9)),
                                                "ctx.think_s": float(r.uniform(5, 300))})
                              for k in range(12)],
            "GET /": [("10.60.0.1", {"net.bytes_up": 300.0})]}, d)
    store.put_model("portal", "__system__", MP.ATTRSEL, {
        "roles": {"body.len": "split", "ctx.sess_age_s": "split", "ctx.sess_pos": "split",
                  "ctx.think_s": "split", "net.bytes_up": "target"},
        "split_cands": {0: [("ctx.sess_age_s", 1), ("ctx.sess_pos", 1), ("ctx.think_s", 3),
                            ("body.len", 1)]},
        "targets_sys": {0: ["net.bytes_up"]}})
    reg = MP.get_registry(store, "portal")
    for a in ("body.len", "ctx.sess_age_s", "ctx.sess_pos", "ctx.think_s"):
        reg.get(a).type = "numeric"
    num = PB.request_targets(store, "portal", MP.get_ptree(store, "portal"), T0 + 5 * DAY,
                             ("numeric",), {})
    got = num[0][orc.node_for("POST /comment")]
    assert got == ["body.len"], got


def test_one_size_measured_twice_takes_one_slot_and_the_payload_comes_first():
    """Pack O seed 0, portal comment node: the P06 slots held net.bytes_down and
    net.pkts_down (the same response size) ranked before the body size."""
    store = MetricStore()
    orc = OracleLearner(store, "portal", T0, ["POST /comment", "GET /"], config=CFG,
                        target_filter=lambda a: a == "net.bytes_up")
    r = np.random.default_rng(1)
    for d in range(4):
        _day(store, orc, [], "portal", {
            "POST /comment": [(f"10.60.0.{k}", {"body.len": float(r.uniform(1500, 3000)),
                                                "net.bytes_up": 1800.0,
                                                "net.bytes_down": float(r.uniform(500, 900)),
                                                "net.pkts_down": float(r.integers(2, 4)),
                                                "net.dur_ms": float(r.uniform(10, 90))})
                              for k in range(12)],
            "GET /": [("10.60.0.1", {"net.bytes_up": 300.0})]}, d)
    store.put_model("portal", "__system__", MP.ATTRSEL, {
        "roles": {a: "split" for a in ("body.len", "net.bytes_down", "net.pkts_down", "net.dur_ms")}
        | {"net.bytes_up": "target"},
        "split_cands": {0: [("net.bytes_down", 2), ("net.pkts_down", 2), ("net.dur_ms", 2),
                            ("body.len", 1)]},
        "targets_sys": {0: ["net.bytes_up"]}})
    reg = MP.get_registry(store, "portal")
    for a in ("body.len", "net.bytes_down", "net.pkts_down", "net.dur_ms"):
        reg.get(a).type = "numeric"
    num = PB.request_targets(store, "portal", MP.get_ptree(store, "portal"), T0 + 5 * DAY,
                             ("numeric",), {})
    got = num[0][orc.node_for("POST /comment")]
    assert got[0] == "body.len" and "net.bytes_down" in got and "net.pkts_down" not in got, got
