"""B13 BudgetEngine: spec unit test (c), the common-mode peer guard.

Spec test mapping:
  (c) test_c_class_wide_surge_is_guarded_but_individual_surge_alarms
  (d), actors and young entities: test_b13_budget_exfil.py
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np

from helpers import make_store, put_model, run_engine

from test_b13_budget import DAY, S, T_MID, Feed, acc, budget_row, events

from app.engines.behavior.budget import MODEL, BudgetEngine
from app.engines.behavior.lib import m_template as MT
from app.engines.behavior.lib.classkeys import ORG

DT = 3600.0
INTERNAL = "erp.corp"
EXT = "files.example.net"
EXT2 = "drop.example.org"


def put_class(st, members: List[str], rid: str = "r1") -> None:
    put_model(st, ORG[0], ORG[1], "model.class", {
        "assign": {f"{S}|{e}": {"role": rid, "sub": None, "prob": 1.0, "static": [],
                                "pool": None, "super": "human"} for e in members},
        "roles": {rid: {"name": "clerks", "members": [f"{S}|{e}" for e in members],
                        "medoid": f"{S}|{members[0]}", "lineage": [], "version": 1}},
        "subs": {}, "statics": {}, "pools": {}, "version": 1})


def put_dest_names(st, names: List[str]) -> Dict[str, int]:
    """model.template@(s, __system__) with R2's destination names (so
    m_template.dest_name resolves the ids B13 checks for 'external')."""
    m = MT.new_model()
    ids = {}
    for n in names:
        did = MT.dest_id_of(n)
        m["prev"]["dests"][did] = [n, 1.0, T_MID, {}]
        ids[n] = did
    put_model(st, S, "__system__", MT.MODEL, m)
    return ids


def stream(now: float, total_up: float, did: int, n: int = 6):
    return [(now - DT + (i + 0.5) * DT / n, total_up / n, did) for i in range(n)]


# ===================================================================== (c)
def test_c_class_wide_surge_is_guarded_but_individual_surge_alarms():
    st, eng = make_store(), BudgetEngine()
    feed = Feed(st)
    ents = ["10.1.0.1", "10.1.0.2", "10.1.0.3", "10.1.0.4"]
    put_class(st, ents)
    rng = np.random.default_rng(11)
    m = 2e6                                                   # ~48 MB a day each
    alarms: Dict[int, Dict[str, int]] = {20: {}, 21: {}, 22: {}}
    for d in range(23):
        for k in range(24):
            now = T_MID + d * DAY + (k + 1) * DT
            for i, e in enumerate(ents):
                f = 2.5 if d == 20 or (d == 22 and i == 0) else 1.0
                up = m * f * math.exp(0.3 * rng.normal())
                feed.tick(e, now, DT, up=up, down=4 * up, req=20.0, writes=2.0, active=True)
            run_engine(eng, st, now, training=d < 20, dt=DT)
            if d >= 20:
                for e in ents:
                    if acc(st, now, e).get("budget_vol"):
                        alarms[d][e] = alarms[d].get(e, 0) + 1
            if d == 20 and k == 23:
                row = budget_row(st, now, ents[1])
                assert row["bytes_up.day"][0] > 2.0 * row["bytes_up.day"][1] / 1.5
    assert all(st.get_model(S, e, MODEL)["fit"]["src"][0] == 1 for e in ents)
    assert alarms[20] == {}, alarms[20]                       # (c): the peer guard holds
    assert alarms[21] == {}, alarms[21]
    assert set(alarms[22]) == {ents[0]}, alarms[22]           # the same surge alone alarms
    ev = [x for x in events(st) if x.extra["axis"] == "volume"]
    assert ev and {x.entity for x in ev} == {ents[0]}
    assert all(x.extra["peer_ratio"] is not None and x.extra["peer_ratio"] > 1.5 for x in ev)
