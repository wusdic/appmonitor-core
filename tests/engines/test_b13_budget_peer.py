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
from app.engines.behavior.lib import emit
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


# ============================================ round 4: untrusted own pattern
def _run_backup(use_fill: bool = True):
    """Three mature peers (22 d of ~2 MB / h) and a nightly backup host that
    joined 5 days later: 300 MB at 02:00 every night, committed with trust 0
    (a HIGH lib-4 match every night), ordinary hours like the peers."""
    from app.engines.behavior import budget as B
    from helpers import set_trust
    st, eng = make_store(), BudgetEngine()
    orig = B.BudgetEngine._observed_fill
    if not use_fill:
        B.BudgetEngine._observed_fill = staticmethod(lambda *a, **k: None)
    try:
        feed = Feed(st)
        peers = ["10.1.0.1", "10.1.0.2", "10.1.0.3"]
        bk = "10.1.0.9"
        put_class(st, peers + [bk])
        rng = np.random.default_rng(5)
        out = []
        for d in range(23):
            for k in range(24):
                now = T_MID + d * DAY + (k + 1) * DT
                for e in peers:
                    up = 2e6 * math.exp(0.3 * rng.normal())
                    feed.tick(e, now, DT, up=up, down=4 * up, req=20.0, writes=2.0, active=True)
                if d >= 5:
                    backup = k == 2
                    up = (3e8 if backup else 2e6) * math.exp(0.1 * rng.normal())
                    feed.tick(bk, now, DT, up=up, down=4e6, req=20.0, writes=2.0, active=True)
                    set_trust(st, S, bk, [now], 0.0 if backup else 1.0)
                run_engine(eng, st, now, training=d < 21, dt=DT)
                if d >= 21 and k in (2, 3, 4, 12):
                    row = budget_row(st, now, bk)
                    out.append((acc(st, now, bk).get("budget_vol", 0),
                                row["bytes_up.8h"][2], row["bytes_up.7d"][2]))
        return st.get_model(S, bk, MODEL), out, emit.read_dict(st, S, bk, emit.DEGRADED, now)
    finally:
        B.BudgetEngine._observed_fill = orig


def test_untrusted_recurring_pattern_replaces_the_peers_level():
    """Integration §10.7 item 1 (pack B 10.20.9.5): the backup host's night
    rows are never trusted, so the young host's own ring has no 02:00 hour
    and no valid 7-d window; the peer fallback used the PEERS' level there
    (6.8 GB of 7-d upload against a 20.8 MB 'usual'). With the recurring
    observed pattern the host keeps its own level: no alarm on its usual
    nights, and the 8-h / 7-d tail p are unremarkable."""
    model, out, dg = _run_backup(True)
    fit = model["fit"]
    assert int(fit["src"][0]) == 2                          # the young-entity peer path
    assert dg.get("budget_vol") == "fallback:peer"          # contract M: weaker reference
    alarms = [a for a, _, _ in out]
    p8 = np.array([p for _, p, _ in out], dtype=float)
    p7 = np.array([p for _, _, p in out], dtype=float)
    assert sum(alarms) == 0
    assert np.nanmin(p8) > 1e-3 and np.nanmin(p7) > 1e-3
    # the failure mode: the peers' level at the host's untrusted phases
    _, bad, _ = _run_backup(False)
    p8b = np.array([p for _, p, _ in bad], dtype=float)
    p7b = np.array([p for _, _, p in bad], dtype=float)
    assert min(np.nanmin(p8b), np.nanmin(p7b)) < 1e-6
