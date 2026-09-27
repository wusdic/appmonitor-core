"""B13 BudgetEngine: spec unit tests (c) and (d), actor aggregation through
model.link and peer-pooled thresholds for a young entity.

Spec test mapping:
  (c) test_c_class_wide_surge_is_guarded_but_individual_surge_alarms
  (d) test_d_small_novel_upload_recorded_below_floor (+ its positive control
      and the actor / young-entity cases, which share the same trained store)
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pytest

from helpers import make_store, put_model, run_engine

from test_b13_budget import (DAY, S, T_MID, Feed, acc, budget_row, events, last_alarms)

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


# =================================================== (d) + actors + young
@pytest.fixture(scope="module")
def trained():
    """Three entities of one class, 20 training days at 3600 s: steady uploads
    to an internal destination only (their exfil history is empty)."""
    st, eng = make_store(), BudgetEngine()
    feed = Feed(st)
    ents = ["10.2.0.1", "10.2.0.2", "10.2.0.3"]
    put_class(st, ents)
    ids = put_dest_names(st, [INTERNAL, EXT, EXT2])
    rng = np.random.default_rng(2)
    for d in range(20):
        for k in range(24):
            now = T_MID + d * DAY + (k + 1) * DT
            for e in ents:
                up = 1e6 * math.exp(0.3 * rng.normal())
                feed.tick(e, now, DT, up=up, down=3 * up, stream=stream(now, up, ids[INTERNAL]))
            run_engine(eng, st, now, training=True, dt=DT)
    return st, eng, feed, ents, ids, rng


def live_day(st, eng, feed, ents, ids, rng, d, extra, hook=None, quiet=None):
    """One live day; extra(e, hour) -> [(total_up, dest)] novel uploads;
    quiet(e, hour) -> True: the entity is silent that hour."""
    out = {"acc": {}, "rows": {}}
    for k in range(24):
        now = T_MID + d * DAY + (k + 1) * DT
        for e in ents:
            if quiet is not None and quiet(e, k):
                feed.tick(e, now, DT, active=False)
                continue
            up = 1e6 * math.exp(0.3 * rng.normal())
            rows = stream(now, up, ids[INTERNAL])
            nov = 0.0
            for b, dest in extra(e, k):
                rows += stream(now, b, ids[dest], n=3)
                nov += b
            feed.tick(e, now, DT, up=up + nov, down=3 * up, stream=rows)
        if hook:
            hook(k, now)
        run_engine(eng, st, now, training=False, dt=DT)
        for e in ents:
            a = acc(st, now, e)
            if a.get("budget_exfil"):
                out["acc"][e] = out["acc"].get(e, 0) + 1
            out["rows"][e] = budget_row(st, now, e)
    return out, now


def test_d_small_novel_upload_recorded_below_floor(trained):
    st, eng, feed, ents, ids, rng = trained
    a = ents[0]
    # (d) 3 MB over the working day to a destination A never used, external
    out, now = live_day(st, eng, feed, ents, ids, rng, 20,
                        lambda e, k: [(375e3, EXT)] if e == a and 9 <= k < 17 else [])
    assert out["acc"] == {}
    assert not [x for x in events(st) if x.extra["axis"] == "exfil"]
    row = out["rows"][a]
    assert row["up_novel.day"][0] == pytest.approx(3e6, rel=1e-4)        # recorded
    assert row["up_novel.day"][0] > row["up_novel.day"][1]               # beyond z_q ...
    assert row["up_novel.day"][2] < 1e-3                                 # ... small tail p
    assert out["rows"][ents[1]]["up_novel.day"][0] == 0.0
    prof = st.profile(S, a).extra["budget"]
    assert prof["quantities"]["up_novel"]["abs_floor"] == 20e6
    v = prof["exfil_today"]["up_novel_bytes"]
    assert v is not None and 0 < v <= 3e6 + 1
    # positive control, next day: 40 MB to another new external destination alarms
    out, now = live_day(st, eng, feed, ents, ids, rng, 21,
                        lambda e, k: [(5e6, EXT2)] if e == a and 9 <= k < 17 else [])
    assert set(out["acc"]) == {a}
    ev = [x for x in events(st, a) if x.extra["axis"] == "exfil"]
    assert len(ev) == 1                     # one event per quantity and local day
    assert ev[0].extra["quantity"] == "up_novel" and ev[0].extra["scope"] == "entity"
    assert "MB" in ev[0].description and ev[0].axes == ["exfil"]


def test_actor_chain_aggregates_split_exfil(trained):
    st, eng, feed, ents, ids, rng = trained
    b, c = ents[1], ents[2]
    link = {"version": 1, "links": [], "actors": {"act-1": {"members": [f"{S}|{b}", c]}}}

    def extra(e, k):                    # 12 MB each: below the 20 MB floor per IP
        if e == b and 8 <= k < 12:
            return [(3e6, EXT2)]
        if e == c and 13 <= k < 17:
            return [(3e6, EXT2)]
        return []

    def hook(k, now):
        if k == 0:
            put_model(st, S, "__system__", "model.link", link)

    # the actor hops from b to c at noon: b falls silent, c is the one seen last
    out, now = live_day(st, eng, feed, ents, ids, rng, 22, extra, hook,
                        quiet=lambda e, k: (e == b and k >= 12) or (e == c and k < 12))
    ev = [x for x in events(st) if x.extra["axis"] == "exfil" and x.ts > now - DAY]
    assert len(ev) == 1, [(x.entity, x.extra["scope"]) for x in ev]
    assert ev[0].entity == c and ev[0].extra["scope"] == "actor"
    assert sorted(ev[0].extra["actor"]) == sorted([b, c])
    assert "by actor" in ev[0].description
    assert out["rows"][b]["up_novel.day"][0] == pytest.approx(12e6, rel=1e-4)
    assert out["rows"][c]["up_novel.day"][0] == pytest.approx(12e6, rel=1e-4)
    put_model(st, S, "__system__", "model.link", {"version": 1, "links": [], "actors": {}})


def test_young_entity_scored_with_peer_pooled_tail(trained):
    st, eng, feed, ents, ids, rng = trained
    n = "10.2.0.99"
    put_class(st, ents + [n])
    def extra(e, k):
        return [(6e6, EXT)] if e == n and 10 <= k < 15 else []

    out, now = live_day(st, eng, feed, ents + [n], ids, rng, 23, extra)
    m = st.get_model(S, n, MODEL)
    assert m["fit"]["src"][4] == 2 and m["fit"]["days"][4] == 0         # up_novel: peer
    assert m["fit"]["src"][0] == 2
    assert out["acc"].get(n)
    ev = [x for x in events(st, n) if x.extra["axis"] == "exfil"]
    assert ev and ev[0].extra["fit_source"] == "peer" and ev[0].extra["history_days"] == 0
    # its ordinary internal traffic (novel only to itself, but internal) never
    # alarms: every finding starts with the upload (the bytes also count as volume)
    t_up = T_MID + 23 * DAY + 10 * DT
    assert all(x.ts > t_up for x in events(st, n))
    assert {x.extra["quantity"] for x in events(st, n)} <= {"up_novel", "bytes_up"}
