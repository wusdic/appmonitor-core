"""B08 NoveltyEngine (docs/lib3/engines.md '## B08'): spec unit tests (a)-(e).

Engines only talk to the store, so raw categorical sets (act.tokens,
tls.sni_etld1_set, l4.*_set, act.stream) are written directly as R1/R2 would.

Spec test mapping:
  (a) test_a_stable_client_new_template
  (b) test_b_explorer_new_template_not_significant
  (c) test_c_sensitive_class_rare_first_seen
  (d) test_d_class_adoption_discount
  (e) test_e_external_upload_adoption_record
Edge cases (empty store, silence, NaN inputs, training, cadence, perf, ...)
live in test_b08_novelty_edges.py.
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional

import numpy as np
import pytest

from helpers import DT, T0, add_obs_tick, make_store, run_engine, set_trust

from app.engines.behavior.lib import emit
from app.engines.behavior.lib import m_template as MT
from app.engines.behavior.lib import m_vocab as V
from app.engines.behavior.lib.classkeys import ORG
from app.engines.behavior.novelty import NoveltyEngine
from app.models.schema import RawMetric

S = "erp"
HOST = "erp.corp"


# ---------------------------------------------------------------- helpers
def tok(path: str, method: str = "GET", host: str = HOST, status: str = "2xx") -> str:
    return f"{method} {host} {path}|{status}"


def write_tick(store, e: str, now: float, tokens: Optional[Dict[str, float]] = None,
               sni: Optional[Dict[str, float]] = None, dports: Optional[Dict[str, float]] = None,
               peers: Optional[Dict[str, float]] = None, stream=None, system: str = S) -> None:
    m = {}
    if tokens:
        m["act.tokens"] = dict(tokens)
        m["act.events"] = float(sum(tokens.values()))
    if sni:
        m["tls.sni_etld1_set"] = dict(sni)
    if dports:
        m["l4.dport_set"] = dict(dports)
    if peers:
        m["l4.peer_set"] = dict(peers)
    add_obs_tick(store, system, e, now, m)
    if stream is not None:
        arr = np.array(stream, dtype=MT.STREAM_DTYPE)
        arr.flags.writeable = False
        store.add_raw(RawMetric(name="act.stream", value=arr, ts=now, system=system, entity=e))
        store.add_raw(RawMetric(name="act.stream_frac", value=1.0, ts=now, system=system,
                                entity=e))


def put_class(store, members: Dict[str, str], system: str = S) -> None:
    """model.class@org with {ip: role} assignments (prob 1)."""
    assign = {f"{system}|{ip}": {"role": rid, "sub": None, "prob": 1.0, "static": [],
                                 "pool": None, "super": "human"} for ip, rid in members.items()}
    roles: Dict[str, Dict] = {}
    for ip, rid in members.items():
        roles.setdefault(rid, {"name": rid, "members": [], "version": 1})["members"].append(
            f"{system}|{ip}")
    store.put_model(ORG[0], ORG[1], "model.class", {"assign": assign, "roles": roles,
                                                    "version": 1})


def score_row(store, e: str, now: float, system: str = S) -> Dict[str, float]:
    return emit.read_row(store, system, e, emit.SCORE, now)


def pm_row(store, e: str, now: float, system: str = S) -> Dict[str, float]:
    return emit.read_row(store, system, e, emit.PM, now)


def last(store, e: str, system: str = S) -> Dict:
    return store.profile(system, e).extra["categorical"]["last"]


def events(store, e: Optional[str] = None, kinds=("first_seen", "rare_access", "class_adopted"),
           since: Optional[float] = None, system: str = S) -> List:
    return store.events(system, e, since=since, kinds=list(kinds), limit=1000)


def trust_all(store, ents: Iterable[str], ts: float, system: str = S) -> None:
    for e in ents:
        set_trust(store, system, e, [ts], 1.0)


# ------------------------------------------------------------------ spec (a)
def test_a_stable_client_new_template():
    st = make_store()
    eng = NoveltyEngine()
    e = "10.0.0.5"
    base = {tok("/api/v1/sync", "POST"): 20.0}
    for i in range(200):
        now = T0 + i * DT
        write_tick(st, e, now, tokens=base)
        run_engine(eng, st, now, training=True)
    now = T0 + 200 * DT
    write_tick(st, e, now, tokens={**base, tok("/api/v1/export"): 1.0})
    run_engine(eng, st, now)
    pm = pm_row(st, e, now)["novelty"]
    lst = last(st, e)
    assert lst["k_new"] == 1
    assert lst["p_rate"] < 1e-3                  # the Good-Turing rate test alone
    assert pm < 1e-3
    assert score_row(st, e, now)["novelty"] > 3.0
    # the next ordinary tick is quiet again (the known template scores 0)
    now += DT
    write_tick(st, e, now, tokens=base)
    run_engine(eng, st, now)
    assert score_row(st, e, now)["novelty"] == 0.0


# ------------------------------------------------------------------ spec (b)
def test_b_explorer_new_template_not_significant():
    st = make_store()
    eng = NoveltyEngine()
    ex = "10.0.1.7"
    peers = ["10.0.1.1", "10.0.1.2", "10.0.1.3"]
    pool = [tok(f"/kb/article/{k}") for k in range(260)]
    common = tok("/home")
    nxt = 0
    for i in range(60):
        now = T0 + i * DT
        for p in peers:                                   # peers read the whole pool
            write_tick(st, p, now, tokens={common: 5.0, **{t: 1.0 for t in pool}})
        new = pool[nxt:nxt + 3]
        nxt += 3
        write_tick(st, ex, now, tokens={common: 7.0, **{t: 1.0 for t in new}})
        run_engine(eng, st, now, training=True)
    model = V.get(st, S, ex)
    p_new = V.novelty_rate(model, "tmpl")
    assert 0.25 <= p_new <= 0.35                          # an explorer: p_new ~ 0.3
    now = T0 + 60 * DT
    for p in peers:
        write_tick(st, p, now, tokens={common: 5.0, **{t: 1.0 for t in pool}})
    write_tick(st, ex, now, tokens={common: 7.0, pool[250]: 1.0})
    run_engine(eng, st, now)
    lst = last(st, ex)
    assert lst["k_new"] == 1
    assert lst["p_rate"] > 0.05
    assert pm_row(st, ex, now)["novelty"] > 0.05


# ------------------------------------------------------------------ spec (c)
def _class_of_ten(st, eng, n_ticks=40, sensitive_user="10.1.0.0"):
    ips = [f"10.1.0.{k}" for k in range(10)]
    put_class(st, {ip: "clerk" for ip in ips})
    base = {tok("/orders"): 10.0, tok("/dashboard"): 6.0, tok("/orders/view/{num}"): 4.0}
    for i in range(n_ticks):
        now = T0 + i * DT
        for ip in ips:
            t = dict(base)
            if ip == sensitive_user:
                t[tok("/hr/salary/export")] = 1.0
            write_tick(st, ip, now, tokens=t)
        run_engine(eng, st, now, training=True)
    return ips, base, T0 + n_ticks * DT


def test_c_sensitive_class_rare_first_seen():
    st = make_store()
    eng = NoveltyEngine()
    ips, base, now = _class_of_ten(st, eng)
    cfg = {"sensitive_patterns": ["/hr/", "salary"]}
    ck = "class:clerk"
    cm = V.get(st, S, ck)
    assert cm is not None and cm["members"] == 10
    assert abs(cm["dims"]["tmpl"][f"GET {HOST} /hr/salary/export"][1] - 1.0) < 0.05   # df = 1
    e = ips[3]
    for ip in ips:
        t = dict(base)
        if ip in (ips[0], e):
            t[tok("/hr/salary/export")] = 1.0
        write_tick(st, ip, now, tokens=t)
    run_engine(eng, st, now, config=cfg)
    fs = [ev for ev in events(st, e, kinds=("first_seen",)) if ev.ts == now]
    assert len(fs) == 1
    ev = fs[0]
    assert ev.extra["dim"] == "tmpl" and ev.extra["value"].endswith("/hr/salary/export")
    assert ev.extra["tier"] == "class"
    assert ev.extra["bits"] >= 12.0
    assert abs(ev.extra["idf"] - (math.log(11.0 / 2.0) + 1.0)) < 0.05
    assert "privilege" in ev.axes
    assert ev.extra["sensitive"] is True
    ra = [x for x in events(st, e, kinds=("rare_access",)) if x.ts == now]
    assert len(ra) == 1 and ra[0].severity.value == "medium"
    axes = emit.read_dict(st, S, e, emit.AXES, now)
    assert "privilege" in axes["novelty"]
    # the habitual user of the path is not reported
    assert not [x for x in events(st, ips[0]) if x.ts == now]
    # dedupe: the second access of the same value emits nothing new
    now2 = now + DT
    for ip in ips:
        t = dict(base)
        if ip in (ips[0], e):
            t[tok("/hr/salary/export")] = 1.0
        write_tick(st, ip, now2, tokens=t)
    run_engine(eng, st, now2, config=cfg)
    assert not [x for x in events(st, e) if x.ts == now2]


# ------------------------------------------------------------------ spec (d)
def test_d_class_adoption_discount():
    st = make_store()
    eng = NoveltyEngine()
    ips = [f"10.2.0.{k}" for k in range(8)]
    put_class(st, {ip: "erp_clerk" for ip in ips})
    base = {tok("/orders"): 10.0, tok("/dashboard"): 6.0, tok("/orders/view/{num}"): 4.0}
    for i in range(40):
        now = T0 + i * DT
        for ip in ips:
            write_tick(st, ip, now, tokens=base)
        run_engine(eng, st, now, training=True)
    t0 = T0 + 40 * DT
    new = tok("/v2/orders")
    adopters = ips[:6]
    starts = {ip: t0 + (0 if k < 3 else 4) * DT for k, ip in enumerate(adopters)}
    for j in range(8):
        now = t0 + j * DT
        trust_all(st, ips, now)
        for ip in ips:
            t = dict(base)
            if ip in starts and now >= starts[ip]:
                t[new] = 2.0
            write_tick(st, ip, now, tokens=t)
        run_engine(eng, st, now)
        for ip, ts in starts.items():
            if ts == now:
                lst = last(st, ip)
                sc = score_row(st, ip, now)["novelty"]
                assert lst["k_new"] == 1 and lst["k_adopted"] == 1
                assert lst["novelty_raw"] > 1.0
                assert sc <= 0.1 * lst["novelty_raw"] + 1e-4    # float32 ring, 4-dp profile
                fs = [x for x in events(st, ip, kinds=("first_seen",)) if x.ts == now]
                assert all(x.extra["adopted"] and x.severity.value == "info" for x in fs)
    ca = events(st, "class:erp_clerk", kinds=("class_adopted",))
    assert len(ca) == 1
    assert ca[0].severity.value == "info" and ca[0].extra["value"] == f"GET {HOST} /v2/orders"
    recs = V.adoption_records(st, S, "class:erp_clerk")
    rec = [r for r in recs if r["value"] == f"GET {HOST} /v2/orders"][0]
    assert rec["adopted"] and len(rec["members"]) == 6
    assert rec["flags"]["external"] is False and rec["flags"]["upload"] is False
    # the two members that did not adopt emitted nothing
    for ip in ips[6:]:
        assert not events(st, ip, since=t0)


# ------------------------------------------------------------------ spec (e)
def test_e_external_upload_adoption_record():
    st = make_store()
    eng = NoveltyEngine()
    ips = [f"10.4.0.{k}" for k in range(6)]
    put_class(st, {ip: "api" for ip in ips})
    base = {tok("/v1/resource", host="api.corp"): 12.0}
    sni0 = {"corp-sso.example.com": 2.0}
    for i in range(40):
        now = T0 + i * DT
        for ip in ips:
            write_tick(st, ip, now, tokens=base, sni=sni0)
        run_engine(eng, st, now, training=True)
    now = T0 + 40 * DT
    dest = "example.org"                         # telemetry-sync.example.org at eTLD+1
    did = MT.dest_id_of(dest)
    cfg = {"org_domains": ["corp"]}
    adopters = ips[:4]
    for ip in ips:
        trust_all(st, [ip], now)
        if ip in adopters:
            rows = [(now - 600 + 10 * k, 0, 2, 30000.0 / 4, 256.0, did, 0) for k in range(4)]
            write_tick(st, ip, now, tokens=base, sni={**sni0, dest: 4.0}, stream=rows)
        else:
            write_tick(st, ip, now, tokens=base, sni=sni0)
    run_engine(eng, st, now, config=cfg)
    for ip in adopters:
        lst = last(st, ip)
        assert lst["k_adopted"] == 1
        assert lst["novelty_raw"] > 1.0
        assert score_row(st, ip, now)["novelty"] <= 0.1 * lst["novelty_raw"] + 1e-4
        fs = [x for x in events(st, ip, kinds=("first_seen",)) if x.ts == now]
        assert fs and fs[0].extra["adopted"] and fs[0].extra["tier"] == "system"
        assert "exfil" in fs[0].axes
    recs = V.adoption_records(st, S, "class:api")
    rec = [r for r in recs if r["dim"] == "sni" and r["value"] == dest][0]
    assert rec["flags"]["external"] is True and rec["flags"]["upload"] is True
    assert rec["adopted"] is True and set(rec["members"]) == set(adopters)
    assert len(events(st, "class:api", kinds=("class_adopted",))) == 1


# ------------------------------------------------ round 4 (evaluator): small class
def test_rare_access_in_a_class_of_eight():
    """Pack A's ERP users are a class of 8 in which only the HR persona uses
    /hr/salary/export (T7). The smoothed class IDF (1 + 1)/(8 + 1) = 22 %
    never calls that value rare, so T7 produced only an entity-tier INFO
    first_seen; the Jeffreys median of the peers' share (1 of 7) does."""
    from app.engines.behavior.novelty import peer_rare
    st = make_store()
    eng = NoveltyEngine()
    ips = [f"10.1.0.{k}" for k in range(8)]
    put_class(st, {ip: "clerk" for ip in ips})
    base = {tok("/orders"): 10.0, tok("/dashboard"): 6.0, tok("/orders/view/{num}"): 4.0}
    for i in range(40):
        now = T0 + i * DT
        for ip in ips:
            t = dict(base)
            if ip == ips[0]:
                t[tok("/hr/salary/export")] = 1.0
            write_tick(st, ip, now, tokens=t)
        run_engine(eng, st, now, training=True)
    now = T0 + 40 * DT
    cfg = {"sensitive_patterns": ["/hr/", "salary"]}
    cm = V.get(st, S, "class:clerk")
    assert cm["members"] == 8
    assert V.idf(cm, "tmpl", f"GET {HOST} /hr/salary/export") < math.log(5.0) + 1.0
    assert peer_rare(cm, "tmpl", f"GET {HOST} /hr/salary/export")
    e = ips[3]
    for ip in ips:
        t = dict(base)
        if ip in (ips[0], e):
            t[tok("/hr/salary/export")] = 1.0
        write_tick(st, ip, now, tokens=t)
    run_engine(eng, st, now, config=cfg)
    ra = [x for x in events(st, e, kinds=("rare_access",)) if x.ts == now]
    assert len(ra) == 1 and ra[0].severity.value == "medium"
    assert "privilege" in ra[0].axes
    assert not [x for x in events(st, ips[0], kinds=("rare_access",)) if x.ts == now]


def test_peer_rare_is_the_jeffreys_median_of_the_peer_share():
    from app.engines.behavior.novelty import peer_rare

    def cls(n, df):
        return {"kind": "class", "n_ent": float(n), "dims": {"tmpl": {"v": [1.0, float(df)]}}}
    assert peer_rare(cls(8, 1), "tmpl", "v")          # 1 of 7 peers
    assert peer_rare(cls(4, 0), "tmpl", "v")          # nobody of 3
    assert not peer_rare(cls(4, 1), "tmpl", "v")      # 1 of 3
    assert not peer_rare(cls(9, 2), "tmpl", "v")      # 2 of 8
    assert not peer_rare(cls(2, 0), "tmpl", "v")      # 1 peer: undecidable
    assert not peer_rare(None, "tmpl", "v")


def test_lib4_category_counts_its_grain_coverage_in_canonical_mode():
    """Round 4 (evaluator): a lib-4 match describes a 15-min grain; at 60 s
    the same sustained activity matched on 15 ticks per grain and weighed
    15x its 900-s count in the 'cat' dimension (pack E: JSD drift on the
    machine personas). The category now counts dt / 900 per match."""
    from app.models.schema import Severity, SignatureMatch
    cfg = {"grain_mode": "canonical", "strict": True}
    masses = {}
    for dt in (60.0, 900.0):
        st = make_store()
        eng = NoveltyEngine()
        e = "10.0.0.9"
        t = T0 - (T0 % 3600.0)
        for i in range(int(3600 // dt)):
            t += dt
            write_tick(st, e, t, tokens={tok("/api/v1/sync", "POST"): 20.0 * dt / 900.0})
            st.add_match(SignatureMatch(system=S, entity=e, ts=t - 1.0, signature_id="health",
                                        label="h", category="health", confidence=1.0,
                                        matched_terms=[], severity=Severity.INFO, evidence={}))
            run_engine(eng, st, t, dt=dt, config=cfg, training=True)
        run = V.get(st, S, e)["run"]
        win = run["win"][1]                       # the 24-h window (slow decay)
        f = 2.0 ** (-(t - win["t0"]) / 86400.0)
        masses[dt] = win["tot"]["cat"] * f
    # one hour of a sustained category: ~4 grain-units at both cadences
    assert masses[60.0] == pytest.approx(masses[900.0], rel=0.1)
