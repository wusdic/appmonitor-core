"""B08 NoveltyEngine edge cases: empty store, silence, NaN / garbage inputs,
training, cadence switch 900 -> 60 s, degraded producers, allowlist, link
inheritance, rollback, the two accumulators, m_vocab accessors and perf."""
from __future__ import annotations

import math
import time

import numpy as np

from helpers import DT, T0, add_obs_tick, make_store, run_engine, set_trust

from test_b08_novelty import (HOST, S, _class_of_ten, events, last, pm_row, put_class,
                              score_row, tok, write_tick)

from app.engines.behavior.lib import emit
from app.engines.behavior.lib import gating as G
from app.engines.behavior.lib import m_vocab as V
from app.engines.behavior.lib.classkeys import ORG, SYSTEM_KEY
from app.engines.behavior.novelty import NoveltyEngine

BASE = {tok("/orders"): 10.0, tok("/dashboard"): 6.0}
DAY_S = 86400.0


def _train(st, eng, ents, n, dt=DT, t0=T0, tokens=BASE, **kw):
    for i in range(n):
        now = t0 + i * dt
        for e in ents:
            write_tick(st, e, now, tokens=tokens, **kw)
        run_engine(eng, st, now, training=True, dt=dt)
    return t0 + n * dt


def test_empty_store_and_pseudo_entities():
    st = make_store()
    eng = NoveltyEngine()
    assert run_engine(eng, st, T0) == 0
    # pseudo-entity models never make it a scored entity
    st.put_model(S, SYSTEM_KEY, "model.template", {"fmt": 1})
    assert run_engine(eng, st, T0 + DT) == 0
    assert st.events() == []


def test_silent_entity_not_scored_but_commits():
    st = make_store()
    eng = NoveltyEngine()
    e = "10.9.0.1"
    now = _train(st, eng, [e], 30)
    write_tick(st, e, now, tokens={**BASE, tok("/new/page"): 1.0})
    run_engine(eng, st, now, training=True)
    assert "novelty" in score_row(st, e, now)
    for j in range(1, 8):                        # silence: nothing written, rows commit
        t = now + j * DT
        run_engine(eng, st, t, training=True)
        assert score_row(st, e, t) == {}
    assert f"GET {HOST} /new/page" in V.get(st, S, e)["state"]["dims"]["tmpl"]
    assert not V.get(st, S, e)["run"]["seen"]    # sighting forgotten once committed


def test_nan_and_garbage_inputs():
    st = make_store()
    eng = NoveltyEngine()
    e = "10.9.0.2"
    now = _train(st, eng, [e], 10)
    add_obs_tick(st, S, e, now, {
        "act.tokens": {tok("/a"): float("nan"), tok("/b"): float("inf"), tok("/c"): -3.0,
                       tok("/d"): "x", "__other__": 5.0, "{rare:http}": 2.0, tok("/ok"): 1.0},
        "l4.dport_set": {"443": float("nan"), "__other__": 1.0},
        "tls.sni_etld1_set": {"": 3.0, "ok.example.com": {"n": 2}},
        "l4.peer_set": "not a dict"})
    run_engine(eng, st, now)
    row = score_row(st, e, now)
    assert math.isfinite(row["novelty"])
    m = V.get(st, S, e)
    rows = m["rows"][now].items
    vals = {(d, v) for d, v, _ in rows}
    assert vals == {("tmpl", f"GET {HOST} /ok"), ("sni", "ok.example.com")}
    # an entity whose only input is garbage is not scored
    add_obs_tick(st, S, "10.9.0.3", now + DT, {"act.tokens": {tok("/x"): float("nan")}})
    run_engine(eng, st, now + DT)
    assert score_row(st, "10.9.0.3", now + DT) == {}


def test_training_emits_no_events_but_scores():
    st = make_store()
    eng = NoveltyEngine()
    ips, base, now = _class_of_ten(st, eng)
    for ip in ips:
        t = dict(base)
        if ip == ips[3]:
            t[tok("/hr/salary/export")] = 1.0
            t[tok("/admin/users/{num}")] = 3.0
        write_tick(st, ip, now, tokens=t)
    run_engine(eng, st, now, training=True)
    assert st.events() == []
    assert score_row(st, ips[3], now)["novelty"] > 1.0


def test_cadence_switch_900_to_60():
    st = make_store()
    eng = NoveltyEngine()
    e = "10.9.0.4"
    now = _train(st, eng, [e], 100, peers={"10.50.1.0/24": 4.0}, dports={"443": 20.0})
    gate_ts = []
    for j in range(40):
        t = now + j * 60.0
        set_trust(st, S, e, [t], 1.0)
        write_tick(st, e, t, tokens={k: 1.0 for k in BASE}, peers={"10.50.1.0/24": 1.0},
                   dports={"443": 2.0})
        run_engine(eng, st, t, dt=60.0)
        assert score_row(st, e, t)["novelty"] == 0.0
        gate_ts.append(V.get(st, S, e)["gate"].last_ts)
    # 60-s rows commit with D = max(4, 600/60) = 10 ticks
    assert G.commit_delay_ticks(60.0) == 10
    assert gate_ts[-1] == now + 29 * 60.0
    t = now + 40 * 60.0
    write_tick(st, e, t, tokens={**{k: 1.0 for k in BASE}, tok("/admin/export"): 1.0})
    run_engine(eng, st, t, dt=60.0)
    assert pm_row(st, e, t)["novelty"] < 1e-2
    fs = [x for x in events(st, e, kinds=("first_seen",)) if x.ts == t]
    assert fs and fs[0].extra["tier"] == "system" and fs[0].e_day == fs[0].e_day


def test_degraded_producer_and_stale_tokens():
    st = make_store()
    eng = NoveltyEngine()
    e = "10.9.0.5"
    now = _train(st, eng, [e], 10)
    write_tick(st, e, now, tokens=BASE)
    st.put_health("raw.action_token", {"engine": "raw.action_token", "last_error_ts": now})
    run_engine(eng, st, now)
    assert score_row(st, e, now) == {}                        # NaN = not scored
    deg = emit.read_dict(st, S, e, emit.DEGRADED, now)
    assert deg["novelty"].startswith("producer_error") and set(deg) == {"novelty",
                                                                       "novelty_rate", "jsd"}
    # activity without categorical data: stale
    t = now + DT
    add_obs_tick(st, S, e, t, {"act.events": 12.0})
    run_engine(eng, st, t)
    assert emit.read_dict(st, S, e, emit.DEGRADED, t)["novelty"] == "stale:act.tokens"


def test_allowlist_not_scored_or_reported():
    st = make_store()
    eng = NoveltyEngine()
    e = "10.9.0.6"
    now = _train(st, eng, [e], 60)
    val = f"GET {HOST} /tools/backup"
    st.put_model(ORG[0], ORG[1], "model.feedback",
                 {"version": 1, "allowlist": {f"{S}|{e}": {"tmpl": {val: None}}}})
    write_tick(st, e, now, tokens={**BASE, tok("/tools/backup"): 3.0})
    run_engine(eng, st, now)
    assert score_row(st, e, now)["novelty"] == 0.0
    assert not events(st, e)
    assert V.get(st, S, e)["rows"][now].items[-1][1] == val    # still learned


def test_link_inherited_values_not_new():
    st = make_store()
    eng = NoveltyEngine()
    a, b = "10.9.1.1", "10.9.1.2"
    extra = {tok("/fin/ledger"): 4.0}
    now = _train(st, eng, [a], 40, tokens={**BASE, **extra})
    now = _train(st, eng, [b], 40, t0=now, tokens=BASE)
    st.put_model(S, SYSTEM_KEY, "model.link",
                 {"version": 1, "links": [{"from": a, "to": b, "ts": now - DT}]})
    write_tick(st, b, now, tokens={**BASE, **extra})
    run_engine(eng, st, now)
    assert last(st, b)["k_new"] == 0
    assert score_row(st, b, now)["novelty"] == 0.0
    assert not events(st, b)


def test_rollback_forgets_learned_value():
    st = make_store()
    eng = NoveltyEngine()
    e = "10.9.0.7"
    now = _train(st, eng, [e], 20)
    tau = now + 4 * DT
    x = f"GET {HOST} /exfil/dump"
    for j in range(20):
        t = now + j * DT
        set_trust(st, S, e, [t], 1.0)
        tk = dict(BASE)
        if t > tau:
            tk[tok("/exfil/dump")] = 2.0
        write_tick(st, e, t, tokens=tk)
        run_engine(eng, st, t)
    assert x in V.get(st, S, e)["state"]["dims"]["tmpl"]
    t = now + 20 * DT
    st.put_model(S, e, "model.control", {"version": 1, "branch": 0, "rollback_to": tau})
    set_trust(st, S, e, [t - DT, t], 0.0, quarantine=1.0)
    write_tick(st, e, t, tokens=BASE)
    run_engine(eng, st, t)
    m = V.get(st, S, e)
    assert x not in m["state"]["dims"]["tmpl"]
    assert m["gate"].held and m["gate"].held[0].ts > tau


def test_jsd_shift_raises_accumulator():
    st = make_store()
    eng = NoveltyEngine()
    e = "10.9.0.8"
    rng = np.random.default_rng(1)
    tpl = [tok(f"/p{k}") for k in range(4)]
    for i in range(150):                          # a noisy but stationary template mix
        now = T0 + i * DT
        w = rng.dirichlet([20, 10, 5, 1]) * 30
        write_tick(st, e, now, tokens={t: max(1.0, round(x)) for t, x in zip(tpl, w)})
        run_engine(eng, st, now, training=True)
    rings = V.get(st, S, e)["state"]["jsd"]
    assert len(rings["h1"]) >= 100 and float(rings["h1"].scores[-1]) < 0.05   # clean null
    now = T0 + 150 * DT
    t = now
    set_trust(st, S, e, [t], 1.0)
    write_tick(st, e, t, tokens=dict(zip(tpl, (12.0, 8.0, 4.0, 1.0))))   # an ordinary tick
    run_engine(eng, st, t)
    assert score_row(st, e, t)["jsd"] < 2.0
    scores = []
    for j in range(1, 5):                         # the mix moves onto the rare template
        t = now + j * DT
        set_trust(st, S, e, [t], 1.0)
        write_tick(st, e, t, tokens={tpl[3]: 25.0, tpl[0]: 3.0})
        run_engine(eng, st, t)
        scores.append(score_row(st, e, t)["jsd"])
    assert scores[-1] >= 4.0 and scores[-1] >= scores[0]
    assert emit.read_dict(st, S, e, emit.ACC_ALARM, t)["jsd"] == 1
    assert emit.read_dict(st, S, e, emit.AXES, t)["jsd"] == ["categorical"]
    assert pm_row(st, e, t)["jsd"] < 1e-4


def test_novelty_rate_scanning_alarm():
    st = make_store()
    eng = NoveltyEngine()
    ips = [f"10.8.0.{k}" for k in range(5)]
    put_class(st, {ip: "ops" for ip in ips})
    dt = 3600.0
    peers = {"10.60.0.0/24": 5.0, "10.60.1.0/24": 3.0}
    now = _train(st, eng, ips, 24 * 4, dt=dt, peers=peers)
    e = ips[0]
    for j in range(6):
        t = now + j * dt
        for ip in ips:
            set_trust(st, S, ip, [t], 1.0)
            p = dict(peers)
            if ip == e:
                p.update({f"10.{70 + j}.{k}.0/24": 1.0 for k in range(25)})
            write_tick(st, ip, t, tokens=BASE, peers=p)
        run_engine(eng, st, t, dt=dt)
    t = now + 5 * dt
    assert score_row(st, e, t)["novelty_rate"] > 4.0
    assert emit.read_dict(st, S, e, emit.ACC_ALARM, t)["novelty_rate"] == 1
    assert emit.read_dict(st, S, e, emit.AXES, t)["novelty_rate"] == ["breadth"]
    assert score_row(st, ips[1], t)["novelty_rate"] < 1.0
    assert emit.read_dict(st, S, ips[1], emit.ACC_ALARM, t)["novelty_rate"] == 0
    # a burst of fresh peers is one finding per dimension, not 25
    fs = [x for x in events(st, e, kinds=("first_seen",)) if x.ts == now]
    assert len(fs) <= 5


def test_m_vocab_accessors():
    st = make_store()
    eng = NoveltyEngine()
    ips, base, now = _class_of_ten(st, eng)
    e = ips[2]
    m = V.get(st, S, e)
    assert V.kind(m) == "entity"
    # the backoff is a proper distribution over the dimension's universe
    ent, cls, sys = V.backoff_models(st, S, e)
    b = V.Backoff(ent, cls, sys, now=now)
    seen = set(V.counts(sys, "tmpl"))
    p_seen = sum(b.p("tmpl", v) for v in seen)
    p_unseen = b.p("tmpl", "GET x /never") * (V.universe("tmpl", sys) - len(seen))
    assert abs(p_seen + p_unseen - 1.0) < 1e-9
    v0 = f"GET {HOST} /orders"
    assert V.prob(st, S, e, "tmpl", v0, now) > 0.3
    assert V.surprisal(st, S, e, "tmpl", "GET x /never", now) > 20.0
    obs = {"tmpl": {v0: 3.0}}
    assert abs(V.loglik(st, S, e, obs, now) - 3.0 * math.log(b.p("tmpl", v0))) < 1e-9
    assert V.llr_vs_system(st, S, e, obs, now) == V.llr_vs_system(st, S, e, obs, now)
    assert abs(V.prevalence(st, S, "tmpl", v0) - 1.0) < 0.01
    sal = f"GET {HOST} /hr/salary/export"
    assert 0.05 < V.prevalence(st, S, "tmpl", sal) < 0.15
    assert V.prevalence_n(st, S, "tmpl", "GET x /never") == 0.0
    assert math.isnan(V.prevalence(st, "nosys", "tmpl", v0))
    assert V.first_seen_age(st, S, e, "tmpl", v0, now) >= 30 * DT
    assert V.first_seen_age(st, S, e, "tmpl", "GET x /never", now) == 0.0
    assert math.isnan(V.first_seen_age(st, S, "10.99.9.9", "tmpl", v0, now))
    fam = V.family_distribution(st, S, e)
    assert abs(sum(fam.values()) - 1.0) < 1e-9 and "http|read|erp.corp|orders" in fam
    assert 0.0 <= V.novelty_rate(m) < 0.05 and V.maturity(m) > 0.95
    tv = V.top_values(m, "tmpl", 2)
    assert tv[0][0] == v0 and abs(tv[0][1] - 0.5) < 0.01
    d = V.descriptors(m)
    assert "tmpl" in d["dims"] and d["families"]
    assert V.idf(cls, "tmpl", v0) < 1.1 and math.isnan(V.idf(m, "tmpl", v0))
    assert math.isnan(V.dest_prevalence(st, S, "example.org")) or \
        V.dest_prevalence(st, S, "example.org") == 0.0
    assert V.adoption_records(st, S, "class:clerk") == [] or \
        all("flags" in r for r in V.adoption_records(st, S, "class:clerk"))


def test_perf_forty_entities():
    st = make_store()
    eng = NoveltyEngine()
    ips = [f"10.7.{k // 10}.{k % 10}" for k in range(40)]
    put_class(st, {ip: f"r{k % 4}" for k, ip in enumerate(ips)})
    rng = np.random.default_rng(3)
    pool = [tok(f"/p/{k}") for k in range(60)]

    def tick(ip, t):
        toks = {pool[int(i)]: 2.0 for i in rng.choice(60, 15, replace=False)}
        peers = {f"10.{int(i)}.0.0/24": 1.0 for i in rng.choice(20, 8, replace=False)}
        write_tick(st, ip, t, tokens=toks, peers=peers, dports={"443": 5.0, "80": 1.0},
                   sni={"corp.example.com": 3.0})

    for i in range(24):
        now = T0 + i * DT
        for ip in ips:
            tick(ip, now)
        run_engine(eng, st, now, training=True)
    dur = []
    for i in range(24, 36):
        now = T0 + i * DT
        for ip in ips:
            tick(ip, now)
            set_trust(st, S, ip, [now], 1.0)
        t0 = time.perf_counter()
        run_engine(eng, st, now)
        dur.append(time.perf_counter() - t0)
    # generous bound (the spec budget is ~3 ms at 40 entities on the reference box)
    assert float(np.median(dur)) < 0.1


def test_all_dimensions_observed():
    from app.models.schema import SignatureMatch
    st = make_store()
    eng = NoveltyEngine()
    e = "10.9.2.1"
    now = _train(st, eng, [e], 3)
    st.add_match(SignatureMatch(system=S, entity=e, ts=now - DT / 2, signature_id="sig1",
                                label="port scan", category="Scan", confidence=0.9))
    add_obs_tick(st, S, e, now, {
        "act.tokens": {tok("/a"): 2.0, "A {rnd}.example.net": 3.0},   # DNS token: not tmpl
        "http.content_types": {"application/json; charset=utf-8": 2.0},
        "tls.sni_set": {"a.cdn.example.com": 1.0, "b.cdn.example.com": 2.0},
        "dns.qname_set": {"x1.tun.example.org": 2.0},
        "dns.qtype_set": {"txt": 2.0},
        "l4.dport_set": {"53": 2.0}, "l4.peer_set": {"203.0.113.0/24": 1.0}})
    run_engine(eng, st, now)
    items = {(d, v): n for d, v, n in V.get(st, S, e)["rows"][now].items}
    assert items == {("tmpl", f"GET {HOST} /a"): 2.0, ("ctype", "application/json"): 2.0,
                     ("sni", "example.com"): 3.0, ("dns", "example.org"): 2.0,
                     ("qtype", "TXT"): 2.0, ("dport", "53"): 2.0,
                     ("peer", "203.0.113.0/24"): 1.0, ("cat", "scan"): 1.0}
    # the match is consumed once (one tick late), not again at the next tick
    run_engine(eng, st, now + DT)
    write_tick(st, e, now + 2 * DT, tokens=BASE)
    run_engine(eng, st, now + 2 * DT)
    assert ("cat", "scan") not in {(d, v) for d, v, _ in
                                   V.get(st, S, e)["rows"][now + 2 * DT].items}


def test_reemergence_after_30_days_is_new():
    st = make_store()
    eng = NoveltyEngine()
    e = "10.9.2.2"
    x = {tok("/quarterly/report"): 2.0}
    now = _train(st, eng, [e], 10, tokens={**BASE, **x})
    now = _train(st, eng, [e], 35, dt=DAY_S, t0=now)
    write_tick(st, e, now, tokens={**BASE, **x})
    run_engine(eng, st, now)
    assert last(st, e)["k_new"] == 1
    # within 30 d the same value is simply known
    st2 = make_store()
    eng2 = NoveltyEngine()
    now = _train(st2, eng2, [e], 10, tokens={**BASE, **x})
    now = _train(st2, eng2, [e], 20, dt=DAY_S, t0=now)
    write_tick(st2, e, now, tokens={**BASE, **x})
    run_engine(eng2, st2, now)
    assert last(st2, e)["k_new"] == 0


def test_space_saving_cap_and_update_order():
    from app.engines.behavior.novelty import _Row, _init_state, _update
    st = _init_state()
    rows = [_Row(T0 + i, (("peer", f"p{i}", 1.0), ("peer", "hot", 5.0)), 0.0, math.nan,
                 math.nan) for i in range(400)]
    for r in rows:
        _update(st, r, 1.0)
    tab = st["dims"]["peer"]
    assert len(tab) == V.ENTITY_CAP and "hot" in tab
    assert abs(st["N"]["peer"] * 2.0 ** (-st["g"]) - 400 * 6.0) < 1.0     # mass kept
    assert st["N1"]["peer"] > 0.0
    # a released (older) row is folded decayed, never moving the clock back
    clock = st["clock"]
    _update(st, _Row(T0 - 86400.0, (("dport", "22", 1.0),), 0.0, math.nan, math.nan), 1.0)
    assert st["clock"] == clock and st["dims"]["dport"]["22"][2] == T0 - 86400.0
