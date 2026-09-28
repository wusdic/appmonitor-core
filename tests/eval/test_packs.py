"""Packs A-E / smoke / mini: timelines, calendar, truth completeness, every
scenario perturbs the features it declares, truth.py helpers, and the mini
pack end-to-end through runner -> metrics -> report."""
import copy
import dataclasses
import datetime as dt
import math
import re
from collections import Counter
from typing import Any, Dict

import numpy as np
import pytest

from app.eval import packs as P
from app.eval import truth as T
from app.pipeline.generator import BASE_KEYS, TWINS, Clock, TrafficGenerator

LETTERS = ["A", "B", "C", "D", "E"]
ALL = LETTERS + ["smoke", "mini"]
EXPECTED_IDS = ({f"T{i}" for i in range(1, 22)} | {"T4b", "T6b", "T9b"}
                | {f"L{i}" for i in range(1, 17)})


def _gen(name, seed=0):
    return TrafficGenerator(seed=seed, pack=P.get_pack(name))


# --------------------------------------------------------------------------- #
# Timelines
# --------------------------------------------------------------------------- #
def test_pack_timelines():
    # spec v2.1 (cadence.md §10, deliberate): warm-ups by packs.warmup_phases
    ticks = {"A": 984, "B": 1824, "C": 1416, "D": 1104, "E": 2112, "smoke": 180, "mini": 200}
    for name in ALL:
        p = P.get_pack(name)
        assert p.n_ticks == ticks[name], name
        assert all(agg == (d >= 900) for _, d, agg in p.phases)
        assert p.start_epoch + sum(n * d for n, d, _ in p.phases) == pytest.approx(p.end_epoch)
        assert p.scenario_start == pytest.approx(p.end_epoch - p.phases[-1][0] * p.phases[-1][1])
        assert p.config["tz"] == p.tz and p.seeds == [0, 1, 2, 3, 4]
    assert [p[:2] for p in P.get_pack("A").phases] == [(312, 3600.0), (288, 900.0), (384, 900.0)]
    e = P.get_pack("E").phases
    assert e[-1] == (1440, 60.0, False) and e[0][:2] == (672, 900.0)
    assert P.get_pack("C").tz == "Europe/Berlin"
    a = P.get_pack("A")
    first = Clock(a.tz).local(a.scenario_start)
    assert first.weekday() == 0 and first.hour == 0
    assert Clock(a.tz).local(a.scenario("T1").t_start).hour == 10     # tick 40 = 10:00


def test_warmup_phases_rule():
    """cadence.md §10: every live <= 900 pack's last warm-up phase runs at the
    live cadence over >= 1 full workday and >= 1 full non-workday; 16 d in
    all; D (live 3600) is 16 d at 3600; smoke keeps 120 warm-up ticks."""
    for name in ("A", "B", "C", "D"):
        p = P.get_pack(name)
        warm = p.phases[:-1]
        live = p.phases[-1][1]
        assert sum(n * d for n, d, _ in warm) == pytest.approx(16 * 86400.0), name
        if live <= 900.0:
            n, d, _ = warm[-1]
            assert d == live and n * d % 86400.0 == 0.0, name
            c = Clock(p.tz, p.calendar)
            d0 = c.local(p.scenario_start).date()
            k = int(n * d // 86400)
            kinds = [c.day_kind(d0 - dt.timedelta(days=i))[0] for i in range(1, k + 1)]
            assert any(kinds) and not all(kinds), name
            # k is the smallest such span (>= 2)
            assert k == 2 or all(kinds[:k - 1]) or not any(kinds[:k - 1]), name
        else:
            assert [w[:2] for w in warm] == [(384, 3600.0)]
    assert [w[:2] for w in P.get_pack("B").phases] == [(288, 3600.0), (384, 900.0), (1152, 900.0)]
    assert [w[:2] for w in P.get_pack("C").phases] == [(264, 3600.0), (480, 900.0), (672, 900.0)]
    sm = P.get_pack("smoke")
    assert sum(n for n, _, _ in sm.phases[:-1]) == 120
    c = Clock(sm.tz, sm.calendar)
    days = {c.day_kind(c.local(sm.start_epoch + h * 3600.0).date())[0]
            for h in range(int((sm.scenario_start - sm.start_epoch) // 3600))}
    assert days == {True, False}


def _days(p):
    c = Clock(p.tz, p.calendar)
    n = int(round((p.end_epoch - p.scenario_start) / 86400))
    d0 = c.local(p.scenario_start).date()
    return c, [d0 + dt.timedelta(days=i) for i in range(n)]


def test_calendars():
    c, days = _days(P.get_pack("B"))
    kinds = [(d.weekday(), c.day_kind(d)) for d in days]
    assert any(wd == 5 and k == (True, True) for wd, k in kinds)        # 调休 Saturday
    assert any(wd < 5 and k[0] is False for wd, k in kinds)             # holiday
    assert any(wd >= 5 and k[0] is False for wd, k in kinds)            # weekend
    # Pack C crosses the Berlin DST switch on scenario day 3
    p = P.get_pack("C")
    c, days = _days(p)
    offs = [c.local(c.epoch(d, 12.0)).utcoffset().total_seconds() for d in days]
    assert offs[:2] == [3600.0] * 2 and offs[2:] == [7200.0] * (len(days) - 2)
    # Pack D: a month end and a holiday
    p = P.get_pack("D")
    c, days = _days(p)
    assert any(d.month != days[0].month for d in days)
    assert any(d.weekday() < 5 and not c.day_kind(d)[0] for d in days)
    l1 = p.scenario("L1")
    ld = {Clock(p.tz).local(a).date() for a, _ in l1.params["windows"]}
    assert ld == {dt.date(2025, 9, 29), dt.date(2025, 9, 30)}


def test_get_pack_names_and_freshness():
    assert set(P.PACKS) == set(LETTERS) | {"smoke"}
    assert P.get_pack("pack_a").name == P.get_pack("Pack A").name == "A"
    assert P.get_pack("mini").name == "mini"
    assert P.get_pack("A") is not P.get_pack("A")
    with pytest.raises(KeyError):
        P.get_pack("Z")


# --------------------------------------------------------------------------- #
# Truth
# --------------------------------------------------------------------------- #
@pytest.mark.parametrize("name", ALL)
def test_truth_complete(name):
    p = P.get_pack(name)
    g = TrafficGenerator(seed=0, pack=p)
    assert T.validate(g.truth) == []                  # also: one scenario per entity
    for r in g.truth:
        assert r["pack"] == name
        assert p.scenario_start - 1e-6 <= r["t_start"] < r["t_end"] <= p.end_epoch + 1e-6
        for k in T.row_keys(r):
            assert k in g.personas or r["scenario_id"] == "T20", (r["scenario_id"], k)
        if r["label"] == "malicious":
            assert T.deadline_s(r, p.scenario_dt) > 0
    if name in LETTERS:
        for tw in TWINS:
            assert tw in g.personas
        assert p.fixtures["twins"] == [k.split("|") for k in TWINS]
        assert set(BASE_KEYS) <= set(g.personas)


def test_all_scenarios_covered():
    ids = set()
    for n in LETTERS:
        ids |= {T.base_id(sc.scenario_id) for sc in P.get_pack(n).scenarios}
    assert ids == EXPECTED_IDS
    e = {sc.scenario_id for sc in P.get_pack("E").scenarios if not sc.truth.get("background")}
    assert e == {"T1'", "T5'", "T12'"}


def test_truth_helpers():
    g = _gen("A")
    tr = g.truth
    assert {frozenset(s) for s in T.alias_sets(tr)} == {
        frozenset({"erp-prod|10.20.1.12", "erp-prod|10.20.1.112"})}
    w = T.scenario_windows(tr)
    assert "L11" not in w and w["T1"][1] == pytest.approx(g.t_clip)
    ctrl = T.control_entities(tr, g.personas)
    assert "oa-portal|10.30.2.21" in ctrl and "api-gateway|10.40.4.52" not in ctrl
    assert "erp-prod|10.20.1.114" not in ctrl                    # negative control is in L6
    assert T.by_id(tr, "T9")["looks_like"] == "10.30.2.21"
    assert T.malicious(tr) and T.legitimate(tr)
    # class members of a class scenario lose [t_start - 1 h, t_end + 24 h]
    d = _gen("D")
    t21 = T.by_id(d.truth, "T21")
    pop = list(d.personas)
    cw = T.control_windows(d.truth, pop, P.get_pack("D").scenario_start, d.t_clip)
    k = "api-gateway|10.40.4.51"
    assert k in cw and len(cw[k]) == 2
    assert cw[k][0][1] == pytest.approx(t21["t_start"] - 3600.0)
    assert cw[k][1] == pytest.approx((t21["t_end"] + 86400.0, d.t_clip))
    other = "erp-prod|10.20.4.30"
    assert cw[other] == [(P.get_pack("D").scenario_start, d.t_clip)]
    assert T.deadline_s({"max_ttd": "2 ticks"}, 900) == 1800
    assert T.deadline_s({"max_ttd": "30min"}, 60) == 1800
    assert T.deadline_s({"max_ttd": "3d"}, 900) == 3 * 86400


# --------------------------------------------------------------------------- #
# Every scenario perturbs its declared features
# --------------------------------------------------------------------------- #
FEATURE_METRIC = {
    "bytes_up": "bytes_up", "updown_log": "bytes_up", "bytes_per_flow": "bytes_up",
    "req_bytes_avg": "bytes_up",
    "http_requests": "requests", "intensity": "hour_profile",
    "new_peer_count": "new_peers", "sni_entropy": "new_peers", "periodicity": "new_peers",
    "timing_regularity": "new_peers", "distinct_peers": "peers",
    "ja3_diversity": "stacks",
    "new_template_ratio": "new_templates", "distinct_templates": "new_templates",
    "path_entropy": "template_mix", "objs": "objs",
    "http_4xx_rate": "n4xx", "http_write_ratio": "writes", "http_get_ratio": "gets",
    "dns_txt_ratio": "txt", "dns_qname_len": "txt", "dns_name_entropy": "txt",
    "dns_queries": "dns", "syn_ratio": "syn", "http_5xx_rate": "n5xx", "rtt": "rtt",
    "retransmit_rate": "retrans", "cross_system": "xsys", "entity": "presence",
    "think_time": "count_cv",
}
WINDOW_DAYS = {"T3": 3.0, "T15": None, "L5": None, "L12": None, "L1": None, "T5": None,
               "L10": None, "T19": None}
_NUM = re.compile(r"\d+")


def _tok(o):
    return f"{o.http_method} {o.http_host} {_NUM.sub('{n}', (o.http_path or '').split('?')[0])}"


def _metrics(obs, keys, system, clock):
    m: Dict[str, Any] = {k: 0.0 for k in ("bytes_up", "requests", "n4xx", "n5xx", "writes", "gets",
                                          "txt", "dns", "syn", "retrans", "xsys")}
    hours = np.zeros(24)
    toks, stacks, peers, objs = Counter(), set(), set(), set()
    rtt, pres = [], Counter()
    per_tick = Counter()
    for o in obs:
        k = f"{o.system}|{o.entity}"
        if k not in keys:
            continue
        c = int((o.extra or {}).get("count", 1))
        m["xsys"] += c * (o.system != system)
        pres[k] += c
        m["bytes_up"] += (o.extra or {}).get("bytes_up_total", o.bytes_up)
        m["retrans"] += o.retransmits
        rtt.append(o.rtt_ms)
        if o.tcp_flags == "SYN":
            m["syn"] += c
            peers.add(o.peer)
        if o.app_proto == "dns":
            m["dns"] += c
            m["txt"] += c if o.dns_qtype == "TXT" else 0
        if o.app_proto != "http":
            continue
        m["requests"] += c
        per_tick[int(o.ts // 900)] += c
        hours[clock.local(o.ts).hour] += c
        m["n4xx"] += c * (400 <= o.http_status < 500)
        m["n5xx"] += c * (o.http_status >= 500)
        m["writes"] += c * (o.http_method in ("POST", "PUT", "DELETE", "PATCH"))
        m["gets"] += c * (o.http_method == "GET")
        toks[_tok(o)] += c
        stacks.add((o.user_agent, o.ja3, o.ttl))
        peers.add(o.http_host)
        objs.update(_NUM.findall(o.http_path or ""))
    pt = np.array(list(per_tick.values()), float)
    m["count_cv"] = float(pt.std() / pt.mean()) if pt.size > 1 else 0.0
    m.update(hours=hours, toks=toks, stacks=frozenset(stacks), peers_set=peers,
             peers=len(peers), objs=len(objs), rtt=float(np.mean(rtt)) if rtt else 0.0,
             presence=dict(pres))
    return m


def _changed(metric, a, b):
    """True if metric differs materially between the scenario run (a) and
    the clean run (b)."""
    if metric == "stacks":
        return a["stacks"] != b["stacks"]
    if metric == "presence":
        return a["presence"] != b["presence"] and any(
            abs(a["presence"].get(k, 0) - b["presence"].get(k, 0)) > 5
            for k in set(a["presence"]) | set(b["presence"]))
    if metric == "new_peers":
        return bool(a["peers_set"] - b["peers_set"])
    if metric == "new_templates":
        return bool(set(a["toks"]) - set(b["toks"]))
    if metric == "template_mix":
        ks = sorted(set(a["toks"]) | set(b["toks"]))
        pa = np.array([a["toks"][k] for k in ks], float)
        pb = np.array([b["toks"][k] for k in ks], float)
        if pa.sum() == 0 or pb.sum() == 0:
            return pa.sum() != pb.sum()
        return 0.5 * np.abs(pa / pa.sum() - pb / pb.sum()).sum() > 0.2
    if metric == "count_cv":            # request timing regularity (script vs human)
        return abs(a[metric] - b[metric]) > 0.3 * max(b[metric], 0.05)
    if metric == "hour_profile":
        d = np.abs(a["hours"] - b["hours"]).sum()
        return d > 0.1 * max(b["hours"].sum(), 1.0) and d >= 5
    x, y = float(a[metric]), float(b[metric])
    return abs(x - y) > 0.1 * max(abs(y), 1.0) and abs(x - y) >= 1.0


def _probe_cases():
    out = []
    for name in LETTERS:
        for sc in P.get_pack(name).scenarios:
            if sc.truth.get("background"):
                continue
            out.append((name, sc.scenario_id))
    return out


def _probe(name, sid, with_scenario):
    p = P.get_pack(name)
    sc = p.scenario(sid)
    dt_s = p.scenario_dt
    days = WINDOW_DAYS.get(T.base_id(sid), 1.0)
    w0 = sc.t_start
    w1 = sc.t_end if days is None else min(sc.t_end, sc.t_start + days * 86400.0)
    start = p.scenario_start + math.floor((w0 - p.scenario_start) / dt_s) * dt_s
    n = int(math.ceil((w1 - start) / dt_s))
    keys = set(sc.keys()) | {sp.key for sp in sc.spawns}
    extra = [sc.params.get("looks_like_key"), sc.params.get("from_key")]
    pop = [k for k in BASE_KEYS if k in keys or k in extra]
    probe = dataclasses.replace(p, population=pop, start_epoch=start,
                                scenarios=[copy.deepcopy(sc)] if with_scenario else [])
    g = TrafficGenerator(seed=0, pack=probe)
    obs = []
    for _ in range(n):
        obs.extend(g.step(dt_s))
    return _metrics(obs, keys, sc.system, Clock(p.tz, p.calendar)), sc


@pytest.mark.parametrize("name,sid", _probe_cases())
def test_scenario_perturbs_declared_features(name, sid):
    a, sc = _probe(name, sid, True)
    b, _ = _probe(name, sid, False)
    feats = sc.truth["perturbed_features"]
    assert feats, sid
    metrics = {FEATURE_METRIC[f] for f in feats}        # every declared feature is mapped
    bad = [m for m in sorted(metrics) if not _changed(m, a, b)]
    assert not bad, (sid, bad)


def test_directional_effects():
    a, _ = _probe("A", "T1", True)
    b, _ = _probe("A", "T1", False)
    assert a["requests"] < 0.3 * b["requests"] and "c2.example.net" in a["peers_set"]
    a, _ = _probe("A", "T12", True)
    b, _ = _probe("A", "T12", False)
    assert a["n4xx"] - b["n4xx"] > 0.7 * (a["requests"] - b["requests"]) > 0
    a, _ = _probe("B", "T8", True)
    b, _ = _probe("B", "T8", False)
    assert a["objs"] > 10 * max(b["objs"], 1)          # object-id breadth (act.objs)
    a, _ = _probe("A", "T5", True)
    b, _ = _probe("A", "T5", False)
    assert a["hours"][2:4].sum() > 0 and b["hours"][2:4].sum() == 0


# --------------------------------------------------------------------------- #
# End to end: runner -> metrics -> report on the mini pack (raw engines only)
# --------------------------------------------------------------------------- #
def raw_registry():
    from app.core.engine import Registry
    from app.engines.raw.action_token import ActionTokenEngine
    from app.engines.raw.active_probe import ActiveProbeEngine
    from app.engines.raw.client_stack import ClientStackEngine
    from app.engines.raw.dns import DNSEngine
    from app.engines.raw.http import HTTPEngine
    from app.engines.raw.l2l3 import L2L3Engine
    from app.engines.raw.l4flow import L4FlowEngine
    from app.engines.raw.tls import TLSEngine
    reg = Registry()
    reg.add(L2L3Engine(), L4FlowEngine(), HTTPEngine(), TLSEngine(), DNSEngine(),
            ActiveProbeEngine(), ActionTokenEngine(), ClientStackEngine())
    return reg


def test_mini_pack_end_to_end():
    from app.eval.metrics import score_run
    from app.eval.report import build_report, render_html
    from app.eval.runner import run_pack
    res = run_pack("mini", seed=0, registry_factory=raw_registry)
    assert res.aborted is None and res.exceptions == []
    assert res.pack == "mini" and len(res.tick_ts) == 128
    assert [p["training"] for p in res.phases] == [True, False]
    assert res.config["tz"] == "Asia/Shanghai" and res.config["strict"] is True
    sids = {r["scenario_id"] for r in res.truth}
    assert {"T1", "T6", "T17", "L14"} <= sids
    assert len([r for r in res.truth if r["label"] == "malicious"]) >= 2
    assert set(res.personas) == set(P.MINI_KEYS)
    assert res.personas["erp-prod|10.20.1.11"]["archetype"] == "interactive"
    assert "work_start" in res.personas["erp-prod|10.20.1.11"]["params"]
    assert res.tick_ts[-1] == pytest.approx(P.get_pack("mini").end_epoch)
    sc = score_run(res)
    assert {s["scenario_id"] for s in sc["scenarios"]} == {"T1", "T6", "T17"}
    assert {s["scenario_id"] for s in sc["legit"]} >= {"L14"}
    rep = build_report([sc])
    assert rep["summary"]["n_runs"] == 1 and "<html" in render_html(rep).lower()


# --------------------------------------------------------------------------- #
# Stated perturbation magnitudes (generator.md THREAT SCENARIOS)
# --------------------------------------------------------------------------- #
def _entity_ticks(name, sid, days_before, days_after, with_scenario, seed=0):
    """Per-tick (vt, log1p bytes_up, http count, obs) of the scenario entity,
    alone, from `days_before` before the onset at the pack cadence."""
    p = P.get_pack(name)
    sc = p.scenario(sid)
    key = sc.keys()[0]
    start = p.scenario_start - days_before * 86400.0
    probe = dataclasses.replace(p, population=[key], start_epoch=start,
                                scenarios=[copy.deepcopy(sc)] if with_scenario else [])
    g = TrafficGenerator(seed=seed, pack=probe)
    rows = []
    n = int((sc.t_start - start) / p.scenario_dt + days_after * 86400.0 / p.scenario_dt)
    for _ in range(n):
        obs = g.step(p.scenario_dt)
        up = sum((o.extra or {}).get("bytes_up_total", o.bytes_up) for o in obs)
        c = sum((o.extra or {}).get("count", 1) for o in obs if o.app_proto == "http")
        rows.append((g.vt, math.log1p(up), c, obs))
    return p, sc, rows


def test_t2_low_and_slow_per_tick_z_below_2_5_on_days_1_2():
    p, sc, clean = _entity_ticks("B", "T2", 14, 2, False)
    _, _, att = _entity_ticks("B", "T2", 14, 2, True)
    clk = Clock(p.tz, p.calendar)
    ts = np.array([r[0] for r in clean])
    x0 = np.array([r[1] for r in clean])
    x1 = np.array([r[1] for r in att])
    act = np.array([r[2] > 0 for r in clean])
    hrs = np.array([clk.local(t - 450).hour for t in ts])
    work = np.array([clk.day_kind(clk.local(t - 450).date())[0] for t in ts])
    pre = ts <= sc.t_start
    zs = []
    for i in np.where((ts > sc.t_start) & (ts <= sc.t_start + 2 * 86400) & act & work)[0]:
        ref = pre & act & work & (np.abs(hrs - hrs[i]) <= 1)
        zs.append((x1[i] - x0[ref].mean()) / x0[ref].std())
    zs = np.array(zs)
    # sessions are a continuous-time process since round 2 (a working 15-min
    # tick with no session start and no carried session is idle): ~48 active
    # working ticks in the two days instead of ~70 with per-tick sessions
    assert len(zs) > 40
    assert np.mean(np.abs(zs) < 2.5) >= 0.95          # invisible to per-tick detectors
    assert np.median(zs) > 0.5                         # ... but a real, growing shift
    ups = [o for r in att for o in r[3] if o.http_host == "ext-store.example.net"]
    assert ups and all(1.5e5 < o.bytes_up < 1.6 * 2.7e5 for o in ups
                       if o.ts < sc.t_start + 86400)   # 200 KB (x1.6 on day 2)


def test_t8_enumeration_breadth_at_normal_rate():
    p, sc, att = _entity_ticks("B", "T8", 7, 1, True)
    _, _, clean = _entity_ticks("B", "T8", 7, 1, False)

    def window(rows):
        ids, cnt = set(), []
        for vt, _, c, obs in rows:
            if sc.t_start < vt <= sc.t_end:
                cnt.append(c)
                for o in obs:
                    ids.update(re.findall(r"\d+", o.http_path or ""))
        return ids, np.array(cnt)
    ia, ca = window(att)
    ib, cb = window(clean)
    assert len(ia) > 1000 and len(ib) < 80              # ~1.3 k ids vs ~40-60 a day
    assert ca.mean() < 1.3 * cb.max() and ca.max() <= 1.5 * np.quantile(cb, 0.95)
    assert ca.std() / ca.mean() < 0.5 * cb.std() / cb.mean()   # script-regular timing


def test_human_driven_onsets_are_inside_the_users_working_time():
    from app.pipeline.generator import Tick, build_human, build_persona
    p = P.get_pack("A")
    clk = Clock(p.tz, p.calendar)
    who = {"T6": "erp-prod|10.20.1.11", "T6b": "erp-prod|10.20.1.18",
           "T7": "erp-prod|10.20.1.13", "T9": "oa-portal|10.30.2.21",
           "T9b": "oa-portal|10.30.2.29", "T13": "erp-prod|10.20.1.16"}
    for sid, key in who.items():
        sc = p.scenario(sid)
        act = build_persona(key).models[0].activity(Tick(sc.t_start, 900.0, clk, 0))
        assert act.min() >= 0.8, sid
        assert (sc.t_start - p.scenario_start) % 900.0 == 0
        assert sc.t_start >= p.scenario_start + sc.truth["nominal_tick"] * 900.0
    l8 = p.scenario("L8")
    hm = build_human("erp-prod|10.20.1.14", tag=P.L8_TAG)
    assert hm.activity(Tick(l8.t_start, 900.0, clk, 0)).min() >= 0.8


def test_class_scenario_deadlines_at_pack_cadence():
    d = P.get_pack("D")
    t21 = d.scenario("T21")
    on = sorted(t21.truth["onsets"].values())
    assert on[-1] - on[0] == pytest.approx(7200.0)                 # staggered over 2 h
    assert t21.truth["max_ttd_s"] == pytest.approx(on[2] - on[0] + 4 * 3600.0)
    for sid in ("L1",):
        assert d.scenario(sid).truth["max_member_severity"] == "info"
    assert P.get_pack("C").scenario("L10").truth["max_member_severity"] == "info"


def test_t16_rare_path_is_rare_at_class_tier():
    from app.pipeline.generator import _role_peers, build_persona, class_rare_template
    key = "oa-portal|10.30.2.24"
    t = class_rare_template(key)
    me = build_persona(key).models[0]
    i = me.vocab.index(t)
    df = sum(pm.visit[i] >= 0.01 for pm in _role_peers(key))
    assert me.visit[i] < 0.002 and 1 <= df <= 2
    a, _ = _probe("A", "T16", True)
    assert any(tok.startswith(f"{t.method} {t.host} {t.fmt.split('?')[0]}") for tok in a["toks"])
