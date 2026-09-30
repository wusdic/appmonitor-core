"""P12 SystemProfile (progressive.md §6.18, §6.20; card P12): characteristics,
scenario-adaptive strategies and system families on synthetic systems."""
from __future__ import annotations

import time
from typing import Any, Dict, List

import numpy as np
import pytest

from ptree_sim import DAY, MON, Sim, daily

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.system_profile import SysTracker, SystemProfileEngine, who_level_bits
from app.models.schema import ORG, SYSTEM_ENTITY

OA_IPS = ["192.168.1.21", "192.168.1.23", "10.168.7.121", "192.168.1.30", "192.168.1.31"]
USERS = dict(zip(OA_IPS, ["jack", "rose", "mike", "anna", "bill"]))


def oa_day(system="oa", host="oa.corp", ips=OA_IPS, app="oa"):
    def fn(d, t0, r):
        out = []
        if (d % 7) >= 5:
            return out
        for ip in ips:
            t = t0 + 9 * 3600 + r.uniform(0, 1200)
            sess = f"s{d}{ip}"
            out.append((t, system, ip, {"http.route": f"POST {host} /{app}/login", "http.method": "POST",
                                        "http.host": host, "body.kv.username": USERS.get(ip, "x"),
                                        "body.len": float(r.integers(1024, 2048)), "sess.key": sess,
                                        "client.stack": f"ua-{ip}", "net.dst": f"{host}:8080"}))
            for k in range(6):
                t += r.uniform(20, 240)
                rt = f"GET {host} /{app}/page{k % 5}"
                out.append((t, system, ip, {"http.route": rt, "http.method": "GET", "http.host": host,
                                            "sess.key": sess, "client.stack": f"ua-{ip}",
                                            "net.dst": f"{host}:8080", "net.bytes_down": float(r.integers(2e3, 9e3))}))
        return out
    return fn


def portal_day(d, t0, r):
    out = []
    for k in range(250):
        ip = f"203.{int(r.integers(0, 4))}.{int(r.integers(0, 256))}.{int(r.integers(1, 255))}"
        t = t0 + r.uniform(0, DAY)
        out.append((t, "portal", ip, {"http.route": "POST portal /login", "http.method": "POST",
                                      "http.host": "portal", "body.kv.username": f"u{int(r.integers(0, 10**6))}",
                                      "sess.key": f"p{d}{k}", "client.stack": f"ua{int(r.integers(0, 50))}",
                                      "net.dst": "portal:443"}))
    return out


def tls_day(d, t0, r):
    ips = [f"192.168.5.{i}" for i in range(1, 30)]
    out = []
    for ip in ips:
        for k in range(8):
            out.append((t0 + 8 * 3600 + r.uniform(0, 10 * 3600), "mail", ip,
                        {"ev.ch": "tls", "tls.sni": "mail.corp.example.com", "net.dst": "mail:993",
                         "net.bytes_up": float(r.integers(500, 5e4)), "client.stack": "outlook"}))
    return out


def snat_day(d, t0, r):
    out = []
    for k in range(300):
        out.append((t0 + 8 * 3600 + r.uniform(0, 9 * 3600), "hr", "192.168.100.99",
                    {"http.route": f"GET hr /hr/p{k % 6}", "http.method": "GET", "http.host": "hr",
                     "sess.key": f"h{d}-{k % 40}", "client.stack": f"ua{k % 25}", "net.dst": "hr:80"}))
    return out


def run(events, days, p12=None, dt=3600.0, config=None, daily=None):
    sim = Sim(dt=dt, config=config)
    eng = p12 or SystemProfileEngine()
    sim.add(events)
    per_day = int(DAY / dt)
    for k in range(int(days * DAY / dt)):
        sim.tick()
        from helpers import ctx
        eng.safe_run(ctx(sim.st, sim.now, window_s=dt, config=sim.cfg), None)
        if daily is not None and (k + 1) % per_day == 0:
            daily(sim, (k + 1) // per_day)
    return sim, eng


EXPECTED = {"oa": {"who": ("ip", "grp"), "P07": ("on",), "P08": ("on",), "P10": ("on",)},
            "portal": {"who": ("prefix", "reg", "none")},
            "mail": {"P07": ("off",), "P08": ("off",), "content": ("off",)},
            "hr": {"who": ("none",)}}


def correct_share(sim) -> float:
    ok = n = 0
    for s, dims in EXPECTED.items():
        p = prof(sim, s)
        ch = (p or {}).get("chosen") or {}
        for dim, allowed in dims.items():
            n += 1
            ok += int(ch.get(dim) in allowed)
    return ok / n


def prof(sim, s):
    return sim.st.get_model(MP.tree_key(sim.st, s), SYSTEM_ENTITY, MP.SYSPROF)


def test_scenarios_get_the_expected_strategies():
    ev = daily(oa_day(), 7, seed=1) + daily(portal_day, 7, seed=2) + daily(tls_day, 7, seed=3) + \
        daily(snat_day, 7, seed=4)
    curve = []
    sim, eng = run(ev, 7, daily=lambda sm, d: curve.append(correct_share(sm)))
    # the strategies get more precise with observation time: no decision on day 1
    # (one full day of measurement first), all correct once the utilities and
    # the full-day SNAT evidence are in, and never worse from one day to the next
    print("P12 share of correct (system, dimension) choices by day:", [round(x, 2) for x in curve])
    assert curve[0] < 0.5 and curve[-1] == 1.0
    assert all(b >= a for a, b in zip(curve, curve[1:])), curve
    oa, portal, mail, hr = (prof(sim, s) for s in ("oa", "portal", "mail", "hr"))
    for p in (oa, portal, mail, hr):
        assert p is not None and p["chosen"]
    # departmental OA: stable IPs, forms, sessions -> who at a fine level, content and workflow on
    assert oa["chosen"]["who"] in ("ip", "grp"), oa["reasons"]
    assert oa["chosen"]["P07"] == "on" and oa["chosen"]["P08"] == "on" and oa["chosen"]["P10"] == "on"
    assert oa["characteristics"]["snat"] is False
    # random-IP portal: churn blocks per-IP who
    assert portal["characteristics"]["churn"] > 0.3
    assert portal["chosen"]["who"] in ("prefix", "reg", "none"), portal["arms"]["who"]
    # opaque TLS: no content engines, no body parsing, a hint for the view
    assert mail["chosen"]["P07"] == "off" and mail["chosen"]["P08"] == "off"
    assert mail["chosen"]["content"] == "off"
    assert any(h["kind"] == "opaque" for h in mail["hints"])
    # one source, 25 client stacks and 40 concurrent sessions behind it, no trusted proxy
    assert hr["characteristics"]["snat"] is True and hr["chosen"]["who"] == "none"
    assert any(h["kind"] == "snat" for h in hr["hints"])
    # a trusted proxy address is not a suspect
    sim2, _ = run(daily(snat_day, 3, seed=4), 3, config={
        "progressive": {"enabled": True, "trusted_proxies": ["192.168.100.99/32"]}, "tz": "Asia/Shanghai"})
    assert prof(sim2, "hr")["characteristics"]["snat"] is False
    # stable across days: at most the initial decision + 2 switches
    assert sum(1 for h in oa["history"] if h["dim"] == "who") <= 2


def _inject(st, events, t1, dt):
    by = {}
    for e in events:
        if t1 - dt < e[0] <= t1:
            by.setdefault(e[1], []).append(e)
    for s, evs in by.items():
        bb = EV.BatchBuilder(s, EV.KIND_TXN)
        for ts, _, ip, attrs in sorted(evs, key=lambda x: x[0]):
            bb.add(ts, ip, dict({"net.src": ip, "ev.ch": "http"}, **attrs), 1.0)
        batch = bb.build(t1 - dt, t1)
        batch.learn[:] = True
        st.add_batch(s, EV.EVT_BATCH, t1, batch)


def test_bindings_switch_off_where_they_do_not_pay():
    """A fitter that runs and saves nothing (gain 0, cost 20 µs/event) is
    switched off within 10 days; the P08 of the OA keeps paying and stays on.
    (P12 alone on injected batches: only its own inputs matter here.)"""
    from helpers import ctx, make_store
    ev = daily(oa_day(), 10, seed=1) + daily(portal_day, 10, seed=2)
    st = make_store()
    eng = SystemProfileEngine()
    cfg = {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}
    dt = 3600.0
    t = MON
    for _ in range(int(10 * DAY / dt)):
        t += dt
        _inject(st, ev, t, dt)
        for s, g, ms in (("portal", 0.0, 0.333), ("oa", 1.2, 0.02)):
            st.put_model(s, SYSTEM_ENTITY, MP.PBIND, {"updated": t, "gain": {"bits_per_event": g, "ms": ms}})
        eng.safe_run(ctx(st, t, window_s=dt, config=cfg), None)
    p_portal = st.get_model("portal", SYSTEM_ENTITY, MP.SYSPROF)
    p_oa = st.get_model("oa", SYSTEM_ENTITY, MP.SYSPROF)
    assert p_portal["chosen"]["P08"] == "off", p_portal["reasons"]
    assert p_oa["chosen"]["P08"] == "on"
    ev_ = [e for e in st.events() if e.kind == "strategy_changed" and e.system == "portal"]
    assert any(e.extra["dim"] == "P08" and e.extra["to"] == "off" for e in ev_)


def test_replicas_form_a_family_and_share_one_tree():
    ev = []
    for i, s in enumerate(("oa", "oa-r2", "oa-r3")):
        ev += daily(oa_day(system=s, host="oa.corp", ips=OA_IPS[:3] if i == 0 else OA_IPS[3:]), 5, seed=10 + i)
    ev += daily(oa_day(system="fin", host="fin.corp", app="fin", ips=["192.168.2.10", "192.168.2.11"]), 5,
                seed=20)
    sim, eng = run(ev, 5)
    fam = sim.st.get_model(ORG, ORG, MP.SYSFAM)
    mem = fam["member"]
    assert mem.get("oa") and mem.get("oa") == mem.get("oa-r2") == mem.get("oa-r3")
    assert "fin" not in mem
    key = mem["oa"]
    assert MP.tree_key(sim.st, "oa-r2") == key
    assert MP.get_ptree(sim.st, key) is not None               # adopted tree
    released = [s for s in ("oa", "oa-r2", "oa-r3") if MP.get_ptree(sim.st, s) is None]
    assert len(released) == 3                                  # members' own trees checkpointed and released
    assert any(sim.st.get_checkpoint(s, SYSTEM_ENTITY, MP.CHECKPOINT) for s in released)
    # net.dst member map for the dst hierarchy; one sysprof for the family, visible per member
    assert fam["dst_members"][key]
    assert prof(sim, "oa-r2") is not None and prof(sim, "oa-r2")["tree_key"] == key
    assert any(e.kind == "family_changed" for e in sim.st.events())


def test_who_level_bits_charges_ungrouped_ips():
    ev = daily(portal_day, 2, seed=2)
    sim, _ = run(ev, 2)
    tr = MP.get_ptree(sim.st, "portal").kinds[EV.KIND_TXN]
    bits, n = who_level_bits(tr, sim.now)
    assert n > 100
    # no P11 groups: the grp level cannot be cheaper than coding the address without a model
    assert bits[3] >= 30.0
    assert bits[0] > bits[2]                                  # random /32s cost more than their /16


def _tracker_cost(n_ips, n_attrs, n_events=4000, seed=0):
    r = np.random.default_rng(seed)
    tr = SysTracker("x", MON)
    b = EV.BatchBuilder("x", EV.KIND_TXN)
    extra = {f"meta.f{j}": float(j) for j in range(n_attrs)}
    for i in range(n_events):
        ip = f"10.{(i * 7919) % n_ips // 65536}.{(i * 7919) % n_ips // 256 % 256}.{(i * 7919) % n_ips % 256}"
        b.add(MON + i * 0.5, ip, dict(extra, **{"ev.ch": "http", "http.route": f"GET x /r{i % 20}",
                                                "client.stack": f"s{i % 7}", "sess.key": f"k{i % 300}",
                                                "body.len": 100.0}), 1.0)
    batch = b.build(MON, MON + n_events * 0.5)
    batch.learn[:] = True
    from collections import Counter
    from app.engines.behavior.lib.phier import Regions
    wctx = ({}, Counter(), Regions([("r1", ["10.0.0.0/8"])]), {"reg:r1": 2.0 ** 24})
    t0 = time.perf_counter()
    for k in range(5):
        tr.observe(batch, None, MON + 3600.0 * (k + 1), 20000 + k // 3, [], wctx)
    us = (time.perf_counter() - t0) * 1e6 / (5 * n_events)
    return us, tr.nbytes()


def test_tracker_memory_and_time_do_not_grow_with_ips_or_attributes():
    small = _tracker_cost(100, 0)
    many_ips = _tracker_cost(50_000, 0)
    many_attrs = _tracker_cost(100, 300)
    print("P12 tracker (us/event, bytes)", {"ips100": small, "ips50000": many_ips, "attrs300": many_attrs})
    # memory: LRU / SpaceSaving-capped per-IP state, flat in attributes
    assert many_ips[1] <= small[1] + 400_000                    # caps: LRU 256, SpaceSaving k, 8192 day counts
    assert many_attrs[1] <= 1.1 * small[1]
    # time per event: bounded by a constant (the who code costs O(distinct IPs of a tick) <= O(events)),
    # flat in the number of attributes
    assert many_ips[0] <= 120.0 and small[0] <= 40.0
    assert many_attrs[0] <= 2.0 * small[0] + 5.0


def test_who_code_dhcp_pool_prefers_prefix_and_stable_department_prefers_ip():
    """System-level who code (§6.18.2): a DHCP pool (60 users re-addressed every
    day inside a /22) is coded cheaper at the prefix level than per IP; a stable
    department of 30 IPs spread over 30 /24s is cheapest per IP."""
    from collections import Counter
    from app.engines.behavior.lib.phier import Regions
    from app.engines.behavior.system_profile import WhoCode
    r = np.random.default_rng(3)
    wctx = ({}, Counter(), Regions([]), {})
    pool, dept = WhoCode(), WhoCode()
    dept_ips = [f"10.{i}.0.5" for i in range(30)]
    for d in range(14):
        leases = r.choice(1024, 60, replace=False)
        for h in range(8):
            t = MON + d * DAY + (9 + h) * 3600.0
            pool.observe({f"10.50.{int(x) // 256}.{int(x) % 256}": 3.0 for x in leases}, t, 20000 + d, *wctx)
            dept.observe({ip: 3.0 for ip in dept_ips}, t, 20000 + d, *wctx)
    pb, _ = pool.bits()
    db, _ = dept.bits()
    assert min(pb[1], pb[2]) < pb[0], pb
    assert db[0] < min(db[1], db[2]), db
