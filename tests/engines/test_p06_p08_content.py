"""P06 ContentBounds, P07 PayloadGrammar, P08 Binding (behavior.content_bounds /
payload_grammar / binding; docs/lib3/progressive.md §6.10-§6.12, cards P06-P08).

The pattern tree is maintained by tests/pcontent_oracle.OracleLearner, a
stand-in for P04 (routing, node summaries, requested pair sketches); every
constraint asserted here is fitted by the engines under test."""
from __future__ import annotations

import random
import time

import numpy as np
import pytest

from helpers import set_trust
from pcontent_oracle import (OracleLearner, batch_of, ga_login_events, ga_login_row, org_truth,
                             statement)

from app.core.engine import Context
from app.core.store import MetricStore
from app.engines.behavior.binding import STATE, BindingEngine
from app.engines.behavior.content_bounds import ContentBoundsEngine
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.payload_grammar import PayloadGrammarEngine
from app.eval import pmetrics as PMX

CFG = {"progressive": {"enabled": True}, "grain_mode": "tick", "strict": True}
DAY = 86400.0
T0 = 1_788_220_800.0          # 2026-09-01 00:00 +08:00 (a Tuesday)


def _ctx(store, now, cfg=CFG):
    return Context(store=store, now=now, window_s=60, config=dict(cfg))


def _engines():
    return [ContentBoundsEngine(), PayloadGrammarEngine(), BindingEngine()]


def _day(store, orc, engines, system, rows_by_route, d, cfg=CFG, trust=None):
    """One learning day: a batch per route at noon, the oracle learns it,
    then the three engines run one hour later (past the learning delay)."""
    T = T0 + d * DAY + 12 * 3600
    bb = EV.BatchBuilder(system)
    i = 0
    for route, rows in rows_by_route.items():
        for ip, attrs in rows:
            a = {"http.route": route}
            a.update(attrs)
            bb.add(T - 3600 + i * 7.0, ip, a, 1.0)
            i += 1
    if i:
        b = bb.build(T - 43200, T)
        store.add_batch(system, EV.EVT_BATCH, T, b)
        orc.learn(b, trust=trust)
    for e in engines:
        e.safe_run(_ctx(store, T + 3600, cfg))
    return T + 3600


# =================================================================== P06/P07
def test_p06_p07_fit_every_node_and_skip_clean_ones():
    store = MetricStore()
    orc = OracleLearner(store, "oa", T0, ["POST /login", "GET /home"], config=CFG)
    engs = _engines()
    r = np.random.default_rng(0)
    for d in range(12):
        rows = {"POST /login": [(ip, {"body.len": float(r.uniform(1024, 2048)),
                                      "body.kv.username": u,
                                      "body.keys": frozenset({"username", "password"})})
                                for ip, u in (("192.168.1.21", "jack"), ("192.168.1.23", "rose"))],
                "GET /home": [("192.168.1.21", {"q.kv.page": "1"})]}
        now = _day(store, orc, engs, "oa", rows, d)
    pb = MP.get_model(store, "oa", MP.PBOUNDS)
    pg = MP.get_model(store, "oa", MP.PGRAMMAR)
    login = orc.node_for("POST /login")
    home = orc.node_for("GET /home")
    rec = pb["nodes"][0][login]["attrs"]["body.len"]
    assert 1000 <= rec["band90"][0] < rec["band90"][1] <= 2100 and rec["n_rng"] == pytest.approx(24)
    assert pb["nodes"][0][home]["status"] == "none"            # P06 declares "none applicable"
    g = pg["nodes"][0][login]["attrs"]
    assert g["body.kv.username"]["grammar"] == "[a-z]{4}"
    assert g["body.keys"]["required"] == ["password", "username"]
    assert pg["nodes"][0][home]["attrs"]["q.kv.page"]["grammar"] == "[0-9]"
    # nothing new -> nothing refitted, and the hourly engine is not due twice in an hour
    e6 = engs[0]
    assert e6.run(_ctx(store, now + 1800)) == 0
    assert pb["gain"]["bits_per_event"] >= 0.0


def test_p07_arm_off_marks_nodes_off():
    store = MetricStore()
    orc = OracleLearner(store, "mail", T0, ["TLS mail"], config=CFG)
    store.put_model("mail", "__system__", MP.SYSPROF, {"chosen": {"p07": "off", "p08": "off"}})
    engs = _engines()
    for d in range(3):
        _day(store, orc, engs, "mail", {"TLS mail": [("10.0.0.1", {"body.kv.x": "abc"})]}, d)
    pg = MP.get_model(store, "mail", MP.PGRAMMAR)
    assert pg["applicable"] is False
    assert all(e["status"] == "off" for e in pg["nodes"][0].values())
    assert MP.get_model(store, "mail", MP.PBIND) is None


def test_inert_when_disabled():
    store = MetricStore()
    OracleLearner(store, "oa", T0, ["POST /login"], config=CFG)
    for e in _engines():
        assert e.run(_ctx(store, T0 + DAY, {"grain_mode": "tick"})) == 0
    assert MP.get_model(store, "oa", MP.PBOUNDS) is None


# ======================================================================= P08
def _p08_world(d, rnd):
    ga = [("192.168.1.21", "jack"), ("192.168.1.23", "rose"), ("10.168.7.121", "mike")]
    login = [(ip, {"body.kv.username": u, "body.kv.captcha": str(rnd.randint(1000, 9999))})
             for ip, u in ga]
    login += [("192.168.5.7", {"body.kv.username": ["amy", "bob"][k % 2]}) for k in range(2)]
    portal = [(f"10.70.{rnd.randint(0, 3)}.{rnd.randint(1, 200)}",
               {"body.kv.username": "u%05d" % rnd.randint(0, 99999)}) for _ in range(40)]
    backup = [(h, {"body.kv.username": "svc_backup"}) for h in ("10.1.1.5", "10.1.1.6", "10.1.1.7")]
    return {"POST /login": login, "POST /portal/login": portal, "POST /backup": backup}


def test_p08_screen_request_and_fit_bindings():
    store = MetricStore()
    orc = OracleLearner(store, "oa", T0, ["POST /login", "POST /portal/login", "POST /backup"],
                        config=CFG)
    engs = _engines()
    rnd = random.Random(0)
    for d in range(16):
        _day(store, orc, engs, "oa", _p08_world(d, rnd), d)
    want = MP.get_model(store, "oa", MP.PWANT)["pairs"]
    specs = {(s["x"], s["y"], s["dir"]) for s in want["specs"]}
    assert ("net.src", "body.kv.username", "fwd") in specs
    assert ("body.kv.username", "net.src", "rev") in specs
    assert not any("captcha" in s[0] or "captcha" in s[1] for s in specs)
    pb = MP.get_model(store, "oa", MP.PBIND)["nodes"][0]
    login = pb[orc.node_for("POST /login")]["pairs"]["net.src->body.kv.username"]
    assert login["fd"]["holds"]
    tab = login["table"]
    assert {x: tab[x]["top"] for x in ("192.168.1.21", "192.168.1.23", "10.168.7.121")} == \
        {"192.168.1.21": "jack", "192.168.1.23": "rose", "10.168.7.121": "mike"}
    assert all(tab[x]["bound"] and tab[x]["LB"] >= 0.8 for x in ("192.168.1.21", "10.168.7.121"))
    assert tab["192.168.5.7"]["set"] == ["amy", "bob"]              # shared terminal
    assert "first" in tab["192.168.1.21"] and tab["192.168.1.21"]["last"] > tab["192.168.1.21"]["first"]
    portal = pb.get(orc.node_for("POST /portal/login"), {}).get("pairs", {})
    for rec in portal.values():                                      # portal: no binding pays
        assert not rec["fd"]["holds"] and not any(e.get("bound") for e in rec["table"].values())
    backup = pb[orc.node_for("POST /backup")]["pairs"]["body.kv.username->net.src"]
    assert backup["table"]["svc_backup"]["set"] == ["10.1.1.5", "10.1.1.6", "10.1.1.7"]
    st = MP.get_model(store, "oa", STATE)
    assert st["probe"].n_rows() <= 64 * 64


def _rebind_run(quarantine: bool):
    store = MetricStore()
    orc = OracleLearner(store, "oa", T0, ["POST /login"], config=CFG)
    engs = _engines()
    rnd = random.Random(1)
    ga = [("192.168.1.21", "jack"), ("192.168.1.23", "rose"), ("10.168.7.121", "mike")]
    for d in range(12):
        _day(store, orc, engs, "oa", {"POST /login": [(ip, {"body.kv.username": u}) for ip, u in ga]}, d)
    for d in range(12, 17):
        rows = [(ip, {"body.kv.username": u}) for ip, u in ga[:2]]
        rows += [("10.168.7.121", {"body.kv.username": "mike.w"})] * 2
        if quarantine:
            set_trust(store, "oa", "10.168.7.121", [T0 + d * DAY + 11 * 3600], 0.0, quarantine=1.0)
        _day(store, orc, engs, "oa", {"POST /login": rows}, d)
    nid = orc.node_for("POST /login")
    rec = MP.get_model(store, "oa", MP.PBIND)["nodes"][0][nid]["pairs"]["net.src->body.kv.username"]
    evs = [e for e in store.events() if e.kind == "binding_changed"]
    return rec, evs


def test_p08_rebinding_trusted_and_not_from_quarantined_ip():
    rec, evs = _rebind_run(False)
    e = rec["table"]["10.168.7.121"]
    assert e["top"] == "mike.w" and e.get("bound") and e["rebound"]["from"] == "mike"
    assert rec["table"]["192.168.1.21"]["top"] == "jack"
    assert any(ev.extra.get("new") == "mike.w" for ev in evs)
    rec_q, evs_q = _rebind_run(True)
    assert rec_q["table"]["10.168.7.121"].get("rebound") is None
    assert not any(ev.extra.get("new") == "mike.w" for ev in evs_q)


# ========================================================== convergence (PG2)
CONV_DAYS = (3, 5, 7, 10, 14, 21, 28, 35)


def _converge(seed: int):
    gen, pt = org_truth(seed)
    row = ga_login_row(pt)
    store = MetricStore()
    orc = OracleLearner(store, "oa", gen.day_start(1), ["POST /login"],
                        {"GA": list(row["gen"]["members"])}, config=CFG)
    engs = _engines()
    evs = ga_login_events(row, gen, range(1, max(CONV_DAYS) + 1), seed=seed)
    nid = orc.node_for("POST /login", "GA")
    r = np.random.default_rng(1000 + seed)
    out = []
    for d in range(1, max(CONV_DAYS) + 1):
        de = [e for dd, e in evs if dd == d]
        T = gen.day_start(d) + 12 * 3600
        if de:
            b = batch_of("oa", de, "POST /login", T - 43200, T)
            store.add_batch("oa", EV.EVT_BATCH, T, b)
            orc.learn(b)
        for e in engs:
            e.safe_run(_ctx(store, T + 3600))
        if d in CONV_DAYS:
            s = statement(store, "oa", nid, "POST /login")
            ls = PMX.LStmt(s, "oa", {})
            _, det = PMX.content_matches(row, ls, r)
            _, hit, _ = PMX.bindings_match(row, ls)
            hold = PMX.holdout_check(ls, [row], pt, r, n=400)
            c = s["evidence"]["content"]
            lbs = [v for b in s["evidence"]["bindings"].values() for v in b["LB"].values()]
            out.append({"day": d, "det": det, "bind": hit, "hold": hold["ok"],
                        "n_constraints": len(hold["constraints"]),
                        "band_cov": c["body.len"]["coverage"], "range_cover": c["body.len"]["cover"],
                        "lb": min(lbs) if lbs else 0.0, "U_s": c["body.kv.username"]["U_s"],
                        "closed": c["body.kv.username"].get("closed")})
    return out


@pytest.mark.parametrize("seed", [0, 1, 2])
def test_precision_rises_with_observation_time_on_the_requirement_example(seed):
    """综合部 GA logging into OA (pack O's truth program). Every published
    constraint holds on held-out truth events at its stated confidence at every
    snapshot (calibrated), and the stated precision tightens with time: the
    range's exceedance bound and the grammar's unseen-shape mass fall, the
    band's coverage bound and the bindings' lower bounds rise, and more truth
    constraints are recovered (bindings 3/3 by day 14, the closed username set
    once n_c >= 20 and U <= 0.02 on the confidence channel)."""
    out = _converge(seed)
    assert all(o["hold"] for o in out), [(o["day"], o["hold"]) for o in out]
    cover = [o["range_cover"] for o in out]
    assert all(b < a for a, b in zip(cover, cover[1:]))
    us = [o["U_s"] for o in out]
    assert all(b < a for a, b in zip(us, us[1:]))
    cov = [o["band_cov"] for o in out]
    # closed-band coverage (pbounds.closed_coverage) counts the endpoint point
    # masses, so the day-3 bound (a handful of values, all inside their own
    # closed 5-95 % band) starts high and dips once the sample shows its spread;
    # from there it rises with observation time (measured 0.61 -> 0.73)
    assert cov[-1] >= cov[0] and cov[-1] > min(cov) + 0.05
    lbs = [o["lb"] for o in out]
    assert all(b >= a - 1e-9 for a, b in zip(lbs, lbs[1:]))
    by = {o["day"]: o for o in out}
    assert by[14]["bind"] == 3 and by[3]["bind"] == 0
    assert all(o["det"]["body.keys"] for o in out)
    # the closed username set (U <= 0.02 at n_c >= 20, the MEDIUM who-closed level
    # and n_conf) appears once the confidence channel holds ~24 units: after day 10
    # (spec thresholds 0.01 / 50 would need ~day 24, PG1 asks at day 14)
    assert by[10]["closed"] is None and by[14]["closed"] == ["jack", "mike", "rose"]
    assert all(o["det"]["body.kv.username"] for o in out if o["day"] >= 14)
    nc = [o["n_constraints"] for o in out]
    assert all(b >= a for a, b in zip(nc, nc[1:])) and nc[-1] > nc[0]


# =========================================================== resource bound
def _deep(store, key):
    from app.eval.pscale import deep_sizeof
    seen: set = set()
    return sum(deep_sizeof(store.get_model(key, "__system__", n), seen)
               for n in (MP.PBOUNDS, MP.PGRAMMAR, MP.PBIND, STATE, MP.PWANT))


def _scale_point(n_ip: int, n_attr: int, days: int = 4, per_day: int = 750, seed: int = 0):
    rnd = random.Random(seed)
    store = MetricStore()
    routes = [f"POST /r{k}" for k in range(5)]
    orc = OracleLearner(store, "portal", T0, routes, config=CFG,
                        target_filter=lambda a: a.startswith(("body.", "meta.")))
    engs = _engines()
    cpu = 0.0
    n_ev = 0
    for d in range(days):
        T = T0 + d * DAY + 12 * 3600
        bb = EV.BatchBuilder("portal")
        for i in range(per_day):
            ip = f"10.{(i * 7 + d) % 97}.{rnd.randrange(n_ip) // 250}.{rnd.randrange(250)}"
            a = {"http.route": routes[i % 5], "body.len": float(rnd.randint(200, 4000)),
                 "body.kv.user": "u%06d" % rnd.randrange(10 ** 6),
                 "body.keys": frozenset({"user", "pw"})}
            for k in range(n_attr):
                a[f"meta.f{k:03d}"] = "v%d" % rnd.randrange(8) if k % 2 else float(rnd.random())
            bb.add(T - 3600 + i, ip, a, 1.0)
        b = bb.build(T - 43200, T)
        store.add_batch("portal", EV.EVT_BATCH, T, b)
        orc.learn(b)
        c = _ctx(store, T + 3600)
        t0 = time.perf_counter()
        for e in engs:
            e.safe_run(c)
        cpu += time.perf_counter() - t0
        n_ev += per_day
    return _deep(store, "portal"), cpu / n_ev * 1e6


def test_resources_bounded_in_ips_and_attributes():
    """The fitters' state and CPU per event do not grow with the number of
    client IPs or of attributes (§7.2; PG4 axes): memory is bounded by the
    tree's nodes x m_t targets, the probe (R_total = 4096 rows of <= 24 columns),
    <= 32 pair-nodes x 64 tracked sources, and Y_MAX screened attributes."""
    from app.eval.pmetrics import loglog_slope
    ips = (200, 2000, 20000)
    mem_i, cpu_i = zip(*[_scale_point(n, 10, days=3, per_day=300) for n in ips])
    attrs = (30, 100, 200)
    mem_a, cpu_a = zip(*[_scale_point(2000, a, days=3, per_day=300) for a in attrs])
    for m in mem_i + mem_a:
        assert m < 4 * 2 ** 20                                       # < 4 MB per tree
    for c in cpu_i + cpu_a:
        assert c < 2000                                              # < 2 ms per event, all three
    # IPs: flat (measured 0.72 / 0.72 / 0.72 MB at 200 / 2 000 / 20 000 IPs)
    assert loglog_slope(ips, mem_i) <= 0.15, mem_i
    assert loglog_slope(ips, cpu_i) <= 0.3, cpu_i
    # attributes: grows only until the caps bind (m_t targets per node, 24 probe
    # columns, Y_MAX screened names), then flat (measured 1.24 / 1.76 / 1.76 MB)
    assert mem_a[2] <= 1.1 * mem_a[1], mem_a
    assert cpu_a[2] <= 1.4 * cpu_a[1], cpu_a


# ================================================= content target requests
def test_fitters_request_content_targets_that_p05_found_informative():
    """P06 / P07 ask P04 (model.pwant) for the content attributes P05 ranks as
    informative but that the node does not model (pack O's real lattice made
    the login body size a split candidate only): numeric for P06, key sets /
    text for P07; nothing P05 dropped; a request the node never fills is given
    up after 5 evidence units, and a filled one keeps being requested."""
    from app.engines.behavior.lib import pbounds as PB
    store = MetricStore()
    orc = OracleLearner(store, "oa", T0, ["POST /login", "GET /home"], config=CFG,
                        target_filter=lambda a: a == "net.bytes_up")
    r = np.random.default_rng(0)
    for d in range(4):
        _day(store, orc, [], "oa", {
            "POST /login": [(ip, {"body.len": float(r.uniform(1024, 2048)), "net.bytes_up": 1500.0,
                                  "meta.noise": float(r.random()),
                                  "body.keys": frozenset({"username", "password"}),
                                  "body.kv.username": u})
                            for ip, u in (("192.168.1.21", "jack"), ("192.168.1.23", "rose"))] * 5,
            "GET /home": [("192.168.1.21", {"net.bytes_up": 300.0})]}, d)
    store.put_model("oa", "__system__", MP.ATTRSEL, {
        "roles": {"body.len": "split", "meta.noise": "dropped", "body.keys": "split",
                  "body.kv.username": "target", "net.bytes_up": "target"},
        "split_cands": {0: [("meta.noise", 1), ("body.len", 1), ("body.keys", 0)]},
        "targets_sys": {0: ["net.bytes_up", "body.kv.username"]}})
    reg = MP.get_registry(store, "oa")
    for a in ("body.len", "meta.noise"):
        reg.get(a).type = "numeric"                 # (few events: P02 still says ordinal)
    ptm = MP.get_ptree(store, "oa")
    book: dict = {}
    t = T0 + 5 * DAY
    num = PB.request_targets(store, "oa", ptm, t, ("numeric",), book)
    login, home = orc.node_for("POST /login"), orc.node_for("GET /home")
    assert num[0][login] == ["body.len"] and "meta.noise" not in sum(num[0].values(), [])
    txt = PB.request_targets(store, "oa", ptm, t, ("set", "text"), {})
    assert txt[0][login] == ["body.keys"]            # username: a system target P04 keeps anyway
    # GET /home never carries body.len: after 5 more evidence units the request is dropped
    for d in range(5, 12):
        _day(store, orc, [], "oa", {"GET /home": [("192.168.1.21", {"net.bytes_up": 300.0})]}, d)
    orc.tree.nodes[login].target("body.len", "num", log=True)      # P04 filled the request
    for a in ("body.len", "meta.noise"):
        reg.get(a).type = "numeric"
    num2 = PB.request_targets(store, "oa", ptm, T0 + 13 * DAY, ("numeric",), book)
    assert home not in num2.get(0, {}) and num2[0][login] == ["body.len"]
