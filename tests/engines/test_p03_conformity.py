"""P03 ConformityEngine (behavior.conformity; docs/lib3/progressive.md §6.16,
card P03 tests (a)-(i)), plus the resource bound and the convergence of its
evidence with observation time."""
from __future__ import annotations

import datetime as _dt
import math
import time
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from helpers import ctx, make_store

from app.engines.behavior import conformity as CF
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pbounds as PB
from app.engines.behavior.lib import pscore as SC
from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import ptree as PT
from app.models.schema import ORG, SYSTEM_ENTITY, Severity

TZ = _dt.timezone(_dt.timedelta(hours=8))
T0 = _dt.datetime(2026, 9, 1, tzinfo=TZ).timestamp()          # Tuesday 00:00 local
DAY = 86400.0
CFG = {"progressive": {"enabled": True}, "tz": "Asia/Shanghai",
       "sensitive_patterns": [r"^/admin(/|$)"]}
GA = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
FIN = ["192.168.2.10", "192.168.2.11", "192.168.2.12"]
APPROVE = "POST fin /fin/approval/{num}/approve"
LOGIN = "POST oa /login"


def workdays(n: int, start: float = T0) -> List[float]:
    out, d = [], 0
    while len(out) < n:
        t = start + d * DAY
        if _dt.datetime.fromtimestamp(t, TZ).weekday() < 5:
            out.append(t)
        d += 1
    return out


def local_minute(ts: float) -> float:
    return ((ts + 8 * 3600) % DAY) / 60.0


class Fx:
    """A store holding pattern trees built by hand (the P04 summaries), the
    fitted models a test needs and P11's groups."""

    def __init__(self) -> None:
        self.st = make_store()
        self.st.put_model(ORG, ORG, MP.WHO_GROUPS, {
            "groups": {"G1": {"id": "G1", "name": "综合部", "members": GA},
                       "G2": {"id": "G2", "name": "财务部", "members": FIN}},
            "ip2g": dict({ip: "G1" for ip in GA}, **{ip: "G2" for ip in FIN})})
        self.eng = CF.ConformityEngine()
        self.now = T0
        self.t_eng = 0.0

    def tree(self, key: str, routes: Sequence[str]) -> PT.Tree:
        m = MP.get_ptree(self.st, key)
        if m is None:
            m = PT.PTreeModel(key)
            self.st.put_model(key, SYSTEM_ENTITY, MP.PTREE, m, version=1)
        tr = m.tree(EV.KIND_TXN, T0, create=True)
        if tr.nodes[tr.root].split is None:
            tr.split(tr.root, "http.route", 0, [[r] for r in routes], T0)
        return tr

    def node(self, key: str, route: str) -> PN.Node:
        tr = self.tree(key, [route])
        sp = tr.nodes[tr.root].split
        return tr.nodes[sp.child_for(route)]

    def learn(self, key: str, route: str, ip: str, ts: float, targets: Optional[Dict[str, Any]] = None) -> None:
        tr = self.tree(key, [route])
        hier = MP.hierarchies(self.st, key, CFG)
        keys = [hier.gen("net.src", l, ip) for l in range(PN.WHO_LEVELS)]
        day = int((ts + 8 * 3600) // DAY)
        wd = 0 if _dt.datetime.fromtimestamp(ts, TZ).weekday() < 5 else 1
        for nd in (tr.nodes[tr.root], self.node(key, route)):
            nd.update_core(ts, 1.0, 1.0, keys, ip, wd, local_minute(ts), day)
            for a, v in (targets or {}).items():
                kind = "num" if isinstance(v, float) else "cat"
                nd.update_target(a, v, ts, 1.0, 1.0, kind=kind, day=day)
        for nd in tr.nodes.values():
            if nd.state == "candidate" and nd.n_c(ts) >= 1:
                nd.state = "confirmed"

    def put(self, key: str, name: str, nid: int, attrs: Optional[Dict[str, Any]] = None,
            pairs: Optional[Dict[str, Any]] = None) -> None:
        m = self.st.get_model(key, SYSTEM_ENTITY, name) or {"fmt": 1, "nodes": {}}
        ent: Dict[str, Any] = {"status": "fitted"}
        if attrs is not None:
            ent["attrs"] = attrs
        if pairs is not None:
            ent["pairs"] = pairs
        m["nodes"].setdefault(0, {})[nid] = ent
        self.st.put_model(key, SYSTEM_ENTITY, name, m)

    def score(self, system: str, events: Sequence[Tuple[float, str, Dict[str, Any]]],
              dt: float = 3600.0) -> Tuple[EV.EventBatch, EV.EventBatch]:
        t1 = max(e[0] for e in events) + 1.0
        b = EV.BatchBuilder(system, EV.KIND_TXN)
        for ts, ip, attrs in sorted(events, key=lambda e: e[0]):
            wd = _dt.datetime.fromtimestamp(ts, TZ).weekday() < 5
            a = {"net.src": ip, "ctx.tod_min": local_minute(ts),
                 "ctx.daytype": "workday" if wd else "nonworkday"}
            a.update(attrs)
            b.add(ts, ip, a)
        batch = b.build(t1 - dt, t1)
        self.st.add_batch(system, EV.EVT_BATCH, t1, batch)
        self.now = t1
        x = time.perf_counter()
        self.eng.safe_run(ctx(self.st, t1, window_s=dt, config=CFG), None)
        self.t_eng += time.perf_counter() - x
        return batch, self.st.batch_at(system, EV.PAT_ASSIGN, t1)

    def violations(self, ip: Optional[str] = None) -> List[Any]:
        return [e for e in self.st.events(kinds=("pattern_violation",), limit=10000)
                if ip is None or e.entity == ip]


def finance_fixture(days: int = 21, per_day: int = 10) -> Tuple[Fx, int]:
    fx = Fx()
    for d in workdays(days):
        for k in range(per_day):
            ts = d + (10 * 60 + 5 * k) * 60.0 - 8 * 3600.0 + 8 * 3600.0
            fx.learn("finance", APPROVE, "192.168.2.10", ts)
        for ip in ("192.168.2.11", "192.168.2.12"):
            fx.learn("finance", "GET fin /fin/ledger", ip, d + 11 * 3600.0)
    nid = fx.node("finance", APPROVE).id
    return fx, nid


def _rate_history(fx: "Fx", route: str, n: int = 20000, t: float = T0) -> None:
    """n finished one-login IP-hours at the route's node in P03's own tally
    (the reference an hour count is ranked against, conformity.RateTally)."""
    st = fx.st.get_model(ORG, ORG, CF.STATE)
    if not isinstance(st, CF.ConfState):
        st = CF.ConfState()
    nd = fx.node("portal", route)
    st.tally().close_hour(MP.tree_key(fx.st, "portal"), [(0, nd.id, f"h{i}", 1.0) for i in range(n)], t)
    fx.st.put_model(ORG, ORG, CF.STATE, st)


def _approve(ip: str, ts: float, user: str = "") -> Tuple[float, str, Dict[str, Any]]:
    a = {"http.route": APPROVE, "http.method": "POST"}
    if user:
        a["body.kv.username"] = user
    return ts, ip, a


# ======================================================================== (a)
def test_a_outsider_group_high_candidate_on_closed_write_node():
    fx, nid = finance_fixture()
    nd = fx.node("finance", APPROVE)
    t = fx.now = workdays(22)[-1] + 10.5 * 3600
    assert nd.n_c(t) >= 150
    l, mem, U = CF._who_closed(nd, t)
    assert l == 0 and mem == {"192.168.2.10"} and U <= 0.005
    b, asg = fx.score("finance", [_approve("192.168.1.23", t)])
    assert asg.get("p_who", 0) <= 0.005
    v = fx.violations("192.168.1.23")
    who = [e for e in v if e.extra["type"] == "who"]
    assert who and who[0].severity == Severity.HIGH and who[0].extra["high_candidate"]
    assert "outsider_group" in who[0].extra["flags"] and who[0].extra["sensitivity"] == 2.5
    assert "system_new" in who[0].extra["flags"] and "lateral" in who[0].axes
    assert who[0].dedupe_key.startswith(f"pv|{nid}|who|192.168.1.23|")
    # the member itself is never flagged
    b2, asg2 = fx.score("finance", [_approve("192.168.2.10", t + 60)])
    assert asg2.get("p_who", 0) == 1.0 and not fx.violations("192.168.2.10")


def test_a_unknown_ip_medium_readdress_low_concurrent():
    fx, nid = finance_fixture()
    rev = {"x": "body.kv.username", "y": "net.src", "dir": "rev",
           "fd": {"g3": 0.0, "holds": True},
           "table": {"lucy": {"top": "192.168.2.10", "bound": True, "n": 150.0, "k": 150.0,
                              "LB": 0.97, "p_viol": 0.006}}}
    fx.put("finance", MP.PBIND, nid, pairs={"body.kv.username->net.src": rev})
    days = workdays(25)
    # .10 last active 2 days before the readdress
    fx.score("finance", [_approve("192.168.2.10", days[21] + 10 * 3600, "lucy")])
    t = days[23] + 10 * 3600
    b, asg = fx.score("finance", [_approve("192.168.2.51", t, "lucy")])
    v = fx.violations("192.168.2.51")
    who = [e for e in v if e.extra["type"] == "who"][0]
    assert "unknown_ip" in who.extra["flags"] and "readdress_candidate" in who.extra["flags"]
    assert who.severity == Severity.LOW
    assert asg.get("readdr", 0) == "192.168.2.10"
    # a never-seen IP without the binding hint: MEDIUM
    fx.score("finance", [_approve("192.168.2.77", t + 120)])
    w77 = [e for e in fx.violations("192.168.2.77") if e.extra["type"] == "who"][0]
    assert "unknown_ip" in w77.extra["flags"] and w77.severity == Severity.MEDIUM
    # concurrent use: lucy's own IP active within the hour
    t2 = days[24] + 10 * 3600
    fx.score("finance", [_approve("192.168.2.10", t2, "lucy"), _approve("192.168.2.99", t2 + 600, "lucy")])
    v99 = fx.violations("192.168.2.99")
    cont = [e for e in v99 if e.extra["type"] == "content"]
    assert cont and "concurrent_use" in cont[0].extra["flags"]
    assert "credential" in cont[0].axes
    assert Severity.MEDIUM in {e.severity for e in v99} or Severity.HIGH in {e.severity for e in v99}
    assert CF.SEV_RANK[cont[0].severity] >= CF.SEV_RANK[Severity.MEDIUM]


# ======================================================================== (b)
def test_b_open_population_has_no_who_constraint():
    fx = Fx()
    rng = np.random.default_rng(0)
    for d in workdays(10):
        for i in range(60):
            ip = f"{rng.integers(1, 223)}.{rng.integers(0, 255)}.{rng.integers(0, 255)}.{rng.integers(1, 255)}"
            fx.learn("portal", "POST portal /login", ip, d + (8 + 10 * rng.random()) * 3600)
    t = workdays(11)[-1] + 12 * 3600
    b, asg = fx.score("portal", [(t, "100.61.1.1", {"http.route": "POST portal /login"})])
    assert math.isnan(asg.get("p_who", 0))
    assert not fx.violations()


# ======================================================================== (c)
def test_c_cross_binding_is_a_credential_finding():
    fx = Fx()
    users = dict(zip(GA, ["jack", "rose", "mike"]))
    for d in workdays(15):
        for k, ip in enumerate(GA):
            fx.learn("oa", LOGIN, ip, d + (9 * 60 + 3 * k) * 60.0)
    nd = fx.node("oa", LOGIN)
    table = {ip: {"n": 15.0, "top": u, "k": 15.0, "bound": True, "LB": 0.95, "p_viol": 0.02}
             for ip, u in users.items()}
    fx.put("oa", MP.PBIND, nd.id, pairs={"net.src->body.kv.username": {
        "x": "net.src", "y": "body.kv.username", "dir": "fwd", "fd": {"g3": 0.0, "holds": True},
        "table": table, "bound_values": {u: [ip] for ip, u in users.items()}}})
    t = workdays(16)[-1] + 9 * 3600 + 600
    _, asg = fx.score("oa", [(t, "192.168.1.21", {"http.route": LOGIN, "body.kv.username": "rose"})])
    v = [e for e in fx.violations("192.168.1.21") if e.extra["type"] == "content"]
    assert v and "cross_binding" in v[0].extra["flags"] and v[0].axes == ["credential"]
    assert v[0].severity == Severity.MEDIUM
    # a borrowed credential is learned damped (integration fix, A2 non-adoption)
    assert asg.get("damp", 0) == pytest.approx(0.1)
    # the bound value itself is clean and learned at full weight
    _, asg = fx.score("oa", [(t + 60, "192.168.1.23", {"http.route": LOGIN, "body.kv.username": "rose"})])
    assert not [e for e in fx.violations("192.168.1.23") if e.extra["type"] == "content"]
    assert asg.get("damp", 0) == 1.0
    # a value bound nowhere (a rename such as D2's mike -> mike.w) is not damped as a borrowed one
    _, asg = fx.score("oa", [(t + 120, "10.168.7.121", {"http.route": LOGIN, "body.kv.username": "mike.w"})])
    assert asg.get("damp", 0) == 1.0 or asg.get("p_ev", 0) <= 1e-4


# ======================================================================== (d)
def test_d_login_at_0305_against_a_learned_window():
    fx = Fx()
    rng = np.random.default_rng(1)
    days = workdays(50)
    for d in days[:45]:
        for ip in GA:
            fx.learn("oa", LOGIN, ip, d + (9 * 60 + 21 * rng.random()) * 60.0)
    nd = fx.node("oa", LOGIN)
    fx.put("oa", MP.PWIN, nd.id, attrs={})
    m = fx.st.get_model("oa", SYSTEM_ENTITY, MP.PWIN)
    m["nodes"][0][nd.id]["when"] = {"workday": [[540, 561]], "nonworkday": []}
    # the node's ordinary events are scored first (they also calibrate its p-values)
    for d in days[45:48]:
        fx.score("oa", [(d + (9 * 60 + 21 * rng.random()) * 60.0, ip, {"http.route": LOGIN}) for ip in GA])
    assert not fx.violations()
    t = days[48] + 3 * 3600 + 5 * 60                              # 03:05
    assert nd.when.evidence(0, t) >= 60
    b, asg = fx.score("oa", [(t, "10.168.7.121", {"http.route": LOGIN})])
    assert asg.get("p_when", 0) <= 1e-3
    w = [e for e in fx.violations("10.168.7.121") if e.extra["type"] == "when"]
    assert w and w[0].severity == Severity.MEDIUM                # a write at night
    assert "03:05" in w[0].description
    # a login inside the window is clean
    fx.score("oa", [(days[49] + 9 * 3600 + 600, "192.168.1.21", {"http.route": LOGIN})])
    assert not [e for e in fx.violations("192.168.1.21") if e.extra["type"] == "when"]


# ======================================================================== (e)
def test_e_missing_required_predecessor():
    from app.engines.behavior.workflow import WorkflowEngine
    fx = Fx()
    form, gen = "GET oa /report/form", "POST oa /report/generate"
    for d in workdays(20):
        for ip in GA[1:]:
            fx.learn("oa", gen, ip, d + 17 * 3600 + 120)
    fx.tree("oa", [gen, form])
    p10 = WorkflowEngine(mine_period_s=6 * 3600.0)
    days = workdays(21)
    now = T0
    for d in days[:20]:
        for h in range(24):
            t1 = d + (h + 1) * 3600.0
            if h == 17:
                b = EV.BatchBuilder("oa", EV.KIND_TXN)
                for ip in GA[1:]:
                    b.add(d + 17 * 3600 + 10, ip, {"http.route": form, "net.src": ip})
                    b.add(d + 17 * 3600 + 120, ip, {"http.route": gen, "net.src": ip})
                fx.st.add_batch("oa", EV.EVT_BATCH, t1, b.build(t1 - 3600, t1))
            fx.st.ensure_retention("evt.", max_age_s=6 * 3600.0)
            p10.safe_run(ctx(fx.st, t1, window_s=3600.0, config=CFG), None)
            now = t1
    flow = fx.st.get_model("oa", SYSTEM_ENTITY, MP.PFLOW)
    req = flow["scopes"]["*"]["requires"]
    assert any(r["to"] == gen and r["from"] == form for r in req)
    t = days[20] + 17 * 3600 + 120
    b, asg = fx.score("oa", [(t, "192.168.1.23", {"http.route": gen, "http.method": "POST"})])
    assert asg.get("p_seq", 0) <= 0.1
    v = [e for e in fx.violations("192.168.1.23") if e.extra["type"] == "seq"]
    assert v and v[0].extra["p"] <= 0.05 and v[0].severity == Severity.MEDIUM
    assert "GET /report/form" in v[0].description


# ======================================================================== (f)
def test_f_prequential_scoring_never_writes_the_tree():
    fx, nid = finance_fixture(days=8)
    m = MP.get_ptree(fx.st, "finance")
    n_nodes, ver = len(m.kinds[0].nodes), m.version
    snap = fx.node("finance", APPROVE).n_c(fx.now)
    t = workdays(9)[-1] + 10 * 3600
    fx.score("finance", [_approve("192.168.2.10", t)] * 3)
    assert len(m.kinds[0].nodes) == n_nodes and m.version == ver
    assert fx.node("finance", APPROVE).n_c(t) == pytest.approx(snap * 2 ** (-(t - fx.now) / PN.PS.H_L), rel=0.05) \
        or fx.node("finance", APPROVE).n_c(t) <= snap


# ======================================================================== (g)
def test_g_sidak_per_tick_scores():
    fx, nid = finance_fixture()
    t = workdays(22)[-1] + 10.5 * 3600
    fx.score("finance", [_approve("192.168.1.23", t), _approve("192.168.1.23", t + 30),
                         _approve("192.168.2.10", t + 60)])
    row = fx.st.latest_derived("finance", "192.168.1.23", CF.CONF_SERIES).value
    w = row["conf_who"]
    U = CF._who_closed(fx.node("finance", APPROVE), fx.now)[2]
    assert w["n"] == 2
    assert w["p"] == pytest.approx(1 - (1 - U) ** 2, rel=0.2)
    assert w["score"] == pytest.approx(-math.log10(w["p"]))
    ok = fx.st.latest_derived("finance", "192.168.2.10", CF.CONF_SERIES).value
    assert ok["conf_who"]["p"] == 1.0


# ======================================================================== (h)
def test_h_per_day_multiplicity_busy_vs_quiet_ip():
    """An IP with 500 clean events a day at one node emits no more content
    findings than one with 5 (p_day), and the per IP-day rate stays small."""
    fx = Fx()
    rng = np.random.default_rng(3)
    route = "POST oa /upload"
    for d in workdays(12):
        for i in range(40):
            fx.learn("oa", route, "192.168.3.20", d + (9 + 8 * rng.random()) * 3600,
                     {"body.len": float(np.exp(rng.normal(7.0, 0.4)))})
    nd = fx.node("oa", route)
    t = workdays(13)[-1]
    rec = PB.fit_numeric(nd.targets["body.len"], t, int((t + 8 * 3600) // DAY), n_c=nd.n_c(t),
                         n_eff=nd.n_m(t), unit="B")
    fx.put("oa", MP.PBOUNDS, nd.id, attrs={"body.len": rec})
    busy, quiet = "192.168.3.21", "192.168.3.22"
    n_days = 80
    for k in range(n_days):
        t0 = t + (k + 1) * DAY + 10 * 3600
        evs = [(t0 + i * 10, busy, {"http.route": route, "body.len": float(np.exp(rng.normal(7.0, 0.4)))})
               for i in range(500)]
        evs += [(t0 + i * 10 + 5, quiet, {"http.route": route, "body.len": float(np.exp(rng.normal(7.0, 0.4)))})
                for i in range(5)]
        fx.score("oa", evs)
    nb = len([e for e in fx.violations(busy) if e.extra["type"] == "content"])
    nq = len([e for e in fx.violations(quiet) if e.extra["type"] == "content"])
    assert nb <= nq + 2
    assert (nb + nq) / (2 * n_days) <= 0.02


# ======================================================================== (i)
def test_i_intensity_sketch_guarantees():
    hs = CF.HourSS(k=1024)
    rng = np.random.default_rng(0)
    heavy = ("10.60.7.7", 0, 5)
    true: Dict[Any, int] = {}
    keys = [(f"10.{i // 60000}.{(i // 250) % 256}.{i % 250}", 0, 5) for i in range(20000)]
    stream = [keys[int(j)] for j in rng.integers(0, len(keys), 30000)] + [heavy] * 400
    rng.shuffle(stream)
    for k in stream:
        hs.add(k)
        true[k] = true.get(k, 0) + 1
    assert hs.guaranteed(heavy) <= 400 and hs.guaranteed(heavy) >= 400 - hs.N / hs.k
    for k, g in hs.rows():
        assert g <= true[k]                                   # never above the truth
    light_flagged = [k for k, g in hs.rows() if true[k] <= hs.N / hs.k and g > hs.N / hs.k]
    assert not light_flagged


def test_i_intensity_scored_on_guaranteed_counts_only():
    fx = Fx()
    route = "POST portal /login"
    rng = np.random.default_rng(2)
    for d in workdays(8):
        for i in range(50):
            fx.learn("portal", route, f"10.60.{i}.{int(rng.integers(1, 250))}", d + (10 + rng.random()) * 3600)
    nd = fx.node("portal", route)
    fx.put("portal", MP.PBOUNDS, nd.id, attrs={"rate.ip_h": {
        "kind": "num", "band90": [1.0, 3.0], "band98": [1.0, 5.0], "qgrid": [1.0, 1.0, 2.0, 3.0, 6.0],
        "range": [1.0, 6.0], "n_c": 20000.0, "hard": True, "cover": 1e-4, "log": False}})
    fx.eng = CF.ConformityEngine(k_int=64)
    _rate_history(fx, route)
    t = workdays(9)[-1] + 20 * 3600
    evs = [(t + i * 5, "10.60.7.7", {"http.route": route}) for i in range(400)]
    evs += [(t + i, f"10.61.{i // 200}.{i % 200}", {"http.route": route}) for i in range(1500)]
    fx.score("portal", evs)
    v = [e for e in fx.violations() if e.extra["type"] == "content" and "intensity" in e.extra["flags"]]
    assert {e.entity for e in v} == {"10.60.7.7"}
    # the finished hour is published for P04 (pat.rate)
    fx.score("portal", [(t + 3 * 3600, "10.61.9.9", {"http.route": route})])
    rates = fx.st.batches_since("portal", EV.PAT_RATE, -1)
    rows = rates[-1][1]["rows"]
    assert any(r[2] == "10.60.7.7" and r[3] >= 400 - 1900 / 64 for r in rows)


# ============================================================ novelty, damping
def test_novel_sensitive_action_and_outlier_damping():
    from app.engines.behavior.lib import pdfg as DF
    fx, nid = finance_fixture(days=10)
    flow = DF.PFlowModel()
    flow.state = DF.FlowState()
    for i in range(200):
        flow.state.acts.add(APPROVE, T0 + i, 1.0, 1.0)
        flow.state.acts.add("GET fin /fin/ledger", T0 + i, 1.0, 1.0)
    flow["scopes"] = {}
    fx.st.put_model("finance", SYSTEM_ENTITY, MP.PFLOW, flow)
    t = workdays(11)[-1] + 10 * 3600
    b, asg = fx.score("finance", [(t, "192.168.2.11", {"http.route": "GET fin /admin/export"})])
    v = [e for e in fx.violations("192.168.2.11") if e.extra["type"] == "novel"]
    assert v and v[0].severity == Severity.MEDIUM and "privilege" in v[0].axes
    # the outsider's approval event is learned with damp 0.1
    b, asg = fx.score("finance", [_approve("192.168.1.23", t + 60)])
    assert asg.get("damp", 0) == pytest.approx(0.1) or asg.get("p_ev", 0) > 1e-4


def test_inert_when_disabled():
    fx, nid = finance_fixture(days=6)
    fx.eng.safe_run(ctx(fx.st, fx.now + 3600, window_s=3600, config={"progressive": {"enabled": False}}), None)
    assert fx.st.batch_at("finance", EV.PAT_ASSIGN, fx.now + 3600) is None


# ==================================================== convergence (越久越准)
def test_evidence_against_an_outsider_grows_with_observation_time():
    """The same outsider event is judged more surely the longer the node was
    observed (U on the confidence channel falls), while members stay clean."""
    ps = []
    for days in (6, 10, 20, 40):
        fx, nid = finance_fixture(days=days)
        t = workdays(days + 1)[-1] + 10.5 * 3600
        b, asg = fx.score("finance", [_approve("192.168.1.23", t), _approve("192.168.2.10", t + 5)])
        ps.append(asg.get("p_who", 0))
        assert asg.get("p_who", 1) == 1.0
    assert all(ps[i + 1] < ps[i] for i in range(len(ps) - 1))
    assert ps[-1] < 0.003


# ================================================================= resources
def _scale_run(n_ips: int, extra: int = 0, **params: Any) -> Tuple[int, float]:
    fx = Fx()
    fx.eng = CF.ConformityEngine(**params)
    route = "GET portal /news"
    rng = np.random.default_rng(5)
    for d in workdays(6):
        for i in range(40):
            fx.learn("portal", route, f"10.60.{i}.1", d + (10 + rng.random()) * 3600,
                     {"net.bytes_up": float(rng.integers(300, 500))})
    nd = fx.node("portal", route)
    t = workdays(7)[-1]
    rec = PB.fit_numeric(nd.targets["net.bytes_up"], t, int((t + 8 * 3600) // DAY), n_c=nd.n_c(t),
                         n_eff=nd.n_m(t), unit="B")
    fx.put("portal", MP.PBOUNDS, nd.id, attrs={"net.bytes_up": rec})
    ips = [f"10.{61 + i // 60000}.{(i // 250) % 256}.{i % 250 + 1}" for i in range(n_ips)]
    total, n = 0.0, 0
    for h in range(3):
        evs = []
        for k in range(1500):
            a = {"http.route": route, "net.bytes_up": float(rng.integers(300, 500))}
            for j in range(extra):
                a[f"meta.f{j:03d}"] = float(rng.random())
            evs.append((t + h * 3600 + 10 * 3600 + k, ips[int(rng.integers(0, n_ips))], a))
        x = fx.t_eng
        fx.score("portal", evs)
        total += fx.t_eng - x
        n += len(evs)
    st = fx.st.get_model(ORG, ORG, CF.STATE)
    return st.nbytes(), total / n


def test_bounded_in_ips_and_attributes():
    """PPC-3 / S1: P03 keeps per-source state only in capped sketches (hourly
    intensity k_int, per-day cells, last activity); its cost per event does not
    grow with the number of IPs or of attributes."""
    caps = dict(k_int=512, cells=2000, last_active=2000)
    b1, us1 = _scale_run(500, **caps)
    b2, us2 = _scale_run(20000, **caps)
    assert b2 <= 2000 * 250 + 2000 * 200 + 512 * 300 + 200_000       # the caps bind
    assert b2 < 4 * b1
    # past the caps the cost per event is flat in the number of IPs (an event of
    # a never-seen IP costs more than a known one's: it is a who finding; 4 000
    # and 40 000 IPs are both almost all never-seen within 4 500 events)
    _, us4 = _scale_run(4000, **caps)
    _, us5 = _scale_run(40000, **caps)
    assert us5 < 1.5 * us4 + 30e-6
    b3, us3 = _scale_run(500, extra=300, **caps)
    assert us3 < 1.5 * us1 + 30e-6                                  # unchecked attributes cost nothing


def test_fast_numeric_path_equals_pbounds_p_value():
    rng = np.random.default_rng(7)
    num = PN.NumSummary(log=True)
    for i in range(600):
        num.update(float(np.exp(rng.normal(7.0, 0.5))), T0 + i * 60, 1.0, 1.0, day=int(i // 100))
    rec = PB.fit_numeric(num, T0 + 36000, 6, n_c=400.0, n_eff=300.0, unit="B")
    fast = CF._NumFast(rec)
    th, tl = rec["tail_hi"], rec["tail_lo"]
    assert th and tl and len(th) == 4
    lo98, hi98 = rec["band98"]
    n_tail = 0
    for v in list(np.exp(rng.normal(7.0, 1.5, 300))) + [0.0, -1.0, 1e9, float("nan"), "x"]:
        a, fa = PB.p_value(rec, v)
        b, fb = fast.p(v)
        assert fa == fb
        if isinstance(v, float) and math.isfinite(v) and v > 0 and not (lo98 <= v <= hi98):
            # beyond band98 P03 states the PREDICTIVE GPD tail (round 4): the
            # plug-in tail of p_value with the fit's parameter uncertainty and
            # the observed extreme (the record's range)
            if v > hi98:
                want = SC.tail_p(v, th[0], th[1], th[2], PB.TAIL_MASS, k=th[3], x_max=rec["range"][1])
            else:
                want = SC.tail_p(-v, -tl[0], tl[1], tl[2], PB.TAIL_MASS, k=tl[3], x_max=-rec["range"][0])
            assert b == pytest.approx(want, rel=1e-9, abs=1e-15)
            assert b >= a - 1e-15 or a > 1e-3            # never more extreme than the plug-in far out
            n_tail += 1
            continue
        assert (math.isnan(a) and math.isnan(b)) or a == pytest.approx(b, rel=1e-9, abs=1e-12)
    assert n_tail > 20


def test_low_and_slow_outsider_is_learned_damped():
    """A9: a SALES address reading the finance approval list once a day is a who
    violation every day and its events are learned with damp 0.1, so it never
    reaches the node's heavy set by persistence alone."""
    fx, nid = finance_fixture()
    days = workdays(33)
    damps = []
    for d in days[22:32]:
        b, asg = fx.score("finance", [_approve("192.168.3.33", d + 10.5 * 3600),
                                      _approve("192.168.2.10", d + 10.6 * 3600)])
        damps.append((asg.get("damp", 0), asg.get("damp", 1)))
    assert all(a == pytest.approx(0.1) and b == 1.0 for a, b in damps)
    who = [e for e in fx.violations("192.168.3.33") if e.extra["type"] == "who"]
    assert len({e.dedupe_key for e in who}) == 10          # once per day, every day


def test_node_routine_rate_of_extreme_scores():
    """CalStore.per_day: the H_m-decayed history of a node's scores read as a
    daily rate (the routine rule of the when / content / seq / intensity
    findings): 2 scores a day at p = 1e-5 read as ~2 a day at that level,
    none at a more extreme level, and the estimate does not depend on the age."""
    for days in (3, 20):
        cal = CF.CalStore()
        for d in range(days):
            for h in (10, 15):
                cal.add("k", 1e-5, T0 + d * DAY + h * 3600.0)
        now = T0 + days * DAY
        r = cal.per_day("k", 1e-5, now)
        assert 1.5 <= r <= 2.6, (days, r)
        assert cal.per_day("k", 1e-8, now) == 0.0
        assert cal.per_day("other", 1e-5, now) == 0.0


def test_intensity_routine_is_judged_on_magnitude():
    """A node whose tail hours are routine (three sources at 4x the band top
    every day) stops reporting such hours, but a 400-an-hour burst on the same
    node is still a finding: beyond the digest's maximum every count has the
    same rank p, so the routine is kept on the count's magnitude."""
    fx = Fx()
    route = "POST portal /login"
    rng = np.random.default_rng(4)
    for d in workdays(8):
        for i in range(50):
            fx.learn("portal", route, f"10.60.{i}.{int(rng.integers(1, 250))}", d + (10 + rng.random()) * 3600)
    nd = fx.node("portal", route)
    fx.put("portal", MP.PBOUNDS, nd.id, attrs={"rate.ip_h": {
        "kind": "num", "band90": [1.0, 3.0], "band98": [1.0, 5.0], "qgrid": [1.0, 1.0, 2.0, 3.0, 6.0],
        "range": [1.0, 6.0], "n_c": 20000.0, "mass": 20000.0, "hard": True, "cover": 1e-4, "log": False}})
    fx.eng = CF.ConformityEngine(k_int=64)
    _rate_history(fx, route)
    days = workdays(14)[8:]
    busy = ["10.62.0.1", "10.62.0.2", "10.62.0.3"]

    def hour(t0, extra=()):
        evs = [(t0 + k * 60.0, ip, {"http.route": route}) for ip in busy for k in range(12)]
        evs += list(extra)
        evs += [(t0 + i * 3.0, f"10.61.{i // 200}.{i % 200}", {"http.route": route}) for i in range(300)]
        fx.score("portal", sorted(evs, key=lambda e: e[0]))
    for d in days[:5]:
        hour(d + 14 * 3600)
    t = days[5] + 14 * 3600
    burst = [(t + i * 5.0, "10.60.7.7", {"http.route": route}) for i in range(400)]
    hour(t, burst)
    fx.score("portal", [(t + 2 * 3600, "10.61.9.9", {"http.route": route})])
    v_last = [e for e in fx.violations() if e.extra["type"] == "content" and "intensity" in e.extra["flags"]
              and e.ts >= t]
    assert {e.entity for e in v_last} == {"10.60.7.7"}


def test_low_volume_member_with_standing_is_not_flagged_at_the_covering_node():
    """A source with its own standing (>= MEMBER_EV units over days) is a member
    of the covering node though it holds < 5 % of the node's mass and evidence
    (pack O: the finance approver at finance's root was outside the 95 % heavy
    set, flagged on every login - 8 HIGH incidents - and damped, so its own
    approval nodes never confirmed)."""
    fx = Fx()
    big = [f"192.168.2.{20 + i}" for i in range(4)]
    for d in workdays(15):
        for ip in big:
            for k in range(30):
                fx.learn("finance", "GET fin /fin/ledger", ip, d + (9 * 60 + 9 * k) * 60.0 + hash(ip) % 7)
        fx.learn("finance", "GET fin /fin/ledger", "192.168.2.10", d + 15 * 3600.0)
    fx.st.put_model(ORG, ORG, MP.WHO_GROUPS, {
        "groups": {"G2": {"id": "G2", "name": "财务部", "members": big + ["192.168.2.10"]}},
        "ip2g": {ip: "G2" for ip in big + ["192.168.2.10"]}})
    nd = fx.node("finance", "GET fin /fin/ledger")
    t = workdays(16)[-1] + 15 * 3600.0
    l, mem, U = CF._who_closed(nd, t)
    assert l == 0 and "192.168.2.10" not in mem            # outside the heavy set ...
    assert nd.who.levels[0].evidence("192.168.2.10", t) >= CF.MEMBER_EV   # ... with standing
    fx.score("finance", [(t, "192.168.2.10", {"http.route": "GET fin /fin/ledger", "http.method": "GET"})])
    assert not [e for e in fx.violations("192.168.2.10") if e.extra["type"] == "who"]
    # a never-seen address is still a violation
    fx.score("finance", [(t + 600, "192.168.2.99", {"http.route": "GET fin /fin/ledger", "http.method": "GET"})])
    assert [e for e in fx.violations("192.168.2.99") if e.extra["type"] == "who"]


def test_reference_snapshot_keeps_low_volume_sources_with_standing():
    """P04's daily reference snapshot (§6.8.3) keeps every source with standing
    in its who list, so P03's dual anchor (current vs reference) does not flag
    a low-volume legitimate user that the 95 %-mass heavy set leaves out."""
    import types
    from app.engines.behavior.pattern_tree import PatternTreeEngine
    fx = Fx()
    big = [f"192.168.2.{20 + i}" for i in range(4)]
    for d in workdays(10):
        for ip in big:
            for k in range(30):
                fx.learn("finance", "GET fin /fin/ledger", ip, d + (9 * 60 + 9 * k) * 60.0)
        fx.learn("finance", "GET fin /fin/ledger", "192.168.2.10", d + 15 * 3600.0)
    tr = fx.tree("finance", ["GET fin /fin/ledger"])
    nd = fx.node("finance", "GET fin /fin/ledger")
    t = workdays(11)[-1] + 4 * 3600.0
    lc = types.SimpleNamespace(now=t, store=fx.st, key="finance", aux={"held": {}})
    PatternTreeEngine._snapshots(PatternTreeEngine(), lc, tr, EV.KIND_TXN, "finance")
    heavy, _ = nd.who.heavy_set(0, t)
    assert "192.168.2.10" not in heavy
    assert "192.168.2.10" in nd.ref["who"] and set(map(str, heavy)) <= set(nd.ref["who"])


def test_foreign_to_the_closed_system_is_judged_at_the_root_not_only_the_young_node():
    """A source foreign at the covering node AND at every closed ancestor is
    judged with the most confident closed population (Bonferroni over the
    levels): a young single-user node (U ~ 1/n_c) does not hide that the
    source never used the closed system (pack O A1 on day 16: node U 0.029 ->
    LOW and no incident; finance root U 0.0013)."""
    fx = Fx()
    for d in workdays(15):
        for ip in ("192.168.2.11", "192.168.2.12", "192.168.2.13"):
            for k in range(6):
                fx.learn("finance", "GET fin /fin/ledger", ip, d + (9 * 60 + 40 * k) * 60.0)
    for d in workdays(15)[-9:]:
        for k in range(3):
            fx.learn("finance", APPROVE, "192.168.2.10", d + (10 * 60 + 30 * k) * 60.0)
    t = fx.now = workdays(23)[-1] + 10.5 * 3600
    nd = fx.node("finance", APPROVE)
    l, mem, U = CF._who_closed(nd, t)
    tr = fx.tree("finance", [APPROVE])
    lr, _, Ur = CF._who_closed(tr.nodes[tr.root], t)
    assert l == 0 and U > 0.01 and lr is not None and 2 * Ur <= 0.01
    fx.score("finance", [_approve("192.168.1.23", t)])
    who = [e for e in fx.violations("192.168.1.23") if e.extra["type"] == "who"]
    assert who and who[0].severity == Severity.HIGH


# ---------------------------------------------------------- novelty (groups_views round)
SALES_IPS = [f"192.168.3.{i}" for i in range(20, 26)]
WEEKLY = "POST fin /report/weekly"


def _novel_fixture():
    from app.engines.behavior.lib import pdfg as DF
    fx, nid = finance_fixture(days=10)
    wg = dict(fx.st.get_model(ORG, ORG, MP.WHO_GROUPS))
    wg["groups"] = dict(wg["groups"], G3={"id": "G3", "name": "销售部", "members": SALES_IPS})
    wg["ip2g"] = dict(wg["ip2g"], **{ip: "G3" for ip in SALES_IPS})
    fx.st.put_model(ORG, ORG, MP.WHO_GROUPS, wg)
    flow = DF.PFlowModel()
    flow.state = DF.FlowState()
    for i in range(200):
        flow.state.acts.add(APPROVE, T0 + i, 1.0, 1.0)
        flow.state.acts.add("GET fin /fin/ledger", T0 + i, 1.0, 1.0)
    flow["scopes"] = {}
    fx.st.put_model("finance", SYSTEM_ENTITY, MP.PFLOW, flow)
    return fx


def _novel_findings(fx, ips):
    return {ip: [e.severity for e in fx.violations(ip) if e.extra["type"] == "novel"] for ip in ips}


def test_new_action_of_a_whole_group_is_one_finding_then_low():
    """A department starting a new write action (SALES' Friday report, 20 IPs
    within the hour) is a new behaviour of the GROUP: the first source is
    reported (nothing could know yet), every further member doing it the same
    day is capped at LOW - not one MEDIUM incident per member (pack O: 39
    MEDIUM `novel` findings for the weekly report, the largest FAR source)."""
    fx = _novel_fixture()
    t = workdays(11)[-1] + 16 * 3600
    for k, ip in enumerate(SALES_IPS):
        fx.score("finance", [(t + 600 * k, ip, {"http.route": WEEKLY, "http.method": "POST"})])
    f = _novel_findings(fx, SALES_IPS)
    assert f[SALES_IPS[0]] == [Severity.MEDIUM]
    assert all(all(s == Severity.LOW for s in f[ip]) for ip in SALES_IPS[1:])
    # a single source's new action is not "coordinated": still MEDIUM
    fx.score("finance", [(t + 7200, "192.168.2.11", {"http.route": "POST fin /fin/purge", "http.method": "POST"})])
    assert _novel_findings(fx, ["192.168.2.11"])["192.168.2.11"] == [Severity.MEDIUM]


def test_recurring_action_is_not_new_when_the_dictionary_lost_it():
    """An action performed by >= 2 sources on an earlier date is a recurring
    action of the system (weekly / monthly), whatever P10's decayed dictionary
    still holds; one source alone never establishes an action."""
    fx = _novel_fixture()
    days = workdays(16)
    t1, t2 = days[10] + 16 * 3600, days[15] + 16 * 3600          # one week apart
    for k, ip in enumerate(SALES_IPS[:3]):
        fx.score("finance", [(t1 + 600 * k, ip, {"http.route": WEEKLY, "http.method": "POST"})])
    fx.score("finance", [(t1 + 5000, "192.168.2.11", {"http.route": "GET fin /admin/export"})])
    n0 = {ip: len(v) for ip, v in _novel_findings(fx, SALES_IPS + ["192.168.2.11"]).items()}
    # a week later: the report again (P10's dictionary never learned it) and the probe again
    for k, ip in enumerate(SALES_IPS[:3]):
        fx.score("finance", [(t2 + 600 * k, ip, {"http.route": WEEKLY, "http.method": "POST"})])
    fx.score("finance", [(t2 + 5000, "192.168.2.11", {"http.route": "GET fin /admin/export"})])
    n1 = {ip: len(v) for ip, v in _novel_findings(fx, SALES_IPS + ["192.168.2.11"]).items()}
    assert all(n1[ip] == n0[ip] for ip in SALES_IPS[:3])
    assert n1["192.168.2.11"] == n0["192.168.2.11"] + 1
    b, asg = fx.score("finance", [(t2 + 9000, SALES_IPS[4], {"http.route": WEEKLY, "http.method": "POST"})])
    assert "recurring_action" in str(asg.get("flags", 0))


def test_lateral_read_into_a_closed_system_is_medium():
    """A source whose group never used the SYSTEM reading a node closed to one
    approver (pack O A9: a sales address reading finance's approval list,
    GET, sensitivity 1.5) is a lateral move: MEDIUM, axis lateral - LOW gave
    no incident at all. A colleague-free read by a group that already uses the
    system stays LOW."""
    fx = _novel_fixture()
    LIST = "GET fin /fin/approval/list"
    for d in workdays(10):
        for k in range(4):
            fx.learn("finance", LIST, "192.168.2.10", d + (10 * 60 + 7 * k) * 60.0)
    t = workdays(11)[-1] + 10.5 * 3600
    fx.score("finance", [(t, SALES_IPS[0], {"http.route": LIST, "http.method": "GET"})])
    v = [e for e in fx.violations(SALES_IPS[0]) if e.extra["type"] == "who"]
    assert v and v[0].severity == Severity.MEDIUM
    assert {"outsider_group", "system_new"} <= set(v[0].extra["flags"]) and "lateral" in v[0].axes
    # a finance colleague (group already in the system) reading it: not a lateral move
    fx.score("finance", [(t + 600, "192.168.2.11", {"http.route": LIST, "http.method": "GET"})])
    w = [e for e in fx.violations("192.168.2.11") if e.extra["type"] == "who"]
    assert all(CF.SEV_RANK[e.severity] <= CF.SEV_RANK[Severity.LOW] for e in w)


def test_hour_count_beyond_every_ip_hour_ever_seen_is_ranked_against_all_of_them():
    """The rate.ip_h digest's n is its DECAYED mass of IP-hours (a few hundred
    on a portal login node), so its conformal rank p of a count beyond its
    maximum never went below ~1/(mass + 1) = 2.5e-3: 400 logins in one hour
    (pack O A7) were never an intensity finding. P03's own tally of every
    finished IP-hour at the node ranks such a count first among all N of
    them: p = 1/(N + 1) -> MEDIUM on the write action once N >= 1e4."""
    fx = Fx()
    route = "POST portal /login"
    rng = np.random.default_rng(5)
    for d in workdays(8):
        for i in range(50):
            fx.learn("portal", route, f"10.60.{i}.{int(rng.integers(1, 250))}", d + (10 + rng.random()) * 3600)
    nd = fx.node("portal", route)
    fx.put("portal", MP.PBOUNDS, nd.id, attrs={"rate.ip_h": {
        "kind": "num", "band90": [1.0, 1.0], "band98": [1.0, 1.95], "qgrid": [1.0] * 31 + [1.05, 2.0],
        "range": None, "n_c": 774.0, "mass": 397.0, "hard": False, "log": False}})
    fx.eng = CF.ConformityEngine(k_int=4096)
    t0 = workdays(9)[-1]
    for h in range(36):                                        # 36 finished hours x 300 one-login IPs
        th = t0 + h * 3600.0
        fx.score("portal", [(th + i * 10.0, f"10.61.{(h * 300 + i) // 250 % 250}.{(h * 300 + i) % 250}",
                             {"http.route": route, "http.method": "POST"}) for i in range(300)])
    t = t0 + 37 * 3600.0
    burst = [(t + i * 5.0, "10.60.7.7", {"http.route": route, "http.method": "POST"}) for i in range(400)]
    fx.score("portal", burst)
    v = [e for e in fx.violations("10.60.7.7") if e.extra["type"] == "content" and "intensity" in e.extra["flags"]]
    assert v and max(CF.SEV_RANK[e.severity] for e in v) >= CF.SEV_RANK[Severity.MEDIUM]
    # none of the one-login sources is a finding
    assert not [e for e in fx.violations() if e.entity != "10.60.7.7" and "intensity" in e.extra["flags"]]


def test_small_hour_count_is_ranked_not_extrapolated_by_a_continuous_tail():
    """An integer hour count of 3 where the node's history holds hundreds of
    2s and dozens of 3s is ordinary: its rank among the node's IP-hours is
    ~1e-3, whatever a GPD tail fitted to counts that are almost all 1 says
    (P06's bounded tail put it at the p floor 1e-9: 40 clean portal / crm
    visitors were intensity findings on pack O seed 0)."""
    fx = Fx()
    route = "POST portal /login"
    rng = np.random.default_rng(6)
    for d in workdays(8):
        for i in range(50):
            fx.learn("portal", route, f"10.60.{i}.{int(rng.integers(1, 250))}", d + (10 + rng.random()) * 3600)
    nd = fx.node("portal", route)
    fx.put("portal", MP.PBOUNDS, nd.id, attrs={"rate.ip_h": {
        "kind": "num", "band90": [1.0, 1.0], "band98": [1.0, 1.95], "qgrid": [1.0] * 31 + [1.05, 2.0],
        "range": [1.0, 2.0], "n_c": 20000.0, "mass": 400.0, "hard": False, "log": False,
        "tail_hi": [1.95, -0.5, 0.05]}})
    _rate_history(fx, route, n=20000)
    st = fx.st.get_model(ORG, ORG, CF.STATE)
    key = MP.tree_key(fx.st, "portal")
    st.tally().close_hour(key, [(0, nd.id, f"x{i}", 2.0) for i in range(300)], T0)
    st.tally().close_hour(key, [(0, nd.id, f"y{i}", 3.0) for i in range(30)], T0)
    fx.st.put_model(ORG, ORG, CF.STATE, st)
    t = workdays(9)[-1] + 20 * 3600
    fx.score("portal", [(t + i * 60.0, "10.60.8.8", {"http.route": route, "http.method": "POST"}) for i in range(3)])
    assert not [e for e in fx.violations("10.60.8.8") if "intensity" in e.extra["flags"]]


def test_new_lease_of_a_configured_dhcp_pool_is_not_an_unknown_source(monkeypatch):
    """A login node used by a DHCP pool closes at /24 (its IPs churn); a fresh
    lease from a /24 of the same configured pool that had not been drawn yet
    is a re-addressed member, not an unknown source (pack O: five MEDIUM
    unknown_ip findings on the make-up Saturday). An address outside every
    pool is still judged."""
    monkeypatch.setitem(CFG, "dhcp_scopes", [{"cidr": "10.50.0.0/22", "name": "研发 DHCP"}])
    fx = Fx()
    rng = np.random.default_rng(7)
    pool = [f"10.50.{b}.{i}" for b in (0, 2, 3) for i in range(1, 21)]
    for d in workdays(12):
        for ip in rng.choice(pool, 30, replace=False):
            fx.learn("oa", LOGIN, str(ip), d + (9.5 + rng.random()) * 3600)
    nd = fx.node("oa", LOGIN)
    lvl, mem, U = CF._who_closed(nd, workdays(13)[-1])
    assert lvl == 1 and U <= 0.02                       # closed at /24
    t = workdays(13)[-1] + 10 * 3600
    fx.score("oa", [(t, "10.50.1.19", {"http.route": LOGIN, "http.method": "POST"})])
    assert not [e for e in fx.violations("10.50.1.19") if e.extra["type"] == "who"
                and CF.SEV_RANK[e.severity] >= CF.SEV_RANK[Severity.MEDIUM]]
    fx.score("oa", [(t + 60, "10.77.1.19", {"http.route": LOGIN, "http.method": "POST"})])
    assert [e for e in fx.violations("10.77.1.19") if e.extra["type"] == "who"
            and CF.SEV_RANK[e.severity] >= CF.SEV_RANK[Severity.MEDIUM]]


def test_colleague_rule_scales_to_a_two_member_group():
    """The node is a pattern of a P11 group when GROUP_MIN_MEMBERS other members
    have standing there - but never more than the group HAS: in a two-member
    group the one colleague is enough. Pack O: 综合部's report writers
    {192.168.1.23, 10.168.7.121} were one group; at the 综合部 login node .23
    had standing, so .121 needed 2 colleagues it could never have, was flagged
    outsider_group, damped 0.1 and untrusted by B28 from day 8 and dropped out
    of the login statement. A colleague's habit in a larger group is still not
    a group pattern (A1, test_one_members_habit_is_not_a_pattern_of_the_group)."""
    m = PT.PTreeModel("oa")
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    nd = tr.nodes[tr.root]

    def keys(ip, g):
        p = ip.split(".")
        return [ip, ".".join(p[:3]) + ".0/24", ".".join(p[:2]) + ".0.0/16", f"grp:{g}", "reg:∅"]
    t = T0
    for d in range(10):
        t = T0 + d * DAY + 36000.0
        for k in range(3):
            nd.update_core(t + 400 * k, 1.0, 1.0, keys("192.168.1.23", "G10"), "192.168.1.23", 0, 600.0,
                           int((t + 8 * 3600) // DAY))
    ip2g = {"192.168.1.23": "G10", "10.168.7.121": "G10"}
    groups = {"G10": {"members": ["10.168.7.121", "192.168.1.23", "10.168.7.0/24"]}}
    assert CF.group_need("G10", "10.168.7.121", groups) == 1
    assert not CF.group_outsider(nd, "G10", "10.168.7.121", t, ip2g, groups)
    # a four-member group still needs two colleagues there
    ip2g4 = dict(ip2g, **{"192.168.1.21": "G10", "192.168.1.30": "G10"})
    groups4 = {"G10": {"members": sorted(ip2g4)}}
    assert CF.group_need("G10", "10.168.7.121", groups4) == 2
    assert CF.group_outsider(nd, "G10", "10.168.7.121", t, ip2g4, groups4)


def test_hour_count_far_beyond_the_node_maximum_is_resolved_below_the_rank_floor():
    """The conformal rank of an hour count cannot go below 1 / (W + 1): on a
    portal login node with ~2 500 IP-hours, 400 logins in one hour (pack O seed
    1, A7) had p 4e-4 like a count of 17 - LOW, no incident. The node's own
    count moments (Cantelli) resolve the magnitude; a count just above the
    maximum keeps its rank p, and a node whose IP-hours routinely reach
    hundreds (crawler-like) does not make 400 extreme."""
    rng = np.random.default_rng(0)
    t = T0
    rt = CF.RateTally()
    counts = np.minimum(1 + rng.geometric(0.6, 2500) - 1, 12)
    counts[:3] = 12
    rt.close_hour("portal", [(EV.KIND_TXN, 7, f"10.0.{i // 250}.{i % 250}", float(c))
                             for i, c in enumerate(counts)], t)
    W = 2500.0
    p17 = rt.p("portal", EV.KIND_TXN, 7, 17.0, t)      # the first bin above the maximum's
    assert p17 == pytest.approx(1.0 / (W + 1.0), rel=1e-6)          # just above the maximum: the rank
    assert rt.p("portal", EV.KIND_TXN, 7, 400.0, t) <= 1e-4          # far beyond it: resolved
    assert rt.p("portal", EV.KIND_TXN, 7, 2.0, t) > 0.05
    ones = CF.RateTally()                                            # every IP-hour a single login
    ones.close_hour("portal", [(EV.KIND_TXN, 7, f"10.1.0.{i % 250}", 1.0) for i in range(300)], t)
    assert ones.p("portal", EV.KIND_TXN, 7, 3.0, t) == pytest.approx(1 / 301.0)  # the rank, not 1e-9
    assert ones.p("portal", EV.KIND_TXN, 7, 30.0, t) > 1e-3           # no finding
    crawl = CF.RateTally()
    cc = np.concatenate([np.ones(2300), rng.integers(50, 300, 200)])
    crawl.close_hour("portal", [(EV.KIND_TXN, 7, f"10.0.{i // 250}.{i % 250}", float(c))
                                for i, c in enumerate(cc)], t)
    assert crawl.p("portal", EV.KIND_TXN, 7, 400.0, t) == pytest.approx(1.0 / (W + 1.0), rel=1e-6)


def test_address_with_standing_stays_a_member_when_its_group_id_changes():
    """A node closed at the GROUP level (18 users: the IP level is not a who
    constraint) keeps its members when P11 re-forms a group under a new id: the
    address's own recurring use of the action (its P11 signature) makes it a
    member, and it is not new to the system. Pack O seed 0: 192.168.1.21's group became G22 (new id) on day
    15; its daily mail use was 'outsider_group, system_new' from then on (a HIGH
    incident on a clean day, B28 trust 0, nothing of it learned any more)."""
    fx, _ = finance_fixture(days=10)
    far = [f"10.{k}.0.5" for k in range(12)]                    # 12 /16s: no prefix level closes
    wg = dict(fx.st.get_model(ORG, ORG, MP.WHO_GROUPS))
    wg["groups"] = dict(wg["groups"], G3={"id": "G3", "name": "G3", "members": far})
    wg["ip2g"] = dict(wg["ip2g"], **{ip: "G3" for ip in far})
    fx.st.put_model(ORG, ORG, MP.WHO_GROUPS, wg)
    MAIL = "GET oa /docs"
    for d in workdays(10):
        for ip in GA + FIN + far:
            for k in range(2):
                fx.learn("oa", MAIL, ip, d + (10 * 60 + 13 * k) * 60.0)
    nd = fx.node("oa", MAIL)
    t = workdays(11)[-1] + 10.5 * 3600
    lvl, mem, U = CF._who_closed(nd, t)
    assert lvl == CF.GRP_LEVEL and "grp:G1" in mem
    # P11's own signature of the address: this action on 10 days (the node's
    # IP-level summary keeps only WHO_K = 8 heavy hitters of its 18 sources)
    from app.engines.behavior import who_groups as WG
    ws = WG.WGState()
    route = CF._route_key(lambda a: {"http.route": MAIL, "http.method": "GET"}.get(a, EV.ABSENT))
    for k, d in enumerate(workdays(10)):
        ws.sigs.add("192.168.1.21", f"{MP.tree_key(fx.st, 'oa')}|{route}", d + 36000, 2.0, 1.0, k)
        ws.sigs.add("192.168.1.21", "mail|TLS mail", d + 36000, 2.0, 1.0, k)
    fx.st.put_model(ORG, ORG, CF.WG_STATE, ws)
    wg = dict(fx.st.get_model(ORG, ORG, MP.WHO_GROUPS))
    wg["groups"] = dict(wg["groups"], G22={"id": "G22", "name": "G22", "members": ["192.168.1.21"]})
    wg["ip2g"] = dict(wg["ip2g"], **{"192.168.1.21": "G22"})
    fx.st.put_model(ORG, ORG, MP.WHO_GROUPS, wg)
    _, asg = fx.score("oa", [(t, "192.168.1.21", {"http.route": MAIL, "http.method": "GET"})])
    assert float(asg.dense("p_who", 1.0)[0]) == 1.0
    assert not [e for e in fx.violations("192.168.1.21") if e.extra["type"] == "who"]
    # a stranger in the same new situation is still foreign
    wg["groups"]["G23"] = {"id": "G23", "name": "G23", "members": ["192.168.9.9"]}
    wg["ip2g"]["192.168.9.9"] = "G23"
    fx.st.put_model(ORG, ORG, MP.WHO_GROUPS, wg)
    fx.score("oa", [(t + 600, "192.168.9.9", {"http.route": MAIL, "http.method": "GET"})])
    w = [e for e in fx.violations("192.168.9.9") if e.extra["type"] == "who"]
    assert w and {"outsider_group", "system_new"} <= set(w[0].extra["flags"])


def test_colleague_using_the_action_counts_though_the_node_summary_cannot_show_it():
    """group_members_at reads the node's IP-level summary (WHO_K = 8 heavy
    hitters) AND the members' own signatures: at a login node shared by 30
    sources the colleague 192.168.1.23 is not among the 8, yet it logs in daily -
    10.168.7.121 is its colleague there, not 'outsider_group' (pack O seed 0:
    damped and untrusted from day 9, out of the 综合部 login statement). A
    member whose signature does not hold the action gives no standing."""
    m = PT.PTreeModel("oa")
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    nd = tr.nodes[tr.root]

    def keys(ip, g):
        p = ip.split(".")
        return [ip, ".".join(p[:3]) + ".0/24", ".".join(p[:2]) + ".0.0/16", f"grp:{g}", "reg:∅"]
    t = T0
    big = [f"192.168.3.{i}" for i in range(20, 50)]
    for d in range(10):
        t = T0 + d * DAY + 36000.0
        for ip in big:
            for k in range(3):
                nd.update_core(t + 60 * k, 1.0, 1.0, keys(ip, "G12"), ip, 0, 600.0, int((t + 8 * 3600) // DAY))
        nd.update_core(t - 900, 1.0, 1.0, keys("192.168.1.23", "G10"), "192.168.1.23", 0, 600.0,
                       int((t + 8 * 3600) // DAY))
    lv0 = nd.who.levels[0]
    assert "192.168.1.23" not in lv0 or lv0.evidence("192.168.1.23", t) < CF.MEMBER_EV
    ip2g = {"192.168.1.23": "G10", "10.168.7.121": "G10"}
    groups = {"G10": {"members": ["10.168.7.121", "192.168.1.23"]}}
    assert CF.group_outsider(nd, "G10", "10.168.7.121", t, ip2g, groups)              # summary only
    assert not CF.group_outsider(nd, "G10", "10.168.7.121", t, ip2g, groups,
                                 uses=lambda mm: mm == "192.168.1.23")
    assert CF.group_outsider(nd, "G10", "10.168.7.121", t, ip2g, groups, uses=lambda mm: False)


def test_value_inside_the_observed_range_is_never_below_its_rank():
    """A value inside the node's observed clean range has p >= 2 / (n_rng + 1),
    whatever the tail model says: a bounded GPD fitted to integer packet
    counts put the observed minimum (pack O: net.pkts_down = 6, range 6-14) at
    p = 1e-9 - an outlier damping of a clean member (192.168.1.21)."""
    rec = {"kind": "num", "range": [6.0, 14.0], "n_rng": 199.0}
    assert CF._in_range_floor(rec, 6.0, 1e-9) == pytest.approx(2.0 / 200.0)
    assert CF._in_range_floor(rec, 14.0, 0.5) == 0.5
    assert CF._in_range_floor(rec, 5.0, 1e-9) == pytest.approx(1.0 / 200.0)   # one integer step: a new extreme
    assert CF._in_range_floor(rec, 4.0, 1e-9) == 1e-9               # further out: the model stands
    # the stated extremes are inverse transforms (exp(log 460) = 460.0000000000001)
    lg = {"kind": "num", "range": [460.0000000000001, 520.0], "n_rng": 99.0}
    assert CF._in_range_floor(lg, 460.0, 1e-9) == pytest.approx(2.0 / 100.0)
    # a hair beyond the observed maximum is a new extreme (rank 1 / (n + 1)), not 1e-9
    dur = {"kind": "num", "range": [0.0, 336.5], "n_rng": 999.0}
    assert CF._in_range_floor(dur, 336.98, 1e-9) == pytest.approx(1.0 / 1000.0)
    assert CF._in_range_floor(dur, 400.0, 1e-9) == 1e-9
    byt = {"kind": "num", "range": [460.0, 515.0], "n_rng": 499.0}           # integer data
    assert CF._in_range_floor(byt, 516.0, 1e-9) == pytest.approx(1.0 / 500.0)
    assert CF._in_range_floor(byt, 517.0, 1e-9) == 1e-9
    assert CF._in_range_floor({"kind": "num"}, 6.0, 1e-9) == 1e-9
    # through the scoring path
    rng = np.random.default_rng(3)
    num = PN.NumSummary(log=False)
    for i in range(400):
        num.update(float(rng.integers(6, 15)), T0 + i * 60, 1.0, 1.0, day=int(i // 40))
    fit = PB.fit_numeric(num, T0 + 36000, 10, n_c=400.0, n_eff=300.0)
    lo = fit["range"][0]
    p_fast, _ = CF._check("num", CF._NumFast(fit), lo, None)
    p_slow, _ = CF._check("num", fit, lo, None)
    assert p_fast >= 2.0 / (fit["n_rng"] + 1.0) - 1e-12 and p_slow >= 2.0 / (fit["n_rng"] + 1.0) - 1e-12


def test_cross_binding_reads_the_sources_binding_credibility_up_the_path():
    """A who split restarts the child's binding table: on pack O the 综合部 login
    node (split on day 10) held 192.168.1.21 -> jack at LB 0.7 on day 17 while
    its parent - every login of .21 since day 1 - held it at 0.95. A2 (rose's
    credential used from .21) is credential-grade at once (MEDIUM), not two days
    later (max_ttd 1 h)."""
    fx = Fx()
    users = dict(zip(GA, ["jack", "rose", "mike"]))
    for d in workdays(15):
        for k, ip in enumerate(GA):
            fx.learn("oa", LOGIN, ip, d + (9 * 60 + 3 * k) * 60.0)
    tr = fx.tree("oa", [LOGIN])
    par = fx.node("oa", LOGIN)
    sp = tr.split(par.id, "net.src", 1, [["192.168.1.0/24", "10.168.7.0/24"]], T0 + 10 * DAY)
    child = tr.nodes[sp.children[0]]
    hier = MP.hierarchies(fx.st, "oa", CFG)
    for d in workdays(15)[-5:]:
        for k, ip in enumerate(GA):
            ts = d + (9 * 60 + 3 * k) * 60.0
            keys = [hier.gen("net.src", l, ip) for l in range(PN.WHO_LEVELS)]
            child.update_core(ts, 1.0, 1.0, keys, ip, 0, local_minute(ts), int((ts + 8 * 3600) // DAY))
    child.state = "confirmed"

    def pairs(lb):
        table = {ip: {"n": 15.0, "top": u, "k": 15.0, "bound": True, "LB": lb, "p_viol": 0.02}
                 for ip, u in users.items()}
        return {"net.src->body.kv.username": {
            "x": "net.src", "y": "body.kv.username", "dir": "fwd", "fd": {"g3": 0.0, "holds": True},
            "table": table, "bound_values": {u: [ip] for ip, u in users.items()}}}
    fx.put("oa", MP.PBIND, par.id, pairs=pairs(0.95))
    m = fx.st.get_model("oa", SYSTEM_ENTITY, MP.PBIND)
    m["nodes"][0][child.id] = {"status": "fitted", "pairs": pairs(0.7)}
    fx.st.put_model("oa", SYSTEM_ENTITY, MP.PBIND, m)
    t = workdays(16)[-1] + 9 * 3600 + 600
    _, asg = fx.score("oa", [(t, "192.168.1.21", {"http.route": LOGIN, "body.kv.username": "rose"})])
    assert int(asg.dense("conf", -1)[0]) == child.id
    v = [e for e in fx.violations("192.168.1.21") if e.extra["type"] == "content"]
    assert v and "cross_binding" in v[0].extra["flags"] and v[0].severity == Severity.MEDIUM
    assert asg.get("damp", 0) == pytest.approx(0.1)


def test_single_address_group_is_judged_by_its_own_recurring_use():
    """A P11 group whose only address is the source (its other sources are
    prefix-mode pools) has no colleague to judge by: its own recurring use of
    the action (P11 signature) is the group's standing there, else it is an
    outsider (pack O seed 0: 192.168.1.21, alone in its group after the split
    of 综合部 by activity, was 'outsider_group' and damped on its daily mail)."""
    m = PT.PTreeModel("mail")
    tr = m.tree(EV.KIND_TXN, T0, create=True)
    nd = tr.nodes[tr.root]
    for d in range(10):
        ts = T0 + d * DAY + 36000.0
        nd.update_core(ts, 1.0, 1.0, ["192.168.2.11", "192.168.2.0/24", "192.168.0.0/16", "grp:G9", "reg:∅"],
                       "192.168.2.11", 0, 600.0, int((ts + 8 * 3600) // DAY))
    ip2g = {"192.168.1.21": "G22"}
    groups = {"G22": {"members": ["192.168.1.21"], "pools": ["192.168.1.0/24"]}}
    t = T0 + 10 * DAY
    assert not CF.group_outsider(nd, "G22", "192.168.1.21", t, ip2g, groups, uses=lambda mm: True)
    assert CF.group_outsider(nd, "G22", "192.168.1.21", t, ip2g, groups, uses=lambda mm: False)
