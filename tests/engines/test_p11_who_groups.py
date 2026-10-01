"""P11 WhoGroupsEngine (behavior.who_groups; docs/lib3/progressive.md §6.15, card P11).

Synthetic organisation: departments whose IPs use distinct action sets on
several systems, written straight into the evt.batch series one hourly tick
at a time (the learned rows P11 reads at t - D)."""
from __future__ import annotations

import ipaddress
import time
import tracemalloc
from typing import Any, Dict, List, Optional, Sequence, Tuple

import numpy as np
import pytest

from helpers import ctx, make_store

from app.engines.behavior import who_groups as WG
from app.engines.behavior.lib import pminhash as MH
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevent as EV
from app.eval.pmetrics import ari
from app.models.schema import ORG, SYSTEM_ENTITY

T0 = 1_790_035_200.0            # 2026-09-21 16:00 UTC = Tuesday 00:00 local (UTC+8)
DAY = 86400.0
DT = 3600.0


def dept_ips() -> Dict[str, List[str]]:
    return {"GA": ["192.168.1.21", "192.168.1.23", "10.168.7.121"],
            "FIN": ["192.168.2.10", "192.168.2.11", "192.168.2.12"],
            "SALES": [f"192.168.3.{i}" for i in range(20, 40)],
            "DEV": [f"10.50.{i // 250}.{i % 250 + 1}" for i in range(60)]}


PROGRAM = {
    "GA": [("oa", "POST oa /login", 1), ("oa", "GET oa /approval/list", 4), ("oa", "POST oa /approval/{num}", 3),
           ("mail", "TLS:mail.corp", 5)],
    "FIN": [("finance", "POST fin /fin/login", 1), ("finance", "GET fin /fin/ledger", 6),
            ("finance", "POST fin /fin/voucher", 3)],
    "SALES": [("crm", "GET crm /crm/customer/{num}", 6), ("crm", "POST crm /crm/visit", 2),
              ("oa", "GET oa /docs", 3)],
    "DEV": [("code", "TLS:git.corp", 8)],
}


class OrgSim:
    def __init__(self, config: Optional[Dict[str, Any]] = None, extra_attrs: int = 0,
                 seed: int = 0, **params: Any) -> None:
        self.st = make_store()
        self.now = T0
        self.cfg = {"progressive": {"enabled": True}, "tz": "Asia/Shanghai"}
        self.cfg.update(config or {})
        self.eng = WG.WhoGroupsEngine(**params)
        self.rng = np.random.default_rng(seed)
        self.extra = int(extra_attrs)
        self.ips = dept_ips()
        self.readdr: List[Tuple[str, str]] = []
        self.pending: Dict[str, List[Tuple[float, str, Dict[str, Any]]]] = {}
        self.t_eng = 0.0
        D = EV.learn_delay_s(DT)
        self.st.ensure_retention("evt.", max_age_s=D + 2 * DT)
        self.st.ensure_retention("pat.", max_age_s=D + 2 * DT)

    def plan_day(self, t_day: float, depts: Sequence[str] = ("GA", "FIN", "SALES", "DEV"),
                 ips: Optional[Dict[str, List[str]]] = None) -> None:
        ips = ips or self.ips
        for d in depts:
            for ip in ips[d]:
                for sysn, act, per in PROGRAM[d]:
                    for _ in range(int(self.rng.poisson(per)) + 1):
                        ts = t_day + (9 + 8 * self.rng.random()) * 3600.0 - 8 * 3600.0
                        if act.startswith("TLS:"):
                            attrs = {"tls.sni": act[4:], "net.bytes_up": float(self.rng.integers(100, 9000))}
                        else:
                            attrs = {"http.route": act, "http.method": act.split()[0]}
                        for j in range(self.extra):
                            attrs[f"meta.f{j:03d}"] = float(self.rng.random())
                        self.pending.setdefault(sysn, []).append((ts, ip, attrs))

    def tick(self) -> None:
        t1 = self.now + DT
        for s, evs in list(self.pending.items()):
            cur = sorted((e for e in evs if self.now < e[0] <= t1), key=lambda e: e[0])
            self.pending[s] = [e for e in evs if e[0] > t1]
            b = EV.BatchBuilder(s, EV.KIND_TXN)
            for ts, ip, attrs in cur:
                b.add(ts, ip, dict(attrs, **{"net.src": ip}))
            self.st.add_batch(s, EV.EVT_BATCH, t1, b.build(self.now, t1))
            if self.readdr:
                bt = self.st.batch_at(s, EV.EVT_BATCH, t1)
                rows = [i for i in range(bt.n) if any(bt.ip_of(i) == new for new, _ in self.readdr)]
                if rows:
                    old = {new: o for new, o in self.readdr}
                    cols = {"readdr": EV.Col(np.asarray(rows, dtype=np.int32),
                                             np.asarray([old[bt.ip_of(i)] for i in rows], dtype=object))}
                    self.st.add_batch(s, EV.PAT_ASSIGN, t1, bt.aligned(cols))
                    self.readdr = []
        self.now = t1
        t = time.perf_counter()
        self.eng.safe_run(ctx(self.st, t1, window_s=DT, config=self.cfg), None)
        self.t_eng += time.perf_counter() - t

    def days(self, n: int, **kw: Any) -> None:
        for _ in range(n):
            day0 = self.now
            wd = ((int((day0 + 8 * 3600) // DAY) + 3) % 7) < 5
            if wd:
                self.plan_day(day0 + 8 * 3600.0, **kw)
            for _ in range(24):
                self.tick()

    def model(self) -> Dict[str, Any]:
        return MP.who_groups(self.st)

    def ari(self, depts: Sequence[str] = ("GA", "FIN", "SALES", "DEV")) -> float:
        ip2g = self.model().get("ip2g") or {}
        t, p = [], []
        for d in depts:
            for ip in self.ips[d]:
                t.append(d)
                p.append(ip2g.get(ip, f"_:{ip}"))
        return float(ari(t, p))


@pytest.fixture(scope="module")
def org7():
    sim = OrgSim(config={"who_group_names": [{"name": "综合部", "ips": dept_ips()["GA"]}],
                         "dhcp_scopes": [{"cidr": "10.50.0.0/22", "name": "研发 DHCP"}]})
    aris = []
    for _ in range(7):
        sim.days(1)
        aris.append(sim.ari() if sim.model().get("groups") else 0.0)
    sim.aris = aris
    return sim


def test_departments_recovered_ari(org7):
    assert org7.aris[-1] >= 0.9
    m = org7.model()
    by = {}
    for g, r in m["groups"].items():
        by[g] = set(r["members"])
    ga = set(dept_ips()["GA"])
    assert any(ga == s for s in by.values())


def test_configured_and_imported_names(org7):
    m = org7.model()
    names = {r["name"]: set(r["members"]) for r in m["groups"].values()}
    assert names.get("综合部") == set(dept_ips()["GA"])
    assert names.get("研发 DHCP") == set(dept_ips()["DEV"])
    g_auto = [r for r in m["groups"].values() if r["name_source"] == "auto"]
    assert all(r["name"].startswith(r["id"]) for r in g_auto)


def test_labels_and_systems(org7):
    m = org7.model()
    fin = next(r for r in m["groups"].values() if "192.168.2.10" in r["members"])
    labels = [x["label"] for x in fin["labels"]]
    assert any("finance" in l for l in labels)
    assert all(x["lift"] >= WG.LIFT_MIN for x in fin["labels"])
    assert fin["systems"].get("finance", 0.0) > 0.9


def test_prefix_covers(org7):
    m = org7.model()
    sales = next(r for r in m["groups"].values() if "192.168.3.20" in r["members"])
    nets = [ipaddress.ip_network(c) for c in sales["covers"]]
    assert nets and all(n.subnet_of(ipaddress.ip_network("192.168.3.0/24")) for n in nets)
    assert all(any(ipaddress.ip_address(ip) in n for n in nets) for ip in dept_ips()["SALES"])
    dev = next(r for r in m["groups"].values() if "10.50.0.1" in r["members"])
    assert any(ipaddress.ip_network(c).subnet_of(ipaddress.ip_network("10.50.0.0/22")) for c in dev["covers"])
    # published for the IP hierarchy's region level
    assert set(m["covers"]) >= {sales["id"], dev["id"]}


def test_precision_rises_with_observation_time(org7):
    """S3 "用的时间越长越精准": the ARI of the learned groups against the departments
    never drops by more than noise and reaches >= 0.9."""
    a = org7.aris
    assert a[0] <= a[-1]
    assert max(a[:2]) < 0.9 or a[1] >= 0.9           # day 1 has at most one clustering run
    assert all(a[i + 1] >= a[i] - 0.05 for i in range(len(a) - 1))


def test_stable_ids_across_runs(org7):
    m0 = org7.model()
    ids0 = {g: set(r["members"]) for g, r in m0["groups"].items()}
    ev0 = len([e for e in org7.st.events() if e.kind == "group_formed"])
    org7.days(3)
    m1 = org7.model()
    ids1 = {g: set(r["members"]) for g, r in m1["groups"].items()}
    for g, mem in ids0.items():
        if len(mem) >= 3:
            assert g in ids1 and len(ids1[g] & mem) / len(ids1[g] | mem) >= 0.9
    assert len([e for e in org7.st.events() if e.kind == "group_formed"]) == ev0


def test_ip_hierarchy_group_level(org7):
    h = MP.hierarchies(org7.st, "oa", org7.cfg)
    g = h.gen("net.src", 3, "192.168.1.21")
    assert g.startswith("grp:") and g == h.gen("net.src", 3, "10.168.7.121")
    assert h.gen("net.src", 3, "8.8.8.8") == "grp:∅"


def test_readdress_joins_provisionally_then_confirmed():
    sim = OrgSim()
    sim.days(4)
    m = sim.model()
    g = m["ip2g"]["192.168.2.11"]
    # 192.168.2.11 moves to 192.168.2.51 (DHCP); P03 flags the new address
    ips = dict(dept_ips())
    ips["FIN"] = ["192.168.2.10", "192.168.2.51", "192.168.2.12"]
    sim.readdr = [("192.168.2.51", "192.168.2.11")]
    sim.plan_day(sim.now + 8 * 3600.0, ips=ips)
    for _ in range(12):
        sim.tick()
    m = sim.model()
    assert m["ip2g"].get("192.168.2.51") == g
    assert "192.168.2.51" in m["groups"][g]["provisional"]
    for _ in range(12):
        sim.tick()
    sim.days(2, ips=ips)
    m = sim.model()
    assert m["ip2g"].get("192.168.2.51") == g
    assert "192.168.2.51" not in (m["groups"][g].get("provisional") or [])


def test_two_run_move_rule():
    """An IP whose behaviour changes department moves only after two
    consecutive runs agree (§6.15 item 4)."""
    sim = OrgSim()
    sim.days(5)
    g_fin = sim.model()["ip2g"]["192.168.2.12"]
    ips = dict(dept_ips())
    ips["FIN"] = ["192.168.2.10", "192.168.2.11"]
    ips["SALES"] = ips["SALES"] + ["192.168.2.12"]
    moved_at = []
    for d in range(10):
        sim.days(1, ips=ips)
        moved_at.append(sim.model()["ip2g"].get("192.168.2.12") != g_fin)
    first = moved_at.index(True)
    assert first >= 1                               # not on the first run that saw the change
    assert all(moved_at[first:])


def test_who_mode_none_contributes_nothing():
    """P12's strategy 'none' removes a system from the signatures; P05's own
    'IP carries no information' verdict is published (views, P03) but only
    advisory for grouping (which systems an IP uses is still who it is)."""
    sim = OrgSim()
    sim.st.put_model("crm", SYSTEM_ENTITY, MP.ATTRSEL, {"who_mode": "none", "roles": {}}, ts=T0)
    sim.st.put_model("finance", SYSTEM_ENTITY, MP.SYSPROF, {"chosen": {"who": "none"}}, ts=T0)
    sim.days(3)
    m = sim.model()
    assert m["mode"]["crm"] == {"mode": "none", "source": "attrsel", "bits": {}}
    assert m["mode"]["finance"]["mode"] == "none" and m["mode"]["finance"]["source"] == "sysprof"
    st = sim.st.get_model(ORG, ORG, WG.STATE)
    keys = set()
    for ip in dept_ips()["SALES"] + dept_ips()["FIN"]:
        ids, _ = st.sigs.weights(ip, sim.now)
        keys |= {st.sigs.items.key_of(int(i)) for i in ids}
    assert any(k.startswith("crm|") for k in keys)
    assert not any(k.startswith("finance|") for k in keys)


def test_sysprof_mode_prefix_keys_by_24():
    """Prefix mode pools an address's first-day rows per /24; an address seen
    on a second local day keeps its own signature (a recurrent client of a
    prefix-mode system is still a member of its department)."""
    sim = OrgSim()
    sim.st.put_model("crm", SYSTEM_ENTITY, MP.SYSPROF, {"chosen": {"who": "prefix"}}, ts=T0)
    sim.days(1)
    st = sim.st.get_model(ORG, ORG, WG.STATE)
    assert "192.168.3.0/24" in st.sigs
    ids, _ = st.sigs.weights("192.168.3.0/24", sim.now)
    assert any(st.sigs.items.key_of(int(i)).startswith("crm|") for i in ids)
    sim.days(2)
    st = sim.st.get_model(ORG, ORG, WG.STATE)
    ids, _ = st.sigs.weights("192.168.3.20", sim.now)
    assert any(st.sigs.items.key_of(int(i)).startswith("crm|") for i in ids)


def test_inert_when_disabled():
    sim = OrgSim(config={"progressive": {"enabled": False}})
    sim.days(2)
    assert sim.st.get_model(ORG, ORG, MP.WHO_GROUPS) is None


# ------------------------------------------------------------------ resources
def _public_run(n_ips: int, extra: int = 0, s_max: int = 2000, days: int = 2) -> Tuple[int, float, int]:
    """n_ips public IPs, each active on random days, one of 3 behaviours."""
    sim = OrgSim(extra_attrs=extra, s_max=s_max)
    rng = np.random.default_rng(1)
    pubs = [f"10.{60 + i // 65000}.{(i // 250) % 256}.{i % 250 + 1}" for i in range(n_ips)]
    rows = 0
    for d in range(days):
        day0 = sim.now
        act = [ip for ip in pubs if rng.random() < 0.5]
        for ip in act:
            kind = int(ipaddress.ip_address(ip)) % 3
            for r in range(3):
                ts = day0 + (1 + 22 * rng.random()) * 3600.0
                attrs = {"http.route": f"GET portal /p{kind}/{r}", "http.method": "GET"}
                for j in range(extra):
                    attrs[f"meta.f{j:03d}"] = float(rng.random())
                sim.pending.setdefault("portal", []).append((ts, ip, attrs))
                rows += 1
        for _ in range(24):
            sim.tick()
    st = sim.st.get_model(ORG, ORG, WG.STATE)
    return st.nbytes(), sim.t_eng / max(rows, 1), len(st.sigs)


def test_bounded_in_ips_and_attributes():
    """PPC-3 / S1: P11's state is capped by S_max whatever the population, and
    its cost per learned row does not depend on the number of attributes."""
    b1, us1, n1 = _public_run(1000)
    b2, us2, n2 = _public_run(8000)
    assert n1 < 2000 and n2 == 2000                  # the LRU cap binds
    assert b2 <= 2000 * 900 + 200_000                # <= S_max x ~0.9 KB (+ items)
    assert b2 / b1 < 8000 / 1000 / 2                 # far below linear growth
    assert us2 < 3.0 * us1 + 50e-6                   # per-row cost ~ flat (incl. clustering)
    b3, us3, _ = _public_run(1000, extra=300)
    assert abs(b3 - b1) <= 0.05 * b1                 # attributes never enter the state
    assert us3 < 2.0 * us1 + 50e-6


def test_ip_agnostic_decision_from_the_who_code():
    """The who mode of a system without P12's verdict: the level whose
    prequential two-part who code is shortest over the tree's leaves (§6.18.2);
    'none' when even the best level costs >= 32 bits (IP carries nothing)."""
    from app.engines.behavior.lib import ptree as PT
    st = make_store()

    def tree_with(code):
        m = PT.PTreeModel("x")
        tr = m.tree(EV.KIND_TXN, T0, create=True)
        leaf = tr.nodes[tr.root]
        leaf.who.code = np.asarray(code, dtype=float) * 100.0
        leaf.who.code_n = 100.0
        st.put_model("x", SYSTEM_ENTITY, MP.PTREE, m, version=1)
    tree_with([9.0, 3.0, 6.0, 0.0, 0.0])            # a DHCP-like population: /24 cheapest
    d = WG.decide_mode(st, "x")
    assert d["mode"] == "prefix" and d["source"] == "code" and d["bits"]["p24"] == pytest.approx(3.0)
    tree_with([2.0, 5.0, 9.0, 0.0, 0.0])            # a stable department: /32 cheapest
    assert WG.decide_mode(st, "x")["mode"] == "ip"
    tree_with([40.0, 36.0, 33.0, 0.0, 0.0])         # random sources: nothing beats 32 bits
    assert WG.decide_mode(st, "x")["mode"] == "none"
    st.put_model("x", SYSTEM_ENTITY, MP.ATTRSEL, {"who_mode": "none"})
    tree_with([2.0, 5.0, 9.0, 0.0, 0.0])
    assert WG.decide_mode(st, "x") ["mode"] == "none"           # P05: IP carries no information
    st.put_model("x", SYSTEM_ENTITY, MP.SYSPROF, {"chosen": {"who": "grp"}})
    assert WG.decide_mode(st, "x") == {"mode": "grp", "source": "sysprof"}


def test_pattern_item_is_the_context_part_of_the_covering_node():
    """The IP x pattern item climbs out of exception nodes and content-split
    children (a net.bytes_up bin is not a who signal) to the nearest node
    reached by a context split; the root means no pattern item."""
    from types import SimpleNamespace as NS
    from app.engines.behavior import who_groups as WG

    def node(nid, parent, split=None, exc=False):
        return NS(id=nid, parent=parent, split=NS(attr=split) if split else None, is_exc=exc)
    nodes = {0: node(0, None, "http.route"), 1: node(1, 0, "net.bytes_up"),
             2: node(2, 1, "ctx.daytype"), 3: node(3, 1), 4: node(4, 2), 5: node(5, 4, exc=True),
             6: node(6, 0, "net.src"), 7: node(7, 6, "http.path"), 8: node(8, 7)}
    tree = NS(nodes=nodes, root=0)
    c = {}
    assert WG._pattern_node(tree, 1, c) == 0           # a route child: the action item has it
    assert WG._pattern_node(tree, 2, c) == 0           # a bytes bin of an action: nothing
    assert WG._pattern_node(tree, 3, c) == 0
    assert WG._pattern_node(tree, 4, c) == 4           # a daytype child is context
    assert WG._pattern_node(tree, 5, c) == 4           # an exception node -> its owner
    # an address split (net.src subset) is not behaviour: it gives no pattern
    # item (integration 2026-09-30, see who_groups.CONTEXT_PREFIX)
    assert WG._pattern_node(tree, 7, c) == 0
    assert WG._pattern_node(tree, 8, c) == 0


def test_join_queue_is_fair_to_every_address():
    """More ungrouped sources than JOIN_PER_TICK: a member-like source with a
    high address is still tried (never-tried first, then longest-waiting),
    not starved behind lower addresses that are re-queued every tick."""
    eng = WG.WhoGroupsEngine()
    st = WG.WGState()
    t = 1_000_000.0
    items = [f"k|GET /a{j}" for j in range(6)]
    for j, it in enumerate(items):
        for m in ("192.168.3.20", "192.168.3.21", "192.168.3.22", "192.168.3.33"):
            st.sigs.add(m, it, t, 10.0 + j, 5.0)
    rnd = MH.Randoms()
    st.reps = {"G1": np.vstack([eng._minhash(st, m, t, rnd) for m in ("192.168.3.20", "192.168.3.21",
                                                                       "192.168.3.22")])}
    st.cohesion = {"G1": 1.0}
    model = {"ip2g": {m: "G1" for m in ("192.168.3.20", "192.168.3.21", "192.168.3.22")},
             "groups": {"G1": {"members": ["192.168.3.20", "192.168.3.21", "192.168.3.22"], "n": 3}}}
    visitors = [f"10.60.{i // 250}.{i % 250 + 1}" for i in range(WG.JOIN_PER_TICK + 50)]
    for v in visitors:
        st.sigs.add(v, "p|GET /", t, 5.0, 5.0)
    joined = None
    for tick in range(3):
        st.join_queue = set(visitors) | {"192.168.3.33"}
        eng._joins(st, t + tick * 3600.0, model)
        if "192.168.3.33" in model["ip2g"]:
            joined = tick
            break
    assert joined is not None and joined <= 1
    assert model["ip2g"]["192.168.3.33"] == "G1"



def test_member_slightly_off_a_large_tight_group_stays_in_it():
    """20 SALES members with near-identical signatures fill each other's 10
    nearest neighbours, so the mutual-kNN graph never links a member with one
    small extra item (pack O A9: a sales IP that also reads finance's approval
    list once a day; J 0.92 to its colleagues at cohesion 0.95). The run
    applies the join rule to the eligible sources it left ungrouped: the
    member stays in its department (pack O day 21: ARI 0.86 -> 0.97)."""
    for seed in range(6):
        rng = np.random.default_rng(seed)
        st = WG.WGState()
        eng = WG.WhoGroupsEngine()
        model = WG.empty_model()
        store = make_store()
        items = {"crm|GET crm /c": 40, "crm|POST crm /v": 15, "mail|TLS mail": 20,
                 "oa|GET oa /docs": 25, "oa|POST oa /login": 5}
        ips = [f"192.168.3.{20 + i}" for i in range(20)]
        for d in range(10):
            t = T0 + d * DAY + 36000.0
            for ip in ips:
                for it, m in items.items():
                    st.sigs.add(ip, it, t, m * (1 + 0.15 * rng.standard_normal()), 3.0, d)
            st.sigs.add(ips[13], "finance|GET fin /list", t, 0.15, 1.0, d)
            for k in range(3):
                st.sigs.add(f"192.168.2.{10 + k}", "finance|GET fin /ledger", t,
                            30 * (1 + 0.15 * rng.standard_normal()), 3.0, d)
                st.sigs.add(f"192.168.2.{10 + k}", "mail|TLS mail", t, 20.0, 3.0, d)
        now = T0 + 10 * DAY
        eng._cluster(ctx(store, now), st, model, now)
        ip2g = model["ip2g"]
        assert ip2g.get(ips[0]) is not None
        assert all(ip2g.get(ip) == ip2g[ips[0]] for ip in ips), seed


def test_group_record_states_what_the_group_does_per_system(org7):
    """The group record carries the group's action mix per system (the user
    view's '综合部 访问 OA：登录、审批…'), and names the members of an action
    only some of them perform."""
    m = org7.model()
    g = m["ip2g"]["192.168.1.21"]
    acts = m["groups"][g]["actions"]
    oa = {a["action"]: a for a in acts["oa"]}
    assert {"POST oa /login", "GET oa /approval/list", "POST oa /approval/{num}"} <= set(oa)
    assert all(a["support"] == 1.0 and a["members"] == [] for a in oa.values())
    assert "mail" in acts
    # members of an action only some of them do (unit, on the helper)
    items = MH.ItemDict()
    i_login, i_appr = items.id_of("oa|POST oa /login"), items.id_of("oa|POST oa /approval/{num}")
    prof = [{i_login: 1.0, i_appr: 1.0}, {i_login: 1.0}, {i_login: 1.0}]
    gp = {i_login: 3.0 / 2 + 1.0, i_appr: 0.5}
    out = WG._group_actions(gp, sum(gp.values()), [0, 1, 2], prof, ["192.168.1.21", "192.168.1.23", "10.168.7.121"],
                            items)
    appr = next(a for a in out["oa"] if a["action"] == "POST oa /approval/{num}")
    assert appr["members"] == ["192.168.1.21"] and appr["support"] == pytest.approx(1 / 3, abs=1e-3)


def _role_org(seed: int = 0) -> Tuple[Dict[str, Any], Any]:
    """Signatures of pack O's shape: 综合部's approver (.21) and its two report
    writers (.23, .121) behave differently; prefix-mode pool sources (one-shot
    addresses pooled per /24) behave like them; 财务部 is elsewhere."""
    rng = np.random.default_rng(seed)
    st = WG.WGState()
    eng = WG.WhoGroupsEngine()
    store = make_store()
    prog = {"appr": {"oa|POST oa /login": 5, "oa|GET oa /docs": 10, "oa|GET oa /approval/list": 15,
                     "oa|POST oa /approval/{num}/approve": 15},
            "rep": {"oa|POST oa /login": 5, "oa|GET oa /docs": 10, "oa|GET oa /report/form": 12,
                    "oa|POST oa /report/generate": 12, "mail|TLS mail": 8},
            "fin": {"finance|POST fin /fin/login": 5, "finance|GET fin /fin/ledger": 20,
                    "finance|POST fin /fin/voucher": 10, "mail|TLS mail": 8}}
    who = {"192.168.1.21": "appr", "192.168.1.0/24": "appr",
           "192.168.1.23": "rep", "10.168.7.121": "rep", "10.168.7.0/24": "rep", "192.168.2.0/24": "rep",
           "192.168.2.10": "fin", "192.168.2.11": "fin", "192.168.2.12": "fin"}
    for d in range(10):
        t = T0 + d * DAY + 36000
        for src, p in who.items():
            for it, m in prog[p].items():
                st.sigs.add(src, it, t, m * (1 + 0.1 * rng.standard_normal()), 3.0, d)
    cfg = {"progressive": {"enabled": True},
           "who_group_names": [{"name": "综合部", "ips": dept_ips()["GA"]},
                               {"name": "财务部", "ips": dept_ips()["FIN"]}]}
    model = WG.empty_model()
    now = T0 + 10 * DAY
    eng._cluster(ctx(store, now, config=cfg), st, model, now)
    return model, st


def test_department_names_are_matched_on_addresses_and_roles_are_named():
    """Configured names are matched on the groups' ADDRESSES (pool sources are
    not people of the department), pool sources are published apart from the
    members and cover a group only where its addresses are the pool's recurring
    population, and a learned role inside a configured department is named
    '<department>·<its top action>' (pack O: 综合部's report writers carried two
    pool sources, Jaccard 2/5 < 0.5, and stayed 'G10·…'; 192.168.2.0/24 - 财务部's
    subnet - was rendered as a member of 综合部's login statement)."""
    model, st = _role_org()
    ip2g = model["ip2g"]
    g_rep, g_appr = ip2g["192.168.1.23"], ip2g["192.168.1.21"]
    assert ip2g["10.168.7.121"] == g_rep and g_appr != g_rep
    rep, appr = model["groups"][g_rep], model["groups"][g_appr]
    assert rep["name"] == "综合部" and rep["name_source"] == "config" and rep["dept"] == "综合部"
    assert set(rep["members"]) == {"192.168.1.23", "10.168.7.121"}
    assert set(rep["pools"]) == {"10.168.7.0/24", "192.168.2.0/24"}
    assert "192.168.2.0/24" not in rep["covers"] and "10.168.7.0/24" in rep["covers"]
    assert appr["name"].startswith("综合部·") and appr["dept"] == "综合部"
    assert appr["members"] == ["192.168.1.21"]
    fin = model["groups"][ip2g["192.168.2.10"]]
    assert fin["name"] == "财务部" and fin["dept"] == "财务部"
    # what the group does is stated over its addresses: an action all of them do
    # lists no members, never a pool source
    for lst in rep["actions"].values():
        for a in lst:
            assert not any("/" in m for m in a["members"])
    gen = next(a for a in rep["actions"]["oa"] if a["action"] == "POST oa /report/generate")
    assert gen["members"] == [] and gen["support"] == 1.0
    items = MH.ItemDict()
    i_login, i_rep = items.id_of("oa|POST oa /login"), items.id_of("oa|POST oa /report/generate")
    prof = [{i_login: 1.0, i_rep: 1.0}, {i_login: 1.0}, {i_login: 1.0, i_rep: 1.0}]
    out = WG._group_actions({i_login: 3.0, i_rep: 2.0}, 5.0, [0, 1, 2], prof,
                            ["192.168.1.23", "10.168.7.121", "192.168.2.0/24"], items)
    rp = next(a for a in out["oa"] if a["action"] == "POST oa /report/generate")
    assert rp["members"] == ["192.168.1.23"] and rp["support"] == pytest.approx(0.5)
