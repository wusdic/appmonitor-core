"""P11 round 3 (groups_views owner): pools, single-member roles, id lineage.

Pack O, round 2: the 研发 DHCP pool (24-h leases in 10.50.0.0/22) was 5-10
learned groups on every seed (dev_pool_grouped 0/5); the finance approver
192.168.2.10 had no learned group; 综合部's group got a new id on day 17 (A5
was then scored against a two-day scope)."""
from __future__ import annotations

import ipaddress
from typing import Dict, List

import numpy as np

from app.engines.behavior import who_groups as WG

from test_p11_who_groups import DAY, OrgSim

POOL = ipaddress.ip_network("10.50.0.0/22")
N_DEV = 40
# each developer's own mix: everybody pushes code, some read mail in the
# morning, some log in to OA - so one day of one lease is a SLICE of the pool
DEV_MIX = [[("code", "TLS:git.corp", 6)] + ([("mail", "TLS:mail.corp", 3)] if k % 3 else [])
           + ([("oa", "POST oa /login", 1)] if k % 4 == 0 else []) for k in range(N_DEV)]


class PoolSim(OrgSim):
    """GA / FIN / SALES at fixed addresses, 研发 on 24-h leases of a /22."""

    def __init__(self, config=None, **kw) -> None:
        super().__init__(config, **kw)
        self.leases: List[Dict[int, str]] = []

    def plan_day(self, t_day: float, depts=("GA", "FIN", "SALES"), ips=None) -> None:
        super().plan_day(t_day, depts=depts, ips=ips)
        hosts = list(POOL.hosts())
        pick = self.rng.choice(len(hosts), size=N_DEV, replace=False)
        lease = {k: str(hosts[int(j)]) for k, j in enumerate(pick)}
        self.leases.append(lease)
        for k, ip in lease.items():
            for sysn, act, per in DEV_MIX[k]:
                for _ in range(int(self.rng.poisson(per)) + 1):
                    ts = t_day + (9 + 8 * self.rng.random()) * 3600.0 - 8 * 3600.0
                    attrs = {"tls.sni": act[4:], "net.bytes_up": float(self.rng.integers(100, 9000))} \
                        if act.startswith("TLS:") else {"http.route": act, "http.method": act.split()[0]}
                    self.pending.setdefault(sysn, []).append((ts, ip, attrs))


def _pool_groups(m) -> Dict[str, int]:
    out: Dict[str, int] = {}
    for g, gr in (m.get("groups") or {}).items():
        k = sum(1 for x in gr.get("members") or [] if "/" not in x and ipaddress.ip_address(x) in POOL)
        if k:
            out[g] = k
    return out


def _run_pool(config) -> List[Dict[str, int]]:
    sim = PoolSim(config=config, seed=11)
    days = []
    for _ in range(16):
        sim.days(1)
        days.append((_pool_groups(sim.model()), sim.model()))
    return days


def _check_pool(days) -> None:
    pg, m = days[-1]
    g, k = max(pg.items(), key=lambda kv: kv[1])
    assert k >= 0.9 * sum(pg.values()), pg                     # one group holds the pool
    assert str(POOL) in (m["groups"][g].get("covers") or [])    # ... and covers its prefix
    # the pool group keeps its id over the last week (lineage, no re-forming)
    ids = [max(p.items(), key=lambda kv: kv[1])[0] for p, _ in days[-7:] if p]
    assert len(set(ids)) == 1, ids


def test_configured_dhcp_scope_is_one_group_covering_its_prefix():
    _check_pool(_run_pool({"dhcp_scopes": [{"cidr": str(POOL), "name": "研发 DHCP"}]}))


def test_turnover_prefix_is_learned_as_a_pool_without_configuration():
    _check_pool(_run_pool(None))


def test_learned_pools_skip_configured_regions():
    sim = PoolSim(config={"ip_classes": [{"name": "dev-net", "cidrs": ["10.50.0.0/16"]}]}, seed=2)
    sim.days(9)
    st = sim.st.get_model("__org__", "__org__", WG.STATE)
    assert WG._pool_nets(sim.cfg, st.sigs, sim.now) == []


def test_single_member_functional_role_forms_its_own_group():
    """财务部: .10 approves (nobody else does), .11 / .12 book vouchers. The
    approver is a stable client the clustering leaves alone; it is a role of
    the department, not an ungrouped address."""
    from test_p11_who_groups import PROGRAM
    prog = dict(PROGRAM)
    prog["FINAP"] = [("finance", "POST fin /fin/login", 1), ("finance", "GET fin /fin/approval/list", 4),
                     ("finance", "POST fin /fin/approval/{num}/approve", 4)]
    import test_p11_who_groups as T
    old = T.PROGRAM
    T.PROGRAM = prog
    try:
        sim = OrgSim(config={"who_group_names": [
            {"name": "财务部", "ips": ["192.168.2.10", "192.168.2.11", "192.168.2.12"]}]}, seed=4)
        sim.ips = dict(sim.ips, FIN=["192.168.2.11", "192.168.2.12"], FINAP=["192.168.2.10"])
        for _ in range(12):
            day0 = sim.now
            if ((int((day0 + 8 * 3600) // DAY) + 3) % 7) < 5:
                sim.plan_day(day0 + 8 * 3600.0, depts=("GA", "FIN", "FINAP", "SALES", "DEV"))
            for _ in range(24):
                sim.tick()
    finally:
        T.PROGRAM = old
    m = sim.model()
    g = m["ip2g"].get("192.168.2.10")
    assert g is not None and m["groups"][g]["members"] == ["192.168.2.10"]
    assert m["groups"][g]["dept"] == "财务部" and m["groups"][g]["name"].startswith("财务部·")
    assert m["ip2g"].get("192.168.2.11") != g
    # departments' members never become one-address roles
    assert all(len(gr["members"]) >= 2 for gg, gr in m["groups"].items() if gg != g)


def test_dissolved_group_gets_its_id_back_when_it_re_forms():
    sim = OrgSim(seed=5)
    sim.days(8)
    m = sim.model()
    g_ga = m["ip2g"]["192.168.1.21"]
    st = sim.st.get_model("__org__", "__org__", WG.STATE)
    # one run that dissolved the group (e.g. its members' rows held for a day)
    st.__dict__.setdefault("retired", {})[g_ga] = (st.prev_members.pop(g_ga), sim.now)
    sim.days(1)
    assert sim.model()["ip2g"]["192.168.1.21"] == g_ga


def test_group_identity_is_matched_on_its_addresses_not_its_pool_sources():
    """综合部 {.23, .121} picks up two /24 pool sources that formed a group of
    their own the run before: it keeps its id (Jaccard over all sources tied
    0.5 / 0.5 and the id went to the pool-only group on pack O seed 1)."""
    prev = {"G12": {"192.168.1.23", "10.168.7.121"}, "G32": {"10.168.7.0/24", "192.168.2.0/24"}}
    new = [{"192.168.1.23", "10.168.7.121", "10.168.7.0/24", "192.168.2.0/24"}]
    assert WG._inherit_ids(prev, new) == {0: "G12"}
    # a merge keeps the id of the largest predecessor it contains
    prev = {"G5": {f"10.50.0.{i}" for i in range(1, 20)}, "G6": {f"10.50.1.{i}" for i in range(1, 8)}}
    new = [{f"10.50.0.{i}" for i in range(1, 20)} | {f"10.50.1.{i}" for i in range(1, 8)}
           | {f"10.50.2.{i}" for i in range(1, 40)}]
    assert WG._inherit_ids(prev, new) == {0: "G5"}


def test_group_name_keeps_its_labels_while_they_still_describe_the_group():
    """Display names do not flip between near-equal top actions every run (the
    approver's role was '综合部·oa GET /approval/{num}' one day and '…/list' the
    next on pack O: a group_changed event and a renamed view each day)."""
    sim = OrgSim(seed=6)
    sim.days(8)
    m = sim.model()
    g = m["ip2g"]["192.168.3.20"]
    labels = [x["label"] for x in m["groups"][g]["labels"]]
    assert len(labels) >= 3
    alt = [labels[2], labels[1]]                   # not today's top-2
    m["groups"][g]["name_labels"] = alt
    m["groups"][g]["name"] = f"{g}·" + "+".join(alt)
    sim.st.put_model("__org__", "__org__", "model.who_groups", m)
    sim.days(1)
    m2 = sim.model()
    lab2 = {x["label"] for x in m2["groups"][g]["labels"]}
    assert set(alt) <= lab2
    assert m2["groups"][g]["auto_name"] == f"{g}·" + "+".join(alt)


def test_merged_group_records_its_predecessors_as_lineage():
    """P03 reads a group's label history through rec 'lineage' (node summaries
    carry the label of learning time): the pool group merged from slices keeps
    the largest slice's id and lists the others."""
    sim = PoolSim(config={"dhcp_scopes": [{"cidr": str(POOL), "name": "研发 DHCP"}]}, seed=3)
    sim.days(6)
    m = sim.model()
    pg = _pool_groups(m)
    g = max(pg, key=pg.get)
    st = sim.st.get_model("__org__", "__org__", WG.STATE)
    mem = sorted(st.prev_members[g])
    a, b = set(mem[: len(mem) * 2 // 3]), set(mem[len(mem) * 2 // 3:])
    st.prev_members = {k: v for k, v in st.prev_members.items() if k != g}
    st.prev_members.update({g: a, "G900": b})                  # as if two slices last run
    sim.days(1)
    m2 = sim.model()
    pg2 = _pool_groups(m2)
    g2 = max(pg2, key=pg2.get)
    assert g2 == g and "G900" in (m2["groups"][g2].get("lineage") or [])
