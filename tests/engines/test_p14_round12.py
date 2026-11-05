"""P14 views, round 12 (views / groups / bindings owner)."""
from __future__ import annotations

from types import SimpleNamespace as NS

from test_p14_round9 import _Lvl
from test_p14_round11 import _Sigs

from app.engines.behavior import views as VW

DAY = 86400.0
LOGIN = "POST oa /login"
GA = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
FIN = ["192.168.2.10", "192.168.2.11", "192.168.2.12"]


def _fam_login(groups, readdr):
    """O-real fam:1, day 35: the family login node, every address logging in."""
    now = 35 * DAY
    ip2g = {m: g for g, gr in groups.items() for m in gr["members"]}
    item = f"fam:1|{LOGIN}"
    ips = sorted({m for gr in groups.values() for m in gr["members"]} | set(GA) | set(FIN))
    sigs = _Sigs({ip: ((9 * DAY) if ip == "192.168.2.11" else (now - 3600.0), {item: 1.0, "x": 1.0})
                  for ip in ips})
    cfg = {"who_group_names": [{"name": "综合部", "ips": list(GA)}, {"name": "财务部", "ips": list(FIN)}]}
    c = NS(ip2g=ip2g, mode="grp", groups=groups, sigs=sigs, key="fam:1", config=cfg, readdr=readdr)
    lv3 = [(f"grp:{g}", 3.0) for g in sorted(groups)]
    lv0 = [(ip, 1.0) for ip in ips if ip != "192.168.2.11"]
    nd = NS(ctx=[], who=NS(levels=[_Lvl(lv0), _Lvl([]), _Lvl([]), _Lvl(lv3)]))
    return {p[0]: p[1] for p in VW.group_parts(c, nd, now, 0.0, LOGIN)}


def test_a_moved_member_is_a_colleague_of_its_old_department_only():
    """O-real seed 2 (round 11, §16.19.11 item 8): P11's G22 mixes 综合部's
    approver 192.168.1.21 with 财务部's .2.10 / .2.12 and the moved clerk .2.51
    (readdressed from the listed .2.11); T11-4 added .2.51 to every configured
    综合部 part of the family login (who Jaccard 1.00 -> 0.75). The moved
    clerk is 财务部's: 综合部's part names its three addresses, 财务部's the
    clerk at the new address."""
    groups = {"G22": {"name": "G22", "members": ["192.168.1.21", "192.168.2.10", "192.168.2.12",
                                                 "192.168.2.51"]},
              "G23": {"name": "G23", "members": ["10.168.7.121", "192.168.1.23"]}}
    by = _fam_login(groups, {"192.168.2.11": "192.168.2.51"})
    assert by["cfg:综合部"] == ["10.168.7.121", "192.168.1.21", "192.168.1.23"]
    assert by["cfg:财务部"] == ["192.168.2.10", "192.168.2.12", "192.168.2.51"]
    # seed 0's G18 (.21 / .2.12 / .2.51: one address of each department) likewise
    groups = {"G18": {"name": "G18", "members": ["192.168.1.21", "192.168.2.12", "192.168.2.51"]},
              "G19": {"name": "G19", "members": ["10.168.7.121", "192.168.1.23"]},
              "G20": {"name": "G20", "members": ["192.168.2.10"]}}
    by = _fam_login(groups, {"192.168.2.11": "192.168.2.51"})
    assert "192.168.2.51" not in by["cfg:综合部"]
    assert "192.168.2.51" in by["cfg:财务部"]


def test_an_unlisted_colleague_follows_the_majority_of_its_group():
    """Without a move, an unlisted address of a learned group is the colleague
    of the department holding most (and at least half) of the group's
    configured members; a group split evenly says nothing about it."""
    groups = {"G22": {"name": "G22", "members": ["192.168.1.21", "192.168.2.10", "192.168.2.12",
                                                 "192.168.2.77"]},
              "G23": {"name": "G23", "members": ["10.168.7.121", "192.168.1.23"]}}
    by = _fam_login(groups, {})
    assert "192.168.2.77" not in by["cfg:综合部"] and "192.168.2.77" in by["cfg:财务部"]
    groups = {"G18": {"name": "G18", "members": ["192.168.1.21", "192.168.2.12", "192.168.2.77"]},
              "G19": {"name": "G19", "members": ["10.168.7.121", "192.168.1.23"]},
              "G20": {"name": "G20", "members": ["192.168.2.10"]}}
    by = _fam_login(groups, {})
    assert all("192.168.2.77" not in m for m in by.values() if m is not None and m != groups["G18"]["members"])
    assert "192.168.2.77" not in by.get("cfg:综合部", []) and "192.168.2.77" not in by.get("cfg:财务部", [])


def test_a_prefix_who_lists_every_seen_prefix_that_fits():
    """O60 seed 0 (round 11, day 7): GET /docs (25 sources, too many to list)
    was stated from three /24s - the heavy set stops at 90 % of the mass - and
    left out 10.168.7.0/24 (综合部's 10.168.7.121) at a stated who of 0.96;
    the address's docs rows were more of the held-out events than the decayed
    summary's share, and the claim failed on who alone. Every prefix the node
    has seen is listed when they all fit in PREFIX_MAX; the stated unseen mass
    is then the level's own."""
    from app.engines.behavior.lib import pnode as PN
    from app.engines.behavior.lib import prender as PR
    w = PN.WhoSummary()
    t = 0.0
    srcs = ([f"192.168.3.{20 + i}" for i in range(20)] + ["192.168.1.21", "192.168.1.23"]
            + ["192.168.2.10", "192.168.2.11", "192.168.2.12", "10.168.7.121"])
    for d in range(10):
        for j, ip in enumerate(srcs):
            t = d * DAY + 30.0 * j
            p24 = ip.rsplit(".", 1)[0] + ".0/24"
            w.update([ip, p24, None, None, None], ip, t, 1.0, 1.0)
    ev, zh, en, c = PR.who_block(w, t + 60.0, 10, {}, {})
    assert ev["level"] == "prefix"
    assert set(ev["items"]) == {"192.168.3.0/24", "192.168.1.0/24", "192.168.2.0/24", "10.168.7.0/24"}
    assert ev["U"] < 0.02


def test_a_group_who_states_the_members_that_left_the_group():
    """O60 seed 0 (round 12 views v6, days 6-7): P11 split 销售部's group G11
    (20 addresses) into G11 (18) and a pair {.3.34, .3.38}. The crm nodes'
    group-level summary attributes an event to its source's group at learning
    time, so G11's share (0.95) still held the pair's events from before the
    split; the statements named G11's 18 current members at a stated who of
    0.95 and held 0.86-0.89 on the evaluator's events. The members that left
    the group recently (P11 rec 'left') are part of the group's who."""
    from app.engines.behavior.lib import pnode as PN
    from app.engines.behavior.lib import prender as PR
    sales = [f"192.168.3.{20 + i}" for i in range(20)]
    gone = ["192.168.3.34", "192.168.3.38"]
    w = PN.WhoSummary()
    t = 0.0
    for d in range(6):
        for j, ip in enumerate(sales):
            t = d * DAY + 30.0 * j
            w.update([ip, "192.168.3.0/24", None, "grp:G11", None], ip, t, 1.0, 1.0)
    cur = [ip for ip in sales if ip not in gone]
    groups = {"G11": {"members": cur, "name": "销售部", "left": {ip: 5 * DAY for ip in gone}},
              "G65": {"members": gone, "name": "销售部·crm POST /crm/visit"}}
    ip2g = {**{ip: "G11" for ip in cur}, **{ip: "G65" for ip in gone}}
    ev, zh, en, c = PR.who_block(w, t + 60.0, 6, ip2g, groups)
    assert ev["level"] == "grp" and ev["items"] == ["grp:G11"]
    assert set(gone) <= set(ev["members"]) and set(cur) <= set(ev["members"])
    assert ev["left_group"] == gone
    assert "192.168.3.34" in zh
    # without the record, the who is the current members only (round-11 behaviour)
    groups["G11"].pop("left")
    ev2 = PR.who_block(w, t + 60.0, 6, ip2g, groups)[0]
    assert not set(gone) & set(ev2["members"])
