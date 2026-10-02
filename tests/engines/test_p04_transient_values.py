"""P04: a split never names a transient value of a learned coarsening
(evaluator round 3, docs/lib3/progressive.md §16.11).

grp:∅ (a source P11 has not grouped yet) is a state, not a population: a
source leaves it when P11 groups it. A child named on it collected pack O's
研发 DHCP leases on day 3.4 (seed 4); once P11 pooled 研发 the re-leased
addresses it had grouped were routed to the departments' `other` node and
flagged `outsider_group` (FAR >= MEDIUM 0.0028 -> 0.0067)."""
from __future__ import annotations

from ptree_sim import DAY, MON, Sim, daily

from app.engines.behavior import pattern_tree as P04
from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pevalue as PE
from app.engines.behavior.lib.phier import GRP_NONE

# twelve one-address groups (more values than the split statistics track, so
# the grouped sources share the `other` slot) and six ungrouped addresses
GROUPED = [f"192.168.1.{i}" for i in range(10, 22)]
UNGROUPED = [f"192.168.1.{i}" for i in range(100, 106)]   # same /24: only the group level separates


def _run(days):
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: [("net.src", 3)]}, "roles": {}}
    sim = Sim(sel=sel)
    ip2g = {ip: f"g{k}" for k, ip in enumerate(GROUPED)}
    sim.st.put_model("__org__", "__org__", MP.WHO_GROUPS,
                     {"ip2g": ip2g, "groups": {g: {"members": [ip]} for ip, g in ip2g.items()}})

    def day(d, t0, rng):
        ev = []
        for ip in GROUPED:
            for _ in range(3):
                ev.append((t0 + rng.uniform(9, 12) * 3600, "oa", ip,
                           {"http.route": "POST oa /login", "net.bytes_up": float(rng.uniform(1024, 2048))}))
        for ip in UNGROUPED:
            for _ in range(6):
                ev.append((t0 + rng.uniform(13, 17) * 3600, "oa", ip,
                           {"http.route": "POST oa /login", "net.bytes_up": float(rng.uniform(6000, 9000))}))
        return ev
    sim.add(daily(day, days, seed=5))
    sim.run_until(MON + days * DAY)
    return sim


def test_named_groups_drop_the_ungrouped_value():
    assert P04._named_groups([[GRP_NONE], ["grp:g1", GRP_NONE], []]) == [["grp:g1"]]
    assert P04._named_groups([["10.0.0.0/24"]]) == [["10.0.0.0/24"]]


def test_no_child_is_named_on_the_ungrouped_value():
    sim = _run(8)
    tr = sim.tree()
    named = [(nd.id, sorted(map(str, vals))) for nd in tr.nodes.values()
             for a, l, vals, neg in nd.ctx[-1:] if a == "net.src" and l == 3 and not neg]
    assert all(GRP_NONE not in v for _, v in named), named
    # the candidate gave way: the two populations are not left mixed for good
    # by a refused group split (it yields its slot like a constant candidate)
    leaf = tr.nodes[sim.top()]
    if leaf.split is None:
        assert ("net.src", 3) in (leaf.meta.get("const") or set()) or leaf.split_stats is not None


def test_a_refused_transient_split_marks_the_candidate_constant():
    """_do_split on a decision whose only named group is grp:∅ refuses and
    makes the candidate yield its slot until the next restart."""
    sim = _run(2)
    tr = sim.tree()
    leaf = tr.nodes[sim.top()]
    assert leaf.split is None
    dec = PE.SplitDecision(0, None)
    dec.groups, dec.other_group = [[GRP_NONE], ["grp:g1", PE.OTHER]], 1
    eng = sim.p04
    n0 = len(tr.nodes)

    class _LC:
        pass
    assert eng._do_split(_LC(), tr, leaf, 0, dec, ("net.src", 3), sim.now) is False
    assert len(tr.nodes) == n0 and leaf.split is None
    assert ("net.src", 3) in leaf.meta["const"]
