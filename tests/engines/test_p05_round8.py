"""P05 round 8: per-attribute evaluation due times (docs/lib3/progressive.md
§16.15.12 item 6, OI-i). When a key has more attributes than one run judges
(the rotation pool does not fit a slice: PG4's 340-attribute point), an
attribute whose role and role statistics hold is re-judged less and less
often (1, 2, 4, 8 run-hours) and a change brings it back to every run. Before,
the cadence was per key: one flapping attribute kept the whole selection
hourly. While every attribute fits a run (pack O) the selection is judged as
before."""
from __future__ import annotations

import math

from ptree_sim import DAY, MON, Sim, daily

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib import pselect as SEL
from app.models.schema import SYSTEM_ENTITY

from test_p05_attr_select import _org


def _st(**kw):
    st = {"U_t": 0.30, "U_tc": 0.30, "U_s_max": 0.40, "CR_p": 0.50, "H": 1.50, "H_p": 1.50, "cov": 0.60,
          "cov_rows": 0.60, "distinct0": 0.01, "CR0": 0.50, "S": 1.0, "n_p": 400, "card": {0: 40},
          "level": 1, "best_levels": [1]}
    st.update(kw)
    return st


def test_a_change_is_a_crossing_of_a_role_threshold_not_a_move():
    """Roles are decided by the side of each threshold: a statistic that stays
    on its side does not bring the attribute back to every run, however far it
    moves. Measured on PG4's 340-attribute point: a split attribute's U_s_max
    swings 1.0 -> 0.15 -> 0.5 bits from hour to hour, far above U_HI; counted
    as a move (against the distance to U_HI) it kept 1 069 evaluations a
    5-day run hourly while no role changed."""
    base = SEL.due_signature(_st())
    assert not SEL.stats_moved(base, SEL.due_signature(_st(U_t=0.33, U_tc=0.33, U_s_max=0.37, H=1.45)))
    swing = SEL.due_signature(_st(U_s_max=1.0))
    assert not SEL.stats_moved(swing, SEL.due_signature(_st(U_s_max=0.15)))
    # next to u_hi (0.05): still above it is no change, below it is
    near = SEL.due_signature(_st(U_t=0.07, U_tc=0.07))
    assert not SEL.stats_moved(near, SEL.due_signature(_st(U_t=0.06, U_tc=0.06)))
    assert SEL.stats_moved(near, SEL.due_signature(_st(U_t=0.04, U_tc=0.04)))
    # coverage across TARGET_COV, the level-0 cardinality across a closed set's size
    assert SEL.stats_moved(base, SEL.due_signature(_st(cov=0.04, cov_rows=0.04)))
    # a level or the set of best split levels changing is a change; their order is not
    assert SEL.stats_moved(base, SEL.due_signature(_st(level=2)))
    assert SEL.stats_moved(base, SEL.due_signature(_st(best_levels=[1, 2])))
    two = SEL.due_signature(_st(best_levels=[1, 2]))
    assert not SEL.stats_moved(two, SEL.due_signature(_st(best_levels=[2, 1])))
    # never evaluated (or a signature of another format)
    assert SEL.stats_moved(None, base)
    assert SEL.stats_moved((0.3, 0.4), base)
    assert not SEL.stats_moved(base, base)


def test_the_due_signature_is_one_small_integer():
    """model.attrsel keeps a signature for every attribute of every key: at
    PG4's 340-attribute point a 14-tuple of statistics per attribute was ~1 MB
    of the 7-day memory (attribute-memory slope 0.187 -> 0.210); the side bits
    and levels it is compared on fit one int."""
    import sys
    sig = SEL.due_signature(_st(best_levels=[1, 3], level=2))
    assert type(sig) is int and sys.getsizeof(sig) <= 32
    assert SEL.due_signature(_st(U_s_max=-math.inf)) != SEL.due_signature(_st())


def test_due_hours_nest_so_that_attributes_fall_due_together():
    assert SEL.due_after(13, 0) == 14
    assert SEL.due_after(13, 1) == 14
    assert SEL.due_after(13, 2) == 16
    assert SEL.due_after(13, 3) == 16
    assert SEL.due_after(16, 3) == 24
    due = {"a": (16, 3), "b": (14, 1)}
    assert SEL.is_due(due, "c", 13)                     # never evaluated: due
    assert not SEL.is_due(due, "a", 15) and SEL.is_due(due, "a", 16)
    names = ["a", "b", "c"]
    roles = {"a": "target", "b": "split", "c": "probe"}
    assert SEL.names_to_evaluate(names, roles, {}, 0.0, 0, due=due, hour=14) == ["b", "c"]
    assert SEL.names_to_evaluate(names, roles, {}, 0.0, 0, due=due, hour=16) == ["a", "b", "c"]
    # without due times (the probe's wanted set) every attribute is listed
    assert SEL.names_to_evaluate(names, roles, {}, 0.0, 0) == ["a", "b", "c"]


def _count_evaluations(monkeypatch):
    calls = []
    ev0 = SEL.evaluate

    def ev(probe, t, hier, names, *a, **k):
        calls.append((float(t), len(list(names))))
        return ev0(probe, t, hier, names, *a, **k)
    monkeypatch.setattr(SEL, "evaluate", ev)
    return calls


def _org_new_fields(d, t0, rng):
    """_org with 100 more constant fields (more attributes than one run
    judges), and every 4 hours a new constant field on every event from then
    on (as the synthetic meta attributes of PG4's 340-attribute point keep
    being classified for the first time over the days)."""
    ev = _org(d, t0, rng)
    for e in ev:
        for j in range(100):
            e[3][f"wide.f{j:03d}"] = "c"
        h = int((e[0] - t0) // (4 * 3600))
        for j in range(6 * d + h + 1):
            e[3][f"late.f{j}"] = "x"
    return ev


def test_a_stable_selection_is_re_judged_less_often(monkeypatch):
    """Once its role holds, an attribute is judged every 2^DUE_MAX_K hours, not
    every hour, whatever happens to the others: with a new field appearing
    every 4 hours the attribute evaluations of day 3 are a fraction of what
    the rotation judged hourly. Round 7: every run judged its E_MAX
    attributes every hour - the key's 6-hour cadence needed STABLE_RUNS runs
    without ANY role change, and each new field's first role is one."""
    calls = _count_evaluations(monkeypatch)
    sim = Sim(p05=True)
    sim.add(daily(_org_new_fields, 3, seed=1))
    sim.run_until(MON + 3 * DAY)
    sel = sim.st.get_model("oa", SYSTEM_ENTITY, MP.ATTRSEL)
    day3 = [n for t, n in calls if t >= MON + 2 * DAY]
    # round 7: ~24 runs x E_MAX (72) attributes on day 3
    assert sum(day3) < 0.5 * 24 * SEL.E_MAX, (sum(day3), len(day3))
    assert sel["due_mode"]
    ks = [int(v[1]) for v in sel["due"].values()]
    assert max(ks) == SEL.DUE_MAX_K, sel["due"]
    # the roles are those of the hourly cadence (test_roles_targets_and_split_candidates_are_learned)
    assert sel["roles"]["noise.a"] in ("dropped", "shape")
    assert "http.route" in {a for a, _ in sel["split_cands"][0]}


def test_while_every_attribute_fits_a_run_every_attribute_is_judged(monkeypatch):
    """The OA-like stream's attributes fit one run (pack O's regime): every
    run judges all of them, as before round 8 (pack O seed 0: backing off
    there changed the mail tree's first split and lost three mail patterns)."""
    calls = _count_evaluations(monkeypatch)
    sim = Sim(p05=True)
    sim.add(daily(_org, 2, seed=1))
    sim.run_until(MON + 2 * DAY)
    sel = sim.st.get_model("oa", SYSTEM_ENTITY, MP.ATTRSEL)
    assert not sel.get("due_mode")
    n_attrs = len([a for a in sel["roles"]])
    day2 = [n for t, n in calls if t >= MON + DAY]
    assert day2 and min(day2) >= n_attrs - 4, (day2[:5], n_attrs)


def test_an_attribute_that_starts_to_carry_information_is_re_judged_hourly_again():
    """Back-off never hides a change for long: an attribute that was noise and
    becomes a function of the action is promoted within one back-off interval
    plus the promotion, and the role change starts its back-off again (k = 0)."""
    def org2(d, t0, rng):
        ev = _org_new_fields(d, t0, rng)
        if d >= 3:
            for e in ev:
                at = e[3]
                at["noise.a"] = "login" if at["http.route"] == "POST oa /login" else (
                    "health" if at["http.route"] == "GET oa /health" else "docs")
        return ev
    sim = Sim(p05=True)
    sim.add(daily(org2, 4, seed=1))
    sim.run_until(MON + 3 * DAY)
    sel = sim.st.get_model("oa", SYSTEM_ENTITY, MP.ATTRSEL)
    assert sel["due_mode"]
    assert sel["roles"]["noise.a"] in ("dropped", "shape")
    assert int(sel["due"]["noise.a"][1]) >= 1
    promoted = None
    for h in range(1, 13):
        sim.run_until(MON + 3 * DAY + h * 3600)
        sel = sim.st.get_model("oa", SYSTEM_ENTITY, MP.ATTRSEL)
        if sel["roles"]["noise.a"] in ("split", "target"):
            promoted = h
            break
    assert promoted is not None, sel["ustat"].get("noise.a")
    print("promoted after", promoted, "h")
    assert promoted <= 2 ** SEL.DUE_MAX_K + 1
    # the role change starts its back-off again: judged at the next run
    hour = int(sel["t"] // 3600.0)
    assert sel["due"]["noise.a"][0] - hour <= 1 and int(sel["due"]["noise.a"][1]) == 0, sel["due"]["noise.a"]
    assert not math.isnan(float(sel["ustat"]["noise.a"]["U_t"]))


def test_probe_rows_record_the_attributes_the_next_run_judges():
    """With more attributes than a probe row keeps (A_ROW), a row records the
    kept roles and the slice the next run judges. With per-attribute due
    times that slice is drawn from the attributes DUE then: recorded from the
    unfiltered rotation instead, an attribute was judged on rows that had not
    recorded it."""
    from app.engines.behavior import attr_select as AS

    def wide(d, t0, rng):
        ev = _org(d, t0, rng)
        for e in ev:
            for j in range(160):
                e[3][f"wide.f{j:03d}"] = "c"
        return ev
    sim = Sim(p05=True)
    sim.add(daily(wide, 2, seed=2))
    sim.run_until(MON + 2 * DAY - 1800)
    eng, st = sim.p05, sim.st
    sel = dict(st.get_model("oa", SYSTEM_ENTITY, MP.ATTRSEL))
    reg = MP.get_registry(st, "oa")
    assert len(reg.names()) > AS.A_ROW
    now = sim.now
    hour = int(now // eng.eval_period_s)
    # every other attribute judged an hour ago and not due for 8 hours
    due = dict(sel["due"])
    for j, a in enumerate(sorted(due)):
        if j % 2:
            due[a] = (hour + 8, SEL.DUE_MAX_K)
    sel["due"] = due
    st.put_model("oa", SYSTEM_ENTITY, MP.ATTRSEL, sel, ts=now)
    eng._want_cache.clear()
    want = eng._wanted(st, "oa", now, None)
    nxt = SEL.names_to_evaluate(reg.names(), sel["roles"], sel.get("dropped_at") or {}, now,
                                eng.run_index.get("oa", 0), ustat=sel.get("ustat") or {},
                                rotate=sel.get("redundant_exact") or (), due=due, hour=hour + 1)
    assert set(nxt) <= want, sorted(set(nxt) - want)[:10]


def test_node_targets_are_refitted_when_their_inputs_change(monkeypatch):
    """The per-node target overrides depend on the trees, the system targets,
    the node-local pool and the who proxies; a role change elsewhere (a new
    constant field classified invariant every 4 hours) does not re-fit them.
    Before, every run with any role change re-fitted every node's targets."""
    calls = []
    f0 = SEL.node_targets_from_probe

    def f(tree, probe, t, *a, **k):
        calls.append(float(t))
        return f0(tree, probe, t, *a, **k)
    monkeypatch.setattr(SEL, "node_targets_from_probe", f)
    sim = Sim(p05=True)
    sim.add(daily(_org_new_fields, 3, seed=1))
    sim.run_until(MON + 3 * DAY)
    sel = sim.st.get_model("oa", SYSTEM_ENTITY, MP.ATTRSEL)
    day3 = [t for t in calls if t >= MON + 2 * DAY]
    assert len(day3) <= 8, (len(day3), sel["runs"])
    assert sel["node_overrides"]                    # still fitted
