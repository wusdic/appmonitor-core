"""pselect round 6: the probe builds a row only when its reservoir takes it."""
from __future__ import annotations

import numpy as np

from app.engines.behavior.lib import pselect as SEL


def _rows(n, n_attr, rng):
    out = []
    for i in range(n):
        row = {f"a{j}": int(rng.integers(0, 5)) for j in range(n_attr) if rng.random() < 0.8}
        row["net.src"] = f"10.0.0.{int(rng.integers(1, 40))}"
        out.append((("http", int(rng.integers(0, 6))), row, float(rng.uniform(0.5, 2.0)),
                    1000.0 + 3.0 * i, float(rng.random())))
    return out


def test_lazy_offer_keeps_the_same_reservoir_and_builds_only_taken_rows():
    rng = np.random.default_rng(7)
    rows = _rows(6000, 30, rng)
    a, b = SEL.StratifiedProbe(R=256), SEL.StratifiedProbe(R=256)
    built = [0]

    def maker(r):
        def f():
            built[0] += 1
            return r
        return f
    for st, row, m, t, u in rows:
        a.offer(st, row, m, t, u)
        b.offer_lazy(st, maker(row), m, t, u)
    ra, wa, sa = a.rows(rows[-1][3])
    rb, wb, sb = b.rows(rows[-1][3])
    assert sa == sb and np.array_equal(wa, wb)
    for name in ["a0", "a7", "a29", "net.src"]:
        assert a.column(ra, name) == b.column(rb, name)
    assert a.n_offered == b.n_offered == len(rows)
    # most offered rows are refused once the reservoir is full
    assert built[0] < 0.5 * len(rows), built[0]


def test_one_run_judges_a_bounded_number_of_attributes():
    """P05's per-run work must not grow with the number of attributes (PG4
    CPU-per-event slope against attributes <= 0.2): split roles beyond the best
    SPLIT_EVERY by U_s, constants (invariant) and shapes rotate with the
    never-evaluated and dropped attributes; a run judges at most E_MAX."""
    def mk(n_split, n_inv, n_shape, n_drop, n_target=3):
        names, roles, ustat = [], {}, {}
        for kind, n in (("split", n_split), ("invariant", n_inv), ("shape", n_shape), ("dropped", n_drop),
                        ("target", n_target)):
            for i in range(n):
                a = f"{kind}{i:03d}"
                names.append(a)
                roles[a] = kind
                ustat[a] = {"U_s": 1.0 / (i + 1), "U_t": 0.1}
        return names, roles, ustat
    small = mk(20, 10, 2, 10)
    big = mk(76, 40, 30, 150)
    for run in range(5):
        todo_s = SEL.names_to_evaluate(small[0], small[1], {}, 0.0, run, ustat=small[2])
        assert set(todo_s) == set(small[0])               # a pack-O-sized registry: everything, every run
        todo_b = SEL.names_to_evaluate(big[0], big[1], {}, 0.0, run, ustat=big[2])
        assert len(todo_b) <= SEL.E_MAX, len(todo_b)
        best = sorted([a for a in big[0] if big[1][a] == "split"], key=lambda a: -big[2][a]["U_s"])
        assert set(best[:SEL.SPLIT_EVERY]) <= set(todo_b)    # the best split candidates every run
    # every attribute is judged within a bounded number of runs
    seen = set()
    for run in range(40):
        seen |= set(SEL.names_to_evaluate(big[0], big[1], {}, 0.0, run, ustat=big[2]))
    assert seen == set(big[0])


def test_a_rare_field_is_not_redundant_with_what_marks_its_presence():
    """O-real seed 0 finance (round 6): a login form's username is present on
    ~2 % of the events, exactly those whose content type is form-encoded. g3
    over all rows is a share of the probe's mass, so ctype -> username looked
    functional (g3 ~0.004 < RED_G3) and the username was classed 'redundant'
    with http.ctype: no node target, no source -> username binding stated.
    Redundancy is judged on the rows where the dependent attribute is present
    too; a true copy (present everywhere) is still redundant."""
    from app.engines.behavior.lib.phier import Hierarchies
    rng = np.random.default_rng(3)
    t0 = 1_780_000_000.0
    pr = SEL.StratifiedProbe(R=4096)
    users = ["lucy"] * 8 + ["tom", "kate"]          # one clerk logs in most often
    for i in range(4000):
        login = rng.random() < 0.02
        row = {"http.route": "POST /fin/login" if login else "GET /fin/page",
               "http.ctype": "application/x-www-form-urlencoded" if login else "text/html",
               "net.src": f"192.168.2.{int(rng.integers(10, 16))}"}
        row["http.ctype_copy"] = "c:" + row["http.ctype"]
        if login:
            row["body.kv.username"] = users[int(rng.integers(0, len(users)))]
        pr.offer(("http", row["http.route"]), row, 1.0, t0 + i, (i * 0.61803398875) % 1.0 + 1e-9)
    t = t0 + 4000
    pr.rebalance(t)
    hier = Hierarchies(registry={a: {"type": "categorical"}
                                 for a in ("http.route", "http.ctype", "http.ctype_copy", "body.kv.username")})
    kept = ["http.ctype", "http.ctype_copy", "body.kv.username"]
    red = SEL.redundancy(pr, t, hier, kept, cost=lambda a: 0.0, cov=lambda a: 1.0)
    assert "body.kv.username" not in red, red
    # the exact copy is still recognised
    assert red.get("http.ctype_copy") == "http.ctype" or red.get("http.ctype") == "http.ctype_copy", red


def test_per_run_pools_are_bounded_and_untouched_below_the_cap():
    """PG4 (round 6): who_proxies walks every kept attribute and the node
    target pass every node-local one, per run - 86-89 split roles per system
    on the 340-attribute point against <= 40 kept on pack O / O-real. Both
    lists are capped; below the cap they are the full lists in their order."""
    small = [f"a{i:02d}" for i in range(20)]
    ust = {a: {"U_t": 0.01 * i} for i, a in enumerate(small)}
    assert SEL.bounded_by_u(small, ust, SEL.LOCAL_POOL_MAX) == small
    big = [f"b{i:03d}" for i in range(120)]
    ustb = {a: {"U_t": (i * 37 % 120) / 100.0} for i, a in enumerate(big)}
    got = SEL.bounded_by_u(big, ustb, SEL.LOCAL_POOL_MAX)
    assert len(got) == SEL.LOCAL_POOL_MAX
    best = sorted(big, key=lambda a: -ustb[a]["U_t"])[:SEL.LOCAL_POOL_MAX]
    assert set(got) == set(best) and got == [a for a in big if a in set(best)]
    roles = {a: "split" for a in small}
    out = {"roles": roles, "ustat": ust, "targets_sys": {0: small[:5]}, "split_cands": {0: []}}
    assert SEL.proxy_candidates(out) == small
    roles = {a: ("split" if i % 3 else "shape") for i, a in enumerate(big)}
    out = {"roles": roles, "ustat": ustb, "targets_sys": {0: big[100:110]},
           "split_cands": {0: [(a, 0) for a in big[50:60]]}}
    pc = SEL.proxy_candidates(out, prev_proxies=["b007"])
    assert len(pc) == SEL.PROXY_EVAL_MAX
    assert set(big[100:110]) <= set(pc) and set(big[50:60]) <= set(pc) and "b007" in pc


def test_daily_value_groups_are_bounded_and_rotate():
    """PG4 (round 6): value groups are re-fitted for <= VG_MAX attributes a
    day - P04's targets and split candidates every day, the others by age -
    and for all of them when there are no more (pack O / O-real)."""
    small = [f"c{i:02d}" for i in range(10)]
    assert SEL.value_group_slice(small, {}, {}) == small
    big = [f"c{i:03d}" for i in range(100)]
    out = {"targets_sys": {0: big[90:95]}, "split_cands": {0: [(a, 1) for a in big[80:85]]}}
    last: dict = {}
    seen = set()
    for day in range(10):
        sl = SEL.value_group_slice(big, out, last)
        assert len(sl) == SEL.VG_MAX
        assert set(big[90:95]) | set(big[80:85]) <= set(sl)
        for a in sl:
            last[a] = float(day)
        seen |= set(sl)
    assert seen == set(big)
