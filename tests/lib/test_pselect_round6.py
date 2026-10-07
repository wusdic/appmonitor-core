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
