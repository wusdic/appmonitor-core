"""lib/pnode round 7 (tree owner): the held-out record of a statement."""
from __future__ import annotations

from app.engines.behavior.lib import pnode as PN

DAY = 86400.0
T0 = 1_780_000_000.0 - (1_780_000_000.0 % DAY)


def _test(hr, t0, hits):
    """One held-out test: 60 one-event units, each checking every key of
    `hits` ({key: hit(i)}), spread over a day (so 'when' is judged on it)."""
    for i in range(60):
        t = t0 + i * 1400.0
        cl = ("src", i, int(t // DAY))
        for k, f in hits.items():
            hr.add(k, bool(f(i)), 0.9, t, cluster=cl)


def test_a_restated_constraint_takes_its_failed_tests_with_it():
    """(round 7) M45 restarts the counts of a constraint the node re-states
    materially (a young node's narrow window widened, a band moved), but the
    TESTS it failed stayed in the record for ~14 tests: the statement - now
    with the right window - kept stating a confidence pulled down by a window
    it no longer states (pack O days >= 7: stated 0.88, held 0.94). A test
    that failed only on the dropped constraint is a pass of what is still
    stated; fails on round-6 code (p_hold stays at 3/9 passes)."""
    hr = PN.HoldRecord()
    for d in range(6):                    # the first band: too narrow, it fails every test
        _test(hr, T0 + d * DAY, {"who": lambda i: True, "body.len": lambda i: i % 2 == 0,
                                 "body.kv.x": lambda i: True})
    for d in range(6, 9):                 # three passes
        _test(hr, T0 + d * DAY, {"who": lambda i: True, "body.len": lambda i: True,
                                 "body.kv.x": lambda i: True})
    ps, n = hr.tests()
    assert ps < 0.5 * n
    hr.drop(["body.len"])                 # P04 _hold_check: the band re-stated (M45)
    ps2, n2 = hr.tests()
    assert abs(n2 - n) < 1e-9
    assert ps2 > 0.95 * n2, (ps2, n2)     # every failure was the dropped band's
    assert "body.len" not in hr.failures()


def test_a_test_failed_by_a_kept_constraint_stays_failed():
    hr = PN.HoldRecord()
    for d in range(4):                    # body.kv.x AND body.len fail
        _test(hr, T0 + d * DAY, {"body.kv.x": lambda i: i % 2 == 0, "body.len": lambda i: i % 2 == 0})
    for d in range(4, 6):                 # body.len alone fails
        _test(hr, T0 + d * DAY, {"body.kv.x": lambda i: True, "body.len": lambda i: i % 3 == 0})
    _test(hr, T0 + 6 * DAY, {"body.kv.x": lambda i: True, "body.len": lambda i: True})
    ps, n = hr.tests()
    hr.drop(["body.kv.x"])
    ps2, n2 = hr.tests()
    assert abs(ps2 - ps) < 1e-9, (ps, ps2)    # body.len failed every failed test: none is forgiven
    assert hr.failures().get("body.len", 0.0) > 4.0


def test_a_failure_older_than_the_record_is_forgiven_at_its_forgotten_weight():
    hr = PN.HoldRecord()
    for i in range(50):                   # one failed test of 50 units
        hr.add("body.len", False, 0.9, T0 + i, cluster=("s", i))
    for j in range(1, 15):                # 14 passed tests after it
        for i in range(50):
            hr.add("body.len", True, 0.9, T0 + j * DAY + i, cluster=("s", j, i))
    hr.add("body.len", True, 0.9, T0 + 20 * DAY, cluster=("s", "close"))   # closes the last test
    ps, n = hr.tests()
    hr.drop(["body.len"])
    ps2, n2 = hr.tests()
    # the failed test is 14 tests old: half weight (HOLD_TEST_HALF_N = 14)
    assert abs((ps2 - ps) - 0.5) < 1e-6, (ps, ps2)
    assert ps2 <= n2 + 1e-9


def test_the_record_keeps_a_test_record_without_the_windows():
    """'~when' facet: the tests judged on every constraint but the windows
    (P14 composes a group's part with its own windows from it)."""
    hr = PN.HoldRecord()
    for d in range(5):
        _test(hr, T0 + d * DAY, {"who": lambda i: True, "when": lambda i: i % 2 == 0})
    ft = hr.facet_tests()
    assert ft["~when"][0] == ft["~when"][1] > 4.0
    assert ft["when"][0] < 0.5


def test_clustered_batch_checks_are_compact_and_give_the_same_tolerance():
    """(round 7, PG4 memory) a batch's per-(constraint, cluster) checks are
    flat arrays (pnode._KC), not a dict of two-float lists per cluster: the
    same cluster-robust tolerance (cluster_tol sums in the constraint's own
    cluster order), a fraction of the memory."""
    import sys
    import numpy as np
    rng = np.random.default_rng(3)
    hr = PN.HoldRecord()
    ref: dict = {}
    for i in range(400):
        c = ("10.0.0.%d" % int(rng.integers(0, 40)), 1)     # < HOLD_BATCH_U units: one batch
        for k in ("body.len", "net.dur_ms", "who"):
            if k == "net.dur_ms" and i % 3 == 0:
                continue                           # a key absent on some events
            hit = bool(rng.random() < 0.9)
            hr.add(k, hit, 0.9, T0 + i, cluster=c)
            v = ref.setdefault(k, {}).setdefault(c, [0.0, 0.0])
            v[0] += 1.0
            v[1] += 1.0 if hit else 0.0
    for k, d in ref.items():
        kc = hr.bc[k]
        assert [tuple(x) for x in kc.values()] == [tuple(v) for v in d.values()]
        n = sum(v[0] for v in d.values())
        assert PN.cluster_tol(0.9, n, kc) == PN.cluster_tol(0.9, n, d)
    def size(o, seen=None):
        seen = set() if seen is None else seen
        if id(o) in seen:
            return 0
        seen.add(id(o))
        n = sys.getsizeof(o)
        if isinstance(o, dict):
            n += sum(size(v, seen) for v in o.values())   # cluster keys are shared with the batch's units
        elif isinstance(o, list):
            n += sum(size(v, seen) for v in o)
        elif isinstance(o, PN._KC):
            n += size(o.n, seen) + size(o.h, seen) + size(o.pos, seen)
        return n
    assert size(hr.bc) < 0.4 * size(ref), (size(hr.bc), size(ref))
