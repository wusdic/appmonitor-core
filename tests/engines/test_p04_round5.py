"""P04 / P05 round 5 (docs/lib3/progressive.md §16.13): P05 does not
re-evaluate an unchanged key (opt-in); the reference statement is identified
by a serial number."""
from __future__ import annotations

from ptree_sim import DAY, MON, Sim, daily, is_workday

GA = ["192.168.1.21", "192.168.1.23", "192.168.1.25"]
POP = [f"192.168.3.{i}" for i in range(20, 32)]
LOGIN = {"http.route": "POST oa /login", "http.method": "POST"}


def _login(ts, ip, size):
    return (ts, "oa", ip, dict(LOGIN, **{"net.bytes_up": float(size)}))


def test_p05_does_not_re_evaluate_an_unchanged_key():
    """(opt-in, progressive.defaults.p05_skip_unchanged) An hour without new
    rows (and no registry, tree or hierarchy change) changes nothing P05
    measures: the evaluation is skipped and does not count as a run. Before
    round 5 every hourly run re-measured the same probe (PG4 servers pack: 774
    of 1 394 evaluations a day without a new row)."""
    cfg = {"progressive": {"enabled": True, "defaults": {"p05_skip_unchanged": True}}, "tz": "Asia/Shanghai"}
    sim = Sim(p05=True, config=cfg)

    def day(d, t0, rng):
        if d >= 2:
            return []
        return [_login(t0 + (9 * 60 + rng.uniform(0, 120)) * 60, ip, rng.uniform(600, 2000))
                for ip in GA + POP for _ in range(2)]
    sim.add(daily(day, 4, seed=5))
    sim.run_until(MON + 2 * DAY + 6 * 3600)
    p05 = sim.p05
    ev, sk = 0, 0
    for _ in range(int(2 * DAY // sim.dt)):                     # two days without any row
        sim.tick()
        ev += int(p05.last_stats.get("evaluations", 0))
        sk += int(p05.last_stats.get("skipped_unchanged", 0))
    assert ev == 0 and sk > 0, (ev, sk)
    sim.add([_login(sim.now + 600.0, GA[0], 1500.0)])
    for _ in range(8):
        sim.tick()
        ev += int(p05.last_stats.get("evaluations", 0))
    assert ev >= 1                                              # a new row: evaluated again


def test_reference_statement_identity_is_a_serial_not_an_address():
    """The held-out record compares a node's constraints with those of the
    reference it last saw, keyed by the reference's identity. id() of the
    reference dict was that key: an address the next reference can reuse
    (the comparison then skipped a material restatement) and a value that
    differs between processes (two identical runs differed in state)."""
    sel = {"targets_sys": {0: ["net.bytes_up"]}, "split_cands": {0: [], 1: []}, "roles": {}}
    sim = Sim(sel=sel)

    def day(d, t0, rng):
        if not is_workday(t0):
            return []
        return [_login(t0 + (9 * 60 + rng.uniform(0, 21)) * 60, ip, rng.uniform(1024, 2048)) for ip in GA]
    sim.add(daily(day, 12, seed=3))
    sim.run_until(MON + 12 * DAY)
    tr = sim.tree()
    refs = [(nd, nd.meta.get("hold_fp")) for nd in tr.nodes.values() if nd.meta.get("hold_fp")]
    assert refs, "no node was checked against a reference"
    for nd, fp in refs:
        sn, t = fp["_ref"]
        assert sn == nd.ref["sn"] and t == nd.ref["t"]
        assert sn != id(nd.ref)

