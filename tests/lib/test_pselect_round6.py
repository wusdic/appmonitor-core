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


def test_a_text_field_is_compared_on_its_values_not_its_shape():
    """O-real seed 0 finance (round 6, evaluator): the login username is a TEXT
    attribute; level_codes compares text at its shape level, where every
    username is the one shape [a-z]{4}, constant on the rows where it is
    present - so ctype -> username was functional there as well and the
    username stayed 'redundant' with http.ctype on days 1-27 (no node target,
    no source -> username binding screened by P08). A text attribute with few
    values is compared on its values; a text copy of another attribute is
    still redundant. Fails with shape-level text codes."""
    from app.engines.behavior.lib.phier import Hierarchies
    rng = np.random.default_rng(5)
    t0 = 1_780_000_000.0
    pr = SEL.StratifiedProbe(R=4096)
    users = ["lucy", "lucy", "lucy", "tomz", "kate"]          # one shape, three values
    for i in range(4000):
        login = rng.random() < 0.05
        row = {"http.route": "POST /fin/login" if login else "GET /fin/page",
               "http.ctype": "application/x-www-form-urlencoded" if login else "text/html",
               "net.src": f"192.168.2.{int(rng.integers(10, 16))}"}
        row["hdr.ctype_text"] = "t-" + row["http.ctype"][:4]
        if login:
            row["body.kv.username"] = users[int(rng.integers(0, len(users)))]
        pr.offer(("http", row["http.route"]), row, 1.0, t0 + i, (i * 0.61803398875) % 1.0 + 1e-9)
    t = t0 + 4000
    pr.rebalance(t)
    reg = {"http.route": {"type": "categorical"}, "http.ctype": {"type": "categorical"},
           "hdr.ctype_text": {"type": "text"}, "body.kv.username": {"type": "text"}}
    hier = Hierarchies(registry=reg)
    assert hier.kind("body.kv.username") == "text"
    kept = ["http.ctype", "hdr.ctype_text", "body.kv.username"]
    red = SEL.redundancy(pr, t, hier, kept, cost=lambda a: 0.0, cov=lambda a: 1.0)
    assert "body.kv.username" not in red, red
    assert red.get("hdr.ctype_text") == "http.ctype" or red.get("http.ctype") == "hdr.ctype_text", red


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


def _rotating_offers(n_want, n_names, per_want, rows_per_want, rng):
    """Rows offered under `n_want` want sets that rotate like P05's hourly
    wanted slice (names_to_evaluate): a fixed core plus a rotating slice."""
    names = [f"f{j:03d}" for j in range(n_names)]
    core = names[:per_want // 2]
    out = []
    t = 1000.0
    for h in range(n_want):
        start = (per_want // 2 + h * 7) % (n_names - per_want)
        want = set(core) | set(names[per_want // 2 + start: per_want + start])
        for _ in range(rows_per_want):
            # an action's rows carry the same fields (one schema per want set)
            row = {a: int(rng.integers(0, 4)) for a in want if int(a[1:]) % 5 != h % 5}
            out.append((want, ("http", int(rng.integers(0, 4))), row, float(rng.uniform(0.5, 2.0)), t,
                        float(rng.random())))
            t += 5.0
    return out


def test_probe_interning_stays_compact_when_the_wanted_slice_rotates():
    """EV6-7 (evaluator, round 6): since round 6 the wanted slice of P05's probe
    rows rotates every hour, so a probe holds rows of hundreds of want sets and
    schemas. Interned as frozensets and per-schema name -> position dicts they
    were ~4 MB of the 340-attribute PG4 point and failed PG4's memory slope
    against attributes (0.180 -> 0.214); as bit masks over the probe's name
    index and key tuples (index dicts built on demand, dropped by release())
    they are a fraction of it."""
    from app.eval.pscale import deep_sizeof
    rng = np.random.default_rng(3)
    pr = SEL.StratifiedProbe(R=8192)
    for want, st, row, m, t, u in _rotating_offers(300, 400, 80, 3, rng):
        pr.offer(st, row, m, t, u, pr.want_id(want))
    rows = pr.rows(t)[0]
    pr.column(rows, "f001")
    pr.release()
    assert len(pr._wants) == 300
    meta = deep_sizeof([pr._wants, pr._want_ix, pr._schema_list, pr._schemas, pr._sid_want,
                        getattr(pr, "_nix", None), getattr(pr, "_ixc", None)], set())
    assert meta < 600_000, meta            # round-6 interning: 2.0 MB; EV6-7: 0.27 MB


def test_probe_columns_equal_a_plain_reference_under_rotating_want_sets():
    """Representation-free check of EV6-7: every column equals the value, ABSENT
    (wanted, the event lacked it) or MISSING (not wanted when recorded)."""
    rng = np.random.default_rng(11)
    pr = SEL.StratifiedProbe(R=100000)
    ref = []
    offers = _rotating_offers(40, 120, 30, 5, rng)
    for k, (want, st, row, m, t, u) in enumerate(offers):
        w = None if k % 17 == 0 else want
        pr.offer(st, row, m, t, u, pr.want_id(w))
        ref.append((row, w))
    rows, _, _ = pr.rows(t)
    assert len(rows) == len(ref)
    # rows() lists the reservoirs stratum by stratum; match rows by content
    got = {}
    for name in [f"f{j:03d}" for j in range(120)] + ["nope"]:
        col = pr.column(rows, name)
        for i, v in enumerate(col):
            got.setdefault(i, {})[name] = v
    want_rows = sorted(repr(sorted(
        {n: (r[n] if n in r else (SEL.ABSENT if (w is None or n in w) else SEL.MISSING))
         for n in [f"f{j:03d}" for j in range(120)] + ["nope"]}.items(), key=lambda kv: kv[0]))
        for r, w in ref)
    have_rows = sorted(repr(sorted(d.items(), key=lambda kv: kv[0])) for d in got.values())
    assert have_rows == want_rows


def _gen(h, a, level, vals):
    u = list(vals)
    codes, k = SEL.gen_codes(h, a, level, np.arange(len(u)), u)
    return codes.tolist(), k


def test_generalisation_memo_follows_the_windows_edges_not_their_number():
    """EV6-8 (evaluator, round 6): gen_codes' module-level memo was keyed on the
    SIZES of the learned models (len(windows), len(ip2g), n_groups,
    len(regions)). P09 moving a window's edges kept the key, so P05 coded
    ctx.tod_min on the old windows - and a run's results depended on what the
    process had memoised before (a resumed run differed from a straight one)."""
    from app.engines.behavior.lib import phier as PH
    reg = {"ctx.tod_min": {"type": "time"}}
    a = PH.Hierarchies(reg, windows=[(0, 600, "w:a"), (600, 1440, "w:b")])
    b = PH.Hierarchies(reg, windows=[(0, 300, "w:a"), (300, 1440, "w:b")])
    vals = [100.0, 400.0, 700.0]
    assert _gen(a, "ctx.tod_min", 2, vals) == ([0, 0, 1], 2)
    assert _gen(b, "ctx.tod_min", 2, vals) == ([0, 1, 1], 2)


def test_generalisation_memo_follows_group_membership_and_day_type_windows():
    from app.engines.behavior.lib import phier as PH
    reg = {"net.src": {"type": "ip"}, "ctx.when": {"type": "time"}}
    ips = ["10.0.0.1", "10.0.0.2", "10.0.0.3"]
    a = PH.Hierarchies(reg, ip2g={"10.0.0.1": "g1", "10.0.0.2": "g1", "10.0.0.3": "g2"})
    b = PH.Hierarchies(reg, ip2g={"10.0.0.1": "g1", "10.0.0.2": "g2", "10.0.0.3": "g2"})
    assert _gen(a, "net.src", 3, ips) == ([0, 0, 1], 2)
    assert _gen(b, "net.src", 3, ips) == ([0, 1, 1], 2)     # same sizes, one address moved
    when = [("workday", 100.0), ("workday", 400.0), ("workday", 700.0)]
    c = PH.Hierarchies(reg, windows_dt={"wd": [(0, 600, "w:a"), (600, 1440, "w:b")]})
    d = PH.Hierarchies(reg, windows_dt={"wd": [(0, 300, "w:a"), (300, 1440, "w:b")]})
    assert _gen(c, "ctx.when", 2, when)[0] == [0, 0, 1]
    assert _gen(d, "ctx.when", 2, when)[0] == [0, 1, 1]


def test_generalisation_memo_does_not_share_a_record_version_across_registries():
    """EV6-8: two systems (or two runs in one process) with a categorical
    attribute of the same name and record version but different value groups
    must not share generalisations (the memo key holds the record itself)."""
    from app.engines.behavior.lib import phier as PH
    vals = ["a", "b", "c"]
    r1 = {"x": {"type": "categorical", "hier": {"groups": {"a": "g1", "b": "g1", "c": "g2"}}}}
    r2 = {"x": {"type": "categorical", "hier": {"groups": {"a": "g1", "b": "g2", "c": "g2"}}}}
    h1, h2 = PH.Hierarchies(r1), PH.Hierarchies(r2)
    assert _gen(h1, "x", 1, vals)[0] == [0, 0, 1]
    assert _gen(h2, "x", 1, vals)[0] == [0, 1, 1]

    class Rec:                                   # registry records carry a version number
        def __init__(self, groups):
            self.type, self.version, self.hier = "categorical", 1, {"groups": groups}
    h3 = PH.Hierarchies({"y": Rec({"a": "g1", "b": "g1", "c": "g2"})})
    h4 = PH.Hierarchies({"y": Rec({"a": "g1", "b": "g2", "c": "g2"})})
    assert _gen(h3, "y", 1, vals)[0] == [0, 0, 1]
    assert _gen(h4, "y", 1, vals)[0] == [0, 1, 1]
