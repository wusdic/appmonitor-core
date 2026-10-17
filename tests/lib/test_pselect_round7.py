"""P05 (round 7, tree owner): attribute selection."""
from __future__ import annotations

import numpy as np

from app.engines.behavior.lib import pselect as SEL


def _finance_probe(seed: int, clerk_share: float, n: int = 4000):
    """A finance stream: 30 % form posts carrying a username, almost all of them
    by the service account svc_sync; the clerks' share of the username rows is
    `clerk_share`. http.ctype is form-encoded exactly on those rows."""
    rng = np.random.default_rng(seed)
    t0 = 1_780_000_000.0
    pr = SEL.StratifiedProbe(R=4096)
    clerks = ["lucy", "tom", "kate"]
    for i in range(n):
        login = rng.random() < 0.30
        row = {"http.route": "POST /fin/login" if login else "GET /fin/page",
               "http.ctype": "application/x-www-form-urlencoded" if login else "text/html",
               "net.src": f"192.168.2.{int(rng.integers(10, 16))}"}
        row["hdr.ctype_copy"] = "c:" + row["http.ctype"]
        if login:
            row["body.kv.username"] = (clerks[int(rng.integers(0, 3))] if rng.random() < clerk_share
                                       else "svc_sync")
        pr.offer(("http", row["http.route"]), row, 1.0, t0 + i, (i * 0.61803398875) % 1.0 + 1e-9)
    t = t0 + n
    pr.rebalance(t)
    return pr, t


def _hier():
    from app.engines.behavior.lib.phier import Hierarchies
    return Hierarchies(registry={"http.route": {"type": "categorical"},
                                 "http.ctype": {"type": "categorical"},
                                 "hdr.ctype_copy": {"type": "categorical"},
                                 "body.kv.username": {"type": "text"}})


def test_a_field_dominated_by_one_value_is_not_redundant_with_what_marks_its_rows():
    """O-real seed 0 finance (round 6 open issue 3): the service account
    svc_sync carries > 99 % of the username's rows, so g3(http.ctype ->
    username) is < 0.01 over all rows AND over the rows where the username is
    present (both are shares of mass), and the username was classed
    'redundant' with http.ctype on days 1-27 although the three clerks' values
    are not a function of the content type. Redundancy is judged on the rows
    where the dependent takes a non-modal value too. Fails on round-6 code."""
    pr, t = _finance_probe(11, clerk_share=0.006)
    hier = _hier()
    rows, w, _ = pr.rows(t)
    col = pr.column(rows, "body.kv.username")
    vals = [v for v in col if v is not SEL.ABSENT and v != SEL.MISSING]
    assert vals.count("svc_sync") / len(vals) > 0.99 and len(set(vals)) >= 3   # the case is the O-real one
    kept = ["http.ctype", "hdr.ctype_copy", "body.kv.username"]
    red = SEL.redundancy(pr, t, hier, kept, cost=lambda a: 0.0, cov=lambda a: 1.0)
    assert "body.kv.username" not in red, red
    # a true function holds on every subset of rows: the copy stays redundant
    assert red.get("hdr.ctype_copy") == "http.ctype" or red.get("http.ctype") == "hdr.ctype_copy", red


def test_a_dependent_constant_off_its_mode_is_still_redundant():
    """Guard: a dependent whose non-modal values are themselves a function of
    the keeper (one username per content type) stays redundant."""
    rng = np.random.default_rng(2)
    t0 = 1_780_000_000.0
    pr = SEL.StratifiedProbe(R=4096)
    for i in range(3000):
        r = rng.random()
        ct = "form" if r < 0.30 else ("json" if r < 0.31 else "html")
        row = {"http.route": "POST /x", "http.ctype": ct,
               "body.kv.username": {"form": "svc_sync", "json": "api", "html": "web"}[ct]}
        pr.offer(("http", "POST /x"), row, 1.0, t0 + i, (i * 0.61803398875) % 1.0 + 1e-9)
    t = t0 + 3000
    pr.rebalance(t)
    red = SEL.redundancy(pr, t, _hier(), ["http.ctype", "body.kv.username"],
                         cost=lambda a: 0.0, cov=lambda a: 1.0)
    assert red.get("body.kv.username") == "http.ctype" or red.get("http.ctype") == "body.kv.username", red


def test_level_lookup_tables_equal_level_codes_on_any_subset():
    """(round 7, PG4 cost) P05's node targets compute level codes of every
    node's rows of every pool attribute; LevelLUT generalises each distinct
    value of the column once per level. Same level, codes and K as
    level_codes on the sub-column, for numeric, categorical and text columns."""
    from app.engines.behavior.lib.phier import Hierarchies
    rng = np.random.default_rng(7)
    n = 3000
    cols = {
        "net.bytes_up": [float(v) for v in rng.lognormal(7, 2, n)],
        "body.kv.opinion": [["agree", "back", "ok", "later"][int(i)] for i in rng.integers(0, 4, n)],
        "body.kv.username": ["u%03d" % int(i) for i in rng.integers(0, 200, n)],
    }
    reg = {"net.bytes_up": {"type": "numeric"}, "body.kv.opinion": {"type": "categorical"},
           "body.kv.username": {"type": "text"}}
    hier = Hierarchies(registry=reg)
    for a, col in cols.items():
        ca, ua = SEL.codes_uniq(col)
        lut = SEL.LevelLUT(hier, a, ca, ua, text_values=True)
        for size in (5, 40, 400, 2500):
            ix = np.sort(rng.choice(n, size, replace=False))
            l1, c1, k1 = lut.level_codes(ix)
            l2, c2, k2 = SEL.level_codes(hier, a, (), text_values=True, raw=SEL.sub_codes(ca, ua, ix))
            assert (l1, k1) == (l2, k2), (a, size, l1, k1, l2, k2)
            assert np.array_equal(c1, c2), (a, size)


def test_batched_joint_entropies_equal_w_plugin_bit_for_bit():
    """(round 7, PG4 cost) w_plugin_rows: many (target, grouping) joint
    entropies in two numba kernels around one np.log2 - the same floats as
    w_plugin per pair (masks, empty rows, a single cell, wide tables)."""
    from app.engines.behavior.lib import pnumba as PNB
    rng = np.random.default_rng(11)
    for n in (1, 7, 50, 333, 4096):
        R = 9
        X = rng.integers(0, 70, (R, n)).astype(np.int64)
        X[1] = 0
        Y = rng.integers(0, 64, (R, n)).astype(np.int64)
        K = rng.integers(1, 70, R).astype(np.int64)
        M = rng.random((R, n)) < 0.7
        M[2] = False
        M[3] = True
        w = rng.lognormal(0, 1.5, n)
        w[rng.random(n) < 0.1] = 0.0
        ref = [SEL.w_plugin((X[r] * int(K[r]) + Y[r])[M[r]], w[M[r]]) for r in range(R)]
        got = SEL.w_plugin_rows(X, Y, K, M, w)
        assert [(float(h).hex(), c) for h, c in got] == [(float(h).hex(), c) for h, c in ref]
        old = PNB.set_enabled(False)
        try:
            assert SEL.w_plugin_rows(X, Y, K, M, w) == ref
        finally:
            PNB.set_enabled(old)


def test_evaluate_is_the_same_with_and_without_the_kernels():
    """evaluate()'s split utilities and context gains through w_plugin_rows
    equal the per-pair numpy path (targets recorded on part of the rows
    included: the username is recorded on the login rows only)."""
    from app.engines.behavior.lib import pnumba as PNB
    pr, t = _finance_probe(3, 0.4)
    hier = _hier()
    names = ["http.route", "http.ctype", "hdr.ctype_copy", "body.kv.username", "net.src"]
    tp = ["http.ctype", "body.kv.username", "hdr.ctype_copy", "http.route"]
    a = SEL.evaluate(pr, t, hier, names, targets_prev=tp, splits_prev=["http.route"])
    old = PNB.set_enabled(False)
    try:
        b = SEL.evaluate(pr, t, hier, names, targets_prev=tp, splits_prev=["http.route"])
    finally:
        PNB.set_enabled(old)
    assert a == b
    assert any(v.get("U_s") for v in a.values())


def _dept_probe(n_dup: int = 6, n: int = 3000):
    """A stream where n_dup synthetic attributes are each a relabelling of the
    source's department (PG4's 340-attribute point: meta.f### by_dept), and one
    near-copy differs on a handful of rows. As in P05 with more attributes than
    a probe row keeps, each hour's rows record a different slice of them
    (the others are MISSING on those rows, not absent)."""
    rng = np.random.default_rng(5)
    t0 = 1_780_000_000.0
    pr = SEL.StratifiedProbe(R=4096)
    depts = ["GA", "FIN", "SALES", "DEV", "PUB"]
    reg = {"dept": {"type": "categorical"}, "near": {"type": "categorical"},
           "http.route": {"type": "categorical"}}
    dups = [f"meta.f{j:03d}" for j in range(n_dup)]
    for a in dups:
        reg[a] = {"type": "categorical"}
    for i in range(n):
        d = depts[int(rng.integers(0, 5))]
        row = {"dept": d, "http.route": f"GET /p{int(rng.integers(0, 4))}"}
        for j, a in enumerate(dups):
            row[a] = f"{d.lower()}-{j % 7}"
        row["near"] = d if i % 500 else "GA"
        hour = i // 100
        want = ["dept", "http.route", "near"] + [a for j, a in enumerate(dups) if (j + hour) % 3 != 0]
        wid = pr.want_id(want)
        pr.offer(("http", "x"), {k: v for k, v in row.items() if k in want}, 1.0, t0 + i,
                 (i * 0.61803398875) % 1.0 + 1e-9, wid)
    t = t0 + n
    pr.rebalance(t)
    from app.engines.behavior.lib.phier import Hierarchies
    return pr, t, Hierarchies(registry=reg)


def test_exact_duplicates_outside_the_pairwise_window_are_redundant():
    """(round 7, PG4) The pairwise redundancy test compares the 3 M_T strongest
    attributes only; 30 department-valued synthetic attributes (one partition)
    kept 'split' roles at PG4's 340-attribute point. Attributes outside the
    window with EXACTLY the partition of a representative on the rows that
    recorded both are redundant with it; a near-copy (differs on a few rows) is
    not judged by this rule; the window's own verdicts are unchanged. Fails on
    round-6 code (no `wider` argument)."""
    pr, t, hier = _dept_probe()
    rows, _w, _ = pr.rows(t)
    col = pr.column(rows, "meta.f001")
    assert any(v == SEL.MISSING for v in col) and any(v != SEL.MISSING for v in col)
    window = ["dept", "http.route"]
    wider = [f"meta.f{j:03d}" for j in range(6)] + ["near"]
    base = SEL.redundancy(pr, t, hier, window, cost=lambda a: 0.0, cov=lambda a: 1.0)
    xo = set()
    red = SEL.redundancy(pr, t, hier, window, cost=lambda a: 0.0, cov=lambda a: 1.0,
                         wider=wider, exact_out=xo)
    for j in range(6):
        assert red.get(f"meta.f{j:03d}") == "dept", red
    assert "near" not in red
    assert {k: v for k, v in red.items() if k in window} == base
    assert xo == {f"meta.f{j:03d}" for j in range(6)}


def test_the_exact_path_compares_attributes_of_one_kind_only():
    """(round 7) The exact path compares attributes of one hierarchy kind only:
    equivalence at the evaluation level says nothing of the other levels a split
    may use (pack O oa: hdr.user-agent.len's coarse length bins split GET /home
    by department, while the header itself splits at its shape levels). A text
    header and a categorical code of it, both outside the window, are not made
    redundant by the exact path. Fails without the kind check."""
    rng = np.random.default_rng(3)
    t0 = 1_780_000_000.0
    pr = SEL.StratifiedProbe(R=4096)
    uas = ["Mozilla/5.0 (X11; Linux x86_64) Firefox/1%02d" % k for k in range(6)] + \
          ["Mozilla/5.0 (Windows NT 10.0; Win64; x64) Chrome/12%d.0" % k for k in range(6)]
    for i in range(2000):
        k = int(rng.integers(0, len(uas)))
        row = {"http.route": f"GET /p{int(rng.integers(0, 3))}", "hdr.user-agent": uas[k],
               "client.ua_code": f"c{k}", "client.os": "linux" if k < 6 else "windows"}
        pr.offer(("http", "x"), row, 1.0, t0 + i, (i * 0.61803398875) % 1.0 + 1e-9)
    t = t0 + 2000
    pr.rebalance(t)
    from app.engines.behavior.lib.phier import Hierarchies
    hier = Hierarchies(registry={"http.route": {"type": "categorical"}, "hdr.user-agent": {"type": "text"},
                                 "client.ua_code": {"type": "categorical"}, "client.os": {"type": "categorical"}})
    assert hier.kind("hdr.user-agent") != hier.kind("client.ua_code")
    red = SEL.redundancy(pr, t, hier, ["http.route", "client.os"], cost=lambda a: 0.0,
                         cov=lambda a: 1.0, wider=["hdr.user-agent", "client.ua_code"])
    assert "client.ua_code" not in red and "hdr.user-agent" not in red, red


def test_exact_redundant_attributes_rotate_instead_of_every_run():
    """Exact duplicates are re-judged on the rotating slice: the every-run set
    does not grow with them."""
    names = ["a", "b"] + [f"m{j}" for j in range(40)]
    roles = {"a": "split", "b": "target"}
    roles.update({f"m{j}": "redundant" for j in range(40)})
    every = SEL.names_to_evaluate(names, roles, {}, 0.0, 0, a_probe=4)
    rot = SEL.names_to_evaluate(names, roles, {}, 0.0, 0, a_probe=4,
                                rotate=[f"m{j}" for j in range(40)])
    assert len(every) == 42
    assert {"a", "b"} <= set(rot) and len(rot) <= 2 + max(4, SEL.E_MIN_SLICE)


def test_numeric_copies_outside_the_window_are_redundant():
    """(round 7, PG4) Route-valued numeric copies (PG4's meta.f### by_route:
    the route's level plus small noise, recorded on different hours' rows)
    outside the pairwise window are the same quantity as the window's keeper
    (Spearman |rho| >= RED_RHO on the rows recording both)."""
    rng = np.random.default_rng(9)
    t0 = 1_780_000_000.0
    pr = SEL.StratifiedProbe(R=4096)
    base = {f"GET /p{k}": float(25 * k) for k in range(40)}       # 40 routes, levels 25 apart
    names = [f"meta.g{j}" for j in range(5)]
    for i in range(3000):
        r = f"GET /p{int(rng.integers(0, 40))}"
        row = {"http.route": r}
        for a in names:
            row[a] = round(base[r] + float(rng.normal(0.0, 2.0)), 3)
        want = ["http.route"] + [a for j, a in enumerate(names) if (j + i // 100) % 3 != 0]
        pr.offer(("http", "x"), {k: v for k, v in row.items() if k in want}, 1.0, t0 + i,
                 (i * 0.61803398875) % 1.0 + 1e-9, pr.want_id(want))
    t = t0 + 3000
    pr.rebalance(t)
    from app.engines.behavior.lib.phier import Hierarchies
    hier = Hierarchies(registry={"http.route": {"type": "categorical"},
                                 **{a: {"type": "numeric"} for a in names}})
    xo = set()
    red = SEL.redundancy(pr, t, hier, ["http.route", "meta.g0"], cost=lambda a: 0.0,
                         cov=lambda a: 1.0, wider=names[1:], exact_out=xo)
    assert all(red.get(a) in ("meta.g0", "http.route") for a in names[1:]), red
    assert xo == set(names[1:])

