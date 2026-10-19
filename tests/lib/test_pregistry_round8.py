"""(round 8, harness owner) P02 cost with the number of attributes."""
from __future__ import annotations

import numpy as np

from app.engines.behavior.lib import pregistry as RG


def _reg(n_attr: int = 6):
    reg = RG.AttrRegistry("oa")
    r = np.random.default_rng(0)
    for k in range(n_attr):
        reg.observe(f"meta.f{k:03d}", [f"v{int(x)}" for x in r.integers(0, 5, 40)], 1000.0, 1.0)
    return reg


def test_update_stats_recomputes_only_attributes_that_changed(monkeypatch):
    """An attribute nothing was added to since the last pass keeps its entropy
    and stability (they read stored counts only): the hourly pass costs the
    attributes observed in the hour, not every registered one."""
    reg = _reg()
    calls = []
    real = RG._distributions
    monkeypatch.setattr(RG, "_distributions", lambda *a, **k: calls.append(1) or real(*a, **k))
    reg.update_stats(4600.0)
    assert len(calls) == 6
    before = {n: (r.entropy, r.stability) for n, r in reg.records.items()}
    calls.clear()
    reg.update_stats(8200.0)                          # an hour later, nothing observed
    assert calls == []
    reg.observe("meta.f002", ["v1", "v9"], 8300.0, 1.0)
    reg.set_role("meta.f004", "dropped")
    reg.update_stats(11800.0)
    assert len(calls) == 1                            # f002 (f004's top was released: empty)
    for n, r in reg.records.items():
        if n not in ("meta.f002", "meta.f004"):
            assert (r.entropy, r.stability) == before[n]


def test_skipped_statistics_equal_a_full_recomputation():
    reg = _reg()
    r = np.random.default_rng(1)
    for h in range(1, 30):
        t = 1000.0 + 3600.0 * h
        for k in r.choice(6, 2, replace=False):
            reg.observe(f"meta.f{k:03d}", [f"v{int(x)}" for x in r.integers(0, 7, 10)], t - 10.0, 1.0)
        reg.update_stats(t)                           # the skipping pass
        snap = {n: (x.entropy, x.stability) for n, x in reg.records.items()}
        for rec in reg.records.values():
            rec.stats_dirty = True
        reg.update_stats(t)                           # every record recomputed
        assert {n: (x.entropy, x.stability) for n, x in reg.records.items()} == snap


def _mixed_stream(seed: int):
    """Continuous floats (new value on almost every row), integral floats,
    epoch-second floats of a *_ts attribute, -0.0, inf / nan, ints, numeric
    strings and categorical strings, with unit and fractional evidence."""
    r = np.random.default_rng(seed)
    out = []
    t = 1.7e9
    for step in range(220):
        t += float(r.choice([60.0, 900.0, 3600.0]))
        n = int(r.integers(1, 40))
        cont = r.uniform(0.0, 1000.0, n).tolist()
        integ = [float(x) for x in r.integers(0, 40, n)]
        ts = (t - r.uniform(0, 600, n)).tolist()
        odd = [float(x) for x in r.choice([-0.0, 0.0, 1.5, np.inf, -np.inf, np.nan, 2.0], n)]
        mixed = [x if i % 3 else str(int(x)) for i, x in enumerate(r.integers(0, 9, n).tolist())]
        cat = [f"v{int(x)}" for x in r.integers(0, 6, n)]
        w = (r.random(n) * 3.0 + 0.1)
        ev = None if step % 4 else np.minimum(1.0, r.random(n) * 1.5)
        for nm, vals in (("meta.f001", cont), ("meta.f002", integ), ("ctx.req_ts", ts),
                         ("meta.f003", odd), ("meta.f004", mixed), ("meta.f005", cat)):
            out.append(("observe", (nm, vals, t, w, ev, r.random(n) < 0.05, 1, None)))
        if step % 4 == 3:
            out.append(("update_types", (t,)))
            out.append(("update_stats", (t,)))
        if step == 120:
            out.append(("set_role", ("meta.f001", "dropped")))
        if step == 160:
            out.append(("set_role", ("meta.f001", "split")))
    return out


def test_float_rows_are_bit_identical_to_the_per_row_reference():
    """(round 8, cost) Finite Python floats are classified without the value
    memo and their HLL items hashed without the hash memos; the registry must
    equal round 4's per-row observe float for float (the recorded pack O
    stream of test_pregistry_batch_equivalence has few continuous values)."""
    import importlib.util, os
    p = os.path.join(os.path.dirname(__file__), "test_pregistry_batch_equivalence.py")
    spec = importlib.util.spec_from_file_location("_pr_eq", p)
    EQ = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(EQ)
    for seed in (0, 1):
        a, b = RG.AttrRegistry("oa"), EQ.RefRegistry("oa")
        for meth, args in _mixed_stream(seed):
            getattr(a, meth)(*args)
            getattr(b, meth)(*args)
        assert list(a.records) == list(b.records)
        for nm in b.records:
            assert EQ.canon(a.records[nm]) == EQ.canon(b.records[nm]), (seed, nm)
        assert {r.type for r in a.records.values()} >= {"numeric", "time", "categorical"}


def test_unmemoised_hash_equals_psketch_h64():
    from app.engines.behavior.lib import psketch as PS
    for s in ("1.5", "-0.0", "inf", "nan", "123456.789", "测试", "\udc80x"):
        assert RG._h64_nomemo(s) == PS._h64(s)


def test_float_values_do_not_enter_the_bounded_memos():
    RG._CLS_MEMO[0].clear()
    RG._H64.clear()
    reg = RG.AttrRegistry("oa")
    r = np.random.default_rng(5)
    reg.observe("meta.f001", r.uniform(0, 1000, 500).tolist(), 1000.0, 1.0)
    assert len(RG._CLS_MEMO[0]) == 0 and len(RG._H64) == 0
    reg.observe("meta.f002", ["a", "b", 3], 1000.0, 1.0)
    assert len(RG._CLS_MEMO[0]) == 3 and len(RG._H64) == 3
