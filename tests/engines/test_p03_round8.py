"""P03 round 8 (scoring cost): what P03 derives from a fitted content record
alone - its prepared numeric grid (_NumFast) and its checks of values - is
kept across ticks while the record object lives (conformity._RecCache).

O-servers-20, 7 days (cProfile on P03): preparing the numeric records of the
visited nodes every tick was 0.51 of 4.8 s, and re-checking the same (record,
value) pairs every tick 0.35 s. The outputs are the reference outputs.
"""
from __future__ import annotations

import math

import numpy as np

import test_p03_conformity as T3

from app.engines.behavior import conformity as CF


def _rec(n_c=40.0, lo=100.0, hi=900.0):
    qg = [float(x) for x in np.linspace(lo, hi, 21)]
    return {"kind": "num", "qgrid": qg, "band98": [lo, hi], "n_c": n_c, "range": [lo, hi],
            "n_rng": n_c, "log": False}


def test_a_records_prepared_grid_is_kept_while_the_record_lives():
    rc = CF._RecCache()
    rc.rotate(T3.T0)
    r = _rec()
    a = rc.numfast(r)
    assert rc.numfast(r) is a                              # the same record: no rebuild
    rc.rotate(T3.T0 + 3600.0)                              # next hour: still used, kept
    assert rc.numfast(r) is a
    r2 = _rec()                                            # a refit: a new record object
    assert rc.numfast(r2) is not a
    r["n_c"] = 41.0                                        # changed in place: rebuilt
    b = rc.numfast(r)
    assert b is not a and b.n == 41.0
    rc.rotate(T3.T0 + 3 * 3600.0)
    rc.rotate(T3.T0 + 4 * 3600.0)                          # unused for an hour: dropped
    assert id(r) not in rc.cur and id(r) not in rc.prev
    for v in (50.0, 100.0, 450.0, 900.0, 901.0, 2000.0, "x"):
        assert CF._check("num", rc.numfast(r), v, None) == CF._check("num", CF._NumFast(r), v, None)


def test_checks_are_kept_across_ticks_for_the_same_record(monkeypatch):
    fx, _ = T3.finance_fixture(days=5)
    rec = {"kind": "text", "n": 50.0, "closed": ["jack", "rose"], "U": 0.01}
    calls = []
    real = CF._check

    def spy(typ, r, v, num):
        calls.append(v)
        return real(typ, r, v, None)
    monkeypatch.setattr(CF, "_check", spy)
    for tick in range(3):                                  # three ticks, one context each
        tc = CF._TreeCtx(fx.eng, fx.st, "finance", "finance", T3.CFG, fx.now + 900.0 * tick)
        for v in ("jack", "mike", ["unhashable"]):
            assert tc.check("text", rec, v) == real("text", rec, v, None)
    assert calls == ["jack", "mike", ["unhashable"], ["unhashable"], ["unhashable"]]
    rec2 = dict(rec)                                       # a refit record: checked again
    tc = CF._TreeCtx(fx.eng, fx.st, "finance", "finance", T3.CFG, fx.now + 3600.0)
    tc.check("text", rec2, "jack")
    assert calls[-1] == "jack" and len(calls) == 6


def _numeric_fixture():
    fx, nid = T3.finance_fixture(days=8)
    nd = fx.node("finance", T3.APPROVE)
    fx.put("finance", CF.MP.PBOUNDS, nd.id, attrs={"body.len": dict(_rec(), unit="bytes")})
    return fx


def _run(fx, cached: bool, monkeypatch):
    if not cached:
        monkeypatch.setattr(CF._RecCache, "numfast", lambda self, rec: CF._NumFast(rec))
        monkeypatch.setattr(CF._RecCache, "check", lambda self, typ, rec, v: CF._check(typ, rec, v, None))
    out = []
    t = T3.workdays(9)[-1] + 10 * 3600.0
    rng = np.random.default_rng(4)
    for k in range(12):
        ev = [T3._approve(ip, t + 60.0 * j) for j, ip in enumerate(("192.168.2.10", "192.168.1.23"))]
        ev = [(ts, ip, dict(a, **{"body.len": float(rng.choice([150.0, 500.0, 880.0, 1500.0, 30.0]))}))
              for ts, ip, a in ev]
        _, asg = fx.score("finance", ev, dt=900.0)
        out.append({c: [None if (isinstance(x, float) and math.isnan(x)) else x
                        for x in np.asarray(asg.dense(c, np.nan)).tolist()]
                    for c in ("p_who", "p_content", "p_when", "p_ev", "damp")})
        t += 900.0
    out.append(sorted((e.entity, e.extra.get("type"), e.p_value) for e in fx.violations()))
    return out


def test_scoring_with_the_record_cache_returns_the_reference_outputs(monkeypatch):
    ref = _run(_numeric_fixture(), False, monkeypatch)
    monkeypatch.undo()
    got = _run(_numeric_fixture(), True, monkeypatch)
    assert got == ref
    assert any(x is not None for row in ref[:-1] for x in row["p_content"])
