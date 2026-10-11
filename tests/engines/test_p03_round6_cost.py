"""P03 round 6 (scoring cost): the rewrites that cut P03's per-event cost
return what the reference code returns.

  * the tree context (models, hierarchies) is built only for a system with a
    new non-empty batch (O-servers: ~85 contexts a tick for ~11 systems with
    events, 86 % of P03's CPU in the Regions CIDR parse of the hierarchies);
  * Regions share the parsed networks and the per-address memo;
  * the numeric range floor is _in_range_floor with the record-only parts
    computed once per record and tick;
  * content checks and P10's sequence scores are memoised per tree context.

The end-to-end equivalence (every pat.assign / pat.rate batch, P03's score
rows, pattern_violation events and P03's state) was checked on 3 days of pack
O and of O-servers-100 against the pre-round-6 code (score harness digests,
docs/lib3/progressive.md §16.14)."""
from __future__ import annotations

import math

import numpy as np
import pytest

import test_p03_conformity as T3

from app.engines.behavior import conformity as CF
from app.engines.behavior.lib import phier as PH
from app.engines.behavior.lib import pdfg as DF


def _same(a, b):
    return (isinstance(a, float) and isinstance(b, float) and math.isnan(a) and math.isnan(b)) or a == b


def test_fast_range_floor_equals_the_reference():
    rng = np.random.default_rng(11)
    recs = [{"kind": "num"}, {"kind": "num", "range": [6.0, 14.0]},
            {"kind": "num", "range": [6.0, 14.0], "n_rng": None},
            {"kind": "num", "range": [6.0, 14.0], "n_rng": 0.0},
            {"kind": "num", "range": [], "n_rng": 10.0},
            {"kind": "num", "range": ["a", "b"], "n_rng": 10.0}]
    for _ in range(300):
        lo = float(rng.choice([rng.integers(0, 1000), rng.random() * 1e3, 460.0000000000001,
                               -rng.random() * 10]))
        hi = lo + float(rng.choice([0.0, rng.integers(1, 100), rng.random() * 50]))
        recs.append({"kind": "num", "range": [lo, hi], "n_rng": float(rng.integers(1, 2000))})
    vals = [None, "x", float("nan"), float("inf"), 0.0, -1.0]
    for rec in recs:
        fast = CF._NumFast(rec)
        r = rec.get("range") or [0.0, 1.0]
        try:
            lo, hi = float(r[0]), float(r[1])
        except (TypeError, ValueError):
            lo, hi = 0.0, 1.0
        xs = vals + [lo, hi, lo - 1.0, hi + 1.0, hi + 0.5, lo - 0.01 * (hi - lo), hi + 2.0,
                     float(rng.uniform(lo - 5, hi + 5)), float(round(hi)) + 1.0, hi * (1 + 1e-12)]
        for x in xs:
            for p in (float("nan"), 1e-9, 0.004, 0.5, 1.0):
                try:
                    ref = CF._in_range_floor(rec, x, p)
                except Exception as e:                       # the fast path raises the same
                    with pytest.raises(type(e)):
                        fast.floor(x, p)
                    continue
                assert _same(fast.floor(x, p), ref), (rec, x, p)


def test_regions_share_parses_and_memo_and_answer_as_before():
    covers = [("G1", ["192.168.1.0/24", "bad", "10.0.0.0/8"]), ("G2", ["192.168.0.0/16"])]
    a, b = PH.Regions(covers), PH.Regions(covers)
    assert a._memo is b._memo and [n for n, _ in a._nets] == [n for n, _ in b._nets]
    assert PH.Regions([("G2", ["192.168.0.0/16"])])._memo is not a._memo
    for ip, want in (("192.168.1.7", "reg:G1"), ("192.168.9.1", "reg:G2"), ("10.2.3.4", "reg:G1"),
                     ("172.16.0.1", PH.REG_NONE), ("not-an-ip", PH.REG_NONE), ("::1", PH.REG_NONE)):
        assert a.of(ip) == want and b.of(ip) == want


def test_a_context_is_built_only_for_systems_with_new_batches(monkeypatch):
    fx, _ = T3.finance_fixture(days=5)
    built = []

    class Counting(CF._TreeCtx):
        def __init__(self, eng, store, s, *a, **kw):
            built.append(s)
            super().__init__(eng, store, s, *a, **kw)
    monkeypatch.setattr(CF, "_TreeCtx", Counting)
    t = T3.workdays(6)[-1] + 10 * 3600.0
    fx.score("finance", [T3._approve("192.168.2.10", t)])
    assert built == ["finance"]
    built.clear()
    fx.score("oa", [(t + 900.0, "192.168.1.21", {"http.route": T3.LOGIN})], dt=900.0)
    # finance's batch is still retained (batch_systems lists it): no context
    assert built == ["oa"]
    st = fx.st.get_model(CF.ORG, CF.ORG, CF.STATE)
    assert ("finance", "evt.batch") in st.last_batch and "finance" in st.hour


def test_memoised_checks_and_sequence_scores_are_the_reference_calls(monkeypatch):
    fx, _ = T3.finance_fixture(days=5)
    tc = CF._TreeCtx(fx.eng, fx.st, "finance", "finance", T3.CFG, fx.now)
    rec = {"kind": "text", "n": 50.0, "closed": ["jack", "rose"], "U": 0.01}
    calls = []
    real = CF._check

    def spy(typ, r, v, num):
        calls.append(v)
        return real(typ, r, v, None)
    monkeypatch.setattr(CF, "_check", spy)
    for v in ("jack", "mike", "jack", "mike", ["unhashable"], ["unhashable"]):
        assert tc.check("text", rec, v) == real("text", rec, v, None)
    assert calls == ["jack", "mike", ["unhashable"], ["unhashable"]]
    seq = []
    monkeypatch.setattr(DF, "seq_scores", lambda *a: seq.append(a) or {"p_seq": 0.5})
    tc.pflow = {"scopes": {}}
    for _ in range(3):
        assert tc.seq_scores(DF.STAR, "GET a", "GET b", 5) == {"p_seq": 0.5}
    tc.seq_scores(DF.STAR, "GET a", "GET b", 7)
    assert len(seq) == 2 and seq[0][-1] == fx.now
