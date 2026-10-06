"""Pattern tree / attribute selection perf rewrites == their reference
implementations (round 5 cost work, docs/lib3/progressive.md §16.13).

Each optimised routine is compared with the straightforward version it
replaced (kept here verbatim as the reference, or kept in the library as
`*_ref`) on random inputs: exact for discrete / structural outputs,
bit-identical floats (the rewrites perform the same IEEE operations in the
same order). The stream-level check (the same learned trees from a recorded
pack O stream) is tests/engines/test_p04_p05_stream_equivalence.py."""
from __future__ import annotations

import copy
import hashlib
import math
import random

import numpy as np

from app.engines.behavior.lib import phier as PH
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import pselect as SEL
from app.engines.behavior.lib import psketch as PS


# ------------------------------------------------------------- references
def ref_dss_add(self: PS.DecayedSpaceSaving, key, t, w=1.0, ev=0.0) -> None:
    """DecayedSpaceSaving.add before round 5 (verbatim)."""
    w = float(w)
    ev = float(ev)
    if not (0.0 <= w < math.inf and 0.0 <= ev < math.inf) or (w == 0.0 and ev == 0.0):
        return
    t = float(t)
    L = self._lm.L
    if L is None:
        self._lm.L = L = t
    elif (t - L) / self._lm._hmin > PS.RESCALE_EXP:
        self._rescale_to(t)
        L = t
    dlt = t - L
    gm = [w * 2.0 ** (dlt / h) for h in self._mh]
    ge = [ev * 2.0 ** (dlt / h) for h in self._eh]
    tm = self._tot_m
    for c in range(len(gm)):
        tm[c] += gm[c]
    te = self._tot_e
    for c in range(len(ge)):
        te[c] += ge[c]
    i = self._idx.get(key)
    if i is not None:
        row = self._m[i]
        for c in range(len(gm)):
            row[c] += gm[c]
        erow = self._e[i]
        for c in range(len(ge)):
            erow[c] += ge[c]
        return
    if len(self._keys) < self.k:
        self._idx[key] = len(self._keys)
        self._keys.append(key)
        self._m.append(gm)
        self._err.append([0.0] * len(gm))
        self._e.append(ge)
        return
    p = self.primary
    m = self._m
    i = min(range(len(m)), key=lambda j: m[j][p])
    del self._idx[self._keys[i]]
    self._keys[i] = key
    self._idx[key] = i
    old = m[i]
    self._err[i] = list(old)
    m[i] = [old[c] + gm[c] for c in range(len(gm))]
    self._e[i] = ge
    ee = self._evict_e
    for c in range(len(ge)):
        ee[c] += ge[c]


def ref_dv_add(self: PS.DecayedVector, t, w=1.0) -> None:
    """DecayedVector.add before round 5 (verbatim)."""
    t = float(t)
    old = self._lm.ensure(t)
    if old is not None:
        self.v *= PS._shrink(t, old, self.hl)
        self._lm.L = t
    n = self.v.size
    if n <= 8 and isinstance(w, (int, float)):
        d = t - self._lm.L
        v = self.v
        w = float(w)
        for i, h in enumerate(self._hs):
            v[i] += w * 2.0 ** (d / h)
        return
    if n <= 16 and isinstance(w, (list, tuple)) and len(w) == n:
        d = t - self._lm.L
        v = self.v
        for i, h in enumerate(self._hs):
            v[i] += float(w[i]) * 2.0 ** (d / h)
        return
    self.v += np.exp2((t - self._lm.L) / self.hl) * np.asarray(w, dtype=np.float64)


def ref_ring_add(self: PN.NumSummary, day: int, y: float, ev: float) -> None:
    if self.seg_day is None:
        self.seg_day = day
    i = day % PN.RING_DAYS
    r = self.ring[i]
    if not (r[0] == day):
        r[:] = [day, y, y, 0.0]
    r[1] = min(r[1], y)
    r[2] = max(r[2], y)
    r[3] += ev


def ref_h64(item) -> int:
    s = item if type(item) is str else str(item)
    return int.from_bytes(hashlib.blake2b(s.encode("utf-8", "surrogatepass"), digest_size=8).digest(), "big")


def ref_gen_codes_num(hier, a, level, codes0, uniq):
    """gen_codes' numeric fast path before round 5 (dict numbering of _num_levels)."""
    fast = SEL._num_levels(hier, a, level, uniq)
    m0 = {}
    lut0 = np.asarray([m0.setdefault(SEL._hashable(g), len(m0)) for g in fast] or [0], dtype=np.int64)
    return (lut0[codes0] if codes0.size else codes0), len(m0)


def _dss_state(s: PS.DecayedSpaceSaving):
    return (s._lm.L, list(s._keys), dict(s._idx), [list(r) for r in s._m], [list(r) for r in s._err],
            [list(r) for r in s._e], list(s._tot_m), list(s._tot_e), list(s._evict_e))


# ------------------------------------------------------------------- tests
def test_space_saving_add_equals_the_reference():
    rng = random.Random(7)
    for trial in range(40):
        k = rng.choice([1, 3, 8, 16])
        mh = rng.choice([PS.HALF_LIVES, (3600.0, 86400.0), (3600.0,)])
        eh = rng.choice([PS.EV_HALF_LIVES, (86400.0,), ()])
        pr = min(PS.CH_M, len(mh) - 1)
        a = PS.DecayedSpaceSaving(k, mh, eh, primary=pr)
        b = PS.DecayedSpaceSaving(k, mh, eh, primary=pr)
        t = 1.7e9
        other = PS.DecayedSpaceSaving(4)                # interleaved adds (shared growth-factor cache)
        for i in range(600):
            t += rng.choice([0.0, 0.0, 1.0, 37.5, 900.0, 86400.0 * rng.random(), 86400.0 * 20])
            key = rng.choice(["a", "b", "c", "d", "e", "f", 1, (2, 3)][: rng.randint(2, 8)])
            w = rng.choice([1.0, 0.5, 0.0, rng.random(), 3.0])
            ev = rng.choice([1.0, 0.0, rng.random()])
            a.add(key, t, w, ev)
            other.add("x", t + rng.choice([0.0, 5.0]), 1.0, 1.0)
            ref_dss_add(b, key, t, w, ev)
            assert _dss_state(a) == _dss_state(b), (trial, i)


def test_decayed_vector_add_equals_the_reference():
    rng = random.Random(11)
    for trial in range(40):
        hl = rng.choice([PS.HALF_LIVES, [PS.H_S] * 3 + [PS.H_M] * 3 + [PS.H_L] * 3, [PS.H_M, PS.H_L]])
        a = PS.DecayedVector(hl)
        b = PS.DecayedVector(hl)
        t = 1.7e9
        for i in range(500):
            t += rng.choice([0.0, 1.0, 600.0, 86400.0 * rng.random(), 86400.0 * 30])
            if rng.random() < 0.5:
                w = rng.choice([1.0, 0.25, rng.random()])
            else:
                w = [rng.random() for _ in range(len(hl))]
            a.add(t, w)
            ref_dv_add(b, t, w)
            assert a.v.tobytes() == b.v.tobytes() and a._lm.L == b._lm.L, (trial, i)


def test_tdigest_compress_equals_the_reference():
    rng = np.random.default_rng(5)
    for trial in range(1500):
        n = int(rng.integers(2, 400))
        mu = [rng.normal(size=n), np.round(rng.normal(size=n) * 3),
              rng.exponential(size=n) * 1e3, np.repeat(rng.normal(size=max(1, n // 10)), 10)[:n]][trial % 4]
        w = rng.exponential(size=mu.size) if trial % 3 else np.ones(mu.size)
        if trial % 7 == 0:
            w[int(rng.integers(0, mu.size))] *= 1e6
        d = float(rng.choice([2.0, 7.0, 20.0, 50.0, 100.0]))
        a, b = PS.TDigest(d), PS.TDigest(d)
        a._compress(mu.copy(), w.copy())
        b._compress_ref(mu.copy(), w.copy())
        assert a._mu.tobytes() == b._mu.tobytes() and a._w.tobytes() == b._w.tobytes(), trial


def test_source_extremes_note_many_equals_sequential_notes():
    rng = random.Random(3)
    for trial in range(60):
        a, b = PN.SourceExtremes(K=4), PN.SourceExtremes(K=4)
        day = 20000
        for i in range(300):
            day += rng.choice([0, 0, 1, 40])
            key = rng.choice(["10.0.0.1", "10.0.0.2", "10.0.0.3", "10.0.0.4", "10.0.0.5", "10.0.0.6"])
            items = [(rng.choice(["x", "y", "z"]), rng.choice([rng.random() * 100, math.inf, 5.0]))
                     for _ in range(rng.randint(0, 5))]
            ts = 1.7e9 + i * rng.choice([0.0, 60.0])
            rep = rng.choice([None, "10.0.0.9"])
            a.note_many(key, items, day, ts, rep=rep)
            for at, x in items:
                b.note(key, at, x, day, ts, rep=rep)
            assert a.d == b.d, (trial, i)


def test_ring_add_equals_the_reference():
    rng = random.Random(2)
    a, b = PN.NumSummary(), PN.NumSummary()
    for i in range(3000):
        day = 20000 + rng.randint(0, 40)
        y = rng.choice([rng.random(), -0.0, 0.0, 5.0])
        ev = rng.random()
        a._ring_add(day, y, ev)
        ref_ring_add(b, day, y, ev)
        assert a.ring.tobytes() == b.ring.tobytes() and a.seg_day == b.seg_day


def test_hll_hash_memo_is_the_hash():
    for item in ["192.168.1.21", "", "shared:10.0.0.1", 12, ("a", 1), "测试"]:
        assert PS._h64(item) == ref_h64(item)
        assert PS._h64(item) == ref_h64(item)            # from the memo


def test_numeric_level_codes_equal_the_dict_numbering():
    class H:
        def __init__(self, e, lg):
            self.e, self.lg = np.asarray(e), lg

        def kind(self, a):
            return "num"

        def _num_edges(self, a):
            return self.e, self.lg

        def gen(self, a, l, v):
            return PH.STAR

    rng = np.random.default_rng(0)
    for t in range(800):
        n = int(rng.integers(0, 300))
        vals = list(np.round(rng.exponential(size=n) * 100, 1))
        if t % 5 == 0 and n:
            vals[0] = "x"
        if t % 7 == 0 and n:
            vals[-1] = float("nan")
        if t % 11 == 0 and n:
            vals[n // 2] = True
        codes0, uniq = SEL.codes_uniq(vals)
        h = H(np.sort(rng.exponential(size=7) * 100), bool(t % 2))
        for lv in (1, 2, 3):
            got = SEL.gen_codes(h, "z", lv, codes0, uniq)
            want = ref_gen_codes_num(h, "z", lv, codes0, uniq)
            assert got[1] == want[1] and got[0].dtype == want[0].dtype and np.array_equal(got[0], want[0])


def test_batched_penalised_gains_equal_one_by_one():
    rng = np.random.default_rng(0)
    for trial in range(1500):
        n = int(rng.integers(0, 600))
        w = rng.exponential(size=n) * rng.choice([1.0, 1e-3, 1e3])
        if trial % 11 == 0:
            w[:] = 0.0
        b = rng.integers(0, int(rng.integers(1, 30)), size=n)
        gs = []
        for _ in range(int(rng.integers(1, 8))):
            kg = int(rng.integers(1, 70))
            gs.append((rng.integers(0, kg, size=n), kg))
        hb = (float(rng.random() * 3), int(np.unique(b).size))
        hgs = [SEL.w_plugin(g, w) for g, _ in gs]
        want = [SEL.penalised_gain(b, 5, g, kg, w, n, hb[0], hg) for (g, kg), hg in zip(gs, hgs)]
        assert SEL.penalised_gains_b(b, hb, gs, hgs, w, n) == want
        bs = [(rng.integers(0, int(rng.integers(1, 20)), size=n), (float(rng.random()), int(rng.integers(1, 9))))
              for _ in range(int(rng.integers(1, 9)))]
        g, kg = gs[0]
        want = [SEL.penalised_gain(bb, 5, g, kg, w, n, hbb, hgs[0]) for bb, hbb in bs]
        assert SEL.penalised_gains_g(bs, g, kg, hgs[0], w, n) == want
        assert SEL.w_plugin_many([g for g, _ in gs], w) == [SEL.w_plugin(g, w) for g, _ in gs]


def test_same_source_memo_is_the_function():
    names = ["http.route", "http.path", "body.len", "body.kv.user", "body.kv.user.len", "body.keys",
             "net.src", "net.bytes_up", "resp.len", "net.bytes_down", "ctx.tod_min", "ctx.when", "tls.ja3",
             "client.stack", "hdr.user-agent", "body.kv.viewstate.len", "x", "x.len"]
    for a in names:
        for b in names:
            assert SEL.same_source(a, b) == SEL._same_source(a, b)
            assert SEL.same_source(a, b) == SEL._same_source(a, b)


def test_numeric_bins_follow_a_registry_refresh():
    """phier.num_bin reads the edges list once per edges object: a refreshed
    record (P02 replaces the array) is re-read at once."""
    rec = {"type": "numeric", "hier": {"edges": np.asarray([10.0, 20.0, 30.0])}}
    h = PH.Hierarchies(registry={"x": rec})
    assert h.num_bin("x", 25.0) == 2
    rec["hier"]["edges"] = np.asarray([24.0, 26.0])
    assert h.num_bin("x", 25.0) == 1
    rec["hier"]["log"] = True
    assert h.num_bin("x", 25.0) == 0                      # log(25) < 24
