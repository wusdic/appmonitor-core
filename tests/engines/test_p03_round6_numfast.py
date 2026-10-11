"""conformity._NumFast round 6 (cost): the pure-Python grid, the inlined
in-band p and the prepared predictive GPD tails return the pre-round-6
_NumFast's bits (its code is kept here verbatim as the reference, with
_in_range_floor) on fitted records of several shapes and on values inside
the band, in both tails, at the extremes and outside the domain."""
from __future__ import annotations

import bisect
import math
from typing import Any, List, Mapping, Optional, Tuple

import numpy as np
import pytest

from app.engines.behavior import conformity as CF
from app.engines.behavior.lib import pbounds as PB
from app.engines.behavior.lib import pnode as PN
from app.engines.behavior.lib import pscore as SC
from app.engines.behavior.conformity import NAN, RANGE_MARGIN, _nan, _tail_k

T0 = 1.7e9


# ---------------------------------------------------------------- reference (pre-round-6 code)
def _ref_in_range_floor(rec: Mapping[str, Any], v: Any, p: float) -> float:
    """A value inside the record's observed clean range is at least as common
    as the observed extreme it does not pass: p >= (1 + 1) / (n_rng + 1). The
    tail models can say otherwise - a bounded GPD (xi < 0) fitted to integer
    counts ends at its estimated endpoint, and pack O's 192.168.1.21 sending
    net.pkts_down = 6 (the node's observed minimum, range 6-14) got p = 1e-9,
    was damped as an outlier and lost its trust for the day (B28)."""
    if _nan(p):
        return p
    rng = rec.get("range") if isinstance(rec, Mapping) else None
    n_rng = rec.get("n_rng") if isinstance(rec, Mapping) else None
    if not rng or n_rng is None or not (float(n_rng) >= 1.0):
        return p
    try:
        x = float(v)
        lo, hi = float(rng[0]), float(rng[1])
    except (TypeError, ValueError):
        return p
    # the stated range is the inverse transform of the observed extremes
    # (exp(log 460) = 460.0000000000001): compare with a relative tolerance, and
    # a value within RANGE_MARGIN of the range width beyond an extreme is a new
    # extreme of the same sample, whose exchangeable rank p is 1 / (n_rng + 1)
    # (pack O portal: net.dur_ms 336.98 against an observed maximum of ~336.5 got
    # p = 1e-9 from a bounded GPD; bytes_up 460 'below' 460.0000000000001)
    tol = 1e-9 * max(1.0, abs(lo), abs(hi))
    if lo - tol <= x <= hi + tol:
        return max(float(p), SC.conformal_rank_p(1.0, float(n_rng)))
    m = RANGE_MARGIN * max(hi - lo, tol)
    if all(abs(z - round(z)) <= 1e-6 * max(1.0, abs(z)) for z in (x, lo, hi)):
        m = max(m, 1.0)                  # integer data: one resolution step (bytes 516 vs max 515)
    if lo - m <= x <= hi + m:
        return max(float(p), SC.conformal_rank_p(0.0, float(n_rng)))
    return p



class _RefNumFast:
    """lib/pbounds.p_value(rec, v) without a live digest, with the record's
    quantile grid prepared once per record and batch (the per-event cost of
    rebuilding the grid dominated P03's time). Same maths; checked against
    p_value in tests/engines/test_p03_conformity.py."""

    __slots__ = ("rec", "lg", "qg", "ql", "ps", "lo", "hi", "vmin", "vmax", "n", "rng", "th", "tl", "ok")

    def __init__(self, rec: Mapping[str, Any]) -> None:
        self.rec = rec
        self.lg = bool(rec.get("log"))
        qg = rec.get("qgrid") or []
        self.qg = np.asarray(qg, dtype=np.float64)
        self.ok = self.qg.size >= 2 and bool(np.all(np.isfinite(self.qg))) and "band98" in rec
        self.ps = np.linspace(0.0, 1.0, self.qg.size) if self.ok else None
        self.ql = self.qg.tolist()
        self.n = float(rec.get("n_c", NAN))
        self.rng = rec.get("range")
        self.th, self.tl = rec.get("tail_hi"), rec.get("tail_lo")
        if self.ok:
            self.lo = PB._fwd(rec["band98"][0], self.lg)
            self.hi = PB._fwd(rec["band98"][1], self.lg)
            self.vmin, self.vmax = float(self.qg[0]), float(self.qg[-1])

    def p(self, v: Any) -> Tuple[float, List[str]]:
        flags: List[str] = []
        try:
            x = float(v)
        except (TypeError, ValueError):
            return NAN, flags
        if not math.isfinite(x):
            return NAN, flags
        rng = self.rng
        if rng is not None:
            if x > rng[1]:
                flags.append("above_range")
            elif x < rng[0]:
                flags.append("below_range")
        nn = self.n
        if not (nn >= PB.N_MIN) or not self.ok:
            return NAN, flags
        y = PB._fwd(x, self.lg)
        if not math.isfinite(y):
            return SC.conformal_rank_p(0, nn), flags
        if self.lo <= y <= self.hi:
            if y <= self.vmin:
                F = 0.0
            elif y >= self.vmax:
                F = 1.0
            else:
                # np.interp over the grid (quantile levels i / (m - 1)), in pure
                # Python: a scalar np.interp call costs more than the search
                ql = self.ql
                j = bisect.bisect_right(ql, y)             # ql[j-1] <= y < ql[j]
                x0, x1 = ql[j - 1], ql[j]
                m1 = len(ql) - 1
                F = ((j - 1) + ((y - x0) / (x1 - x0) if x1 > x0 else 0.0)) / m1
            return SC.numeric_p(F), flags
        if y > self.hi:
            if self.th:
                th = self.th
                return SC.tail_p(x, th[0], th[1], th[2], PB.TAIL_MASS, k=_tail_k(th),
                                 x_max=self._extreme(True)), flags
            return SC.conformal_rank_p(0.0 if y > self.vmax else 0.01 * nn, nn), flags
        if self.tl:
            tl = self.tl
            xm = self._extreme(False)
            return SC.tail_p(-x, -tl[0], tl[1], tl[2], PB.TAIL_MASS, k=_tail_k(tl),
                             x_max=None if xm is None else -xm), flags
        return SC.conformal_rank_p(0.0 if y < self.vmin else 0.01 * nn, nn), flags

    def _extreme(self, upper: bool) -> Optional[float]:
        """The largest (smallest) observed value in natural units: the record's
        clean range, else the digest's extreme."""
        rng = self.rng
        try:
            if rng is not None:
                return float(rng[1] if upper else rng[0])
            v = self.vmax if upper else self.vmin
            return float(PB._inv(v, self.lg))
        except (TypeError, ValueError, IndexError, OverflowError):
            return None




def _ref_check(rec: Any, v: Any) -> Tuple[float, List[str]]:
    p, fl = rec.p(v)
    return _ref_in_range_floor(rec.rec, v, p), fl


def _same(a: float, b: float) -> bool:
    return (a != a and b != b) or a == b


def _fits():
    rng = np.random.default_rng(5)
    out = []
    for log, gen in ((True, lambda: float(rng.lognormal(7, 0.5))), (False, lambda: float(rng.normal(50, 8))),
                     (False, lambda: float(rng.integers(1, 40))), (True, lambda: float(rng.pareto(2.5) * 100 + 1)),
                     (False, lambda: float(rng.gamma(2.0, 30.0)))):
        num = PN.NumSummary(log=log)
        for i in range(3000):
            num.update(gen(), T0 + i * 60, 1.0, 1.0, day=int(i // 400))
        fit = PB.fit_numeric(num, T0 + 3000 * 60, 8, n_c=3000.0, n_eff=2000.0)
        out.append(fit)
        lean = dict(fit)
        lean.pop("tail_hi", None)
        lean.pop("range", None)
        out.append(lean)
        few = dict(fit, n_c=5.0)
        out.append(few)
    return out


def test_numfast_returns_the_reference_bits():
    rng = np.random.default_rng(9)
    n = 0
    for fit in _fits():
        new, ref = CF._NumFast(fit), _RefNumFast(fit)
        q = [float(x) for x in (fit.get("qgrid") or [1.0])]
        lo = min(q) if q else 0.0
        hi = max(q) if q else 1.0
        span = (hi - lo) or 1.0
        xs = [float(x) for x in rng.uniform(lo - span, hi + 2 * span, 400)]
        if fit.get("log"):
            xs = [math.exp(x) for x in xs]
        rg = fit.get("range") or [0.0, 1.0]
        xs += [rg[0], rg[1], rg[0] - 1.0, rg[1] + 1.0, rg[1] * 1.001, rg[1] * 10, 0.0, -1.0, 1e300,
               float("nan"), float("inf"), -float("inf"), "x", None, 7]
        for v in xs:
            a, fa = CF._check("num", new, v, None)
            b, fb = _ref_check(ref, v)
            assert _same(a, b) and fa == fb, (fit.get("log"), v, a, b)
            n += 1
    assert n > 5000


def test_prepared_gpd_tail_equals_the_one_shot_predictive():
    rng = np.random.default_rng(2)
    for i in range(5000):
        xi = float(rng.uniform(-0.6, 0.6))
        sigma = float(rng.uniform(0.01, 1000.0)) if i % 50 else float(rng.choice([-1.0, float("nan")]))
        k = float(rng.integers(1, 64))
        zm = None if i % 3 == 0 else float(rng.uniform(-5.0, 5000.0))
        g = SC.GpdPredictive(xi, sigma, k, zm)
        for z in (float(rng.uniform(-10.0, 10000.0)), 0.0, 1e-9, 1e6):
            assert _same(g.sf(z), SC.gpd_predictive_sf(z, xi, sigma, k, zm))
