"""Numeric content constraints of the progressive core (docs/lib3/progressive.md §6.10).

STATUS: implemented (W-P4, P06 maths). Pure functions over a pnode.NumSummary
(t-digest at H_m, moments, the daily (min, max, n_obs) ring of the current
confidence segment, exceedance reservoirs); no store access, no state.

fit_numeric(num, t, day_now, ...) -> record
    band90  = [Q(0.05), Q(0.95)]    "90 % in ..." (shape, H_m), Q of the digest
              truncated to the clean range (clean_mass: the band describes the
              range's rows; violations learned at full weight leave it)
    band98  = [Q(0.01), Q(0.99)]
    range   = observed [min, max] over the ring's days of the current
              confidence segment;  n_rng = evidence units observed on those days
    cover   = Beta(2, n_rng - 1) 95 % quantile: an upper bound of P(next row outside
              this range) (round 4; the predictive rank bound 2 / (n_rng + 1) is cover_pred)
    hard    = n_rng >= 30 and approx share <= 0.2 ("100 % in [a, b]" may be rendered)
    tail_hi / tail_lo = [u, xi, sigma, n] in natural units: GPD (lib/evt.gpd_pwm_fit) on the
              exceedances over Q(0.90) (under Q(0.10)) when >= 30 of them
    qgrid   = 33 quantiles of the transformed value y (compact distribution kept
              in the fitted model, so a fitted record scores without the summary)
    disp90 / disp_range = display rounding (1-2-5 x 10^k grid in the attribute's
              unit, bytes in KB / MB, coverage of the rounded band kept in [0.88, 0.95])
    Values are in the attribute's natural unit (the summary's log transform is undone).
    When the approx share exceeds 0.2 no hard bound is published: `range` and `cover`
    are omitted and the observed extremes go to `observed` (§5.1.2).

p_value(rec, v, num=None, t=None, n=None) -> (p, flags)
    §6.10 scoring: inside band98 two-sided mid-p from the CDF; beyond, the doubled
    GPD tail (tail mass 0.1) or, without a tail fit, the conformal rank
    (1 + #{as extreme}) / (n + 1); n is evidence on the confidence channel.

bin_gain(num, sys_edges) -> bits/event the node's value distribution saves over the
    system's (plug-in KL against the registry's 8 equal-mass bins); the gain P12
    compares across arms (§6.18.2).
"""
from __future__ import annotations

import fnmatch
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import pmdl
from . import pscore
from .evt import gpd_pwm_fit
from . import timebins as TB

BAND_LO, BAND_HI = 0.05, 0.95
BAND98_LO, BAND98_HI = 0.01, 0.99
COVER_RANGE = (0.88, 0.95)            # rounded band coverage kept in this interval
TOL_MAX = 0.03                         # ... widened by <= 2 binomial s.e., at most 0.03
HARD_N = 30.0                          # n_rng needed to render "100 % in [a, b]"
APPROX_MAX = 0.2
GPD_MIN = 30
TAIL_MASS = 0.1
QGRID = 33
N_MIN = 20.0
BYTE_GLOBS = ("*bytes*", "body.len", "*.size", "*size_b*", "hdr.content-length", "net.bytes_*")
KB = 1024.0

NAN = float("nan")


# ------------------------------------------------------------------ helpers
def _fin(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


def unit_of(attr: str, byte_globs: Sequence[str] = BYTE_GLOBS) -> str:
    """'B' for byte-valued attributes (configurable globs), else ''."""
    for g in byte_globs:
        if fnmatch.fnmatchcase(attr, g):
            return "B"
    return ""


def _inv(y: float, log: bool) -> float:
    if not _fin(y):
        return NAN
    return float(math.exp(y)) if log else float(y)


def _fwd(v: float, log: bool) -> float:
    v = float(v)
    if log:
        return math.log(v) if v > 0 else -math.inf
    return v


def grid_steps(scale: float, unit: str) -> List[float]:
    """1-2-5 x 10^k grid steps (natural units), coarse to fine, around `scale`."""
    base = KB if unit == "B" and scale >= KB else 1.0
    s = max(abs(scale), 1e-12) / base
    k0 = int(math.floor(math.log10(s))) + 1
    out = []
    for k in range(k0, k0 - 7, -1):
        for m in (5.0, 2.0, 1.0):
            out.append(m * (10.0 ** k) * base)
    return out


def fmt_num(v: float, unit: str) -> Tuple[str, str]:
    """Human number and unit: bytes as B / KB / MB (1024-based)."""
    if not _fin(v):
        return "?", unit
    if unit == "B":
        for u, f in (("MB", KB * KB), ("KB", KB)):
            if abs(v) >= f:
                return _trim(v / f), u
        return _trim(v), "B"
    return _trim(v), unit


def _trim(x: float) -> str:
    if abs(x - round(x)) < 1e-9:
        return str(int(round(x)))
    s = f"{x:.3g}"
    return s


def round_band(lo: float, hi: float, cdf, unit: str = "",
               cover: Tuple[float, float] = COVER_RANGE, n: float = math.inf) -> Dict[str, Any]:
    """Display rounding of a 90 % band (§6.10): the coarsest 1-2-5 grid not
    coarser than the band's width whose rounded band keeps its coverage (cdf in
    natural units) in `cover`, widened by two binomial standard errors of a
    90 % coverage estimated from n evidence units (with n = 200 the empirical
    coverage of the true 1-2 KB band is itself 0.90 +- 0.02). Nearest rounding
    of both endpoints is tried first, then outward rounding; the coverage
    check is what keeps the rendering honest. Falls back to the raw endpoints."""
    if not (_fin(lo) and _fin(hi)) or hi <= lo:
        return {"lo": lo, "hi": hi, "coverage": NAN, "grid": None, "text": _text(lo, hi, unit)}
    tol = min(TOL_MAX, 2.0 * math.sqrt(0.09 / n)) if (n > 0 and math.isfinite(n)) else 0.0
    c_lo, c_hi = cover[0] - tol, cover[1] + tol
    best = None
    for g in grid_steps(max(abs(hi), abs(lo)), unit):
        if g > (hi - lo) * 1.0001:
            continue
        near = (round(lo / g) * g, round(hi / g) * g)
        out = (math.floor(lo / g + 1e-9) * g, math.ceil(hi / g - 1e-9) * g)
        for a, b in (near, out):
            if b <= a or (lo >= 0 and a < 0):
                continue
            c = float(cdf(b) - cdf(a))
            if c_lo - 1e-9 <= c <= c_hi + 1e-9:
                best = {"lo": float(a), "hi": float(b), "coverage": c, "grid": float(g)}
                break
        if best is not None:
            break
    if best is None:
        best = {"lo": float(lo), "hi": float(hi), "coverage": float(cdf(hi) - cdf(lo)), "grid": None}
    elif lo > 0 and best["lo"] <= 0 and best.get("grid") is not None:
        # a positive lower edge never reads 0: on a grid as coarse as the upper
        # edge's, a wide band of a heavy-tailed size ('1.1 KB - 230 KB' mail
        # uploads, '0.8 KB - 1.6 MB' git pushes) rendered '0-200 KB', which
        # states no lower bound at all; the lower edge is rounded on its own
        # 1-2-5 grid (coarsest step <= the edge) under the same coverage check
        for g2 in _lo_steps(lo, best["hi"], unit):
            for a2 in (round(lo / g2) * g2, math.floor(lo / g2 + 1e-9) * g2):
                if a2 <= 0 or a2 >= best["hi"]:
                    continue
                c = float(cdf(best["hi"]) - cdf(a2))
                if c_lo - 1e-9 <= c <= c_hi + 1e-9:
                    best = dict(best, lo=float(a2), coverage=c, grid_lo=float(g2))
                    break
            if best["lo"] > 0:
                break
    best["text"] = _text(best["lo"], best["hi"], unit)
    return best


def _lo_steps(lo: float, hi: float, unit: str) -> List[float]:
    """1-2-5 steps not coarser than a positive lower edge, in the unit the
    band is displayed in (KB steps when the upper edge reads in KB)."""
    scale = max(lo, KB) if (unit == "B" and hi >= KB) else lo
    return [g for g in grid_steps(scale, unit) if g <= lo * 1.0001]


def round_range(lo: float, hi: float, unit: str = "") -> Dict[str, Any]:
    """Outward rounding of an observed range to the coarsest 1-2-5 grid not
    wider than a quarter of the range (so '512-3071 B' reads '0.5-3 KB')."""
    if not (_fin(lo) and _fin(hi)):
        return {"lo": lo, "hi": hi, "text": _text(lo, hi, unit)}
    w = hi - lo
    if w <= 0:
        return {"lo": lo, "hi": hi, "text": _text(lo, hi, unit)}
    for g in grid_steps(max(abs(hi), abs(lo)), unit):
        if g <= w / 4.0 + 1e-12:
            a = math.floor(lo / g + 1e-9) * g
            b = math.ceil(hi / g - 1e-9) * g
            if lo >= 0:
                a = max(a, 0.0)
            if lo > 0 and a <= 0:
                # a positive observed minimum never reads 0 (see round_band): it is
                # floored on its own 1-2-5 grid ('800 B - 293 KB' read '0-300 KB')
                g2 = next(iter(_lo_steps(lo, b, unit)), None)
                if g2 is not None:
                    a = math.floor(lo / g2 + 1e-9) * g2
            return {"lo": float(a), "hi": float(b), "grid": float(g), "text": _text(a, b, unit)}
    return {"lo": lo, "hi": hi, "text": _text(lo, hi, unit)}


def _text(lo: float, hi: float, unit: str) -> str:
    if unit == "B" and _fin(hi):
        # render both ends in the unit of the upper end
        f, u = (KB * KB, "MB") if abs(hi) >= KB * KB else (KB, "KB") if abs(hi) >= KB else (1.0, "B")
        if _fin(lo) and 0 < abs(lo) < 0.1 * f:
            # a lower edge far below the upper one reads in its own unit ('0.5 KB–2 MB')
            a, ua = fmt_num(lo, unit)
            return f"{a} {ua}–{_trim(hi / f)} {u}"
        return f"{_trim(lo / f)}–{_trim(hi / f)} {u}"
    a, _ = fmt_num(lo, unit)
    b, _ = fmt_num(hi, unit)
    return f"{a}–{b}" + (f" {unit}" if unit else "")


# ---------------------------------------------------------------------- fit
def _close(a: float, b: float) -> bool:
    return abs(float(a) - float(b)) <= 1e-9 * (1.0 + abs(float(b)))


def clean_range(num: Any, day_now: int, excl: Optional[Mapping[int, Sequence[float]]] = None,
                day_of: Any = None) -> Tuple[float, float, float, int]:
    """(min, max, n_rng, days dropped) of the node's daily ring over the
    current confidence segment (pnode.NumSummary.observed_range), without
    the values of rows P03 judged violations (§6.9.3: a violation never
    teaches the pattern; an undamped 12 KB injection login must not become
    the login's stated maximum). excl = {local day: [y values]} (transformed
    scale). A day whose extreme is such a value takes its next extreme from
    the exceedance reservoirs (values beyond the running Q(0.10) / Q(0.90),
    with their timestamps; day_of(ts) -> local day); when they hold none for
    that day, that SIDE of the day leaves the range (its other extreme is a
    clean observation and stays). The minimum and the maximum may then be
    taken over different days: n_rng is the smaller of the two samples
    (Σ n_obs over the days each side was taken from, a day whose excluded
    maximum lies below the stated maximum counting for that side too, since
    all its clean rows do; the excluded rows leave n_obs), so the rank bound
    2 / (n_rng + 1) >= 1 / (n_lo + 1) + 1 / (n_hi + 1) stays conservative.
    (2026-10-01: dropping the whole day lost 7 of 10 days of pack O's GA
    login node, whose daily maximum was often a damped row: n_rng = 3.)
    Days dropped = days that lost both sides."""
    if not excl:
        lo, hi, n = num.observed_range(int(day_now))
        return lo, hi, n, 0
    r = num.ring
    ok = np.isfinite(r[:, 0]) & (r[:, 0] > day_now - len(r)) & (r[:, 0] <= day_now)
    if getattr(num, "seg_day", None) is not None:
        ok &= r[:, 0] >= num.seg_day
    if not ok.any():
        return NAN, NAN, 0.0, 0
    res_hi = res_lo = None
    los, his, n_lo, n_hi, dropped = [], [], 0.0, 0.0, 0
    lo_out, hi_out = [], []                 # (excluded extreme, n) of sides that left the range
    for row in r[ok]:
        d, lo, hi, n = int(row[0]), float(row[1]), float(row[2]), float(row[3])
        bad = [float(y) for y in (excl.get(d) or ())]
        bad_lo = bad_hi = NAN
        if bad:
            if any(_close(hi, y) for y in bad):
                bad_hi = hi
                if res_hi is None:
                    res_hi = [(float(y), float(t)) for y, _w, t in num.hi_res.items()]
                c = [y for y, t in res_hi if day_of is not None and day_of(t) == d and y < hi
                     and not any(_close(y, b) for b in bad)]
                hi = max(c) if c else NAN
            if any(_close(lo, y) for y in bad):
                bad_lo = lo
                if res_lo is None:
                    res_lo = [(float(y), float(t)) for y, _w, t in num.lo_res.items()]
                c = [y for y, t in res_lo if day_of is not None and day_of(t) == d and y > lo
                     and not any(_close(y, b) for b in bad)]
                lo = min(c) if c else NAN
            # the excluded extremes were rows of the ring's count: they leave it
            n = max(0.0, n - float(_fin(bad_hi)) - float(_fin(bad_lo) and not _close(bad_lo, bad_hi)))
        if not (_fin(lo) or _fin(hi)):
            dropped += 1
        if _fin(lo):
            los.append(lo)
            n_lo += n
        elif _fin(bad_lo):
            lo_out.append((bad_lo, n))
        if _fin(hi):
            his.append(hi)
            n_hi += n
        elif _fin(bad_hi):
            hi_out.append((bad_hi, n))
    if not los or not his:
        return NAN, NAN, 0.0, dropped
    ymin, ymax = float(min(los)), float(max(his))
    # a day whose maximum left the range still lies wholly below the stated
    # maximum when its excluded value does: its clean rows are part of the
    # sample the maximum was taken over (and symmetrically for the minimum)
    n_hi += sum(n for v, n in hi_out if v <= ymax)
    n_lo += sum(n for v, n in lo_out if v >= ymin)
    return ymin, ymax, float(min(n_lo, n_hi)), dropped


def fit_numeric(num: Any, t: float, day_now: int, n_c: float = NAN, n_eff: float = NAN,
                approx_share: float = 0.0, unit: str = "", pin: Optional[Mapping[str, Any]] = None,
                excl: Optional[Mapping[int, Sequence[float]]] = None, day_of: Any = None
                ) -> Optional[Dict[str, Any]]:
    """Fitted numeric constraint of one node attribute (§6.10); None when the
    summary is empty. n_c / n_eff: the attribute's evidence on the confidence
    channel / at H_m (the caller scales the node's evidence by the attribute's
    presence). pin: operator bounds {'lo', 'hi'} (B23 `expected`), which widen
    the published range and never narrow it."""
    td = num.td
    if td.total() <= 0 or td.n_centroids() == 0:
        return None
    lg = bool(num.log)
    ymin, ymax, n_rng, n_drop = clean_range(num, int(day_now), excl, day_of)
    rng = [_inv(ymin, lg), _inv(ymax, lg)] if n_rng > 0 else None
    approx = float(approx_share) if _fin(approx_share) else 0.0
    # the band is a statement about the same rows as the range (round 4)
    pl, ph = clean_mass(td, ymin, ymax) if (rng is not None and approx <= APPROX_MAX) else (0.0, 1.0)
    span = ph - pl
    # every quantile of the fit in one pass over the digest (td_quantiles: the
    # same floats as td.quantile(p) per p, the centroid cumsum taken once)
    ps = [BAND_LO, BAND_HI, BAND98_LO, BAND98_HI, 0.9, 0.1] + [i / (QGRID - 1) for i in range(QGRID)]
    if not (pl <= 0.0 and ph >= 1.0):
        ps = [pl + p * span for p in ps]
    qv = td_quantiles(td, ps)
    y05, y95, y01, y99 = qv[0], qv[1], qv[2], qv[3]
    band90 = [_inv(y05, lg), _inv(y95, lg)]
    band98 = [_inv(y01, lg), _inv(y99, lg)]
    cov90 = min(1.0, closed_coverage(td, y05, y95) / span) if _fin(y05) and _fin(y95) else NAN
    rec: Dict[str, Any] = {
        "kind": "num", "log": lg, "unit": unit,
        "band90": band90, "band98": band98, "coverage_emp": cov90,
        "coverage": coverage_lb(cov90, n_eff),
        "n_rng": float(n_rng), "n_eff": float(n_eff) if _fin(n_eff) else NAN,
        "n_c": float(n_c) if _fin(n_c) else NAN, "approx": approx,
        "mass": float(td.total(t)), "excluded_days": int(n_drop),
        "qgrid": [float(v) for v in qv[6:]],
    }
    if pin:
        if rng is not None:
            if _fin(pin.get("lo")):
                rng[0] = min(rng[0], float(pin["lo"]))
            if _fin(pin.get("hi")):
                rng[1] = max(rng[1], float(pin["hi"]))
        rec["pinned"] = {k: pin[k] for k in ("lo", "hi") if k in pin}
    if rng is not None:
        if approx <= APPROX_MAX:
            rec["range"] = rng
            # the rank bound 2 / (n + 1) is predictive (averaged over samples);
            # the realised range's own exceedance mass is Beta(2, n - 1) and its
            # 95 % upper bound is the per-statement guarantee - the STATED cover
            # (round 4), as the band states its coverage's 95 % lower bound
            # (coverage_lb): "下次越界概率 ≤ x %" is then an upper bound of this
            # range's exceedance. With the predictive value a correct range drawn
            # from a sample that happened to miss a tail failed its held-out
            # check: pack O seed 0, day 21, the 综合部 login range 1-2.6 KB from 39
            # logins (none of the 5 % below 1 KB drawn) stated 0.05, held 0.115
            # (Beta(2, 38): 95 % bound 0.12)
            rec["cover_pred"] = 2.0 / (n_rng + 1.0)
            rec["cover_hi"] = float(pmdl.beta_quantile(0.95, 2.0, max(n_rng - 1.0, 1e-6))) \
                if n_rng >= 1 else 1.0
            rec["cover"] = rec["cover_hi"]
            rec["hard"] = bool(n_rng >= HARD_N)
        else:
            rec["observed"] = rng
            rec["hard"] = False
    else:
        rec["hard"] = False
    rec["tail_hi"] = _tail(num, qv[4], upper=True)
    rec["tail_lo"] = _tail(num, qv[5], upper=False)
    cdf_nat = (lambda x: min(1.0, max(0.0, (td.cdf(_fwd(x, lg)) - pl) / span))
               if (not lg or x > 0) else 0.0)
    rec["disp90"] = round_band(band90[0], band90[1], cdf_nat, unit,
                               n=n_eff if _fin(n_eff) else math.inf)
    if rng is not None:
        rec["disp_range"] = round_range(rng[0], rng[1], unit)
    rec["confidence"] = confidence(rec)
    return rec


CLEAN_MIN = 1.0                        # digest mass (rows' worth) the clean range must hold to be read


def clean_mass(td: Any, ymin: float, ymax: float) -> Tuple[float, float]:
    """(F(ymin-), F(ymax)) of the digest at the clean range's ends, (0, 1) when
    the digest lies within it: the band is then read from the digest
    truncated to the range, F_c(y) = (F(y) - F(ymin-)) / (F(ymax) - F(ymin-)).

    Why: the range is taken over the CLEAN rows (clean_range leaves out the
    rows P03 judged violations - the P06 ledger - and the damped rows, which
    never reach the ring), so every conforming row of the range's days lies
    in it and digest mass beyond it is mass of rows that must not teach the
    pattern (§6.9.3). P04 learns a violation a delay after P03 judges it and,
    unless P03 also damped it, at full weight: the digest holds it, the ring
    does not. Pack O seeds 0-1, day 21: A10's ~150 comment posts without
    viewstate, above the comment range, entered portal POST /comment at full
    weight on the last day; the range stayed 1.5-3.0 KB but the band read
    1.6-8.3 KB (wider than the range itself; truth 1.6-2.9 KB). Mass below
    the range that a decayed digest keeps from before the ring's days (or the
    confidence segment) leaves too: the band and the range describe the same
    rows. Whatever share of the digest the clean rows hold: pack O seed 1,
    day 21, A10's rows were ~3/4 of the decayed digest mass of the comment
    node (whose confidence segment had restarted on day 19, n_rng 45), and a
    "range holds >= half the digest" guard left its band at 1.8-8.3 KB. Only a
    range holding less than CLEAN_MIN rows' worth of digest mass (nothing to
    read a quantile from) leaves the digest as it is."""
    if not (_fin(ymin) and _fin(ymax)) or ymax < ymin:
        return 0.0, 1.0
    vmin, vmax = float(getattr(td, "vmin", -math.inf)), float(getattr(td, "vmax", math.inf))
    pl = td.cdf(ymin - 1e-9 * (1.0 + abs(ymin))) if vmin < ymin else 0.0
    ph = td.cdf(ymax + 1e-9 * (1.0 + abs(ymax))) if vmax > ymax else 1.0
    if not (_fin(pl) and _fin(ph)) or (ph - pl) * float(td.total()) < CLEAN_MIN:
        return 0.0, 1.0
    return float(max(0.0, pl)), float(min(1.0, ph))


def _tail(num: Any, u: float, upper: bool) -> Optional[List[float]]:
    """GPD on the exceedances beyond u, fitted in the attribute's natural unit
    ([u_nat, xi, sigma, n]): the log transform shapes the band, but a
    peaks-over-threshold tail is a statement about excesses in the observed
    unit (fitting it on log values turns the drop from a dense body into a
    sparse shoulder into a spurious power law)."""
    if not _fin(u):
        return None
    lg = bool(num.log)
    ex = num.exceedances(upper=upper)
    ex = ex[np.isfinite(ex)]
    sel = ex[ex > u] if upper else ex[ex < u]
    if sel.size < GPD_MIN:
        return None
    un = _inv(u, lg)
    xs = np.exp(sel) if lg else sel
    exc = (xs - un) if upper else (un - xs)
    exc = exc[exc > 0]
    if exc.size < GPD_MIN:
        return None
    xi, sigma = gpd_pwm_fit(exc)
    if not (_fin(xi) and _fin(sigma) and sigma > 0):
        return None
    return [float(un), float(xi), float(sigma), int(exc.size)]


def closed_coverage(td: Any, lo: float, hi: float) -> float:
    """Digest mass in the closed band [lo, hi] (the stated claim is "within
    lo-hi", endpoints included). cdf(hi) - cdf(lo) drops the point mass at lo:
    a constant attribute (net.pkts_down = 2 on every login) has band [2, 2]
    and open-interval coverage 0, which made its statement confidence 1e-308
    and every statement of the node read "置信 0.00" (measured on pack O)."""
    if not (_fin(lo) and _fin(hi)):
        return NAN
    e_lo = 1e-9 * (1.0 + abs(float(lo)))
    e_hi = 1e-9 * (1.0 + abs(float(hi)))
    return float(min(1.0, max(0.0, td.cdf(float(hi) + e_hi) - td.cdf(float(lo) - e_lo))))


def coverage_lb(c: float, n: float, q: float = 0.05) -> float:
    """Lower confidence bound of the true coverage of a band whose in-sample
    coverage is c on n evidence units: an interval between order statistics
    that holds k of n observations covers Beta(k, n - k + 1) of the
    distribution, so the stated coverage is its q-quantile (the band itself is
    chosen from the same sample, and at n = 20 a 90 % in-sample band covers
    0.72-0.95 of fresh events). It rises to c as evidence grows: the published
    "90 % in [a, b]" claim becomes more precise with observation time."""
    if not (_fin(c) and _fin(n)) or n <= 0:
        return NAN
    k = max(1e-6, float(c) * float(n))
    return float(pmdl.beta_quantile(q, k, max(1e-6, float(n) - k) + 1.0))


def confidence(rec: Mapping[str, Any]) -> float:
    """Statement confidence of a numeric constraint (§6.17.2): the stated
    (lower-bound) coverage of the band over its nominal 0.9, times 1 - cover
    for a hard range. (The spec's 1 - |c - nominal| / nominal is ~1 by
    construction for an in-sample band; the lower bound is what grows with time.)"""
    c = rec.get("coverage")
    ce = rec.get("coverage_emp", c)
    conf = min(1.0, float(c) / 0.9) if _fin(c) else (0.5 if _fin(ce) else 0.0)
    if rec.get("hard") and _fin(rec.get("cover")):
        conf = min(conf, 1.0 - float(rec["cover"]))
    return float(conf)


def material_change(old: Optional[Mapping[str, Any]], new: Mapping[str, Any], rel: float = 0.25) -> bool:
    """A band endpoint moved by more than 25 % (§6.8.2 cver rule)."""
    if not old or "band90" not in old:
        return True
    for a, b in zip(old["band90"], new["band90"]):
        if not (_fin(a) and _fin(b)):
            continue
        if abs(a - b) > rel * max(abs(a), 1e-9):
            return True
    return False


# ------------------------------------------------------------------ scoring
def _grid_cdf(qgrid: Sequence[float], y: float) -> float:
    g = np.asarray(qgrid, dtype=np.float64)
    if g.size < 2 or not np.all(np.isfinite(g)):
        return NAN
    ps = np.linspace(0.0, 1.0, g.size)
    if y <= g[0]:
        return 0.0
    if y >= g[-1]:
        return 1.0
    return float(np.interp(y, g, ps))


def p_value(rec: Mapping[str, Any], v: Any, num: Any = None, t: Optional[float] = None,
            n: Optional[float] = None, n_min: float = N_MIN) -> Tuple[float, List[str]]:
    """(p, flags) of value v against a fitted numeric constraint (§6.10):
    inside band98 the two-sided mid-p from the CDF (the live t-digest when
    `num` is given, else the record's compact quantile grid); beyond, the
    doubled GPD tail 2 * 0.1 * sf(excess) in natural units, or the conformal
    rank (1 + #{as extreme}) / (n + 1) without a tail fit. n = evidence on the
    confidence channel (default the record's n_c). flags: 'above_range' /
    'below_range' when v lies outside the published hard range."""
    flags: List[str] = []
    try:
        x = float(v)
    except (TypeError, ValueError):
        return NAN, flags
    if not math.isfinite(x):
        return NAN, flags
    nn = float(n if n is not None else rec.get("n_c", NAN))
    rng = rec.get("range")
    if rng is not None:
        if x > rng[1]:
            flags.append("above_range")
        elif x < rng[0]:
            flags.append("below_range")
    if not (nn >= n_min):
        return NAN, flags
    lg = bool(rec.get("log"))
    y = _fwd(x, lg)
    live = num is not None and num.td.total() > 0
    qg = rec.get("qgrid") or []
    if not live and len(qg) < 2:
        return NAN, flags
    if not math.isfinite(y):
        return pscore.conformal_rank_p(0, nn), flags
    if live:
        lo, hi = num.td.quantile(BAND98_LO), num.td.quantile(BAND98_HI)
        vmin, vmax = num.td.vmin, num.td.vmax
        F = num.td.cdf
    else:
        lo, hi = _fwd(rec["band98"][0], lg), _fwd(rec["band98"][1], lg)
        vmin, vmax = qg[0], qg[-1]
        F = lambda z: _grid_cdf(qg, z)          # noqa: E731
    if lo <= y <= hi:
        return pscore.numeric_p(F(y)), flags
    th, tl = rec.get("tail_hi"), rec.get("tail_lo")
    if y > hi:
        if th:
            return pscore.tail_p(x, th[0], th[1], th[2], TAIL_MASS), flags
        return pscore.conformal_rank_p(0.0 if y > vmax else 0.01 * nn, nn), flags
    if tl:
        return pscore.tail_p(-x, -tl[0], tl[1], tl[2], TAIL_MASS), flags
    return pscore.conformal_rank_p(0.0 if y < vmin else 0.01 * nn, nn), flags


# ------------------------------------------------- batched digest reads
def td_quantiles(td: Any, ps: Sequence[float]) -> List[Any]:
    """[td.quantile(p) for p in ps] for a psketch.TDigest with the centroid
    cumsum computed once: the same branches and the same float operations as
    TDigest.quantile (tests/lib/test_pbounds_batched_reads.py). Any other
    digest is read p by p."""
    if not (hasattr(td, "_mu") and hasattr(td, "_w") and hasattr(td, "_flush")):
        return [td.quantile(p) for p in ps]
    td._flush()
    mu, w = td._mu, td._w
    n = mu.size
    if n == 0:
        return [math.nan] * len(ps)
    qs = [min(1.0, max(0.0, float(q))) for q in ps]
    if n == 1:
        return [float(mu[0])] * len(qs)
    tot = float(w.sum())
    cum = np.cumsum(w) - w / 2.0
    c0, cl = cum[0], cum[-1]
    vmin, vmax = td.vmin, td.vmax
    m0, ml = float(mu[0]), float(mu[-1])
    targets = [q * tot for q in qs]
    js = np.searchsorted(cum, np.asarray(targets, dtype=np.float64), side="right").tolist()
    out: List[Any] = []
    for target, j in zip(targets, js):
        if target <= c0:
            frac = target / c0 if c0 > 0 else 0.0
            out.append(vmin + (m0 - vmin) * frac)
        elif target >= cl:
            rest = tot - cl
            frac = (target - cl) / rest if rest > 0 else 1.0
            out.append(ml + (vmax - ml) * frac)
        else:
            a, b = cum[j - 1], cum[j]
            frac = (target - a) / (b - a) if b > a else 0.0
            out.append(float(mu[j - 1] + (mu[j] - mu[j - 1]) * frac))
    return out


def td_cdfs(td: Any, xs: Sequence[float]) -> List[Any]:
    """[td.cdf(x) for x in xs] for a psketch.TDigest with the centroid cumsum
    computed once (TDigest.cdf's branches and float operations)."""
    if not (hasattr(td, "_mu") and hasattr(td, "_w") and hasattr(td, "_flush")):
        return [td.cdf(x) for x in xs]
    td._flush()
    mu, w = td._mu, td._w
    n = mu.size
    if n == 0:
        return [math.nan] * len(xs)
    vmin, vmax = td.vmin, td.vmax
    xs = [float(x) for x in xs]
    tot = float(w.sum())
    out: List[Any] = []
    if n == 1:
        for x in xs:
            if x < vmin:
                out.append(0.0)
            elif x >= vmax:
                out.append(1.0)
            else:
                span = vmax - vmin
                out.append((x - vmin) / span if span > 0 else 0.5)
        return out
    cum = np.cumsum(w) - w / 2.0
    m0, ml = float(mu[0]), float(mu[-1])
    js = np.searchsorted(mu, np.asarray(xs, dtype=np.float64), side="right").tolist()
    for x, j in zip(xs, js):
        if x < vmin:
            out.append(0.0)
        elif x >= vmax:
            out.append(1.0)
        elif x < m0:
            span = m0 - vmin
            out.append(float(cum[0] * ((x - vmin) / span if span > 0 else 1.0) / tot))
        elif x >= ml:
            span = vmax - ml
            frac = (x - ml) / span if span > 0 else 1.0
            out.append(float((cum[-1] + (tot - cum[-1]) * frac) / tot))
        else:
            a, b = float(mu[j - 1]), float(mu[j])
            frac = (x - a) / (b - a) if b > a else 0.5
            out.append(float((cum[j - 1] + (cum[j] - cum[j - 1]) * frac) / tot))
    return out


# --------------------------------------------------------------------- gain
def bin_gain(num: Any, sys_digest: Any, n_bins: int = 8) -> float:
    """Bits per event the node's distribution saves against the system's:
    KL(node || system) over the system's n_bins equal-mass bins (raw scale).
    A plug-in estimate (not prequential; P03 holds the losses)."""
    if sys_digest is None or sys_digest.total() <= 0 or num.td.total() <= 0:
        return 0.0
    edges = td_quantiles(sys_digest, [k / n_bins for k in range(1, n_bins)])
    lg = bool(num.log)
    xs = [_fwd(e, lg) for e in edges if not (lg and e <= 0)]
    cv = iter(td_cdfs(num.td, xs))
    cdf = [0.0 if (lg and e <= 0) else float(next(cv)) for e in edges]
    c = np.clip(np.asarray([0.0] + cdf + [1.0]), 0.0, 1.0)
    p = np.maximum(np.diff(np.maximum.accumulate(c)), 0.0)
    s = p.sum()
    if s <= 0:
        return 0.0
    p = p / s
    nz = p[p > 0]
    return float(max(0.0, np.sum(nz * np.log2(nz * n_bins))))


# ------------------------------------------------ fitter bookkeeping (P06-P08)
DIRTY_FRAC = 0.1
DIRTY_UNITS = 20.0


def fit_mark(node: Any, t: float) -> Dict[str, Any]:
    """What a fitter remembers about a node at its last fit (§6.20 dirty nodes)."""
    return {"t": float(t), "n_c": float(node.n_c(t)), "ver": int(node.version),
            "cver": int(node.cver), "state": str(node.state)}


def new_evidence(prev: Mapping[str, Any], node: Any, t: float) -> float:
    """Confidence-channel evidence that arrived since the last fit:
    n_c(t) - n_c(t_prev) 2^(-(t - t_prev)/H_l) (exact under forward decay;
    negative after a confidence reset)."""
    from .psketch import H_L
    return float(node.n_c(t) - float(prev["n_c"]) * 2.0 ** (-(float(t) - float(prev["t"])) / H_L))


def is_dirty(prev: Optional[Mapping[str, Any]], node: Any, t: float,
             frac: float = DIRTY_FRAC, units: float = DIRTY_UNITS) -> bool:
    """A node needs a refit when never fitted, when its structure / content
    version or state changed (drift, confirmation, reset), or when its
    evidence grew by >= 10 % or >= 20 units since the last fit (§6.20)."""
    if not prev:
        return True
    if int(prev.get("ver", -1)) != int(node.version) or int(prev.get("cver", -1)) != int(node.cver) \
            or str(prev.get("state")) != str(node.state):
        return True
    d = new_evidence(prev, node, t)
    if d < 0:
        return True
    return d >= min(units, max(frac * float(prev["n_c"]), 0.5))


def tree_changed(marks: Dict[Any, Any], kind: int, root: Any, t: float) -> bool:
    """True when a tree (kind) received evidence or changed structure since the
    previous call (marks is the caller's per-kind bookkeeping, updated here):
    a tree that learned nothing is skipped without touching its nodes (§6.20)."""
    prev = marks.get(kind)
    marks[kind] = fit_mark(root, t)
    if not prev:
        return True
    if int(prev.get("ver", -1)) != int(root.version) or str(prev.get("state")) != str(root.state):
        return True
    return abs(new_evidence(prev, root, t)) > 1e-9 * max(1.0, float(prev["n_c"]))


def fit_digest(td: Any, t: float, unit: str = "", n: float = NAN) -> Optional[Dict[str, Any]]:
    """Band / p99 of a bare TDigest target such as rate.ip_h (§6.16.4):
    no daily ring, so no hard range."""
    if td is None or td.total() <= 0 or td.n_centroids() == 0:
        return None
    qv = td_quantiles(td, [BAND_LO, BAND_HI, BAND98_LO, BAND98_HI, 0.99]
                      + [i / (QGRID - 1) for i in range(QGRID)])
    rec = {"kind": "num", "log": False, "unit": unit,
           "band90": [qv[0], qv[1]], "band98": [qv[2], qv[3]],
           "p99": qv[4], "hard": False, "mass": float(td.total(t)),
           "qgrid": [float(v) for v in qv[5:]],
           "coverage_emp": closed_coverage(td, qv[0], qv[1])}
    rec["coverage"] = coverage_lb(rec["coverage_emp"], n)
    rec["disp90"] = round_band(rec["band90"][0], rec["band90"][1], td.cdf, unit)
    rec["confidence"] = confidence(rec)
    return rec


# ------------------------------------------------------ fitter helpers (P06-P08)
CPINS = "model.cpins"


def pins_for(config: Mapping[str, Any], store: Any, key: str, attr: str) -> Optional[Dict[str, Any]]:
    """Operator pins of an attribute: config progressive.content_pins
    {attr glob: pin} then model.cpins@(key) {attr: pin} (a B23 `expected` hook
    writes the latter). Pins only widen published ranges."""
    out: Dict[str, Any] = {}
    cp = ((config or {}).get("progressive") or {}).get("content_pins") or {}
    for g, pin in cp.items():
        if fnmatch.fnmatchcase(attr, g) and isinstance(pin, Mapping):
            out.update(pin)
    m = store.get_model(key, "__system__", CPINS)
    if isinstance(m, Mapping) and isinstance(m.get(attr), Mapping):
        out.update(m[attr])
    return out or None


def chosen_arm(store: Any, key: str, names: tuple, default: str = "on") -> str:
    """P12's chosen arm for a dimension (model.sysprof['chosen']); `default`
    before P12 ran. Accepts any of the dimension's names."""
    sp = store.get_model(key, "__system__", "model.sysprof")
    ch = (sp or {}).get("chosen") if isinstance(sp, Mapping) else None
    if isinstance(ch, Mapping):
        for n in names:
            if n in ch:
                return str(ch[n])
    return default


def local_day(now: float, config: Mapping[str, Any]) -> int:
    return TB.local_datetime(float(now), (config or {}).get("tz") or TB.DEFAULT_TZ).date().toordinal()


def empty_model() -> Dict[str, Any]:
    return {"fmt": 1, "version": 0, "updated": None, "nodes": {}, "fit": {}, "gain": {},
            "last_run": None}


def lookup(model: Any, kind: int, nid: int, attr: Optional[str] = None) -> Any:
    """Fitted entry of a node in model.pbounds / model.pgrammar ({'status',
    'attrs', ...}), or one attribute's record (P03 / P14 accessor)."""
    if not isinstance(model, Mapping):
        return None
    ent = ((model.get("nodes") or {}).get(int(kind)) or {}).get(int(nid))
    if ent is None or attr is None:
        return ent
    return (ent.get("attrs") or {}).get(attr)


# ------------------------------------------------- content target requests
REQ_NODES = 32          # nodes per tree that receive requested content targets
REQ_PER_NODE = 3        # requested attributes per node and fitter
REQ_TRY_UNITS = 5.0     # evidence and active days a node must gain before an unmet
REQ_TRY_DAYS = 2        # request is given up (a login node is busy with page views all day
REQ_RETRY_DAYS = 7      # but sees the login body only at 9 am); retried after 7 active days
CONTEXT_NS = ("ctx", "ev", "sess")   # derived time / session / event context: not content (P09, P10 model it)
PAYLOAD_NS = ("body", "q", "hdr", "resp")   # the content of a request / response itself


def _same_quantity(a: str, b: str) -> bool:
    try:
        from .pselect import same_source
    except Exception:                          # pragma: no cover
        return a == b
    return bool(same_source(a, b))
M_T_P04 = 8             # m_t: P04 models the first m_t system targets without a request


def request_targets(store: Any, key: str, ptm: Any, t: float, types: Sequence[str],
                    book: Dict[str, Any], max_nodes: int = REQ_NODES,
                    per_node: int = REQ_PER_NODE) -> Dict[int, Dict[int, List[str]]]:
    """Content targets a fitter asks P04 to keep at its busiest nodes
    (model.pwant '<fitter>'.'targets', §5.6): the registry attributes of the
    fitter's types (P06 numeric; P07 set / text) that P05 found informative
    (role split or target), in P05's own ranking (split candidates, then
    targets), that the node does not model yet. Without such a request a
    split-only attribute (pack O's login body size and key set) is never a
    target, so the requirement's "90 % of submissions 1-2 KB" cannot be fitted.
    The derived context namespaces (CONTEXT_NS: time of day, session age /
    position, think time) are never requested: they are P09's / P10's, and
    ranked first they took every request slot of pack O's portal comment node.
    No attribute name is written in code. A request the node never fills (the
    attribute is absent there) is given up after the node gained REQ_TRY_UNITS
    evidence units on REQ_TRY_DAYS more active days, so the slot moves to the next
    candidate (retried after REQ_RETRY_DAYS). book: the caller's persistent
    bookkeeping {kind: {nid: {attr: [n_c, active days, state]}}}."""
    reg = None
    try:
        from . import m_ptree as MP
        reg = MP.get_registry(store, key)
        sel = MP.get_model(store, key, MP.ATTRSEL)
    except Exception:                          # pragma: no cover
        sel = None
    if reg is None:
        return {}
    roles = (sel or {}).get("roles") or {} if isinstance(sel, Mapping) else {}
    out: Dict[int, Dict[int, List[str]]] = {}
    for kind, tree in ptm.kinds.items():
        order: List[str] = []
        sc = ((sel or {}).get("split_cands") or {}) if isinstance(sel, Mapping) else {}
        for a, *_ in (sc.get(kind) or sc.get(str(kind)) or []):
            if a not in order:
                order.append(a)
        ts_ = ((sel or {}).get("targets_sys") or {}) if isinstance(sel, Mapping) else {}
        for a in (ts_.get(kind) or ts_.get(str(kind)) or []):
            if a not in order:
                order.append(a)
        if not roles:                          # before P05's first run: registry coverage
            order += sorted((a for a in reg.names() if a not in order),
                            key=lambda a: (-float(reg.coverage(a)), a))
        cands = []
        for a in order:
            if a.split(".", 1)[0] in CONTEXT_NS:
                # the derived context of an event (time of day, session age and
                # position, think time) is P09's / P10's (windows, workflow delay
                # bands), not content: pack O's portal comment node gave its 3
                # request slots to ctx.sess_age_s / ctx.sess_pos / ctx.think_s
                # (P05 split candidates, ranked first) and never received its
                # body size (PG1 portal comment content: band from 1 observation)
                continue
            rec = reg.get(a)
            if rec is None or getattr(rec, "type", None) not in types:
                continue
            if getattr(rec, "state", "active") == "gone":
                continue
            if roles and roles.get(a) not in ("split", "target"):
                continue
            cands.append(a)
        if not cands:
            continue
        numeric = any(ty not in ("set", "text") for ty in types)
        if numeric:
            # sizes (P06): the payload's own field before its transport measures
            # (stable order otherwise), and one quantity measured twice takes one
            # request slot (lib/pselect.same_source: request body and upstream
            # bytes / packets, response bytes and packets). Pack O seed 0, portal
            # comment node: the slots held net.bytes_down AND net.pkts_down (P05
            # ranked the transport measures first), the body size P04 had no room
            # for was given up and the statement had no "提交数据量" band
            cands.sort(key=lambda a: 0 if a.split(".", 1)[0] in PAYLOAD_NS else 1)
        nodes = [nd for nd in tree.nodes.values()
                 if nd.parent is not None and not getattr(nd, "is_exc", False)
                 and nd.state not in ("retired", "dormant")]
        nodes.sort(key=lambda nd: (-nd.mass_at(t), nd.id))
        kb = book.setdefault(kind, {})
        live = set()
        ovs = ((sel or {}).get("node_overrides") or {}) if isinstance(sel, Mapping) else {}
        ovs = ovs.get(kind) or ovs.get(str(kind)) or {}
        sys_base = list(ts_.get(kind) or ts_.get(str(kind)) or [])[:M_T_P04]
        for nd in nodes[:max_nodes]:
            live.add(nd.id)
            nb = kb.setdefault(nd.id, {})
            nc = float(nd.n_c(t))
            nd_days = int(nd.n_days())
            # what P04 models at this node on P05's word (nearest override, else
            # the system targets): those need no request
            cur, ov = nd, None
            while cur is not None and ov is None:
                ov = ovs.get(cur.id, ovs.get(str(cur.id)))
                cur = tree.nodes.get(cur.parent) if cur.parent is not None else None
            base = set(ov) if ov else set(sys_base)
            want: List[str] = []           # new requests: at most per_node
            keep: List[str] = []           # filled requests kept alive: at most per_node
            for a in cands:
                if len(want) >= per_node and len(keep) >= per_node:
                    break
                if a in base:
                    nb.pop(a, None)
                    continue
                if numeric and any(_same_quantity(a, b) for b in keep + want):
                    nb.pop(a, None)
                    continue
                if a in nd.targets:
                    if a in nb and len(keep) < per_node:
                        nb[a] = [nc, nd_days, 1]   # filled by our request: keep it (P04 drops
                        keep.append(a)             # extras that nobody requests any more)
                    continue                   # else P04 models it on its own
                if len(want) >= per_node:
                    continue
                st = nb.get(a)
                if st is None or (st[2] < 0 and nd_days - st[1] >= REQ_RETRY_DAYS):
                    st = nb[a] = [nc, nd_days, 0]
                elif st[2] >= 0 and nc - st[0] >= REQ_TRY_UNITS and nd_days - st[1] >= REQ_TRY_DAYS:
                    st[2] = -1                 # absent at this node: give the slot up
                    st[1] = nd_days
                if st[2] < 0:
                    continue
                want.append(a)
            want = keep + want
            if len(nb) > 64:                   # bounded bookkeeping per node
                for a in [x for x in nb if x not in want][:len(nb) - 64]:
                    del nb[a]
            if want:
                out.setdefault(kind, {})[nd.id] = want
        for nid in [n for n in kb if n not in live]:
            del kb[nid]
    return out
