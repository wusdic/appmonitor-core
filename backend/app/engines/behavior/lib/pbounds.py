"""Numeric content constraints of the progressive core (docs/lib3/progressive.md §6.10).

STATUS: implemented (W-P4, P06 maths). Pure functions over a pnode.NumSummary
(t-digest at H_m, moments, the daily (min, max, n_obs) ring of the current
confidence segment, exceedance reservoirs); no store access, no state.

fit_numeric(num, t, day_now, ...) -> record
    band90  = [Q(0.05), Q(0.95)]    "90 % in ..." (shape, H_m)
    band98  = [Q(0.01), Q(0.99)]
    range   = observed [min, max] over the ring's days of the current
              confidence segment;  n_rng = evidence units observed on those days
    cover   = 2 / (n_rng + 1)       P(next observed row outside range), exchangeability
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
    best["text"] = _text(best["lo"], best["hi"], unit)
    return best


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
            return {"lo": float(a), "hi": float(b), "grid": float(g), "text": _text(a, b, unit)}
    return {"lo": lo, "hi": hi, "text": _text(lo, hi, unit)}


def _text(lo: float, hi: float, unit: str) -> str:
    if unit == "B" and _fin(hi):
        # render both ends in the unit of the upper end
        f, u = (KB * KB, "MB") if abs(hi) >= KB * KB else (KB, "KB") if abs(hi) >= KB else (1.0, "B")
        return f"{_trim(lo / f)}–{_trim(hi / f)} {u}"
    a, _ = fmt_num(lo, unit)
    b, _ = fmt_num(hi, unit)
    return f"{a}–{b}" + (f" {unit}" if unit else "")


# ---------------------------------------------------------------------- fit
def fit_numeric(num: Any, t: float, day_now: int, n_c: float = NAN, n_eff: float = NAN,
                approx_share: float = 0.0, unit: str = "", pin: Optional[Mapping[str, Any]] = None
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
    q = td.quantile
    y05, y95, y01, y99 = q(BAND_LO), q(BAND_HI), q(BAND98_LO), q(BAND98_HI)
    band90 = [_inv(y05, lg), _inv(y95, lg)]
    band98 = [_inv(y01, lg), _inv(y99, lg)]
    cov90 = float(td.cdf(y95) - td.cdf(y05)) if _fin(y05) and _fin(y95) else NAN
    ymin, ymax, n_rng = num.observed_range(int(day_now))
    rng = [_inv(ymin, lg), _inv(ymax, lg)] if n_rng > 0 else None
    approx = float(approx_share) if _fin(approx_share) else 0.0
    rec: Dict[str, Any] = {
        "kind": "num", "log": lg, "unit": unit,
        "band90": band90, "band98": band98, "coverage_emp": cov90,
        "coverage": coverage_lb(cov90, n_eff),
        "n_rng": float(n_rng), "n_eff": float(n_eff) if _fin(n_eff) else NAN,
        "n_c": float(n_c) if _fin(n_c) else NAN, "approx": approx,
        "mass": float(td.total(t)),
        "qgrid": [float(q(i / (QGRID - 1))) for i in range(QGRID)],
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
            rec["cover"] = 2.0 / (n_rng + 1.0)
            # the rank bound is predictive (averaged over samples); the realised
            # range's own exceedance mass is Beta(2, n - 1): its 95 % upper bound
            # is the per-statement guarantee
            rec["cover_hi"] = float(pmdl.beta_quantile(0.95, 2.0, max(n_rng - 1.0, 1e-6))) \
                if n_rng >= 1 else 1.0
            rec["hard"] = bool(n_rng >= HARD_N)
        else:
            rec["observed"] = rng
            rec["hard"] = False
    else:
        rec["hard"] = False
    rec["tail_hi"] = _tail(num, q(0.9), upper=True)
    rec["tail_lo"] = _tail(num, q(0.1), upper=False)
    cdf_nat = (lambda x: td.cdf(_fwd(x, lg)) if (not lg or x > 0) else 0.0)
    rec["disp90"] = round_band(band90[0], band90[1], cdf_nat, unit,
                               n=n_eff if _fin(n_eff) else math.inf)
    if rng is not None:
        rec["disp_range"] = round_range(rng[0], rng[1], unit)
    rec["confidence"] = confidence(rec)
    return rec


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


# --------------------------------------------------------------------- gain
def bin_gain(num: Any, sys_digest: Any, n_bins: int = 8) -> float:
    """Bits per event the node's distribution saves against the system's:
    KL(node || system) over the system's n_bins equal-mass bins (raw scale).
    A plug-in estimate (not prequential; P03 holds the losses)."""
    if sys_digest is None or sys_digest.total() <= 0 or num.td.total() <= 0:
        return 0.0
    edges = [sys_digest.quantile(k / n_bins) for k in range(1, n_bins)]
    lg = bool(num.log)
    cdf = []
    for e in edges:
        if lg and e <= 0:
            cdf.append(0.0)
        else:
            cdf.append(float(num.td.cdf(_fwd(e, lg))))
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
    q = td.quantile
    rec = {"kind": "num", "log": False, "unit": unit,
           "band90": [q(BAND_LO), q(BAND_HI)], "band98": [q(BAND98_LO), q(BAND98_HI)],
           "p99": q(0.99), "hard": False, "mass": float(td.total(t)),
           "qgrid": [float(q(i / (QGRID - 1))) for i in range(QGRID)],
           "coverage_emp": float(td.cdf(q(BAND_HI)) - td.cdf(q(BAND_LO)))}
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
