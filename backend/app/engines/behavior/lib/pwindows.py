"""Activity windows on the circular day (docs/lib3/progressive.md §6.13, P09).

Pure functions, no store access; P09 (`behavior.time_window`) calls them on the
when-summaries P04 keeps per pattern node (hist96 per day type at H_m, and the
optional weighted minute reservoir, §5.5.4). P03 / P14 / eval may call
`in_windows`, `render_zh/en` and `as_intervals` on the published windows.

Algorithm (per node and day type):
  1. density  hist96 shares (mass, H_m); counts are scaled to the node's
              EVIDENCE n (PPC-9: shares from mass, sample size from evidence, so
              an aggregated or HT-thinned node does not look more certain than
              the observations behind it).
  2. unwrap   cut the circle at the middle of the longest run of minimal
              3-slot-smoothed density, so a window never straddles the cut
              (a 22:00-02:00 shift is one window, not two).
  3. blocks   Bayesian Blocks (Scargle, Norris, Jackson & Chiang 2013), optimal
              partition by dynamic programming, block fitness
              N_k (ln N_k - ln T_k), prior ncp = 4 - ln(73.53 p0 n^-0.478),
              p0 = 0.05; O(M^2) on M cells:
                slot mode    96 cells of 15 min;
                minute mode  (reservoir with >= MIN_POINTS points of the day type)
                             one 1-minute cell per occupied minute plus one
                             zero-count cell per empty run, so edges fall on
                             whole minutes (M <= 2 R + 1 = 513).
  4. windows  maximal runs of adjacent blocks with rate >= max(KAPPA x r_bg,
              F_MEAN x n/1440) and rate > 0, where r_bg = the time-weighted rate
              of the quietest blocks covering Q_OFF = 15 % of the day (the
              "off" hours); windows closer than one slot (15 min, the
              resolution of the density model) join; windows with < MIN_SHARE
              of the mass are dropped.
              A flat day (every block, or none, active) is one all-day window
              [0, 1440). Minute mode then trims sparse end groups that a flat
              density would put behind such a gap with prob. < 1 % and snaps
              edges over neighbouring points (Bayesian-Blocks edge effects).
  5. confidence coverage (mass share inside), n distinct dates and day stability
              (share of dates whose sampled arrivals all fall inside), the
              latter two from the reservoir's timestamps (None in slot mode).

Deviation from the text of §6.13 (measured in tests/lib/test_pwindows.py):
r_bg = mass/1440 rejects every broad pattern (08:00-20:00 activity has only 2x
the mean rate, below kappa = 3), and any background that includes the other
active blocks drops half of a broad pattern whenever Bayesian Blocks splits it
by chance; the background is therefore the rate of the quietest 15 % of the
day, with half the mean rate as the floor.

Window representation: [start, end) in local minutes, end exclusive, start >
end when the window wraps midnight (the convention of lib/phier.window_of).
The label is 'w:HHMM-HHMM' with the exclusive end (09:00 <= t < 09:21 ->
'w:0900-0921', rendered '09:00–09:21').
"""
from __future__ import annotations

import datetime as _dt
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DAY_MIN = 1440
SLOTS = 96
SLOT_MIN = 15
P0 = 0.05
KAPPA = 3.0
MIN_SHARE = 0.05
Q_OFF = 0.15                     # the quietest 15 % of the day is the background ...
F_MEAN = 0.5                     # ... and an active block has >= half the mean rate
MERGE_GAP = 15.0                 # windows closer than one slot are one window
MIN_POINTS = 10                  # reservoir points of a day type needed for minute mode
CLEAN_W = 0.5                    # a point weighing < 1/2 of a typical arrival is a damped outlier
REGIME_DATES = 3                 # §6.9.2: a time window change persists >= 3 workdays ...
REGIME_ALPHA = 1e-3              # ... and differs from the earlier arrivals at this level
REGIME_SINGLE_DATES = 5          # a change shown by one source only persists >= 5 dates
REGIME_POINTS = 6                # arrivals on each side of a change (3 workdays x 2); with so
                                 # few points only a clean separation passes alpha after Bonferroni

W_MAX = 8                        # windows per node and day type
MOVE_MIN = 10                    # an endpoint move > 10 min is a material change (cver + 1)
CONC_SHARE = 0.5                 # minute reservoir requested when >= 50 % of mass ...
CONC_SLOTS = 4                   # ... lies in <= 4 slots (§5.5.4)
DAYTYPES = ("wd", "nwd")
DT_LONG = {"wd": "workday", "nwd": "nonworkday"}


# ================================================================ helpers
def ncp_prior(n: float, p0: float = P0) -> float:
    """Scargle et al. 2013 eq. 21: ncp = 4 - ln(73.53 p0 n^-0.478)."""
    n = max(float(n), 1.0)
    return 4.0 - math.log(73.53 * p0 * n ** -0.478)


def smooth3(h: np.ndarray) -> np.ndarray:
    """Circular 3-slot box kernel."""
    h = np.asarray(h, dtype=np.float64)
    return (np.roll(h, 1) + h + np.roll(h, -1)) / 3.0


def unwrap_cut(hist: np.ndarray) -> int:
    """Minute at which to cut the circle: the start of the middle slot of the
    longest circular run of minimal smoothed density (ties -> earliest run)."""
    s = smooth3(hist)
    if s.size == 0:
        return 0
    lo = float(s.min())
    tol = 1e-12 + 1e-9 * float(s.max())
    m = s <= lo + tol
    if m.all():
        return 0
    n = m.size
    start0 = int(np.argmin(m))                    # a slot that is not minimal
    best_len, best_mid, run, run_start = 0, 0, 0, None
    for k in range(1, n + 1):
        i = (start0 + k) % n
        if m[i]:
            if run == 0:
                run_start = i
            run += 1
            if run > best_len:
                best_len, best_mid = run, (run_start + run // 2) % n
        else:
            run = 0
    return int(best_mid * (DAY_MIN // n))


def bayesian_blocks(counts: np.ndarray, widths: np.ndarray, ncp: float) -> List[Tuple[int, int]]:
    """Optimal partition of consecutive cells into blocks (Scargle et al. 2013,
    Algorithm 1, binned form). counts >= 0 (may be fractional), widths > 0.
    Returns [(first cell, last cell)] in order. O(M^2) time, O(M) memory."""
    c = np.asarray(counts, dtype=np.float64)
    w = np.asarray(widths, dtype=np.float64)
    M = c.size
    if M == 0:
        return []
    cumN = np.concatenate(([0.0], np.cumsum(c)))
    cumT = np.concatenate(([0.0], np.cumsum(w)))
    best = np.zeros(M)
    last = np.zeros(M, dtype=np.int64)
    for r in range(M):
        N = cumN[r + 1] - cumN[: r + 1]
        T = cumT[r + 1] - cumT[: r + 1]
        with np.errstate(divide="ignore", invalid="ignore"):
            fit = np.where(N > 0, N * (np.log(np.maximum(N, 1e-300)) - np.log(T)), 0.0)
        A = fit - ncp
        A[1:] += best[:r]
        j = int(np.argmax(A))
        last[r] = j
        best[r] = A[j]
    out: List[Tuple[int, int]] = []
    r = M - 1
    while r >= 0:
        j = int(last[r])
        out.append((j, r))
        r = j - 1
    return out[::-1]


def _circ(m: float) -> float:
    return float(m) % DAY_MIN


def in_window(minute: float, s: float, e: float) -> bool:
    m = _circ(minute)
    if s == 0 and e >= DAY_MIN:
        return True
    return (s <= m < e) if s <= e else (m >= s or m < e)


def in_windows(minute: float, windows: Iterable[Sequence[float]]) -> bool:
    return any(in_window(minute, w[0], w[1]) for w in windows)


def window_len(s: float, e: float) -> float:
    if s == 0 and e >= DAY_MIN:
        return float(DAY_MIN)
    return float(e - s) if s <= e else float(DAY_MIN - s + e)


def as_intervals(windows: Iterable[Sequence[float]]) -> List[List[int]]:
    """Windows as non-wrapping [m0, m1] intervals (a wrapping window becomes
    [s, 1440] and [0, e]); the format of the statement contract (pmetrics)."""
    out: List[List[int]] = []
    for w in windows:
        s, e = int(w[0]), int(w[1])
        if s <= e:
            out.append([s, e])
        else:
            out.append([s, DAY_MIN])
            if e > 0:
                out.append([0, e])
    return sorted(out)


def hhmm(m: float) -> str:
    m = int(round(m)) % (DAY_MIN + 1)
    if m >= DAY_MIN:
        return "24:00"
    return f"{m // 60:02d}:{m % 60:02d}"


def label(s: float, e: float) -> str:
    return "w:" + hhmm(s).replace(":", "") + "-" + hhmm(e).replace(":", "")


def concentrated(hist: np.ndarray, share: float = CONC_SHARE, slots: int = CONC_SLOTS) -> bool:
    """>= `share` of the mass in <= `slots` slots (P09 then asks P04 for the
    minute reservoir, §5.5.4)."""
    h = np.asarray(hist, dtype=np.float64)
    tot = h.sum()
    if tot <= 0:
        return False
    top = np.sort(h)[::-1][:slots].sum()
    return bool(top >= share * tot)


def moved(old: Sequence[Sequence[float]], new: Sequence[Sequence[float]], tol: float = MOVE_MIN) -> bool:
    """A material change of a window set: count changed or an endpoint moved
    by more than `tol` minutes (circular distance)."""
    if len(old or ()) != len(new or ()):
        return True
    for a, b in zip(sorted(old), sorted(new)):
        for x, y in zip(a[:2], b[:2]):
            d = abs(float(x) - float(y)) % DAY_MIN
            if min(d, DAY_MIN - d) > tol:
                return True
    return False


# ================================================================ fitting
def _cells_slots(hist: np.ndarray, cut: int, n: float) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """96 slot cells starting at the cut: (counts scaled to n, widths, starts
    in unwrapped minutes)."""
    h = np.asarray(hist, dtype=np.float64)
    tot = h.sum()
    k = cut // SLOT_MIN
    hs = np.roll(h, -k)
    counts = hs / tot * n if tot > 0 else np.zeros(SLOTS)
    widths = np.full(SLOTS, float(SLOT_MIN))
    starts = np.arange(SLOTS, dtype=np.float64) * SLOT_MIN
    return counts, widths, starts


def _cells_minutes(minutes: np.ndarray, cut: int, weights: Optional[np.ndarray] = None
                   ) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """1-minute cells for occupied minutes, one zero cell per empty run,
    covering [0, 1440) in unwrapped minutes; a cell counts the weights of its
    points (1 each without weights)."""
    u = np.floor((np.asarray(minutes, dtype=np.float64) - cut) % DAY_MIN).astype(np.int64)
    u = np.clip(u, 0, DAY_MIN - 1)
    occ, inv = np.unique(u, return_inverse=True)
    w = np.ones(u.size) if weights is None else np.asarray(weights, dtype=np.float64)
    cnt = np.bincount(inv, weights=w, minlength=occ.size)
    counts: List[float] = []
    widths: List[float] = []
    starts: List[float] = []
    pos = 0
    for m, c in zip(occ.tolist(), cnt.tolist()):
        if m > pos:
            counts.append(0.0)
            widths.append(float(m - pos))
            starts.append(float(pos))
        counts.append(float(c))
        widths.append(1.0)
        starts.append(float(m))
        pos = m + 1
    if pos < DAY_MIN:
        counts.append(0.0)
        widths.append(float(DAY_MIN - pos))
        starts.append(float(pos))
    return np.asarray(counts), np.asarray(widths), np.asarray(starts)


def _windows_from_blocks(blocks: List[Tuple[int, int]], counts: np.ndarray, widths: np.ndarray,
                         starts: np.ndarray, kappa: float, min_share: float
                         ) -> Tuple[List[Tuple[float, float, float]], Dict[str, Any]]:
    """[(start, end, mass share)] in unwrapped minutes."""
    tot = float(counts.sum())
    if tot <= 0 or not blocks:
        return [], {"blocks": len(blocks)}
    B = []
    for j, r in blocks:
        N = float(counts[j:r + 1].sum())
        T = float(widths[j:r + 1].sum())
        B.append((float(starts[j]), float(starts[r] + widths[r]), N, T, N / T if T > 0 else 0.0))
    mean_rate = tot / DAY_MIN
    # background = the time-weighted rate of the quietest blocks that together
    # cover Q_OFF of the day ("off" hours); a block is active when its rate is
    # >= KAPPA x that background and >= F_MEAN x the day's mean rate (the
    # floor keeps sparse noise out of windows)
    order = sorted(range(len(B)), key=lambda i: B[i][4])
    t_acc, m_acc = 0.0, 0.0
    for i in order:
        take = min(B[i][3], Q_OFF * DAY_MIN - t_acc)
        if take <= 0:
            break
        t_acc += take
        m_acc += B[i][4] * take
    r_off = m_acc / t_acc if t_acc > 0 else 0.0
    thr = max(F_MEAN * mean_rate, kappa * r_off)
    inw = [b[4] > 0 and b[4] >= thr for b in B]
    if all(inw) or not any(inw):
        return [(0.0, float(DAY_MIN), 1.0)], {"blocks": len(B), "all_day": True, "thr": thr}
    wins: List[List[float]] = []
    for (s, e, N, T, rate), ok in zip(B, inw):
        # adjacent active blocks join; so do windows separated by less than one
        # slot (below the resolution of the node's density model)
        if ok and wins and s - wins[-1][1] < MERGE_GAP:
            gap_mass = sum(b[2] for b in B if b[0] >= wins[-1][1] - 1e-9 and b[1] <= s + 1e-9)
            wins[-1][1] = e
            wins[-1][2] += N + gap_mass
        elif ok:
            wins.append([s, e, N])
    out = [(s, e, N / tot) for s, e, N in wins if N / tot >= min_share]
    info = {"blocks": len(B), "thr": thr}
    if not out:
        # flat day: nothing stands out of the background -> all day
        return [(0.0, float(DAY_MIN), 1.0)], dict(info, all_day=True)
    out.sort(key=lambda x: -x[2])
    return sorted(out[:W_MAX]), info


SNAP_SPACING = 3.0               # edge snapping: gaps <= 3 x the window's mean point spacing
TRIM_ALPHA = 0.01                # edge trimming: an end group behind a gap that a flat density
                                 # would produce with prob. < 1 % (mu ln(n / 0.01), mu = median gap / ln 2) ...
TRIM_MIN = 5.0                   # ... and > 5 min from the rest leaves the window,
TRIM_TAIL = 0.1                  # ... with at most 10 % of the window's points beyond it


def _snap(wins: List[Tuple[float, float, float]], u: np.ndarray, tot: float
          ) -> List[Tuple[float, float, float]]:
    """Minute mode: a sparse edge point is often split off into the adjacent
    empty block (a Bayesian-Blocks edge effect: its cell is merged with the
    empty run). Extend each window over neighbouring reservoir points whose
    gap to the window edge is <= SNAP_SPACING x the window's mean spacing
    (at least 1 minute), never into another window."""
    if not len(u) or not wins:
        return wins
    u = np.sort(np.floor(u))
    out = []
    for i, (s, e, sh) in enumerate(wins):
        lo_lim = wins[i - 1][1] if i > 0 else 0.0
        hi_lim = wins[i + 1][0] if i + 1 < len(wins) else float(DAY_MIN)
        inside = u[(u >= s) & (u < e)]
        if inside.size == 0:
            out.append((s, e, sh))
            continue
        added = 0
        # the reverse edge effect: a few sparse points far outside the core were
        # merged into the window's block -> trim them (a gap a flat density would
        # produce with prob. < 1 %, >= 5 min, <= 10 % of the window's points beyond)
        while inside.size >= 4:
            gaps = np.diff(inside)
            mu = max(float(np.median(gaps)), 0.5) / math.log(2.0)   # exponential spacing under a flat density
            gt = max(TRIM_MIN, mu * math.log(gaps.size / TRIM_ALPHA))
            big = np.flatnonzero(gaps > gt)
            k_max = max(1, int(TRIM_TAIL * inside.size))
            if big.size and inside.size - 1 - big[-1] <= k_max:        # a small group after the last big gap
                e = float(inside[big[-1]]) + 1.0
            elif big.size and big[0] + 1 <= k_max:                      # ... or before the first
                s = float(inside[big[0] + 1])
            else:
                break
            keep = inside[(inside >= s) & (inside < e)]
            added -= int(inside.size - keep.size)
            inside = keep
        g = max(1.0, SNAP_SPACING * (e - s) / inside.size)
        while True:
            nxt = u[(u >= e) & (u < hi_lim)]
            if nxt.size and nxt[0] - e < g:
                e = min(hi_lim, float(nxt[0]) + 1.0)
                added += int(np.sum(u == nxt[0]))
                continue
            prv = u[(u < s) & (u >= lo_lim)]
            if prv.size and s - (prv[-1] + 1.0) < g:
                s = max(lo_lim, float(prv[-1]))
                added += int(np.sum(u == prv[-1]))
                continue
            break
        out.append((s, e, sh + added / tot if tot > 0 else sh))
    return out


def _accepted(wins_u: List[Tuple[float, float, float]], pts: Sequence[Sequence[Any]],
              wts: np.ndarray, cut: int, tz_offset_s: float) -> List[Tuple[float, float, float]]:
    """The §6.9.2 acceptance rule applied to a window: at a node whose clean
    arrivals come from >= 2 sources, a window that only ONE source's arrivals
    support is that source's idiosyncrasy or anomaly until it persisted
    REGIME_SINGLE_DATES dates (the rule regime_cut applies to a single source's
    change). Pack O, A2 (192.168.1.21 logging in as rose at 09:10 on days
    17-19, flagged and held, its held rows released and learned at full
    weight): the 综合部 login node stated '08:32-08:51、09:10-09:11' (seeds 0,
    3). A node of one source (an approver, an exception) is not affected; a
    dropped window's arrivals count as outside the windows (coverage)."""
    if len(wins_u) <= 1 or not pts or len(pts[0]) < 4:
        return wins_u
    clean = [(float(p[0]), float(p[1]), str(p[3])) for p, w in zip(pts, wts) if w >= CLEAN_W and p[3]]
    if len({c[2] for c in clean}) < 2:
        return wins_u
    out = []
    for s, e, sh in wins_u:
        inside = [c for c in clean if s - 1e-9 <= (c[0] - cut) % DAY_MIN <= e + 1e-9]
        srcs = {c[2] for c in inside}
        if len(srcs) == 1 and len({_local_date(c[1], tz_offset_s) for c in inside}) < REGIME_SINGLE_DATES:
            continue
        out.append((s, e, sh))
    return out or wins_u


def _local_date(ts: float, tz_offset_s: float) -> int:
    return int((float(ts) + tz_offset_s) // 86400.0)


def fit_daytype(hist: np.ndarray, n: float, points: Optional[Sequence[Sequence[Any]]] = None,
                tz_offset_s: float = 0.0, kappa: float = KAPPA, min_share: float = MIN_SHARE,
                p0: float = P0, min_points: int = MIN_POINTS) -> Optional[Dict[str, Any]]:
    """Windows of one node and day type.

    hist    the node's hist96 for the day type (mass, any positive scale)
    n       evidence units behind it (H_m channel): the Bayesian-Blocks sample size
    points  [(minute, ts[, weight[, source]])] of the minute reservoir for this
            day type, or None; a weight (default 1, relative to a typical
            arrival) scales the point's count, so a row P04 learned with
            outlier damping (0.1) or low trust weighs that much

    Returns {'windows': [[s, e]], 'labels', 'shares', 'coverage', 'res':
    'minute'|'slot', 'n', 'n_points', 'dates', 'stability', 'blocks', 'cut',
    'all_day'} or None when there is no mass."""
    h = np.asarray(hist, dtype=np.float64)
    if h.size != SLOTS or h.sum() <= 0 or n <= 0:
        return None
    cut = unwrap_cut(h)
    pts = list(points or [])
    minute_mode = len(pts) >= min_points
    wts = np.asarray([float(p[2]) if len(p) > 2 else 1.0 for p in pts], dtype=np.float64)
    if minute_mode:
        mins = np.asarray([p[0] for p in pts], dtype=np.float64)
        counts, widths, starts = _cells_minutes(mins, cut, wts)
        ncp = ncp_prior(float(wts.sum()), p0)
    else:
        counts, widths, starts = _cells_slots(h, cut, n)
        ncp = ncp_prior(n, p0)
    blocks = bayesian_blocks(counts, widths, ncp)
    wins_u, info = _windows_from_blocks(blocks, counts, widths, starts, kappa, min_share)
    if minute_mode and not info.get("all_day"):
        # edge snapping / trimming on the clean arrivals (a damped outlier, weight
        # < 1/2 of a typical arrival, never extends a window)
        clean = wts >= CLEAN_W
        wins_u = _snap(wins_u, (mins[clean] - cut) % DAY_MIN, float(clean.sum()) or 1.0)
        wins_u = _accepted(wins_u, pts, wts, cut, tz_offset_s)
    windows: List[List[int]] = []
    for s, e, _ in wins_u:
        if info.get("all_day"):
            windows.append([0, DAY_MIN])
            continue
        a = int(round(s + cut)) % DAY_MIN
        b = int(round(e + cut)) % DAY_MIN
        if b == a:                                  # a window of the whole circle
            windows.append([0, DAY_MIN])
        else:
            windows.append([a, b])
    windows = sorted(windows)
    # coverage: reservoir fraction in minute mode (edges are minute-exact),
    # slot mass share in slot mode (edges are slot edges)
    if minute_mode:
        inside = np.asarray([in_windows(m, windows) for m in mins], dtype=np.float64)
        cov = float(np.sum(inside * wts) / max(float(wts.sum()), 1e-12))
        if not info.get("all_day"):
            cov = predictive_coverage(cov, float(np.sum(wts >= CLEAN_W)), len(windows))
    else:
        tot = h.sum()
        mids = np.arange(SLOTS) * SLOT_MIN + SLOT_MIN / 2.0
        cov = float(sum(h[i] for i in range(SLOTS) if in_windows(mids[i], windows)) / tot)
    dates, stab = None, None
    if pts:
        by_date: Dict[int, List[bool]] = {}
        for p, w in zip(pts, wts):
            if w < CLEAN_W:
                continue                        # damped outliers are not the pattern's days
            by_date.setdefault(_local_date(p[1], tz_offset_s), []).append(in_windows(p[0], windows))
        dates = len(by_date)
        stab = float(np.mean([all(v) for v in by_date.values()])) if by_date else None
    return {"windows": windows, "labels": [label(s, e) for s, e in windows],
            "shares": [round(float(x[2]), 4) for x in wins_u], "coverage": cov,
            "res": "minute" if minute_mode else "slot", "n": float(n), "n_points": len(pts),
            "w_points": round(float(wts.sum()), 3),
            "dates": dates, "stability": stab, "blocks": int(info.get("blocks", 0)),
            "cut": int(cut), "all_day": bool(info.get("all_day", False))}


def _ks_sf(lam: float) -> float:
    """Kolmogorov distribution survival function Q(lam) = 2 sum (-1)^(j-1) e^(-2 j^2 lam^2)."""
    if lam <= 0.2:
        return 1.0
    s, j = 0.0, 1
    while j <= 100:
        term = math.exp(-2.0 * j * j * lam * lam)
        s += term if j % 2 else -term
        if term < 1e-12:
            break
        j += 1
    return float(min(1.0, max(0.0, 2.0 * s)))


def _wks(ua: np.ndarray, wa: np.ndarray, ub: np.ndarray, wb: np.ndarray) -> Tuple[float, float]:
    """Weighted two-sample Kolmogorov-Smirnov (D, p), effective sample sizes
    (sum w)^2 / sum w^2 (Kish)."""
    grid = np.unique(np.concatenate([ua, ub]))
    oa, ob = np.argsort(ua), np.argsort(ub)
    ca = np.concatenate(([0.0], np.cumsum(wa[oa])))
    cb = np.concatenate(([0.0], np.cumsum(wb[ob])))
    Fa = ca[np.searchsorted(ua[oa], grid, side="right")] / max(ca[-1], 1e-12)
    Fb = cb[np.searchsorted(ub[ob], grid, side="right")] / max(cb[-1], 1e-12)
    D = float(np.max(np.abs(Fa - Fb))) if grid.size else 0.0
    na = float(wa.sum()) ** 2 / max(float((wa * wa).sum()), 1e-12)
    nb = float(wb.sum()) ** 2 / max(float((wb * wb).sum()), 1e-12)
    ne = na * nb / max(na + nb, 1e-12)
    return D, _ks_sf(D * math.sqrt(ne))


def regime_cut(points: Sequence[Sequence[Any]], hist: np.ndarray, tz_offset_s: float = 0.0,
               alpha: float = REGIME_ALPHA, min_dates: int = REGIME_DATES,
               min_points: int = REGIME_POINTS) -> Optional[Dict[str, Any]]:
    """The arrival-time change P09 owns (§6.9.2, §16.2 M8): the local date from
    which a node's arrivals of one day type follow a different time-of-day law.

    Candidates are the boundaries between consecutive local dates of the
    reservoir points with >= min_dates dates (and >= min_points points) on
    each side; each is tested by a weighted two-sample Kolmogorov-Smirnov test
    (minutes unwrapped at the quiet cut of `hist`), Bonferroni over the
    candidates. The most significant boundary with adjusted p <= alpha is the
    change. It is *accepted* (§6.9.2) when the later arrivals come from >= 2
    sources (a coordinated change: a department's new schedule) or persist
    for >= REGIME_SINGLE_DATES dates; otherwise it is provisional.
    points [(minute, ts, weight, source)]; returns {'since' (ts of the first
    local midnight of the new regime), 'p', 'dates_after', 'sources_after',
    'accepted'} or None."""
    pts = [p for p in points if len(p) >= 2]
    if len(pts) < 2 * min_points:
        return None
    cut = unwrap_cut(np.asarray(hist, dtype=np.float64))
    dates = np.asarray([_local_date(p[1], tz_offset_s) for p in pts], dtype=np.int64)
    ud = np.unique(dates)
    if ud.size < 2 * min_dates:
        return None
    u = np.asarray([(float(p[0]) - cut) % DAY_MIN for p in pts], dtype=np.float64)
    w = np.asarray([float(p[2]) if len(p) > 2 else 1.0 for p in pts], dtype=np.float64)
    cands = []
    for k in range(min_dates, ud.size - min_dates + 1):
        after = dates >= ud[k]
        if after.sum() < min_points or (~after).sum() < min_points:
            continue
        cands.append((k, after))
    if not cands:
        return None
    best = None
    for k, after in cands:
        D, p = _wks(u[after], w[after], u[~after], w[~after])
        pa = min(1.0, p * len(cands))
        if best is None or pa < best[0] or (pa == best[0] and D > best[2]):
            best = (pa, k, D, after)
    pa, k, D, after = best
    if pa > alpha:
        return None
    srcs = {str(p[3]) for p, a in zip(pts, after) if a and len(p) > 3}
    n_after = int(ud.size - k)
    return {"since": float(int(ud[k]) * 86400.0 - tz_offset_s), "p": float(pa), "D": float(D),
            "dates_after": n_after, "sources_after": len(srcs),
            "accepted": bool(len(srcs) >= 2 or n_after >= REGIME_SINGLE_DATES)}


def predictive_coverage(cov_in: float, n: float, n_windows: int) -> float:
    """The probability that the NEXT arrival falls inside the windows, from
    their in-sample coverage `cov_in` over n reservoir arrivals: in minute mode
    each window's two edges are snapped to arrivals (order statistics of the
    sample), so - the rank bound P06/P07 use for ranges and lengths - a new
    arrival lands beyond a window's edges with probability 2/(n+1) even when
    every sampled arrival is inside: P(inside) = 1 - (n (1 - cov_in) +
    2 n_windows) / (n + 1), never above cov_in.

    Measured on pack O (round 2): statements said '覆盖 100 %' while 5-20 % of
    the held-out arrivals of their truth windows fell outside the learned
    edges; 'when' was the constraint that failed most often in PG1 precision
    (24 of 36 failing statements at day 14, seed 0), and P04's hold records,
    which test the windows at the stated coverage, failed with it."""
    if n <= 0:
        return float(cov_in)
    out = n * (1.0 - float(cov_in)) + 2.0 * max(0, int(n_windows))
    return float(max(0.0, min(float(cov_in), 1.0 - out / (n + 1.0))))


def confidence(rec: Mapping[str, Any]) -> float:
    """Statement confidence of a when-constraint: coverage x day stability
    (§6.17.2); coverage alone when the stability is unknown (slot mode)."""
    c = float(rec.get("coverage") or 0.0)
    st = rec.get("stability")
    return c * (float(st) if st is not None else 1.0)


def entropy_gain_bits(hist: np.ndarray) -> float:
    """log2(96) - H(slot shares): bits per event the node's time density saves
    against a uniform day (the P12 full-information utility of the when facet)."""
    h = np.asarray(hist, dtype=np.float64)
    tot = h.sum()
    if tot <= 0:
        return 0.0
    p = h[h > 0] / tot
    return float(math.log2(SLOTS) + np.sum(p * np.log2(p)))


# ============================================================== rendering
def render_zh(dt: str, rec: Mapping[str, Any], n_dates: Optional[int] = None) -> str:
    """'工作日 09:00–09:21（覆盖 97 %，21 个工作日）'."""
    name = {"wd": "工作日", "nwd": "非工作日"}.get(dt, dt)
    unit = {"wd": "个工作日", "nwd": "个非工作日"}.get(dt, "天")
    if rec.get("all_day"):
        span = "全天"
    else:
        span = "、".join(f"{hhmm(s)}–{hhmm(e)}" for s, e in rec.get("windows") or [])
    cov = 100.0 * float(rec.get("coverage") or 0.0)
    d = rec.get("dates") if rec.get("dates") is not None else n_dates
    tail = f"，{int(d)} {unit}" if d else ""
    return f"{name} {span}（覆盖 {cov:.0f} %{tail}）"


def render_en(dt: str, rec: Mapping[str, Any], n_dates: Optional[int] = None) -> str:
    name = {"wd": "workdays", "nwd": "non-workdays"}.get(dt, dt)
    if rec.get("all_day"):
        span = "all day"
    else:
        span = ", ".join(f"{hhmm(s)}–{hhmm(e)}" for s, e in rec.get("windows") or [])
    cov = 100.0 * float(rec.get("coverage") or 0.0)
    d = rec.get("dates") if rec.get("dates") is not None else n_dates
    tail = f", {int(d)} dates" if d else ""
    return f"{name} {span} (coverage {cov:.0f} %{tail})"


def tz_offset(config: Optional[Mapping[str, Any]], at: Optional[float] = None) -> float:
    """UTC offset (s) of the configured tz at `at` (default: mid-January 2026)."""
    try:
        from zoneinfo import ZoneInfo
        tz = ZoneInfo((config or {}).get("tz") or "Asia/Shanghai")
        when = (_dt.datetime.fromtimestamp(float(at), tz) if at is not None
                else _dt.datetime(2026, 1, 15, tzinfo=tz))
        off = when.utcoffset()
        return float(off.total_seconds()) if off is not None else 0.0
    except Exception:
        return 8 * 3600.0


# ============================================================== accessors
def lookup(model: Any, kind: int, nid: int) -> Optional[Dict[str, Any]]:
    """The fitted entry of a node in model.pwin (P03 / P10 / P14 accessor)."""
    if not isinstance(model, Mapping):
        return None
    nodes = model.get("nodes") or {}
    sub = nodes.get(kind, nodes.get(str(kind)))
    if not isinstance(sub, Mapping):
        return None
    return sub.get(nid, sub.get(str(nid)))


def part_when(when: Any, members: Iterable[str], tz_offset_s: float = 0.0,
              min_points: int = MIN_POINTS) -> Optional[Dict[str, Any]]:
    """The statement-contract `when` block of ONE learned group's part of a
    node, fitted like the node's own windows (fit_daytype, minute mode) on the
    node's minute-reservoir arrivals whose source is a member of the group;
    None when the node keeps no reservoir or the part has fewer than
    `min_points` arrivals on every day type (the view then states the node's
    windows). Integration addition (2026-09-30): on a login route shared by
    综合部 (09:00-09:21), 财务部 (09:05-09:30) and 销售部 (08:30-09:30) the
    node's union window matched no department's truth (IoU 0.17-0.42)."""
    res = getattr(when, "res", None)
    if res is None or not len(res):
        return None
    mem = {str(m) for m in members}
    by: Dict[str, Optional[Dict[str, Any]]] = {}
    for d, dk in enumerate(DAYTYPES):
        pts = [(float(it[1]), float(t)) for it, _w, t in res.items()
               if len(it) > 2 and int(it[0]) == d and str(it[2]) in mem]
        if len(pts) < min_points:
            by[dk] = None
            continue
        h = np.zeros(SLOTS)
        for m, _ in pts:
            h[int(m // 15) % SLOTS] += 1.0
        rec = fit_daytype(h, float(len(pts)), pts, tz_offset_s=tz_offset_s)
        if rec is not None:
            rec["confidence"] = confidence(rec)
            rec["text_zh"] = render_zh(dk, rec)
            rec["text_en"] = render_en(dk, rec)
        by[dk] = rec
    fitted = [r for r in by.values() if r]
    if not fitted:
        return None
    out: Dict[str, Any] = {DT_LONG[dk]: (as_intervals(by[dk]["windows"]) if by.get(dk) else [])
                           for dk in DAYTYPES}
    w = [len(r.get("windows") or []) and float(r.get("n_points") or r.get("n") or 1.0) for r in fitted]
    tot = sum(w) or 1.0
    out["coverage"] = float(sum(wi * float(r["coverage"]) for wi, r in zip(w, fitted)) / tot)
    out["confidence"] = float(min(r["confidence"] for r in fitted))
    out["part"] = True
    out["n_points"] = int(sum(float(r.get("n_points") or 0) for r in fitted))
    out["by_daytype"] = by
    return out
