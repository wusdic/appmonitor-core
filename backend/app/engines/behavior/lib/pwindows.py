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
import functools
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

DAY_MIN = 1440
SLOTS = 96
SLOT_MIN = 15
P0 = 0.05
KAPPA = 3.0
MIN_SHARE = 0.05
FLAT_SHARE = 0.5                 # windows holding less than half of the arrivals describe no time-of-day law
Q_OFF = 0.15                     # the quietest 15 % of the day is the background ...
F_MEAN = 0.5                     # ... and an active block has >= half the mean rate
MERGE_GAP = 15.0                 # windows closer than one slot are one window
MIN_POINTS = 10                  # reservoir points of a day type needed for minute mode
CLEAN_W = 0.5                    # a point weighing < 1/2 of a typical arrival is a damped outlier
REGIME_DATES = 3                 # §6.9.2: a time window change persists >= 3 workdays ...
REGIME_ALPHA = 1e-3              # ... and differs from the earlier arrivals at this level
REGIME_SINGLE_DATES = 5          # a change shown by one source only persists >= 5 dates
STALE_ALPHA = 0.05               # _persistence: a not currently supported segment is stale at this level
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
                         starts: np.ndarray, kappa: float, min_share: float,
                         accept: Optional[Any] = None
                         ) -> Tuple[List[Tuple[float, float, float]], Dict[str, Any]]:
    """[(start, end, mass share)] in unwrapped minutes. `accept(s, e)`: an
    active block must also pass it (current persistence, _persists) before
    adjacent blocks join into windows."""
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
    if accept is not None:
        acc = [ok and accept(b[0], b[1]) for b, ok in zip(B, inw)]
        if any(acc):
            inw = acc
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
    if not out or sum(x[2] for x in out) < FLAT_SHARE:
        # flat day: nothing stands out of the background, or what stands out
        # holds a minority of the arrivals (round 4: pack O seed 1, the 60-s
        # health monitors of OA and finance on days 13-14 stated a 10-minute
        # workday window holding 5 % of their arrivals - a chance cluster of 8
        # of 166 reservoir minutes three times the background rate) -> all day
        return [(0.0, float(DAY_MIN), 1.0)], dict(info, all_day=True)
    out.sort(key=lambda x: -x[2])
    return sorted(out[:W_MAX]), info


SESSION_GAP_MIN = 30.0           # _sessions: a source's arrivals > 30 min apart are separate sessions


def _sessions(pts: Sequence[Sequence[Any]], tz_offset_s: float) -> List[Any]:
    """(round 6) The independent unit of each arrival for _extend: its
    source's session - the source's arrivals of one local date split where
    they are more than SESSION_GAP_MIN apart (the usual inactivity rule).
    An arrival without a source is a unit of its own."""
    keys: List[Any] = [None] * len(pts)
    grp: Dict[Tuple[int, str], List[int]] = {}
    for i, p in enumerate(pts):
        if len(p) > 3 and p[3]:
            grp.setdefault((_local_date(float(p[1]), tz_offset_s), str(p[3])), []).append(i)
        else:
            keys[i] = ("pt", i)
    for k, idx in grp.items():
        idx.sort(key=lambda j: float(pts[j][1]))
        sess, last = 0, None
        for j in idx:
            ts = float(pts[j][1])
            if last is not None and ts - last > SESSION_GAP_MIN * 60.0:
                sess += 1
            keys[j] = (k[0], k[1], sess)
            last = ts
    return keys


def _extend(wins: List[Tuple[float, float, float]], u: np.ndarray, units: Sequence[Any]
            ) -> List[Tuple[float, float, float]]:
    """(round 6) The edges of a minute-mode window are its extreme arrivals,
    which lie INSIDE the true window: k independent arrivals of a block of
    constant rate (Bayesian Blocks' model) leave on average 1 / (k + 1) of the
    window beyond each edge. The unbiased (UMVU) estimate of a uniform law's
    support extends each extreme by R / (k - 1) (R = the range of the window's
    clean arrivals). k counts the INDEPENDENT units - a source's session
    (_sessions): the page views of one session share its start, so a
    session is one draw of the arrival law (pack O seed 0, finance approvals 10:00-11:30: 30 views
    of ~10 sessions spanned 10:17-11:28; extended by R / 29 the window still
    missed 10:00-10:15, by R / 9 it does not). Edges are rounded to whole
    minutes and never reach a neighbouring window; with many units the
    extension is below half a minute and the window does not change.
    The stated coverage keeps the hull's rank bound (predictive_coverage):
    the extension only adds room, so the bound stays a valid lower bound
    (a coverage raised by the extension's expected edge mass over-stated the
    held share of clustered arrivals: pack O seed 0, nominal 0.92 -> 0.95,
    'when' checks passing 0.919 -> 0.905).
    Why (pack O seeds 3-4, D1): after 综合部's login moved to 08:30-08:51 the
    window was fitted from the first 9 arrivals of the new law (3 sources x 3
    workdays) and read 08:38-08:51 / 08:31-08:44 (IoU 0.62) on day 16: the
    hull of 9 uniform points misses 20 % of the window on average and IoU <
    0.7 one time in five; extended, one time in seventeen (tests/lib)."""
    if not wins or not len(u):
        return wins
    u = np.floor(np.asarray(u, dtype=np.float64))
    unit_arr = list(units)
    order = np.argsort(u, kind="stable")
    u = u[order]
    unit_arr = [unit_arr[int(i)] for i in order]
    out: List[Tuple[float, float, float]] = []
    for i, (s, e, sh) in enumerate(wins):
        lo_lim = out[-1][1] if out else 0.0
        hi_lim = wins[i + 1][0] if i + 1 < len(wins) else float(DAY_MIN)
        m = (u >= s) & (u < e)
        inside = u[m]
        k = len({unit_arr[j] for j in np.flatnonzero(m)})
        if inside.size < 2 or k < 2:
            out.append((s, e, sh))
            continue
        g = float(inside[-1] - inside[0]) / (k - 1.0)
        s2 = float(max(lo_lim, min(s, math.floor(s - g + 0.5))))
        e2 = float(min(hi_lim, max(e, math.floor(e + g + 0.5))))
        out.append((s2, e2, sh))
    return out


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


def _persistence(pts: Sequence[Sequence[Any]], wts: np.ndarray, cut: int, tz_offset_s: float) -> Optional[Any]:
    """The §6.9.2 acceptance rule for a segment of the day [s, e] (unwrapped
    minutes) at a node whose clean arrivals come from >= 2 sources (None - no
    constraint - at a node of one source: an approver, an exception):
      * a segment only ONE source's arrivals support is that source's
        idiosyncrasy or anomaly until it persisted REGIME_SINGLE_DATES dates
        (round 3, R4);
      * a segment is STALE when its support stopped: over the node's last
        REGIME_SINGLE_DATES + 1 dates (of the day type; the latest may be
        today, still partial) it is not currently supported (>= 2 sources on
        >= REGIME_DATES of those dates, or arrivals on >= REGIME_SINGLE_DATES of
        them) AND its source-dates (a source arriving in it on a date) there
        are significantly fewer than its rate on the earlier dates predicts
        (Poisson lower tail <= STALE_ALPHA).
    Applied to the active Bayesian blocks before they join into windows and to
    the windows after snapping (_accepted). Why (pack O): seed 3, day 21,
    '09:09-09:14' = the three members' pre-D1 logins (two dates - too few for
    regime_cut) + A2's 09:10 logins of 192.168.1.21 (three workdays, released
    undamped): one source on three of the last six dates, 3 source-dates
    against 12 expected, p = 0.002 (IoU 0.69 before). Seed 0 (round 4 run 1):
    the pre-D1 block 09:03-09:21 joined the D1 block 08:32-08:50 across a
    13-minute gap (< MERGE_GAP): '08:32-09:21' from day 17 to 21 (checklist
    IoU 0.37). A sparse segment of a broad law (2 sources, < 1 source-date a
    date: finance's ledger) is never significantly stale."""
    if not pts or len(pts[0]) < 4:
        return None
    clean = [(float(p[0]), _local_date(float(p[1]), tz_offset_s), str(p[3]))
             for p, w in zip(pts, wts) if w >= CLEAN_W and p[3]]
    if len({c[2] for c in clean}) < 2:
        return None
    dates = sorted({c[1] for c in clean})
    recent = set(dates[-(REGIME_SINGLE_DATES + 1):])
    n_past = len(dates) - len(recent)
    rc = [((c[0] - cut) % DAY_MIN, c[1], c[2]) for c in clean]

    def persists(s: float, e: float) -> bool:
        inside = [c for c in rc if s - 1e-9 <= c[0] <= e + 1e-9]
        srcs = {c[2] for c in inside}
        if len(srcs) == 1 and len({c[1] for c in inside}) < REGIME_SINGLE_DATES:
            return False
        if n_past <= 0:
            return True
        rec_in = [c for c in inside if c[1] in recent]
        r_src = {c[2] for c in rec_in}
        r_days = {c[1] for c in rec_in}
        if (len(r_src) >= 2 and len(r_days) >= REGIME_DATES) or len(r_days) >= REGIME_SINGLE_DATES:
            return True                         # currently supported (the R4 rule on the recent dates)
        sd = {(c[1], c[2]) for c in inside}
        k_rec = sum(1 for d, _ in sd if d in recent)
        lam = (len(sd) - k_rec) / float(n_past) * len(recent)
        return _poisson_cdf(k_rec, lam) > STALE_ALPHA
    return persists


def _poisson_cdf(k: int, lam: float) -> float:
    if lam <= 0:
        return 1.0
    term = math.exp(-lam)
    acc = term
    for j in range(1, int(k) + 1):
        term *= lam / j
        acc += term
    return float(min(1.0, acc))


def _accepted(wins_u: List[Tuple[float, float, float]], pts: Sequence[Sequence[Any]],
              wts: np.ndarray, cut: int, tz_offset_s: float) -> List[Tuple[float, float, float]]:
    """Windows that persist (_persistence); all of them when none does. A
    dropped window's arrivals count as outside the windows (coverage)."""
    if len(wins_u) <= 1:
        return wins_u
    ok = _persistence(pts, wts, cut, tz_offset_s)
    if ok is None:
        return wins_u
    out = [w for w in wins_u if ok(w[0], w[1])]
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
    cw = wts
    if minute_mode:
        mins = np.asarray([p[0] for p in pts], dtype=np.float64)
        counts, widths, starts = _cells_minutes(mins, cut, cw)
        ncp = ncp_prior(float(cw.sum()), p0)
    else:
        counts, widths, starts = _cells_slots(h, cut, n)
        ncp = ncp_prior(n, p0)
    blocks = bayesian_blocks(counts, widths, ncp)
    wins_u, info = _windows_from_blocks(blocks, counts, widths, starts, kappa, min_share,
                                        _persistence(pts, wts, cut, tz_offset_s) if minute_mode else None)
    if minute_mode and not info.get("all_day"):
        # edge snapping / trimming on the clean arrivals (a damped outlier, weight
        # < 1/2 of a typical arrival, never extends a window)
        clean = wts >= CLEAN_W
        wins_u = _snap(wins_u, (mins[clean] - cut) % DAY_MIN, float(clean.sum()) or 1.0)
        wins_u = _accepted(wins_u, pts, wts, cut, tz_offset_s)
        wins_u = _extend(wins_u, (mins[clean] - cut) % DAY_MIN,
                         _sessions([p for p, ok in zip(pts, clean) if ok], tz_offset_s))
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
        cov = float(np.sum(inside * cw) / max(float(cw.sum()), 1e-12))
        if not info.get("all_day"):
            cov = predictive_coverage(cov, float(np.sum(wts >= CLEAN_W)), len(windows))
    else:
        tot = h.sum()
        mids = np.arange(SLOTS) * SLOT_MIN + SLOT_MIN / 2.0
        cov = float(sum(h[i] for i in range(SLOTS) if in_windows(mids[i], windows)) / tot)
        if not info.get("all_day"):
            # (round 6) the slot share is an in-sample share like the minute
            # mode's: the windows were chosen to hold the sample, so the next
            # arrival falls beyond them with probability up to the rank bound
            # (O-real seed 0, crm-02 days 7-14: '09:00-18:00' stated coverage
            # 1.00 on n ~ 100 arrivals, held 0.96-0.98 - every check failed)
            cov = predictive_coverage(cov, float(n), len(windows))
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
            "w_points": round(float(cw.sum()), 3),
            "dates": dates, "stability": stab, "blocks": int(info.get("blocks", 0)),
            "cut": int(cut), "all_day": bool(info.get("all_day", False))}


FWD_PRIOR = 2.0                  # out-of-sample coverage: Beta prior strength centred on the in-sample coverage


CV_FOLDS = 5                     # date folds of the out-of-sample coverage


def cv_coverage(hist: np.ndarray, pts: Sequence[Sequence[Any]], tz_offset_s: float = 0.0,
                folds: int = CV_FOLDS) -> Optional[Tuple[float, float]]:
    """Out-of-sample coverage of the windows a node's arrivals give: blocked
    cross-validation by DATE - the dates are dealt into min(folds, #dates)
    folds; the windows fitted on the other folds' arrivals are checked on the
    clean arrivals of each fold (the arrivals of one date stay together: they
    are not independent). Returns (weighted hits, weighted tests) or None
    (fewer than 3 dates, slot mode, an all-day fit, no fold with >= MIN_POINTS
    training points). The training fits saw fewer arrivals than the stated
    windows: their edges (order statistics, the rank bound of
    predictive_coverage) leave 2 / (n + 1) per window outside where the full
    fit leaves 2 / (N + 1); the difference is credited, so the estimate
    measures the stated windows, not smaller ones.
    Why (pack O, round 3, day 14, seeds 0-1): 'when' was the constraint that
    failed most often in PG1 precision (13 / 11 constraints); besides drift,
    the in-sample coverage of minute-exact windows overstated how many fresh
    arrivals fall inside: finance ledger windows stated 0.94 held 0.80, the
    approval list 0.88 / 0.77, mail department parts 0.86-0.87 / 0.63-0.71
    (the edges and gaps are chosen on the very arrivals they are scored on).
    (A forward split - train on the older two thirds of the dates - was
    tried first: on a department's part of the mail node, ~20 arrivals, it
    halved the stated coverage, 0.86 -> 0.47, against ~0.68 held: a third of
    the dates missing from training is too pessimistic for small samples.)"""
    rows = [p for p in pts if len(p) >= 2]
    if not rows:
        return None
    wts = [float(p[2]) if len(p) > 2 else 1.0 for p in rows]
    day = [_local_date(float(p[1]), tz_offset_s) for p in rows]
    dates = sorted({d for d, w in zip(day, wts) if w >= CLEAN_W})
    if len(dates) < 3:
        return None
    K = min(int(folds), len(dates))
    fold_of = {d: i % K for i, d in enumerate(dates)}
    n_all = float(np.sum(wts))
    hits, tot = 0.0, 0.0
    for j in range(K):
        train = [p for p, d in zip(rows, day) if fold_of.get(d, -1) != j]
        test = [(float(p[0]), w) for p, w, d in zip(rows, wts, day) if w >= CLEAN_W and fold_of.get(d, -1) == j]
        if len(train) < MIN_POINTS or not test:
            continue
        r2 = fit_daytype(hist, float(len(train)), train, tz_offset_s=tz_offset_s)
        if r2 is None or r2.get("res") != "minute":
            continue
        t_ = float(sum(w for _, w in test))
        h_ = float(sum(w for m, w in test if in_windows(m, r2["windows"])))
        n_tr = float(np.sum([float(p[2]) if len(p) > 2 else 1.0 for p in train]))
        edge = 2.0 * len(r2["windows"]) * (1.0 / (n_tr + 1.0) - 1.0 / (n_all + 1.0))
        hits += min(t_, h_ + max(0.0, edge) * t_)
        tot += t_
    if tot <= 0:
        return None
    return hits, tot


def fit_regime(hist: np.ndarray, n: float, pts: Sequence[Sequence[Any]], tz_offset_s: float = 0.0,
               since0: Optional[float] = None, provisional0: bool = False,
               young: bool = True) -> Optional[Dict[str, Any]]:
    """Windows of one node (or one group's part of it) and day type from its
    arrivals [(minute, ts, weight, source)]: P09's pipeline shared by the
    node fit (behavior.time_window) and part_when.
      1 the current regime: `since0` (a P04 alarm / an accepted change) or the
        latest significant change of the time-of-day law (regime_cut); its
        arrivals replace the full set when there are enough of them - and,
        when they come from an alarm rather than a tested change, they span
        >= REGIME_DATES dates (a time-of-day law is a statement about days:
        the 60-s health monitors of OA and finance, pack O seed 1, day 14,
        stated a 23-minute window of 38 % coverage fitted on the minutes since
        an alarm);
      2 fit_daytype (Bayesian Blocks, snapping, current-persistence acceptance);
      3 coverage: the in-sample predictive coverage, lowered to the date
        cross-validated coverage (cv_coverage; the posterior mean under a Beta
        prior of strength FWD_PRIOR centred on the in-sample value) when that
        is smaller: the out-of-sample check reveals windows that hold less than
        their own sample says (edges and gaps chosen on it, a drifting law),
        never more than the rank bound of the in-sample value;
      4 (round 4) a change younger than REGIME_DATES dates - found by
        regime_cut with one date allowed after the boundary, or, when its
        arrivals are too few for the KS test, by testing the established law
        (fitted without the node's last REGIME_DATES - 1 dates) on the young
        dates (recent_drift) - is not fitted: the windows of the established
        law are stated, marked 'drift' and provisional, at the confidence the
        young arrivals show; regime_cut fits the new law once it has its
        dates. `young` False: no step 4 (the recursive established fit)."""
    pts = list(pts)
    since, provisional, cut = since0, bool(provisional0), None
    young_cut = None
    if since is None:
        cut = regime_cut(pts, hist, tz_offset_s=tz_offset_s, min_after=1 if young else REGIME_DATES)
        if cut is not None and cut["dates_after"] < REGIME_DATES:
            young_cut, cut = cut, None          # 4 below: a change too young to fit
        elif cut is not None:
            since, provisional = cut["since"], not cut["accepted"]
    if young_cut is not None:
        est = [p for p in pts if float(p[1]) < young_cut["since"]]
        rec = fit_regime(hist, n, est, tz_offset_s=tz_offset_s, young=False)
        if rec is not None:
            dr = recent_drift(rec, [p for p in pts if float(p[1]) >= young_cut["since"]], tz_offset_s,
                              alpha=1.0)
            if dr is not None:
                rec["drift"] = dict(dr, change={k: young_cut[k] for k in ("p", "dates_after", "sources_after")})
                rec["provisional"] = True
            return rec
    recent = [p for p in pts if float(p[1]) >= since] if since is not None else pts
    need = REGIME_POINTS if cut is not None else MIN_POINTS
    spans = cut is not None or len({_local_date(float(p[1]), tz_offset_s) for p in recent}) >= REGIME_DATES
    use_recent = since is not None and len(recent) >= need and spans
    use = recent if use_recent else pts
    rec = fit_daytype(hist, max(float(n), float(len(pts))), use, tz_offset_s=tz_offset_s,
                      min_points=min(need, MIN_POINTS))
    if rec is None:
        return None
    if since is not None:
        rec["since"] = float(since)
        rec["provisional"] = bool(provisional)
        rec["regime"] = "new" if use_recent else "mixed"
        if cut is not None:
            rec["change"] = {k: cut[k] for k in ("p", "dates_after", "sources_after", "accepted")}
    if rec.get("res") == "minute" and not rec.get("all_day"):
        fc = cv_coverage(hist, use, tz_offset_s)
        if fc is not None:
            h, m = fc
            c_in = float(rec["coverage"])
            rec["coverage_in"] = c_in
            rec["cv"] = [round(h, 3), round(m, 3)]
            rec["coverage"] = float(min(c_in, (h + FWD_PRIOR * c_in) / (m + FWD_PRIOR)))
            if rec["coverage"] < FLAT_SHARE:
                _flat(rec, use, tz_offset_s)
    if young and since0 is None:
        # 4b a change whose arrivals are too few for the KS test (3 logins on
        # one date): the established law (the dates before the young ones)
        # tested on the young dates
        yd = _young_dates(use, tz_offset_s)
        if yd:
            est_pts = [p for p in use if _local_date(float(p[1]), tz_offset_s) < yd]
            est = fit_regime(hist, n, est_pts, tz_offset_s=tz_offset_s, young=False)
            if est is not None:
                dr = recent_drift(est, [p for p in use if _local_date(float(p[1]), tz_offset_s) >= yd],
                                  tz_offset_s)
                if dr is not None:
                    est["drift"] = dr
                    est["provisional"] = True
                    return est
    return rec


def _flat(rec: Dict[str, Any], pts: Sequence[Sequence[Any]], tz_offset_s: float) -> None:
    """(round 6) fit_daytype's flat-day rule applied to the out-of-sample
    coverage: windows that hold less than FLAT_SHARE of the arrivals of dates
    they were not fitted on describe no time-of-day law (their edges and gaps
    follow the sample), so the day type is stated as one all-day window, as
    fit_daytype does when the in-sample windows hold a minority. O-real seed
    0, days 22-23: portal's non-workday POST /login (a normal law over
    07:00-23:00, sampled 1 in 4) stated '20:01-20:59' at coverage 0.20 and
    held 0.01-0.03 of the held-out arrivals."""
    rec["windows_fitted"] = rec.get("windows")
    rec["windows"] = [[0, DAY_MIN]]
    rec["labels"] = [label(0, DAY_MIN)]
    rec["shares"] = [1.0]
    rec["coverage"] = 1.0
    rec["all_day"] = True
    rec["flat_cv"] = True
    by: Dict[int, bool] = {}
    for p in pts:
        if (float(p[2]) if len(p) > 2 else 1.0) >= CLEAN_W:
            by[_local_date(float(p[1]), tz_offset_s)] = True
    rec["stability"] = 1.0 if by else rec.get("stability")


DRIFT_ALPHA = 0.01               # recent_drift: binomial lower tail of a date's in-window arrivals (Bonferroni)


def _young_dates(pts: Sequence[Sequence[Any]], tz_offset_s: float) -> Optional[int]:
    """The first of the node's last REGIME_DATES - 1 local dates (the dates a
    change regime_cut could not yet fit falls on) when the node has >= 2
    REGIME_DATES dates, else None."""
    dates = sorted({_local_date(float(p[1]), tz_offset_s) for p in pts})
    if len(dates) < 2 * REGIME_DATES:
        return None
    return int(dates[-(REGIME_DATES - 1)])


def _binom_cdf(k: int, n: int, p: float) -> float:
    p = min(1.0, max(0.0, float(p)))
    return float(min(1.0, sum(math.comb(n, j) * p ** j * (1.0 - p) ** (n - j) for j in range(int(k) + 1))))


def recent_drift(rec: Mapping[str, Any], young: Sequence[Sequence[Any]],
                 tz_offset_s: float = 0.0, alpha: float = DRIFT_ALPHA) -> Optional[Dict[str, Any]]:
    """Do the windows `rec` (fitted on the established dates) hold on the
    `young` arrivals (the dates after them, < REGIME_DATES of them)? Per young
    date, the number k of clean arrivals inside the windows is tested against
    Binomial(n, coverage) - the windows' own claim; a date whose lower tail
    p (x the dates tested) is <= alpha, with its outside arrivals from >= 2
    sources (one source: an idiosyncrasy or an anomaly, the persistence
    rule's business - unless every arrival has one source), shows the windows
    no longer hold (alpha 1: regime_cut already found the change). Returns
    {'date', 'p', 'k', 'n', 'sources', 'coverage'}: 'coverage' is the
    posterior mean (k + FWD_PRIOR c) / (n + FWD_PRIOR) of the young arrivals,
    the windows' stated confidence meanwhile (confidence()).
    Why (pack O, round 4): D1 moved 综合部's logins from 09:00-09:21 to
    08:30-08:51 on day 12; at the day-14 snapshot the login and home
    statements read '08:49-09:21' (the young date's block joined the old one)
    at 0.77-0.85 and held 0.08-0.13 (seed 0: 4 of 13 failing statements); at
    day 15 regime_cut placed the change one date early (it needs 3 dates after
    a boundary) and the parts read the union 08:32-09:21 until day 17."""
    if rec.get("res") != "minute" or rec.get("all_day") or not rec.get("windows"):
        return None
    c = float(rec.get("coverage") or 0.0)
    if not 0.0 < c < 1.0:
        return None
    clean = [(float(p[0]), _local_date(float(p[1]), tz_offset_s), str(p[3]) if len(p) > 3 else "")
             for p in young if (float(p[2]) if len(p) > 2 else 1.0) >= CLEAN_W]
    if not clean:
        return None
    dates = sorted({x[1] for x in clean})
    single = len({x[2] for x in clean}) <= 1
    wins = rec["windows"]
    hit = None
    for d in dates:
        day = [x for x in clean if x[1] == d]
        k = sum(1 for x in day if in_windows(x[0], wins))
        out_src = {x[2] for x in day if not in_windows(x[0], wins)}
        p = min(1.0, _binom_cdf(k, len(day), c) * len(dates))
        if p <= alpha and out_src and (len(out_src) >= 2 or single):
            hit = (d, p, len(out_src))
            break
    if hit is None:
        return None
    since = [x for x in clean if x[1] >= hit[0]]
    k = sum(1 for x in since if in_windows(x[0], wins))
    n = len(since)
    return {"date": int(hit[0]), "p": float(hit[1]), "k": int(k), "n": int(n), "sources": int(hit[2]),
            "coverage": float((k + FWD_PRIOR * c) / (n + FWD_PRIOR))}


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


KS_EXACT_MN = 4096               # _wks: exact two-sample KS law when m x n <= this (small samples)


@functools.lru_cache(maxsize=4096)
def _ks2_exact_sf(k: int, m: int, n: int) -> float:
    """Exact two-sided two-sample KS tail P(D_{m,n} >= k / (m n)) under H0
    (Hodges 1957: lattice paths from (0, 0) to (m, n) that stay strictly
    inside |i/m - j/n| < d, all C(m + n, m) paths equally likely). D of two
    samples of sizes m, n is a multiple of 1 / (m n): k = round(D m n)."""
    m, n = int(m), int(n)
    if m <= 0 or n <= 0 or k <= 0:
        return 1.0
    prev: List[int] = []
    for i in range(m + 1):
        cur = [0] * (n + 1)
        for j in range(n + 1):
            if abs(i * n - j * m) >= k:                  # |i/m - j/n| >= d (integer test)
                continue
            if i == 0 and j == 0:
                cur[j] = 1
                continue
            cur[j] = (prev[j] if i > 0 else 0) + (cur[j - 1] if j > 0 else 0)
        prev = cur
    return float(min(1.0, max(0.0, 1.0 - prev[n] / math.comb(m + n, m))))


def _wks(ua: np.ndarray, wa: np.ndarray, ub: np.ndarray, wb: np.ndarray) -> Tuple[float, float]:
    """Weighted two-sample Kolmogorov-Smirnov (D, p), effective sample sizes
    (sum w)^2 / sum w^2 (Kish).
    Round 6: small samples use the EXACT law of D (_ks2_exact_sf, on the
    effective sizes rounded) instead of Kolmogorov's limit, which is
    conservative there: 9 established against 6 new arrivals, completely
    separated (D = 1), read p = 0.0015 > REGIME_ALPHA from the limit against
    the exact 2 / C(15, 6) = 0.0004 - pack O seed 3, day 15: 综合部's login
    part after D1 (two new workdays) was not cut at the change, and its
    window stayed the union of both laws, 08:38-09:20, for two more days."""
    grid = np.unique(np.concatenate([ua, ub]))
    oa, ob = np.argsort(ua), np.argsort(ub)
    ca = np.concatenate(([0.0], np.cumsum(wa[oa])))
    cb = np.concatenate(([0.0], np.cumsum(wb[ob])))
    Fa = ca[np.searchsorted(ua[oa], grid, side="right")] / max(ca[-1], 1e-12)
    Fb = cb[np.searchsorted(ub[ob], grid, side="right")] / max(cb[-1], 1e-12)
    D = float(np.max(np.abs(Fa - Fb))) if grid.size else 0.0
    na = float(wa.sum()) ** 2 / max(float((wa * wa).sum()), 1e-12)
    nb = float(wb.sum()) ** 2 / max(float((wb * wb).sum()), 1e-12)
    m, n = int(round(na)), int(round(nb))
    if 0 < m * n <= KS_EXACT_MN:
        # the D of the effective sizes' lattice at or below the observed D
        return D, _ks2_exact_sf(int(math.floor(D * m * n + 1e-6)), m, n)
    ne = na * nb / max(na + nb, 1e-12)
    return D, _ks_sf(D * math.sqrt(ne))


def regime_cut(points: Sequence[Sequence[Any]], hist: np.ndarray, tz_offset_s: float = 0.0,
               alpha: float = REGIME_ALPHA, min_dates: int = REGIME_DATES,
               min_points: int = REGIME_POINTS, min_after: Optional[int] = None) -> Optional[Dict[str, Any]]:
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
    'accepted'} or None. min_after (default min_dates): fewer dates allowed
    after a boundary - fit_regime passes 1 to place a young change on its
    real first date (with 3 dates required after it, a change two dates old
    was placed one date early: pack O seed 0, day 15, the union 08:32-09:21)."""
    pts = [p for p in points if len(p) >= 2]
    if len(pts) < 2 * min_points:
        return None
    cut = unwrap_cut(np.asarray(hist, dtype=np.float64))
    dates = np.asarray([_local_date(p[1], tz_offset_s) for p in pts], dtype=np.int64)
    ud = np.unique(dates)
    m_after = min_dates if min_after is None else max(1, int(min_after))
    if ud.size < min_dates + m_after:
        return None
    u = np.asarray([(float(p[0]) - cut) % DAY_MIN for p in pts], dtype=np.float64)
    w = np.asarray([float(p[2]) if len(p) > 2 else 1.0 for p in pts], dtype=np.float64)
    cands = []
    for k in range(min_dates, ud.size - m_after + 1):
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
    dr = rec.get("drift")
    if isinstance(dr, Mapping) and dr.get("coverage") is not None:
        c = min(c, float(dr["coverage"]))       # recent_drift: the windows stopped holding
    st = rec.get("stability")
    return c * (float(st) if st is not None else 1.0)


def drift_hold(drift: Mapping[str, Any], nominal: Optional[float]) -> Optional[float]:
    """(round 6) The probability that windows whose latest dates contradict
    them (recent_drift: k of the young dates' n clean arrivals inside) HOLD
    at their stated coverage c - theta, the true share inside, >= c - eps,
    eps = pnode.hold_eps(c), the tolerance of "holds" of P04's held-out tests
    and the statement contract. theta's
    posterior is the one whose mean recent_drift states as the coverage,
    Beta(k + FWD_PRIOR c, n - k + FWD_PRIOR (1 - c)) (the stated coverage c
    as a prior of strength FWD_PRIOR, then the young arrivals). Pack O D1,
    day 14: 0 of 3 young logins inside 09:00-09:21 gives ~0.007 (round 5
    stated the windows at that posterior's mean, 0.36, and they held 0)."""
    from scipy.special import betainc
    from . import pnode as PN
    if not isinstance(drift, Mapping) or nominal is None:
        return None
    c_nom = float(nominal)
    x = max(0.0, c_nom - PN.hold_eps(c_nom))
    p, any_ = 1.0, False
    for dr in drift.values():
        if not isinstance(dr, Mapping) or dr.get("n") is None:
            continue
        k, n = float(dr.get("k") or 0.0), float(dr["n"])
        any_ = True
        a = k + FWD_PRIOR * c_nom
        b = max(n - k, 0.0) + FWD_PRIOR * (1.0 - c_nom)
        p *= float(1.0 - betainc(a, max(b, 1e-9), x)) if x > 0 else 1.0
    return float(p) if any_ else None


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


def block_coverage(by: Mapping[str, Optional[Mapping[str, Any]]]) -> Dict[str, Any]:
    """The statement-contract coverage of a `when` block: ONE number that the
    windows of EVERY stated day type hold (eval/pmetrics checks each day
    type's held-out arrivals against it, P04's hold tests use min(nominal)
    likewise) = the smallest day-type coverage; 'coverage_by_daytype' keeps
    each. Was the evidence-weighted mean (round 4): pack O seed 1, day 21,
    portal's non-workday windows (6 dates, coverage 0.74-0.82) were stated at
    the workday-dominated 0.88-0.90 and held 0.68-0.79."""
    cov = {DT_LONG[dk]: float(r["coverage"]) for dk, r in by.items()
           if r and r.get("coverage") is not None and dk in DT_LONG}
    return {"coverage": (min(cov.values()) if cov else None), "coverage_by_daytype": cov}


def part_when(when: Any, members: Iterable[str], tz_offset_s: float = 0.0,
              min_points: int = MIN_POINTS, drop: Optional[set] = None) -> Optional[Dict[str, Any]]:
    """The statement-contract `when` block of ONE learned group's part of a
    node, fitted by the node's own pipeline (fit_regime: learning-mass weights,
    the current regime, current-persistence acceptance, cross-validated coverage) on
    the node's minute-reservoir arrivals whose source is a member of the
    group; None when the node keeps no reservoir or the part has fewer than
    `min_points` arrivals on every day type (the view then states the node's
    windows). Integration addition (2026-09-30): on a login route shared by
    综合部 (09:00-09:21), 财务部 (09:05-09:30) and 销售部 (08:30-09:30) the
    node's union window matched no department's truth (IoU 0.17-0.42).
    Round 4: the part was fitted on its raw arrivals (no weights, no regime):
    the 综合部 part of pack O's shared login / home nodes stated the window
    before D1 (08:49-09:21, held-out coverage 0.08-0.13 against a stated
    0.80-0.90, seed 0 day 14) and the mail parts their in-sample coverage.
    `drop` = {(ts, source)} of rows P03 judged violations (P06's ledger,
    behavior.content_bounds.point_ledger), left out as in the node fit."""
    res = getattr(when, "res", None)
    if res is None or not len(res):
        return None
    mem = {str(m) for m in members}
    items = res.items()
    med = float(np.median([w for _it, w, _t in items])) if items else 1.0
    by: Dict[str, Optional[Dict[str, Any]]] = {}
    for d, dk in enumerate(DAYTYPES):
        pts = [(float(it[1]), float(t), 1.0 if med <= 0 else min(1.0, float(w) / med), str(it[2]))
               for it, w, t in items if len(it) > 2 and int(it[0]) == d and str(it[2]) in mem
               and not (drop and (round(float(t), 3), str(it[2])) in drop)]
        if len(pts) < min_points:
            by[dk] = None
            continue
        h = np.zeros(SLOTS)
        for m, _t, w, _s in pts:
            h[int(m // 15) % SLOTS] += w
        rec = fit_regime(h, float(len(pts)), pts, tz_offset_s=tz_offset_s)
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
    out.update(block_coverage(by))
    out["confidence"] = float(min(r["confidence"] for r in fitted))
    drift = {DT_LONG[dk]: by[dk]["drift"] for dk in DAYTYPES if by.get(dk) and by[dk].get("drift")}
    if drift:
        out["drift"] = drift
    out["part"] = True
    out["n_points"] = int(sum(float(r.get("n_points") or 0) for r in fitted))
    out["by_daytype"] = by
    return out
