"""Scenario adaptation maths of the progressive core: strategy selection for P12
(docs/lib3/progressive.md §6.18, §6.20, §10.2; card P12).

STATUS: implemented (W-P6). Pure functions and small state objects; no store
access. The engine (behavior/system_profile.py) measures, this module decides.

Requirement S20 ("不同的场景怎么自动适配哪些算法引擎"): every system gets, per
strategy dimension, the arm that (1) satisfies the arm's declared preconditions
on the measured system characteristics and (2) has the best MEASURED utility in
one currency (PPC-5):

    U(arm) = gain - LAMBDA_C * cost          bits per event
    gain   = prequential gain in bits/event against the arm's baseline (every
             model is scored before it learns, so the gain is held out by
             construction); cost = measured µs per event; LAMBDA_C = 0.001 bits/µs.

Dimensions (§6.18.2) and how each is decided:

    who      ip | grp | prefix | reg | none   full information: every who
             summary codes each learned IP at every level (P04's two-part code),
             so all five arms are measured every day -> Hedge.
    P06      on (always, when numeric targets exist).
    P07/P08  on | off   full information while on (the fitter reports its gain);
             while off a 1-day probe every 14 days -> Hedge on the probe days.
    P10      on | off   must run to be measured -> budgeted UCB on the gain/cost
             ratio (Tran-Thanh et al. 2012), exploration 1 day in 14.
    win      on | off   (window-event tree) on while metric windows exist.
    tier     XS | S | M | L from the node demand and the daily evidence (P15
             caps it by the global budget, §6.19).
    e_max    8 | 32 | 128 earned per-IP B-engine models (§10.2).
    B-engine applicability per engine: on | class | off from preconditions and
             a B23 label veto.

Hedge runs on U relative to the day's best arm, clipped to [-8, 0] bits/event
(its regret bound needs bounded losses; a gain/cost ratio is unbounded as
cost -> 0; the spec's [-8, 8] clip of absolute U would make every who level
that saves more than 8 bits look alike); the ratio is used only
for the knapsack packing and the UCB index. A switch needs the challenger's U
to exceed the incumbent's by the switching margin on 3 consecutive days
(`strategy_changed` INFO). The margin is 0.05 bits/event (the spec value) for
the first days, then twice the measured day-to-day standard deviation of the
utility difference (never below 0.002 bits/event): noisy utilities need a
difference beyond their noise (no flapping), and an arm whose
gain is exactly zero (a fitter that finds nothing, e.g. bindings on a portal
of random IPs) differs from 'off' only by its cost, deterministically, and the
fixed 0.05 would keep it on forever whenever its cost is below 50 µs/event.

Everything here is O(#arms) per system and day.
"""
from __future__ import annotations

import math
import zlib
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

LAMBDA_C = 0.001                 # bits per µs (the same exchange rate as P05)
PRICE_FLOOR = 0.01               # CPU shadow price floor (share of LAMBDA_C) while the budget is slack
ETA = 0.5                        # Hedge learning rate per day
U_CLIP = 8.0                     # bits/event
SWITCH_MARGIN = 0.05             # bits/event, spec value (noisy utilities)
SWITCH_MARGIN_MIN = 1e-4         # bits/event: steady (noise-free) differences switch on their sign
SWITCH_DAYS = 3
DIFF_HISTORY = 7                 # days of U kept per arm for the noise estimate
MARGIN_MIN_N = 5                 # daily utilities per arm before the margin adapts
PROBE_EVERY_D = 14
UCB_SD_MIN = 0.01                # floor of the UCB bonus scale (bits/event)
UCB_DECAY = 0.9                  # per-day decay of the UCB statistics (non-stationarity)
BITS_NONE = 32.0                 # an IPv4 address coded without any model
WHO_ARMS = ("ip", "grp", "prefix", "reg", "none")
WHO_LEVELS = {"ip": (0,), "prefix": (1, 2), "grp": (3,), "reg": (4,)}
ONOFF = ("on", "off")
TIER_ORDER = ("XS", "S", "M", "L")
TIER_NODES = {"XS": 32, "S": 256, "M": 1024, "L": 4096}
XS_EVIDENCE_DAY = 200.0          # a tree below 200 evidence units / day stays XS (§6.20)
E_MAX_ARMS = (8, 32, 128)

# preconditions (§6.18.2)
IP_INFO_MIN = 0.05               # CR(net.src@/32)
CHURN_MAX = 0.3                  # new IPs per day / population
GRP_COVER_MIN = 0.5              # P11 groups cover >= 50 % of events
PAYLOAD_MIN = 0.01               # payload_vis (body or query); the spec's 0.05 switched P07 off
                                 # on the pack-O OA system (4 % of its events carry a form body)
PAYLOAD_DAY_MIN = 10.0           # ... or >= 10 events with a body / query per day
SESS_P10_MIN = 0.6
ROUTES_P10_MIN = 5
SESS_B10_MIN = 0.3
STACK_VIS_MIN = 0.2
VETO_FP_MIN = 5
VETO_PREC_MAX = 0.2
MIN_WHO_EVIDENCE = 30.0          # evidence units before the who dimension is decided

BC_BIMODAL = 0.555               # bimodality coefficient of a uniform distribution


# =========================================================== measurements
def clip_u(u: float) -> float:
    return float(min(U_CLIP, max(-U_CLIP, u)))


def shadow_price(usage_share: Optional[float], lam: float = LAMBDA_C) -> float:
    """The exchange rate actually charged for CPU: LAMBDA_C scaled by how much
    of its CPU budget the P-core uses (usage / budget, floored at PRICE_FLOOR,
    at most 1) - the Lagrange multiplier of the budget constraint, small while
    the budget is slack, the nominal rate when it binds. With the nominal rate
    a periodic fitter on a small system can never pay: P08 costs ~300 µs per
    event on pack O's OA (5 s/day over ~17 000 events/day) while its bindings
    save ~0.02 bits per system event (they concern only the login node), and
    §7.3's own estimate (5-60 ms per tree and hour) gives the same order. An
    arm whose gain is exactly zero still loses to 'off' at any price > 0."""
    if usage_share is None or not math.isfinite(float(usage_share)):
        return lam
    return float(lam * min(1.0, max(PRICE_FLOOR, float(usage_share))))


def utility(gain: Optional[float], cost_us: Optional[float], lam: float = LAMBDA_C) -> Optional[float]:
    """U = gain - lambda_c * cost (bits/event); None when the gain is unmeasured."""
    if gain is None or not math.isfinite(float(gain)):
        return None
    c = float(cost_us) if cost_us is not None and math.isfinite(float(cost_us)) else 0.0
    return float(gain) - lam * max(0.0, c)


def who_arm_bits(level_bits: Sequence[float]) -> Dict[str, float]:
    """Bits per event of each who arm from the per-level bits/event of P04's
    two-part code (levels /32, /24, /16, grp, reg): prefix takes the better of
    /24 and /16; none codes the address with no model (32 bits)."""
    lb = [float(x) if x is not None and math.isfinite(float(x)) else math.inf for x in level_bits]
    while len(lb) < 5:
        lb.append(math.inf)
    out = {"ip": lb[0], "prefix": min(lb[1], lb[2]), "grp": lb[3], "reg": lb[4], "none": BITS_NONE}
    return out


def who_utilities(level_bits: Sequence[float]) -> Dict[str, float]:
    """U(arm) = bits(none) - bits(arm): the gain of modelling who at that level
    over not modelling it (who summaries are always kept, so cost = 0)."""
    b = who_arm_bits(level_bits)
    return {a: (BITS_NONE - v) if math.isfinite(v) else -U_CLIP for a, v in b.items()}


def who_preconditions(ch: Mapping[str, Any]) -> Dict[str, Tuple[bool, str]]:
    """Which who arms may be chosen (§6.18.2, §5.1.4):
      ip:     ip_info at /32 >= 0.05 (unknown -> allowed), churn < 0.3/day, not snat
      grp:    P11 groups cover >= 50 % of the events
      reg:    the system has regions (configured or learned covers)
      prefix, none: always.
    An SNAT-suspect system (all users behind one unresolved proxy address) runs
    with who = none only: the proxy must not be profiled as a user."""
    snat = bool(ch.get("snat"))
    info = ch.get("ip_info") or {}
    cr32 = info.get(0, info.get("0")) if isinstance(info, Mapping) else None
    churn = ch.get("churn")
    out: Dict[str, Tuple[bool, str]] = {}
    if snat:
        for a in WHO_ARMS:
            out[a] = (a == "none", "snat_suspect" if a != "none" else "ok")
        return out
    ok_ip, why = True, "ok"
    if cr32 is not None and float(cr32) < IP_INFO_MIN:
        ok_ip, why = False, f"ip_info/32={float(cr32):.3f}<{IP_INFO_MIN}"
    elif churn is not None and float(churn) >= CHURN_MAX:
        ok_ip, why = False, f"churn={float(churn):.2f}>={CHURN_MAX}"
    out["ip"] = (ok_ip, why)
    gc = ch.get("grp_cover")
    out["grp"] = ((gc is not None and float(gc) >= GRP_COVER_MIN),
                  "ok" if gc is not None and float(gc) >= GRP_COVER_MIN else f"grp_cover={gc}")
    rc = ch.get("reg_cover")
    out["reg"] = ((rc is not None and float(rc) >= GRP_COVER_MIN),
                  "ok" if rc is not None and float(rc) >= GRP_COVER_MIN else f"reg_cover={rc}")
    out["prefix"] = (True, "ok")
    out["none"] = (True, "ok")
    return out


def content_preconditions(ch: Mapping[str, Any], who: Optional[str]) -> Dict[str, Tuple[bool, str]]:
    """P06 on while numeric targets exist; P07 / P08 need payload visibility
    (body or query) >= 0.05; P08 also needs who != none or a session key;
    P10 needs identifiable sessions (>= 0.6) and >= 5 route templates."""
    pv = ch.get("payload_vis") or {}
    body = float(pv.get("body", 0.0) or 0.0)
    if float(pv.get("body_day", 0.0) or 0.0) >= PAYLOAD_DAY_MIN:
        body = max(body, PAYLOAD_MIN)             # a few content events a day are enough to learn
    sk = float(ch.get("sess_key_cov", 0.0) or 0.0)
    sess = float(ch.get("sess_ident", 0.0) or 0.0)
    routes = float(ch.get("route_card", 0.0) or 0.0)
    out = {"P06": (bool(ch.get("numeric_targets", True)), "ok"),
           "P07": (body >= PAYLOAD_MIN, "ok" if body >= PAYLOAD_MIN else f"payload_vis={body:.3f}")}
    p08 = body >= PAYLOAD_MIN and (who != "none" or sk >= PAYLOAD_MIN)
    out["P08"] = (p08, "ok" if p08 else ("payload" if body < PAYLOAD_MIN else "who=none,no sess.key"))
    p10 = sess >= SESS_P10_MIN and routes >= ROUTES_P10_MIN
    out["P10"] = (p10, "ok" if p10 else f"sess_ident={sess:.2f},routes={routes:.0f}")
    return out


# ================================================================== Hedge
class Hedge:
    """Exponential weights over a fixed arm set (full information). Log
    weights, renormalised so the max is 0; `leader(allowed)` is the arm of
    largest weight among the allowed ones (the deterministic follow-the-leader
    read-out used for selection; `probs` gives the Hedge distribution)."""

    __slots__ = ("arms", "logw", "n")

    def __init__(self, arms: Sequence[str]) -> None:
        self.arms = tuple(arms)
        self.logw = {a: 0.0 for a in self.arms}
        self.n = 0

    def update(self, U: Mapping[str, Optional[float]], eta: float = ETA) -> None:
        """One day of utilities; arms without a measurement keep their weight.
        Hedge is invariant to a per-day constant added to every arm, so the
        day's utilities are taken relative to the day's best measured arm and
        clipped to [-U_CLIP, 0]: bounded losses (the regret bound) without
        saturating arms whose absolute utility exceeds 8 bits/event (who
        levels differ by tens of bits from 'none' but by a few from each other)."""
        vals = {a: float(U[a]) for a in self.arms
                if U.get(a) is not None and math.isfinite(float(U[a]))}
        if not vals:
            return
        top = max(vals.values())
        any_m = False
        for a, u in vals.items():
            self.logw[a] += eta * clip_u(u - top)
            any_m = True
        if any_m:
            self.n += 1
            m = max(self.logw.values())
            for a in self.arms:
                self.logw[a] = max(self.logw[a] - m, -700.0)

    def probs(self, allowed: Optional[Iterable[str]] = None) -> Dict[str, float]:
        al = [a for a in self.arms if allowed is None or a in set(allowed)]
        if not al:
            return {}
        m = max(self.logw[a] for a in al)
        w = {a: math.exp(self.logw[a] - m) for a in al}
        z = sum(w.values())
        return {a: v / z for a, v in w.items()}

    def leader(self, allowed: Optional[Iterable[str]] = None) -> Optional[str]:
        al = [a for a in self.arms if allowed is None or a in set(allowed)]
        if not al:
            return None
        return max(al, key=lambda a: (self.logw[a], -self.arms.index(a)))

    def to_dict(self) -> Dict[str, Any]:
        return {"arms": list(self.arms), "logw": dict(self.logw), "n": self.n}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Hedge":
        h = cls(d.get("arms") or ())
        h.logw.update({str(k): float(v) for k, v in (d.get("logw") or {}).items() if k in h.logw})
        h.n = int(d.get("n", 0))
        return h


# ===================================================== budgeted UCB (P10)
class BudgetedUCB:
    """Budgeted UCB over arms that must run to be measured (Tran-Thanh et al.
    2012, fractional KUBE-style index): per arm the decayed sums of gain
    (bits/event), squared gain, cost (µs/event) and the number of measured
    days n. Index

        I(a) = (g_bar(a) + max(sd(a), SD_MIN) sqrt(2 ln N / n_a)) / max(c_bar(a), c_min)

    i.e. an optimistic gain per unit cost, the bonus scaled by the observed
    spread of the arm's daily gains (a constant gain needs little exploration).
    An arm is worth running when its optimistic ratio beats the exchange rate
    LAMBDA_C (one bit per 1000 µs): I(a) >= LAMBDA_C <=> optimistic U >= 0.
    Unmeasured arms have I = +inf (explored first); exploration is additionally
    forced 1 day in 14 by the caller (the probe schedule)."""

    __slots__ = ("g", "g2", "c", "n")

    def __init__(self) -> None:
        self.g: Dict[str, float] = {}
        self.g2: Dict[str, float] = {}
        self.c: Dict[str, float] = {}
        self.n: Dict[str, float] = {}

    def observe(self, arm: str, gain: float, cost_us: float, decay: float = UCB_DECAY) -> None:
        for a in list(self.n):
            self.g[a] *= decay
            self.g2[a] *= decay
            self.c[a] *= decay
            self.n[a] *= decay
        self.g[arm] = self.g.get(arm, 0.0) + float(gain)
        self.g2[arm] = self.g2.get(arm, 0.0) + float(gain) ** 2
        self.c[arm] = self.c.get(arm, 0.0) + max(0.0, float(cost_us))
        self.n[arm] = self.n.get(arm, 0.0) + 1.0

    def mean(self, arm: str) -> Tuple[Optional[float], Optional[float]]:
        n = self.n.get(arm, 0.0)
        if n <= 0:
            return None, None
        return self.g[arm] / n, self.c[arm] / n

    def sd(self, arm: str) -> float:
        n = self.n.get(arm, 0.0)
        if n <= 1e-9:
            return 0.0
        m = self.g[arm] / n
        return math.sqrt(max(0.0, self.g2.get(arm, 0.0) / n - m * m))

    def index(self, arm: str, c_min: float = 1.0) -> float:
        n = self.n.get(arm, 0.0)
        if n <= 1e-9:
            return math.inf
        N = max(2.0, sum(self.n.values()) + 1.0)
        g, c = self.mean(arm)
        bonus = max(self.sd(arm), UCB_SD_MIN) * math.sqrt(2.0 * math.log(N) / n)
        return (float(g) + bonus) / max(float(c), c_min)

    def worth_running(self, arm: str, lam: float = LAMBDA_C) -> bool:
        return self.index(arm) >= lam

    def to_dict(self) -> Dict[str, Any]:
        return {"g": dict(self.g), "g2": dict(self.g2), "c": dict(self.c), "n": dict(self.n)}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "BudgetedUCB":
        u = cls()
        for k in ("g", "g2", "c", "n"):
            getattr(u, k).update({str(a): float(v) for a, v in (d.get(k) or {}).items()})
        return u


# ============================================================ hysteresis
def switch_margin(u_cur: Sequence[float], u_new: Sequence[float]) -> float:
    """Margin a challenger must beat the incumbent by: 0.05 bits/event (the spec
    value) until both arms have >= 5 daily utilities, then 2 x the standard
    deviation of the daily difference, sqrt(var_cur + var_new) from the last
    DIFF_HISTORY days of each arm (never below 0.002): a steady cost-only
    difference switches at a small margin, a noisy one only beyond its noise
    (module docstring)."""
    a = [float(x) for x in u_cur if x is not None and math.isfinite(float(x))]
    b = [float(x) for x in u_new if x is not None and math.isfinite(float(x))]
    if len(a) < MARGIN_MIN_N or len(b) < MARGIN_MIN_N:
        return SWITCH_MARGIN
    sd = math.sqrt(float(np.var(a, ddof=1)) + float(np.var(b, ddof=1)))
    return float(max(SWITCH_MARGIN_MIN, 2.0 * sd))


class Switcher:
    """Per-dimension incumbent with hysteresis. `step` is called once per day
    with the day's utilities U (None = unmeasured), the allowed arms and the
    challenger proposed by the selector; it returns (arm, changed, reason).

    * no incumbent yet            -> take the proposal
    * incumbent not allowed       -> switch at once (preconditions are hard)
    * challenger's U >= incumbent's U + margin on SWITCH_DAYS consecutive
      days (same challenger)      -> switch
    Unmeasured days neither advance nor reset the streak. The last
    DIFF_HISTORY daily utilities of every measured arm are kept for the margin."""

    __slots__ = ("cur", "cand", "streak", "uhist", "since")

    def __init__(self, cur: Optional[str] = None) -> None:
        self.cur = cur
        self.cand: Optional[str] = None
        self.streak = 0
        self.uhist: Dict[str, List[float]] = {}
        self.since: Optional[int] = None

    def _record(self, U: Mapping[str, Optional[float]]) -> None:
        for a, u in U.items():
            if u is not None and math.isfinite(float(u)):
                self.uhist[a] = (self.uhist.get(a, []) + [float(u)])[-DIFF_HISTORY:]

    def step(self, proposal: Optional[str], U: Mapping[str, Optional[float]],
             allowed: Iterable[str], day: int, days: int = SWITCH_DAYS) -> Tuple[Optional[str], bool, str]:
        al = list(allowed)
        self._record(U)
        if proposal is None:
            return self.cur, False, "no_proposal"
        if self.cur is None:
            self.cur, self.since = proposal, day
            self.cand, self.streak = None, 0
            return self.cur, True, "initial"
        if self.cur not in al:
            old = self.cur
            self.cur, self.since = (proposal if proposal in al else (al[0] if al else None)), day
            self.cand, self.streak = None, 0
            return self.cur, self.cur != old, "precondition"
        if proposal == self.cur:
            self.cand, self.streak = None, 0
            return self.cur, False, "incumbent"
        u_new, u_cur = U.get(proposal), U.get(self.cur)
        if u_new is None or u_cur is None:
            return self.cur, False, "unmeasured"
        diff = float(u_new) - float(u_cur)
        if self.cand != proposal:
            self.cand, self.streak = proposal, 0
        margin = switch_margin(self.uhist.get(self.cur, ()), self.uhist.get(proposal, ()))
        if diff >= margin:
            self.streak += 1
        else:
            self.streak = 0
        if self.streak >= days:
            self.cur, self.since = proposal, day
            self.cand, self.streak = None, 0
            return self.cur, True, f"utility+{diff:.3f}>={margin:.3f}x{days}d"
        return self.cur, False, f"streak {self.streak}/{days} (d={diff:.3f}, m={margin:.3f})"

    def to_dict(self) -> Dict[str, Any]:
        return {"cur": self.cur, "cand": self.cand, "streak": self.streak,
                "uhist": {a: list(v) for a, v in self.uhist.items()}, "since": self.since}

    @classmethod
    def from_dict(cls, d: Mapping[str, Any]) -> "Switcher":
        s = cls(d.get("cur"))
        s.cand, s.streak = d.get("cand"), int(d.get("streak", 0))
        s.uhist = {str(a): [float(x) for x in v] for a, v in (d.get("uhist") or {}).items()}
        s.since = d.get("since")
        return s


# ================================================================ packing
def pack_knapsack(items: Sequence[Tuple[str, float, float]], budget_us: Optional[float],
                  mandatory: Iterable[str] = ()) -> List[str]:
    """Greedy knapsack by utility per unit cost (§6.18.2): items (name, gain
    bits/event, cost µs/event) with positive U are taken in decreasing
    gain/cost order while their cost fits the remaining CPU budget
    (µs/event); mandatory items are taken first. budget None = unlimited."""
    chosen = [n for n, _, _ in items if n in set(mandatory)]
    left = math.inf if budget_us is None else float(budget_us)
    left -= sum(max(0.0, c) for n, _, c in items if n in set(chosen))
    rest = [(n, g, c) for n, g, c in items if n not in set(chosen) and (g - LAMBDA_C * max(c, 0.0)) > 0]
    rest.sort(key=lambda x: -(x[1] / max(x[2], 1e-3)))
    for n, g, c in rest:
        if max(0.0, c) <= left:
            chosen.append(n)
            left -= max(0.0, c)
    return chosen


# ======================================================= characteristics
def bimodality_coefficient(hist: Sequence[float], centers: Sequence[float]) -> Optional[float]:
    """Sarle's bimodality coefficient BC = (g^2 + 1) / (k + 3 (n-1)^2 / ((n-2)(n-3)))
    of a weighted histogram (g skewness, k excess kurtosis, n the total weight);
    BC > 5/9 suggests bimodality (e.g. log inter-event gaps: short gaps inside
    sessions, long gaps between them). None with fewer than 20 samples."""
    h = np.asarray(hist, dtype=np.float64)
    x = np.asarray(centers, dtype=np.float64)
    n = float(h.sum())
    if n < 20 or h.size != x.size:
        return None
    mu = float((h * x).sum() / n)
    d = x - mu
    m2 = float((h * d ** 2).sum() / n)
    if m2 <= 1e-12:
        return 0.0
    g = float((h * d ** 3).sum() / n) / m2 ** 1.5
    k = float((h * d ** 4).sum() / n) / m2 ** 2 - 3.0
    corr = 3.0 * (n - 1) ** 2 / ((n - 2) * (n - 3))
    den = k + corr
    return float((g * g + 1.0) / den) if den > 0 else None


def heaps_beta(points: Sequence[Tuple[float, float]]) -> Optional[float]:
    """Heaps exponent: slope of log(distinct) against log(events) over the
    cumulative (events, distinct) points (>= 3 points with events > 0)."""
    pts = [(float(e), float(d)) for e, d in points if e > 0 and d > 0]
    if len(pts) < 3:
        return None
    x = np.log([p[0] for p in pts])
    y = np.log([p[1] for p in pts])
    if float(np.ptp(x)) < 1e-9:
        return None
    return float(np.polyfit(x, y, 1)[0])


def recommend_tier(n_nodes: float, evidence_per_day: float, cur: Optional[str] = None) -> str:
    """Node tier from demand (§6.20): XS while the tree receives < 200
    evidence units a day and still fits XS comfortably (< 75 % of its 32
    nodes: a small but real system that has grown its patterns is not pruned
    back because its volume is low); else the smallest tier holding 1.5 x the
    nodes the tree uses (one tier down only below 0.5 x of the lower tier:
    hysteresis)."""
    if evidence_per_day < XS_EVIDENCE_DAY and float(n_nodes) < 0.75 * TIER_NODES["XS"]:
        return "XS"
    need = 1.5 * float(n_nodes)
    for t in TIER_ORDER[1:]:
        if TIER_NODES[t] >= need:
            if cur in TIER_ORDER and TIER_ORDER.index(cur) > TIER_ORDER.index(t):
                lower_ok = float(n_nodes) <= 0.5 * TIER_NODES[t]
                return t if lower_ok else cur
            return t
    return "L"


def earned_cap(n_candidates: int) -> int:
    """E_max arm (§10.2): the smallest of 8 / 32 / 128 that holds the IPs whose
    earned gain passes the test; 8 when there are none."""
    for e in E_MAX_ARMS:
        if n_candidates <= e:
            return e
    return E_MAX_ARMS[-1]


# B-engine applicability (§6.18.2 last row); engine -> (characteristic, minimum)
B_RULES: Dict[str, Tuple[str, float]] = {
    "B10": ("sess_ident", SESS_B10_MIN),          # per-IP PPM needs sessions
    "B11": ("automation_or_prefilter", 0.0),      # timing / beacon: machine-like IPs
    "B12": ("automation_or_prefilter", 0.0),
    "B09": ("stack_vis", STACK_VIS_MIN),          # client identity needs JA3 / UA
    "B15": ("stack_vis", STACK_VIS_MIN),
    "B13": ("upload_routes", 0.0),                # exfil budgets need upload-capable routes
}
B_FAMILIES: Dict[str, Tuple[str, ...]] = {        # detector families an engine owns (veto)
    "B10": ("sequence",), "B11": ("timing",), "B12": ("timing",), "B13": ("budget",),
    "B09": ("identity",), "B15": ("identity",), "B07": ("temporal",), "B08": ("categorical",),
    "B14": ("change",), "B06": ("shape",), "B04": ("intensity", "shape"),
}


def b_applicability(ch: Mapping[str, Any], vetoed: Iterable[str] = ()) -> Dict[str, str]:
    """{engine: 'on' | 'class' | 'off'} per system. A failed precondition sends
    the engine to class-only (its class / system tiers keep running: they are
    cheap and cover every IP); a detector family vetoed by B23 labels (>= 5 fp,
    label precision < 0.2 over 30 d) also drops to class-only."""
    vs = set(vetoed)
    out: Dict[str, str] = {}
    for eng, (key, lo) in B_RULES.items():
        if key == "automation_or_prefilter":
            ok = float(ch.get("automation", 0.0) or 0.0) > lo or bool(ch.get("periodic_ips"))
        elif key == "upload_routes":
            ok = bool(ch.get("upload_routes", True))
        else:
            v = ch.get(key)
            ok = v is None or float(v) >= lo
        out[eng] = "on" if ok else "class"
    for eng, fams in B_FAMILIES.items():
        if fams and all(f in vs for f in fams):
            out[eng] = "class"
    return out


def label_veto(counts: Mapping[str, Tuple[int, int]]) -> List[str]:
    """Families with >= 5 fp labels and label precision tp/(tp+fp) < 0.2."""
    out = []
    for fam, (tp, fp) in counts.items():
        if fp >= VETO_FP_MIN and tp / max(1, tp + fp) < VETO_PREC_MAX:
            out.append(fam)
    return sorted(out)


def probe_day(system: str, day: int, every: int = PROBE_EVERY_D) -> bool:
    """True on the system's exploration day (1 in `every`, crc32 phase)."""
    ph = zlib.crc32(system.encode("utf-8")) % every
    return (int(day) % every) == ph


# ============================================================== decision
def new_state() -> Dict[str, Any]:
    return {"who": {"hedge": Hedge(WHO_ARMS).to_dict(), "sw": Switcher().to_dict()},
            "P07": {"hedge": Hedge(ONOFF).to_dict(), "sw": Switcher().to_dict()},
            "P08": {"hedge": Hedge(ONOFF).to_dict(), "sw": Switcher().to_dict()},
            "P10": {"ucb": BudgetedUCB().to_dict(), "sw": Switcher().to_dict()},
            "tier": None, "day": None, "history": []}


def decide(state: Dict[str, Any], ch: Mapping[str, Any], meas: Mapping[str, Any], day: int,
           system: str, explore: bool = True) -> Dict[str, Any]:
    """One daily decision for one tree (mutates and returns `state`).

    ch    characteristics (engine-measured, see system_profile.py)
    meas  {'who': [bits/event per level] | None, 'who_n': evidence,
           'P07'|'P08'|'P10'|'P06': {'gain': bits/event, 'cost': µs/event} | None,
           'n_nodes', 'ev_day', 'n_earned'}
    Returns {'chosen': {dim: arm}, 'arms': {dim: {arm: {gain, cost, U, weight}}},
             'reasons': {dim: str}, 'changed': [(dim, old, new, reason)], 'probe': [dims]}."""
    chosen: Dict[str, str] = {}
    arms: Dict[str, Dict[str, Dict[str, Any]]] = {}
    reasons: Dict[str, str] = {}
    lam = shadow_price(meas.get("cpu_usage_share"))
    changed: List[Tuple[str, Optional[str], str, str]] = []
    probing: List[str] = []
    is_probe = bool(explore) and probe_day(system, day)

    # ---- who (full information, Hedge)
    ws = state.setdefault("who", {"hedge": Hedge(WHO_ARMS).to_dict(), "sw": Switcher().to_dict()})
    hed, sw = Hedge.from_dict(ws["hedge"]), Switcher.from_dict(ws["sw"])
    pre = who_preconditions(ch)
    allowed = [a for a in WHO_ARMS if pre[a][0]]
    lb = meas.get("who")
    U_who: Dict[str, Optional[float]] = {a: None for a in WHO_ARMS}
    if lb is not None and float(meas.get("who_n", 0.0) or 0.0) >= MIN_WHO_EVIDENCE:
        U_who = dict(who_utilities(lb))
        hed.update(U_who)
    bits = who_arm_bits(lb) if lb is not None else {}
    probs = hed.probs(WHO_ARMS)
    arms["who"] = {a: {"gain": U_who.get(a), "cost": 0.0, "U": U_who.get(a), "bits": bits.get(a),
                       "weight": probs.get(a), "allowed": pre[a][0], "why": pre[a][1]}
                   for a in WHO_ARMS}
    if hed.n > 0 or sw.cur is not None or not pre["ip"][0]:
        prop = hed.leader(allowed) if hed.n > 0 else (sw.cur or _who_default(ch, allowed))
        old = sw.cur
        cur, ch_, why = sw.step(prop, U_who, allowed, day)
        if cur is not None:
            chosen["who"] = cur
            reasons["who"] = why
            if ch_ and old is not None:
                changed.append(("who", old, cur, why))
    ws["hedge"], ws["sw"] = hed.to_dict(), sw.to_dict()

    # ---- content engines
    cpre = content_preconditions(ch, chosen.get("who"))
    chosen["P06"] = "on" if cpre["P06"][0] else "off"
    chosen["P09"] = "on"
    chosen["win"] = "on" if ch.get("win_events", True) else "off"
    for dim in ("P07", "P08"):
        st = state.setdefault(dim, {"hedge": Hedge(ONOFF).to_dict(), "sw": Switcher().to_dict()})
        hed, sw = Hedge.from_dict(st["hedge"]), Switcher.from_dict(st["sw"])
        m = meas.get(dim) or {}
        u_on = utility(m.get("gain"), m.get("cost"), lam) if m else None
        ok, why = cpre[dim]
        U = {"on": u_on, "off": 0.0 if u_on is not None else None}
        if u_on is not None and (sw.cur in (None, "on") or st.get("probing")):
            hed.update(U)
        allowed_c = ["on", "off"] if ok else ["off"]
        # the first choice is 'on' when allowed (a full-information arm is only
        # measured while it runs); so is the first day a failed precondition
        # holds again (the arm was off for lack of data, not for its utility)
        fresh = sw.cur is None or (ok and st.get("pre_ok") is False)
        if fresh and sw.cur is not None:
            sw.cur, sw.cand, sw.streak = None, None, 0
            hed = Hedge(ONOFF)
        prop = ("on" if ok else "off") if sw.cur is None else hed.leader(allowed_c)
        old = sw.cur if not fresh else st.get("last_arm")
        cur, ch_, why2 = sw.step(prop, U, allowed_c, day)
        st["pre_ok"] = bool(ok)
        arm = cur or "off"
        st["probing"] = False
        if arm == "off" and ok and is_probe:
            st["probing"] = True
            probing.append(dim)
            arm_out = "on"
        else:
            arm_out = arm
        chosen[dim] = arm_out
        reasons[dim] = why if not ok else why2
        if ch_ and old is not None and old != arm:
            changed.append((dim, old, arm, why2))
        st["last_arm"] = arm
        pr = hed.probs(ONOFF)
        arms[dim] = {"on": {"gain": m.get("gain"), "cost": m.get("cost"), "U": u_on,
                            "weight": pr.get("on"), "allowed": ok, "why": why},
                     "off": {"gain": 0.0, "cost": 0.0, "U": 0.0, "weight": pr.get("off"),
                             "allowed": True, "why": "ok"}}
        st["hedge"], st["sw"] = hed.to_dict(), sw.to_dict()

    # ---- P10 (budgeted UCB, must run to be measured)
    st = state.setdefault("P10", {"ucb": BudgetedUCB().to_dict(), "sw": Switcher().to_dict()})
    ucb, sw = BudgetedUCB.from_dict(st["ucb"]), Switcher.from_dict(st["sw"])
    m = meas.get("P10") or {}
    ok, why = cpre["P10"]
    if m and m.get("gain") is not None and (sw.cur in (None, "on") or st.get("probing")):
        ucb.observe("on", float(m["gain"]), float(m.get("cost") or 0.0))
    g_on, c_on = ucb.mean("on")
    u_on = utility(g_on, c_on, lam) if g_on is not None else None
    run = ok and ucb.worth_running("on", lam)
    prop = "on" if run else "off"
    fresh = sw.cur is not None and ok and st.get("pre_ok") is False
    old = sw.cur if not fresh else st.get("last_arm")
    if fresh:
        sw.cur, sw.cand, sw.streak = None, None, 0
    cur, ch_, why2 = sw.step(prop, {"on": u_on, "off": 0.0 if u_on is not None else None},
                             ["on", "off"] if ok else ["off"], day)
    st["pre_ok"] = bool(ok)
    arm = cur or "off"
    st["probing"] = False
    if arm == "off" and ok and is_probe:
        st["probing"] = True
        probing.append("P10")
        chosen["P10"] = "on"
    else:
        chosen["P10"] = arm
    reasons["P10"] = why if not ok else why2
    if ch_ and old is not None and old != arm:
        changed.append(("P10", old, arm, why2))
    st["last_arm"] = arm
    idx = ucb.index("on")
    arms["P10"] = {"on": {"gain": g_on, "cost": c_on, "U": u_on,
                          "index": None if not math.isfinite(idx) else idx, "allowed": ok, "why": why},
                   "off": {"gain": 0.0, "cost": 0.0, "U": 0.0, "allowed": True, "why": "ok"}}
    st["ucb"], st["sw"] = ucb.to_dict(), sw.to_dict()

    # ---- budget packing (§6.18.2): when the P-core exceeds its CPU budget the
    # optional arms that are on are packed by gain / cost into the share of
    # their current cost the budget allows; the rest are switched off
    use = meas.get("cpu_usage_share")
    if use is not None and float(use) > 1.0:
        items = []
        for dim in ("P07", "P08", "P10"):
            m = meas.get(dim) or {}
            if chosen.get(dim) == "on" and m.get("gain") is not None:
                items.append((dim, float(m["gain"]), float(m.get("cost") or 0.0)))
        if items:
            budget_us = sum(c for _, _, c in items) / float(use)
            keep = set(pack_knapsack(items, budget_us))
            for dim, _, _ in items:
                if dim not in keep:
                    chosen[dim] = "off"
                    reasons[dim] = f"budget (P-core at {float(use):.1f}x its CPU share)"

    # ---- P00 body parsing: off only when nothing downstream reads bodies
    body_used = chosen["P07"] == "on" or chosen["P08"] == "on"
    pv = float((ch.get("payload_vis") or {}).get("body", 0.0) or 0.0)
    chosen["content"] = "on" if (body_used or pv >= PAYLOAD_MIN or is_probe) else "off"

    # ---- tier and earned cap
    t_old = state.get("tier")
    tier = recommend_tier(float(meas.get("n_nodes", 0.0) or 0.0), float(meas.get("ev_day", 0.0) or 0.0), t_old)
    state["tier"] = tier
    chosen["tier"] = tier
    chosen["e_max"] = str(earned_cap(int(meas.get("n_earned", 0) or 0)))

    # ---- B-engine applicability
    for eng, mode in b_applicability(ch, ch.get("vetoed") or ()).items():
        chosen[eng] = mode

    # lower-case aliases read by consumers that accept 'p0x' only (P09/P10 today)
    for k in ("P07", "P08", "P09", "P10"):
        chosen[k.lower()] = chosen[k]
    state["day"] = int(day)
    hist = state.setdefault("history", [])
    for c in changed:
        hist.append({"day": int(day), "dim": c[0], "from": c[1], "to": c[2], "why": c[3]})
    state["history"] = hist[-64:]
    return {"chosen": chosen, "arms": arms, "reasons": reasons, "changed": changed, "probe": probing,
            "lambda": lam}


def _who_default(ch: Mapping[str, Any], allowed: Sequence[str]) -> Optional[str]:
    """Before any who measurement: P05's who_mode when allowed, else the
    finest allowed arm."""
    wm = ch.get("who_mode")
    if wm in allowed:
        return str(wm)
    for a in ("ip", "grp", "prefix", "reg", "none"):
        if a in allowed:
            return a
    return None
