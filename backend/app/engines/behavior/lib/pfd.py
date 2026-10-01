"""Bindings: approximate functional dependencies of the progressive core
(docs/lib3/progressive.md §6.12).

STATUS: implemented (W-P4, P08 maths). Pure functions and small bounded
structures; no store access.

Screening (hourly, on a stratified probe sample of learned rows)
    ProbeReservoir      per stratum (route template / bootstrap key) a decayed
                        weighted reservoir of R_k rows, at most S_max strata; each
                        row keeps only who / session / client-stack and payload
                        candidate columns. Rows carry the HT weight
                        stratum mass / rows kept, so a rare stratum (three daily GA
                        logins among portal page views) keeps min(R_k, size) rows.
    screen(rows, ...)   pairs (X, Y): X a who level at the system's who-granularity
                        (net.src@ip, @/24, @grp), client.stack, sess.key; Y a
                        categorical / text attribute that is not shape-only; kept when
                        H(Y) >= 0.5 bit, >= 2 values of X with >= 3 rows and
                        g3(X -> Y) <= 0.2, or (set bindings) the median number of
                        distinct y per heavy x <= 4 with H(Y) >= 1 bit and the
                        values repeating (median rows per distinct y >= 2). The reverse
                        direction (Y -> X) is screened the same way. <= Q_pairs per system.
    Pair names: X = 'net.src' (level 0) or 'net.src@<level>' (x_name / parse_x);
    P04 keeps pnode.PairSketch at node.pairs[(X, Y)] with x = gen(attr, level, value).
Fit (per node, from its PairSketch; n_x, k_x evidence on the confidence channel)
    y*_x = argmax_y c(x, y), k_x = c(x, y*_x)
    prior_x = leave-one-out empirical Bayes over the other heavy x' (pmdl.eb_beta_prior)
    LB_x = 5 % quantile of Beta(a0 + k_x, b0 + n_x - k_x); binding(x) <=> n_x >= 5, LB_x >= 0.8
    FD holds <=> bindings (set bindings included) cover >= 80 % of the mass of the
                 judged sources (n_x >= 5: "heavy x" read as "enough evidence to be
                 judged", deviation) and g3 = 1 - sum k_x / sum n_x <= 0.05 over the
                 judged sources without a set binding (g3_all: over every tracked source)
    one-to-one <=> the reverse FD (each y from one x) also holds
    set binding(x) <=> no binding, U_x = (N1_x + E_x + 0.5)/(n_x + 1) <= 0.05, <= 4 values >= 95 %
    shared:<ip> keys never get per-IP bindings.
    Value history (ValueHistory / classify, used by P08 since 2026-10-01): which of
    x's values count in the fit. Superseded (a rename: the current value has >= 5
    clean events over >= 2 normal days and the old value is absent from x's last 5)
    and pending (a minority newcomer not yet confirmed: >= 5 clean events, >= 2
    normal days, >= 5 days, no governor episode, not the established value of
    another source) values are excluded (fit_pair(exclude=...)), so a borrowed
    credential never joins x's binding by persistence and a rename rebinds at
    every node holding x. (RebindTracker / rebind_baseline: the per-node form,
    kept for reference; segment baselines are still accepted by fit_pair.)
Scoring (P03)
    check_forward(rec, x, y)   p_bind = (b0 + n_x - k_x)/(a0 + b0 + n_x) for y != y*_x,
                               flag cross_binding when y is another x's bound value;
                               U_x outside a set binding.
    check_reverse(rec, x, y, ...)  y's own sources closed and x not among them:
                               foreign_source, refined by the concurrency test into
                               concurrent_use (a bound source of y active within
                               T_conc) or readdress_candidate (x unknown and no bound
                               source of y active since the start of the local day).
"""
from __future__ import annotations

import heapq
import math
from collections import deque
from typing import Any, Callable, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import pmdl
from . import psketch as PS
from .phier import Shaped

N_BIND = 5.0
LB_BIND = 0.8
FD_COVER = 0.8
FD_G3 = 0.05
SET_U = 0.05
SET_K = 4
SET_COVER = 0.95
S_CAP = 20.0
SCREEN_HY = 0.5
SCREEN_HY_SET = 1.0
SCREEN_G3 = 0.2
SET_MIN_CARD = 8         # payload side of a set binding: distinct values system-wide (deviation)
SCREEN_LAMBDA = 0.5      # X removes >= half of the mode-prediction error of Y (deviation: not in §6.12)
SCREEN_MIN_ROWS = 3
SCREEN_SET_MED = 4
SCREEN_SET_REP = 2.0
Q_PAIRS = 16
T_CONC = 3600.0
REBIND_N = 5
REBIND_DAYS = 2
RECENT = 5
R_K = 32                 # rows a stratum may always hold in the probe (r_min, §6.4)
R_TOTAL = 4096           # probe rows per tree (R_p, §6.4), allocated ~ sqrt(stratum mass)
R_MAX = 1024             # rows per stratum at most
S_MAX = 64               # strata per system
OTHER_STRATUM = "__other__"
NAN = float("nan")


def x_name(attr: str, level: int) -> str:
    return attr if int(level) == 0 else f"{attr}@{int(level)}"


def parse_x(name: str) -> Tuple[str, int]:
    if "@" in name:
        a, l = name.rsplit("@", 1)
        if l.isdigit():
            return a, int(l)
    return name, 0


def is_shared(x: Any) -> bool:
    return isinstance(x, str) and x.startswith("shared:")


# ================================================================= probe
class ProbeReservoir:
    """Stratified probe of learned rows for pair screening (§6.4 / §6.12).

    One time-decayed (H_m keys) reservoir per stratum, at most S_max strata;
    the R_total rows are allocated in proportion to sqrt(stratum mass), at
    least r_min (or the stratum's size) and at most r_max per stratum, the way
    §6.4 stratifies P05's probe: a mass-proportional sample would leave three
    daily GA logins out of a probe dominated by health checks, and a fixed
    per-stratum size (64, the first version) held ~1.5 days of pack O's OA
    logins, too few rows per source (>= 3) to screen per-IP bindings.
    Shrinking a reservoir keeps its largest keys, i.e. exactly the smaller
    reservoir's sample. Memory: <= R_total rows of <= a_row columns."""

    REBALANCE_EVERY = 512

    def __init__(self, r_k: int = R_K, s_max: int = S_MAX, a_row: int = 24, seed: int = 0,
                 r_total: int = R_TOTAL, r_max: int = R_MAX) -> None:
        self.r_k = int(r_k)                 # r_min: rows a stratum always may hold
        self.s_max = int(s_max)
        self.a_row = int(a_row)
        self.seed = int(seed)
        self.r_total = int(r_total)
        self.r_max = int(r_max)
        self.res: Dict[str, PS.WeightedReservoir] = {}
        self.mass: Dict[str, PS.DecayedVector] = {}
        self.offered = 0
        self._last_t = 0.0

    def offer(self, stratum: str, row: Mapping[str, Any], mass: float, t: float, u: float) -> None:
        self.offered += 1
        k = str(stratum)
        if k not in self.res and len(self.res) >= self.s_max:
            k = OTHER_STRATUM
        r = self.res.get(k)
        if r is None:
            r = self.res[k] = PS.WeightedReservoir(self.r_k, PS.H_M, self.seed + len(self.res))
            self.mass[k] = PS.DecayedVector([PS.H_M])
        self.mass[k].add(t, float(mass))
        self._last_t = max(self._last_t, float(t))
        if len(row) > self.a_row:
            row = dict(list(row.items())[:self.a_row])
        r.offer(dict(row), 1.0, t, u)
        if self.offered % self.REBALANCE_EVERY == 0:
            self.rebalance(self._last_t)

    def rebalance(self, t: float) -> None:
        """Capacities ~ sqrt(mass) within r_total; over-full strata keep their
        largest keys (a valid smaller reservoir)."""
        if not self.res:
            return
        sq = {k: math.sqrt(max(float(self.mass[k].read(t)[0]), 0.0)) for k in self.res}
        tot = sum(sq.values())
        n = len(self.res)
        for k, r in self.res.items():
            share = sq[k] / tot if tot > 0 else 1.0 / n
            cap = int(min(self.r_max, max(self.r_k, round(self.r_total * share))))
            r.R = cap
            h = r._heap                      # min-heap of keys (psketch.WeightedReservoir)
            while len(h) > cap:
                heapq.heappop(h)

    def rows(self, t: float, with_strata: bool = False) -> Tuple[Any, ...]:
        """(rows, HT weights = stratum mass / rows kept in the stratum[, strata])."""
        self.rebalance(t)
        out, w, sk = [], [], []
        for k, r in self.res.items():
            its = r.items()
            if not its:
                continue
            m = float(self.mass[k].read(t)[0])
            for it, _, _ in its:
                out.append(it)
                w.append(m / len(its))
                sk.append(k)
        if with_strata:
            return out, np.asarray(w, dtype=np.float64), sk
        return out, np.asarray(w, dtype=np.float64)

    def n_rows(self) -> int:
        return int(sum(len(r) for r in self.res.values()))

    def nbytes(self) -> int:
        cols = sum(len(it) for r in self.res.values() for it, _, _ in r.items())
        return int(cols * 120 + 200 * len(self.res) + 400)


# =============================================================== screening
def _entropy(counts: Mapping[Any, float]) -> float:
    tot = sum(v for v in counts.values() if v > 0)
    if tot <= 0:
        return 0.0
    return float(-sum((v / tot) * math.log2(v / tot) for v in counts.values() if v > 0))


def _median(xs: Sequence[float]) -> float:
    s = sorted(xs)
    n = len(s)
    return float(s[n // 2]) if n % 2 else 0.5 * (s[n // 2 - 1] + s[n // 2])


def screen_pair(xs: Sequence[Any], ys: Sequence[Any], w: Sequence[float]) -> Optional[Dict[str, Any]]:
    """Screening statistics of X -> Y on aligned sample values (None / ABSENT
    already removed): {'hy', 'g3', 'heavy', 'med_set', 'set', 'fd', 'n'}."""
    joint: Dict[Any, Dict[Any, float]] = {}
    rows: Dict[Any, int] = {}
    ym: Dict[Any, float] = {}
    for x, y, ww in zip(xs, ys, w):
        d = joint.setdefault(x, {})
        d[y] = d.get(y, 0.0) + ww
        rows[x] = rows.get(x, 0) + 1
        ym[y] = ym.get(y, 0.0) + ww
    if not joint:
        return None
    hy = _entropy(ym)
    heavy = [x for x, n in rows.items() if n >= SCREEN_MIN_ROWS and not is_shared(x)]
    N = sum(ym.values())
    g3 = 1.0 - sum(max(d.values()) for d in joint.values()) / N if N > 0 else 1.0
    med = float(_median([len(joint[x]) for x in heavy])) if heavy else math.inf
    rep = float(_median([rows[x] / len(joint[x]) for x in heavy])) if heavy else 0.0
    # Goodman-Kruskal lambda: the share of the mode-prediction error of Y that X
    # removes. g3 alone admits "status -> IP" when one IP dominates Y (the mode
    # already errs by only ~g3); a binding must explain Y, not restate its mode.
    g3_0 = 1.0 - max(ym.values()) / N if N > 0 else 0.0
    lam = (g3_0 - g3) / g3_0 if g3_0 > 1e-12 else 0.0
    fd = hy >= SCREEN_HY and len(heavy) >= 2 and g3 <= SCREEN_G3 and lam >= SCREEN_LAMBDA
    # a small per-x value set only means something when the values repeat
    # (median rows per distinct value >= 2); otherwise 3 rows of 3 random
    # values would look like a set binding
    # (one heavy x is enough for a set binding: "svc_backup is used only from
    # these hosts" is a statement about that one account)
    st = (not fd) and hy >= SCREEN_HY_SET and len(heavy) >= 1 and med <= SCREEN_SET_MED \
        and rep >= SCREEN_SET_REP
    return {"hy": float(hy), "g3": float(g3), "lambda": float(lam), "heavy": len(heavy),
            "med_set": med if math.isfinite(med) else None,
            "fd": bool(fd), "set": bool(st), "n": len(xs)}


def screen(rows: Sequence[Mapping[str, Any]], w: np.ndarray, x_cands: Sequence[Tuple[str, int]],
           y_cands: Sequence[str], gen: Callable[[str, int, Any], Any],
           q_pairs: int = Q_PAIRS, absent: Any = None,
           strata: Optional[Sequence[str]] = None,
           card: Optional[Callable[[str], float]] = None) -> List[Dict[str, Any]]:
    """Candidate pairs from the probe rows (§6.12). x_cands: (attr, level);
    y_cands: attribute names. Returns specs {'x', 'y', 'dir': 'fwd'|'rev', 'stats',
    'strata'} ranked by (1 - g3) lambda min(H(Y), 4), at most q_pairs.
    With `strata` (the stratum of each row: its route), a pair is screened
    inside each stratum as well as pooled and kept when it passes in any: a
    binding is a property of a context (综合部's login), and pooling it with a
    portal's random users dilutes it below the g3 bar. `card(attr)` (the
    registry's distinct count): a set binding needs its payload side to be an
    identifier-like attribute (>= SET_MIN_CARD distinct values system-wide);
    'json is only sent by these two IPs' in an 8-row stratum is not one."""
    out = []
    # generalised X values once per row (not once per (X, Y) pair)
    xcols: Dict[Tuple[str, int], List[Any]] = {}
    for (xa, xl) in x_cands:
        col = []
        memo: Dict[Any, Any] = {}
        for r in rows:
            xv = r.get(xa, absent)
            if xv is absent or xv is None:
                col.append(None)
                continue
            try:
                g = memo.get(xv, memo)
                if g is memo:
                    g = memo[xv] = gen(xa, xl, xv)
            except TypeError:                     # unhashable value
                g = gen(xa, xl, xv)
            col.append(None if g is absent else g)
        xcols[(xa, xl)] = col
    wl = [float(x) for x in w]
    for yname in y_cands:
        idx = []
        yv_all = {}
        for i, r in enumerate(rows):
            v = r.get(yname, absent)
            if v is not absent and v is not None and not isinstance(v, Shaped) \
                    and isinstance(v, (str, int, float, bool)):
                idx.append(i)
                yv_all[i] = v
        if len(idx) < 2 * SCREEN_MIN_ROWS:
            continue
        for (xa, xl) in x_cands:
            col = xcols[(xa, xl)]
            xs, ys, ww = [], [], []
            for i in idx:
                g = col[i]
                if g is None:
                    continue
                xs.append(g)
                ys.append(yv_all[i])
                ww.append(wl[i])
            if len(xs) < 2 * SCREEN_MIN_ROWS:
                continue
            xn = x_name(xa, xl)
            groups: Dict[Optional[str], List[int]] = {None: list(range(len(xs)))}
            if strata is not None:
                for j, i in enumerate(i for i in idx if col[i] is not None):
                    groups.setdefault(str(strata[i]), []).append(j)
            for gk, sel in groups.items():
                if len(sel) < 2 * SCREEN_MIN_ROWS or (gk is not None and len(groups) == 2):
                    continue                     # one stratum only: the pooled test is the same
                gx = [xs[j] for j in sel]
                gy = [ys[j] for j in sel]
                gw = [ww[j] for j in sel]
                id_like = card is None or float(card(yname)) >= SET_MIN_CARD
                st = screen_pair(gx, gy, gw)
                if st is not None and (st["fd"] or (st["set"] and id_like)):
                    out.append({"x": xn, "y": yname, "dir": "fwd", "stats": st, "strata": [gk]})
                rs = screen_pair(gy, gx, gw)
                if rs is not None and (rs["fd"] or (rs["set"] and id_like)):
                    out.append({"x": yname, "y": xn, "dir": "rev", "stats": rs, "strata": [gk]})
    out.sort(key=lambda d: (-(1.0 - d["stats"]["g3"]) * max(d["stats"].get("lambda", 0.0), 0.0)
                            * min(d["stats"]["hy"], 4.0), d["x"], d["y"]))
    seen: Dict[Tuple[str, str], Dict[str, Any]] = {}
    res = []
    for d in out:
        k = (d["x"], d["y"])
        if k in seen:
            seen[k]["strata"] = sorted(set(seen[k]["strata"]) | set(d["strata"]), key=str)
            continue
        if len(res) >= q_pairs:
            continue
        d = dict(d, strata=list(d["strata"]))
        seen[k] = d
        res.append(d)
    for d in res:
        d["strata"] = [x for x in d["strata"] if x is not None]
    return res


# ===================================================================== fit
def _decay(t: float, t0: float) -> float:
    return 2.0 ** (-(float(t) - float(t0)) / PS.H_L)


def pair_counts(ps: Any, t: float, seg: Optional[Mapping[Hashable, Mapping[str, Any]]] = None,
                exclude: Optional[Mapping[str, Any]] = None) -> Dict[Hashable, Dict[str, Any]]:
    """{x: {'n', 'mass', 'y': {y: evidence}, 'U'}} on the confidence channel,
    with the per-x segment baselines (rebinding) subtracted and the values
    `exclude[str(x)]` (superseded or still pending, ValueHistory.classify)
    removed from x's counts."""
    out: Dict[Hashable, Dict[str, Any]] = {}
    for x in ps.x.keys():
        tab = ps.y.get(x)
        if tab is None:
            continue
        ys = {k: e for k, _, _, e in tab.items(t, 0)}
        n = tab.total_evidence(t)
        U = tab.unseen(t)
        ex = (exclude or {}).get(str(x))
        if ex:
            drop = {k for k in ys if str(_jv(k)) in ex}
            if drop and len(drop) < len(ys):
                n = max(0.0, n - sum(ys[k] for k in drop))
                ys = {k: e for k, e in ys.items() if k not in drop}
                n1 = sum(1 for e in ys.values() if e < PS.N1_EVIDENCE)
                U = min(1.0, (n1 + 0.5) / (n + 1.0))
        sg = (seg or {}).get(x)
        if sg:
            f = _decay(t, sg["t"])
            n = max(0.0, n - float(sg.get("n0", 0.0)) * f)
            k0 = sg.get("k0") or {}
            ys = {k: max(0.0, e - float(k0.get(k, 0.0)) * f) for k, e in ys.items()}
            ys = {k: e for k, e in ys.items() if e > 1e-9}
            n1 = sum(1 for e in ys.values() if e < PS.N1_EVIDENCE)
            U = min(1.0, (n1 + 0.5) / (n + 1.0))
        out[x] = {"n": float(n), "mass": float(ps.x.guaranteed(x, t)), "y": ys, "U": float(U)}
    return out


def fit_pair(ps: Any, t: float, seg: Optional[Mapping[Hashable, Mapping[str, Any]]] = None,
             n_bind: float = N_BIND, lb_bind: float = LB_BIND,
             exclude: Optional[Mapping[str, Any]] = None) -> Optional[Dict[str, Any]]:
    """Fitted binding of one node pair (§6.12). None without tracked x.
    `exclude` {str(x): {str(y): reason}}: values kept out of x's counts
    (ValueHistory.classify: superseded by a rename, or a minority value still
    pending confirmation); they are listed on x's table entry."""
    cnt = pair_counts(ps, t, seg, exclude)
    heavy = {x: c for x, c in cnt.items() if c["n"] >= 1.0 and c["y"]}
    if not heavy:
        return None
    tops = {}
    for x, c in heavy.items():
        y, k = max(c["y"].items(), key=lambda kv: (kv[1], str(kv[0])))
        tops[x] = (y, min(k, c["n"]))
    table: Dict[str, Dict[str, Any]] = {}
    per = [x for x in heavy if not is_shared(x)]
    ks = {x: tops[x][1] for x in per}
    ns = {x: heavy[x]["n"] for x in per}
    bound_mass = 0.0
    tot_mass = sum(c["mass"] for c in heavy.values())
    # leave-one-out prior sums (O(1) per x instead of O(#x))
    pur = {o: ks[o] / ns[o] for o in per if ns[o] > 0}
    SK, SN = sum(ks.values()), sum(ns.values())
    SP, SP2, NP = sum(pur.values()), sum(v * v for v in pur.values()), len(pur)
    for x, c in heavy.items():
        y, k = tops[x]
        n = c["n"]
        ent: Dict[str, Any] = {"n": n, "top": _jv(y), "k": k, "mass": c["mass"]}
        ex = (exclude or {}).get(str(x))
        if ex:
            for yy, why in ex.items():
                ent.setdefault(str(why), []).append(yy)
        if not is_shared(x):
            a0, b0 = _loo_prior(SK - ks[x], SN - ns[x],
                                SP - pur.get(x, 0.0), SP2 - pur.get(x, 0.0) ** 2,
                                NP - (1 if x in pur else 0), len(per) - 1)
            lb = pmdl.beta_quantile(0.05, a0 + k, b0 + max(0.0, n - k))
            ent.update({"LB": float(lb), "a0": float(a0), "b0": float(b0),
                        "p_viol": float((b0 + n - k) / (a0 + b0 + n))})
            # a rename (a superseded value: the successor already has >= REBIND_N
            # clean events over >= REBIND_DAYS normal days in the value history)
            # moves the binding to the new value, y*_x <- y' (§6.12), with the
            # confidence of the new segment: the evidence-unit count of the new
            # segment may still be below n_bind (decay, burst runs)
            renamed = "superseded" in set(map(str, ((exclude or {}).get(str(x)) or {}).values()))
            if (n >= n_bind or renamed) and lb >= lb_bind:
                ent["bound"] = True
                bound_mass += c["mass"]
        if not ent.get("bound"):
            ys = sorted(c["y"].items(), key=lambda kv: -kv[1])
            acc, sset = 0.0, []
            for yy, e in ys:
                if acc >= SET_COVER * n:
                    break
                sset.append(_jv(yy))
                acc += e
            if c["U"] <= SET_U and len(sset) <= SET_K and acc >= SET_COVER * n - 1e-9 and len(sset) >= 2:
                ent["set"] = sorted(sset, key=lambda v: (type(v).__name__, str(v)))
                ent["U"] = float(c["U"])
        table[_jx(x)] = ent
    N = sum(c["n"] for c in heavy.values())
    # set-bound sources (shared terminals) are explained by their sets: they
    # count toward the coverage of the dependency but not toward g3
    # the dependency is judged on the sources with enough evidence to be judged
    # (n_x >= n_bind): a DHCP pool whose personas show up on a new address every
    # day contributes many one-login sources that are neither bound nor
    # counter-examples, and must not veto 综合部's bindings at a shared login node
    judged = {x for x in heavy if heavy[x]["n"] >= n_bind or table[_jx(x)].get("bound")}
    ents = dict(zip(heavy, table.values()))
    tot_mass = sum(heavy[x]["mass"] for x in judged)
    set_mass = sum(heavy[x]["mass"] for x in judged if ents[x].get("set"))
    fx = [x for x in judged if not ents[x].get("set")]
    Nf = sum(heavy[x]["n"] for x in fx)
    g3 = 1.0 - sum(tops[x][1] for x in fx) / Nf if Nf > 0 else 1.0
    fa = [x for x in heavy if not ents[x].get("set")]
    Na = sum(heavy[x]["n"] for x in fa)
    g3_all = 1.0 - sum(tops[x][1] for x in fa) / Na if Na > 0 else 1.0
    holds = bool(tot_mass > 0 and (bound_mass + set_mass) / tot_mass >= FD_COVER
                 and g3 <= FD_G3 and bound_mass > 0)
    # reverse FD (each y from one x)
    rev: Dict[Any, Dict[Any, float]] = {}
    for x, c in heavy.items():
        for y, e in c["y"].items():
            rev.setdefault(y, {})[x] = rev.get(y, {}).get(x, 0.0) + e
    Nr = sum(sum(d.values()) for d in rev.values())
    g3_rev = 1.0 - sum(max(d.values()) for d in rev.values()) / Nr if Nr > 0 else 1.0
    bound_values: Dict[str, List[str]] = {}
    for x, ent in table.items():
        if ent.get("bound"):
            bound_values.setdefault(str(ent["top"]), []).append(x)
    hy = pmdl.entropy_plugin(np.asarray([sum(d.values()) for d in rev.values()]))
    hyx = sum(heavy[x]["n"] / N * pmdl.entropy_plugin(np.asarray(list(heavy[x]["y"].values())))
              for x in heavy) if N > 0 else 0.0
    return {"fd": {"g3": float(g3), "g3_all": float(g3_all), "n": float(N), "holds": holds,
                   "judged": len(judged),
                   "one_to_one": bool(holds and g3_rev <= FD_G3), "g3_rev": float(g3_rev),
                   "bound_share": float(bound_mass / tot_mass) if tot_mass > 0 else 0.0},
            "table": table, "bound_values": bound_values,
            "gain": float(max(0.0, hy - hyx)),
            "confidence": float(min([e["LB"] for e in table.values() if e.get("bound")] or [0.0]))}


def _loo_prior(sk: float, sn: float, sp: float, sp2: float, npur: int, n_others: int
               ) -> Tuple[float, float]:
    """pmdl.eb_beta_prior over the other heavy sources from running sums:
    mean m = (sum k + 1/2) / (sum n + 1), strength min(20, s_mom, sum n), with
    s_mom the method-of-moments strength of the others' purities (ddof 1);
    Jeffreys (1/2, 1/2) with fewer than two others."""
    if n_others < 2:
        return 0.5, 0.5
    m = (sk + 0.5) / (sn + 1.0)
    s_mom = math.inf
    if npur >= 2:
        mu = sp / npur
        var = max(0.0, (sp2 - npur * mu * mu) / (npur - 1))
        if var > 1e-12:
            s_mom = mu * (1.0 - mu) / var - 1.0
            s_mom = s_mom if s_mom > 0 else 1e-6
    st = max(min(S_CAP, s_mom, sn), 1e-6)
    return float(m * st), float((1.0 - m) * st)


def _jv(v: Any) -> Any:
    return v if isinstance(v, (str, int, float, bool)) else str(v)


def _jx(x: Any) -> str:
    return str(x)


def material_change(old: Optional[Mapping[str, Any]], new: Mapping[str, Any]) -> bool:
    """A bound value or set changed (§6.8.2 cver rule)."""
    def sig(r: Optional[Mapping[str, Any]]) -> Dict[str, Any]:
        if not r:
            return {}
        return {x: (e.get("top") if e.get("bound") else tuple(e.get("set") or ()))
                for x, e in (r.get("table") or {}).items() if e.get("bound") or e.get("set")}
    return sig(old) != sig(new)


# ================================================================ scoring
def check_forward(rec: Mapping[str, Any], x: Any, y: Any) -> Tuple[float, List[str]]:
    """(p, flags) of the event (x, y) against the X -> Y binding of a node."""
    if not rec:
        return NAN, []
    ent = (rec.get("table") or {}).get(str(x))
    if ent is None:
        return NAN, []
    ys = str(_jv(y))
    if ent.get("bound") and (rec.get("fd") or {}).get("holds"):
        if ys == str(ent["top"]):
            return 1.0, []
        flags = ["unbound_value"]
        if ys in (rec.get("bound_values") or {}):
            flags = ["cross_binding"]
        return float(ent["p_viol"]), flags
    if ent.get("set"):
        if ys in set(map(str, ent["set"])):
            return 1.0, []
        return float(ent.get("U", NAN)), ["outside_set"]
    return NAN, []


def check_reverse(rec: Mapping[str, Any], x: Any, y: Any, t: float,
                  last_active: Optional[Callable[[str], Optional[float]]] = None,
                  x_known: bool = True, day_start: Optional[float] = None,
                  t_conc: float = T_CONC) -> Tuple[float, List[str]]:
    """(p, flags) of the event (x, y) against the reverse Y -> X binding
    (rec fitted on the pair (Y, X)): y's sources are closed and x is not
    among them -> foreign_source, refined by the concurrency test."""
    if not rec:
        return NAN, []
    ent = (rec.get("table") or {}).get(str(_jv(y)))
    if ent is None:
        return NAN, []
    xs = str(x)
    if ent.get("bound"):
        sources = [str(ent["top"])]
        p = float(ent["p_viol"])
    elif ent.get("set"):
        sources = [str(s) for s in ent["set"]]
        p = float(ent.get("U", NAN))
    else:
        return NAN, []
    if xs in sources:
        return 1.0, []
    flags = ["foreign_source"]
    if last_active is not None:
        acts = [a for a in (last_active(s) for s in sources) if a is not None]
        if any(abs(float(t) - a) <= t_conc for a in acts):
            flags.append("concurrent_use")
        elif not x_known and (day_start is None or not any(a >= day_start for a in acts)):
            flags.append("readdress_candidate")
            p = float(ent.get("U", p)) if ent.get("set") else float(ent.get("p_viol", p))
    return p, flags


# ============================================================== rebinding
class RebindTracker:
    """Per (node, pair) recency of bound sources (bounded by the pair's
    tracked x, <= 64): last RECENT values, candidate new values with their
    counts and normal days, first / last seen."""

    def __init__(self, cap: int = 64) -> None:
        self.cap = int(cap)
        self.x: "Dict[str, Dict[str, Any]]" = {}

    def observe(self, x: str, y: Any, ts: float, day: int, bound_to: Optional[str],
                normal: bool) -> None:
        st = self.x.get(x)
        if st is None:
            if len(self.x) >= self.cap:
                old = min(self.x, key=lambda k: self.x[k]["last"])
                del self.x[old]
            st = self.x[x] = {"recent": deque(maxlen=RECENT), "cand": {}, "first": ts, "last": ts}
        st["first"] = min(st["first"], ts)
        st["last"] = max(st["last"], ts)
        ys = str(_jv(y))
        st["recent"].append(ys)
        if bound_to is not None and ys != bound_to:
            c = st["cand"].setdefault(ys, {"n": 0, "days": set()})
            c["n"] += 1
            if normal:
                c["days"].add(int(day))
            if len(st["cand"]) > 4:
                drop = min(st["cand"], key=lambda k: st["cand"][k]["n"])
                del st["cand"][drop]

    def rebind_candidate(self, x: str, bound_to: str) -> Optional[str]:
        st = self.x.get(x)
        if st is None or bound_to in st["recent"]:
            return None
        best = None
        for y, c in st["cand"].items():
            if c["n"] >= REBIND_N and len(c["days"]) >= REBIND_DAYS:
                if best is None or c["n"] > st["cand"][best]["n"]:
                    best = y
        return best

    def clear_candidates(self, x: str) -> None:
        st = self.x.get(x)
        if st is not None:
            st["cand"].clear()

    def seen(self, x: str) -> Optional[Tuple[float, float]]:
        st = self.x.get(x)
        return None if st is None else (st["first"], st["last"])

    def nbytes(self) -> int:
        return int(len(self.x) * 600 + 100)


def rebind_baseline(ps: Any, x: Any, y_new: Any, t: float) -> Dict[str, Any]:
    """Segment baseline that restarts x's confidence segment at t keeping the
    new value's evidence: n0 = n_x - k_{y'}, k0 = every other value's evidence."""
    tab = ps.y.get(x)
    if tab is None:
        return {"t": float(t), "n0": 0.0, "k0": {}}
    ys = {k: e for k, _, _, e in tab.items(t, 0)}
    n = tab.total_evidence(t)
    kn = ys.get(y_new, 0.0)
    return {"t": float(t), "n0": float(max(0.0, n - kn)),
            "k0": {k: float(e) for k, e in ys.items() if k != y_new}, "y": _jv(y_new)}


# ========================================================= value history
NEWCOMER_GAP = 86400.0      # a value first seen >= 1 d after x's earliest value is a newcomer
PERSIST_S = 5 * 86400.0     # §6.9.2: a single source's change persists >= 5 d
HIST_X = 2048               # sources tracked per pair (LRU)
HIST_Y = 8                  # values tracked per source
HIST_DAYS = 8               # normal days remembered per value (>= REBIND_DAYS)


class ValueHistory:
    """Temporal history of the values of each source x of one pair (X, Y)
    over the whole tree (not per node: a child created by a split sees the
    history its rows already had). Per x: per value y [first ts, last ts,
    clean events, clean normal days], and the last RECENT values; bounded by
    an LRU of HIST_X sources and HIST_Y values per source.

    "Clean" events are trusted rows P03 did not damp (§6.9.3): a borrowed
    credential's damped rows never count toward its confirmation. The history
    feeds `classify`, which decides which minority values may count in the
    binding fit (§6.9.2 acceptance of a change applied to a binding):
      superseded  the source's current value (majority of its last RECENT
                  events) qualifies as a rename (>= REBIND_N clean events over
                  >= REBIND_DAYS normal days) and y was last seen before the
                  current value first appeared and is absent from the recent
                  events (D2: mike -> mike.w; replaces per-node segment baselines)
      pending     y appeared >= NEWCOMER_GAP after the source's earliest value
                  (a newcomer next to an established value) and is not yet
                  confirmed: >= REBIND_N clean events over >= REBIND_DAYS normal
                  days, persisting >= PERSIST_S, the source without a governor
                  episode since y appeared, and y not the established value of
                  another source of the node (a credential bound elsewhere is
                  never adopted by persistence alone: anomaly A2, 192.168.1.21
                  logging in as rose for five days)
    Values present from the start (a shared terminal's users) are neither."""

    def __init__(self, cap_x: int = HIST_X, cap_y: int = HIST_Y) -> None:
        from collections import OrderedDict
        self.cap_x, self.cap_y = int(cap_x), int(cap_y)
        self.x: "OrderedDict[str, Dict[str, Any]]" = OrderedDict()

    def observe(self, x: str, y: Any, ts: float, day: int, normal: bool, clean: bool = True) -> None:
        st = self.x.get(x)
        if st is None:
            if len(self.x) >= self.cap_x:
                self.x.popitem(last=False)
            st = self.x[x] = {"v": {}, "recent": deque(maxlen=RECENT), "ver": 0}
        else:
            self.x.move_to_end(x)
        st["ver"] = int(st.get("ver", 0)) + 1
        ys = str(_jv(y))
        v = st["v"]
        r = v.get(ys)
        if r is None:
            if len(v) >= self.cap_y:
                drop = min(v, key=lambda k: (v[k][2], v[k][1]))
                del v[drop]
            r = v[ys] = [float(ts), float(ts), 0, []]
        r[0] = min(r[0], float(ts))
        r[1] = max(r[1], float(ts))
        if clean:
            r[2] += 1
            if normal and int(day) not in r[3]:
                r[3].append(int(day))
                if len(r[3]) > HIST_DAYS:
                    del r[3][0]
        st["recent"].append(ys)

    def get(self, x: str) -> Optional[Dict[str, Any]]:
        return self.x.get(str(x))

    def version(self, xs: Iterable[Any]) -> int:
        """Sum of the observation counters of the sources xs: changes whenever
        one of them was observed (a node holding them must be refitted, its
        verdicts may have changed although its own evidence barely grew)."""
        return int(sum(int((self.x.get(str(x)) or {}).get("ver", 0)) for x in xs))

    def seen(self, x: str) -> Optional[Tuple[float, float]]:
        st = self.x.get(str(x))
        if not st or not st["v"]:
            return None
        return (min(r[0] for r in st["v"].values()), max(r[1] for r in st["v"].values()))

    def nbytes(self) -> int:
        return int(100 + sum(200 + 120 * len(s["v"]) for s in self.x.values()))


def _qualifies(r: Sequence[Any]) -> bool:
    return int(r[2]) >= REBIND_N and len(r[3]) >= REBIND_DAYS


def classify(hx: Optional[Mapping[str, Any]], values: Iterable[Any], t: float,
             established_elsewhere: Iterable[str] = (),
             episode_since: Optional[Callable[[float], bool]] = None) -> Dict[str, str]:
    """{str(y): 'superseded' | 'pending'} for the values of one source x
    (ValueHistory docstring). hx: ValueHistory.get(x); values: x's values at
    the node; established_elsewhere: the established values of the node's
    other sources; episode_since(ts) -> True when x had a governor episode
    (an incident regime) since ts. Values the history does not know are left
    alone, and at least one value always remains."""
    if not hx:
        return {}
    vals = {str(_jv(y)) for y in values}
    known = {y: hx["v"][y] for y in vals if y in hx["v"]}
    if len(known) < 2:
        return {}
    recent = list(hx.get("recent") or ())
    out: Dict[str, str] = {}
    if recent:
        cnt: Dict[str, int] = {}
        for y in recent:
            cnt[y] = cnt.get(y, 0) + 1
        cur = max(cnt, key=lambda y: (cnt[y], -recent[::-1].index(y)))   # majority, ties -> latest
        rc = known.get(cur)
        if rc is not None and _qualifies(rc):
            for y, r in known.items():
                if y != cur and y not in recent and r[1] < rc[0]:
                    out[y] = "superseded"
    rest = {y: r for y, r in known.items() if y not in out}
    if len(rest) < 2:
        return out
    t_est = min(r[0] for r in rest.values())
    other = set(map(str, established_elsewhere))
    for y, r in rest.items():
        if r[0] - t_est < NEWCOMER_GAP:
            continue                                  # co-existed from the start
        ok = (_qualifies(r) and float(t) - r[0] >= PERSIST_S and y not in other
              and not (episode_since is not None and episode_since(r[0])))
        if not ok:
            out[y] = "pending"
    return out
