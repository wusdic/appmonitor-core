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
    FD holds <=> bindings (set bindings included) cover >= 80 % of heavy-x mass and
                 g3 = 1 - sum k_x / sum n_x <= 0.05 over the sources without a set binding
    one-to-one <=> the reverse FD (each y from one x) also holds
    set binding(x) <=> no binding, U_x = (N1_x + E_x + 0.5)/(n_x + 1) <= 0.05, <= 4 values >= 95 %
    shared:<ip> keys never get per-IP bindings.
    Rebinding (legitimate rename): a new y' with >= 5 trusted events over >= 2 normal days
    and no y*_x among x's last 5 events -> y*_x <- y'; the pair's confidence segment
    for x restarts (subtractive baseline on the forward-decayed counts: exact, §6.1).
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

import math
from collections import deque
from typing import Any, Callable, Dict, Hashable, List, Mapping, Optional, Sequence, Tuple

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
SCREEN_MIN_ROWS = 3
SCREEN_SET_MED = 4
SCREEN_SET_REP = 2.0
Q_PAIRS = 16
T_CONC = 3600.0
REBIND_N = 5
REBIND_DAYS = 2
RECENT = 5
R_K = 64                 # rows per stratum in the probe
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
    """Stratified probe of learned rows for pair screening (bounded:
    <= S_max strata x R_k rows, each row <= a_row columns)."""

    def __init__(self, r_k: int = R_K, s_max: int = S_MAX, a_row: int = 24, seed: int = 0) -> None:
        self.r_k = int(r_k)
        self.s_max = int(s_max)
        self.a_row = int(a_row)
        self.seed = int(seed)
        self.res: Dict[str, PS.WeightedReservoir] = {}
        self.mass: Dict[str, PS.DecayedVector] = {}
        self.offered = 0

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
        if len(row) > self.a_row:
            row = dict(list(row.items())[:self.a_row])
        r.offer(dict(row), 1.0, t, u)

    def rows(self, t: float) -> Tuple[List[Dict[str, Any]], np.ndarray]:
        """(rows, HT weights = stratum mass / rows kept in the stratum)."""
        out, w = [], []
        for k, r in self.res.items():
            its = r.items()
            if not its:
                continue
            m = float(self.mass[k].read(t)[0])
            for it, _, _ in its:
                out.append(it)
                w.append(m / len(its))
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
    fd = hy >= SCREEN_HY and len(heavy) >= 2 and g3 <= SCREEN_G3
    # a small per-x value set only means something when the values repeat
    # (median rows per distinct value >= 2); otherwise 3 rows of 3 random
    # values would look like a set binding
    # (one heavy x is enough for a set binding: "svc_backup is used only from
    # these hosts" is a statement about that one account)
    st = (not fd) and hy >= SCREEN_HY_SET and len(heavy) >= 1 and med <= SCREEN_SET_MED \
        and rep >= SCREEN_SET_REP
    return {"hy": float(hy), "g3": float(g3), "heavy": len(heavy),
            "med_set": med if math.isfinite(med) else None,
            "fd": bool(fd), "set": bool(st), "n": len(xs)}


def screen(rows: Sequence[Mapping[str, Any]], w: np.ndarray, x_cands: Sequence[Tuple[str, int]],
           y_cands: Sequence[str], gen: Callable[[str, int, Any], Any],
           q_pairs: int = Q_PAIRS, absent: Any = None) -> List[Dict[str, Any]]:
    """Candidate pairs from the probe rows (§6.12). x_cands: (attr, level);
    y_cands: attribute names. Returns specs {'x', 'y', 'dir': 'fwd'|'rev', 'stats'}
    ranked by (1 - g3) * min(H(Y), 4), at most q_pairs."""
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
            st = screen_pair(xs, ys, ww)
            if st is not None and (st["fd"] or st["set"]):
                out.append({"x": xn, "y": yname, "dir": "fwd", "stats": st})
            rs = screen_pair(ys, xs, ww)
            if rs is not None and (rs["fd"] or rs["set"]):
                out.append({"x": yname, "y": xn, "dir": "rev", "stats": rs})
    out.sort(key=lambda d: (-(1.0 - d["stats"]["g3"]) * min(d["stats"]["hy"], 4.0), d["x"], d["y"]))
    seen, res = set(), []
    for d in out:
        k = (d["x"], d["y"])
        if k in seen:
            continue
        seen.add(k)
        res.append(d)
        if len(res) >= q_pairs:
            break
    return res


# ===================================================================== fit
def _decay(t: float, t0: float) -> float:
    return 2.0 ** (-(float(t) - float(t0)) / PS.H_L)


def pair_counts(ps: Any, t: float, seg: Optional[Mapping[Hashable, Mapping[str, Any]]] = None
                ) -> Dict[Hashable, Dict[str, Any]]:
    """{x: {'n', 'mass', 'y': {y: evidence}, 'U'}} on the confidence channel,
    with the per-x segment baselines (rebinding) subtracted."""
    out: Dict[Hashable, Dict[str, Any]] = {}
    for x in ps.x.keys():
        tab = ps.y.get(x)
        if tab is None:
            continue
        ys = {k: e for k, _, _, e in tab.items(t, 0)}
        n = tab.total_evidence(t)
        U = tab.unseen(t)
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
             n_bind: float = N_BIND, lb_bind: float = LB_BIND) -> Optional[Dict[str, Any]]:
    """Fitted binding of one node pair (§6.12). None without tracked x."""
    cnt = pair_counts(ps, t, seg)
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
        if not is_shared(x):
            a0, b0 = _loo_prior(SK - ks[x], SN - ns[x],
                                SP - pur.get(x, 0.0), SP2 - pur.get(x, 0.0) ** 2,
                                NP - (1 if x in pur else 0), len(per) - 1)
            lb = pmdl.beta_quantile(0.05, a0 + k, b0 + max(0.0, n - k))
            ent.update({"LB": float(lb), "a0": float(a0), "b0": float(b0),
                        "p_viol": float((b0 + n - k) / (a0 + b0 + n))})
            if n >= n_bind and lb >= lb_bind:
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
                ent["set"] = sorted(sset)
                ent["U"] = float(c["U"])
        table[_jx(x)] = ent
    N = sum(c["n"] for c in heavy.values())
    # set-bound sources (shared terminals) are explained by their sets: they
    # count toward the coverage of the dependency but not toward g3
    set_mass = sum(heavy[x]["mass"] for x, e in zip(heavy, table.values()) if e.get("set"))
    fx = [x for x, e in zip(heavy, table.values()) if not e.get("set")]
    Nf = sum(heavy[x]["n"] for x in fx)
    g3 = 1.0 - sum(tops[x][1] for x in fx) / Nf if Nf > 0 else 1.0
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
    return {"fd": {"g3": float(g3), "n": float(N), "holds": holds,
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
