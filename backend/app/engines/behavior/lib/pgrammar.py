"""Payload grammar induction of the progressive core (docs/lib3/progressive.md §6.11).

STATUS: implemented (W-P4, P07 maths). Pure functions over pnode summaries; no
store access, no state. Nothing is pre-set: every length range, class and key
set is an observed quantity of the node's current confidence segment, published
with its coverage and unseen mass.

Text attributes (pnode.TextSummary: shapes SS(8), exact values SS(16), length
histogram, charset-class counts)
    fit_text(ts, t)      1 group the tracked shapes by skeleton (class / literal
                            sequence without run lengths, lib/phier.skeleton)
                         2 <= 3 skeletons covering >= 99 % of mass -> their alternation
                            with per-run length ranges (anti-unification), skeletons
                            sharing a prefix factored into prefix + optional suffix:
                            L4, L3, L5 . L1 -> [a-z]{3,5}(\\.[a-z])?
                         3 otherwise the charset-class union and the total length range
                         4 closed value set when the exact-value SS covers >= 99 % of
                            mass, its unseen mass U <= 0.01 and n_c >= 50 (policy clear
                            / hmac): p(new value) = U
    published: grammar (regex), charset, len [lo, hi] with len_cover = 2 / (n + 1),
    c_g (mass the grammar covers), U_s = (N1_shapes + E + 0.5) / (N + 1), closed / U.
    A class token A (mixed alphanumeric run > 8, lib/phier) becomes the class union
    the charset counts actually saw (e.g. [a-z0-9]), [A-Za-z0-9] when unknown
    (shape-only policy).
Set attributes (pnode.SetSummary: templates SS, element presence SS)
    fit_set(ss, t)       required R = {k : presence >= 0.99}, optional 0.01 <= presence < 0.99,
                         p(new key) = (N1_keys + E + 0.5) / (N + 1),
                         p(missing required k) = (misses_k + 0.5) / (N + 1)
Categorical attributes (pnode.CatSummary)
    fit_cat(cs, t)       closed value set as for text values
Scoring (P03): check_text / check_set / check_cat -> (p, flags); flags among
    'new_value', 'grammar', 'length', 'injection_shape', 'new_key', 'missing_key'.
    A value that fails the grammar scores U_s, a length outside the range the
    conformal rank 1 / (n + 1), a value outside a closed set U. NaN below n_min.
Operator pins (B23 `expected`, §6.11): {len_min, len_max} widen the published
    range (never narrow it): [a-z]{3,8} pinned at 10 renders [a-z]{3,10}.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .phier import Shaped, shape_length, skeleton

MAX_SKELETONS = 3
SKEL_COVER = 0.99
SHAPE_MIN_SHARE = 0.002           # shapes below this share of mass do not widen the grammar
REQUIRED = 0.99
OPTIONAL = 0.01
CLOSED_COVER = 0.99
CLOSED_U = 0.01
CLOSED_N = 50.0
N_MIN = 3.0                        # evidence needed before anything is published
N_SCORE = 20.0                     # support needed to score (§6.16.1)
DANGEROUS = frozenset("'\"`=<>;() ")
_CLS_RX = {"L": "a-z", "U": "A-Z", "D": "0-9"}
X_RX = "\\u0080-\\U0010ffff"
NAN = float("nan")

Token = Tuple[str, int]           # (class 'L'|'U'|'D'|'X'|'A' or a literal character, run length)


# ------------------------------------------------------------------ shapes
def parse_shape(shp: str) -> List[Token]:
    """lib/phier level-1 shape -> [(class or literal, run length)]."""
    out: List[Token] = []
    for tok in str(shp).split(" "):
        if not tok or tok == "E0":
            continue
        if tok[0] in "LUDXA" and tok[1:].isdigit():
            out.append((tok[0], int(tok[1:])))
            continue
        if len(tok) > 2 and tok.endswith("}") and "{" in tok[1:]:
            i = tok.index("{", 1)
            lit, n = tok[:i], tok[i + 1:-1]
            if n.isdigit():
                out.append((" " if lit == "SP" else lit, int(n)))
                continue
        out.append((" " if tok == "SP" else tok, 1))
    return out


def _key(tokens: Sequence[Token]) -> Tuple[str, ...]:
    return tuple(c for c, _ in tokens)


def _merge_runs(runs: Sequence[Sequence[Token]]) -> List[Tuple[str, int, int]]:
    """Anti-unification of token sequences with one skeleton: per-position
    (class, min run, max run)."""
    first = runs[0]
    out = [(c, n, n) for c, n in first]
    for r in runs[1:]:
        out = [(c, min(lo, n), max(hi, n)) for (c, lo, hi), (_, n) in zip(out, r)]
    return out


def _class_body(c: str, alnum: str) -> str:
    if c in _CLS_RX:
        return _CLS_RX[c]
    if c == "A":
        return alnum
    if c == "X":
        return X_RX
    return re.escape(c) if c not in "-]\\^[" else "\\" + c


def _atom(c: str, alnum: str) -> str:
    if c in "LUDAX":
        return "[" + _class_body(c, alnum) + "]"
    return re.escape(c) if c != " " else " "


def _quant(lo: int, hi: int) -> str:
    if lo == hi:
        return "" if lo == 1 else "{%d}" % lo
    return "{%d,%d}" % (lo, hi)


def _seq_rx(seq: Sequence[Tuple[str, int, int]], alnum: str) -> str:
    return "".join(_atom(c, alnum) + _quant(lo, hi) for c, lo, hi in seq)


def _group_by_key(items: Sequence[Sequence[Tuple[str, int, int]]]
                  ) -> List[List[Tuple[str, int, int]]]:
    by: Dict[Tuple[str, ...], List[Tuple[str, int, int]]] = {}
    order: List[Tuple[str, ...]] = []
    for it in items:
        k = tuple(c for c, _, _ in it)
        if k not in by:
            by[k] = list(it)
            order.append(k)
        else:
            by[k] = [(c, min(a, x), max(b, y)) for (c, a, b), (_, x, y) in zip(by[k], it)]
    return [by[k] for k in order]


def _build(items: Sequence[Sequence[Tuple[str, int, int]]], alnum: str) -> str:
    """Regex of a set of anti-unified skeletons with prefix factoring."""
    items = _group_by_key(items)
    nonempty = [it for it in items if it]
    if not nonempty:
        return ""
    has_empty = len(nonempty) < len(items)
    # longest common class prefix of the non-empty items
    m = min(len(it) for it in nonempty)
    k = 0
    while k < m and len({it[k][0] for it in nonempty}) == 1:
        k += 1
    if k == 0:
        if len(nonempty) == 1:
            body = _seq_rx(nonempty[0], alnum)
        else:
            body = "(" + "|".join(_seq_rx(it, alnum) for it in nonempty) + ")"
        return "(" + body + ")?" if has_empty else body
    prefix = [(nonempty[0][i][0], min(it[i][1] for it in nonempty),
               max(it[i][2] for it in nonempty)) for i in range(k)]
    rest = [it[k:] for it in nonempty]
    tail = _build(rest, alnum)
    rx = _seq_rx(prefix, alnum)
    if tail:
        if all(r for r in rest):
            rx += tail
        else:
            rx += tail if tail.endswith(")?") else "(" + tail + ")?"
    return "(" + rx + ")?" if has_empty else rx


def alnum_class(chars: Optional[np.ndarray]) -> str:
    """Class body for an A run from TextSummary charset counts
    (index 0 L, 1 U, 2 D); [A-Za-z0-9] when nothing was counted."""
    if chars is None or len(chars) < 3 or float(np.sum(chars[:3])) <= 0:
        return "A-Za-z0-9"
    parts = [b for i, b in ((1, "A-Z"), (0, "a-z"), (2, "0-9")) if chars[i] > 0]
    return "".join(parts) if parts else "A-Za-z0-9"


def charset_rx(classes: Iterable[str], lo: int, hi: int, alnum: str) -> str:
    cl = set(classes)
    if "A" in cl:                     # L / U / D already inside the A class
        cl -= {c for c, r in _CLS_RX.items() if r in alnum}
    body = ""
    for c in sorted(cl, key=lambda x: ("LUDAX".find(x) if x in "LUDAX" else 9, x)):
        body += _class_body(c, alnum)
    return "[" + body + "]" + ("{%d}" % lo if lo == hi else "{%d,%d}" % (lo, hi))


def instance(shp: str) -> str:
    """A concrete string of a shape (for matching shape-only values)."""
    out = []
    for c, n in parse_shape(shp):
        ch = {"L": "a", "U": "A", "D": "0", "X": "é"}.get(c)
        if c == "A":
            out.append(("a0" * n)[:n])
        elif ch is not None:
            out.append(ch * n)
        else:
            out.append(c * n)
    return "".join(out)


# --------------------------------------------------------------------- fits
def fit_text(ts: Any, t: float, n_min: float = N_MIN, closed_n: float = CLOSED_N,
             closed_u: float = CLOSED_U, pin: Optional[Mapping[str, Any]] = None
             ) -> Optional[Dict[str, Any]]:
    """Grammar of a pnode.TextSummary (§6.11); None below n_min evidence."""
    ss = ts.shapes
    N = ss.total_evidence(t)
    tot = ss.total(t)
    if N < n_min or tot <= 0:
        return None
    raw = ss.items(t)
    items = [(k, g) for k, _, g, _ in raw if g > 0]
    if not items:
        # all tracked counts are inherited errors (a churning, e.g. random,
        # value space): classes and lengths come from the upper-bound counts
        items = [(k, c) for k, c, _, _ in raw if c > 0]
    if not items:
        return None
    chars = None
    try:
        chars = ts.chars.read(t)
    except Exception:
        chars = None
    alnum = alnum_class(chars)
    by_skel: Dict[str, float] = {}
    for k, g in items:
        sk = skeleton(k)
        by_skel[sk] = by_skel.get(sk, 0.0) + g
    ranked = sorted(by_skel.items(), key=lambda kv: -kv[1])
    chosen, cum = [], 0.0
    for sk, g in ranked[:MAX_SKELETONS]:
        chosen.append(sk)
        cum += g
        if cum >= SKEL_COVER * tot:
            break
    rec: Dict[str, Any] = {"kind": "text", "n": float(N), "U_s": float(ss.unseen(t)),
                           "mass": float(tot)}
    if cum >= SKEL_COVER * tot:
        inc = [(k, g) for k, g in items if skeleton(k) in chosen]
        top_of = {}
        for k, g in inc:
            sk = skeleton(k)
            if sk not in top_of or g > top_of[sk][1]:
                top_of[sk] = (k, g)
        inc = [(k, g) for k, g in inc if g / tot >= SHAPE_MIN_SHARE or top_of[skeleton(k)][0] == k]
        seqs = {}
        for k, g in inc:
            toks = parse_shape(k)
            seqs.setdefault(_key(toks), []).append(toks)
        anti = [_merge_runs(v) for v in seqs.values()]
        rx = _build(anti, alnum)
        mode = "shape"
        c_g = sum(g for _, g in inc) / tot
        lens = [shape_length(k) for k, _ in inc]
        classes = {c for a in anti for c, _, _ in a}
    else:
        inc = [(k, g) for k, g in items if g / tot >= SHAPE_MIN_SHARE] or items
        lens = [shape_length(k) for k, _ in inc]
        classes = {c for k, _ in inc for c, _ in parse_shape(k)}
        lo, hi = (min(lens), max(lens)) if lens else (0, 0)
        # untracked mass: widen the length range to the length histogram's occupied buckets
        un = max(0.0, 1.0 - sum(g for _, g in inc) / tot)
        if un > 0.01:
            try:
                h = ts.lens.read(t)
                occ = [b for b in range(len(h)) if h[b] > 0.001 * max(h.sum(), 1e-12)]
                if occ:
                    lo = min(lo, max(0, 2 ** min(occ) - 1))
                    hi = max(hi, 2 ** (max(occ) + 1) - 2)
            except Exception:
                pass
        lens = [lo, hi]
        rx = charset_rx(classes, lo, hi, alnum) if classes else ""
        mode = "charset"
        c_g = min(1.0, sum(g for _, g in inc) / tot)   # tracked mass: a lower bound
    lo, hi = (int(min(lens)), int(max(lens))) if lens else (0, 0)
    if pin:
        plo = pin.get("len_min")
        phi = pin.get("len_max")
        nlo = min(lo, int(plo)) if plo is not None else lo
        nhi = max(hi, int(phi)) if phi is not None else hi
        if (nlo, nhi) != (lo, hi):
            single = mode == "shape" and len(anti) == 1 and len(anti[0]) == 1
            if single:
                c = anti[0][0][0]
                rx = _atom(c, alnum) + _quant(nlo, nhi)
            else:
                rx = charset_rx(classes, nlo, nhi, alnum)
                mode = "charset"
            lo, hi = nlo, nhi
            rec["pinned"] = {k: pin[k] for k in ("len_min", "len_max") if k in pin}
    rec.update({"grammar": rx, "mode": mode, "charset": sorted(classes), "alnum": alnum,
                "len": [lo, hi], "len_cover": 2.0 / (N + 1.0), "c_g": float(c_g),
                "skeletons": chosen if mode == "shape" else []})
    vs = getattr(ts, "values", None)
    if vs is not None:
        rec.update(_closed(vs, t, closed_n, closed_u))
    rec["confidence"] = float(rec["c_g"] * (1.0 - rec["U_s"]))
    return rec


def _closed(vs: Any, t: float, closed_n: float, closed_u: float) -> Dict[str, Any]:
    vtot = vs.total(t)
    nv = vs.total_evidence(t)
    if vtot <= 0:
        return {}
    its = [(k, g) for k, _, g, _ in vs.items(t) if g > 0]
    cov = sum(g for _, g in its) / vtot
    U = float(vs.unseen(t))
    out: Dict[str, Any] = {"n_values": float(nv), "U": U, "value_cover": float(cov),
                           "top": [[_jv(k), float(g / vtot)] for k, g in its[:8]]}
    if cov >= CLOSED_COVER and U <= closed_u and nv >= closed_n:
        out["closed"] = sorted(_jv(k) for k, _ in its)
    return out


def _jv(k: Any) -> Any:
    return k if isinstance(k, (str, int, float, bool)) else str(k)


def fit_cat(cs: Any, t: float, n_min: float = N_MIN, closed_n: float = CLOSED_N,
            closed_u: float = CLOSED_U) -> Optional[Dict[str, Any]]:
    """Closed value set of a categorical target (pnode.CatSummary)."""
    ss = cs.ss
    N = ss.total_evidence(t)
    if N < n_min or ss.total(t) <= 0:
        return None
    rec = {"kind": "cat", "n": float(N)}
    rec.update(_closed(ss, t, closed_n, closed_u))
    rec["confidence"] = float(1.0 - rec.get("U", 1.0)) if "closed" in rec else 0.0
    return rec


def fit_set(ss: Any, t: float, n_min: float = N_MIN) -> Optional[Dict[str, Any]]:
    """Required / optional keys of a set target (pnode.SetSummary)."""
    N = ss.tpl.total_evidence(t)
    if N < n_min:
        return None
    pres = ss.presence(t)
    if not pres:
        return None
    req = sorted(str(k) for k, p in pres.items() if p >= REQUIRED)
    opt = sorted(str(k) for k, p in pres.items() if OPTIONAL <= p < REQUIRED)
    el = ss.elem
    p_new = min(1.0, (el.n1(t) + el.eviction_evidence(t) + 0.5) / (N + 1.0))
    miss = {}
    for k in req:
        nk = el.evidence(k, t)
        miss[k] = float(min(1.0, (max(0.0, N - nk) + 0.5) / (N + 1.0)))
    conf = (1.0 - p_new) * (1.0 - max(miss.values()) if miss else 1.0)
    return {"kind": "set", "n": float(N), "required": req, "optional": opt,
            "presence": {str(k): float(p) for k, p in pres.items()},
            "p_new_key": float(p_new), "p_missing": miss, "confidence": float(conf)}


def material_change(old: Optional[Mapping[str, Any]], new: Mapping[str, Any]) -> bool:
    """A grammar, key set or closed set changed (§6.8.2 cver rule)."""
    if not old:
        return True
    for k in ("grammar", "required", "closed"):
        if old.get(k) != new.get(k):
            return True
    return False


# ------------------------------------------------------------------ scoring
_RX_CACHE: Dict[str, Any] = {}


def _compiled(rx: str) -> Any:
    c = _RX_CACHE.get(rx)
    if c is None:
        try:
            c = re.compile(rx)
        except re.error:
            c = False
        if len(_RX_CACHE) > 4096:
            _RX_CACHE.clear()
        _RX_CACHE[rx] = c
    return c


def _allowed(rec: Mapping[str, Any], ch: str) -> bool:
    cls = rec.get("charset") or []
    if "a" <= ch <= "z":
        return "L" in cls or ("A" in cls and "a-z" in rec.get("alnum", ""))
    if "A" <= ch <= "Z":
        return "U" in cls or ("A" in cls and "A-Z" in rec.get("alnum", ""))
    if "0" <= ch <= "9":
        return "D" in cls or ("A" in cls and "0-9" in rec.get("alnum", ""))
    return ch in cls


def injection_shape(rec: Mapping[str, Any], v: str) -> bool:
    """The value carries a dangerous character class (quote, '=', space, '<',
    ';', '--', ...) that the grammar's charset never contained."""
    if "--" in v and "-" not in (rec.get("charset") or []):
        return True
    return any(ch in DANGEROUS and not _allowed(rec, ch) for ch in v)


def check_text(rec: Mapping[str, Any], v: Any, n_min: float = N_SCORE) -> Tuple[float, List[str]]:
    """(p, flags) of a text value against a fitted grammar / closed set."""
    flags: List[str] = []
    if rec is None or not (float(rec.get("n", 0.0)) >= n_min):
        return NAN, flags
    ps: List[float] = []
    shaped = isinstance(v, Shaped)
    s = str(v)
    n = float(rec["n"])
    if "closed" in rec and not shaped and s not in set(map(str, rec["closed"])):
        ps.append(float(rec.get("U", 0.5 / (n + 1.0))))
        flags.append("new_value")
    rx = rec.get("grammar")
    probe = instance(s) if shaped else s
    if rx:
        c = _compiled(rx)
        if c and not c.fullmatch(probe):
            ps.append(float(rec.get("U_s", 0.5 / (n + 1.0))))
            flags.append("grammar")
            if not shaped and injection_shape(rec, s):
                flags.append("injection_shape")
    ln = shape_length(s) if shaped else len(s)
    lo, hi = rec.get("len", [None, None])
    if lo is not None and (ln < lo or ln > hi):
        ps.append(1.0 / (n + 1.0))
        flags.append("length")
    return (float(min(ps)) if ps else 1.0), flags


def check_set(rec: Mapping[str, Any], v: Any, n_min: float = N_SCORE) -> Tuple[float, List[str]]:
    flags: List[str] = []
    if rec is None or not (float(rec.get("n", 0.0)) >= n_min):
        return NAN, flags
    try:
        keys = {str(x) for x in v}
    except TypeError:
        return NAN, flags
    ps = []
    known = set(rec.get("presence", {}))
    if keys - known:
        ps.append(float(rec["p_new_key"]))
        flags.append("new_key")
    for k in rec.get("required", []):
        if k not in keys:
            ps.append(float(rec["p_missing"].get(k, 0.5)))
            flags.append("missing_key")
            break
    return (float(min(ps)) if ps else 1.0), flags


def check_cat(rec: Mapping[str, Any], v: Any, n_min: float = N_SCORE) -> Tuple[float, List[str]]:
    if rec is None or not (float(rec.get("n", 0.0)) >= n_min) or "closed" not in rec:
        return NAN, []
    if _jv(v) in set(rec["closed"]) or str(v) in set(map(str, rec["closed"])):
        return 1.0, []
    return float(rec.get("U", 1.0)), ["new_value"]


# --------------------------------------------------------------------- gain
def shape_gain(node_ss: Any, sys_ss: Any, t: float, alpha: float = 0.5) -> float:
    """Bits per event saved by coding the node's level-1 keys (shapes) with
    the node distribution instead of the system's (plug-in KL, smoothed)."""
    if sys_ss is None or node_ss.total(t) <= 0 or sys_ss.total(t) <= 0:
        return 0.0
    keys, sh, other = node_ss.distribution(t)
    skeys, ssh, sother = sys_ss.distribution(t)
    sp = {k: float(s) for k, s in zip(skeys, ssh)}
    K = len(skeys) + 1
    g = 0.0
    for k, p in zip(keys, sh):
        if p <= 0:
            continue
        q = (sp.get(k, 0.0) + alpha / (K * 100.0)) / (1.0 + alpha / 100.0)
        g += float(p) * math.log2(float(p) / max(q, 1e-12))
    return float(max(0.0, g))
