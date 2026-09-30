"""Progressive-core evaluation gates PG1-PG11 (docs/lib3/progressive.md §12).

Two stages, like metrics.py:

  score_prun(run)        one RunResult of an org pack (O, O60, O-real*, O-red, ...)
                         -> a JSON-able dict of per-run measurements
  compute_pgates(scores) per-run dicts (packs x seeds) -> the gates, each
                         {name, value, target, pass, details}, medians over seeds
                         with bootstrap 95 % CIs; `pass` None when not computable

What the scorer reads. Only (a) the truth the generator published
(run.truth rows A*/D*/R*, run.ptruth: pattern / group / strategy / system /
attribute truth, opportunities, who log) and (b) what the engines published:
the end-of-day P-model snapshots the runner took (run.psnaps: model.pviews,
model.pbind, model.pwin, model.sysprof, model.attr, model.attrsel and, on full
days, model.ptree and the fitted models; model.who_groups, model.sysfam,
model.facets; group views), discrete events (pattern_violation,
pattern_absent, attribute_new/gone, binding_changed, ...), incidents and the
behavior.p series. Nothing reads the engines' internals; held-out and contrast
samples are drawn from the truth program (orggen.sample_value & co.).

Statement contract (what P14 must publish; the scorer's only view of a learned
pattern). `model.pviews@(s,'__system__')` holds statement objects (§6.17.2),
anywhere in the model (they are collected recursively: any dict with an
`evidence` dict and a `pattern_id` or `text_zh`). The machine-readable part is
`evidence`:

  route: 'POST /login' (or method + route; host tokens and {num}/{id}-style
         placeholders are normalised), system (default: the view's key),
  context: [[attr, level, values, negated], ...], depth: int, is_exc: bool,
  who: {level: 'ip'|'grp'|'prefix'|'reg'|'any', items: [ip | cidr | 'grp:<g>' |
        'reg:<name>' | 'shared:<ip>' | '*'], members: [ips], U, closed, confidence},
  when: {workday: [[m0, m1], ...], nonworkday: [...], coverage, confidence},
  content: {attr: {band90, range, n_rng, cover, approx, grammar (regex or
            {regex}), c_g, U_s, required | required_keys, closed | closed_values,
            U, p99, observed, confidence}},
  bindings: {attr: {x: 'net.src' | 'sess.key', table: {x: y | [y, ...]},
             LB: {x: lb}, reverse: {y: [x, ...]}}},
  workflow: [{from: 'GET /report/form', to: 'POST /report/generate', dep, band: [s0, s1]}],
  negative: bool, target_system: str (group-view negative statements),
  hint: str (configuration hints such as the SNAT one).

Statement top level: state, confidence, support, view, subject, text_zh,
text_en, pattern_id. Minutes are local minutes of day, sizes bytes.
"""
from __future__ import annotations

import copy
import ipaddress
import json
import math
import re
from typing import Any, Callable, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

from ..pipeline import orggen as OG
from .metrics import (SEV_RANK, _median, bootstrap_ci, check, gate, ks_uniform, sev_rank,
                      top_features)

try:
    from ..engines.behavior.lib.detectors import DETECTORS
except Exception:  # pragma: no cover
    DETECTORS = []

CONFIRMED = frozenset({"confirmed", "stable"})
RENDERED = frozenset({"confirmed", "stable", "evolving", "stale"})
_PH = re.compile(r"\{[^}]*\}")
_NUMSEG = re.compile(r"(?<=/)\d+(?=/|$)")
_METHODS = {"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS", "TLS"}


# --------------------------------------------------------------------------- #
# Small helpers
# --------------------------------------------------------------------------- #
def _f(v: Any, default: float = math.nan) -> float:
    try:
        x = float(v)
        return x
    except (TypeError, ValueError):
        return default


def norm_route(method: Any, route: Any = None) -> Tuple[str, str]:
    """('POST', '/approval/{}/approve') from 'POST oa.corp.local /approval/{num}/approve',
    ('POST', '/approval/{id}/approve') or similar."""
    if route is None:
        parts = str(method or "").split()
        m = parts[0].upper() if parts and parts[0].upper() in _METHODS else ""
        rest = parts[1:] if m else parts
        route = rest[-1] if rest else ""
        method = m
    r = str(route or "").strip()
    if str(method).upper() != "TLS":
        r = _PH.sub("{}", r)
        r = _NUMSEG.sub("{}", r)
        if len(r) > 1:
            r = r.rstrip("/")
    return str(method or "").upper(), r


def _intervals(v: Any) -> List[Tuple[float, float]]:
    out = []
    for x in v or []:
        try:
            a, b = float(x[0]), float(x[1])
        except (TypeError, ValueError, IndexError):
            continue
        if b > a:
            out.append((a, b))
    return sorted(out)


def _union_len(iv: Sequence[Tuple[float, float]]) -> float:
    tot, cur = 0.0, None
    for a, b in sorted(iv):
        if cur is None or a > cur[1]:
            if cur is not None:
                tot += cur[1] - cur[0]
            cur = [a, b]
        else:
            cur[1] = max(cur[1], b)
    if cur is not None:
        tot += cur[1] - cur[0]
    return tot


def window_iou(a: Sequence, b: Sequence) -> Optional[float]:
    """IoU of two interval sets on the minute line; None when both are empty."""
    A, B = _intervals(a), _intervals(b)
    if not A and not B:
        return None
    inter = []
    for x0, x1 in A:
        for y0, y1 in B:
            lo, hi = max(x0, y0), min(x1, y1)
            if hi > lo:
                inter.append((lo, hi))
    u = _union_len(A + B)
    return _union_len(inter) / u if u > 0 else 0.0


def _in_windows(minute: float, iv: Sequence) -> bool:
    return any(a <= minute < b for a, b in _intervals(iv))


def jaccard(a: Iterable[Any], b: Iterable[Any]) -> float:
    A, B = set(a), set(b)
    if not A and not B:
        return 1.0
    return len(A & B) / len(A | B)


def _net(s: str) -> Optional[Any]:
    try:
        return ipaddress.ip_network(str(s), strict=False)
    except ValueError:
        return None


def _ip(s: str) -> Optional[Any]:
    try:
        return ipaddress.ip_address(str(s))
    except ValueError:
        return None


def ari(labels_true: Sequence[Any], labels_pred: Sequence[Any]) -> Optional[float]:
    """Adjusted Rand index (Hubert & Arabie 1985)."""
    n = len(labels_true)
    if n < 2 or n != len(labels_pred):
        return None
    ct: Dict[Tuple[Any, Any], int] = {}
    for a, b in zip(labels_true, labels_pred):
        ct[(a, b)] = ct.get((a, b), 0) + 1
    comb = lambda x: x * (x - 1) / 2.0
    sum_ij = sum(comb(v) for v in ct.values())
    rows: Dict[Any, int] = {}
    cols: Dict[Any, int] = {}
    for (a, b), v in ct.items():
        rows[a] = rows.get(a, 0) + v
        cols[b] = cols.get(b, 0) + v
    sa = sum(comb(v) for v in rows.values())
    sb = sum(comb(v) for v in cols.values())
    exp = sa * sb / comb(n)
    mx = 0.5 * (sa + sb)
    if mx == exp:
        return 1.0
    return float((sum_ij - exp) / (mx - exp))


def loglog_slope(xs: Sequence[float], ys: Sequence[float]) -> Optional[float]:
    x = np.asarray([v for v, w in zip(xs, ys) if v and w and v > 0 and w > 0], dtype=float)
    y = np.asarray([w for v, w in zip(xs, ys) if v and w and v > 0 and w > 0], dtype=float)
    if x.size < 2 or np.ptp(np.log(x)) == 0:
        return None
    return float(np.polyfit(np.log(x), np.log(y), 1)[0])


def ece(conf: Sequence[float], obs: Sequence[float], bins: int = 10) -> Optional[float]:
    """Expected calibration error of stated confidences against observed
    outcomes (1 = the statement held on held-out events, 0 = it did not)."""
    c = np.asarray(conf, dtype=float)
    o = np.asarray(obs, dtype=float)
    ok = np.isfinite(c) & np.isfinite(o)
    c, o = np.clip(c[ok], 0, 1), np.clip(o[ok], 0, 1)
    if c.size == 0:
        return None
    idx = np.minimum((c * bins).astype(int), bins - 1)
    tot = 0.0
    for b in range(bins):
        m = idx == b
        if m.any():
            tot += m.sum() / c.size * abs(float(c[m].mean()) - float(o[m].mean()))
    return float(tot)


# --------------------------------------------------------------------------- #
# Regex sampling (for grammar containment checks)
# --------------------------------------------------------------------------- #
def _parse_class(rx: str, i: int) -> Tuple[str, int]:
    """[...] at rx[i] -> (alphabet, index after ']')."""
    j = i + 1
    neg = j < len(rx) and rx[j] == "^"
    if neg:
        j += 1
    chars: List[str] = []
    first = True
    while j < len(rx) and (rx[j] != "]" or first):
        first = False
        c = rx[j]
        if c == "\\" and j + 1 < len(rx):
            nxt = rx[j + 1]
            chars += list({"d": "0123456789", "w": "abcdefghijklmnopqrstuvwxyz0123456789_",
                           "s": " "}.get(nxt, nxt))
            j += 2
            continue
        if j + 2 < len(rx) and rx[j + 1] == "-" and rx[j + 2] != "]":
            chars += [chr(x) for x in range(ord(c), ord(rx[j + 2]) + 1)]
            j += 3
            continue
        chars.append(c)
        j += 1
    alpha = "".join(sorted(set(chars)))
    if neg:
        alpha = "".join(ch for ch in map(chr, range(32, 127)) if ch not in alpha)
    return alpha, j + 1


def sample_regex(rx: str, r: np.random.Generator, n: int, max_rep: int = 12) -> List[str]:
    """Strings drawn from a regex of the P07 subset: literals, escapes, [..]
    classes, groups with |, quantifiers ? * + {m} {m,n}. Repetition counts are
    drawn from the extremes half of the time (bounds are what matters)."""
    rx = rx.strip()
    if rx.startswith("^"):
        rx = rx[1:]
    if rx.endswith("$") and not rx.endswith("\\$"):
        rx = rx[:-1]

    def parse_seq(i: int) -> Tuple[List[Any], int]:
        alts: List[List[Any]] = [[]]
        while i < len(rx) and rx[i] != ")":
            c = rx[i]
            if c == "|":
                alts.append([])
                i += 1
                continue
            if c == "(":
                j = i + 1
                if rx[j:j + 2] == "?:":
                    j += 2
                node, i = parse_seq(j)
                i += 1
                atom: Any = ("grp", node)
            elif c == "[":
                alpha, i = _parse_class(rx, i)
                atom = ("cls", alpha)
            elif c == "\\" and i + 1 < len(rx):
                nxt = rx[i + 1]
                atom = ("cls", {"d": "0123456789", "w": "abcdefghijklmnopqrstuvwxyz0123456789_",
                                "s": " "}.get(nxt, nxt))
                i += 2
            elif c == ".":
                atom = ("cls", "".join(map(chr, range(33, 127))))
                i += 1
            else:
                atom = ("cls", c)
                i += 1
            lo, hi = 1, 1
            if i < len(rx) and rx[i] in "?*+{":
                q = rx[i]
                if q == "?":
                    lo, hi, i = 0, 1, i + 1
                elif q == "*":
                    lo, hi, i = 0, max_rep, i + 1
                elif q == "+":
                    lo, hi, i = 1, max_rep, i + 1
                else:
                    k = rx.index("}", i)
                    body = rx[i + 1:k]
                    if "," in body:
                        a, b = body.split(",", 1)
                        lo, hi = int(a or 0), int(b) if b else max_rep
                    else:
                        lo = hi = int(body)
                    i = k + 1
                if i < len(rx) and rx[i] in "?+":
                    i += 1
            alts[-1].append((atom, lo, hi))
        return [("alt", alts)], i

    tree, _ = parse_seq(0)

    def gen(node: List[Any]) -> str:
        out = []
        for item in node:
            if item[0] == "alt":
                alts = item[1]
                seq = alts[int(r.integers(0, len(alts)))]
                for atom, lo, hi in seq:
                    if r.random() < 0.5:
                        k = int(lo if r.random() < 0.5 else hi)
                    else:
                        k = int(r.integers(lo, hi + 1))
                    for _ in range(k):
                        if atom[0] == "cls":
                            a = atom[1]
                            out.append(a[int(r.integers(0, len(a)))] if a else "")
                        else:
                            out.append(gen(atom[1]))
        return "".join(out)

    return [gen(tree) for _ in range(n)]


def regex_contained(rx: str, outer: str, r: np.random.Generator, n: int = 2000) -> Optional[bool]:
    """True when every sampled string of `rx` fullmatches `outer`."""
    try:
        samples = sample_regex(rx, r, n)
        o = re.compile(outer)
    except (ValueError, re.error, IndexError):
        return None
    return all(o.fullmatch(s) for s in samples)


def _grammar_rx(g: Any) -> Optional[str]:
    if g is None:
        return None
    if isinstance(g, Mapping):
        g = g.get("regex")
    return str(g) if g else None


# --------------------------------------------------------------------------- #
# Learned statements (the P14 contract above)
# --------------------------------------------------------------------------- #
def _collect(obj: Any, out: List[Dict[str, Any]], depth: int = 0) -> None:
    if depth > 12:
        return
    if isinstance(obj, Mapping):
        if isinstance(obj.get("evidence"), Mapping) and ("pattern_id" in obj or "text_zh" in obj):
            out.append(dict(obj))
            return
        for v in obj.values():
            _collect(v, out, depth + 1)
    elif isinstance(obj, (list, tuple)):
        for v in obj:
            _collect(v, out, depth + 1)


class Who:
    __slots__ = ("level", "ips", "members", "prefixes", "groups", "regions", "shared", "any",
                 "U", "closed", "confidence")

    def __init__(self, d: Mapping[str, Any], ip_classes: Mapping[str, List[str]]) -> None:
        d = d or {}
        self.level = str(d.get("level") or "").lower() or None
        self.ips: Set[str] = set()
        self.members: Set[str] = {str(x) for x in d.get("members") or []}
        self.prefixes: List[Any] = []
        self.groups: Set[str] = set()
        self.regions: Set[str] = set()
        self.shared: Set[str] = set()
        self.any = self.level in ("any", "none", "*")
        for it in d.get("items") or []:
            s = str(it[0] if isinstance(it, (list, tuple)) else it)
            if s in ("*", "any"):
                self.any = True
            elif s.startswith("shared:"):
                self.shared.add(s[7:])
                self.ips.add(s[7:])
            elif s.startswith("grp:"):
                self.groups.add(s[4:])
            elif s.startswith("reg:"):
                self.regions.add(s[4:])
                for c in ip_classes.get(s[4:], []):
                    n = _net(c)
                    if n is not None:
                        self.prefixes.append(n)
            elif "/" in s:
                n = _net(s)
                if n is not None:
                    self.prefixes.append(n)
            elif _ip(s) is not None:
                self.ips.add(s)
        self.U = _f(d.get("U"))
        self.closed = bool(d.get("closed", not self.any and bool(self.ips or self.members)))
        self.confidence = _f(d.get("confidence"))

    def ipset(self) -> Set[str]:
        return (self.ips | self.members) - set()

    def contains(self, ip: str) -> bool:
        if self.any:
            return True
        if ip in self.ips or ip in self.members:
            return True
        a = _ip(ip)
        return a is not None and any(a in p for p in self.prefixes)


class LStmt:
    """A normalised learned statement."""

    def __init__(self, raw: Mapping[str, Any], system: str,
                 ip_classes: Mapping[str, List[str]]) -> None:
        ev = raw.get("evidence") or {}
        self.raw = raw
        self.system = str(ev.get("system") or raw.get("system") or system)
        rt = str(ev.get("route") or "").split()
        if ev.get("method") and rt and rt[0].upper() not in _METHODS:
            self.method, self.route = norm_route(ev.get("method"), ev.get("route"))
        else:
            self.method, self.route = norm_route(ev.get("route") or ev.get("act") or "")
        self.state = str(raw.get("state") or ev.get("state") or "").lower()
        self.confidence = _f(raw.get("confidence"))
        self.support = _f(raw.get("support"))
        self.view = str(raw.get("view") or "system")
        self.text_zh = str(raw.get("text_zh") or "")
        self.text_en = str(raw.get("text_en") or "")
        self.pattern_id = str(raw.get("pattern_id") or raw.get("id") or "")
        self.who = Who(ev.get("who") or {}, ip_classes)
        w = ev.get("when") or {}
        self.when = {"workday": _intervals(w.get("workday")),
                     "nonworkday": _intervals(w.get("nonworkday"))}
        self.when_cov = _f(w.get("coverage", w.get("confidence")))
        self.content: Dict[str, Dict[str, Any]] = {str(k): dict(v) for k, v in
                                                   (ev.get("content") or {}).items()
                                                   if isinstance(v, Mapping)}
        self.bindings: Dict[str, Dict[str, Any]] = {str(k): dict(v) for k, v in
                                                    (ev.get("bindings") or {}).items()
                                                    if isinstance(v, Mapping)}
        self.workflow = [dict(e) for e in ev.get("workflow") or [] if isinstance(e, Mapping)]
        self.context = list(ev.get("context") or [])
        self.depth = _f(ev.get("depth"))
        self.is_exc = bool(ev.get("is_exc"))
        self.negative = bool(ev.get("negative"))
        self.target_system = str(ev.get("target_system") or "")
        self.hint = str(ev.get("hint") or "")

    @property
    def confirmed(self) -> bool:
        return self.state in CONFIRMED

    def binding_table(self, attr: str) -> Dict[str, Set[str]]:
        b = self.bindings.get(attr) or {}
        out: Dict[str, Set[str]] = {}
        for x, y in (b.get("table") or {}).items():
            if isinstance(y, Mapping):
                y = y.get("y", y.get("value", y.get("values")))
            vals = {str(v) for v in y} if isinstance(y, (list, tuple, set)) else {str(y)}
            out[str(x)] = vals
        return out


def ip_classes_of(config: Mapping[str, Any]) -> Dict[str, List[str]]:
    out: Dict[str, List[str]] = {}
    for c in (config or {}).get("ip_classes") or []:
        if isinstance(c, Mapping) and c.get("name"):
            out[str(c["name"])] = [str(x) for x in c.get("cidrs") or []]
    for c in (config or {}).get("dhcp_scopes") or []:
        if isinstance(c, Mapping) and c.get("name"):
            out[str(c["name"])] = [str(c.get("cidr"))]
    return out


def statements(snap: Mapping[str, Any], ip_classes: Mapping[str, List[str]],
               families: Optional[Mapping[str, List[str]]] = None) -> List[LStmt]:
    """Every system-view statement of one snapshot (group views excluded)."""
    out: List[LStmt] = []
    for key, rec in (snap.get("systems") or {}).items():
        raw: List[Dict[str, Any]] = []
        _collect(rec.get("model.pviews"), raw)
        for st in raw:
            ls = LStmt(st, key, ip_classes)
            if ls.view == "group":
                continue
            out.append(ls)
    return out


def group_statements(snap: Mapping[str, Any], ip_classes: Mapping[str, List[str]]
                     ) -> Dict[str, List[LStmt]]:
    out: Dict[str, List[LStmt]] = {}
    for key, pv in (snap.get("group_views") or {}).items():
        raw: List[Dict[str, Any]] = []
        _collect(pv, raw)
        out[key] = [LStmt(st, str(key), ip_classes) for st in raw]
    return out


# --------------------------------------------------------------------------- #
# Truth access
# --------------------------------------------------------------------------- #
class PTruth:
    def __init__(self, ptruth: Mapping[str, Any]) -> None:
        self.raw = ptruth or {}
        self.rows: List[Dict[str, Any]] = list(self.raw.get("pattern_truth") or [])
        self.by_tid = {r["tid"]: r for r in self.rows}
        days = self.raw.get("days") or {}
        self.day_start: List[float] = [float(x) for x in days.get("day_start") or []]
        self.workday: List[bool] = list(days.get("workday") or [])
        self.n_days = int(days.get("n_days") or max(0, len(self.day_start) - 1))
        import datetime as _dt
        self.start = _dt.date.fromisoformat(days["start"]) if days.get("start") else None
        self.opp = self.raw.get("opportunities") or {}
        self.groups = self.raw.get("group_truth") or {}
        self.strategy = self.raw.get("strategy_truth") or {}
        self.systems = self.raw.get("system_truth") or {}
        self.attrs = list(self.raw.get("attr_truth") or [])
        self.who_log = self.raw.get("who_log") or {}

    def day_end(self, d: int) -> float:
        return self.day_start[d] if d < len(self.day_start) else math.inf

    def day_index(self, iso: str) -> int:
        import datetime as _dt
        return (_dt.date.fromisoformat(iso) - self.start).days + 1 if self.start else 0

    def day_of_ts(self, ts: float) -> int:
        import bisect
        return bisect.bisect_right(self.day_start, ts)

    def valid_at_day(self, d: int) -> List[Dict[str, Any]]:
        return [r for r in self.rows if int(r["valid_from_day"]) <= d < int(r["valid_to_day"])]

    def opportunities(self, tid: str, upto_day: int, from_day: int = 1) -> Tuple[int, int]:
        n, dates = 0, 0
        for iso, per in (self.opp.get(tid) or {}).items():
            di = self.day_index(iso)
            if from_day <= di <= upto_day:
                c = sum(int(v) for v in per.values())
                if c:
                    n += c
                    dates += 1
        return n, dates

    def lineage_opps(self, row: Mapping[str, Any], upto_day: int) -> Tuple[int, int]:
        return self.opportunities(row["tid"], upto_day, int(row["valid_from_day"]))

    def workdays_between(self, d0: int, d1: int) -> int:
        """Workdays in [d0, d1] (1-based days)."""
        return sum(1 for d in range(max(1, d0), min(d1, self.n_days) + 1)
                   if d - 1 < len(self.workday) and self.workday[d - 1])

    def family_of(self, system: str) -> Set[str]:
        fams = (self.systems.get("families") or {})
        out = {system}
        for fam, mem in fams.items():
            if system == fam or system in mem:
                out |= set(mem) | {fam}
        return out


def _sys_match(stmt_sys: str, row: Mapping[str, Any], pt: PTruth) -> bool:
    if stmt_sys == row["system"] or stmt_sys in (row.get("systems") or []):
        return True
    return stmt_sys in pt.family_of(row["system"]) or stmt_sys.startswith("fam:") and (
        row["system"] in pt.family_of(stmt_sys[4:]) or stmt_sys[4:] in pt.family_of(row["system"]))


def _route_of(row: Mapping[str, Any]) -> Tuple[str, str]:
    return norm_route(row["method"], row["route"])


# --------------------------------------------------------------------------- #
# PG1 component checks
# --------------------------------------------------------------------------- #
def who_compatible(truth_who: Mapping[str, Any], w: Who) -> bool:
    lvl = truth_who.get("level")
    if lvl == "ip":
        truth = set(truth_who.get("value") or [])
        return jaccard(w.ipset(), truth) >= 0.8
    if lvl == "grp":
        truth = set(truth_who.get("members") or [])
        return jaccard(w.ipset(), truth) >= 0.8
    if lvl in ("prefix", "reg"):
        truth = [(_net(c), float(s)) for c, s in truth_who.get("value") or []]
        truth = [(n, s) for n, s in truth if n is not None]
        if w.level == "grp" and "grp" in (truth_who.get("alt") or []) and w.members:
            inside = sum(1 for ip in w.members if any(_ip(ip) in n for n, _ in truth))
            return inside / len(w.members) >= 0.9
        return prefix_cover_ok(truth, w.prefixes)
    if lvl == "any":
        return w.any or len(w.regions) >= 3 or len(w.prefixes) >= 3
    return False


def prefix_cover_ok(truth: List[Tuple[Any, float]], learned: List[Any],
                    cover: float = 0.9, outside: float = 0.2) -> bool:
    if not truth or not learned:
        return False
    tot = sum(s for _, s in truth) or 1.0
    covered = 0.0
    for n, s in truth:
        inside = sum(min(n.num_addresses, p.num_addresses) for p in learned
                     if p.version == n.version and (p.subnet_of(n) or n.subnet_of(p)))
        covered += s * min(1.0, inside / n.num_addresses)
    out_space, all_space = 0.0, 0.0
    for p in learned:
        all_space += p.num_addresses
        ins = sum(min(n.num_addresses, p.num_addresses) for n, _ in truth
                  if n.version == p.version and (p.subnet_of(n) or n.subnet_of(p)))
        out_space += max(0.0, p.num_addresses - ins)
    return covered / tot >= cover and (out_space / all_space if all_space else 1.0) <= outside


def when_compatible(row: Mapping[str, Any], s: LStmt, thr: float = 0.7) -> bool:
    for dt in row.get("daytypes") or []:
        iou = window_iou((row.get("windows") or {}).get(dt) or [], s.when.get(dt) or [])
        if iou is None or iou < thr:
            return False
    return True


def _rel_ok(a: float, b: float, tol: float) -> bool:
    return abs(a - b) <= tol * max(abs(b), 1e-9)


def _grammar_ok(rx: Optional[str], tcon: Mapping[str, Any], row: Mapping[str, Any],
                r: np.random.Generator, n: int = 1000) -> bool:
    if not rx:
        return False
    try:
        cre = re.compile(rx)
    except re.error:
        return False
    attr_key = None
    fields = (row.get("gen") or {}).get("fields") or {}
    for k in fields:
        if f"body.kv.{k}" == tcon.get("_attr"):
            attr_key = k
    spec = fields.get(attr_key) if attr_key else None
    vals = tcon.get("closed_values")
    if spec is not None:
        fresh = OG.truth_values(spec, r, n, vals)
    elif vals:
        fresh = [vals[int(i)] for i in r.integers(0, len(vals), n)]
    else:
        return False
    if not fresh:
        return False
    acc = np.mean([bool(cre.fullmatch(str(v))) for v in fresh])
    g = tcon.get("grammar")
    if not g:
        return acc >= 0.99
    contrast = OG.contrast_values(g, r, n)
    rej = np.mean([not cre.fullmatch(v) for v in contrast])
    return bool(acc >= 0.99 and rej >= 0.99)


def content_matches(row: Mapping[str, Any], s: LStmt, r: np.random.Generator,
                    core_only: bool = True) -> Tuple[bool, Dict[str, bool]]:
    detail: Dict[str, bool] = {}
    for attr, tc in (row.get("content") or {}).items():
        if core_only and not tc.get("core", False):
            continue
        lc = s.content.get(attr)
        ok = lc is not None
        if ok and "band90" in tc:
            b = lc.get("band90")
            ok = bool(b) and _rel_ok(_f(b[0]), tc["band90"][0], 0.2) and \
                _rel_ok(_f(b[1]), tc["band90"][1], 0.2)
            if ok and "range" in tc and lc.get("range"):
                rg = lc["range"]
                ok = _rel_ok(_f(rg[0]), tc["range"][0], 0.25) and \
                    _rel_ok(_f(rg[1]), tc["range"][1], 0.25)
            elif ok and "range" in tc:
                ok = False
        if ok and "required_keys" in tc:
            ok = set(map(str, lc.get("required") or lc.get("required_keys") or [])) == \
                set(tc["required_keys"])
        if ok and tc.get("kind") in ("bound", "choice", "const") and "closed_values" in tc:
            ok = set(map(str, lc.get("closed") or lc.get("closed_values") or [])) == \
                set(map(str, tc["closed_values"]))
        if ok and tc.get("kind") == "bound" and tc.get("grammar"):
            ok = _grammar_ok(_grammar_rx(lc.get("grammar")), dict(tc, _attr=attr), row, r)
        detail[attr] = bool(ok)
    return all(detail.values()) if detail else True, detail


def bindings_match(row: Mapping[str, Any], s: LStmt) -> Tuple[bool, int, int]:
    tot, hit = 0, 0
    for attr, table in (row.get("bindings") or {}).items():
        lt = s.binding_table(attr)
        for x, y in table.items():
            tot += 1
            want = set(y) if isinstance(y, list) else {str(y)}
            if lt.get(x) == want:
                hit += 1
    return (tot == 0 or hit / tot >= 0.95), hit, tot


def _edges(stmts: Sequence[LStmt]) -> List[Tuple[str, Tuple[str, str], Tuple[str, str], float,
                                                  Tuple[float, float]]]:
    out = []
    for s in stmts:
        for e in s.workflow:
            a, b = norm_route(e.get("from")), norm_route(e.get("to"))
            band = e.get("band") or [math.nan, math.nan]
            out.append((s.system, a, b, _f(e.get("dep")), (_f(band[0]), _f(band[1]))))
    return out


def workflow_match(row: Mapping[str, Any], edges: Sequence, pt: PTruth) -> bool:
    for frm, to, band in row.get("workflow") or []:
        fr = pt.by_tid.get(frm)
        if fr is None:
            continue
        a, b = _route_of(fr), _route_of(row)
        ok = False
        for sys_, ea, eb, dep, eband in edges:
            if ea == a and eb == b and _sys_match(sys_, row, pt) and dep >= 0.8 and \
                    eband[0] <= band[1] and eband[1] >= band[0]:
                ok = True
                break
        if not ok:
            return False
    return True


def recover(row: Mapping[str, Any], stmts: Sequence[LStmt], edges: Sequence, pt: PTruth,
            r: np.random.Generator) -> Dict[str, Any]:
    """Whether truth pattern `row` is recovered by one confirmed statement,
    plus per-component recoveries (who, when, content, bindings, workflow)."""
    route = _route_of(row)
    cands = [s for s in stmts if s.confirmed and (s.method, s.route) == route
             and _sys_match(s.system, row, pt)]
    comp = {"who": False, "when": False, "content": False,
            "bindings": None if not row.get("bindings") else False,
            "workflow": None if not row.get("workflow") else False}
    wf_ok = workflow_match(row, edges, pt) if row.get("workflow") else True
    if row.get("workflow"):
        comp["workflow"] = wf_ok
    best = None
    for s in cands:
        w = who_compatible(row["who"], s.who)
        t = when_compatible(row, s)
        c, _ = content_matches(row, s, r)
        b, _, _ = bindings_match(row, s)
        comp["who"] |= w
        comp["when"] |= t
        comp["content"] |= c
        if comp["bindings"] is not None:
            comp["bindings"] |= b
        if w and t and c and b and wf_ok:
            best = s
            break
    return {"recovered": best is not None, "components": comp, "n_cands": len(cands),
            "stmt": best.pattern_id if best is not None else None}


# --------------------------------------------------------------------------- #
# Held-out sampling (PG1 precision, PG2 calibration)
# --------------------------------------------------------------------------- #
def _pick_ip(row: Mapping[str, Any], r: np.random.Generator, restrict: Optional[Who]) -> Optional[str]:
    gen = row.get("gen") or {}
    mem = gen.get("members")
    if mem:
        ips = [ip for ip in mem if restrict is None or restrict.contains(ip)]
        if not ips:
            return None
        return ips[int(r.integers(0, len(ips)))]
    cidrs = gen.get("regions") or ([[gen["prefix"], 1.0]] if gen.get("prefix") else [])
    if not cidrs:
        return None
    for _ in range(20):
        w = np.asarray([float(s) for _, s in cidrs])
        c = cidrs[int(r.choice(len(cidrs), p=w / w.sum()))][0]
        n = ipaddress.ip_network(c, strict=False)
        ip = str(n.network_address + int(r.integers(1, max(2, n.num_addresses - 1))))
        if restrict is None or restrict.contains(ip):
            return ip
    return None


def holdout_events(rows: Sequence[Mapping[str, Any]], r: np.random.Generator, n: int,
                   restrict: Optional[Who] = None) -> List[Dict[str, Any]]:
    """Fresh events from the truth program of `rows` (equal weight per row)."""
    out: List[Dict[str, Any]] = []
    if not rows:
        return out
    tries = 0
    while len(out) < n and tries < 4 * n:
        tries += 1
        row = rows[int(r.integers(0, len(rows)))]
        ip = _pick_ip(row, r, restrict)
        if ip is None:
            continue
        wins = row.get("windows") or {}
        dts = [d for d in ("workday", "nonworkday") if wins.get(d)]
        if not dts:
            continue
        lens = [sum(b - a for a, b in wins[d]) for d in dts]
        dt = dts[int(r.choice(len(dts), p=np.asarray(lens) / sum(lens)))]
        iv = wins[dt]
        L = np.asarray([b - a for a, b in iv], dtype=float)
        a, b = iv[int(r.choice(len(iv), p=L / L.sum()))]
        gen = row.get("gen") or {}
        ev: Dict[str, Any] = {"ip": ip, "daytype": dt, "minute": float(r.uniform(a, b)),
                              "tid": row["tid"]}
        if gen.get("size"):
            ev["body.len"] = OG.sample_size(gen["size"], r)
        if gen.get("up"):
            ev["net.bytes_up"] = OG.sample_size(gen["up"], r) + 300
        users = gen.get("usernames") or {}
        uname = users.get(ip)
        if uname is None and users:
            vals = list(users.values())
            uname = vals[int(r.integers(0, len(vals)))]
        fields = gen.get("fields") or {}
        keys = set(fields)
        if gen.get("size") and gen.get("fmt"):
            keys.add("viewstate" if gen.get("fmt") == "form" else "remark")
        if fields:
            ev["body.keys"] = keys
        for k, spec in fields.items():
            ev[f"body.kv.{k}"] = str(OG.sample_value(spec, r, {"username": uname or ""}))
        out.append(ev)
    return out


def _tol(nominal: float, m: int) -> float:
    nominal = min(max(nominal, 0.0), 1.0)
    return max(0.02, 3.0 * math.sqrt(max(nominal * (1 - nominal), 1e-4) / max(m, 1)))


def holdout_check(s: LStmt, rows_valid: Sequence[Mapping[str, Any]], pt: PTruth,
                  r: np.random.Generator, n: int = 400) -> Dict[str, Any]:
    """Every constraint of a statement against held-out events of its context."""
    rows = [row for row in rows_valid if _route_of(row) == (s.method, s.route)
            and _sys_match(s.system, row, pt)]
    ctx_who = any(str(c[0]).startswith("net.src") for c in s.context if isinstance(c, (list, tuple)) and c)
    restrict = s.who if (ctx_who or s.is_exc) and (s.who.ipset() or s.who.prefixes) else None
    evs = holdout_events(rows, r, n, restrict)
    conf = s.confidence if math.isfinite(s.confidence) else 0.9
    res: List[Tuple[str, float, float, bool]] = []
    if not evs:
        return {"supported": False, "ok": False, "constraints": [], "min_hold": 0.0}

    def add(name: str, nominal: float, hits: List[bool]) -> None:
        if not hits:
            return
        h = float(np.mean(hits))
        nom = nominal if math.isfinite(nominal) else conf
        res.append((name, nom, h, h >= nom - _tol(nom, len(hits))))

    if s.who.closed and not s.who.any and (s.who.ipset() or s.who.prefixes):
        nom = 1.0 - s.who.U if math.isfinite(s.who.U) else conf
        add("who", nom, [s.who.contains(e["ip"]) for e in evs])
    for dt in ("workday", "nonworkday"):
        sub = [e for e in evs if e["daytype"] == dt]
        if s.when.get(dt) and sub:
            add(f"when.{dt}", s.when_cov if math.isfinite(s.when_cov) else conf,
                [_in_windows(e["minute"], s.when[dt]) for e in sub])
    for attr, c in s.content.items():
        vals = [e.get(attr) for e in evs if e.get(attr) is not None]
        if not vals:
            continue
        if c.get("band90"):
            lo, hi = _f(c["band90"][0]), _f(c["band90"][1])
            add(f"{attr}.band90", _f(c.get("coverage"), 0.9), [lo <= v <= hi for v in vals])
        if c.get("range"):
            lo, hi = _f(c["range"][0]), _f(c["range"][1])
            nom = 1.0 - _f(c.get("cover")) if math.isfinite(_f(c.get("cover"))) else conf
            add(f"{attr}.range", nom, [lo <= v <= hi for v in vals])
        rx = _grammar_rx(c.get("grammar"))
        if rx:
            try:
                cre = re.compile(rx)
                cg, us = _f(c.get("c_g")), _f(c.get("U_s"))
                nom = cg * (1 - us) if math.isfinite(cg) and math.isfinite(us) else conf
                add(f"{attr}.grammar", nom, [bool(cre.fullmatch(str(v))) for v in vals])
            except re.error:
                res.append((f"{attr}.grammar", conf, 0.0, False))
        req = c.get("required") or c.get("required_keys")
        if req and attr.endswith(".keys"):
            add(f"{attr}.required", 0.99, [set(map(str, req)) <= set(v) for v in vals])
        closed = c.get("closed") or c.get("closed_values")
        if closed:
            u = _f(c.get("U"))
            add(f"{attr}.closed", 1.0 - u if math.isfinite(u) else conf,
                [str(v) in set(map(str, closed)) for v in vals])
    for attr in s.bindings:
        table = s.binding_table(attr)
        lbs = (s.bindings[attr].get("LB") or {})
        for x, ys in table.items():
            sub = [e for e in evs if e["ip"] == x and e.get(attr) is not None]
            if sub:
                add(f"{attr}.bind.{x}", _f(lbs.get(x), conf), [str(e[attr]) in ys for e in sub])
    holds = [h for _, _, h, _ in res]
    return {"supported": True, "ok": all(ok for *_, ok in res), "constraints": res,
            "min_hold": float(min(holds)) if holds else 1.0}


# --------------------------------------------------------------------------- #
# PG1 / PG2 per snapshot
# --------------------------------------------------------------------------- #
def eligible(row: Mapping[str, Any], pt: PTruth, day: int) -> bool:
    if not (int(row["valid_from_day"]) <= day < int(row["valid_to_day"])):
        return False
    if row.get("idle"):
        return False
    n, dates = pt.lineage_opps(row, day)
    return n >= 20 and dates >= 3


def pg1_snapshot(snap: Mapping[str, Any], pt: PTruth, ip_classes: Mapping[str, List[str]],
                 day: int, seed: int = 0, systems: Optional[Set[str]] = None,
                 precision_n: int = 300) -> Dict[str, Any]:
    stmts = statements(snap, ip_classes)
    if systems is not None:
        stmts = [s for s in stmts if s.system in systems]
    edges = _edges(stmts)
    r = np.random.default_rng([seed, day, 1])
    rows = [row for row in pt.rows if eligible(row, pt, day)
            and (systems is None or row["system"] in systems)]
    per: Dict[str, Any] = {}
    for row in rows:
        per[row["tid"]] = recover(row, stmts, edges, pt, r)
        per[row["tid"]]["period"] = row.get("period")
    comps = {}
    for c in ("who", "when", "content", "bindings", "workflow"):
        vals = [p["components"][c] for p in per.values() if p["components"][c] is not None]
        comps[c] = float(np.mean(vals)) if vals else None
    recall = float(np.mean([p["recovered"] for p in per.values()])) if per else None
    by_period: Dict[str, Optional[float]] = {}
    for per_ in ("daily", "weekly", "continuous", "sporadic", "monthly"):
        v = [p["recovered"] for p in per.values() if p["period"] == per_]
        by_period[per_] = float(np.mean(v)) if v else None
    # precision + calibration over rendered confirmed statements
    valid = pt.valid_at_day(day)
    rp = np.random.default_rng([seed, day, 2])
    prec_hits, conf, obs = [], [], []
    for s in stmts:
        if not s.confirmed or s.negative:
            continue
        h = holdout_check(s, valid, pt, rp, precision_n)
        prec_hits.append(bool(h["ok"]))
        if math.isfinite(s.confidence):
            conf.append(s.confidence)
            obs.append(1.0 if h["ok"] else 0.0)
    depths = [s.depth for s in stmts if s.confirmed and math.isfinite(s.depth)]
    return {
        "day": day, "n_truth": len(rows), "n_stmt": len(stmts),
        "n_confirmed": sum(1 for s in stmts if s.confirmed),
        "recall": recall, "components": comps, "recall_by_period": by_period,
        "precision": float(np.mean(prec_hits)) if prec_hits else None,
        "ece": ece(conf, obs), "mean_depth": float(np.mean(depths)) if depths else None,
        "per_pattern": {k: {"recovered": v["recovered"], "components": v["components"],
                            "period": v["period"], "stmt": v["stmt"]} for k, v in per.items()},
        "stmt_conf": {k: _stmt_conf(stmts, pt.by_tid[k]) for k, v in per.items() if v["recovered"]},
        "who_U": {k: _stmt_U(stmts, v["stmt"]) for k, v in per.items() if v["recovered"]},
    }


def _stmt_conf(stmts: Sequence[LStmt], row: Mapping[str, Any]) -> Optional[float]:
    route = _route_of(row)
    c = [s.confidence for s in stmts if s.confirmed and (s.method, s.route) == route
         and who_compatible(row["who"], s.who) and math.isfinite(s.confidence)]
    return max(c) if c else None


def _stmt_U(stmts: Sequence[LStmt], pid: Optional[str]) -> Optional[float]:
    for s in stmts:
        if s.pattern_id == pid and s.who.closed and math.isfinite(s.who.U):
            return s.who.U
    return None


def pg2_convergence(pg1: Mapping[int, Mapping[str, Any]], pt: PTruth,
                    drift_days: Tuple[int, int] = (12, 15)) -> Dict[str, Any]:
    days = sorted(pg1)
    rec = {d: pg1[d].get("recall") for d in days}
    viol = []
    prev = None
    for d in days:
        if drift_days[0] <= d <= drift_days[1]:
            continue
        if prev is not None and rec[prev] is not None and rec[d] is not None \
                and rec[d] < rec[prev] - 0.05:
            viol.append([prev, d])
        prev = d
    ttr: Dict[str, Optional[int]] = {}
    for period in ("daily", "weekly"):
        ttr[period] = next((d for d in days if (pg1[d].get("recall_by_period") or {}).get(period)
                            is not None and pg1[d]["recall_by_period"][period] >= 0.8), None)
    dep = [(d, pg1[d].get("mean_depth")) for d in days if d < drift_days[0]
           and pg1[d].get("mean_depth") is not None]
    depth_ok = all(b[1] >= a[1] - 1e-9 for a, b in zip(dep, dep[1:])) if len(dep) >= 2 else None
    unchanged = {r["tid"] for r in pt.rows if int(r["valid_from_day"]) == 1
                 and int(r["valid_to_day"]) > pt.n_days}
    med_conf = [(d, _median(v for k, v in (pg1[d].get("stmt_conf") or {}).items()
                            if k in unchanged)) for d in days]
    med_conf = [(d, v) for d, v in med_conf if v is not None]
    conf_ok = all(b[1] >= a[1] - 1e-9 for a, b in zip(med_conf, med_conf[1:])) \
        if len(med_conf) >= 2 else None
    med_u = [(d, _median((pg1[d].get("who_U") or {}).values())) for d in days]
    med_u = [(d, v) for d, v in med_u if v is not None]
    u_ok = all(b[1] <= a[1] + 1e-9 for a, b in zip(med_u, med_u[1:])) if len(med_u) >= 2 else None
    eces = [pg1[d].get("ece") for d in days if pg1[d].get("ece") is not None]
    return {"recall": rec, "precision": {d: pg1[d].get("precision") for d in days},
            "recall_violations": viol, "recall_monotone": (not viol) if rec else None,
            "days_to_80": ttr, "depth_nondecreasing": depth_ok,
            "median_conf": med_conf, "conf_nondecreasing": conf_ok,
            "median_U": med_u, "U_nonincreasing": u_ok,
            "ece": eces[-1] if eces else None}


def false_splits(snaps: Mapping[int, Mapping[str, Any]], pt: PTruth,
                 ip_classes: Mapping[str, List[str]]) -> Dict[str, Any]:
    """Learned splits on attributes independent of every truth constraint
    (attr_truth cls 'independent'), from statement contexts and ptree node
    contexts, per system-month."""
    indep = {a["name"] for a in pt.attrs if a.get("cls") == "independent"}
    if not indep or not snaps:
        return {"n": None, "per_system_month": None}
    found: Set[Tuple[str, str]] = set()
    for snap in snaps.values():
        for s in statements(snap, ip_classes):
            for c in s.context:
                if isinstance(c, (list, tuple)) and c and str(c[0]) in indep:
                    found.add((s.system, s.pattern_id))
        for key, rec in (snap.get("systems") or {}).items():
            for nid, ctx in _ptree_contexts(rec.get("model.ptree")):
                if any(str(c[0]) in indep for c in ctx if isinstance(c, (list, tuple)) and c):
                    found.add((key, str(nid)))
    n_sys = len({r["system"] for r in pt.rows}) or 1
    months = max(pt.n_days / 30.0, 1e-9)
    return {"n": len(found), "per_system_month": len(found) / n_sys / months}


def _ptree_contexts(tree: Any) -> List[Tuple[Any, List[Any]]]:
    out: List[Tuple[Any, List[Any]]] = []
    if not isinstance(tree, Mapping):
        return out
    kinds = tree.get("kinds") or {}
    for t in kinds.values():
        for nid, node in ((t or {}).get("nodes") or {}).items():
            if isinstance(node, Mapping):
                out.append((nid, list(node.get("ctx") or [])))
    return out


def ptree_node_count(tree: Any) -> Optional[int]:
    if not isinstance(tree, Mapping):
        return None
    return sum(len((t or {}).get("nodes") or {}) for t in (tree.get("kinds") or {}).values())


def login_bindings(snap: Mapping[str, Any], pt: PTruth, ipc: Mapping[str, List[str]], day: int,
                   activities: Sequence[str] = ("GA.oa.login", "FIN.finance.login")
                   ) -> Tuple[int, int]:
    """(matched, total) IP -> username binding pairs of the GA and FIN login
    patterns valid on `day` (PG1 target 'GA and FIN bindings 6/6'), each
    pattern against its best who-compatible confirmed statement."""
    stmts = statements(snap, ipc)
    hit = tot = 0
    for row in pt.valid_at_day(day):
        if row["activity"] not in activities or row["step"] != 0 or not row.get("bindings"):
            continue
        n = sum(len(t) for t in row["bindings"].values())
        best = 0
        for s in stmts:
            if s.confirmed and (s.method, s.route) == _route_of(row) and _sys_match(s.system, row, pt) \
                    and who_compatible(row["who"], s.who):
                best = max(best, bindings_match(row, s)[1])
        hit += best
        tot += n
    return hit, tot


# --------------------------------------------------------------------------- #
# PG3 specificity
# --------------------------------------------------------------------------- #
def _find(stmts: Sequence[LStmt], system: str, method: str, route: str) -> List[LStmt]:
    k = norm_route(method, route)
    return [s for s in stmts if s.confirmed and (s.method, s.route) == k and
            (s.system == system or s.system.startswith(system + "-") or s.system == "fam:" + system)]


def groups_from(who_groups: Any) -> Tuple[Dict[str, str], List[Any]]:
    """(ip -> group id, prefix covers) from model.who_groups (format-tolerant)."""
    ip2g: Dict[str, str] = {}
    covers: List[Any] = []
    if not isinstance(who_groups, Mapping):
        return ip2g, covers
    for ip, g in (who_groups.get("ip2g") or {}).items():
        ip2g[str(ip)] = str(g)
    groups = who_groups.get("groups") or {}
    items = groups.items() if isinstance(groups, Mapping) else enumerate(groups)
    for gid, g in items:
        if not isinstance(g, Mapping):
            continue
        gid = str(g.get("id", gid))
        for ip in g.get("members") or []:
            ip2g.setdefault(str(ip), gid)
        for c in g.get("covers") or g.get("prefixes") or []:
            n = _net(c[0] if isinstance(c, (list, tuple)) else c)
            if n is not None:
                covers.append((gid, n))
    return ip2g, covers


def pg3_specificity(snap: Mapping[str, Any], pt: PTruth, ip_classes: Mapping[str, List[str]]
                    ) -> Dict[str, Any]:
    stmts = statements(snap, ip_classes)
    out: Dict[str, Any] = {}
    g = pt.groups
    ga_row = next((r for r in pt.rows if r["activity"].startswith("GA.") and r["method"] == "POST"
                   and r["route"] == "/login" and int(r["valid_to_day"]) > pt.n_days), None)
    if ga_row is not None:
        truth = set(ga_row["who"]["value"])
        best = None
        for s in _find(stmts, "oa", "POST", "/login"):
            if jaccard(s.who.ipset(), truth) == 1.0:
                b = bindings_match(ga_row, s)
                best = max(best or 0, b[1])
        out["ga_login_who"] = best is not None
        out["ga_bindings"] = best
    fin_row = next((r for r in pt.rows if r["activity"].startswith("FIN.finance.approval")
                    and r["method"] == "POST"), None)
    if fin_row is not None:
        truth = set(fin_row["who"]["value"])
        out["fin_approval_who"] = any(s.who.ipset() == truth and not s.who.prefixes
                                      for s in _find(stmts, fin_row["system"], "POST", fin_row["route"]))
    portal = [s for s in stmts if s.system == "portal" and s.confirmed]
    pl = [s for s in portal if (s.method, s.route) == ("POST", "/login")]
    if pl:
        out["portal_login_level"] = all(s.who.level in ("prefix", "reg", "any", "none", None)
                                        and not (s.who.ips and len(s.who.ips) <= 8 and s.who.closed)
                                        for s in pl)
    else:
        out["portal_login_level"] = None
    out["portal_bindings"] = sum(len(s.binding_table(a)) for s in portal for a in s.bindings)
    n_pub = int((g.get("PUB") or {}).get("n") or 0)
    exc_ips = {ip for s in portal if s.is_exc for ip in s.who.ipset()}
    out["portal_exception_share"] = len(exc_ips) / n_pub if n_pub else None
    dev = g.get("DEV") or {}
    if dev.get("cidr"):
        pool = _net(dev["cidr"])
        bad = sum(1 for s in stmts if s.confirmed and any(
            _ip(ip) is not None and _ip(ip) in pool for ip in s.who.ips))
        code = [s for s in stmts if s.system == "code" and s.confirmed]

        def pool_level(w: Who) -> bool:
            if w.level == "grp" and w.members:
                return sum(1 for ip in w.members if _ip(ip) in pool) / len(w.members) >= 0.9
            return any(p == pool or (pool.subnet_of(p) and p.prefixlen >= pool.prefixlen - 2)
                       for p in w.prefixes if p.version == pool.version)
        out["dev_single_ip_patterns"] = bad
        out["dev_who_ok"] = (bad == 0 and all(pool_level(s.who) or s.who.level == "grp"
                                              for s in code)) if code else None
    ip2g, covers = groups_from((snap.get("org") or {}).get("model.who_groups"))
    if ip2g or covers:
        truth_lab, pred_lab = [], []
        for code_, rec in g.items():
            if rec.get("kind") != "static":
                continue
            for m in rec.get("members") or []:
                truth_lab.append(code_)
                pred_lab.append(ip2g.get(m["ip"], f"_single:{m['ip']}"))
        out["ari"] = ari(truth_lab, pred_lab)
        if dev.get("members"):
            pool_ips = {m["ip"] for m in dev["members"]}
            labs = [ip2g[ip] for ip in pool_ips if ip in ip2g]
            one_group = bool(labs) and max(labs.count(x) for x in set(labs)) / len(pool_ips) >= 0.9
            pool = _net(dev["cidr"])
            one_prefix = any(n == pool or (pool.subnet_of(n) and n.prefixlen >= pool.prefixlen - 2)
                             for _, n in covers)
            out["dev_pool_grouped"] = one_group or one_prefix
    else:
        out["ari"] = None
    return out


# --------------------------------------------------------------------------- #
# Events / incidents helpers
# --------------------------------------------------------------------------- #
def _pv(run: Any) -> List[Dict[str, Any]]:
    return [e for e in (getattr(run, "events", None) or []) if e.get("kind") == "pattern_violation"]


def _ev_type(e: Mapping[str, Any]) -> str:
    x = e.get("extra") or {}
    return str(x.get("type") or x.get("vtype") or "")


def _ev_flags(e: Mapping[str, Any]) -> Set[str]:
    x = e.get("extra") or {}
    f = x.get("flags") or []
    if isinstance(f, Mapping):
        f = [k for k, v in f.items() if v]
    return {str(v) for v in f}


def _inc_max_sev_between(inc: Mapping[str, Any], t0: float, t1: float) -> int:
    best = -1
    for h in inc.get("history") or []:
        if t0 <= _f(h.get("ts")) <= t1:
            best = max(best, sev_rank(h.get("severity")))
    if not inc.get("history") and t0 <= _f(inc.get("opened")) <= t1:
        best = max(best, sev_rank(inc.get("severity")))
    return best


def _incidents_on(run: Any, system: str, ips: Iterable[str]) -> List[Dict[str, Any]]:
    ips = set(ips)
    out = []
    for inc in getattr(run, "incidents", None) or []:
        ents = {inc.get("entity")} | set(inc.get("entities") or [])
        if inc.get("system") == system and ents & ips:
            out.append(inc)
    return out


# --------------------------------------------------------------------------- #
# PG5 drift adaptation
# --------------------------------------------------------------------------- #
def _stmts_by_day(snaps: Mapping[int, Mapping[str, Any]], ipc: Mapping[str, List[str]]
                  ) -> Dict[int, List[LStmt]]:
    return {int(d): statements(s, ipc) for d, s in snaps.items()}


def pg5_drift(run: Any, pt: PTruth, sbd: Mapping[int, List[LStmt]]) -> Dict[str, Any]:
    truth = {r["scenario_id"]: r for r in getattr(run, "truth", None) or []}
    days = sorted(sbd)
    out: Dict[str, Any] = {}
    d1 = truth.get("D1")
    if d1 and d1.get("tids_new"):
        new = pt.by_tid.get(sorted(d1["tids_new"])[0])
        old_tid = next((t for t in d1.get("tids_old") or [] if t.endswith("#0") or "#0@" in t), None)
        old = pt.by_tid.get(old_tid) if old_tid else None
        day0 = int(d1["day"])
        conf_day = gone_day = None
        for d in days:
            if d < day0:
                continue
            ga = [s for s in sbd[d] if s.confirmed and (s.method, s.route) == _route_of(new)
                  and who_compatible(new["who"], s.who)]
            if conf_day is None and any((window_iou(new["windows"]["workday"],
                                                    s.when["workday"]) or 0) >= 0.7 for s in ga):
                conf_day = d
            if gone_day is None and old is not None and not any(
                    (window_iou(old["windows"]["workday"], s.when["workday"]) or 0) >= 0.7 for s in ga):
                gone_day = d
        ips = d1.get("entities") or []
        t_end = pt.day_end(conf_day) if conf_day else pt.day_end(pt.n_days)
        n_inc = sum(1 for inc in _incidents_on(run, "oa", ips)
                    if _inc_max_sev_between(inc, pt.day_start[day0 - 1], t_end) >= SEV_RANK["low"])
        out["D1"] = {"confirm_workdays": pt.workdays_between(day0, conf_day) if conf_day else None,
                     "gone_workdays": pt.workdays_between(day0, gone_day) if gone_day else None,
                     "incidents_low": n_inc,
                     "pass": (conf_day is not None and pt.workdays_between(day0, conf_day) <= 3
                              and gone_day is not None and pt.workdays_between(day0, gone_day) <= 5
                              and n_inc <= 1) if days else None}
    d2 = truth.get("D2")
    if d2:
        ip = d2["entities"][0]
        new_v = (d2.get("params") or {}).get("value")
        day0 = int(d2["day"])
        lin = [r for r in pt.rows if r.get("bindings") and ip in
               ((r["bindings"].get("body.kv.username") or {}))]
        ev_by_day: Dict[int, int] = {}
        for r in lin:
            if int(r["valid_from_day"]) < day0 or r["method"] != "POST":
                continue
            for iso, per in (pt.opp.get(r["tid"]) or {}).items():
                c = int(per.get(ip, 0))
                if c:
                    di = pt.day_index(iso)
                    ev_by_day[di] = ev_by_day.get(di, 0) + c
        need_day, cum, dates = None, 0, 0
        for di in sorted(ev_by_day):
            cum += ev_by_day[di]
            dates += 1
            if cum >= 5 and dates >= 2:
                need_day = di
                break
        sw_day = None
        for d in days:
            if d < day0:
                continue
            if any(new_v in s.binding_table("body.kv.username").get(ip, set()) and
                   len(s.binding_table("body.kv.username").get(ip, set())) == 1
                   for s in sbd[d] if s.confirmed):
                sw_day = d
                break
        ev_at = sum(v for k, v in ev_by_day.items() if sw_day is not None and k <= sw_day)
        out["D2"] = {"switch_day": sw_day, "events_at_switch": ev_at if sw_day else None,
                     "needed_day": need_day,
                     "pass": (sw_day is not None and need_day is not None and sw_day <= need_day)
                     if days else None}
    d3 = truth.get("D3")
    if d3 and d3.get("tids_new"):
        day0 = int(d3["day"])
        new_rows = [pt.by_tid[t] for t in d3["tids_new"] if t in pt.by_tid]
        old_rows = [pt.by_tid[t] for t in d3.get("tids_old") or [] if t in pt.by_tid]
        conf_day = next((d for d in days if d >= day0 and all(
            any(s.confirmed and (s.method, s.route) == _route_of(r) for s in sbd[d])
            for r in new_rows)), None)
        stale_day = next((d for d in days if d >= day0 and not any(
            s.confirmed and (s.method, s.route) == _route_of(r) for r in old_rows
            for s in sbd[d])), None)
        cw = pt.workdays_between(day0, conf_day) if conf_day else None
        sw = pt.workdays_between(day0, stale_day) if stale_day else None
        out["D3"] = {"confirm_workdays": cw, "stale_workdays": sw,
                     "pass": (cw is not None and cw <= 5 and sw is not None and sw <= 3)
                     if days else None}
    for did, system in (("D4", "portal"), ("D5", None)):
        dr = truth.get(did)
        if not dr:
            continue
        t0, t1 = float(dr["t_start"]), float(dr["t_end"])
        if did == "D4":
            ips = {ip for iso, per in (pt.who_log.get("portal") or {}).items() for ip in per}
            incs = _incidents_on(run, "portal", ips)
        else:
            cidr = _net((pt.groups.get("DEV") or {}).get("cidr") or "0.0.0.0/32")
            incs = [i for i in getattr(run, "incidents", None) or []
                    if _ip(str(i.get("entity"))) is not None and _ip(str(i.get("entity"))) in cidr]
        n = sum(1 for i in incs if _inc_max_sev_between(i, t0, t1) >= SEV_RANK["low"])
        out[did] = {"incidents_low": n, "pass": n == 0}
    a2 = truth.get("A2")
    if a2 and a2.get("non_adoption") and days:
        na = a2["non_adoption"]
        d = max(x for x in days if x <= int(na["day"])) if any(x <= int(na["day"]) for x in days) else None
        if d is not None:
            vals = set()
            for s in sbd[d]:
                if s.confirmed:
                    vals |= s.binding_table(na["attr"]).get(na["ip"], set())
            out["non_adoption_binding"] = {"day": d, "values": sorted(vals),
                                           "pass": vals == {na["value"]}}
    a7 = truth.get("A7")
    if a7 and days:
        pre_d = max((x for x in days if pt.day_end(x) <= float(a7["t_start"])), default=None)
        post_d = days[-1]

        def p99(d: int) -> Optional[float]:
            for s in sbd[d]:
                if s.system == "portal" and s.confirmed and (s.method, s.route) == ("POST", "/login"):
                    c = s.content.get("rate.ip_h") or {}
                    v = c.get("p99")
                    if v is None and c.get("band98"):
                        v = c["band98"][1]
                    if v is not None:
                        return _f(v)
            return None
        a, b = (p99(pre_d) if pre_d else None), p99(post_d)
        out["rate_p99"] = {"pre": a, "post": b,
                           "pass": (abs(b - a) <= 0.1 * a) if (a and b is not None) else None}
    return out


# --------------------------------------------------------------------------- #
# PG6 anomalies and FAR
# --------------------------------------------------------------------------- #
def pg6_anomalies(run: Any, pt: PTruth, sbd: Mapping[int, List[LStmt]]) -> Dict[str, Any]:
    dt = float(getattr(run, "scenario_dt", 900.0) or 900.0)
    pvs = _pv(run)
    out: Dict[str, Any] = {}
    for row in getattr(run, "truth", None) or []:
        sid = str(row.get("scenario_id"))
        if row.get("label") != "malicious" or not sid.startswith("A"):
            continue
        ents = set(row.get("entities") or [])
        t_first = float(row.get("t_first", row["t_start"]))
        tick_end = t_first + dt - ((t_first - (pt.day_start[0] if pt.day_start else 0.0)) % dt)
        exp_types = set(row.get("expected_types") or [])
        exp_flags = set(row.get("expected_flags") or [])
        hits = [e for e in pvs if e.get("system") == row["system"] and e.get("entity") in ents
                and t_first <= float(e["ts"]) <= tick_end + dt]
        typed = [e for e in hits if _ev_type(e) in exp_types or (_ev_flags(e) & exp_flags)]
        dl = max(4 * dt, 3600.0)
        req = sev_rank(row.get("required_severity", "low"))
        incs = _incidents_on(run, row["system"], ents)
        inc_ok = any(_inc_max_sev_between(i, float(row["t_start"]), t_first + dl) >= req for i in incs)
        tp = [i for i in incs if _inc_max_sev_between(i, float(row["t_start"]), t_first + dl) >= req]
        top_ok = None
        if tp:
            pf = {str(x) for x in row.get("perturbed_features") or []} | \
                 {f"conf_{t}" for t in exp_types}
            tops = top_features(tp[0].get("explanation") or {}, 1)
            top_ok = bool(tops) and any(t in pf or any(p in t for p in pf) for t in tops)
        rec = {"violation": bool(typed), "incident": inc_ok, "detected": bool(typed) and inc_ok,
               "flags": sorted({f for e in hits for f in _ev_flags(e)}),
               "types": sorted({_ev_type(e) for e in hits}), "top_reason_ok": top_ok}
        if row.get("check_days"):
            later = int(max(row["check_days"]))
            t0, t1 = pt.day_start[later - 1], pt.day_end(later)
            rec["who_on_last_day"] = any(e.get("system") == row["system"] and e.get("entity") in ents
                                         and t0 <= float(e["ts"]) < t1 and _ev_type(e) == "who"
                                         for e in pvs)
            hn = row.get("heavy_node") or {}
            d = max((x for x in sbd if x <= later), default=None)
            if d is not None and hn:
                nodes = [s for s in sbd[d] if s.system == hn.get("system") and
                         (s.method, s.route) == norm_route(hn.get("method"), hn.get("route"))]
                rec["outside_heavy_set"] = (not any(ents & s.who.ipset() for s in nodes)) if nodes else None
        out[sid] = rec
    return out


def pg6_far(run: Any, pt: PTruth) -> Dict[str, Any]:
    """FAR per clean entity-day: entities in no anomaly / perturbation row,
    entity-days = (system, ip, date) with traffic in the who log."""
    dirty: Set[str] = set()
    for r in getattr(run, "truth", None) or []:
        if r.get("label") == "malicious" or str(r.get("scenario_id", "")).startswith("R"):
            dirty |= set(map(str, r.get("entities") or []))
            for x in (r.get("params") or {}).values():
                if isinstance(x, str) and _ip(x) is not None:
                    dirty.add(x)
    ed = 0
    days_of: Dict[Tuple[str, str], Set[str]] = {}
    for s, per_date in (pt.who_log or {}).items():
        for iso, per in per_date.items():
            for ip in per:
                if ip in dirty:
                    continue
                days_of.setdefault((s, ip), set()).add(iso)
                ed += 1
    low = med = 0
    for inc in getattr(run, "incidents", None) or []:
        k = (inc.get("system"), inc.get("entity"))
        if k not in days_of:
            continue
        mx = max([sev_rank(h.get("severity")) for h in inc.get("history") or []] +
                 [sev_rank(inc.get("severity"))])
        low += mx >= SEV_RANK["low"]
        med += mx >= SEV_RANK["medium"]
    pv_low = sum(1 for e in _pv(run) if (e.get("system"), e.get("entity")) in days_of
                 and sev_rank(e.get("severity")) >= SEV_RANK["low"])
    ks = None
    conf_idx = [i for i, d in enumerate(DETECTORS) if str(d).startswith("conf")]
    if conf_idx and getattr(run, "series", None):
        vals = []
        for key, ser in run.series.items():
            s, _, ip = key.partition("|")
            if (s, ip) not in days_of:
                continue
            p = np.asarray(ser.get("p"))
            if p.ndim == 2 and p.shape[1] > max(conf_idx):
                vals.append(p[:, conf_idx].reshape(-1))
        if vals:
            ks, _ = ks_uniform(np.concatenate(vals))
    return {"entity_days": ed, "inc_low": low, "inc_medium": med, "pv_low": pv_low,
            "far_low": low / ed if ed else None, "far_medium": med / ed if ed else None,
            "pv_far_low": pv_low / ed if ed else None, "ks_conf": ks}


# --------------------------------------------------------------------------- #
# PG7 open schema
# --------------------------------------------------------------------------- #
def _attr_records(model: Any) -> Dict[str, Mapping[str, Any]]:
    if not isinstance(model, Mapping):
        return {}
    for key in ("attrs", "attributes", "registry"):
        if isinstance(model.get(key), Mapping):
            model = model[key]
            break
    return {str(k): v for k, v in model.items() if isinstance(v, Mapping) and
            ("type" in v or "role_sys" in v or "first_seen" in v)}


def pg7_schema(run: Any, pt: PTruth) -> Dict[str, Any]:
    snaps = getattr(run, "psnaps", None) or {}
    dt = float(getattr(run, "scenario_dt", 900.0) or 900.0)
    if not snaps or not pt.attrs:
        return {"registered_first_tick": None}
    last = snaps[max(snaps)]
    reg_ok, type_ok, role_ok, kept, dropped, inv = [], [], [], [], [], []
    for a in pt.attrs:
        name = a["name"]
        for s in a.get("systems") or []:
            recs = _attr_records(((last.get("systems") or {}).get(s) or {}).get("model.attr"))
            rec = recs.get(name)
            if rec is None:
                reg_ok.append(False)
                continue
            fs = _f(rec.get("first_seen"))
            reg_ok.append(math.isfinite(fs) and fs <= float(a["appears"]) + dt + 1e-6)
            if a.get("type"):
                type_ok.append(str(rec.get("type")) == str(a["type"]))
            d_role = pt.day_of_ts(float(a["appears"]) + 86400.0)
            snap_r = snaps.get(d_role) or {}
            model_r = ((snap_r.get("systems") or {}).get(s) or {}).get("model.attr")
            rr = _attr_records(model_r).get(name)
            role_ok.append(bool(rr and rr.get("role_sys")))
            role = str(rec.get("role_sys") or "")
            if a.get("cls") == "informative":
                kept.append(role in ("split", "target"))
            elif a.get("cls") in ("noise", "independent"):
                dropped.append(role == "dropped")
            elif a.get("cls") == "constant":
                inv.append(role == "invariant")
    fr = lambda xs: float(np.mean(xs)) if xs else None
    return {"registered_first_tick": fr(reg_ok), "type_correct": fr(type_ok),
            "role_within_24h": fr(role_ok), "informative_kept": fr(kept),
            "noise_dropped": fr(dropped), "constants_invariant": fr(inv)}


# --------------------------------------------------------------------------- #
# PG8 scenario adaptation
# --------------------------------------------------------------------------- #
def _chosen(sysprof: Any) -> Dict[str, str]:
    if not isinstance(sysprof, Mapping):
        return {}
    ch = sysprof.get("chosen") or {}
    return {str(k): str(v) for k, v in ch.items()} if isinstance(ch, Mapping) else {}


def _harmonic(c: int) -> float:
    return float(sum(1.0 / j for j in range(1, c + 1))) if c < 64 else \
        math.log(c) + 0.5772156649 + 1.0 / (2 * c)


def who_code_lengths(who_log: Mapping[str, Mapping[str, int]], groups: Mapping[str, str],
                     group_size: Mapping[str, int], regions: Mapping[str, Any],
                     alpha: float = 1.0) -> Dict[str, float]:
    """Offline prequential two-part code length (bits) of the source IPs of one
    system's events at each who level (§6.18.2): L = -log2 p(g) + log2|g|,
    unseen items escape with -log2(alpha/(N+alpha)) + the item's own bits.
    Counts are evidence units (§6.5.4): the c events of one IP on one day
    count H(c) (harmonic), so a burst is not c independent observations."""
    nets = [(n, name) for name, n in regions.items()]

    def item(level: str, ip: str) -> Tuple[str, float, float]:
        a = ipaddress.ip_address(ip)
        if level == "ip":
            return ip, 0.0, 32.0
        if level == "/24":
            return str(ipaddress.ip_network(f"{ip}/24", strict=False)), 8.0, 24.0
        if level == "/16":
            return str(ipaddress.ip_network(f"{ip}/16", strict=False)), 16.0, 16.0
        if level == "grp":
            g = groups.get(ip)
            n_g = len(set(groups.values())) + 1
            if g is None:
                return "grp:0", 32.0, math.log2(n_g)
            return g, math.log2(max(1, group_size.get(g, 1))), math.log2(n_g)
        if level == "reg":
            for n, name in nets:
                if a in n:
                    return name, math.log2(n.num_addresses), math.log2(len(nets) + 1)
            return "reg:0", 32.0, math.log2(len(nets) + 1)
        return "*", 32.0, 0.0

    out: Dict[str, float] = {}
    days = sorted(who_log)
    for level in ("ip", "/24", "/16", "grp", "reg", "none"):
        counts: Dict[str, float] = {}
        N = 0.0
        bits = 0.0
        for iso in days:
            for ip in sorted(who_log[iso]):
                c0 = int(who_log[iso][ip])
                if c0 <= 0 or _ip(ip) is None:
                    continue
                c = _harmonic(c0)
                g, addr, idb = item(level, ip)
                bits += c * addr
                n = counts.get(g, 0.0)
                if n == 0:
                    bits += -math.log2(alpha / (N + alpha)) + idb
                    n, N, c = 1.0, N + 1.0, c - 1.0
                if c > 0:
                    bits += -(math.lgamma(n + c) - math.lgamma(n)
                              - math.lgamma(N + alpha + c) + math.lgamma(N + alpha)) / math.log(2)
                counts[g] = n + c
                N += c
        out[level] = bits
    return out


def pg8_adaptation(run: Any, pt: PTruth, day: int = 14) -> Dict[str, Any]:
    snaps = getattr(run, "psnaps", None) or {}
    if not snaps or not pt.strategy:
        return {"arms_ok_share": None}
    d = max((x for x in snaps if x <= day), default=None)
    per_sys: Dict[str, Any] = {}
    oks = []
    cfg = getattr(run, "config", None) or {}
    ipc = ip_classes_of(cfg)
    regions = {}
    for name, cidrs in ipc.items():
        for c in cidrs:
            n = _net(c)
            if n is not None:
                regions[f"reg:{name}:{c}"] = n
    groups: Dict[str, str] = {}
    gsize: Dict[str, int] = {}
    for code, rec in pt.groups.items():
        if rec.get("kind") in ("static", "service", "shared", "automation", "nat"):
            for m in rec.get("members") or []:
                groups[m["ip"]] = code
        elif rec.get("kind") == "pool":
            for m in rec.get("members") or []:
                groups[m["ip"]] = code
        gsize[code] = len({m["ip"] for m in rec.get("members") or []}) if rec.get("kind") != "pool" \
            else _net(rec["cidr"]).num_addresses
    for s, arms in pt.strategy.items():
        ch = _chosen(((snaps.get(d) or {}).get("systems") or {}).get(s, {}).get("model.sysprof")) \
            if d is not None else {}
        if not ch:
            per_sys[s] = {"chosen": None}
            continue
        ok = all(ch.get(dim) in allowed for dim, allowed in arms.items() if dim in ch)
        oks.append(ok)
        hist = []
        for x in sorted(snaps):
            if x > 7:
                c = _chosen((snaps[x].get("systems") or {}).get(s, {}).get("model.sysprof"))
                if c:
                    hist.append(c)
        switches = sum(1 for a, b in zip(hist, hist[1:]) if a != b)
        cl = who_code_lengths(pt.who_log.get(s) or {}, groups, gsize, regions)
        lv = {"ip": "ip", "grp": "grp", "prefix": "/24", "reg": "reg", "none": "none"}.get(
            ch.get("who", ""), None)
        best = min(cl.values()) if cl else None
        chosen_len = cl.get(lv) if lv else None
        if ch.get("who") == "prefix":
            chosen_len = min(cl.get("/24", math.inf), cl.get("/16", math.inf))
        per_sys[s] = {"chosen": ch, "ok": ok, "switches_after_7": switches, "code_bits": cl,
                      "who_within_5pct": (chosen_len <= 1.05 * best)
                      if (best and chosen_len is not None and math.isfinite(chosen_len)) else None}
    return {"arms_ok_share": float(np.mean(oks)) if oks else None,
            "max_switches": max([v.get("switches_after_7", 0) for v in per_sys.values()] or [0]),
            "who_within_5pct": _allv(v.get("who_within_5pct") for v in per_sys.values()),
            "systems": per_sys}


def _allv(xs: Iterable[Optional[bool]]) -> Optional[bool]:
    v = [x for x in xs if x is not None]
    return all(v) if v else None


# --------------------------------------------------------------------------- #
# PG10 views
# --------------------------------------------------------------------------- #
def _accepts(rx: Optional[str], values: Sequence[str]) -> bool:
    if not rx:
        return False
    try:
        c = re.compile(rx)
    except re.error:
        return False
    return all(c.fullmatch(v) for v in values)


def pg10_views(run: Any, pt: PTruth, ipc: Mapping[str, List[str]], seed: int = 0) -> Dict[str, Any]:
    snaps = getattr(run, "psnaps", None) or {}
    out: Dict[str, Any] = {}
    r = np.random.default_rng([seed, 10])
    for day, check_bound in ((11, False), (pt.n_days, True)):
        if day not in snaps:
            out[f"day{day}"] = None
            continue
        row = next((x for x in pt.valid_at_day(day) if x["activity"] == "GA.oa.login"
                    and x["step"] == 0), None)
        if row is None:
            continue
        users = row["content"]["body.kv.username"]["closed_values"]
        ok = False
        for s in _find(statements(snaps[day], ipc), "oa", "POST", "/login"):
            if jaccard(s.who.ipset(), row["who"]["value"]) < 1.0:
                continue
            tw, lw = row["windows"]["workday"], s.when["workday"]
            if not lw or abs(lw[0][0] - tw[0][0]) > 2 or abs(lw[-1][1] - tw[0][1]) > 2:
                continue
            c = s.content.get("body.len") or {}
            b = c.get("band90")
            if not b or not (_rel_ok(_f(b[0]), 1024.0, 0.2) and _rel_ok(_f(b[1]), 2048.0, 0.2)):
                continue
            rg = c.get("range")
            if rg is not None and not (_rel_ok(_f(rg[0]), 512.0, 0.25) and _rel_ok(_f(rg[1]), 3072.0, 0.25)):
                continue
            if check_bound and (rg is None or c.get("cover") is None):
                continue
            if not check_bound and rg is not None and _f(c.get("n_rng"), 0) >= 30 \
                    and c.get("cover") is None:
                continue
            rx = _grammar_rx((s.content.get("body.kv.username") or {}).get("grammar"))
            if not _accepts(rx, users):
                continue
            if check_bound and not regex_contained(rx, r"[a-z.]{1,10}", r):
                continue
            if not bindings_match(row, s)[0]:
                continue
            ok = True
            break
        out[f"day{day}"] = ok
    last = snaps.get(max(snaps)) if snaps else None
    if last is not None:
        fin = next((x for x in pt.rows if x["activity"] == "FIN.finance.approval" and x["method"] == "POST"), None)
        if fin is not None:
            out["finance_single_ip"] = any(s.who.ipset() == set(fin["who"]["value"]) for s in
                                           _find(statements(last, ipc), fin["system"], "POST", fin["route"]))
        ga = set(ip for m in (pt.groups.get("GA") or {}).get("members") or [] for ip in [m["ip"]])
        neg = None
        for key, sts in group_statements(last, ipc).items():
            mem = set().union(*[s.who.ipset() for s in sts]) if sts else set()
            if ga and jaccard(mem & ga, ga) < 0.8 and "综合部" not in "".join(s.text_zh for s in sts):
                continue
            neg = any(s.negative and (s.target_system == "finance" or "财务" in s.text_zh) for s in sts)
            if neg:
                break
        out["ga_negative_finance"] = neg
    return out


# --------------------------------------------------------------------------- #
# PG11 real-world items
# --------------------------------------------------------------------------- #
def _events(run: Any, kinds: Iterable[str]) -> List[Dict[str, Any]]:
    ks = set(kinds)
    return [e for e in getattr(run, "events", None) or [] if e.get("kind") in ks]


def pg11_items(run: Any, pt: PTruth, sbd: Mapping[int, List[LStmt]],
               ipc: Mapping[str, List[str]]) -> Dict[str, Any]:
    truth = {str(r["scenario_id"]): r for r in getattr(run, "truth", None) or []}
    snaps = getattr(run, "psnaps", None) or {}
    days = sorted(sbd)
    last = sbd[days[-1]] if days else []
    out: Dict[str, Any] = {}
    pvs = _pv(run)
    if "R1" in truth:
        ex = truth["R1"].get("expected") or {}
        if "snat_suspect_within_days" in ex:
            day0 = int(truth["R1"]["day"])
            hit_day = None
            for d in sorted(snaps):
                sp = (snaps[d].get("systems") or {}).get("oa", {}).get("model.sysprof") or {}
                chars = sp.get("characteristics") or {}
                if d >= day0 and (chars.get("snat") or chars.get("snat_suspect")):
                    hit_day = d
                    break
            ch = _chosen((snaps[max(snaps)].get("systems") or {}).get("oa", {}).get("model.sysprof")) \
                if snaps else {}
            proxy = (truth["R1"].get("params") or {}).get("proxy")
            who_v = sum(1 for e in pvs if e.get("entity") == proxy and _ev_type(e) == "who")
            hint = any(s.system == "oa" and ("代理" in s.text_zh or s.hint) for s in last)
            out["R1"] = {"snat_day": hit_day, "who_mode": ch.get("who"), "who_violations": who_v,
                         "hint": hint,
                         "pass": (hit_day is not None and hit_day - day0 <= 1 and ch.get("who") == "none"
                                  and who_v == 0 and hint) if snaps else None}
        else:
            out["R1"] = {"pass": None, "compare": "pg1_oa_within_0.05_of_O"}
    if "R2" in truth:
        ex = truth["R2"].get("expected") or {}
        ip = (truth["R2"].get("params") or {}).get("ip")
        shared = any(ip in s.who.shared for s in last)
        ip_bind = any(ip in s.binding_table(a) for s in last for a in s.bindings
                      if (s.bindings[a].get("x") or "net.src") != "sess.key")
        sess_users: Set[str] = set()
        for s in last:
            for a, b in s.bindings.items():
                if b.get("x") == "sess.key":
                    for ys in s.binding_table(a).values():
                        sess_users |= ys & set(ex.get("users") or [])
        out["R2"] = {"shared_item": shared, "ip_binding": ip_bind, "sess_users": len(sess_users),
                     "pass": (shared and not ip_bind and len(sess_users) >= int(ex.get("sess_bindings_min", 8)))
                     if days else None}
    if "R3" in truth:
        p = truth["R3"].get("params") or {}
        new, old = p.get("new"), p.get("old")
        flags = [e for e in pvs if e.get("entity") == new and "readdress_candidate" in _ev_flags(e)]
        max_sev = max([sev_rank(e.get("severity")) for e in flags] or [-1])
        incs = [i for i in getattr(run, "incidents", None) or [] if i.get("entity") == new and
                max([sev_rank(h.get("severity")) for h in i.get("history") or []] +
                    [sev_rank(i.get("severity"))]) >= SEV_RANK["medium"]]
        old_live = any(old in s.who.ipset() and s.state in CONFIRMED for s in last)
        out["R3"] = {"flagged": bool(flags), "max_flag_sev": max_sev, "incidents_medium": len(incs),
                     "old_still_live": old_live,
                     "pass": (bool(flags) and max_sev <= SEV_RANK["low"] and not incs and not old_live)
                     if days else None}
    if "R4" in truth:
        ex = truth["R4"].get("expected") or {}
        ip = (truth["R4"].get("params") or {}).get("ip")
        want = set(ex.get("set_binding") or [])
        sb = any(s.binding_table(a).get(ip) == want for s in last for a in s.bindings)
        third = truth.get("R4x")
        low = None
        if third:
            ev = [e for e in pvs if e.get("entity") == ip and float(third["t_start"]) <= float(e["ts"])
                  <= float(third["t_start"]) + 3600.0]
            low = any(sev_rank(e.get("severity")) == SEV_RANK["low"] for e in ev)
        out["R4"] = {"set_binding": sb, "third_low": low, "pass": (sb and bool(low)) if days else None}
    if "R5" in truth:
        ex = truth["R5"].get("expected") or {}
        ips = set((truth["R5"].get("params") or {}).get("ips") or [])
        rev = False
        for s in last:
            for a, b in s.bindings.items():
                for y, xs in (b.get("reverse") or {}).items():
                    if str(y) == "svc_sync" and set(map(str, xs)) == ips:
                        rev = True
        who_v = sum(1 for e in pvs if e.get("entity") in ips and _ev_type(e) == "who")
        auto = "automation" in json.dumps((snaps[max(snaps)].get("org") or {}).get("model.facets") or {},
                                          ensure_ascii=False) if snaps else False
        out["R5"] = {"reverse_set": rev, "who_violations": who_v, "automation_facet": auto,
                     "pass": (rev and who_v == 0 and auto) if days else None}
    if "R7" in truth:
        dates = sorted({iso for r in pt.rows if r["route"] == "/fin/close"
                        for iso in (pt.opp.get(r["tid"]) or {})})
        n = None
        if len(dates) >= 2:
            d2 = pt.day_index(dates[1])
            t0, t1 = pt.day_start[d2 - 1], pt.day_end(d2)
            n = sum(1 for e in pvs if e.get("system") == "finance" and _ev_type(e) == "novel"
                    and t0 <= float(e["ts"]) < t1 and sev_rank(e.get("severity")) >= SEV_RANK["medium"])
        out["R7"] = {"novel_medium_second": n, "pass": (n == 0) if n is not None else None}
    if "R8" in truth:
        a, b = (truth["R8"].get("params") or {}).get("days") or [22, 28]
        t0, t1 = pt.day_start[a - 1], pt.day_end(b)
        absent = sum(1 for e in _events(run, ["pattern_absent"]) if t0 <= float(e["ts"]) < t1
                     and sev_rank(e.get("severity")) >= SEV_RANK["low"])
        acc = sum(1 for e in _events(run, ["pattern_replaced", "binding_changed", "pattern_drift"])
                  if t0 <= float(e["ts"]) < t1 and (e.get("kind") != "pattern_drift" or
                                                     (e.get("extra") or {}).get("accepted")))
        out["R8"] = {"absent_low": absent, "accepted": acc,
                     "pass": (absent == 0 and acc == 0) if days else None}
    if "R9" in truth:
        p = truth["R9"].get("params") or {}
        rep, base = p.get("replica"), p.get("system")
        by = int((truth["R9"].get("expected") or {}).get("joined_by_day", 12))
        joined = None
        if by in snaps:
            sf = (snaps[by].get("org") or {}).get("model.sysfam") or {}
            m = sf.get("member") or sf.get("members") or sf.get("tree_key") or {}
            if isinstance(m, Mapping):
                joined = bool(m.get(rep)) and m.get(rep) == m.get(base)
        storm = sum(1 for e in pvs if e.get("system") == rep and _ev_type(e) == "novel")
        out["R9"] = {"joined": joined, "novel_on_replica": storm,
                     "pass": (bool(joined) and storm <= 10) if joined is not None else None}
    if "R10" in truth:
        name = (truth["R10"].get("params") or {}).get("name")
        day0 = int(truth["R10"]["day"])
        t_lim = pt.day_end(next((d for d in range(day0, pt.n_days + 1)
                                 if pt.workday[d - 1]), day0))
        gone = [e for e in _events(run, ["attribute_gone"]) if name in json.dumps(e.get("extra") or {})
                and float(e["ts"]) <= t_lim]
        out["R10"] = {"attribute_gone": bool(gone), "pass": bool(gone)}
    if "R11" in truth:
        ip = (truth["R11"].get("entities") or [None])[0]
        nov = sum(1 for e in pvs if e.get("entity") == ip and _ev_type(e) == "novel")
        counts = {d: ptree_node_count((snaps[d].get("systems") or {}).get("portal", {}).get("model.ptree"))
                  for d in snaps}
        pre = counts.get(max((d for d in counts if d < 13 and counts[d] is not None), default=-1))
        post = counts.get(max((d for d in counts if d >= 14 and counts[d] is not None), default=-1))
        growth = (post - pre) / pre if pre and post is not None else None
        truth_routes = {_route_of(r) for r in pt.rows if r["system"] == "portal"}
        learned = any(s.system == "portal" and s.confirmed and ((s.method, s.route) not in truth_routes
                                                                or ip in s.who.ipset()) for s in last)
        out["R11"] = {"novel": nov, "node_growth": growth, "learned_from_scanner": learned,
                      "pass": (nov > 0 and (growth is None or growth < 0.05) and not learned)
                      if days else None}
    if "R12" in truth:
        marker = any(s.system == "portal" and ("观测到的" in s.text_zh or
                                                any(c.get("observed") for c in s.content.values()))
                     for s in last)
        out["R12"] = {"observed_marker": marker, "pass": marker if days else None,
                      "compare": "pg1_portal_within_0.05_of_O"}
    if "R13" in truth:
        mem = set((truth["R13"].get("params") or {}).get("systems") or [])
        fam_ok = None
        ratio = None
        if snaps:
            last_snap = snaps[max(snaps)]
            sf = (last_snap.get("org") or {}).get("model.sysfam") or {}
            m = sf.get("member") or sf.get("members") or sf.get("tree_key") or {}
            if isinstance(m, Mapping) and m:
                keys = {m.get(x) for x in mem}
                fam_ok = len(keys) == 1 and None not in keys
                if fam_ok:
                    fk = next(iter(keys))
                    size = lambda k: len(json.dumps((last_snap.get("systems") or {}).get(k, {})
                                                    .get("model.ptree"), default=str))
                    base = size("crm")
                    ratio = size(fk) / base if base > 4 else None
        out["R13"] = {"one_family": fam_ok, "size_ratio_vs_crm": ratio,
                      "pass": (fam_ok and (ratio is None or ratio <= 2.0)) if fam_ok is not None else None}
    return out


# --------------------------------------------------------------------------- #
# Per-run scoring
# --------------------------------------------------------------------------- #
def score_prun(run: Any, precision_n: int = 300) -> Dict[str, Any]:
    pt = PTruth(getattr(run, "ptruth", None) or {})
    cfg = getattr(run, "config", None) or {}
    ipc = ip_classes_of(cfg)
    snaps = {int(k): v for k, v in (getattr(run, "psnaps", None) or {}).items()}
    seed = int(getattr(run, "seed", 0))
    pg1 = {d: pg1_snapshot(snaps[d], pt, ipc, d, seed, precision_n=precision_n) for d in sorted(snaps)}
    sbd = _stmts_by_day(snaps, ipc)
    per_sys: Dict[str, Dict[int, Optional[float]]] = {}
    last = max(snaps) if snaps else None
    if last is not None:
        for s in sorted({r["system"] for r in pt.rows}):
            per_sys[s] = {"recall": pg1_snapshot(snaps[last], pt, ipc, last, seed, systems={s},
                                                 precision_n=50)["recall"]}
    d14 = max((d for d in snaps if d <= 14), default=None)
    out = {
        "pack": getattr(run, "pack", ""), "seed": seed, "variant": pt.raw.get("variant", ""),
        "aborted": getattr(run, "aborted", None), "n_snapshots": len(snaps),
        "pg1": {str(d): {k: v for k, v in pg1[d].items() if k not in ("stmt_conf", "who_U")}
                for d in pg1},
        "pg1_day14": ({k: v for k, v in pg1[d14].items() if k not in ("per_pattern", "stmt_conf", "who_U")}
                      if d14 is not None else None),
        "pg1_last_by_system": per_sys,
        "bindings_ga_fin_day14": (list(login_bindings(snaps[d14], pt, ipc, d14))
                                  if d14 is not None else None),
        "pg2": pg2_convergence(pg1, pt),
        "false_splits": false_splits({d: s for d, s in snaps.items() if s.get("full")}, pt, ipc),
        "pg3": pg3_specificity(snaps[last], pt, ipc) if last is not None else {},
        "pg5": pg5_drift(run, pt, sbd),
        "pg6": {"anomalies": pg6_anomalies(run, pt, sbd), "far": pg6_far(run, pt)},
        "pg7": pg7_schema(run, pt),
        "pg8": pg8_adaptation(run, pt),
        "pg10": pg10_views(run, pt, ipc, seed),
        "pg11": pg11_items(run, pt, sbd, ipc),
        "gen_stats": dict(getattr(run, "gen_stats", None) or {}),
        "core_active": bool(any((snaps[d].get("systems") or snaps[d].get("org")) for d in snaps)
                            or _pv(run)),
    }
    return out


# --------------------------------------------------------------------------- #
# Gates over runs
# --------------------------------------------------------------------------- #
def _vals(scores: Sequence[Mapping[str, Any]], fn: Callable[[Mapping[str, Any]], Any]) -> List[Any]:
    out = []
    for s in scores:
        try:
            out.append(fn(s))
        except (KeyError, TypeError, IndexError, AttributeError):
            out.append(None)
    return [v for v in out if v is not None]


def _med_check(name: str, scores: Sequence[Mapping[str, Any]], fn: Callable, target: float,
               ge: bool = True) -> Dict[str, Any]:
    v = [float(x) for x in _vals(scores, fn)]
    m = _median(v)
    ci = bootstrap_ci(v) if v else (None, None)
    ok = None if m is None else (m >= target - 1e-12 if ge else m <= target + 1e-12)
    return check(name, m, (">= " if ge else "<= ") + str(target), ok, ci)


def _share_check(name: str, scores: Sequence[Mapping[str, Any]], fn: Callable, target: float = 1.0
                 ) -> Dict[str, Any]:
    v = [bool(x) for x in _vals(scores, fn)]
    m = float(np.mean(v)) if v else None
    return check(name, m, f">= {target}", None if m is None else m >= target - 1e-12)


def compute_pgates(scores: Sequence[Mapping[str, Any]],
                   scale: Optional[Mapping[str, Any]] = None,
                   pg9: Optional[Mapping[str, Any]] = None) -> Dict[str, Dict[str, Any]]:
    """Gates PG1-PG11 from per-run scores. Pack O scores feed PG1-PG3 and
    PG5-PG8, PG10; O-red is reported separately (a gate passing on O but not on
    O-red is reported as not passed); O-real* feeds PG11; PG4 needs the scale
    results (pscale.pg4_summary) and PG9 a comparison (pg9_compare)."""
    scores = copy.deepcopy(list(scores))
    O = [s for s in scores if s.get("pack") == "O"]
    RED = [s for s in scores if s.get("pack") == "O-red"]
    REAL = [s for s in scores if str(s.get("pack", "")).startswith("O-real")]
    gates: Dict[str, Dict[str, Any]] = {}

    def pg1_checks(sc: Sequence[Mapping[str, Any]], tag: str = "") -> List[Dict[str, Any]]:
        ch = [_med_check(f"recall@14{tag}", sc, lambda s: s["pg1_day14"]["recall"], 0.90),
              _med_check(f"precision@14{tag}", sc, lambda s: s["pg1_day14"]["precision"], 0.90)]
        for c in ("who", "when", "content", "bindings", "workflow"):
            ch.append(_med_check(f"recall_{c}@14{tag}", sc,
                                 lambda s, c=c: s["pg1_day14"]["components"][c], 0.85))
        ch.append(_share_check(f"GA+FIN bindings 6/6 at day 14{tag}", sc,
                               lambda s: None if not s.get("bindings_ga_fin_day14") else
                               s["bindings_ga_fin_day14"][0] == s["bindings_ga_fin_day14"][1] > 0))
        return ch

    ch1 = pg1_checks(O)
    red1 = pg1_checks(RED, " (O-red)") if RED else []
    g = gate("PG1 pattern recovery", ch1[0]["value"], ">= 0.90 recall, >= 0.90 precision at day 14",
             ch1 + red1)
    if RED and any(c["pass"] is False for c in red1):
        g["pass"] = False if g["pass"] is not None else None
    gates["PG1"] = g

    ch2 = [
        _share_check("recall non-decreasing (±0.05) outside days 12-15", O,
                     lambda s: s["pg2"]["recall_monotone"]),
        _share_check("daily patterns: 80 % recall by day 7", O,
                     lambda s: None if s["n_snapshots"] == 0 else
                     (s["pg2"]["days_to_80"]["daily"] or 99) <= 7),
        _share_check("weekly patterns: 80 % recall by day 21", O,
                     lambda s: None if s["n_snapshots"] == 0 else
                     (s["pg2"]["days_to_80"]["weekly"] or 99) <= 21),
        _share_check("mean depth non-decreasing before day 12", O,
                     lambda s: s["pg2"]["depth_nondecreasing"]),
        _share_check("median confidence non-decreasing", O, lambda s: s["pg2"]["conf_nondecreasing"]),
        _share_check("median unseen-IP mass non-increasing", O, lambda s: s["pg2"]["U_nonincreasing"]),
        _med_check("ECE", O, lambda s: s["pg2"]["ece"], 0.05, ge=False),
        _med_check("false splits per system-month (O-red)", RED,
                   lambda s: s["false_splits"]["per_system_month"], 1.0, ge=False),
    ]
    gates["PG2"] = gate("PG2 convergence and calibrated confidence", ch2[0]["value"],
                        "monotone recall, 80 % by 7 d (daily) / 21 d (weekly), ECE <= 0.05", ch2)

    ch3 = [
        _share_check("GA login who = the 3 IPs", O, lambda s: s["pg3"].get("ga_login_who")),
        _share_check("finance approval who = {approver}", O, lambda s: s["pg3"].get("fin_approval_who")),
        _share_check("portal login who in {prefix, reg, any}", O,
                     lambda s: s["pg3"].get("portal_login_level")),
        _share_check("no portal bindings", O, lambda s: None if s["n_snapshots"] == 0 else
                     s["pg3"].get("portal_bindings", 0) == 0),
        _med_check("portal exceptions share", O, lambda s: s["pg3"].get("portal_exception_share"),
                   0.01, ge=False),
        _share_check("DEV who in {grp, pool prefix}", O, lambda s: s["pg3"].get("dev_who_ok")),
        _med_check("P11 ARI vs static departments", O, lambda s: s["pg3"].get("ari"), 0.9),
        _share_check("DEV pool one group / one prefix", O, lambda s: s["pg3"].get("dev_pool_grouped")),
    ]
    gates["PG3"] = gate("PG3 specificity reached", ch3[0]["value"], "GA 3 IPs + 3 bindings, ...", ch3)

    gates["PG4"] = dict(scale) if scale else gate(
        "PG4 resources sublinear", None, "slopes (see §12)",
        [check("scale experiment", None, "pscale.pg4_summary", None)])

    ch5 = [_share_check(f"{k}", O, lambda s, k=k: (s["pg5"].get(k) or {}).get("pass"))
           for k in ("D1", "D2", "D3", "D4", "D5")]
    ch5.append(_share_check("192.168.1.21 still jack on day 21", O,
                            lambda s: (s["pg5"].get("non_adoption_binding") or {}).get("pass")))
    ch5.append(_share_check("portal rate.ip_h p99 within 10 % of pre-A7", O,
                            lambda s: (s["pg5"].get("rate_p99") or {}).get("pass")))
    gates["PG5"] = gate("PG5 drift adaptation latency", ch5[0]["value"], "D1-D5 + non-adoption", ch5)

    det = []
    for s in O:
        if not s.get("core_active"):
            continue
        for sid, rec in (s["pg6"]["anomalies"] or {}).items():
            det.append(bool(rec.get("detected")))
    rec6 = float(np.mean(det)) if det else None
    tops = [rec.get("top_reason_ok") for s in O for rec in (s["pg6"]["anomalies"] or {}).values()
            if rec.get("top_reason_ok") is not None]
    ch6 = [check("recall A1-A10 (violation within 1 tick + incident within max(4 ticks, 1 h))",
                 rec6, ">= 0.95", None if rec6 is None else rec6 >= 0.95),
           _med_check("FAR incidents >= LOW per entity-day", O, lambda s: s["pg6"]["far"]["far_low"],
                      0.1, ge=False),
           _med_check("FAR incidents >= MEDIUM per entity-day", O,
                      lambda s: s["pg6"]["far"]["far_medium"], 0.02, ge=False),
           _med_check("pattern_violation >= LOW per entity-day", O,
                      lambda s: s["pg6"]["far"]["pv_far_low"], 0.05, ge=False),
           _med_check("KS D of conf_* p on clean ticks", O, lambda s: s["pg6"]["far"]["ks_conf"],
                      0.05, ge=False),
           check("B29 top reason = violated constraint", float(np.mean(tops)) if tops else None,
                 ">= 0.9", (float(np.mean(tops)) >= 0.9) if tops else None),
           _share_check("A9: who violation on day 21", O,
                        lambda s: (s["pg6"]["anomalies"].get("A9") or {}).get("who_on_last_day")),
           _share_check("A9: outside heavy set on day 21", O,
                        lambda s: (s["pg6"]["anomalies"].get("A9") or {}).get("outside_heavy_set"))]
    gates["PG6"] = gate("PG6 anomaly detection of the requirement's examples", rec6,
                        ">= 0.95 recall; FAR budgets", ch6,
                        per_anomaly={sid: float(np.mean([bool((s["pg6"]["anomalies"].get(sid) or {})
                                                               .get("detected")) for s in O]))
                                     for sid in sorted({k for s in O for k in s["pg6"]["anomalies"]})})

    ch7 = [_med_check("registered in the first tick", O, lambda s: s["pg7"]["registered_first_tick"], 1.0),
           _med_check("type correct", O, lambda s: s["pg7"]["type_correct"], 0.95),
           _med_check("role within 24 h", O, lambda s: s["pg7"]["role_within_24h"], 1.0),
           _med_check("informative kept", O, lambda s: s["pg7"]["informative_kept"], 0.9),
           _med_check("noise dropped", O, lambda s: s["pg7"]["noise_dropped"], 0.95),
           _med_check("constants -> invariants", O, lambda s: s["pg7"]["constants_invariant"], 1.0)]
    gates["PG7"] = gate("PG7 open schema", ch7[0]["value"], "first tick, types, roles", ch7)

    ch8 = [_med_check("chosen arms in strategy truth (day 14)", O, lambda s: s["pg8"]["arms_ok_share"], 0.95),
           _med_check("switches per system after day 7", O, lambda s: s["pg8"]["max_switches"], 2,
                      ge=False),
           _share_check("who level code length within 5 % of best", O,
                        lambda s: s["pg8"]["who_within_5pct"])]
    gates["PG8"] = gate("PG8 scenario adaptation", ch8[0]["value"], ">= 0.95 of (system, seed)", ch8)

    gates["PG9"] = dict(pg9) if pg9 else gate("PG9 non-regression", None, "packs A-E on vs off",
                                              [check("comparison", None, "pg9_compare", None)])

    ch10 = [_share_check("OA statement day 11 (before D1)", O, lambda s: s["pg10"].get("day11")),
            _share_check("OA statement day 21", O,
                         lambda s: next((v for k, v in s["pg10"].items() if k.startswith("day")
                                         and k != "day11"), None)),
            _share_check("finance approval single IP", O, lambda s: s["pg10"].get("finance_single_ip")),
            _share_check("GA group view negative statement (finance)", O,
                         lambda s: s["pg10"].get("ga_negative_finance"))]
    gates["PG10"] = gate("PG10 views", ch10[0]["value"], "OA example statements", ch10)

    by_seed_o = {s["seed"]: s for s in O}
    for s in REAL:
        base = by_seed_o.get(s["seed"])
        for it, sysid in (("R1", "oa"), ("R12", "portal")):
            rec = (s.get("pg11") or {}).get(it)
            if not rec or rec.get("pass") is not None or not rec.get("compare") or base is None:
                continue
            a = ((base.get("pg1_last_by_system") or {}).get(sysid) or {}).get("recall")
            b = ((s.get("pg1_last_by_system") or {}).get(sysid) or {}).get("recall")
            if a is not None and b is not None:
                rec["recall_delta"] = a - b
                rec["pass"] = (a - b) <= 0.05 and rec.get("observed_marker", True) is not False
    ch11 = []
    items = sorted({k for s in REAL for k in (s.get("pg11") or {})})
    for it in items:
        v = [bool(s["pg11"][it]["pass"]) for s in REAL if (s.get("pg11") or {}).get(it)
             and s["pg11"][it].get("pass") is not None]
        seeds_ok = sum(v)
        ch11.append(check(f"{it} expected adaptation (>= 4 of 5 seeds)", seeds_ok,
                          ">= 4/5", None if not v else seeds_ok >= min(4, len(v)) and
                          seeds_ok / len(v) >= 0.8))
    full = [s for s in REAL if s.get("pack") == "O-real"]
    if full and O:
        by_seed = {s["seed"]: s for s in O}
        drops = []
        for s in full:
            b = by_seed.get(s["seed"])
            if b is None:
                continue
            a0 = (b["pg1"].get("21") or {}).get("recall")
            a1 = (s["pg1"].get("21") or {}).get("recall")
            if a0 is not None and a1 is not None:
                drops.append(a0 - a1)
        m = _median(drops)
        ch11.append(check("PG1 recall drop at day 21 vs O", m, "<= 0.05",
                          None if m is None else m <= 0.05))
    gates["PG11"] = gate("PG11 real-world robustness", ch11[0]["value"] if ch11 else None,
                         "every R-item on >= 4/5 seeds; recall drop <= 0.05", ch11)
    return gates


def pg9_compare(gates_on: Mapping[str, Any], gates_off: Mapping[str, Any]) -> Dict[str, Any]:
    """PG9: packs A-E with the progressive core on vs off: every gate 1-15 not
    worse beyond the bootstrap CI (a gate that passes off must not fail on; a
    numeric headline must not move outside the off-run CI in the bad direction)."""
    checks = []
    for k in sorted(set(gates_on) & set(gates_off)):
        a, b = gates_on[k], gates_off[k]
        worse = b.get("pass") is True and a.get("pass") is False
        checks.append(check(f"{k}: on {a.get('pass')} vs off {b.get('pass')}", a.get("value"),
                            f"not worse than {b.get('value')}", not worse))
    return gate("PG9 non-regression", sum(1 for c in checks if c["pass"]),
                "no gate worse with the core on", checks)


# --------------------------------------------------------------------------- #
# Oracle (self-test of the scorer; never used to score a run)
# --------------------------------------------------------------------------- #
def truth_as_statements(pt: PTruth, day: int, confidence: float = 0.95) -> Dict[str, Any]:
    """A snapshot whose statements restate the truth patterns valid on `day`
    in the contract format above: what a perfect learner would publish. The
    scorer must give it recall 1 and precision ~1; the unit tests use it and
    degraded copies of it, and it doubles as a worked example of the contract."""
    systems: Dict[str, Dict[str, Any]] = {}
    for row in pt.valid_at_day(day):
        who = row["who"]
        if who["level"] == "ip":
            w = {"level": "ip", "items": list(who["value"]), "closed": True, "U": 0.01}
        elif who["level"] == "grp":
            w = {"level": "grp", "items": [f"grp:{who['value']}"], "members": list(who["members"]),
                 "closed": True, "U": 0.01}
        else:
            w = {"level": who["level"], "items": [c for c, _ in who["value"]], "closed": False}
        content: Dict[str, Any] = {}
        for attr, c in (row.get("content") or {}).items():
            d: Dict[str, Any] = {}
            if "band90" in c:
                d.update(band90=list(c["band90"]), range=list(c["range"]), n_rng=200, cover=0.01)
            if "required_keys" in c:
                d["required"] = list(c["required_keys"])
            if c.get("grammar"):
                d.update(grammar=c["grammar"]["regex"], c_g=1.0, U_s=0.005)
            if c.get("closed_values") and c.get("kind") in ("bound", "choice", "const"):
                d.update(closed=list(c["closed_values"]), U=0.005)
            content[attr] = d
        bindings = {a: {"x": "net.src", "table": dict(t), "LB": {x: 0.9 for x in t}}
                    for a, t in (row.get("bindings") or {}).items()}
        wf = []
        for frm, _, band in row.get("workflow") or []:
            fr = pt.by_tid.get(frm)
            if fr is not None:
                wf.append({"from": f"{fr['method']} {fr['route']}",
                           "to": f"{row['method']} {row['route']}", "dep": 0.95, "band": list(band)})
        st = {"pattern_id": f"p:{row['system']}:0:{row['tid']}", "view": "system",
              "state": "stable", "confidence": confidence, "support": 100,
              "text_zh": f"{row['act']}", "evidence": {
                  "route": f"{row['method']} {row['route']}", "system": row["system"],
                  "who": w, "when": dict(row["windows"], coverage=0.97), "content": content,
                  "bindings": bindings, "workflow": wf, "depth": 2,
                  "context": ([["net.src", "grp", [row.get("group")], False]]
                              if who["level"] in ("ip", "grp") else
                              [["net.src", "prefix", [c for c, _ in who["value"]], False]]
                              if who["level"] == "prefix" else [])}}
        systems.setdefault(row["system"], {"model.pviews": {"statements": []}})
        systems[row["system"]]["model.pviews"]["statements"].append(st)
    return {"day": day, "full": True, "systems": systems, "org": {}, "group_views": {}}
