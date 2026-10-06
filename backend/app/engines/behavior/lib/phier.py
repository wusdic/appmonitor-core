"""Generalisation hierarchies of the progressive core (docs/lib3/progressive.md §5.4).

STATUS: implemented (W-P0). `gen(a, l, v)` maps a value to its level-l
representative; level 0 is the value itself, the last level is '*'. Levels
backed by a learned model (who groups, regions, time windows, numeric bins,
categorical value groups, set templates) fall through to the next coarser
level while the model is not ready, so level indices are stable. ABSENT (⊥)
maps to itself at every level.

Kinds (chosen from the attribute's registry type, plus the structural names
the spec defines hierarchies for):

    ip     /32 -> /24 -> /16 -> grp -> reg -> *          (IPv6 /128 -> /64 -> /48)
           shared IPs (B17 shared_ip, P00 snat_suspect) map to 'shared:<ip>' at level 0
    dst    peer:port -> member system -> site -> *       (net.dst, system families §6.20)
    route  template -> 'host /a/b/*' -> 'host /a/*' -> host -> *   (http.route)
    path   raw path -> '/a/b/*' -> '/a/*' -> *                     (http.path, hdr.referer)
    tod    minute -> 15-min slot -> learned window -> day|night -> *   (ctx.tod_min)
    when   (daytype, minute) -> (daytype, slot) -> (daytype, window) -> daypart -> daytype -> *
    status code -> class (2xx) -> *
    num    value -> 8 learned bins -> 4 -> 2 -> *
    ord    value -> above/below the median -> *
    cat    value -> learned value group -> *
    text   value -> shape -> shape with length buckets -> charset:length bucket -> length bucket -> *
    set    frozenset -> template (rare elements removed) -> cardinality bucket -> *
    other  value -> *

Shapes (§5.4.5): runs of character classes L=[a-z], U=[A-Z], D=[0-9], literal
punctuation / space kept, anything else X, with exact run lengths
('jack' -> 'L4', 'mike.w' -> 'L4 . L1'); a mixed alphanumeric run longer than
8 collapses to 'A<n>'. `skeleton()` drops the run lengths ('L . L').
"""
from __future__ import annotations

import bisect
import ipaddress
import math
import re
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from .pevent import ABSENT

STAR = "*"
IP_LEVELS = ("ip", "/24", "/16", "grp", "reg", STAR)
IP6_BITS = (128, 64, 48)
IP4_BITS = (32, 24, 16)
GRP_NONE = "grp:∅"
REG_NONE = "reg:∅"
LEN_BUCKETS = ((1, 1, "1"), (2, 3, "2-3"), (4, 7, "4-7"), (8, 15, "8-15"), (16, 31, "16-31"))
LEN_BIG = "32+"
_PUNCT_KEEP = set(" .-_@/:,;=+&?#%!~*'\"()[]{}<>|\\^$`")


# ================================================================ IP levels
@lru_cache(maxsize=65536)
def _ip_parse(v: str) -> Optional[Tuple[int, int]]:
    """(version, int) or None."""
    try:
        a = ipaddress.ip_address(v)
    except ValueError:
        return None
    return a.version, int(a)


@lru_cache(maxsize=65536)
def ip_prefix(v: str, level: int) -> str:
    """Prefix string at level 0/1/2 (/32,/24,/16 or /128,/64,/48)."""
    p = _ip_parse(v)
    if p is None:
        return v
    ver, x = p
    bits = (IP4_BITS if ver == 4 else IP6_BITS)[level]
    tot = 32 if ver == 4 else 128
    if bits == tot:
        return str(ipaddress.ip_address(x))
    cls = ipaddress.IPv4Network if ver == 4 else ipaddress.IPv6Network
    net = cls((x >> (tot - bits) << (tot - bits), bits))
    return str(net)


def is_ip(v: Any) -> bool:
    return isinstance(v, str) and _ip_parse(v) is not None


def ip_space_bits(v: str, level: int) -> float:
    """log2 |g| of a level-0/1/2 item's address space (0 for /32)."""
    p = _ip_parse(v)
    if p is None:
        return 0.0
    tot = 32 if p[0] == 4 else 128
    bits = (IP4_BITS if p[0] == 4 else IP6_BITS)[level]
    return float(tot - bits)


class Regions:
    """First-match CIDR regions: [(name, cidrs)] from config ip_classes /
    dhcp_scopes, or P11 prefix covers."""

    def __init__(self, regions: Iterable[Tuple[str, Iterable[str]]] = ()) -> None:
        self._nets: List[Tuple[Any, str]] = []
        for name, cidrs in regions:
            for c in cidrs:
                try:
                    self._nets.append((ipaddress.ip_network(str(c), strict=False), str(name)))
                except ValueError:
                    continue
        self._memo: Dict[str, str] = {}

    @classmethod
    def from_config(cls, config: Optional[Mapping[str, Any]]) -> "Regions":
        cfg = config or {}
        out: List[Tuple[str, List[str]]] = []
        for key in ("ip_classes", "dhcp_scopes"):
            for item in cfg.get(key) or ():
                if isinstance(item, Mapping):
                    cid = item.get("cidrs") or ([item["cidr"]] if item.get("cidr") else [])
                    out.append((str(item.get("name", key)), list(cid)))
        return cls(out)

    def __len__(self) -> int:
        return len(self._nets)

    def of(self, ip: str) -> str:
        hit = self._memo.get(ip)
        if hit is not None:
            return hit
        res = REG_NONE
        try:
            a = ipaddress.ip_address(ip)
            for net, name in self._nets:
                if a.version == net.version and a in net:
                    res = f"reg:{name}"
                    break
        except ValueError:
            pass
        if len(self._memo) > 65536:
            self._memo.clear()
        self._memo[ip] = res
        return res


# ================================================================== shapes
class Shaped(str):
    """A value that already IS a level-1 shape (stored by the value policy,
    §5.1.3). shape(Shaped) returns it unchanged, so shaped and clear values of
    one attribute share a hierarchy; equality / hashing are those of str."""
    __slots__ = ()


def _cls(ch: str) -> str:
    if "a" <= ch <= "z":
        return "L"
    if "A" <= ch <= "Z":
        return "U"
    if "0" <= ch <= "9":
        return "D"
    if ch in _PUNCT_KEEP:
        return ch
    return "X"


def shape(v: str) -> str:
    """Level-1 run-length class shape, e.g. 'jack' -> 'L4', 'mike.w' -> 'L4 . L1'.
    Tokens are space-separated; a literal space is 'SP'; a repeated literal is
    'c{n}'; other code points form X runs ('X2'). A Shaped value is returned
    as is."""
    if isinstance(v, Shaped):
        return v
    return _shape(str(v))


# One match per maximal alphanumeric stretch (ASCII a-z A-Z 0-9), per run of
# one repeated kept punctuation character, or per run of other code points (X).
# A stretch mixing classes and longer than 8 is A<n>; otherwise its class runs.
# Same output as round 2's per-character _cls loop (tests/lib/
# test_phier_shape_equivalence.py); P00's value policy shapes every long form
# padding (viewstate, 0.5-4 KB) with it.
_PUNCT_RX = "".join("\\" + c if c in "\\]^-[" else c for c in sorted(_PUNCT_KEEP))
_TOK_RX = re.compile(f"([a-zA-Z0-9]+)|([{_PUNCT_RX}])\\2*|[^a-zA-Z0-9{_PUNCT_RX}]+")
_CLS_RUN_RX = re.compile("[a-z]+|[A-Z]+|[0-9]+")


def _cls_of(c0: str) -> str:
    return "L" if "a" <= c0 <= "z" else "U" if "A" <= c0 <= "Z" else "D"


@lru_cache(maxsize=65536)
def _shape(s: str) -> str:
    if not s:
        return "E0"
    out: List[str] = []
    for m in _TOK_RX.finditer(s):
        t = m.group()
        n = len(t)
        if m.group(1) is not None:                     # alphanumeric stretch
            if n > 8 and _CLS_RUN_RX.fullmatch(t) is None:
                out.append(f"A{n}")
            else:
                out.extend(f"{_cls_of(r[0])}{len(r)}" for r in _CLS_RUN_RX.findall(t))
        elif m.group(2) is not None:                   # one kept punctuation character
            lit = "SP" if t[0] == " " else t[0]
            out.append(lit if n == 1 else f"{lit}{{{n}}}")
        else:
            out.append(f"X{n}")
    return " ".join(out)


def len_bucket(n: int) -> str:
    for lo, hi, lab in LEN_BUCKETS:
        if lo <= n <= hi:
            return lab
    return LEN_BIG if n >= 32 else "0"


@lru_cache(maxsize=65536)
def shape_buckets(shp: str) -> str:
    """Level 2: the shape with run lengths replaced by length buckets."""
    out = []
    for tok in shp.split(" "):
        if tok and tok[0] in "LUDXA" and tok[1:].isdigit():
            out.append(f"{tok[0]}{len_bucket(int(tok[1:]))}")
        else:
            out.append(tok)
    return " ".join(out)


@lru_cache(maxsize=65536)
def skeleton(shp: str) -> str:
    """Class / literal sequence without run lengths ('L4 . L1' -> 'L . L')."""
    out = []
    for tok in shp.split(" "):
        if tok and tok[0] in "LUDXA" and tok[1:].isdigit():
            out.append(tok[0])
        elif "{" in tok:
            out.append(tok.split("{")[0])
        else:
            out.append(tok)
    return " ".join(out)


def charset(v: str) -> str:
    return "".join(sorted({_cls(c) for c in str(v)}))


def charset_len(v: str) -> str:
    """Level 3: '{L,.}:4-7' (character classes present + total length bucket)."""
    s = str(v)
    return "{" + ",".join(sorted({_cls(c) for c in s})) + "}:" + len_bucket(len(s))


def shape_level(v: str, level: int) -> str:
    """Text hierarchy on a raw value: 0 value, 1 shape, 2 shape buckets,
    3 charset:len, 4 len bucket, 5 *."""
    if level <= 0:
        return v
    if level == 1:
        return shape(v)
    if level == 2:
        return shape_buckets(shape(v))
    if isinstance(v, Shaped):
        n = _shape_len(v)
        return ("{?}:" if level == 3 else "len:") + len_bucket(n) if level in (3, 4) else STAR
    if level == 3:
        return charset_len(v)
    if level == 4:
        return "len:" + len_bucket(len(str(v)))
    return STAR


# ================================================================== routes
def _route_parts(route: str) -> Tuple[str, str, List[str]]:
    """'POST host /a/b/{num}?x' -> ('POST', 'host', ['a', 'b', '{num}'])."""
    parts = route.split(" ", 2)
    if len(parts) == 3:
        m, h, p = parts
    elif len(parts) == 2:
        m, h, p = "", parts[0], parts[1]
    else:
        m, h, p = "", "", parts[0]
    p = p.split("?", 1)[0]
    segs = [s for s in p.split("/") if s]
    return m, h, segs


@lru_cache(maxsize=65536)
def route_level(route: str, level: int) -> str:
    """http.route: 0 template, 1 'host /a/b/*', 2 'host /a/*', 3 host, 4 *."""
    if level <= 0:
        return route
    if level >= 4:
        return STAR
    _, h, segs = _route_parts(route)
    if level == 3:
        return h or "-"
    k = 2 if level == 1 else 1
    return f"{h} /" + "/".join(segs[:k]) + ("/*" if len(segs) > k else "")


@lru_cache(maxsize=65536)
def path_level(path: str, level: int) -> str:
    """http.path / hdr.referer: 0 raw, 1 '/a/b/*', 2 '/a/*', 3 *."""
    if level <= 0:
        return path
    if level >= 3:
        return STAR
    p = str(path)
    if "://" in p:
        p = p.split("://", 1)[1]
        p = "/" + p.split("/", 1)[1] if "/" in p else "/"
    segs = [s for s in p.split("?", 1)[0].split("/") if s]
    k = 2 if level == 1 else 1
    return "/" + "/".join(segs[:k]) + ("/*" if len(segs) > k else "")


# ==================================================================== time
def window_of(minute: float, windows: Sequence[Tuple[int, int, str]]) -> Optional[str]:
    """Label of the learned window containing the minute; windows may wrap
    midnight (start > end). None when no window model."""
    if not windows:
        return None
    m = int(minute) % 1440
    for s, e, lab in windows:
        if (s <= e and s <= m < e) or (s > e and (m >= s or m < e)):
            return lab
    return "w:off"


def _day_half(minute: float, day_hours: Tuple[float, float]) -> str:
    h = (float(minute) % 1440) / 60.0
    a, b = day_hours
    is_day = (a <= h < b) if a <= b else (h >= a or h < b)
    return "day" if is_day else "night"


_DT_SHORT = {"workday": "wd", "nonworkday": "nwd", "wd": "wd", "nwd": "nwd"}


# =============================================================== hierarchy
class Hierarchies:
    """Levels, gen and card_hint for every attribute of one tree.

    Inputs (all optional; a missing model makes its level fall through):
      registry     {name: record} with record.type / record.hier / record.card_estimate()
                   (lib/pregistry.AttrRecord) or plain dicts with the same keys
      ip2g         {ip: group id}  (P11 model.who_groups ip2g)
      n_groups     number of groups (card hint of the grp level)
      regions      Regions (config ip_classes / dhcp_scopes, else P11 covers)
      windows      [(start_min, end_min, label)] system-root windows (P09), per
                   day type in `windows_dt` {'wd': [...], 'nwd': [...]} for ctx.when
      shared       set of IPs mapped to 'shared:<ip>' at level 0
      dst_members  {peer:port: member system}; dst_sites {member: site}
      day_hours    (8, 20) day hours for day/night
    """

    def __init__(self, registry: Optional[Mapping[str, Any]] = None,
                 ip2g: Optional[Mapping[str, Any]] = None, n_groups: int = 0,
                 regions: Optional[Regions] = None,
                 windows: Optional[Sequence[Tuple[int, int, str]]] = None,
                 windows_dt: Optional[Mapping[str, Sequence[Tuple[int, int, str]]]] = None,
                 shared: Optional[Iterable[str]] = None,
                 dst_members: Optional[Mapping[str, str]] = None,
                 dst_sites: Optional[Mapping[str, str]] = None,
                 day_hours: Tuple[float, float] = (8, 20)) -> None:
        self.registry = registry or {}
        self.ip2g = dict(ip2g or {})
        self.n_groups = int(n_groups or len(set(self.ip2g.values())))
        self.regions = regions or Regions()
        self.windows = list(windows or [])
        self.windows_dt = {k: list(v) for k, v in (windows_dt or {}).items()}
        self.shared = set(shared or ())
        self.dst_members = dict(dst_members or {})
        self.dst_sites = dict(dst_sites or {})
        self.day_hours = tuple(day_hours)
        self._kind_memo: Dict[str, str] = {}
        self._c_edges: Dict[str, Tuple[Any, Any, Optional[List[float]], bool]] = {}

    # ------------------------------------------------------------ kinds
    def _rec(self, a: str) -> Any:
        return self.registry.get(a) if hasattr(self.registry, "get") else None

    @staticmethod
    def _field(rec: Any, key: str, default: Any = None) -> Any:
        if rec is None:
            return default
        if isinstance(rec, Mapping):
            return rec.get(key, default)
        return getattr(rec, key, default)

    def kind(self, a: str) -> str:
        k = self._kind_memo.get(a)
        if k is not None:
            return k
        rec = self._rec(a)
        typ = self._field(rec, "type", "unknown")
        hier = self._field(rec, "hier", {}) or {}
        if a == "net.dst":
            k = "dst"
        elif a == "http.route":
            k = "route"
        elif a in ("http.path", "hdr.referer"):
            k = "path"
        elif a == "ctx.tod_min":
            k = "tod"
        elif a == "ctx.when":
            k = "when"
        elif typ == "ip" or a in ("net.src", "net.peer_src"):
            k = "ip"
        elif hier.get("status") or a == "http.status" or a.endswith(".status"):
            k = "status"
        elif typ == "numeric" or typ == "time":
            k = "num"
        elif typ == "ordinal":
            k = "ord"
        elif typ == "categorical":
            k = "cat"
        elif typ == "text":
            k = "text"
        elif typ == "set":
            k = "set"
        else:
            k = "other"
        if rec is not None:            # kinds of unregistered names may change once typed
            self._kind_memo[a] = k
        return k

    _LEVELS = {
        "ip": list(IP_LEVELS),
        "dst": ["dst", "member", "site", STAR],
        "route": ["route", "prefix2", "prefix1", "host", STAR],
        "path": ["path", "prefix2", "prefix1", STAR],
        "tod": ["minute", "slot15", "window", "half", STAR],
        "when": ["minute", "slot15", "window", "daypart", "daytype", STAR],
        "status": ["code", "class", STAR],
        "num": ["value", "bin8", "bin4", "bin2", STAR],
        "ord": ["value", "median", STAR],
        "cat": ["value", "group", STAR],
        "text": ["value", "shape", "shape_len", "charset_len", "len", STAR],
        "set": ["set", "template", "card", STAR],
        "other": ["value", STAR],
    }

    def levels(self, a: str) -> List[str]:
        return list(self._LEVELS[self.kind(a)])

    def n_levels(self, a: str) -> int:
        return len(self._LEVELS[self.kind(a)])

    # --------------------------------------------------------------- gen
    def gen(self, a: str, level: int, v: Any) -> Any:
        """Level-`level` representative of value v of attribute a."""
        if v is ABSENT or v is None:
            return ABSENT
        k = self.kind(a)
        L = len(self._LEVELS[k])
        if level >= L - 1:
            return STAR
        if level <= 0:
            if k == "ip" and v in self.shared:
                return f"shared:{v}"
            return v
        fn = getattr(self, "_gen_" + k)
        out = fn(a, level, v)
        return self.gen(a, level + 1, v) if out is None else out

    def _gen_ip(self, a: str, level: int, v: Any) -> Optional[Any]:
        s = str(v)
        if s.startswith("shared:"):
            s = s[7:]
        if level in (1, 2):
            return ip_prefix(s, level) if _ip_parse(s) else None
        if level == 3:
            if not self.ip2g:
                return None
            g = self.ip2g.get(s)
            return f"grp:{g}" if g is not None else GRP_NONE
        if level == 4:
            if not len(self.regions):
                return None
            return self.regions.of(s)
        return None

    def _gen_dst(self, a: str, level: int, v: Any) -> Optional[Any]:
        if level == 1:
            m = self.dst_members.get(str(v))
            return None if m is None else f"sys:{m}"
        if level == 2:
            m = self.dst_members.get(str(v))
            site = self.dst_sites.get(m) if m is not None else None
            return None if site is None else f"site:{site}"
        return None

    def _gen_route(self, a: str, level: int, v: Any) -> Optional[Any]:
        return route_level(str(v), level)

    def _gen_path(self, a: str, level: int, v: Any) -> Optional[Any]:
        return path_level(str(v), level)

    def _gen_tod(self, a: str, level: int, v: Any) -> Optional[Any]:
        m = float(v)
        if level == 1:
            return int(m // 15)
        if level == 2:
            return window_of(m, self.windows)
        if level == 3:
            return _day_half(m, self.day_hours)
        return None

    def _gen_when(self, a: str, level: int, v: Any) -> Optional[Any]:
        try:
            dt, m = v
        except (TypeError, ValueError):
            return STAR
        d = _DT_SHORT.get(str(dt), str(dt))
        if level == 1:
            return (d, int(float(m) // 15))
        if level == 2:
            w = window_of(float(m), self.windows_dt.get(d) or [])
            return None if w is None else (d, w)
        if level == 3:
            return f"{d}_{_day_half(float(m), self.day_hours)}"
        if level == 4:
            return d
        return None

    def _gen_status(self, a: str, level: int, v: Any) -> Optional[Any]:
        try:
            c = int(float(v))
        except (TypeError, ValueError):
            return str(v)
        return f"{c // 100}xx" if c > 0 else "0xx"

    def _num_edges(self, a: str) -> Tuple[Optional[np.ndarray], bool]:
        rec = self._rec(a)
        hier = self._field(rec, "hier", {}) or {}
        e = hier.get("edges")
        return (None if e is None or len(e) == 0 else np.asarray(e, dtype=np.float64),
                bool(hier.get("log", False)))

    def _edges_list(self, a: str) -> Tuple[Optional[List[float]], bool]:
        """_num_edges as a Python list, re-read only when the registry record's
        edges object or log flag changes (P02 replaces the array on a refresh)."""
        rec = self._rec(a)
        hier = self._field(rec, "hier", {}) or {}
        e = hier.get("edges")
        lgv = hier.get("log", False)
        hit = self._c_edges.get(a)
        if hit is not None and hit[0] is e and hit[1] is lgv:
            return hit[2], hit[3]
        lst = None if e is None or len(e) == 0 else np.asarray(e, dtype=np.float64).tolist()
        out = (lst, bool(lgv))
        self._c_edges[a] = (e, lgv, out[0], out[1])
        return out

    def num_bin(self, a: str, v: Any) -> Optional[int]:
        """Level-1 bin index 0..len(edges) of a numeric value (None without edges)."""
        edges, lg = self._edges_list(a)
        if edges is None:
            return None
        try:
            x = float(v)
        except (TypeError, ValueError):
            return None
        if not math.isfinite(x):
            return None
        if lg:
            x = math.log(x) if x > 0 else -math.inf
        return int(bisect.bisect_right(edges, x))

    def _gen_num(self, a: str, level: int, v: Any) -> Optional[Any]:
        b = self.num_bin(a, v)
        if b is None:
            return None
        if level == 1:
            return b
        if level == 2:
            return b // 2
        if level == 3:
            return b // 4
        return None

    def _gen_ord(self, a: str, level: int, v: Any) -> Optional[Any]:
        rec = self._rec(a)
        med = (self._field(rec, "hier", {}) or {}).get("median")
        if med is None:
            return None
        try:
            return "hi" if float(v) > float(med) else "lo"
        except (TypeError, ValueError):
            return None

    def _gen_cat(self, a: str, level: int, v: Any) -> Optional[Any]:
        rec = self._rec(a)
        groups = (self._field(rec, "hier", {}) or {}).get("groups")
        if not groups:
            return None
        g = groups.get(v)
        return v if g is None else f"g:{g}"

    def _gen_text(self, a: str, level: int, v: Any) -> Optional[Any]:
        return shape_level(v if isinstance(v, Shaped) else str(v), level)

    def _gen_set(self, a: str, level: int, v: Any) -> Optional[Any]:
        try:
            st = frozenset(v)
        except TypeError:
            return STAR
        if level == 1:
            rec = self._rec(a)
            keep = (self._field(rec, "hier", {}) or {}).get("set_keep")
            if keep is None:
                return None
            return frozenset(x for x in st if x in keep)
        if level == 2:
            return "card:" + len_bucket(len(st)) if st else "card:0"
        return None

    def _gen_other(self, a: str, level: int, v: Any) -> Optional[Any]:
        return None

    # --------------------------------------------------------- card hints
    def card_hint(self, a: str, level: int) -> float:
        """Number of possible values at a level (for L_split / grouping limits)."""
        k = self.kind(a)
        L = len(self._LEVELS[k])
        if level >= L - 1:
            return 1.0
        rec = self._rec(a)
        est = None
        ce = self._field(rec, "card_estimate", None)
        if callable(ce):
            try:
                est = float(ce())
            except Exception:
                est = None
        if k == "ip":
            return [2.0 ** 32, 2.0 ** 24, 2.0 ** 16, self.n_groups + 1.0,
                    len(self.regions) + 1.0][level]
        if k == "num":
            return [max(est or 2.0 ** 16, 2.0), 8.0, 4.0, 2.0][level]
        if k == "ord":
            return [max(est or 16.0, 2.0), 2.0][level]
        if k == "status":
            return [max(est or 64.0, 2.0), 6.0][level]
        if k == "tod":
            return [1440.0, 96.0, max(2.0, len(self.windows) + 1.0), 2.0][level]
        if k == "when":
            return [2880.0, 192.0, 2.0 * (max(len(v) for v in self.windows_dt.values()) + 1.0)
                    if self.windows_dt else 4.0, 4.0, 2.0][level]
        base = max(est or 64.0, 2.0)
        return max(2.0, base / (4.0 ** level))


def _shape_len(shp: str) -> int:
    n = 0
    for tok in shp.split(" "):
        if tok and tok[0] in "LUDXA" and tok[1:].isdigit():
            n += int(tok[1:])
        elif "{" in tok:
            try:
                n += int(tok.split("{", 1)[1].rstrip("}"))
            except ValueError:
                n += 1
        elif tok:
            n += 1
    return n


def shape_length(shp: str) -> int:
    """Total character length a shape stands for ('L4 . L1' -> 6)."""
    return _shape_len(shp)
