"""Organisation generator for the progressive profile core (docs/lib3/progressive.md §11).

The generator's truth for the progressive core is the *persona program* itself:
one declarative `OrgSpec` produces the traffic AND the ground truth
(`gen.pattern_truth`, `gen.group_truth`, `gen.strategy_truth`,
`gen.system_truth`, `gen.attr_truth` and the gen.truth rows of the drifts
D1-D5, anomalies A1-A10 and real-world perturbations R1-R13), so recovery can
be scored constraint by constraint. Nothing here is read by an engine.

Design points
-------------
* A day is *planned* when the virtual clock first reaches it: every activity
  instance of every actor is drawn from `rng_for(seed, pack, 'org', day,
  activity, actor)`, so the traffic does not depend on the tick length
  (pack O60 compares 900-s aggregated with 60-s event mode on the same days)
  nor on which other actors exist (adding the portal scale population or an
  R-item leaves the departments' traffic bit-identical).
* Events are HTTP requests (clear systems: method, route, status, sizes and the
  capture extension `extra['l7']` with body / query / headers / session cookie)
  or opaque TLS records (SNI, JA3, sizes; no `http_*`, no `l7`).
* Aggregated mode (dt >= 900 s): one Observation per (system, source, channel,
  method, path, status, stack[, forwarded-for]) with the usual totals plus
  `extra['ev_sample']` (§5.1.2): up to 64 per-event rows drawn uniformly
  without replacement from their own RNG stream `rng_for(seed, pack, 'evs',
  key)`. Event mode: one Observation per event with `extra['l7']`/`['meta']`.
* The truth is computed statically from the spec, day by day, and runs of
  equal days are merged into rows with `valid_from` / `valid_to`, so a drift,
  a re-addressing or a rename closes a row and opens a new one without any
  special casing. Each row carries a `gen` block from which the scorer draws
  held-out and contrast samples (`sample_value`, `sample_size`,
  `contrast_values`), i.e. the scorer reads the truth, never the engines'
  internals.
* Opportunities (per truth pattern, per local date, per client IP) and a
  per-(system, date, client) event log are recorded while emitting; they are
  truth-side bookkeeping for PG1's opportunity rule and PG8's offline who
  code lengths.

Calendar note. Pack O starts Mon 2025-09-01 (Asia/Shanghai) with the holiday on
day 5 (Fri) and the make-up Saturday on day 6 as §11.3 requires; day 20 is
then a Saturday, so A8 ("day 20 (a workday)") runs on the last workday on or
before day 20, day 19, and its truth row keeps `nominal_day = 20`.
"""
from __future__ import annotations

import copy
import datetime as _dt
import heapq
import ipaddress
import json
import math
import re
import zlib
from dataclasses import dataclass, field
from functools import lru_cache
from typing import Any, Dict, Iterable, List, Optional, Sequence, Set, Tuple, Union
from urllib.parse import quote_plus

import numpy as np

from ..models.schema import Observation
from .generator import (JA3_A, ORG_STANDARD, UA_FF_LIN, Clock, Stack, crc, fixed_stack, ja3,
                        rng_for, unique_stack)

BODY_CAP = 4096
EV_SAMPLE_MAX = 64
TS_SAMPLE_MAX = 64
AGG_MIN_DT = 900.0
WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
_B64 = np.frombuffer(b"ABCDEFGHIJKLMNOPQRSTUVWXYZabcdefghijklmnopqrstuvwxyz0123456789-_",
                     dtype=np.uint8)
_ZH = ("的一是在不了有和人这中大为上个国我以要他时来用们生到作地于出就分对成会可主发年动同工也能下过子说产种"
       "面而方后多定行学法所民得经十三之进着等部度家电力里如水化高自二理起小物现实加量都两体制机当使点从业"
       "本去把性好应开它合还因由其些然前外天政四日那社义事平形相全表间样与关各重新线内数正心反你明看原又")
_EN = ("please review the attached draft and confirm the numbers before friday meeting notes "
       "customer visit went well follow up next week budget approved pending signature").split()


# --------------------------------------------------------------------------- #
# DSL (§11.2)
# --------------------------------------------------------------------------- #
@dataclass
class ValueSpec:
    """kind: 'bound' (the actor's username) | 'choice' {values, p} | 'hex' {lo, hi}
    | 'digits' {lo, hi} | 'alnum' {lo, hi} | 'decimal' {lo, hi} | 'text' {lo, hi,
    lang} | 'const' {value} | 'name' {n, tag, lo, hi} (random from a pool) |
    'list' {lo, hi, elem} | 'period' | 'links' {lo, hi} | 'by_dept' {map, default}
    | 'by_route' {lo, hi, sd} | 'uniform' {lo, hi} | 'int' {lo, hi}."""

    kind: str
    params: Dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> Dict[str, Any]:
        p = {k: (v.to_dict() if isinstance(v, ValueSpec) else v) for k, v in self.params.items()}
        return {"kind": self.kind, "params": p}

    @staticmethod
    def from_dict(d: Any) -> "ValueSpec":
        if isinstance(d, ValueSpec):
            return d
        p = dict(d.get("params") or {})
        if isinstance(p.get("elem"), dict):
            p["elem"] = ValueSpec.from_dict(p["elem"])
        return ValueSpec(str(d["kind"]), p)


# bytes a TLS session adds to the payload upstream (records, handshake share): a TLS
# row's net.bytes_up is the step's `up` payload plus this framing
TLS_UP_FRAMING = 300


@dataclass
class SizeSpec:
    """Mixture of uniforms [(weight, lo, hi)] in bytes, hard-clipped to `clip`."""

    components: List[Tuple[float, float, float]]
    clip: Tuple[float, float] = (0.0, math.inf)

    def to_dict(self) -> Dict[str, Any]:
        return {"components": [list(c) for c in self.components], "clip": list(self.clip)}

    @staticmethod
    def from_dict(d: Any) -> "SizeSpec":
        if isinstance(d, SizeSpec):
            return d
        return SizeSpec([tuple(map(float, c)) for c in d["components"]],
                        tuple(map(float, d.get("clip") or (0.0, math.inf))))

    def sample(self, r: np.random.Generator) -> int:
        w = np.asarray([c[0] for c in self.components], dtype=float)
        i = int(r.choice(len(w), p=w / w.sum())) if len(w) > 1 else 0
        _, lo, hi = self.components[i]
        v = float(r.uniform(lo, hi)) if hi > lo else float(lo)
        return int(round(min(max(v, self.clip[0]), self.clip[1])))

    def cdf(self, x: float) -> float:
        w = np.asarray([c[0] for c in self.components], dtype=float)
        w = w / w.sum()
        if x < self.clip[0]:
            return 0.0
        if x >= self.clip[1]:
            return 1.0
        s = 0.0
        for wi, (_, lo, hi) in zip(w, self.components):
            s += wi * (1.0 if x >= hi else 0.0 if x < lo else (x - lo) / max(hi - lo, 1e-12))
        return float(s)

    def quantile(self, q: float) -> float:
        lo = max(self.clip[0], min(c[1] for c in self.components))
        hi = min(self.clip[1], max(c[2] for c in self.components))
        a, b = lo, hi
        for _ in range(80):
            m = 0.5 * (a + b)
            if self.cdf(m) < q:
                a = m
            else:
                b = m
        return float(b)

    def support(self) -> Tuple[float, float]:
        return (max(self.clip[0], min(c[1] for c in self.components)),
                min(self.clip[1], max(c[2] for c in self.components)))


@dataclass
class BodySpec:
    fmt: str                                   # 'form' | 'json' | 'none'
    fields: Dict[str, ValueSpec] = field(default_factory=dict)
    size: Optional[SizeSpec] = None


@dataclass
class Step:
    method: str                                # HTTP method; 'TLS' for opaque records
    route_fmt: str                             # '/approval/{id}/approve'; the SNI host for TLS
    body: Optional[BodySpec] = None
    p: float = 1.0
    repeat: Tuple[int, int] = (1, 1)
    think_s: Tuple[float, float] = (2.0, 20.0)
    status: Dict[int, float] = field(default_factory=lambda: {200: 1.0})
    resp: SizeSpec = field(default_factory=lambda: SizeSpec([(1.0, 2000.0, 20000.0)]))
    up: Optional[SizeSpec] = None              # TLS record request size
    query: Dict[str, ValueSpec] = field(default_factory=dict)


@dataclass
class WindowSpec:
    daytypes: str = "workday"                  # 'workday' | 'nonworkday' | 'all'
    start: str = "09:00"
    end: str = "17:00"
    arrival: str = "uniform"                   # 'uniform' | 'normal'
    dow: Optional[Set[int]] = None             # weekday numbers (Mon = 0)
    mend: bool = False                         # only the last 2 workdays of the month

    @property
    def m0(self) -> int:
        return _hm(self.start)

    @property
    def m1(self) -> int:
        return _hm(self.end)


@dataclass
class Activity:
    name: str
    system: Union[str, List[str]]              # a list: one member chosen per session
    dept: str
    who: Union[str, List[str]]                 # 'each' | [actor ids] | 'one_of'
    when: List[WindowSpec]
    steps: List[Step]
    per_day: Tuple[int, int] = (1, 1)
    new_session: bool = False
    p_day: float = 1.0
    period_s: Optional[float] = None           # periodic machine activity
    jitter_s: float = 0.0


@dataclass
class Department:
    code: str
    name: str
    ips: List[str] = field(default_factory=list)
    usernames: Dict[str, str] = field(default_factory=dict)       # ip -> username
    pool: Optional[Tuple[str, int, float]] = None                 # (cidr, n_personas, lease_h)
    regions: Optional[List[Tuple[str, float]]] = None             # public population
    n: int = 0                                                    # public / NAT persona count
    active_days: float = 3.0                                      # public: Poisson mean
    stacks: str = "org"                                           # 'org' | 'dev' | 'public' | 'lib'
    kind: str = "static"          # static | pool | public | automation | nat | shared | service
    nat_ip: str = ""
    shared_users: List[str] = field(default_factory=list)          # shared terminal / NAT users
    readdress: Dict[str, Tuple[int, str]] = field(default_factory=dict)   # ip -> (day, new ip)
    days_active: Optional[Tuple[int, int]] = None                  # [from_day, to_day] (D4)


@dataclass
class SystemSpec:
    id: str
    addr: str                                  # 'ip:port'
    visibility: str = "clear"                  # 'clear' | 'tls_opaque'
    host: str = ""
    family: Optional[str] = None
    https: bool = False                        # clear L7 behind TLS termination (log source)


@dataclass
class Drift:
    id: str
    day: int
    kind: str                                  # 'window' | 'rename' | 'route' | 'growth' | 'churn'
    params: Dict[str, Any] = field(default_factory=dict)
    truth: Dict[str, Any] = field(default_factory=dict)


@dataclass
class Anomaly:
    id: str
    day: int
    time: str
    kind: str                                  # 'session' | 'burst'
    entity: str
    params: Dict[str, Any] = field(default_factory=dict)
    truth: Dict[str, Any] = field(default_factory=dict)


@dataclass
class AttrSchedule:
    day: int
    system: Union[str, List[str]]
    where: str                                 # 'headers' | 'meta'
    name: str
    value: ValueSpec
    by: str = "const"                          # 'const' | 'dept' | 'route' | 'noise'
    cls: str = "noise"          # informative | noise | constant | anomaly_signal | independent
    type: str = "categorical"
    until_day: Optional[int] = None


@dataclass
class Perturbation:
    id: str
    day: int
    kind: str
    params: Dict[str, Any] = field(default_factory=dict)
    truth: Dict[str, Any] = field(default_factory=dict)


@dataclass
class OrgSpec:
    name: str
    start_date: _dt.date
    n_days: int
    tz: str
    calendar: Dict[str, List[str]]
    systems: List[SystemSpec]
    departments: List[Department]
    activities: List[Activity]
    drifts: List[Drift] = field(default_factory=list)
    anomalies: List[Anomaly] = field(default_factory=list)
    attr_schedule: List[AttrSchedule] = field(default_factory=list)
    portal_n: int = 500
    portal_regions: List[Tuple[str, float]] = field(default_factory=list)
    perturbations: List[Perturbation] = field(default_factory=list)
    families: Dict[str, List[str]] = field(default_factory=dict)
    strategy: Dict[str, Dict[str, List[str]]] = field(default_factory=dict)
    oncall_p: float = 0.05                     # holiday on-call probability (R8)
    oncall_holidays: List[str] = field(default_factory=list)
    config: Dict[str, Any] = field(default_factory=dict)
    variant: str = "O"


def _hm(s: str) -> int:
    h, m = s.split(":")
    return int(h) * 60 + int(m)


# --------------------------------------------------------------------------- #
# Values, sizes, grammars (pure; the scorer uses them on truth data)
# --------------------------------------------------------------------------- #
@lru_cache(maxsize=65536)
def pool_name(tag: str, i: int, lo: int = 3, hi: int = 8) -> str:
    h = zlib.crc32(f"{tag}|{i}".encode())
    n = lo + h % (hi - lo + 1)
    out = []
    for j in range(n):
        h = zlib.crc32(f"{h}|{j}".encode())
        out.append(chr(97 + h % 26))
    return "".join(out)


def _rand_chars(r: np.random.Generator, alphabet: str, n: int) -> str:
    if n <= 0:
        return ""
    idx = r.integers(0, len(alphabet), n)
    return "".join(alphabet[i] for i in idx.tolist())


def sample_value(spec: Any, r: np.random.Generator, ctx: Optional[Dict[str, Any]] = None) -> Any:
    """One value of a ValueSpec (or its dict form). ctx: {username, dept, aid,
    route, date}."""
    spec = ValueSpec.from_dict(spec)
    p = spec.params
    ctx = ctx or {}
    k = spec.kind
    if k == "const":
        return p.get("value", "")
    if k == "bound":
        return ctx.get("username", "")
    if k == "choice":
        vals = list(p["values"])
        pr = p.get("p")
        if pr:
            pr = np.asarray(pr, dtype=float)
            return vals[int(r.choice(len(vals), p=pr / pr.sum()))]
        return vals[int(r.integers(0, len(vals)))]
    if k in ("hex", "digits", "alnum"):
        n = int(r.integers(int(p.get("lo", 8)), int(p.get("hi", p.get("lo", 8))) + 1))
        alpha = {"hex": "0123456789abcdef", "digits": "0123456789",
                 "alnum": "abcdefghijklmnopqrstuvwxyz0123456789"}[k]
        s = _rand_chars(r, alpha, n)
        if k == "digits" and n > 1 and s[0] == "0":
            s = "1" + s[1:]
        return s
    if k == "decimal":
        return f"{float(r.uniform(p['lo'], p['hi'])):.2f}"
    if k == "int":
        return int(r.integers(int(p["lo"]), int(p["hi"]) + 1))
    if k == "uniform":
        return round(float(r.uniform(p["lo"], p["hi"])), 3)
    if k == "text":
        n = int(r.integers(int(p.get("lo", 20)), int(p.get("hi", 200)) + 1))
        if p.get("lang", "zh") == "zh":
            return _rand_chars(r, _ZH, n)
        words, tot = [], 0
        while tot < n:
            w = _EN[int(r.integers(0, len(_EN)))]
            words.append(w)
            tot += len(w) + 1
        return " ".join(words)[:n]
    if k == "links":
        n = int(r.integers(int(p.get("lo", 1800)), int(p.get("hi", 2200)) + 1))
        parts, tot = [], 0
        while tot < n:
            u = f"http://spam{int(r.integers(0, 99))}.example.com/{_rand_chars(r, 'abcdefgh', 8)} "
            parts.append(u + _rand_chars(r, _ZH, 20) + " ")
            tot += len(parts[-1])
        return "".join(parts)[:n]
    if k == "name":
        i = int(r.integers(0, int(p.get("n", 50000))))
        return pool_name(str(p.get("tag", "pub")), i, int(p.get("lo", 3)), int(p.get("hi", 8)))
    if k == "list":
        n = int(r.integers(int(p.get("lo", 1)), int(p.get("hi", 5)) + 1))
        return [sample_value(p["elem"], r, ctx) for _ in range(n)]
    if k == "period":
        d = ctx.get("date")
        return d.strftime("%Y-%m") if d is not None else "2025-09"
    if k == "by_dept":
        v = p.get("map", {}).get(ctx.get("dept", ""), p.get("default", ""))
        if isinstance(v, (list, tuple)):
            return v[crc(str(ctx.get("aid", ""))) % len(v)]
        return v
    if k == "by_route":
        base = float(p.get("lo", 0.0)) + (crc(str(ctx.get("route", ""))) % 1000) / 1000.0 * (
            float(p.get("hi", 100.0)) - float(p.get("lo", 0.0)))
        return round(base + float(r.normal(0.0, float(p.get("sd", 1.0)))), 3)
    raise ValueError(f"unknown ValueSpec kind {k!r}")


def sample_size(spec: Any, r: np.random.Generator) -> int:
    return SizeSpec.from_dict(spec).sample(r)


_CLASSES = (("a-z", "abcdefghijklmnopqrstuvwxyz"), ("A-Z", "ABCDEFGHIJKLMNOPQRSTUVWXYZ"),
            ("0-9", "0123456789"))


def grammar_of(values: Iterable[str]) -> Dict[str, Any]:
    """Truth grammar of a finite value set: charset-class union + length range,
    e.g. {jack, rose, mike} -> [a-z]{4}; {jack, rose, mike.w} -> [a-z.]{4,6}."""
    vals = [str(v) for v in values]
    if not vals:
        return {"charset": "", "lo": 0, "hi": 0, "regex": ""}
    chars = set("".join(vals))
    parts, rest = [], set(chars)
    for name, alpha in _CLASSES:
        if chars & set(alpha):
            parts.append(name)
            rest -= set(alpha)
    lit = "".join(sorted(rest))
    charset = "".join(parts) + lit
    lo, hi = min(map(len, vals)), max(map(len, vals))
    return grammar_regex(charset, lo, hi)


_RANGE = re.compile(r"(\w)-(\w)")


def _parse_charset(charset: str) -> List[Tuple[str, str]]:
    """'a-z0-9._' -> [('a','z'), ('0','9'), ('.','.'), ('_','_')]."""
    out: List[Tuple[str, str]] = []
    i = 0
    while i < len(charset):
        m = _RANGE.match(charset, i)
        if m:
            out.append((m.group(1), m.group(2)))
            i = m.end()
        else:
            out.append((charset[i], charset[i]))
            i += 1
    return out


def grammar_regex(charset: str, lo: int, hi: int) -> Dict[str, Any]:
    cls = ""
    for a, b in _parse_charset(charset):
        if a == b:
            cls += ("\\" + a) if a in "\\]^-[" else a
        else:
            cls += f"{a}-{b}"
    q = f"{{{lo}}}" if lo == hi else f"{{{lo},{hi}}}"
    return {"charset": charset, "lo": int(lo), "hi": int(hi), "regex": f"[{cls}]{q}"}


def _charset_alpha(charset: str) -> str:
    return "".join("".join(chr(c) for c in range(ord(a), ord(b) + 1))
                   for a, b in _parse_charset(charset))


def spec_grammar(spec: Any, ctx_values: Optional[Sequence[str]] = None) -> Optional[Dict[str, Any]]:
    """Truth grammar of a ValueSpec (None when the kind has no string shape)."""
    spec = ValueSpec.from_dict(spec)
    p = spec.params
    if spec.kind == "bound":
        return grammar_of(ctx_values or [])
    if spec.kind in ("hex", "digits", "alnum"):
        cs = {"hex": "0-9a-f", "digits": "0-9", "alnum": "a-z0-9"}[spec.kind]
        return grammar_regex(cs, int(p.get("lo", 8)), int(p.get("hi", p.get("lo", 8))))
    if spec.kind == "name":
        return grammar_regex("a-z", int(p.get("lo", 3)), int(p.get("hi", 8)))
    if spec.kind == "choice":
        return grammar_of(p["values"])
    return None


def truth_values(spec: Any, r: np.random.Generator, n: int,
                 ctx_values: Optional[Sequence[str]] = None) -> List[str]:
    """n fresh truth samples of a text constraint."""
    spec = ValueSpec.from_dict(spec)
    if spec.kind == "bound":
        vals = list(ctx_values or [])
        return [vals[int(i)] for i in r.integers(0, len(vals), n)] if vals else []
    return [str(sample_value(spec, r)) for _ in range(n)]


_FOREIGN = "ABCXYZ0123456789'\" =<>;-_%/"


def contrast_values(grammar: Dict[str, Any], r: np.random.Generator, n: int) -> List[str]:
    """n strings outside a truth grammar: half with one character from a class
    outside the charset, half with a length outside [lo, hi]."""
    charset = _charset_alpha(str(grammar.get("charset", "")))
    lo, hi = int(grammar.get("lo", 1)), int(grammar.get("hi", 1))
    foreign = "".join(c for c in _FOREIGN if c not in charset) or "é"
    alpha = charset or "a"
    out: List[str] = []
    for i in range(n):
        if i % 2 == 0:
            m = int(r.integers(max(lo, 1), max(hi, 1) + 1))
            s = list(_rand_chars(r, alpha, m))
            s[int(r.integers(0, m))] = foreign[int(r.integers(0, len(foreign)))]
            out.append("".join(s))
        else:
            if lo > 1 and r.random() < 0.3:
                m = int(r.integers(1, lo))
            else:
                m = hi + int(r.integers(1, 6))
            out.append(_rand_chars(r, alpha, m))
    return out


# --------------------------------------------------------------------------- #
# Actors
# --------------------------------------------------------------------------- #
@dataclass
class Actor:
    aid: str                    # stable id: the base IP, or '<dept>#<i>' for pool / NAT / public
    dept: str
    ip: str = ""                # fixed address ('' for a pool persona)
    username: str = ""
    stack: Stack = ORG_STANDARD
    pool: Optional[Tuple[str, int, float]] = None
    slot: int = 0
    readdress: Optional[Tuple[int, str]] = None
    days: Optional[Set[int]] = None           # active days (public); None = every day
    parity: Optional[int] = None              # shared terminal: works on days with day % 2 == parity
    kind: str = "static"


@dataclass(slots=True)
class OEv:
    ts: float
    system: str
    src: str                    # the real client address
    ch: str                     # 'h' HTTP | 't' TLS
    method: str
    host: str
    path: str
    status: int
    up: int
    down: int
    dur: float
    stack: Stack
    l7: Optional[Dict[str, Any]]
    meta: Optional[Dict[str, Any]]
    tid: Optional[str]
    aid: str
    anomaly: Optional[str] = None
    route: str = ""


def act_of(e: "OEv") -> str:
    """The action of a generated event as the core sees it: method and route
    template for HTTP, the server name for TLS (truth only; PG8)."""
    if e.ch == "h":
        return f"{e.method} {e.route or e.path}"
    return f"TLS {e.host}"


def _net_hosts(cidr: str) -> List[str]:
    net = ipaddress.ip_network(cidr, strict=False)
    return [str(h) for h in net.hosts()]


# --------------------------------------------------------------------------- #
# The organisation of pack O (§11.3) and its variants
# --------------------------------------------------------------------------- #
KB = 1024.0
O_START = _dt.date(2025, 9, 1)            # Monday; day 5 holiday (Fri), day 6 make-up Saturday

O_SYSTEMS = [
    SystemSpec("oa", "192.168.100.100:8080", "clear", "oa.corp.local"),
    SystemSpec("finance", "192.168.100.110:8443", "clear", "fin.corp.local", https=True),
    SystemSpec("crm", "192.168.100.120:8080", "clear", "crm.corp.local"),
    SystemSpec("code", "192.168.100.130:443", "tls_opaque", "git.corp.local"),
    SystemSpec("mail", "192.168.100.140:443", "tls_opaque", "mail.corp.local"),
    SystemSpec("portal", "192.168.100.150:80", "clear", "www.corp.example"),
]

SALES_NAMES = ["amy", "brian", "carol", "david", "emma", "frank", "grace", "henry", "ivy", "james",
               "kevin", "linda", "michelle", "nancy", "oscar", "patricia", "quinn", "ryan",
               "jonathan", "brandon"]
PORTAL_REGIONS = [("10.60.0.0/16", 0.5), ("10.61.0.0/16", 0.3), ("172.16.0.0/16", 0.2)]
OPINIONS = ["同意", "退回", "同意，请尽快办理"]


def _o_params(red: bool) -> Dict[str, Any]:
    """Every department / window / size / name constant of pack O; the red-team
    variant O-red (never used for tuning) draws a different organisation."""
    if not red:
        return {
            "GA": {"ips": ["192.168.1.21", "192.168.1.23", "10.168.7.121"],
                   "users": ["jack", "rose", "mike"], "approver": "192.168.1.21",
                   "reporters": ["192.168.1.23", "10.168.7.121"],
                   "login": ("09:00", "09:21"), "login_new": ("08:30", "08:51"),
                   "login_size": SizeSpec([(0.90, 1 * KB, 2 * KB), (0.05, 0.5 * KB, 1 * KB),
                                           (0.05, 2 * KB, 3 * KB)], (0.5 * KB, 3 * KB)),
                   "rename": ("10.168.7.121", "mike.w")},
            "FIN": {"ips": ["192.168.2.10", "192.168.2.11", "192.168.2.12"],
                    "users": ["lucy", "tom", "kate"], "approver": "192.168.2.10",
                    "books": ["192.168.2.11", "192.168.2.12"], "login": ("08:50", "09:10")},
            "SALES": {"ips": [f"192.168.3.{i}" for i in range(20, 40)], "users": SALES_NAMES,
                      "login": ("08:30", "09:30")},
            "DEV": {"pool": ("10.50.0.0/22", 60, 24.0)},
            "regions": PORTAL_REGIONS, "ip_classes": True,
        }
    return {
        "GA": {"ips": ["192.168.4.31", "192.168.4.37", "10.77.2.14", "10.77.2.19", "192.168.4.44"],
               "users": ["alice", "bob", "chen", "dora", "evan"], "approver": "10.77.2.14",
               "reporters": ["192.168.4.31", "192.168.4.44"],
               "login": ("08:40", "09:05"), "login_new": ("08:10", "08:35"),
               "login_size": SizeSpec([(0.9, 2 * KB, 4 * KB), (0.1, 1 * KB, 2 * KB)],
                                      (1 * KB, 4 * KB)),
               "rename": ("10.77.2.19", "dora.k")},
        "FIN": {"ips": ["192.168.6.5", "192.168.6.6"], "users": ["wang", "zhao"],
                "approver": "192.168.6.6", "books": ["192.168.6.5"], "login": ("09:15", "09:40")},
        "SALES": {"ips": [f"192.168.8.{i}" for i in range(100, 112)],
                  "users": [pool_name("red-sales", i, 4, 7) for i in range(12)],
                  "login": ("09:00", "10:00")},
        "DEV": {"pool": ("10.51.0.0/23", 40, 24.0)},
        "regions": [("10.70.0.0/16", 0.6), ("10.71.0.0/16", 0.25), ("172.20.0.0/16", 0.15)],
        "ip_classes": False,
    }


def _w(start: str, end: str, daytypes: str = "workday", **kw: Any) -> WindowSpec:
    return WindowSpec(daytypes, start, end, **kw)


def _form(fields: Dict[str, ValueSpec], lo: float, hi: float, clip: Optional[Tuple] = None) -> BodySpec:
    return BodySpec("form", fields, SizeSpec([(1.0, lo, hi)], clip or (lo, hi)))


def _login_steps(host_login: str, fields: Dict[str, ValueSpec], size: SizeSpec,
                 home: str = "/home") -> List[Step]:
    return [Step("POST", host_login, BodySpec("form", fields, size), status={302: 1.0},
                 resp=SizeSpec([(1.0, 300.0, 600.0)]), think_s=(1.0, 3.0)),
            Step("GET", home, think_s=(1.0, 3.0), resp=SizeSpec([(1.0, 15000.0, 40000.0)]))]


def _oa_login_fields() -> Dict[str, ValueSpec]:
    return {"username": ValueSpec("bound"), "password": ValueSpec("hex", {"lo": 8, "hi": 16}),
            "captcha": ValueSpec("digits", {"lo": 4, "hi": 4}),
            "csrf": ValueSpec("hex", {"lo": 32, "hi": 32})}


def _docs_steps() -> List[Step]:
    return [Step("GET", "/docs", think_s=(5.0, 60.0), resp=SizeSpec([(1.0, 8000.0, 30000.0)])),
            Step("GET", "/docs/{id}", think_s=(5.0, 60.0), resp=SizeSpec([(1.0, 20000.0, 200000.0)])),
            Step("POST", "/docs/{id}/comment", p=0.4, think_s=(30.0, 300.0),
                 body=BodySpec("form", {"text": ValueSpec("text", {"lo": 20, "hi": 80})},
                               SizeSpec([(1.0, 800.0, 1500.0)])))]


def _mail_steps() -> List[Step]:
    return [Step("TLS", "mail.corp.local", repeat=(2, 8), think_s=(5.0, 60.0),
                 up=SizeSpec([(0.8, 800.0, 6000.0), (0.2, 20000.0, 300000.0)]),
                 resp=SizeSpec([(1.0, 2000.0, 80000.0)]))]


def build_org(variant: str = "O", portal_n: int = 500, n_meta: int = 0,
              real: Iterable[str] = (), red: bool = False, n_days: int = 21,
              independent_attrs: int = 0, r1_trusted: bool = True) -> OrgSpec:
    """The organisation of pack O (§11.3) with drifts D1-D5 and anomalies
    A1-A10 (§11.4). `real` switches R-items on (pack O-real), `red` builds the
    red-team organisation (O-red), `n_meta` adds synthetic meta.f### attributes
    on day 3 (O-scale), `independent_attrs` adds meta.z## false-split probes."""
    P = _o_params(red)
    real = set(real)
    GA, FIN, SALES = P["GA"], P["FIN"], P["SALES"]
    login_size = GA["login_size"]
    holidays = [(O_START + _dt.timedelta(days=4)).isoformat()]
    makeup = [(O_START + _dt.timedelta(days=5)).isoformat()]
    oncall: List[str] = []
    if "R8" in real:
        n_days = max(n_days, 35)
        oncall = [(O_START + _dt.timedelta(days=d - 1)).isoformat() for d in range(22, 29)]
        holidays += oncall
    systems = [copy.copy(s) for s in O_SYSTEMS]
    depts: List[Department] = [
        Department("GA", "综合部", list(GA["ips"]), dict(zip(GA["ips"], GA["users"]))),
        Department("FIN", "财务部", list(FIN["ips"]), dict(zip(FIN["ips"], FIN["users"]))),
        Department("SALES", "销售部", list(SALES["ips"]), dict(zip(SALES["ips"], SALES["users"]))),
        Department("DEV", "研发", pool=P["DEV"]["pool"], kind="pool", stacks="dev"),
        Department("PUB", "公网用户", regions=list(P["regions"]), n=int(portal_n), kind="public",
                   stacks="public"),
        Department("AUTO", "自动化", ["192.168.9.9", "192.168.9.5"], kind="automation", stacks="lib"),
    ]
    oa_login = _oa_login_fields()
    ga_login_steps = _login_steps("/login", oa_login, login_size)
    other_login = SizeSpec([(1.0, 0.6 * KB, 1.4 * KB)], (0.6 * KB, 1.4 * KB))
    acts: List[Activity] = [
        # ---- 综合部 GA -> OA
        Activity("GA.oa.login", "oa", "GA", "each", [_w(*GA["login"])], ga_login_steps),
        Activity("GA.oa.approvals", "oa", "GA", [GA["approver"]], [_w("09:25", "11:30")], [
            Step("GET", "/approval/list", think_s=(30.0, 300.0), resp=SizeSpec([(1.0, 5000.0, 20000.0)])),
            Step("GET", "/approval/{id}", think_s=(30.0, 300.0), resp=SizeSpec([(1.0, 8000.0, 40000.0)])),
            Step("POST", "/approval/{id}/approve", think_s=(30.0, 300.0), body=BodySpec("form", {
                "id": ValueSpec("digits", {"lo": 6, "hi": 6}),
                "opinion": ValueSpec("choice", {"values": OPINIONS, "p": [0.7, 0.1, 0.2]}),
                "sign": ValueSpec("hex", {"lo": 32, "hi": 32})},
                SizeSpec([(1.0, 0.8 * KB, 1.5 * KB)], (0.8 * KB, 1.5 * KB))),
                 status={302: 1.0})], per_day=(3, 8)),
        Activity("GA.oa.documents", "oa", "GA", "each", [_w("09:30", "16:30")], _docs_steps(),
                 per_day=(2, 5)),
        Activity("GA.oa.report", "oa", "GA", list(GA["reporters"]), [_w("17:00", "17:10")], [
            Step("GET", "/report/form", think_s=(60.0, 240.0), resp=SizeSpec([(1.0, 6000.0, 12000.0)])),
            Step("POST", "/report/generate", think_s=(60.0, 240.0), body=BodySpec("json", {
                "dept": ValueSpec("const", {"value": "GA"}), "period": ValueSpec("period"),
                "items": ValueSpec("list", {"lo": 10, "hi": 40, "elem": ValueSpec(
                    "text", {"lo": 10, "hi": 40})})},
                SizeSpec([(1.0, 20 * KB, 60 * KB)], (20 * KB, 60 * KB))),
                 resp=SizeSpec([(1.0, 2000.0, 5000.0)]))], new_session=True),
        Activity("GA.mail", "mail", "GA", "each", [_w("09:25", "09:40"), _w("13:30", "14:00")],
                 _mail_steps()),
        # ---- 财务部 FIN
        Activity("FIN.finance.login", "finance", "FIN", "each", [_w(*FIN["login"])], _login_steps(
            "/fin/login", {"username": ValueSpec("bound"),
                           "password": ValueSpec("hex", {"lo": 8, "hi": 16}),
                           "otp": ValueSpec("digits", {"lo": 6, "hi": 6})},
            SizeSpec([(1.0, 0.6 * KB, 1.2 * KB)], (0.6 * KB, 1.2 * KB)), "/fin/home")),
        Activity("FIN.finance.approval", "finance", "FIN", [FIN["approver"]],
                 [_w("10:00", "11:30"), _w("15:00", "16:00")], [
            Step("GET", "/fin/approval/list", think_s=(20.0, 120.0), resp=SizeSpec([(1.0, 6000.0, 30000.0)])),
            Step("POST", "/fin/approval/{id}/approve", think_s=(20.0, 120.0), body=BodySpec("form", {
                "voucher": ValueSpec("digits", {"lo": 8, "hi": 8}),
                "amount": ValueSpec("decimal", {"lo": 100.0, "hi": 50000.0}),
                "opinion": ValueSpec("choice", {"values": OPINIONS, "p": [0.8, 0.1, 0.1]}),
                "sign": ValueSpec("hex", {"lo": 32, "hi": 32})},
                SizeSpec([(1.0, 0.9 * KB, 1.6 * KB)], (0.9 * KB, 1.6 * KB))), status={302: 1.0})],
                 per_day=(2, 5)),
        Activity("FIN.finance.bookkeeping", "finance", "FIN", list(FIN["books"]), [_w("09:15", "17:30")], [
            Step("GET", "/fin/ledger", think_s=(10.0, 90.0), resp=SizeSpec([(1.0, 10000.0, 50000.0)])),
            Step("GET", "/fin/ledger/{id}", repeat=(1, 4), think_s=(10.0, 90.0)),
            Step("POST", "/fin/voucher/create", think_s=(30.0, 240.0), body=BodySpec("json", {
                "voucher": ValueSpec("digits", {"lo": 8, "hi": 8}),
                "lines": ValueSpec("list", {"lo": 2, "hi": 12, "elem": ValueSpec(
                    "decimal", {"lo": 10.0, "hi": 9000.0})})},
                SizeSpec([(1.0, 2 * KB, 8 * KB)], (2 * KB, 8 * KB))))], per_day=(3, 8)),
        Activity("FIN.oa.login", "oa", "FIN", "each", [_w("09:05", "09:30")],
                 _login_steps("/login", oa_login, other_login)),
        Activity("FIN.oa.documents", "oa", "FIN", "each", [_w("09:30", "17:00")], _docs_steps(),
                 per_day=(1, 3)),
        Activity("FIN.mail", "mail", "FIN", "each", [_w("09:15", "09:30"), _w("13:30", "14:00")],
                 _mail_steps()),
        # ---- 销售部 SALES
        Activity("SALES.oa.login", "oa", "SALES", "each", [_w(*SALES["login"])],
                 _login_steps("/login", oa_login, other_login)),
        Activity("SALES.oa.documents", "oa", "SALES", "each", [_w("09:30", "17:30")], _docs_steps(),
                 per_day=(1, 3)),
        Activity("SALES.oa.weekly", "oa", "SALES", "each", [_w("16:00", "17:00", dow={4})], [
            Step("POST", "/report/weekly", body=BodySpec("json", {
                "dept": ValueSpec("const", {"value": "SALES"}), "week": ValueSpec("period"),
                "visits": ValueSpec("list", {"lo": 3, "hi": 15, "elem": ValueSpec(
                    "text", {"lo": 10, "hi": 60})})},
                SizeSpec([(1.0, 4 * KB, 12 * KB)], (4 * KB, 12 * KB))))]),
        Activity("SALES.crm", "crm", "SALES", "each", [_w("09:00", "18:00")], [
            Step("GET", "/crm/customer/{id}", repeat=(1, 3), think_s=(10.0, 120.0),
                 resp=SizeSpec([(1.0, 5000.0, 30000.0)])),
            Step("POST", "/crm/visit", p=0.6, think_s=(60.0, 600.0), body=BodySpec("form", {
                "customer": ValueSpec("digits", {"lo": 6, "hi": 6}),
                "note": ValueSpec("text", {"lo": 10, "hi": 120})},
                SizeSpec([(1.0, 1200.0, 2400.0)])))], per_day=(2, 6)),
        Activity("SALES.mail", "mail", "SALES", "each", [_w("09:00", "09:30"), _w("13:00", "13:30")],
                 _mail_steps()),
        # ---- 研发 DEV (DHCP pool)
        Activity("DEV.code", "code", "DEV", "each", [_w("10:00", "22:00")], [
            Step("TLS", "git.corp.local", repeat=(5, 40), think_s=(1.0, 10.0),
                 up=SizeSpec([(0.7, 500.0, 5000.0), (0.3, 50000.0, 2000000.0)]),
                 resp=SizeSpec([(0.6, 1000.0, 20000.0), (0.4, 100000.0, 5000000.0)]))],
                 per_day=(3, 10)),
        Activity("DEV.oa.login", "oa", "DEV", "each", [_w("09:30", "11:00")],
                 _login_steps("/login", oa_login, other_login), p_day=0.4),
        Activity("DEV.mail", "mail", "DEV", "each", [_w("10:00", "10:30"), _w("14:00", "14:30")],
                 _mail_steps()),
        # ---- public portal
        Activity("PUB.portal.visit", "portal", "PUB", "each",
                 [_w("07:00", "23:00", "all", arrival="normal")], [
            Step("GET", "/", think_s=(2.0, 15.0), resp=SizeSpec([(1.0, 20000.0, 60000.0)])),
            Step("POST", "/login", think_s=(5.0, 30.0), body=BodySpec("form", {
                "username": ValueSpec("name", {"n": 50000, "tag": "pub", "lo": 3, "hi": 8}),
                "password": ValueSpec("hex", {"lo": 8, "hi": 16})},
                SizeSpec([(1.0, 0.3 * KB, 0.8 * KB)], (0.3 * KB, 0.8 * KB))), status={302: 1.0}),
            Step("GET", "/news/{id}", repeat=(1, 10), think_s=(10.0, 120.0),
                 resp=SizeSpec([(1.0, 10000.0, 80000.0)])),
            Step("POST", "/comment", p=0.2, think_s=(30.0, 300.0), body=BodySpec("form", {
                "news": ValueSpec("digits", {"lo": 5, "hi": 5}),
                "text": ValueSpec("text", {"lo": 20, "hi": 150})},
                SizeSpec([(1.0, 1500.0, 3000.0)])))], per_day=(1, 2)),
        # ---- automation
    ] + [
        Activity(f"AUTO.monitor.{s_}", s_, "AUTO", ["192.168.9.9"], [_w("00:00", "23:59", "all")],
                 [Step("GET", "/health", think_s=(0.0, 0.0), resp=SizeSpec([(1.0, 180.0, 220.0)]))],
                 period_s=60.0, jitter_s=1.0)
        for s_ in ("oa", "finance", "portal")
    ] + [
        Activity("AUTO.backup", "finance", "AUTO", ["192.168.9.5"], [_w("01:00", "01:01", "all")], [
            Step("POST", "/backup/export", body=BodySpec("json", {
                "scope": ValueSpec("const", {"value": "full"}), "date": ValueSpec("period")},
                SizeSpec([(1.0, 200.0, 400.0)])),
                 resp=SizeSpec([(1.0, 50e6, 200e6)]))]),
    ]
    drifts = [
        Drift("D1", 12, "window", {"activity": "GA.oa.login", "when": [_w(*GA["login_new"])]},
              {"max_allowed_severity": "low"}),
        Drift("D2", 13, "rename", {"ip": GA["rename"][0], "value": GA["rename"][1]},
              {"max_allowed_severity": "info"}),
        Drift("D3", 14, "route", {"activity": "GA.oa.approvals",
                                  "replace": [["/approval/", "/flow/"]]},
              {"max_allowed_severity": "low"}),
        Drift("D4", 12, "growth", {"dept": "PUB", "factor": 1.5, "until_day": 18},
              {"max_allowed_severity": "info"}),
        Drift("D5", 15, "churn", {"dept": "DEV", "lease_h": 12.0},
              {"max_allowed_severity": "info"}),
    ]
    anomalies = _o_anomalies(P, red)
    # a string in a payload namespace (hdr.*, body.kv.*, q.kv.*) is typed `text`
    # (progressive.md §5.4.5: payload values use the shape hierarchy, and P07
    # states a few-valued one as its closed set - the categorical constraint -
    # next to its grammar); §5.3 rule 5 (categorical when distinct <= 256)
    # governs non-payload strings. Typed categorical, the username of the
    # requirement (3 values) would lose its grammar (§16.2 A2).
    sched: List[AttrSchedule] = [
        AttrSchedule(8, "oa", "headers", "x-client-ver", ValueSpec("by_dept", {
            "map": {"GA": "5.2.1", "FIN": "5.2.1", "SALES": ["5.1.9", "5.2.1"]},
            "default": "5.2.1"}), by="dept", cls="informative", type="text",
            until_day=15 if "R10" in real else None),
        AttrSchedule(10, ["oa", "portal"], "meta", "waf.score", ValueSpec("int", {"lo": 0, "hi": 2}),
                     by="noise", cls="anomaly_signal", type="ordinal"),
    ]
    sched += synthetic_attrs(n_meta, day=3, systems=["oa", "finance", "crm", "portal"])
    if independent_attrs:
        sched += synthetic_attrs(independent_attrs, day=1, systems=["oa", "finance", "crm", "portal"],
                                 prefix="z", independent=True)
    # who arms (progressive.md §16.2 A1): the who arm is the granularity at
    # which behaviour is conditioned on who, so the right arms are the levels
    # whose held-out behaviour gain is within 5 % (+ 0.01 bits/event) of the
    # best (pmetrics.who_arm_utilities, the oracle of PG8's third clause; the
    # same sets on seeds 0-2): oa - grp / prefix (every department owns its
    # /24s, so /24 refines the departments; per IP loses 0.8 bits/event because
    # 研发's pool users re-address daily); finance - ip (the approver and the
    # bookkeepers do different things inside one department); crm - grp /
    # prefix (one department whose members behave alike); code, mail - grp /
    # prefix / reg (the region is the 研发 DHCP scope, the same partition);
    # portal - prefix (public visitors: per-IP behaviour models lose 2 bits/event)
    strategy = {
        "oa": {"who": ["grp", "prefix"], "P07": ["on"], "P08": ["on"], "P10": ["on"]},
        "finance": {"who": ["ip"], "P07": ["on"], "P08": ["on"], "P10": ["on"]},
        "crm": {"who": ["grp", "prefix"], "P07": ["on"], "P08": ["on", "off"],
                "P10": ["on", "off"]},
        "code": {"who": ["grp", "prefix", "reg"], "P07": ["off"], "P08": ["off"], "P10": ["on", "off"]},
        "mail": {"who": ["grp", "prefix", "reg"], "P07": ["off"], "P08": ["off"], "P10": ["on", "off"]},
        "portal": {"who": ["prefix"], "P07": ["on"], "P08": ["off"],
                   "P10": ["on", "off"]},
    }
    config: Dict[str, Any] = {
        "who_group_names": [{"name": "综合部", "ips": list(GA["ips"])},
                            {"name": "财务部", "ips": list(FIN["ips"])},
                            {"name": "销售部", "ips": list(SALES["ips"])},
                            {"name": "研发", "cidrs": [P["DEV"]["pool"][0]]}],
        "dhcp_scopes": [{"cidr": P["DEV"]["pool"][0], "lease_s": P["DEV"]["pool"][2] * 3600.0,
                         "name": "研发 DHCP"}],
        "sensitive_patterns": [r"^/admin(/|$)"],
        "progressive": {"enabled": True},
    }
    if P["ip_classes"]:
        config["ip_classes"] = [{"name": f"public-{i}", "cidrs": [c]}
                                for i, (c, _) in enumerate(P["regions"])]
    perts: List[Perturbation] = []
    spec = OrgSpec(variant, O_START, int(n_days), "Asia/Shanghai",
                   {"holidays": holidays, "makeup_workdays": makeup}, systems, depts, acts,
                   drifts, anomalies, sched, int(portal_n), list(P["regions"]), perts, {},
                   strategy, 0.05, oncall, config, variant)
    if real:
        _apply_real(spec, real, r1_trusted)
    return spec


def synthetic_attrs(n: int, day: int, systems: List[str], prefix: str = "f",
                    independent: bool = False) -> List[AttrSchedule]:
    """n synthetic attributes: 20 % informative (department- or route-dependent),
    67 % noise, 13 % constant (O-scale); `independent` makes all of them noise
    independent of every truth constraint (O-red false-split probe)."""
    out: List[AttrSchedule] = []
    n_inf = 0 if independent else int(round(0.20 * n))
    n_const = 0 if independent else int(round(0.13 * n))
    for i in range(n):
        name = f"{prefix}{i + 1:03d}" if not independent else f"{prefix}{i + 1:02d}"
        numeric = (i % 2 == 0)
        if i < n_inf:
            if i % 4 < 2:
                v = ValueSpec("by_dept", {"map": {d: f"{d.lower()}-{i % 7}" for d in
                                                  ("GA", "FIN", "SALES", "DEV", "PUB")},
                                          "default": "x"})
                out.append(AttrSchedule(day, systems, "meta", name, v, "dept", "informative",
                                        "categorical"))
            else:
                v = ValueSpec("by_route", {"lo": 0.0, "hi": 1000.0, "sd": 2.0})
                out.append(AttrSchedule(day, systems, "meta", name, v, "route", "informative",
                                        "numeric"))
        elif i < n_inf + n_const:
            out.append(AttrSchedule(day, systems, "meta", name,
                                    ValueSpec("const", {"value": f"c-{i}"}), "const", "constant",
                                    "categorical"))
        else:
            v = (ValueSpec("uniform", {"lo": 0.0, "hi": 1000.0}) if numeric else
                 ValueSpec("choice", {"values": [f"v{j}" for j in range(8)]}))
            out.append(AttrSchedule(day, systems, "meta", name, v, "noise",
                                    "independent" if independent else "noise",
                                    "numeric" if numeric else "categorical"))
    return out


def _o_anomalies(P: Dict[str, Any], red: bool) -> List[Anomaly]:
    GA, FIN, SALES = P["GA"], P["FIN"], P["SALES"]
    ga1, ga2, ga3 = GA["ips"][0], GA["ips"][1], GA["ips"][2]
    rose = GA["users"][1]
    fin_ap, lucy = FIN["approver"], FIN["users"][FIN["ips"].index(FIN["approver"])]
    s25 = SALES["ips"][5]
    s30 = SALES["ips"][10]
    s33 = SALES["ips"][13 % len(SALES["ips"])]
    fin_net = FIN["approver"].rsplit(".", 1)[0]
    appr_steps = [Step("GET", "/fin/approval/list", think_s=(20.0, 60.0)),
                  Step("POST", "/fin/approval/{id}/approve", think_s=(20.0, 60.0), body=BodySpec("form", {
                      "voucher": ValueSpec("digits", {"lo": 8, "hi": 8}),
                      "amount": ValueSpec("decimal", {"lo": 100.0, "hi": 50000.0}),
                      "opinion": ValueSpec("const", {"value": "同意"}),
                      "sign": ValueSpec("hex", {"lo": 32, "hi": 32})},
                      SizeSpec([(1.0, 0.9 * KB, 1.6 * KB)])), status={302: 1.0})]
    login = _login_steps("/login", _oa_login_fields(), GA["login_size"])
    sev = lambda req, tgt=None, **kw: {"required_severity": req, "target_severity": tgt or req, **kw}
    return [
        Anomaly("A1", 16, "10:30", "session", ga2, {"system": "finance", "steps": appr_steps},
                dict(sev("medium", "high"), expected_types=["who"],
                     expected_flags=["outsider_group", "system_new"],
                     expected_axes=["privilege", "lateral"], perturbed_features=["net.src"])),
        Anomaly("A2", 17, "09:10", "session", ga1, {
            "system": "oa", "steps": login, "until_day": 21, "username": rose},
            dict(sev("medium"), expected_types=["content"],
                 expected_flags=["cross_binding", "concurrent_use"], expected_axes=["credential"],
                 perturbed_features=["body.kv.username"],
                 non_adoption={"ip": ga1, "attr": "body.kv.username", "value": GA["users"][0],
                               "day": 21})),
        Anomaly("A3", 18, "11:00", "session", s25, {
            "system": "oa", "steps": [Step("POST", "/login", BodySpec("form", _oa_login_fields(),
                                                                      SizeSpec([(1.0, 12 * KB, 12 * KB)])),
                                           status={200: 1.0})],
            "username": "admin' OR '1'='1", "meta": {"waf.score": 14}},
            dict(sev("medium"), expected_types=["content"], expected_flags=["injection_shape"],
                 expected_axes=["content"], perturbed_features=["body.kv.username", "body.len"])),
        Anomaly("A4", 18, "03:05", "session", ga3, {"system": "oa", "steps": login},
                dict(sev("medium"), expected_types=["when"], expected_axes=["temporal"],
                     perturbed_features=["ctx.tod_min"])),
        Anomaly("A5", 19, "17:02", "session", ga2, {
            "system": "oa", "suppress": ["GA.oa.report"],
            "steps": [Step("POST", "/report/generate", body=BodySpec("json", {
                "dept": ValueSpec("const", {"value": "GA"}), "period": ValueSpec("period"),
                "items": ValueSpec("list", {"lo": 10, "hi": 40, "elem": ValueSpec(
                    "text", {"lo": 10, "hi": 40})})}, SizeSpec([(1.0, 20 * KB, 60 * KB)])))]},
            dict(sev("low"), expected_types=["seq"], expected_axes=["sequence"],
                 perturbed_features=["ctx.prev_act"])),
        Anomaly("A6", 19, "14:00", "session", s30, {
            "system": "oa", "steps": [Step("GET", "/admin/export",
                                           resp=SizeSpec([(1.0, 200000.0, 900000.0)]))]},
            dict(sev("medium"), expected_types=["novel"], expected_axes=["categorical", "privilege"],
                 perturbed_features=["http.route"])),
        Anomaly("A7", 20, "20:00", "burst", "10.60.7.7" if not red else "10.70.7.7", {
            "system": "portal", "n": 400, "span_min": 60, "status": {401: 0.9, 302: 0.1},
            "step": Step("POST", "/login", BodySpec("form", {
                "username": ValueSpec("name", {"n": 50000, "tag": "stuff", "lo": 3, "hi": 8}),
                "password": ValueSpec("hex", {"lo": 8, "hi": 16})},
                SizeSpec([(1.0, 0.3 * KB, 0.8 * KB)])))},
            dict(sev("medium"), expected_types=["content"], expected_flags=["rate.ip_h"],
                 expected_axes=["credential", "volume"], perturbed_features=["rate.ip_h"])),
        Anomaly("A8", 19, "10:15", "session", f"{fin_net}.99", {
            "system": "finance", "username": lucy,
            "steps": _login_steps("/fin/login", {"username": ValueSpec("bound"),
                                                 "password": ValueSpec("hex", {"lo": 8, "hi": 16}),
                                                 "otp": ValueSpec("digits", {"lo": 6, "hi": 6})},
                                  SizeSpec([(1.0, 0.6 * KB, 1.2 * KB)]), "/fin/home") + appr_steps,
            "ensure": {"activity": "FIN.finance.approval", "actor": fin_ap, "at": "10:05"}},
            dict(sev("medium", "high"), expected_types=["who", "content"],
                 expected_flags=["unknown_ip", "concurrent_use"],
                 expected_axes=["privilege", "credential"], perturbed_features=["net.src",
                                                                                "body.kv.username"],
                 nominal_day=20, bound_source=fin_ap)),
        Anomaly("A9", 11, "10:30", "session", s33, {
            "system": "finance", "until_day": 21,
            "steps": [Step("GET", "/fin/approval/list", resp=SizeSpec([(1.0, 6000.0, 30000.0)]))]},
            dict(sev("low"), expected_types=["who"], expected_flags=["outsider_group"],
                 expected_axes=["privilege"], perturbed_features=["net.src"],
                 check_days=[11, 21], heavy_node={"system": "finance", "method": "GET",
                                                  "route": "/fin/approval/list"})),
        Anomaly("A10", 21, "09:00", "burst", "203.0.113.0/24", {
            "system": "portal", "n": 150, "span_min": 180,
            "ips": [f"203.0.113.{i}" for i in range(10, 60)],
            "step": Step("POST", "/comment", BodySpec("form", {
                "news": ValueSpec("digits", {"lo": 5, "hi": 5}),
                "text": ValueSpec("links", {"lo": 1900, "hi": 2100})},
                SizeSpec([(1.0, 2 * KB, 2.3 * KB)])))},
            dict(sev("low"), expected_types=["who", "content"], expected_axes=["content"],
                 perturbed_features=["body.len", "body.kv.text"], who_level="reg")),
    ]


def _apply_real(spec: OrgSpec, real: Set[str], r1_trusted: bool) -> None:
    """Pack O-real items R1-R13 (§11.4)."""
    acts, depts = spec.activities, spec.departments
    oa_login = _oa_login_fields()
    other_login = SizeSpec([(1.0, 0.6 * KB, 1.4 * KB)], (0.6 * KB, 1.4 * KB))
    if "R1" in real:
        spec.perturbations.append(Perturbation("R1", 8, "snat_proxy", {
            "system": "oa", "proxy": "192.168.100.99"}, {
            "trusted": r1_trusted, "expected": ({"pg1_oa_within": 0.05} if r1_trusted else
                                                {"snat_suspect_within_days": 1, "who": "none",
                                                 "no_who_violation_on": "192.168.100.99",
                                                 "view_hint": True})}))
        if r1_trusted:
            spec.config.setdefault("progressive", {})["trusted_proxies"] = ["192.168.100.99/32"]
        spec.strategy["oa"] = ({"who": ["grp", "prefix"], "P07": ["on"], "P08": ["on"], "P10": ["on"]}
                               if r1_trusted else {"who": ["none"], "P07": ["on"],
                                                   "P08": ["off", "on"], "P10": ["on", "off"]})
    if "R2" in real:
        users = [pool_name("nat-branch", i, 4, 7) for i in range(10)]
        depts.append(Department("SBR", "销售部分支(NAT)", ["192.168.30.1"], kind="nat",
                                nat_ip="192.168.30.1", n=10, shared_users=users))
        acts += [Activity("SBR.oa.login", "oa", "SBR", "each", [_w("08:30", "09:30")],
                          _login_steps("/login", oa_login, other_login)),
                 Activity("SBR.oa.documents", "oa", "SBR", "each", [_w("09:30", "17:30")],
                          _docs_steps(), per_day=(1, 2))]
        spec.perturbations.append(Perturbation("R2", 1, "nat_branch", {"ip": "192.168.30.1"}, {
            "expected": {"shared_item": "shared:192.168.30.1", "no_ip_binding": "192.168.30.1",
                         "sess_bindings_min": 8, "users": users}}))
    if "R3" in real:
        fin = next(d for d in depts if d.code == "FIN")
        old = fin.ips[1]
        new = old.rsplit(".", 1)[0] + ".51"
        fin.readdress[old] = (9, new)
        spec.perturbations.append(Perturbation("R3", 9, "readdress", {"old": old, "new": new}, {
            "expected": {"flag": "readdress_candidate", "max_severity": "low",
                         "no_incident_ge": "medium", "old_stale": old, "group": "FIN"}}))
    if "R4" in real:
        depts.append(Department("GAT", "综合部共享终端", ["192.168.1.40"], kind="shared",
                                shared_users=["jack2", "rose2"]))
        acts += [Activity("GAT.oa.login", "oa", "GAT", "each", [_w("09:00", "09:21")],
                          _login_steps("/login", oa_login, SizeSpec(
                              [(1.0, 1 * KB, 2 * KB)], (1 * KB, 2 * KB)))),
                 Activity("GAT.oa.documents", "oa", "GAT", "each", [_w("09:30", "16:30")],
                          _docs_steps(), per_day=(1, 3))]
        spec.anomalies.append(Anomaly("R4x", 17, "09:40", "session", "192.168.1.40", {
            "system": "oa", "steps": _login_steps("/login", oa_login, SizeSpec([(1.0, 1 * KB, 2 * KB)])),
            "username": "tony3"}, {"required_severity": "low", "target_severity": "low",
                                   "expected_types": ["content"], "expected_axes": ["credential"],
                                   "perturbed_features": ["body.kv.username"],
                                   "expected_flags": ["set_binding"]}))
        spec.perturbations.append(Perturbation("R4", 1, "shared_terminal", {"ip": "192.168.1.40"}, {
            "expected": {"set_binding": ["jack2", "rose2"], "third_name": "tony3",
                         "third_severity": "low"}}))
    if "R5" in real:
        ips = ["10.9.0.1", "10.9.0.2", "10.9.0.3"]
        depts.append(Department("SVC", "服务账号", ips, {ip: "svc_sync" for ip in ips},
                                kind="service", stacks="lib"))
        acts.append(Activity("SVC.finance.sync", "finance", "SVC", "each", [_w("00:00", "23:59", "all")], [
            Step("POST", "/fin/sync", body=BodySpec("form", {
                "username": ValueSpec("bound"), "batch": ValueSpec("digits", {"lo": 6, "hi": 6}),
                "token": ValueSpec("hex", {"lo": 40, "hi": 40})}, SizeSpec([(1.0, 400.0, 700.0)])))],
            period_s=600.0, jitter_s=5.0))
        spec.perturbations.append(Perturbation("R5", 1, "service_account", {"ips": ips}, {
            "expected": {"reverse_set": {"svc_sync": ips}, "facet": "automation",
                         "no_who_violation": ips}}))
    if "R7" in real:
        fin = next(d for d in depts if d.code == "FIN")
        acts.append(Activity("FIN.finance.close", "finance", "FIN", [fin.ips[0]],
                             [_w("16:00", "17:00", mend=True)], [
            Step("POST", "/fin/close", body=BodySpec("json", {
                "period": ValueSpec("period"), "mode": ValueSpec("const", {"value": "month"})},
                SizeSpec([(1.0, 500.0, 900.0)])))]))
        spec.perturbations.append(Perturbation("R7", 1, "month_end", {}, {
            "expected": {"route": "/fin/close", "no_novel_ge_on_second": "medium"}}))
    if "R8" in real:
        spec.perturbations.append(Perturbation("R8", 22, "holiday", {"days": [22, 28]}, {
            "expected": {"no_pattern_absent_ge": "low", "no_drift_acceptance": True,
                         "intact_day": 29}}))
    if "R9" in real:
        spec.systems.append(SystemSpec("oa-r2", "192.168.100.101:8080", "clear", "oa.corp.local",
                                       family="oa"))
        spec.families["oa"] = ["oa", "oa-r2"]
        spec.perturbations.append(Perturbation("R9", 10, "replica", {
            "system": "oa", "replica": "oa-r2", "share": 0.5}, {
            "expected": {"family": "oa", "joined_by_day": 12, "no_novel_storm": True}}))
    if "R10" in real:
        spec.perturbations.append(Perturbation("R10", 15, "attr_gone", {"name": "hdr.x-client-ver"}, {
            "expected": {"attribute_gone": "hdr.x-client-ver", "within_normal_days": 1}}))
    if "R11" in real:
        spec.anomalies.append(Anomaly("R11", 13, "14:00", "burst", "10.60.99.9", {
            "system": "portal", "n": 5000, "span_min": 120, "status": {404: 1.0},
            "random_paths": True, "step": Step("GET", "/x")},
            {"required_severity": "info", "target_severity": "info", "expected_types": ["novel"],
             "expected_axes": ["categorical"], "perturbed_features": ["http.route"],
             "expected_adaptation": {"node_growth_max": 0.05, "no_pattern_by_day": 21}}))
    if "R12" in real:
        spec.perturbations.append(Perturbation("R12", 1, "sampling", {
            "system": "portal", "rate": 4, "drop": 0.05}, {
            "expected": {"pg1_portal_within": 0.05, "observed_marker": "观测到的"}}))
    if "R13" in real:
        members = []
        for b in range(1, 13):
            sid = f"crm-{b:02d}"
            spec.systems.append(SystemSpec(sid, f"192.168.110.{b}:8080", "clear", "crm.corp.local",
                                           family="crm-br"))
            members.append(sid)
            ips = [f"192.168.111.{(b - 1) * 5 + i + 1}" for i in range(5)]
            depts.append(Department(f"BR{b:02d}", f"分支{b}", ips,
                                    {ip: pool_name(f"br{b}", i, 3, 7) for i, ip in enumerate(ips)}))
            acts.append(Activity(f"BR{b:02d}.crm", sid, f"BR{b:02d}", "each", [_w("09:00", "18:00")], [
                Step("GET", "/crm/customer/{id}", repeat=(1, 3), think_s=(10.0, 120.0)),
                Step("POST", "/crm/visit", p=0.6, think_s=(60.0, 600.0), body=BodySpec("form", {
                    "customer": ValueSpec("digits", {"lo": 6, "hi": 6}),
                    "note": ValueSpec("text", {"lo": 10, "hi": 120})},
                    SizeSpec([(1.0, 1200.0, 2400.0)])))], per_day=(2, 5)))
        spec.families["crm-br"] = members
        spec.perturbations.append(Perturbation("R13", 1, "branches", {"systems": members}, {
            "expected": {"family": members, "memory_ratio_max": 2.0}}))


def build_servers_org(n_days: int = 7, n_families: int = 12, members: int = 20,
                      singletons: int = 30, idle: int = 30,
                      n_systems: Optional[int] = None) -> OrgSpec:
    """Pack O-servers: families x members (replicas share users, branches have
    their own 5 IPs) + singleton systems + idle systems (traffic on day 1 only).
    `n_systems` sizes the pack at a fixed number of families (PG4: 20, 100,
    300 systems in 12 families): ~80 % family members, the rest split evenly
    between singletons and idle systems."""
    if n_systems is not None:
        members = max(1, int(round(0.8 * n_systems / n_families)))
        rest = max(0, n_systems - members * n_families)
        singletons, idle = rest - rest // 2, rest // 2
    systems: List[SystemSpec] = []
    depts: List[Department] = []
    acts: List[Activity] = []
    families: Dict[str, List[str]] = {}

    def app_steps(app: str) -> List[Step]:
        return [Step("GET", f"/{app}/list", think_s=(5.0, 60.0)),
                Step("GET", f"/{app}/item/{{id}}", repeat=(1, 3), think_s=(5.0, 60.0)),
                Step("POST", f"/{app}/item/{{id}}/save", p=0.5, think_s=(20.0, 120.0),
                     body=BodySpec("form", {"v": ValueSpec("alnum", {"lo": 4, "hi": 12})},
                                   SizeSpec([(1.0, 300.0, 1200.0)])))]

    for f in range(n_families):
        app = f"app{f:02d}"
        fam = f"fam{f:02d}"
        mem = []
        for m in range(members):
            sid = f"{app}-{m:02d}"
            systems.append(SystemSpec(sid, f"10.200.{f}.{m + 1}:8080", "clear", f"{app}.corp.local",
                                      family=fam))
            mem.append(sid)
        families[fam] = mem
        if f < n_families // 2:            # replicas: one user department load-balanced
            ips = [f"10.201.{f}.{i}" for i in range(1, 11)]
            depts.append(Department(f"U{f:02d}", f"用户{f}", ips,
                                    {ip: pool_name(fam, i, 3, 7) for i, ip in enumerate(ips)}))
            acts.append(Activity(f"U{f:02d}.{app}", mem, f"U{f:02d}", "each",
                                 [_w("09:00", "17:30")], app_steps(app), per_day=(2, 4)))
        else:                              # branches: each member its own 5 IPs
            for m, sid in enumerate(mem):
                ips = [f"10.202.{f}.{m * 5 + i + 1}" for i in range(5)]
                code = f"B{f:02d}{m:02d}"
                depts.append(Department(code, f"分支{f}-{m}", ips,
                                        {ip: pool_name(code, i, 3, 7) for i, ip in enumerate(ips)}))
                acts.append(Activity(f"{code}.{app}", sid, code, "each", [_w("09:00", "17:30")],
                                     app_steps(app), per_day=(1, 3)))
    for i in range(singletons):
        sid = f"single{i:02d}"
        systems.append(SystemSpec(sid, f"10.210.0.{i + 1}:8080", "clear", f"{sid}.corp.local"))
        ips = [f"10.211.{i}.{k}" for k in range(1, 6)]
        depts.append(Department(f"S{i:02d}", f"单系统用户{i}", ips))
        acts.append(Activity(f"S{i:02d}.{sid}", sid, f"S{i:02d}", "each", [_w("09:00", "17:30")],
                             app_steps(f"s{i:02d}"), per_day=(1, 3)))
    for i in range(idle):
        sid = f"idle{i:02d}"
        systems.append(SystemSpec(sid, f"10.220.0.{i + 1}:8080", "clear", f"{sid}.corp.local"))
        depts.append(Department(f"I{i:02d}", f"闲置{i}", [f"10.221.0.{i + 1}"]))
        acts.append(Activity(f"I{i:02d}.{sid}", sid, f"I{i:02d}", "each",
                             [_w("10:00", "11:00", dow=None)], [Step("GET", "/status", repeat=(3, 6))],
                             p_day=1.0))
    spec = OrgSpec(f"O-servers-{len(systems)}", O_START, int(n_days), "Asia/Shanghai",
                   {"holidays": [], "makeup_workdays": []}, systems, depts, acts, families=families,
                   strategy={}, config={"progressive": {"enabled": True}}, variant="O-servers",
                   portal_n=0)
    spec.config["idle_systems"] = [f"idle{i:02d}" for i in range(idle)]
    spec.config["idle_until_day"] = 1
    return spec


# --------------------------------------------------------------------------- #
# The generator
# --------------------------------------------------------------------------- #
# --------------------------------------------------------------------------- #
# Per-step arrival law (truth; progressive.md §11.5)
# --------------------------------------------------------------------------- #
STEP_LAW_N = 20000               # simulated sessions per window spec
STEP_LAW_Q = 129                 # quantile-function points published per segment
STEP_LAW_MASS = 0.99             # a step's window = the central 99 % of its arrival law
_STEP_LAW_CACHE: Dict[str, Dict[str, Any]] = {}


def _session_offsets(steps: Sequence[Step], k: int, n: int, r: np.random.Generator
                     ) -> Tuple[np.ndarray, np.ndarray]:
    """Offsets (s) from the session start of every arrival of step k in n
    simulated sessions, and the index of the session each belongs to. The
    same program as OrgGenerator._session: a step with p < 1 is skipped with
    probability 1 - p, it is repeated U{repeat} times, and every record but
    the session's first is preceded by a think time U(think_s) of its step."""
    t = np.zeros(n)
    started = np.zeros(n, dtype=bool)
    offs: List[np.ndarray] = []
    sess: List[np.ndarray] = []
    idx = np.arange(n)
    for kk, st in enumerate(steps[: k + 1]):
        on = r.random(n) < st.p if st.p < 1.0 else np.ones(n, dtype=bool)
        lo, hi = int(st.repeat[0]), int(st.repeat[1])
        reps = r.integers(lo, hi + 1, n)
        for j in range(hi):
            sel = on & (j < reps)
            think = sel & started
            t[think] += r.uniform(float(st.think_s[0]), float(st.think_s[1]), int(think.sum()))
            if kk == k:
                offs.append(t[sel].copy())
                sess.append(idx[sel])
            started |= sel
    if not offs:
        return np.zeros(0), np.zeros(0, dtype=int)
    return np.concatenate(offs), np.concatenate(sess)


def step_arrival_law(when: Sequence[WindowSpec], steps: Sequence[Step], k: int) -> Dict[str, Any]:
    """The arrival law of step k of an activity, as the generator produces it:
    a session starts at a minute drawn from its window spec's arrival law
    (uniform, or N(mid, width / 4) clipped to the window), then the steps
    follow with their think times (_session_offsets). Per window spec the law
    is simulated once (STEP_LAW_N sessions, a fixed RNG: the law is a property
    of the program, not of the seed) and published as

      segments  {daytype: [[weight, [q_0 .. q_1]]]}: the quantile function at
                STEP_LAW_Q equally spaced probabilities, weight = the spec's
                share of sessions (proportional to its width, as _plan_day
                draws it) x the step's mean arrivals per session
      windows   {daytype: [[a, b]]}: per spec the central STEP_LAW_MASS of the
                law, floor / ceil to minutes (step 0 of a non-repeated step:
                the activity's window itself), merged when they overlap.

    Before (round 2) a step's windows were the activity's window widened by
    the sums of the think-time bounds and the scorer drew held-out minutes
    uniformly inside them."""
    key = json.dumps([[[w.daytypes, w.start, w.end, w.arrival] for w in when], [[s.method, s.route_fmt, s.p, list(s.repeat), list(s.think_s)]
                                           for s in steps[: k + 1]], int(k)], sort_keys=True, default=str)
    hit = _STEP_LAW_CACHE.get(key)
    if hit is not None:
        return copy.deepcopy(hit)
    segs: Dict[str, List[List[Any]]] = {"workday": [], "nonworkday": []}
    wins: Dict[str, List[List[int]]] = {"workday": [], "nonworkday": []}
    qs = np.linspace(0.0, 1.0, STEP_LAW_Q)
    tail = 0.5 * (1.0 - STEP_LAW_MASS)
    for i, w in enumerate(when):
        r = np.random.default_rng([crc(key) & 0x7FFFFFFF, i])
        m0, m1 = float(w.m0), float(w.m1)
        if w.arrival == "normal":
            mu, sd = 0.5 * (m0 + m1), (m1 - m0) / 4.0
            start = np.clip(r.normal(mu, sd, STEP_LAW_N), m0, m1)
        else:
            start = r.uniform(m0, m1, STEP_LAW_N)
        off, sid = _session_offsets(steps, k, STEP_LAW_N, r)
        if off.size == 0:
            continue
        x = np.minimum(start[sid] + off / 60.0, 1440.0)
        q = np.round(np.quantile(x, qs), 2)
        a = int(math.floor(float(np.quantile(x, tail))))
        b = int(math.ceil(float(np.quantile(x, 1.0 - tail))))
        exact = (k == 0 and steps[0].repeat[1] <= 1)
        if exact:                                   # the session start law itself
            a, b = int(w.m0), int(min(1440, w.m1))
        if w.m1 >= 1439 and w.m0 == 0:
            a, b = 0, 1440
        weight = round(max(1.0, m1 - m0) * off.size / STEP_LAW_N, 4)
        for dt_ in ("workday", "nonworkday"):
            if w.daytypes in (dt_, "all"):
                wins[dt_].append([a, b])
                segs[dt_].append([weight, [float(v) for v in q]])
    for dt_ in wins:
        merged: List[List[int]] = []
        for a, b in sorted(wins[dt_]):
            if merged and a <= merged[-1][1]:
                merged[-1][1] = max(merged[-1][1], b)
            else:
                merged.append([a, b])
        wins[dt_] = merged
    out = {"windows": wins, "segments": segs}
    _STEP_LAW_CACHE[key] = out
    return copy.deepcopy(out)


@dataclass
class _Eff:
    when: List[WindowSpec]
    steps: List[Step]


class OrgGenerator:
    """Plans and emits the organisation's traffic; publishes its truth.

    OrgGenerator(spec, seed, pack_name, clock=None, ev_sample=True)
    step(t0, t1, aggregated) -> List[Observation]
    ptruth() -> {pattern_truth, group_truth, strategy_truth, system_truth, attr_truth,
                 opportunities, who_log, stats, days}
    truth_rows() -> gen.truth rows (anomalies, drifts, perturbations)
    """

    def __init__(self, spec: OrgSpec, seed: int = 0, pack_name: str = "O",
                 clock: Optional[Clock] = None, ev_sample: bool = True) -> None:
        self.spec = spec
        self.seed = int(seed)
        self.pack = str(pack_name)
        self.clock = clock or Clock(spec.tz, spec.calendar)
        self.ev_sample = bool(ev_sample)
        self.start = spec.start_date
        self.n_days = int(spec.n_days)
        self.sys = {s.id: s for s in spec.systems}
        self.depts = {d.code: d for d in spec.departments}
        self.oncall = {_dt.date.fromisoformat(d) for d in spec.oncall_holidays}
        self.drift = {d.id: d for d in spec.drifts}
        self.pert = {p.id: p for p in spec.perturbations}
        self._evs_rngs: Dict[str, np.random.Generator] = {}
        self._rngs: Dict[str, np.random.Generator] = {}
        self._heap: List[Tuple[float, int, OEv]] = []
        self._seq = 0
        self._planned = 0
        self._lease_cache: Dict[Tuple[str, int], List[str]] = {}
        self.actors: Dict[str, Actor] = {}
        self.members: Dict[str, List[str]] = {}
        self._build_actors()
        self._public_by_day: Dict[int, List[str]] = {}
        self._index_public()
        self._mend = self._month_end_dates()
        self.opportunities: Dict[str, Dict[str, Dict[str, int]]] = {}
        # {tid: {attr: {date: [min, max]}}} of the benign events emitted: the part
        # of a row's content range the data could show (evaluator round 3)
        self.extremes: Dict[str, Dict[str, Dict[str, List[float]]]] = {}
        self.who_log: Dict[str, Dict[str, Dict[str, int]]] = {}
        # (source, action, local hour) counts per system and date: the offline
        # who-arm utility of PG8 (held-out code of who + behaviour given who,
        # pmetrics.who_arm_utilities) needs what each source did, not only how often
        self.act_log: Dict[str, Dict[str, Dict[str, int]]] = {}
        self.attr_first: Dict[str, float] = {}
        self.attr_tick: Dict[str, float] = {}
        self.stats = {"events": 0, "records": 0, "benign": 0, "anomalous": 0, "dropped": 0}
        self._anom_events: Dict[int, List[OEv]] = {}
        self._suppress: Set[Tuple[int, str, str]] = set()
        self._ensure: Dict[int, List[Tuple[str, str, str]]] = {}
        self._anom_span: Dict[str, Tuple[float, float, float]] = {}
        self._tids: Dict[Tuple[str, int, int], str] = {}
        self._ptruth_rows: List[Dict[str, Any]] = self._build_pattern_truth()
        self._plan_anomalies()

    # ------------------------------------------------------------------ time
    def day_start(self, d: int) -> float:
        return self.clock.epoch(self.start + _dt.timedelta(days=d - 1), 0.0)

    def day_of(self, ts: float) -> int:
        return (self.clock.local(ts).date() - self.start).days + 1

    def date_of(self, d: int) -> _dt.date:
        return self.start + _dt.timedelta(days=d - 1)

    def _minute_ts(self, d: int, minute: float) -> float:
        return self.day_start(d) + minute * 60.0     # no DST in the org's zone

    def _month_end_dates(self) -> Set[_dt.date]:
        out: Set[_dt.date] = set()
        d0, d1 = self.start, self.start + _dt.timedelta(days=self.n_days + 40)
        by_month: Dict[Tuple[int, int], List[_dt.date]] = {}
        d = d0 - _dt.timedelta(days=40)
        while d <= d1:
            if self.clock.day_kind(d)[0]:
                by_month.setdefault((d.year, d.month), []).append(d)
            d += _dt.timedelta(days=1)
        for days in by_month.values():
            out |= set(sorted(days)[-2:])
        return out

    # ---------------------------------------------------------------- actors
    def _stack_for(self, dept: Department, aid: str) -> Stack:
        r = rng_for("orgstack", dept.code, aid)
        if dept.stacks == "dev":
            v = int(r.integers(125, 129))
            return Stack(f"ff{v}-linux", UA_FF_LIN.format(v=v), JA3_A.replace("-21", "") + "-21",
                         64, 29200)
        if dept.stacks == "lib":
            return fixed_stack("python-requests", "python-requests/2.31.0", f"org|{aid}")
        if dept.stacks == "public":
            return unique_stack(r, f"pub{crc(aid) % 12}")
        return ORG_STANDARD if r.random() < 0.8 else unique_stack(r, f"org-{aid}")

    def _build_actors(self) -> None:
        for d in self.spec.departments:
            ids: List[str] = []
            if d.kind == "public":
                r = rng_for(self.seed, self.pack, "public", d.code)
                n = int(d.n)
                shares = np.asarray([s for _, s in d.regions], dtype=float)
                reg = r.choice(len(shares), size=n, p=shares / shares.sum())
                used: Set[str] = set()
                for i in range(n):
                    net = ipaddress.ip_network(d.regions[int(reg[i])][0], strict=False)
                    while True:
                        ip = str(net.network_address + int(r.integers(2, net.num_addresses - 1)))
                        if ip not in used:
                            used.add(ip)
                            break
                    aid = ip
                    self.actors[aid] = Actor(aid, d.code, ip, stack=self._stack_for(d, aid),
                                             kind="public")
                    ids.append(aid)
            elif d.kind == "pool":
                cidr, n, _ = d.pool
                for i in range(int(n)):
                    aid = f"{d.code}#{i:03d}"
                    self.actors[aid] = Actor(aid, d.code, "", pool_name(f"dev|{d.code}", i, 4, 7),
                                             self._stack_for(d, aid), pool=d.pool, slot=i,
                                             kind="pool")
                    ids.append(aid)
            elif d.kind == "nat":
                for i, u in enumerate(d.shared_users):
                    aid = f"{d.code}#{i:02d}"
                    self.actors[aid] = Actor(aid, d.code, d.nat_ip, u,
                                             self._stack_for(d, f"{aid}|{i}"), kind="nat")
                    ids.append(aid)
            elif d.kind == "shared":
                for i, u in enumerate(d.shared_users):
                    aid = f"{d.code}#{i:02d}"
                    self.actors[aid] = Actor(aid, d.code, d.ips[0], u, self._stack_for(d, d.ips[0]),
                                             parity=i % 2, kind="shared")
                    ids.append(aid)
            else:
                for ip in d.ips:
                    aid = ip
                    self.actors[aid] = Actor(aid, d.code, ip, d.usernames.get(ip, ""),
                                             self._stack_for(d, aid), readdress=d.readdress.get(ip),
                                             kind=d.kind)
                    ids.append(aid)
            self.members[d.code] = ids

    def _index_public(self) -> None:
        """Active days of the public population (Poisson(3) days each) and the
        D4 growth population (active only inside its window)."""
        pub = [d for d in self.spec.departments if d.kind == "public"]
        for d in pub:
            r = rng_for(self.seed, self.pack, "public-days", d.code)
            for aid in self.members[d.code]:
                k = max(1, int(r.poisson(d.active_days)))
                days = r.choice(self.n_days, size=min(k, self.n_days), replace=False) + 1
                self.actors[aid].days = set(int(x) for x in days)
            for dr in self.spec.drifts:
                if dr.kind != "growth" or dr.params.get("dept") != d.code:
                    continue
                d0, d1 = int(dr.day), int(dr.params.get("until_day", self.n_days))
                extra = int(round((float(dr.params.get("factor", 1.5)) - 1.0) * d.n))
                rr = rng_for(self.seed, self.pack, "growth", dr.id)
                shares = np.asarray([s for _, s in d.regions], dtype=float)
                reg = rr.choice(len(shares), size=extra, p=shares / shares.sum())
                span = max(1, d1 - d0 + 1)
                for i in range(extra):
                    net = ipaddress.ip_network(d.regions[int(reg[i])][0], strict=False)
                    ip = str(net.network_address + int(rr.integers(2, net.num_addresses - 1)))
                    if ip in self.actors:
                        continue
                    k = max(1, int(rr.poisson(d.active_days * span / self.n_days * 3.0)))
                    days = rr.choice(span, size=min(k, span), replace=False) + d0
                    self.actors[ip] = Actor(ip, d.code, ip, stack=self._stack_for(d, ip),
                                            days=set(int(x) for x in days), kind="public")
                    self.members[d.code].append(ip)
            for aid in self.members[d.code]:
                for day in self.actors[aid].days or ():
                    self._public_by_day.setdefault(day, []).append(aid)

    def lease_h(self, dept: str, d: int) -> float:
        h = float(self.depts[dept].pool[2]) if self.depts[dept].pool else 24.0
        for dr in self.spec.drifts:
            if dr.kind == "churn" and dr.params.get("dept") == dept and d >= dr.day:
                h = float(dr.params.get("lease_h", h))
        return h

    def _lease_ips(self, cidr: str, idx: int) -> List[str]:
        key = (cidr, idx)
        hit = self._lease_cache.get(key)
        if hit is None:
            hosts = _net_hosts(cidr)
            r = rng_for(self.seed, self.pack, "lease", cidr, idx)
            perm = r.permutation(len(hosts))
            hit = self._lease_cache[key] = [hosts[i] for i in perm.tolist()]
            if len(self._lease_cache) > 256:
                self._lease_cache.pop(next(iter(self._lease_cache)))
        return hit

    def lease_index(self, dept: str, ts: float) -> int:
        d = self.day_of(ts)
        h = self.lease_h(dept, d)
        hour = self.clock.hour(ts)
        return (d - 1) * 1000 + int(hour // h)

    def ip_at(self, a: Actor, ts: float) -> str:
        if a.pool is not None:
            idx = self.lease_index(a.dept, ts)
            return self._lease_ips(a.pool[0], idx)[a.slot]
        if a.readdress is not None and self.day_of(ts) >= a.readdress[0]:
            return a.readdress[1]
        return a.ip

    def username_at(self, a: Actor, d: int) -> str:
        u = a.username
        for dr in self.spec.drifts:
            if dr.kind == "rename" and dr.params.get("ip") == a.aid and d >= dr.day:
                u = str(dr.params["value"])
        return u

    # --------------------------------------------------------- effective spec
    def eff(self, act: Activity, d: int) -> _Eff:
        when, steps = act.when, act.steps
        for dr in self.spec.drifts:
            if d < dr.day or dr.params.get("activity") != act.name:
                continue
            if dr.kind == "window":
                when = list(dr.params["when"])
            elif dr.kind == "route":
                new = []
                for st in steps:
                    st2 = copy.copy(st)
                    for a, b in dr.params.get("replace", []):
                        st2.route_fmt = st2.route_fmt.replace(a, b)
                    new.append(st2)
                steps = new
        return _Eff(when, steps)

    def _who(self, act: Activity, d: int) -> List[Actor]:
        dept = self.depts[act.dept]
        if dept.kind == "public":
            return [self.actors[a] for a in self._public_by_day.get(d, ())]
        ids = self.members[act.dept] if act.who in ("each", "one_of") else [
            a for a in self.members[act.dept] if a in set(act.who)]
        out = [self.actors[a] for a in ids]
        if dept.kind == "shared":
            out = [a for a in out if a.parity is None or d % 2 == a.parity]
        return out

    def _day_ok(self, w: WindowSpec, date: _dt.date) -> bool:
        work = self.clock.day_kind(date)[0]
        if w.daytypes == "workday" and not work:
            return False
        if w.daytypes == "nonworkday" and work:
            return False
        if w.dow is not None and date.weekday() not in w.dow:
            return False
        if w.mend and date not in self._mend:
            return False
        return True

    # ---------------------------------------------------------------- truth
    def _step_windows(self, when: List[WindowSpec], steps: List[Step], k: int
                      ) -> Dict[str, List[List[int]]]:
        """Each step's own windows: per window spec, the central
        STEP_LAW_MASS hull of the step's arrival law (step_arrival_law), merged
        per day type. Before (round 2), the windows were the activity's window
        widened by the sum of the steps' think-time bounds, and the scorer drew
        held-out minutes uniformly in them: a later step's (or a repeated
        record's) arrivals were placed in a tail the generator almost never
        reaches (e.g. mail records 2-8 of a session)."""
        return step_arrival_law(when, steps, k)["windows"]

    def _step_windows_r2(self, when: List[WindowSpec], steps: List[Step], k: int
                         ) -> Dict[str, List[List[int]]]:
        """Round-2 step windows (support bounds); kept for the regression test
        and the before/after re-scoring only."""
        lo_off = 0.0
        hi_off = 0.0
        n_before = 0
        for j, st in enumerate(steps[: k + 1]):
            reps_lo, reps_hi = st.repeat
            for rep in range(reps_hi):
                first = (n_before == 0)
                if not first:
                    hi_off += st.think_s[1]
                    if st.p >= 1.0 and rep < reps_lo and j < k:
                        lo_off += st.think_s[0]
                n_before += 1
        out: Dict[str, List[List[int]]] = {"workday": [], "nonworkday": []}
        for w in when:
            a = int(math.floor(w.m0 + lo_off / 60.0))
            b = int(min(1440, math.ceil(w.m1 + hi_off / 60.0)))
            if w.m1 >= 1439 and w.m0 == 0:
                a, b = 0, 1440
            for dt_ in ("workday", "nonworkday"):
                if w.daytypes in (dt_, "all"):
                    out[dt_].append([a, b])
        for dt_ in out:
            out[dt_] = sorted(out[dt_])
        return out

    def _who_truth(self, act: Activity, d: int) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        dept = self.depts[act.dept]
        acts = self._who(act, d) if dept.kind != "public" else []
        if dept.kind == "public":
            return ({"level": "reg", "value": [[c, float(s)] for c, s in dept.regions],
                     "group": dept.code}, {"regions": [[c, float(s)] for c, s in dept.regions]})
        if dept.kind == "pool":
            return ({"level": "prefix", "value": [[dept.pool[0], 1.0]], "group": dept.code,
                     "alt": ["grp"]}, {"prefix": dept.pool[0]})
        if dept.kind == "shared" and act.who == "each":
            acts = [self.actors[a] for a in self.members[dept.code]]
        ips = sorted({self.ip_at(a, self.day_start(d) + 43200.0) for a in acts})
        weights = {ip: 1.0 / len(ips) for ip in ips} if ips else {}
        if len(ips) <= 8:
            return ({"level": "ip", "value": ips, "group": dept.code,
                     "shared": dept.kind in ("nat", "shared")}, {"members": weights})
        return ({"level": "grp", "value": dept.code, "members": ips, "group": dept.code},
                {"members": weights})

    def _pattern_on(self, act: Activity, k: int, d: int) -> Optional[Dict[str, Any]]:
        e = self.eff(act, d)
        if k >= len(e.steps):
            return None
        st = e.steps[k]
        systems = act.system if isinstance(act.system, list) else [act.system]
        s0 = self.sys[systems[0]]
        who, gen = self._who_truth(act, d)
        if not who.get("value"):
            return None
        tls = st.method == "TLS"
        route = st.route_fmt
        content: Dict[str, Any] = {}
        bindings: Dict[str, Any] = {}
        dept = self.depts[act.dept]
        members = self._who(act, d) if dept.kind != "public" else []
        if dept.kind == "shared":
            members = [self.actors[a] for a in self.members[dept.code]]
        if dept.kind in ("pool", "nat"):
            users = {a.aid: self.username_at(a, d) for a in members if a.username}
        else:
            users = {self.ip_at(a, self.day_start(d) + 43200.0): self.username_at(a, d)
                     for a in members if a.username}
        gen_fields: Dict[str, Any] = {}
        if tls and st.up is not None:
            lo, hi = st.up.support()
            # the observation is the payload plus the TLS record framing (`_render`),
            # so the truth states the observed quantity (evaluator round 3: the
            # learned mail / git bands were exactly truth + 300 and failed PG1)
            f = TLS_UP_FRAMING
            content["net.bytes_up"] = {"band90": [st.up.quantile(0.05) + f, st.up.quantile(0.95) + f],
                                       "range": [lo + f, hi + f], "core": True}
            gen["up"] = st.up.to_dict()
        if st.body is not None and s0.visibility == "clear":
            b = st.body
            if b.size is not None:
                lo, hi = b.size.support()
                content["body.len"] = {"band90": [b.size.quantile(0.05), b.size.quantile(0.95)],
                                       "range": [lo, hi], "core": True}
                gen["size"] = b.size.to_dict()
            # the padding key is observable only when the capture holds it: a JSON
            # body's trailing 'remark' is cut off whenever the body exceeds the
            # capture cap (P00 keeps the keys of the prefix it can parse), so it
            # is required only when no body of the step can exceed BODY_CAP
            # (integration fix: the truth asked for 'remark' on 20-60 KB reports)
            pad_key = []
            if b.size is not None:
                if b.fmt == "form":
                    pad_key = ["viewstate"]
                elif b.size.support()[1] <= BODY_CAP:
                    pad_key = ["remark"]
            keys = sorted(list(b.fields) + pad_key)
            content["body.keys"] = {"required_keys": keys, "core": True}
            for key, vs in b.fields.items():
                gen_fields[key] = vs.to_dict()
                attr = f"body.kv.{key}"
                if vs.kind == "bound":
                    if dept.kind == "shared":
                        vals = sorted({u for u in users.values()} |
                                      {a.username for a in members})
                    else:
                        vals = sorted(set(users.values()))
                    if not vals:
                        continue
                    content[attr] = {"grammar": grammar_of(vals), "closed_values": vals,
                                     "kind": "bound", "core": True}
                    if dept.kind == "shared":
                        bindings[attr] = {ip: sorted(set(vals)) for ip in users}
                    elif dept.kind not in ("nat", "pool", "public") and users:
                        bindings[attr] = dict(sorted(users.items()))
                elif vs.kind == "choice":
                    content[attr] = {"closed_values": sorted(map(str, vs.params["values"])),
                                     "grammar": spec_grammar(vs), "kind": "choice", "core": True}
                elif vs.kind == "const":
                    content[attr] = {"closed_values": [str(vs.params.get("value", ""))],
                                     "kind": "const", "core": False}
                else:
                    g = spec_grammar(vs)
                    if g is not None:
                        content[attr] = {"grammar": g, "kind": vs.kind, "core": False}
            gen["fmt"] = b.fmt
        gen["fields"] = gen_fields
        if any(ValueSpec.from_dict(v).kind == "bound" for v in gen_fields.values()):
            gen["usernames"] = users
        law = step_arrival_law(e.when, e.steps, k)
        windows = law["windows"]
        # arrival law of the step (eval draws held-out minutes from it: the
        # quantile function of each window spec's segment, segments by weight)
        gen["arrival"] = "normal" if any(w.arrival == "normal" for w in e.when) else "uniform"
        gen["arrival_q"] = law["segments"]
        # the bounds every arrival respects (the window widened by the think-time
        # sums: round 2's `windows`); `windows` hold STEP_LAW_MASS of the law
        gen["support"] = self._step_windows_r2(e.when, e.steps, k)
        daytypes = [t for t in ("workday", "nonworkday") if windows[t]]
        period = ("continuous" if act.period_s else
                  "monthly" if any(w.mend for w in e.when) else
                  "weekly" if any(w.dow for w in e.when) else
                  "sporadic" if act.p_day < 1.0 else "daily")
        return {
            "system": systems[0] if len(systems) == 1 else (s0.family or systems[0]),
            "systems": systems, "kind": "txn", "channel": "tls" if tls else "http",
            "method": st.method, "route": route, "act": f"{st.method} {route}",
            "who": who, "daytypes": daytypes, "windows": windows,
            "content": content, "bindings": bindings,
            "group": act.dept, "activity": act.name, "step": k, "period": period,
            "write": st.method in WRITE_METHODS, "optional": st.p < 1.0,
            "gen": gen,
        }

    def _build_pattern_truth(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        idle = set(self.spec.config.get("idle_systems") or [])
        for act in self.spec.activities:
            n_steps = max(len(self.eff(act, d).steps) for d in range(1, self.n_days + 1))
            n_steps = max(n_steps, len(act.steps))
            for k in range(n_steps):
                seg, cur, cur_sig, start = 0, None, None, 1
                for d in range(1, self.n_days + 2):
                    pat = self._pattern_on(act, k, d) if d <= self.n_days else None
                    sig = json.dumps(pat, sort_keys=True, default=str) if pat is not None else None
                    if sig != cur_sig:
                        if cur is not None:
                            rows.append(self._close(cur, act, k, seg, start, d))
                            seg += 1
                        cur, cur_sig, start = pat, sig, d
                    if cur is not None:
                        self._tids[(act.name, k, d)] = self._tid(act.name, k, seg)
        # workflow edges (consecutive mandatory steps of one activity, same segment day)
        by_tid = {r["tid"]: r for r in rows}
        for r in rows:
            k = r["step"]
            if k == 0 or r["optional"]:
                continue
            act = next(a for a in self.spec.activities if a.name == r["activity"])
            d = r["valid_from_day"]
            prev = self._tids.get((act.name, k - 1, d))
            st = self.eff(act, d).steps[k]
            if prev and prev in by_tid and not by_tid[prev]["optional"]:
                r["workflow"] = [[prev, r["tid"], [float(st.think_s[0]), float(st.think_s[1])]]]
        for r in rows:
            r.setdefault("workflow", [])
            if r["system"] in idle:
                r["idle"] = True
        return rows

    @staticmethod
    def _tid(name: str, k: int, seg: int) -> str:
        return f"{name}#{k}" + (f"@{seg}" if seg else "")

    def _close(self, pat: Dict[str, Any], act: Activity, k: int, seg: int, d0: int, d1: int
               ) -> Dict[str, Any]:
        row = dict(pat)
        row["tid"] = self._tid(act.name, k, seg)
        row["lineage"] = f"{act.name}#{k}"
        row["valid_from_day"], row["valid_to_day"] = int(d0), int(d1)
        row["valid_from"] = self.day_start(d0)
        row["valid_to"] = self.day_start(d1) if d1 <= self.n_days else self.day_start(self.n_days + 1)
        return row

    def tid(self, act: str, k: int, d: int) -> Optional[str]:
        return self._tids.get((act, k, d))

    # -------------------------------------------------------------- planning
    def _rng(self, *parts: Any) -> np.random.Generator:
        return rng_for(self.seed, self.pack, "org", *parts)

    def _plan_anomalies(self) -> None:
        for an in self.spec.anomalies:
            if an.day > self.n_days:            # outside a shortened timeline (tests, scale packs)
                continue
            p = an.params
            if p.get("suppress"):
                for d in range(an.day, int(p.get("until_day", an.day)) + 1):
                    for a in p["suppress"]:
                        self._suppress.add((d, a, an.entity))
            if p.get("ensure"):
                en = p["ensure"]
                self._ensure.setdefault(an.day, []).append((en["activity"], en["actor"], en["at"]))
            evs: List[OEv] = []
            for d in range(an.day, min(self.n_days, int(p.get("until_day", an.day))) + 1):
                r = self._rng("anomaly", an.id, d)
                t = self._minute_ts(d, _hm(an.time))
                sysid = p["system"]
                if an.kind == "burst":
                    ips = p.get("ips") or [an.entity]
                    n = int(p.get("n", 100))
                    span = float(p.get("span_min", 60)) * 60.0
                    ts = np.sort(t + r.uniform(0.0, span, n))
                    base = self.actors.get(ips[0])
                    for i, tt in enumerate(ts.tolist()):
                        ip = ips[int(r.integers(0, len(ips)))]
                        stack = self._stack_for(self.depts.get("PUB") or Department("X", "x"), ip)
                        a = Actor(ip, "ANOM", ip, stack=stack)
                        st = copy.copy(p["step"])
                        if p.get("random_paths"):
                            st.route_fmt = "/" + "/".join(_rand_chars(r, "abcdefghijklmnopqrstuvwxyz0123456789", int(
                                r.integers(3, 10))) for _ in range(int(r.integers(1, 4))))
                        if p.get("status"):
                            st.status = dict(p["status"])
                        evs.append(self._render(sysid, st, a, float(tt), r, "", d, None, an.id,
                                                {"username": ""}, meta=p.get("meta")))
                else:
                    base = self.actors.get(an.entity)
                    stack = base.stack if base is not None else self._stack_for(
                        self.depts.get("FIN") or Department("X", "x"), an.entity)
                    a = Actor(an.entity, base.dept if base else "ANOM", an.entity,
                              base.username if base else "", stack)
                    uname = p.get("username")
                    if uname is None and base is not None:
                        uname = self.username_at(base, d)
                    cookie = _rand_chars(r, "0123456789abcdef", 24)
                    tt = t
                    ids: Dict[str, str] = {}
                    for j, st in enumerate(p["steps"]):
                        if j:
                            tt += float(r.uniform(*st.think_s))
                        evs.append(self._render(sysid, st, a, tt, r, cookie, d, None, an.id,
                                                {"username": uname or ""}, ids=ids,
                                                meta=p.get("meta")))
            for e in evs:
                e.anomaly = an.id
            if evs:
                ts = [e.ts for e in evs]
                self._anom_span[an.id] = (min(ts), max(ts), float(len(evs)))
            for e in evs:
                self._anom_events.setdefault(self.day_of(e.ts), []).append(e)

    def _plan_day(self, d: int) -> List[OEv]:
        date = self.date_of(d)
        out: List[OEv] = []
        holiday_oncall = date in self.oncall
        idle = set(self.spec.config.get("idle_systems") or [])
        idle_until = int(self.spec.config.get("idle_until_day", 0))
        for act in self.spec.activities:
            e = self.eff(act, d)
            systems = act.system if isinstance(act.system, list) else [act.system]
            if idle and systems[0] in idle and d > idle_until:
                continue
            wins = [w for w in e.when if self._day_ok(w, date)]
            oncall = False
            if not wins and holiday_oncall:
                wins = [w for w in e.when if w.daytypes == "workday" and
                        (w.dow is None or date.weekday() in w.dow)]
                oncall = bool(wins)
            if not wins:
                continue
            for a in self._who(act, d):
                if (d, act.name, a.aid) in self._suppress:
                    continue
                r = self._rng(d, act.name, a.aid)
                if oncall and r.random() >= self.spec.oncall_p:
                    continue
                if act.p_day < 1.0 and r.random() >= act.p_day:
                    continue
                if act.period_s:
                    out.extend(self._periodic(act, e, a, wins, d, r, systems))
                    continue
                n = int(r.integers(act.per_day[0], act.per_day[1] + 1))
                lens = np.asarray([max(1, w.m1 - w.m0) for w in wins], dtype=float)
                for _ in range(n):
                    w = wins[int(r.choice(len(wins), p=lens / lens.sum()))] if len(wins) > 1 else wins[0]
                    if w.arrival == "normal":
                        mu, sd = 0.5 * (w.m0 + w.m1), (w.m1 - w.m0) / 4.0
                        m = float(np.clip(r.normal(mu, sd), w.m0, w.m1))
                    else:
                        m = float(r.uniform(w.m0, w.m1))
                    sysid = systems[int(r.integers(0, len(systems)))] if len(systems) > 1 else systems[0]
                    out.extend(self._session(act, e, a, self._minute_ts(d, m), r, d, sysid))
        for act_name, aid, at in self._ensure.get(d, []):
            act = next(x for x in self.spec.activities if x.name == act_name)
            r = self._rng(d, act.name, aid, "ensure")
            sysid = act.system if isinstance(act.system, str) else act.system[0]
            out.extend(self._session(act, self.eff(act, d), self.actors[aid],
                                     self._minute_ts(d, _hm(at)), r, d, sysid))
        out.extend(self._anom_events.get(d, []))
        return out

    def _periodic(self, act: Activity, e: _Eff, a: Actor, wins: List[WindowSpec], d: int,
                  r: np.random.Generator, systems: List[str]) -> List[OEv]:
        out: List[OEv] = []
        per = float(act.period_s)
        for w in wins:
            phase = (crc(f"{act.name}|{a.aid}") % 1000) / 1000.0 * per
            t = self._minute_ts(d, w.m0) + phase
            t_end = self._minute_ts(d, w.m1 if w.m1 < 1439 else 1440)
            while t < t_end:
                tt = t + (float(r.uniform(-act.jitter_s, act.jitter_s)) if act.jitter_s else 0.0)
                for sysid in systems:
                    out.extend(self._session(act, e, a, tt, r, d, sysid, periodic=True))
                t += per
        return out

    def _session(self, act: Activity, e: _Eff, a: Actor, t0: float, r: np.random.Generator,
                 d: int, sysid: str, periodic: bool = False) -> List[OEv]:
        out: List[OEv] = []
        cookie = _rand_chars(r, "0123456789abcdef", 24)
        t = t0
        ids: Dict[str, str] = {}
        uname = self.username_at(a, d)
        for k, st in enumerate(e.steps):
            if st.p < 1.0 and r.random() >= st.p:
                continue
            reps = int(r.integers(st.repeat[0], st.repeat[1] + 1))
            for j in range(reps):
                if out:
                    t += float(r.uniform(*st.think_s))
                if j:
                    ids.pop("id", None)
                tid = self._tids.get((act.name, k, d))
                out.append(self._render(sysid, st, a, t, r, cookie, d, tid, None,
                                        {"username": uname}, ids=ids, act=act))
        return out

    # ------------------------------------------------------------- rendering
    def _attrs(self, system: str, d: int, dept: str, aid: str, route: str,
               r: np.random.Generator) -> Tuple[Dict[str, Any], Dict[str, Any]]:
        hdr: Dict[str, Any] = {}
        meta: Dict[str, Any] = {}
        for a in self.spec.attr_schedule:
            systems = a.system if isinstance(a.system, list) else [a.system]
            if system not in systems or d < a.day or (a.until_day is not None and d >= a.until_day):
                continue
            v = sample_value(a.value, r, {"dept": dept, "aid": aid, "route": route})
            (hdr if a.where == "headers" else meta)[a.name] = v
        return hdr, meta

    def _render(self, sysid: str, st: Step, a: Actor, t: float, r: np.random.Generator,
                cookie: str, d: int, tid: Optional[str], anomaly: Optional[str],
                vctx: Dict[str, Any], ids: Optional[Dict[str, str]] = None,
                meta: Optional[Dict[str, Any]] = None, act: Optional[Activity] = None) -> OEv:
        s = self.sys[sysid]
        ip = self.ip_at(a, t)
        if st.method == "TLS" or s.visibility == "tls_opaque":
            up = st.up.sample(r) if st.up is not None else int(r.integers(500, 5000))
            down = st.resp.sample(r)
            return OEv(t, sysid, ip, "t", "TLS", st.route_fmt if st.method == "TLS" else s.host,
                       "", 0, up + TLS_UP_FRAMING, down, float(r.lognormal(4.0, 0.8)), a.stack, None, None,
                       tid, a.aid, anomaly, st.route_fmt)
        ids = ids if ids is not None else {}
        path = st.route_fmt
        for ph in re.findall(r"\{(\w+)\}", path):
            if ph not in ids:
                ids[ph] = str(int(r.integers(1000, 99999)))
            path = path.replace("{" + ph + "}", ids[ph])
        ctx = {"username": vctx.get("username", ""), "dept": a.dept, "aid": a.aid,
               "date": self.date_of(d), "route": st.route_fmt}
        body, fmt = "", ""
        if st.body is not None and st.body.fmt != "none":
            body, fmt = render_body(st.body, r, ctx)
        codes = list(st.status)
        pr = np.asarray([st.status[c] for c in codes], dtype=float)
        status = int(codes[int(r.choice(len(codes), p=pr / pr.sum()))]) if len(codes) > 1 else int(codes[0])
        down = st.resp.sample(r)
        hdr, mt = self._attrs(sysid, d, a.dept, a.aid, st.route_fmt, r)
        if meta:
            mt.update(meta)
        headers = {"user-agent": a.stack.ua, "host": s.host, "accept-language": "zh-CN,zh;q=0.9"}
        if fmt:
            headers["content-type"] = {"form": "application/x-www-form-urlencoded",
                                       "json": "application/json"}.get(fmt, "text/plain")
        headers.update(hdr)
        bb = body.encode("utf-8")
        l7: Dict[str, Any] = {"query": "", "headers": headers, "resp_len": int(down)}
        if cookie:
            l7["sess"] = cookie
        if fmt:
            l7.update({"body": bb[:BODY_CAP].decode("utf-8", "replace"), "body_len": len(bb),
                       "body_trunc": len(bb) > BODY_CAP, "body_type": fmt})
        up = len(bb) + 380 + len(path) + len(a.stack.ua)
        return OEv(t, sysid, ip, "h", st.method, s.host, path, status, up, int(down),
                   float(r.lognormal(3.5, 0.6)), a.stack, l7, mt or None, tid, a.aid, anomaly,
                   st.route_fmt)

    # -------------------------------------------------------------- emission
    def _ensure_planned(self, t1: float) -> None:
        while self._planned < self.n_days and self.day_start(self._planned + 1) < t1:
            self._planned += 1
            for e in self._plan_day(self._planned):
                self._seq += 1
                heapq.heappush(self._heap, (e.ts, self._seq, e))

    def _evs_rng(self, key: str) -> np.random.Generator:
        g = self._evs_rngs.get(key)
        if g is None:
            g = self._evs_rngs[key] = rng_for(self.seed, self.pack, "evs", key)
        return g

    def _xrng(self, key: str) -> np.random.Generator:
        g = self._rngs.get(key)
        if g is None:
            g = self._rngs[key] = rng_for(self.seed, self.pack, "orgx", key)
        return g

    def step(self, t0: float, t1: float, aggregated: bool) -> List[Observation]:
        self._ensure_planned(t1)
        evs: List[OEv] = []
        while self._heap and self._heap[0][0] < t1:
            evs.append(heapq.heappop(self._heap)[2])
        if not evs:
            return []
        rows: List[Tuple[OEv, str, Dict[str, Any], str]] = []   # (event, observed src, extra, xff)
        proxy = self.pert.get("R1")
        replica = self.pert.get("R9")
        sampling = self.pert.get("R12")
        for e in evs:
            d = self.day_of(e.ts)
            date_iso = self.date_of(d).isoformat()
            if e.tid is not None:
                self.stats["benign"] += 1
                o = self.opportunities.setdefault(e.tid, {}).setdefault(date_iso, {})
                o[e.src] = o.get(e.src, 0) + 1
                self._note_extremes(e, date_iso)
            else:
                self.stats["anomalous"] += 1
            if replica is not None and e.system == replica.params["system"] and d >= replica.day \
                    and self._xrng("replica").random() < float(replica.params.get("share", 0.5)):
                e.system = replica.params["replica"]
            extra: Dict[str, Any] = {}
            if sampling is not None and e.system == sampling.params["system"]:
                if self._xrng("sample").random() >= 1.0 / float(sampling.params["rate"]):
                    self.stats["dropped"] += 1
                    continue
                extra["sample_rate"] = float(sampling.params["rate"])
            src, xff = e.src, ""
            if proxy is not None and e.system == proxy.params["system"] and d >= proxy.day \
                    and e.ch == "h":
                src, xff = proxy.params["proxy"], e.src
                e.l7 = dict(e.l7 or {})
                e.l7["headers"] = dict(e.l7.get("headers") or {}, **{"x-forwarded-for": e.src})
            wl = self.who_log.setdefault(e.system, {}).setdefault(date_iso, {})
            wl[e.src] = wl.get(e.src, 0) + 1
            ak = f"{e.src}\t{act_of(e)}\t{self.clock.local(e.ts).hour}"
            al = self.act_log.setdefault(e.system, {}).setdefault(date_iso, {})
            al[ak] = al.get(ak, 0) + 1
            self.stats["events"] += 1
            rows.append((e, src, extra, xff))
        if sampling is not None:
            drop = float(sampling.params.get("drop", 0.0))
        else:
            drop = 0.0
        obs = self._aggregate(rows) if aggregated else [
            self._obs(e, src, dict(extra, **({"l7": e.l7} if e.l7 is not None else {}),
                                   **({"meta": e.meta} if e.meta else {})), e.ts, e.up, e.down, e.dur)
            for e, src, extra, _ in rows]
        if drop > 0.0:
            rr = self._xrng("drop")
            keep = []
            for o in obs:
                if o.system == sampling.params["system"] and rr.random() < drop:
                    self.stats["dropped"] += int(o.extra.get("count", 1))
                    continue
                keep.append(o)
            obs = keep
        self._note_attr_first(obs, t1)
        self.stats["records"] += len(obs)
        return obs

    def _note_attr_first(self, obs: List[Observation], t1: float) -> None:
        """First OBSERVABLE time of each header / meta attribute (PG7 truth):
        only what reaches a capture record counts – in aggregated mode an
        event's l7 / meta is visible only when the event is in `ev_sample`.
        `attr_tick[k]` = end of the tick that delivered it (an event can carry
        a timestamp a fraction of a second before the window it is delivered
        in, e.g. a 60-s monitor's jittered schedule)."""
        af = self.attr_first
        at = self.attr_tick
        before = set(af)
        for o in obs:
            ex = o.extra or {}
            rows = ex.get("ev_sample")
            items = ([(float(o.ts) + float(r.get("o", 0.0)), r.get("l7"), r.get("meta")) for r in rows]
                     if rows else [(float(o.ts), ex.get("l7"), ex.get("meta"))])
            for ts, l7, meta in items:
                if l7:
                    for hn in (l7.get("headers") or {}):
                        k = f"hdr.{hn}"
                        if ts < af.get(k, math.inf):
                            af[k] = ts
                for mn in (meta or {}):
                    k = f"meta.{mn}"
                    if ts < af.get(k, math.inf):
                        af[k] = ts
        for k in set(af) - before:
            at[k] = float(t1)

    def _aggregate(self, rows: List[Tuple[OEv, str, Dict[str, Any], str]]) -> List[Observation]:
        groups: Dict[tuple, List[Tuple[OEv, Dict[str, Any]]]] = {}
        for e, src, extra, xff in rows:
            k = (e.system, src, e.ch, e.method, e.host, e.path, e.status, e.stack.name, xff)
            groups.setdefault(k, []).append((e, extra))
        out: List[Observation] = []
        for k, items in groups.items():
            items.sort(key=lambda x: x[0].ts)
            evs = [x[0] for x in items]
            w = len(evs)
            t0 = evs[0].ts
            ts = [e.ts for e in evs]
            if w > TS_SAMPLE_MAX:
                idx = np.linspace(0, w - 1, TS_SAMPLE_MAX).round().astype(int)
                ts_sample = [round(ts[i] - t0, 3) for i in idx.tolist()]
            else:
                ts_sample = [round(t - t0, 3) for t in ts]
            up = sum(e.up for e in evs)
            down = sum(e.down for e in evs)
            dur = sum(e.dur for e in evs)
            extra: Dict[str, Any] = {"count": w, "bytes_up_total": int(up),
                                     "bytes_down_total": int(down), "ts_sample": ts_sample}
            extra.update(items[0][1])
            if evs[0].ch == "h":
                extra["retransmits_total"] = 0
            if self.ev_sample:
                r = self._evs_rng(f"{k[0]}|{k[1]}")
                pick = (np.arange(w) if w <= EV_SAMPLE_MAX else
                        np.sort(r.choice(w, size=EV_SAMPLE_MAX, replace=False)))
                sample = []
                for i in pick.tolist():
                    e = evs[i]
                    row: Dict[str, Any] = {"o": round(e.ts - t0, 3), "up": int(e.up),
                                           "down": int(e.down)}
                    if e.ch == "h":
                        row["st"] = int(e.status)
                    if e.l7 is not None:
                        row["l7"] = e.l7
                    if e.meta:
                        row["meta"] = e.meta
                    sample.append(row)
                extra["ev_sample"] = sample
            out.append(self._obs(evs[0], k[1], extra, t0, up / w, down / w, dur / w))
        return out

    def _obs(self, e: OEv, src: str, extra: Dict[str, Any], ts: float, up: float, down: float,
             dur: float) -> Observation:
        s = self.sys[e.system]
        ip, _, port = s.addr.partition(":")
        st = e.stack
        up_i, down_i = int(up), int(down)
        if e.ch == "t":
            return Observation(
                ts=ts, system=e.system, entity=src, peer=ip, l3_proto="ip", l4_proto="tcp",
                dst_port=int(port), bytes_up=up_i, bytes_down=down_i,
                pkts_up=max(1, up_i // 1400 + 2), pkts_down=max(1, down_i // 1400 + 2),
                ttl=st.ttl, win_size=st.win, rtt_ms=2.0, duration_ms=dur, app_proto="tls",
                tls_version="TLS1.3", tls_cipher="TLS_AES_128_GCM_SHA256", tls_sni=e.host,
                ja3=st.ja3, ja3s="771,4865,0-43-51", extra=extra)
        tls = s.https
        ctype = ""
        if e.l7 is not None:
            ctype = (e.l7.get("headers") or {}).get("content-type", "") if e.method in WRITE_METHODS \
                else "text/html"
        return Observation(
            ts=ts, system=e.system, entity=src, peer=ip, l3_proto="ip", l4_proto="tcp",
            dst_port=int(port), bytes_up=up_i, bytes_down=down_i,
            pkts_up=max(1, up_i // 1400 + 2), pkts_down=max(1, down_i // 1400 + 2),
            ttl=st.ttl, win_size=st.win, rtt_ms=2.0, duration_ms=dur, app_proto="http",
            http_method=e.method, http_host=e.host, http_path=e.path, http_status=e.status,
            user_agent=st.ua, content_type=ctype or "text/html",
            tls_version="TLS1.3" if tls else "", tls_cipher="TLS_AES_128_GCM_SHA256" if tls else "",
            tls_sni=e.host if tls else "", ja3=st.ja3 if tls else "", extra=extra)

    # ----------------------------------------------------------------- truth
    def group_truth(self) -> Dict[str, Any]:
        out: Dict[str, Any] = {}
        end = self.day_start(self.n_days + 1)
        for d in self.spec.departments:
            rec: Dict[str, Any] = {"code": d.code, "name": d.name, "kind": d.kind}
            if d.kind == "public":
                rec.update({"regions": [[c, float(s)] for c, s in d.regions],
                            "n": len(self.members[d.code])})
            elif d.kind == "pool":
                mem = []
                for day in range(1, self.n_days + 1):
                    h = self.lease_h(d.code, day)
                    per = max(1, int(round(24.0 / h)))
                    for j in range(per):
                        t_a = self.day_start(day) + j * h * 3600.0
                        t_b = t_a + h * 3600.0
                        for aid in self.members[d.code]:
                            a = self.actors[aid]
                            mem.append({"ip": self.ip_at(a, t_a + 1.0), "aid": aid,
                                        "from": t_a, "to": t_b})
                rec.update({"cidr": d.pool[0], "members": mem, "lease_h": d.pool[2]})
            else:
                mem = []
                for aid in self.members[d.code]:
                    a = self.actors[aid]
                    if a.readdress is not None:
                        t_r = self.day_start(a.readdress[0])
                        mem.append({"ip": a.ip, "aid": aid, "from": self.day_start(1), "to": t_r})
                        mem.append({"ip": a.readdress[1], "aid": aid, "from": t_r, "to": end})
                    else:
                        mem.append({"ip": a.ip, "aid": aid, "from": self.day_start(1), "to": end,
                                    "username": a.username})
                rec["members"] = mem
            out[d.code] = rec
        return out

    def system_truth(self) -> Dict[str, Any]:
        fams = dict(self.spec.families)
        return {"systems": {s.id: {"addr": s.addr, "visibility": s.visibility, "host": s.host,
                                   "family": s.family} for s in self.spec.systems},
                "families": fams,
                "idle": list(self.spec.config.get("idle_systems") or [])}

    def attr_truth(self) -> List[Dict[str, Any]]:
        out = []
        for a in self.spec.attr_schedule:
            name = f"hdr.{a.name}" if a.where == "headers" else f"meta.{a.name}"
            out.append({"name": name, "type": a.type, "cls": a.cls, "by": a.by,
                        "day": a.day, "until_day": a.until_day,
                        "appears": self.attr_first.get(name, self.day_start(a.day)),
                        "appears_tick": self.attr_tick.get(name),
                        "systems": a.system if isinstance(a.system, list) else [a.system]})
        return out

    def truth_rows(self) -> List[Dict[str, Any]]:
        rows: List[Dict[str, Any]] = []
        end = self.day_start(self.n_days + 1)
        for an in self.spec.anomalies:
            span = self._anom_span.get(an.id)
            if span is None:
                continue
            t_first, t_last, n = span
            p = an.params
            ents = list(p.get("ips") or [an.entity])
            rec = {"scenario_id": an.id, "pack": self.pack, "system": p["system"],
                   "entities": ents, "t_start": t_first, "t_end": t_last + 1.0, "t_first": t_first,
                   "label": "malicious", "kind": an.kind, "day": an.day, "time": an.time,
                   "n_events": int(n), "max_ttd": "1h", "progressive": True,
                   "expected_detectors": [], "expected_axes": [], "perturbed_features": []}
            rec.update(copy.deepcopy(an.truth))
            if p.get("until_day"):
                rec["until_day"] = int(p["until_day"])
            rows.append(rec)
        for dr in self.spec.drifts:
            if dr.day > self.n_days:
                continue
            ents: List[str] = []
            if dr.kind in ("window", "route"):
                act = next(a for a in self.spec.activities if a.name == dr.params["activity"])
                ents = sorted({self.actors[x.aid].ip for x in self._who(act, 1)})
            elif dr.kind == "rename":
                ents = [dr.params["ip"]]
            rec = {"scenario_id": dr.id, "pack": self.pack, "label": "legit_change",
                   "system": {"window": "oa", "rename": "oa", "route": "oa", "growth": "portal",
                              "churn": "code"}.get(dr.kind, ""),
                   "entities": ents, "t_start": self.day_start(dr.day),
                   "t_end": (self.day_start(int(dr.params["until_day"]) + 1)
                             if dr.params.get("until_day") else end),
                   "day": dr.day, "drift_kind": dr.kind, "params": _plain(dr.params),
                   "expected_detectors": [], "expected_axes": [], "perturbed_features": [],
                   "progressive": True}
            if dr.kind in ("growth", "churn"):
                rec["scope"] = "dept"
                rec["dept"] = dr.params.get("dept")
            if dr.kind in ("window", "route"):
                rec["tids_old"] = sorted({r["tid"] for r in self._ptruth_rows
                                          if r["activity"] == dr.params["activity"]
                                          and r["valid_to_day"] == dr.day})
                rec["tids_new"] = sorted({r["tid"] for r in self._ptruth_rows
                                          if r["activity"] == dr.params["activity"]
                                          and r["valid_from_day"] == dr.day})
            rec.update(copy.deepcopy(dr.truth))
            rows.append(rec)
        for pt in self.spec.perturbations:
            if pt.day > self.n_days:
                continue
            rec = {"scenario_id": pt.id, "pack": self.pack, "label": "system_change",
                   "system": pt.params.get("system", ""), "entities": [],
                   "t_start": self.day_start(pt.day), "t_end": end, "day": pt.day,
                   "perturbation": pt.kind, "params": _plain(pt.params),
                   "max_allowed_severity": "low", "expected_detectors": [], "expected_axes": [],
                   "perturbed_features": [], "progressive": True}
            rec.update(copy.deepcopy(pt.truth))
            rows.append(rec)
        return rows

    def _note_extremes(self, e: OEv, date_iso: str) -> None:
        """Smallest / largest emitted value per (row, size attribute, date)."""
        vals = []
        if e.ch == "t":
            vals.append(("net.bytes_up", float(e.up)))
        elif e.l7 is not None and e.l7.get("body_len") is not None:
            vals.append(("body.len", float(e.l7["body_len"])))
        for a, v in vals:
            ext = self.extremes.setdefault(e.tid, {}).setdefault(a, {})
            mm = ext.get(date_iso)
            if mm is None:
                ext[date_iso] = [v, v]
            else:
                mm[0], mm[1] = min(mm[0], v), max(mm[1], v)

    def ptruth(self) -> Dict[str, Any]:
        return {
            "pattern_truth": self._ptruth_rows,
            "group_truth": self.group_truth(),
            "strategy_truth": copy.deepcopy(self.spec.strategy),
            "system_truth": self.system_truth(),
            "attr_truth": self.attr_truth(),
            "opportunities": self.opportunities,
            "extremes": self.extremes,
            "who_log": self.who_log,
            "act_log": self.act_log,
            "stats": dict(self.stats),
            "days": {"start": self.start.isoformat(), "n_days": self.n_days,
                     "day_start": [self.day_start(d) for d in range(1, self.n_days + 2)],
                     "workday": [bool(self.clock.day_kind(self.date_of(d))[0])
                                 for d in range(1, self.n_days + 1)],
                     "holidays": list(self.spec.calendar.get("holidays") or [])},
            "variant": self.spec.variant,
        }


def _plain(obj: Any) -> Any:
    if isinstance(obj, (ValueSpec, SizeSpec)):
        return obj.to_dict()
    if isinstance(obj, WindowSpec):
        return {"daytypes": obj.daytypes, "start": obj.start, "end": obj.end}
    if isinstance(obj, dict):
        return {str(k): _plain(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple)):
        return [_plain(v) for v in obj]
    return obj


def render_body(b: BodySpec, r: np.random.Generator, ctx: Dict[str, Any]) -> Tuple[str, str]:
    """A request body for a BodySpec, padded (form 'viewstate' / JSON 'remark')
    to a size drawn from b.size; body length in bytes = that size whenever the
    fields fit."""
    vals = {k: sample_value(v, r, ctx) for k, v in b.fields.items()}
    target = b.size.sample(r) if b.size is not None else None
    if b.fmt == "json":
        if target is None:
            return json.dumps(vals, ensure_ascii=False), "json"
        base = json.dumps(dict(vals, remark=""), ensure_ascii=False)
        n = target - len(base.encode("utf-8"))
        pad = _B64[r.integers(0, 64, max(0, n))].tobytes().decode() if n > 0 else ""
        return json.dumps(dict(vals, remark=pad), ensure_ascii=False), "json"
    s = "&".join(f"{k}={quote_plus(str(v))}" for k, v in vals.items())
    if target is None:
        return s, "form"
    n = target - len(s.encode("utf-8")) - len("&viewstate=")
    pad = _B64[r.integers(0, 64, max(0, n))].tobytes().decode() if n > 0 else ""
    return f"{s}&viewstate={pad}", "form"
