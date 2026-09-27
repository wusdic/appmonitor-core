"""Synthetic observation source, v2 (docs/lib3/generator.md).

In production Observations come from capture adapters (SPAN/TAP decode, flow
collector, active prober). This generator fabricates a realistic multi-system
population on a *virtual* clock so that warm-up can synthesise weeks of
history in seconds and the evaluation packs (backend/app/eval/packs.py) can
inject declarative threat / legitimate-change scenarios with ground truth.

What v2 models, and why:

* Individuated personas. Every parameter of a persona is drawn from a numpy
  Generator seeded with crc32('<system>|<entity>') (never Python hash()), so a
  persona is the same individual in every run and every seed: volume scale,
  a local-time work window with a lunch dip and a weekday mask, a sparse
  Dirichlet(0.4) path preference over the system's 30-40 templates (private
  paths included; /hr/salary/* only for the HR persona 10.20.1.21), a
  personal navigation Markov chain, an object-id range, an error rate, a
  device tuple (70 % the org-standard Chrome/126 stack, 30 % unique), think
  time LogNormal(ln 8 s, 1), an RTT per subnet and a response-size
  multiplier. Identification, portraits and classes need individuals, not a
  shared archetype with noise. The twins 10.30.2.27/.28 share 90 % of their
  parameters: they are the confusable pair of the identification gate.
* Machine personas with their own clocks: API clients (library UA, own JA3,
  poll period with jitter, batch size), integration every 5 min, health
  checks every 30 s +- 1 s with an NTP-like poller, daily backups, the NAT
  host 10.30.2.50 carrying two interactive personas.
* Mechanics. Human counts are Poisson(rate * dt) with the rate integrated
  over 5-minute sub-intervals in LOCAL time (tz and holiday / make-up workday
  calendar from the pack). Events are laid out as sessions inside the tick,
  so obs.ts is a real sub-tick time. Idle ticks are really idle: an entity
  with nothing to do emits nothing (no forced DNS query).
* Aggregated mode (dt >= 900 s): one Observation per (entity, token, outcome,
  destination, client stack) carrying extra = {count, bytes_up_total,
  bytes_down_total, ts_sample <= 64 offsets from obs.ts}. The raw engines
  honour `count`, so a 15-min tick costs ~30 records per entity instead of
  ~10^2-10^3 events. Paths keep their concrete object ids (an id is part of
  the grouping key), so act.objs breadth is exact in both modes.
* Scenarios (Scenario objects, built by eval/packs.py) are declarative:
  kind + params + [t_start, t_end) + mode (additive / replace) + truth
  metadata. The generator turns them into per-tick modifiers of a persona
  (volume, bytes, paths, client stack, schedule, entity renumbering, persona
  swap) or into extra events, and publishes `gen.truth`, which only the
  evaluation metrics read.

Backward compatibility: TrafficGenerator(seed, window_s) without a pack runs
the 20-persona smoke population with a live-tick-triggered demo scenario set
(the v1 anomaly schedule), and step(dt, live=False) -> List[Observation] keeps
its v1 signature, so build.Runtime works unchanged.
"""
from __future__ import annotations

import bisect
import datetime as _dt
import math
import re
import zlib
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, Iterable, List, Optional, Sequence, Tuple
from zoneinfo import ZoneInfo

import numpy as np

from ..models.schema import AcquisitionMethod, Observation, Reachability

INF = float("inf")
DEFAULT_TZ = "Asia/Shanghai"
AGG_MIN_DT = 900.0                 # aggregated mode from this Δt on (generator.md §4)
TS_SAMPLE_MAX = 64
SUB_S = 300.0                      # rate integration sub-interval (local time)
REF_TICK_S = 900.0                 # "per tick" quantities in the spec are per 900 s


def crc(s: str) -> int:
    """Stable 32-bit hash (zlib.crc32); never Python hash(), which is salted."""
    return zlib.crc32(s.encode("utf-8"))


def rng_for(*parts: Any) -> np.random.Generator:
    """numpy Generator seeded from crc32 of each part (ints pass through)."""
    return np.random.default_rng([p if isinstance(p, int) else crc(str(p)) for p in parts])


def _sig(x: np.ndarray) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.clip(x, -40.0, 40.0)))


# --------------------------------------------------------------------------- #
# Network vocabulary
# --------------------------------------------------------------------------- #
RESOLVER_IP = "10.0.0.53"
SYSTEM_HOSTS: Dict[str, Tuple[str, str]] = {   # system -> (ui host, api host)
    "erp-prod": ("erp.corp.local", "erp-api.corp.local"),
    "oa-portal": ("portal.corp.local", "portal-api.corp.local"),
    "api-gateway": ("api.corp.local", "api.corp.local"),
}
HOST_IPS: Dict[str, str] = {
    "erp.corp.local": "10.20.0.10", "erp-api.corp.local": "10.20.0.11",
    "portal.corp.local": "10.30.0.10", "portal-api.corp.local": "10.30.0.11",
    "api.corp.local": "10.40.0.10", "sso.corp.local": "10.0.0.20",
    "cdn.corp.local": "10.0.0.80", "ntp.corp.local": "10.0.0.123",
    "backup.corp.local": "10.0.0.60", "svc.corp.local": "10.0.0.70",
}


def host_ip(host: str) -> str:
    ip = HOST_IPS.get(host)
    if ip is None:                       # external: a stable TEST-NET-3 address
        h = crc(host)
        ip = f"203.0.113.{h % 250 + 2}"
        HOST_IPS[host] = ip
    return ip


# --------------------------------------------------------------------------- #
# Client stacks (UA, JA3, TTL, TCP window)
# --------------------------------------------------------------------------- #
@dataclass(frozen=True)
class Stack:
    name: str
    ua: str
    ja3: str
    ttl: int
    win: int


_CH_CIPH = "4865-4866-4867-49195-49199-49196-49200-52393-52392-49171-49172-156-157-47-53"
_CH_EXT = "0-23-65281-10-11-35-16-5-13-18-51-45-43-27-17513"
_FF_CIPH = "4865-4867-4866-49195-49199-52393-52392-49196-49200-49162-49161-49171-49172-156-157-47-53"
_FF_EXT = "0-23-65281-10-11-35-16-5-34-51-43-13-45-28-21"
_SF_CIPH = "4865-4866-4867-49196-49195-52393-49200-49199-52392-49188-49187-49162-49161-49192"
_SF_EXT = "0-23-65281-10-11-16-5-13-18-51-45-43-27"
_LIB_CIPH = "4866-4867-4865-49196-49200-159-52393-52392-52394-49195-49199-158-49188-49192-107"
_LIB_EXT = "0-11-10-35-22-23-13-43-45-51"


def ja3(ciphers: str, exts: str, extra_ext: Sequence[int] = ()) -> str:
    e = exts + "".join(f"-{x}" for x in extra_ext)
    return f"771,{ciphers},{e},29-23-24,0"


UA_CHROME = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
             "Chrome/{v}.0.0.0 Safari/537.36")
UA_EDGE = UA_CHROME + " Edg/{v}.0.0.0"
UA_FF_WIN = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:{v}.0) Gecko/20100101 Firefox/{v}.0"
UA_FF_LIN = "Mozilla/5.0 (X11; Linux x86_64; rv:{v}.0) Gecko/20100101 Firefox/{v}.0"
UA_SAFARI = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14_5) AppleWebKit/605.1.15 (KHTML, like Gecko) "
             "Version/{v}.5 Safari/605.1.15")

JA3_A = ja3(_CH_CIPH, _CH_EXT)                            # org standard (Chrome/126 Windows)
JA3_A127 = ja3(_CH_CIPH, _CH_EXT, (41,))                  # Chrome/127 rollout (L2)
JA3_LINUX = ja3(_CH_CIPH.replace("-47-53", ""), _CH_EXT + "-21")   # Linux JA3 (T6b)
ORG_STANDARD = Stack("chrome126-win", UA_CHROME.format(v=126), JA3_A, 128, 64240)
CHROME127 = Stack("chrome127-win", UA_CHROME.format(v=127), JA3_A127, 128, 64240)
SPOOF_T6B = Stack("chrome126-ua-linux-ja3", UA_CHROME.format(v=126), JA3_LINUX, 64, 29200)
PY_REQUESTS = "python-requests/2.31.0"


def unique_stack(r: np.random.Generator, tag: str) -> Stack:
    """A non-standard personal stack: Edge, Firefox or Safari-mac (TTL 64)
    with a personal version and a personal extension (so its ja3n is unique)."""
    fam = int(r.integers(0, 3))
    ext = (int(r.integers(60000, 65000)),)
    if fam == 0:
        v = int(r.integers(124, 127))
        return Stack(f"edge{v}-{tag}", UA_EDGE.format(v=v), ja3(_CH_CIPH, _CH_EXT, ext), 128, 64240)
    if fam == 1:
        v = int(r.integers(125, 129))
        if r.random() < 0.5:
            return Stack(f"ff{v}-win-{tag}", UA_FF_WIN.format(v=v), ja3(_FF_CIPH, _FF_EXT, ext),
                         128, 64240)
        return Stack(f"ff{v}-linux-{tag}", UA_FF_LIN.format(v=v), ja3(_FF_CIPH, _FF_EXT, ext),
                     64, 29200)
    v = int(r.integers(16, 18))
    return Stack(f"safari{v}-mac-{tag}", UA_SAFARI.format(v=v), ja3(_SF_CIPH, _SF_EXT, ext),
                 64, 65535)


def library_stack(r: np.random.Generator, tag: str) -> Stack:
    """A client-library stack with its own JA3 (API clients)."""
    lib = ["python-requests/2.31.0", "okhttp/4.12.0", "Go-http-client/1.1"][int(r.integers(0, 3))]
    ext = (int(r.integers(60000, 65000)),)
    return Stack(f"{lib.split('/')[0]}-{tag}", lib, ja3(_LIB_CIPH, _LIB_EXT, ext), 64, 29200)


def fixed_stack(name: str, ua: str, tag: str, ttl: int = 64) -> Stack:
    return Stack(name, ua, ja3(_LIB_CIPH, _LIB_EXT, (crc(tag) % 5000 + 55000,)), ttl, 29200)


# --------------------------------------------------------------------------- #
# Templates (per-system vocabularies)
# --------------------------------------------------------------------------- #
class Template:
    """One request template: method + host + path format. `fmt` may hold one
    placeholder: {id} (object id -> '{num}' token), {p} (page) or {q} (term)."""

    __slots__ = ("method", "host", "fmt", "cat", "token", "pre", "post", "var",
                 "base_up", "base_down", "srv_ms")

    def __init__(self, method: str, host: str, fmt: str, cat: str) -> None:
        self.method, self.host, self.fmt, self.cat = method, host, fmt, cat
        m = re.search(r"\{(id|p|q)\}", fmt)
        if m:
            self.pre, self.post, self.var = fmt[:m.start()], fmt[m.end():], m.group(1)
        else:
            self.pre, self.post, self.var = fmt, "", None
        path, _, query = fmt.partition("?")
        path = path.replace("{id}", "{num}")
        if query:
            names = sorted(kv.split("=", 1)[0] for kv in query.split("&"))
            path += "?" + "&".join(names)
        self.token = f"{method} {host} {path}"
        h = crc(self.token)
        write = method in ("POST", "PUT", "DELETE", "PATCH")
        self.base_up = float(1500 + h % 6000) if write else float(350 + h % 900)
        self.base_down = float(2 ** (11 + (h >> 8) % 6))
        self.srv_ms = float(20 + (h >> 16) % 250)

    def render(self, obj_id: int, u: float) -> str:
        v = self.var
        if v is None:
            return self.fmt
        if v == "id":
            return f"{self.pre}{obj_id}{self.post}"
        if v == "p":
            return f"{self.pre}{1 + int(u * 5)}{self.post}"
        return f"{self.pre}{SEARCH_TERMS[int(u * len(SEARCH_TERMS)) % len(SEARCH_TERMS)]}{self.post}"


SEARCH_TERMS = ("invoice", "order", "q3", "contract", "leave", "budget", "stock", "customer",
                "report", "meeting", "policy", "travel")
CATS = ("auth", "dash", "list", "view", "write", "search", "report", "private", "static")
# navigation structure between categories (row: from, col: to)
_NAV = {
    "auth":    {"dash": 3.0, "list": 1.0, "static": 1.0},
    "dash":    {"list": 2.0, "search": 1.0, "report": 1.0, "private": 1.0, "view": 0.5,
                "static": 0.5, "dash": 0.2},
    "list":    {"view": 3.0, "list": 1.0, "search": 0.5, "write": 0.3, "dash": 0.5},
    "view":    {"view": 2.0, "list": 1.5, "write": 1.0, "dash": 0.5, "report": 0.3},
    "write":   {"view": 1.5, "list": 1.5, "dash": 0.5},
    "search":  {"view": 2.0, "search": 1.0, "list": 0.5},
    "report":  {"report": 1.0, "dash": 1.0, "list": 0.5},
    "private": {"private": 1.5, "dash": 1.0, "list": 0.5, "view": 0.5},
    "static":  {"dash": 1.0, "list": 1.0},
}

_ERP_HUMAN = [
    ("POST", "/login", "auth"), ("GET", "/logout", "auth"),
    ("GET", "/dashboard", "dash"), ("GET", "/home", "dash"), ("GET", "/notifications", "dash"),
    ("GET", "/orders", "list"), ("GET", "/orders/list?page={p}&status=open", "list"),
    ("GET", "/customers", "list"), ("GET", "/invoices", "list"), ("GET", "/products", "list"),
    ("GET", "/suppliers", "list"),
    ("GET", "/orders/view/{id}", "view"), ("GET", "/customers/{id}", "view"),
    ("GET", "/invoices/{id}", "view"), ("GET", "/products/{id}", "view"),
    ("POST", "/orders/create", "write"), ("POST", "/orders/{id}/update", "write"),
    ("POST", "/invoices/{id}/approve", "write"), ("PUT", "/customers/{id}", "write"),
    ("GET", "/search?q={q}", "search"),
    ("GET", "/report/sales_daily", "report"), ("GET", "/report/inventory", "report"),
    ("GET", "/report/aging", "report"), ("GET", "/report/margin", "report"),
    ("GET", "/report/export?fmt=xlsx", "report"),
    ("GET", "/fin/ledger", "private"), ("GET", "/fin/ledger/{id}", "private"),
    ("GET", "/hr/leave", "private"), ("POST", "/hr/leave/apply", "private"),
    ("GET", "/crm/lead", "private"), ("GET", "/crm/lead/{id}", "private"),
    ("GET", "/wh/stock", "private"), ("GET", "/wh/stock/{id}", "private"),
    ("GET", "/static/app.js", "static"), ("GET", "/api/profile", "static"),
    ("GET", "/settings", "static"),
]
_ERP_SALARY = [("GET", "/hr/salary/list", "private"), ("GET", "/hr/salary/{id}", "private"),
               ("POST", "/hr/salary/{id}/update", "write")]
_OA_HUMAN = [
    ("POST", "/login", "auth"), ("GET", "/logout", "auth"),
    ("GET", "/home", "dash"), ("GET", "/inbox", "dash"), ("GET", "/calendar", "dash"),
    ("GET", "/docs", "list"), ("GET", "/tasks", "list"), ("GET", "/approvals", "list"),
    ("GET", "/news", "list"), ("GET", "/meetings?week={p}", "list"),
    ("GET", "/docs/{id}", "view"), ("GET", "/tasks/{id}", "view"),
    ("GET", "/approvals/{id}", "view"), ("GET", "/news/{id}", "view"),
    ("GET", "/mail/{id}", "view"),
    ("POST", "/tasks/{id}/comment", "write"), ("POST", "/approvals/{id}/approve", "write"),
    ("POST", "/docs/upload", "write"), ("POST", "/mail/send", "write"),
    ("GET", "/search?q={q}", "search"),
    ("GET", "/report/weekly", "report"), ("GET", "/report/attendance", "report"),
    ("GET", "/report/projects", "report"),
    ("GET", "/hr/leave", "private"), ("POST", "/hr/leave/apply", "private"),
    ("GET", "/fin/expense", "private"), ("GET", "/fin/expense/{id}", "private"),
    ("GET", "/crm/lead", "private"), ("GET", "/orders/view/{id}", "private"),
    ("GET", "/static/portal.js", "static"), ("GET", "/api/profile", "static"),
    ("GET", "/settings", "static"),
]
_API = [
    ("GET", "/v1/resource", "list"), ("GET", "/v1/resource/{id}", "view"),
    ("POST", "/v1/resource", "write"), ("PUT", "/v1/resource/{id}", "write"),
    ("GET", "/v1/orders", "list"), ("GET", "/v1/orders/{id}", "view"),
    ("POST", "/v1/orders", "write"), ("GET", "/v1/inventory", "list"),
    ("GET", "/v1/customers/{id}", "view"), ("POST", "/v1/events", "write"),
    ("GET", "/v1/status", "dash"), ("GET", "/v1/prices", "list"),
]
_TPL_CACHE: Dict[Tuple[str, str, str], Template] = {}


def tpl(method: str, host: str, fmt: str, cat: str = "view") -> Template:
    k = (method, host, fmt)
    t = _TPL_CACHE.get(k)
    if t is None:
        t = _TPL_CACHE[k] = Template(method, host, fmt, cat)
    return t


def human_vocab(system: str, hr: bool = False) -> List[Template]:
    host = SYSTEM_HOSTS.get(system, ("app.corp.local",))[0]
    rows = list(_OA_HUMAN if system == "oa-portal" else _ERP_HUMAN)
    if hr and system == "erp-prod":
        rows += _ERP_SALARY
    return [tpl(m, host, f, c) for m, f, c in rows]


def api_vocab(system: str) -> List[Template]:
    host = SYSTEM_HOSTS.get(system, ("", "api.corp.local"))[1]
    return [tpl(m, host, f, c) for m, f, c in _API]


# --------------------------------------------------------------------------- #
# Tick context (local time, calendar)
# --------------------------------------------------------------------------- #
class Clock:
    """tz + calendar ({holidays, makeup_workdays} as 'YYYY-MM-DD') lookups."""

    def __init__(self, tz: str = DEFAULT_TZ, calendar: Optional[Dict[str, Any]] = None) -> None:
        self.tz = tz or DEFAULT_TZ
        self.zone = ZoneInfo(self.tz)
        cal = calendar or {}
        self.holidays = {_dt.date.fromisoformat(str(d)) for d in cal.get("holidays") or []}
        self.makeup = {_dt.date.fromisoformat(str(d)) for d in cal.get("makeup_workdays") or []}

    def local(self, t: float) -> _dt.datetime:
        return _dt.datetime.fromtimestamp(t, self.zone)

    def day_kind(self, d: _dt.date) -> Tuple[bool, bool]:
        """(workday, makeup): 调休 make-up days are workdays, holidays are not."""
        if d in self.makeup:
            return True, True
        if d in self.holidays:
            return False, False
        return d.weekday() < 5, False

    def hour(self, t: float) -> float:
        ld = self.local(t)
        return ld.hour + ld.minute / 60.0 + ld.second / 3600.0

    def epoch(self, d: _dt.date, hh: float = 0.0) -> float:
        """Epoch of local date d at hour hh (DST-correct)."""
        h = int(hh)
        m = int(round((hh - h) * 60))
        return _dt.datetime(d.year, d.month, d.day, h, m, tzinfo=self.zone).timestamp()


class Tick:
    """Per-tick local-time grid shared by every persona: M sub-intervals of
    ~300 s with their local hour, weekday, day type and business factor."""

    __slots__ = ("t0", "t1", "dt", "M", "sub", "mids", "hours", "dows", "work", "makeup",
                 "dates", "biz", "index")

    def __init__(self, t0: float, dt: float, clock: Clock, index: int) -> None:
        self.t0, self.dt, self.t1, self.index = t0, dt, t0 + dt, index
        M = max(1, int(round(dt / SUB_S)))
        self.M, self.sub = M, dt / M
        self.mids = t0 + (np.arange(M) + 0.5) * self.sub
        hours = np.empty(M)
        dows = np.empty(M, dtype=np.int64)
        work = np.empty(M, dtype=bool)
        mk = np.empty(M, dtype=bool)
        dates = []
        for i, t in enumerate(self.mids):
            ld = clock.local(float(t))
            hours[i] = ld.hour + ld.minute / 60.0 + ld.second / 3600.0
            dows[i] = ld.weekday()
            d = ld.date()
            dates.append(d)
            work[i], mk[i] = clock.day_kind(d)
        self.hours, self.dows, self.work, self.makeup, self.dates = hours, dows, work, mk, dates
        day = _sig((hours - 8.0) / 0.5) * _sig((19.0 - hours) / 0.5)
        self.biz = np.where(work, 0.15 + 0.85 * day, 0.15)


# --------------------------------------------------------------------------- #
# Persona models
# --------------------------------------------------------------------------- #
class HumanModel:
    """An interactive individual (see module doc for the parameter list)."""

    def __init__(self, key: str, system: str, r: np.random.Generator, hr: bool = False,
                 search: bool = False, base: Optional["HumanModel"] = None,
                 share: float = 0.9, stack: Optional[Stack] = None) -> None:
        self.key, self.system, self.hr, self.search = key, system, hr, search
        vocab = human_vocab(system, hr)
        K = len(vocab)
        vol = float(r.lognormal(0.0, 0.35))
        ws = float(np.clip(r.normal(8.5, 0.7), 6.5, 11.0))
        we = float(np.clip(r.normal(18.0, 1.0), ws + 6.0, 22.0))
        mask = np.ones(7)
        mask[5:] = 0.0
        if r.random() < 0.2:                               # a lighter weekday
            mask[int(r.integers(0, 5))] = 0.6
        pref = r.dirichlet(np.full(K, 0.4))
        if search:
            boost = np.array([8.0 if t.cat == "search" else 3.0 if t.cat == "view" else 1.0
                              for t in vocab])
            pref = pref * boost
        if not hr:
            pref = np.where([("/hr/salary" in t.fmt) for t in vocab], 0.0, pref)
        pref = pref / pref.sum()
        noise = r.lognormal(0.0, 0.5, size=(K, K))
        id_lo = int(r.integers(10_000, 900_000))
        id_w = int(r.integers(60, 400))
        err = float(r.uniform(0.01, 0.08))
        dev_u = float(r.random())
        uniq = unique_stack(r, key.replace("|", "-"))
        size_mult = float(r.lognormal(0.0, 0.35))
        sess_len = float(r.uniform(6.0, 20.0))
        p_login = float(r.uniform(0.3, 0.7))
        if base is not None:                               # twin: share 90 %
            a = share
            vol = a * base.volume + (1 - a) * vol
            ws = a * base.work_start + (1 - a) * ws
            we = a * base.work_end + (1 - a) * we
            mask = base.mask.copy()
            pref = a * base.pref + (1 - a) * pref
            noise = a * base.noise + (1 - a) * noise
            id_lo, id_w = base.id_lo + int((1 - a) * (id_lo - base.id_lo)), base.id_w
            err = a * base.err + (1 - a) * err
            size_mult = a * base.size_mult + (1 - a) * size_mult
            sess_len = a * base.sess_len + (1 - a) * sess_len
            p_login = a * base.p_login + (1 - a) * p_login
            dev_stack = base.stack
        else:
            dev_stack = ORG_STANDARD if dev_u < 0.7 else uniq
        self.vocab, self.K = vocab, K
        self.volume, self.work_start, self.work_end, self.mask = vol, ws, we, mask
        self.pref, self.noise = pref, noise
        self.id_lo, self.id_w, self.err = id_lo, id_w, err
        self.stack = stack or dev_stack
        self.size_mult, self.sess_len, self.p_login = size_mult, sess_len, p_login
        self.think_mu, self.think_sigma = math.log(8.0), 1.0
        self.rate_s = (2.5 if search else 1.6) * vol / 60.0   # events / s at full activity
        self._compile()

    def _compile(self) -> None:
        cats = [t.cat for t in self.vocab]
        S = np.array([[_NAV[a].get(b, 0.05) for b in cats] for a in cats])
        T = S * self.pref[None, :] * self.noise + 1e-9
        T /= T.sum(axis=1, keepdims=True)
        self.T = T
        self.trans_cum = [np.cumsum(row).tolist() for row in T]
        start = np.array([self.p_login if (t.cat == "auth" and t.method == "POST") else
                          (1 - self.p_login) * p if t.cat == "dash" else 0.0
                          for t, p in zip(self.vocab, self.pref)])
        if start.sum() <= 0:
            start = np.array([1.0 if t.cat == "dash" else 0.0 for t in self.vocab])
        self.start = start / start.sum()
        self.start_cum = np.cumsum(self.start).tolist()
        # expected visit share within a session (for portraits / path_pref)
        v, acc = self.start.copy(), np.zeros(self.K)
        for _ in range(int(round(self.sess_len))):
            acc += v
            v = v @ T
        self.visit = acc / acc.sum()

    def activity(self, tk: Tick, force: Sequence[Tuple[float, float]] = ()) -> np.ndarray:
        h = tk.hours
        a = _sig((h - self.work_start) / 0.2) * _sig((self.work_end - h) / 0.2)
        a = np.where((h >= 12.0) & (h < 13.0), a * 0.4, a)
        dayf = np.where(tk.work, np.where(tk.makeup, 1.0, self.mask[tk.dows]), 0.0)
        a = a * dayf
        if force:
            for i, t in enumerate(tk.mids):
                if any(x <= t < y for x, y in force):
                    a[i] = 1.0
        return a

    def params(self) -> Dict[str, Any]:
        order = np.argsort(-self.visit)
        return {
            "persona_id": self.key, "volume_scale": round(self.volume, 4),
            "work_start": round(self.work_start, 3), "work_end": round(self.work_end, 3),
            "lunch": [12.0, 13.0, 0.4], "weekday_mask": [float(x) for x in self.mask],
            "path_pref": {self.vocab[i].token: round(float(self.visit[i]), 5)
                          for i in order[:15] if self.visit[i] > 1e-4},
            "id_range": [self.id_lo, self.id_lo + self.id_w], "error_rate": round(self.err, 4),
            "device": {"name": self.stack.name, "ua": self.stack.ua, "ttl": self.stack.ttl,
                       "org_standard": self.stack is ORG_STANDARD},
            "think_time": {"mu": round(self.think_mu, 4), "sigma": self.think_sigma},
            "size_mult": round(self.size_mult, 4), "rate_per_min": round(self.rate_s * 60, 4),
            "hr": self.hr,
        }


class MachineModel:
    """API client / integration / health / backup persona clock + endpoints."""

    def __init__(self, key: str, system: str, archetype: str, r: np.random.Generator) -> None:
        self.key, self.system, self.archetype = key, system, archetype
        self.volume = float(r.lognormal(0.0, 0.2))
        tag = key.replace("|", "-")
        ip = key.partition("|")[2]
        if archetype == "api":
            vocab = api_vocab(system)
            k = int(r.integers(3, 7))
            idx = sorted(r.choice(len(vocab), size=k, replace=False).tolist())
            self.endpoints = [vocab[i] for i in idx]
            w = r.dirichlet(np.ones(k))
            self.ep_cum = np.cumsum(w).tolist()
            self.stack = library_stack(r, tag)
            self.period = float(r.uniform(90.0, 150.0))
            self.jitter = float(r.uniform(0.05, 0.2))
            self.batch = float(r.uniform(2.5, 3.5)) * self.volume
            self.id_lo = int(r.integers(1000, 90000))
            self.err = float(r.uniform(0.005, 0.03))
        elif archetype == "integration":
            self.endpoints = [tpl("POST", "svc.corp.local", "/api/sync", "write"),
                              tpl("GET", "svc.corp.local", "/api/sync/status", "dash")]
            self.ep_cum = [0.85, 1.0]
            self.stack = fixed_stack(f"svc-sync-{tag}", "svc-sync/1.4 (Java/17)", tag)
            self.period, self.jitter = 300.0, 2.0 / 300.0
            self.batch = 6.0 * self.volume
            self.id_lo, self.err = 0, 0.005
        elif archetype == "health":
            self.endpoints = [tpl("GET", h, "/healthz", "dash")
                              for h in sorted(set(SYSTEM_HOSTS.get(system, ("svc.corp.local",))))]
            self.ep_cum = np.cumsum([1.0 / len(self.endpoints)] * len(self.endpoints)).tolist()
            self.stack = fixed_stack(f"kube-probe-{tag}", "kube-probe/1.28", tag)
            self.period, self.jitter = 30.0, 1.0
            self.batch, self.id_lo, self.err = 1.0, 0, 0.0
        elif archetype == "backup":
            self.endpoints = [tpl("PUT", "backup.corp.local", "/blob/{id}", "write")]
            self.ep_cum = [1.0]
            self.stack = fixed_stack(f"restic-{tag}", "restic/0.16.4", tag)
            self.hour = 1.0 if ip == "10.20.9.5" else 2.0
            self.duration = 40 * 60.0
            self.per_min = 6.0
            self.period, self.jitter, self.batch, self.id_lo, self.err = 0.0, 0.0, 1.0, 0, 0.0
        else:
            raise ValueError(f"unknown machine archetype {archetype!r}")

    def params(self) -> Dict[str, Any]:
        p: Dict[str, Any] = {"persona_id": self.key, "volume_scale": round(self.volume, 4),
                             "ua": self.stack.ua, "stack": self.stack.name}
        if self.archetype == "backup":
            p.update(window_hour=self.hour, duration_s=self.duration)
        else:
            p.update(period_s=round(self.period, 3), jitter=round(self.jitter, 4),
                     batch=round(self.batch, 3),
                     path_pref={t.token: round(b - a, 4) for t, a, b in
                                zip(self.endpoints, [0.0] + self.ep_cum[:-1], self.ep_cum)})
        return p


@dataclass
class Persona:
    """entity + archetype + JSON-able params (the runner reads these three);
    `models` / `present` are runtime-only."""

    entity: str
    archetype: str
    system: str = ""
    params: Dict[str, Any] = field(default_factory=dict)
    models: List[Any] = field(default_factory=list, repr=False, compare=False)
    present: Tuple[float, float] = (-INF, INF)
    emits: bool = True

    @property
    def key(self) -> str:
        return f"{self.system}|{self.entity}"


# --------------------------------------------------------------------------- #
# Population
# --------------------------------------------------------------------------- #
# (system, ip, archetype, flags)
BASE_POPULATION: List[Tuple[str, str, str, str]] = [
    ("erp-prod", "10.20.1.11", "interactive", ""), ("erp-prod", "10.20.1.12", "interactive", ""),
    ("erp-prod", "10.20.1.13", "interactive", ""), ("erp-prod", "10.20.1.15", "interactive", ""),
    ("erp-prod", "10.20.1.16", "interactive", ""), ("erp-prod", "10.20.1.17", "interactive", ""),
    ("erp-prod", "10.20.1.18", "interactive", ""), ("erp-prod", "10.20.1.21", "interactive", "hr"),
    ("erp-prod", "10.20.4.30", "api", ""), ("erp-prod", "10.20.4.32", "api", ""),
    ("erp-prod", "10.20.4.33", "api", ""), ("erp-prod", "10.20.4.31", "integration", ""),
    ("erp-prod", "10.20.9.5", "backup", ""), ("erp-prod", "10.20.9.9", "health", ""),
    ("oa-portal", "10.30.2.21", "interactive", ""), ("oa-portal", "10.30.2.22", "interactive", ""),
    ("oa-portal", "10.30.2.24", "interactive", ""), ("oa-portal", "10.30.2.25", "interactive", ""),
    ("oa-portal", "10.30.2.26", "interactive", ""), ("oa-portal", "10.30.2.29", "interactive", ""),
    ("oa-portal", "10.30.2.27", "interactive", "twin"),
    ("oa-portal", "10.30.2.28", "interactive", "twin"),
    ("oa-portal", "10.30.2.23", "search", ""), ("oa-portal", "10.30.4.40", "api", ""),
    ("oa-portal", "10.30.4.41", "api", ""), ("oa-portal", "10.30.4.42", "api", ""),
    ("oa-portal", "10.30.9.9", "health", ""), ("oa-portal", "10.30.2.50", "nat", ""),
    ("api-gateway", "10.40.4.51", "api", ""), ("api-gateway", "10.40.4.52", "api", ""),
    ("api-gateway", "10.40.4.54", "api", ""), ("api-gateway", "10.40.4.55", "api", ""),
    ("api-gateway", "10.40.4.56", "api", ""), ("api-gateway", "10.40.4.57", "api", ""),
    ("api-gateway", "10.40.4.53", "integration", ""), ("api-gateway", "10.40.9.9", "health", ""),
    ("api-gateway", "10.40.9.6", "backup", ""),
]
BASE_KEYS: List[str] = [f"{s}|{ip}" for s, ip, _, _ in BASE_POPULATION]
RESERVED: Dict[str, List[str]] = {
    "erp-prod": ["10.20.1.14", "10.20.1.66", "10.20.7.77", "10.20.1.70", "10.20.1.71",
                 "10.20.1.72", "10.20.1.112", "10.20.1.114"],
    "oa-portal": ["10.30.2.99"],
}
V1_KEYS = ["erp-prod|10.20.1.11", "erp-prod|10.20.1.12", "erp-prod|10.20.1.13",
           "erp-prod|10.20.4.30", "erp-prod|10.20.4.31", "erp-prod|10.20.9.5",
           "erp-prod|10.20.9.9", "oa-portal|10.30.2.21", "oa-portal|10.30.2.22",
           "oa-portal|10.30.2.23", "oa-portal|10.30.4.40", "oa-portal|10.30.9.9",
           "api-gateway|10.40.4.51", "api-gateway|10.40.4.52", "api-gateway|10.40.4.53",
           "api-gateway|10.40.9.9", "api-gateway|10.40.9.6"]
SMOKE_KEYS = V1_KEYS + ["erp-prod|10.20.1.15", "oa-portal|10.30.2.24", "api-gateway|10.40.4.54"]
TWINS = ("oa-portal|10.30.2.27", "oa-portal|10.30.2.28")
NAT_KEY = "oa-portal|10.30.2.50"
_ARCH = {f"{s}|{ip}": (a, f) for s, ip, a, f in BASE_POPULATION}


def build_human(key: str, hr: bool = False, search: bool = False, tag: Optional[str] = None,
                stack: Optional[Stack] = None) -> HumanModel:
    """Persona v2 of `key` (params from crc32(key)), or a fresh individual from
    the interactive prior when `tag` is given (crc32(tag))."""
    system = key.partition("|")[0]
    if key in TWINS and tag is None:
        base = HumanModel(f"{system}|twin", system, rng_for(f"{system}|twin-base"))
        return HumanModel(key, system, rng_for(key), base=base, share=0.9)
    return HumanModel(key, system, rng_for(tag if tag is not None else key), hr=hr,
                      search=search, stack=stack)


def build_persona(key: str) -> Persona:
    system, _, ip = key.partition("|")
    arch, flags = _ARCH.get(key, ("interactive", ""))
    if arch in ("interactive", "search"):
        m = build_human(key, hr=(flags == "hr"), search=(arch == "search"))
        return Persona(ip, arch, system, m.params(), [m])
    if arch == "nat":
        ms = [build_human(f"{key}#{i}", tag=f"{key}#{i}") for i in (1, 2)]
        return Persona(ip, arch, system, {"persona_id": key, "personas": [m.params() for m in ms],
                                          "work_start": min(m.work_start for m in ms),
                                          "work_end": max(m.work_end for m in ms)}, ms)
    m = MachineModel(key, system, arch, rng_for(key))
    return Persona(ip, arch, system, m.params(), [m])


# --------------------------------------------------------------------------- #
# Scenarios (declarative; built by eval/packs.py)
# --------------------------------------------------------------------------- #
@dataclass
class Spawn:
    """A persona that appears at t_from (new IP). kind: 'prior' (fresh
    individual from the interactive prior, crc32(tag)), 'scanner',
    'dns_tunnel', or 'alias' (no own traffic: an existing persona is
    renumbered onto it)."""

    key: str
    kind: str = "prior"
    t_from: float = -INF
    t_to: float = INF
    tag: str = ""
    archetype: str = "interactive"
    alias_of: str = ""


@dataclass
class Scenario:
    """One declarative scenario. `kind` selects the effect; `params` its knobs;
    [t_start, t_end) its active span (absolute epoch s); `truth` the metadata
    published in gen.truth (expected_detectors, expected_axes, max_ttd_s,
    required_severity | max_allowed_severity, perturbed_features, ...).
    `live_tick` (demo mode only) activates it at the n-th live tick."""

    scenario_id: str
    label: str
    system: str
    entities: List[str]
    kind: str
    t_start: float = math.nan
    t_end: float = INF
    mode: str = "additive"
    params: Dict[str, Any] = field(default_factory=dict)
    spawns: List[Spawn] = field(default_factory=list)
    truth: Dict[str, Any] = field(default_factory=dict)
    live_tick: Optional[int] = None

    def keys(self) -> List[str]:
        return [e if "|" in e else f"{self.system}|{e}" for e in self.entities]

    def active(self, t0: float, t1: float) -> bool:
        return self.t_start < t1 and self.t_end > t0

    def truth_record(self, pack: str) -> Dict[str, Any]:
        rec = {"scenario_id": self.scenario_id, "pack": pack, "system": self.system,
               "entities": list(self.entities), "t_start": float(self.t_start),
               "t_end": float(self.t_end), "label": self.label, "mode": self.mode,
               "kind": self.kind}
        for k in ("expected_detectors", "expected_axes", "perturbed_features"):
            rec[k] = list(self.truth.get(k) or [])
        rec.update({k: v for k, v in self.truth.items() if k not in rec})
        if self.label == "malicious":
            rec.setdefault("required_severity", "low")
        else:
            rec.setdefault("max_allowed_severity", "info")
        return rec


class Mods:
    """Per-(entity, tick) modifiers collected from the active scenarios."""

    __slots__ = ("replace", "vol", "up", "up_rand", "down", "stack", "path_override", "force",
                 "swap", "entity_as", "status", "rtt", "retrans", "extra_tpls", "enum",
                 "backup_hour")

    def __init__(self) -> None:
        self.replace = False
        self.vol = 1.0
        self.up = 1.0
        self.up_rand: Optional[Tuple[float, float]] = None
        self.down = 1.0
        self.stack: Optional[Stack] = None
        self.path_override: Optional[Tuple[float, List[Template]]] = None
        self.force: List[Tuple[float, float]] = []
        self.swap: Optional[HumanModel] = None
        self.entity_as: Optional[str] = None
        self.status: Optional[Tuple[float, int]] = None
        self.rtt = 1.0
        self.retrans = 1.0
        self.extra_tpls: List[Tuple[float, Template]] = []
        self.enum: Optional[Dict[str, Any]] = None
        self.backup_hour: Optional[float] = None


_NO_MODS = Mods()

# Events are small lists: [ts, ch, a, b, c, status, up, down, dur_ms, stack]
#   'h' HTTP  : a=method, b=host, c=path
#   'd' DNS   : a=qtype, b=qname, c=rcode
#   'u' UDP   : a=peer host, b=dport
#   's' SYN   : a=peer ip, b=dport
#   'p' probe : a=reachability
EV_TS, EV_CH, EV_A, EV_B, EV_C, EV_ST, EV_UP, EV_DOWN, EV_DUR, EV_STACK = range(10)


# --------------------------------------------------------------------------- #
# Generator
# --------------------------------------------------------------------------- #
class TrafficGenerator:
    """Deterministic multi-system traffic fabricator (v2).

    TrafficGenerator(seed=42, window_s=60, pack=None)
      pack: a Pack-like object (eval/packs.py) with name, tz, calendar,
            start_epoch, population ('sys|ip' keys), scenarios; None runs the
            smoke population with the live-tick demo scenarios.
    step(dt, live=False, aggregated=None) -> List[Observation]
      advances the virtual clock `vt` by dt; aggregated defaults to dt >= 900.
    Attributes: vt, personas {'sys|ip': Persona}, systems {system: [Persona]},
    truth [dict] (gen.truth, generator.md §5), scenarios, tick, live_ticks.
    """

    def __init__(self, seed: int = 42, window_s: int = 60, pack: Any = None) -> None:
        self.seed = int(seed)
        self.window_s = window_s
        self.pack = pack
        if pack is None:
            self.pack_name = "demo"
            self.clock = Clock(DEFAULT_TZ, None)
            keys = list(SMOKE_KEYS)
            scenarios = demo_scenarios()
            self.vt = 0.0
        else:
            self.pack_name = str(getattr(pack, "name", "pack"))
            self.clock = Clock(getattr(pack, "tz", DEFAULT_TZ) or DEFAULT_TZ,
                               getattr(pack, "calendar", None))
            keys = list(getattr(pack, "population", None) or BASE_KEYS)
            scenarios = list(getattr(pack, "scenarios", None) or [])
            self.vt = float(getattr(pack, "start_epoch", 0.0) or 0.0)
        self.scenarios: List[Scenario] = scenarios
        self.tick = 0
        self.live_ticks = 0
        self.personas: Dict[str, Persona] = {}
        for k in keys:
            self.personas[k] = build_persona(k)
        for sc in scenarios:
            for sp in sc.spawns:
                self._add_spawn(sp)
        self.truth: List[Dict[str, Any]] = [sc.truth_record(self.pack_name) for sc in scenarios
                                            if sc.live_tick is None]
        self._rngs: Dict[str, np.random.Generator] = {}
        self._state: Dict[str, Any] = {}          # renewal clocks, scenario state

    # ------------------------------------------------------------ population
    @property
    def systems(self) -> Dict[str, List[Persona]]:
        out: Dict[str, List[Persona]] = {}
        for p in self.personas.values():
            out.setdefault(p.system, []).append(p)
        return out

    def _add_spawn(self, sp: Spawn) -> None:
        system, _, ip = sp.key.partition("|")
        if sp.kind == "alias":
            src = self.personas.get(sp.alias_of) or build_persona(sp.alias_of)
            p = Persona(ip, src.archetype, system, dict(src.params), [], (sp.t_from, sp.t_to),
                        emits=False)
        elif sp.kind in ("scanner", "dns_tunnel"):
            p = Persona(ip, sp.kind, system, {"persona_id": sp.key, "kind": sp.kind}, [],
                        (sp.t_from, sp.t_to))
        else:
            tag = sp.tag or f"prior|{sp.key}"
            m = build_human(sp.key, tag=tag)
            p = Persona(ip, sp.archetype, system, m.params(), [m], (sp.t_from, sp.t_to))
            p.params["persona_id"] = tag
        self.personas[sp.key] = p

    def _rng(self, key: str) -> np.random.Generator:
        r = self._rngs.get(key)
        if r is None:
            r = self._rngs[key] = rng_for(self.seed, self.pack_name, key)
        return r

    # ------------------------------------------------------------------ step
    def step(self, dt: float, live: bool = False,
             aggregated: Optional[bool] = None) -> List[Observation]:
        dt = float(dt)
        t0 = self.vt
        self.vt = t1 = t0 + dt
        if aggregated is None:
            aggregated = dt >= AGG_MIN_DT
        if live:
            self._activate_live(t0)
        tk = Tick(t0, dt, self.clock, self.tick)
        mods: Dict[str, Mods] = {}
        extra: Dict[str, List[list]] = {}
        for sc in self.scenarios:
            if sc.active(t0, t1):
                fx = getattr(self, f"_fx_{sc.kind}", None)
                if fx is None:
                    raise ValueError(f"unknown scenario kind {sc.kind!r}")
                fx(sc, tk, mods, extra)
        by_key: Dict[str, List[list]] = {}
        for key, p in self.personas.items():
            if not p.emits or not (p.present[0] < t1 and p.present[1] > t0):
                continue
            m = mods.get(key, _NO_MODS)
            evs: List[list] = []
            if not m.replace:
                evs = self._emit(p, key, tk, m)
                if m is not _NO_MODS and evs:
                    self._post(evs, m, self._rng(key))
            if p.present[0] > t0 or p.present[1] < t1:
                evs = [e for e in evs if p.present[0] <= e[EV_TS] < p.present[1]]
            out_key = m.entity_as or key
            if evs:
                by_key.setdefault(out_key, []).extend(evs)
        for key, evs in extra.items():
            if evs:
                by_key.setdefault(key, []).extend(evs)
        obs: List[Observation] = []
        for key in sorted(by_key):
            evs = by_key[key]
            evs.sort(key=_ev_ts)
            m = mods.get(key, _NO_MODS)
            if aggregated:
                obs.extend(self._aggregate(key, evs, m))
            else:
                obs.extend(self._observations(key, evs, m))
        self.tick += 1
        if live:
            self.live_ticks += 1
        return obs

    def _activate_live(self, t0: float) -> None:
        for sc in self.scenarios:
            if sc.live_tick is not None and math.isnan(sc.t_start) \
                    and self.live_ticks >= sc.live_tick:
                sc.t_start = t0
                for sp in sc.spawns:
                    sp.t_from = t0
                    self._add_spawn(sp)
                self.truth.append(sc.truth_record(self.pack_name))

    # -------------------------------------------------------------- emitters
    def _emit(self, p: Persona, key: str, tk: Tick, m: Mods) -> List[list]:
        r = self._rng(key)
        arch = p.archetype
        if arch in ("interactive", "search", "nat"):
            out: List[list] = []
            models = [m.swap] if m.swap is not None else p.models
            for hm in models:
                out.extend(self._emit_human(hm, r, tk, m))
            return out
        if arch == "scanner":
            return self._emit_scanner(key, r, tk)
        if arch == "dns_tunnel":
            return self._emit_dns_tunnel(r, tk, "tunnel.example.net", 2.0 / 60.0, 48, 48)
        mm: MachineModel = p.models[0]
        if arch == "backup":
            return self._emit_backup(mm, key, r, tk, m)
        if arch == "health":
            return self._emit_health(mm, key, r, tk)
        return self._emit_periodic(mm, key, r, tk, m)

    def _emit_human(self, hm: HumanModel, r: np.random.Generator, tk: Tick,
                    m: Mods) -> List[list]:
        act = hm.activity(tk, m.force)
        lam = hm.rate_s * tk.sub * act * m.vol
        tot = float(lam.sum())
        if tot <= 1e-9:
            return []
        n = int(r.poisson(tot))
        if n == 0:
            return []
        # sessions: sizes, start sub-interval (∝ rate), think-time offsets
        k = 1 + int(r.binomial(n - 1, 1.0 / hm.sess_len)) if n > 1 else 1
        sizes = r.multinomial(n, np.full(k, 1.0 / k))
        sizes = sizes[sizes > 0]
        cell = r.choice(tk.M, size=len(sizes), p=lam / tot)
        starts = tk.t0 + (cell + r.random(len(sizes))) * tk.sub
        think = r.lognormal(hm.think_mu, hm.think_sigma, n)
        u_nav = r.random(n)
        u_id = r.random(n)
        u_err = r.random(n)
        u_var = r.random(n)
        s_up = r.lognormal(0.0, 0.3, n)
        s_down = r.lognormal(0.0, 0.5, n)
        vocab, tc, sc_ = hm.vocab, hm.trans_cum, hm.start_cum
        stack = hm.stack
        extra = m.extra_tpls
        u_x = r.random(n) if extra else None
        enum = m.enum
        out: List[list] = []
        i = 0
        t1 = tk.t1
        for size, st in zip(sizes.tolist(), starts.tolist()):
            offs = np.cumsum(think[i:i + size]) - think[i]
            span = float(offs[-1])
            if span >= tk.dt * 0.98:
                offs = offs * (tk.dt * 0.98 / max(span, 1e-9))
                span = float(offs[-1])
            if st + span >= t1:
                st = max(tk.t0, t1 - span - 1e-3)
            state = -1
            for j in range(size):
                u = u_nav[i]
                if state < 0:
                    state = min(bisect.bisect_right(sc_, u), hm.K - 1)
                else:
                    state = min(bisect.bisect_right(tc[state], u), hm.K - 1)
                t = vocab[state]
                if extra and u_x is not None:
                    acc = 0.0
                    for share, xt in extra:
                        acc += share
                        if u_x[i] < acc:
                            t = xt
                            break
                if enum is not None:
                    enum["next"] = enum.get("next", enum["start"]) + 1
                    path = f"/orders/view/{enum['next']}"
                    t = enum["tpl"]
                else:
                    oid = hm.id_lo + int(u_id[i] * u_id[i] * hm.id_w)
                    path = t.render(oid, u_var[i])
                if u_err[i] < hm.err:
                    status = (404, 403, 500, 400)[int(u_err[i] / hm.err * 4) % 4]
                elif t.cat == "auth" and t.method == "POST":
                    status = 302
                else:
                    status = 200
                up = t.base_up * s_up[i]
                down = t.base_down * hm.size_mult * s_down[i]
                out.append([st + float(offs[j]), "h", t.method, t.host, path, status,
                            up, down, t.srv_ms, stack])
                i += 1
            # the browser resolves the system host once per session (cached)
            if u_var[i - 1] < 0.6:
                h = vocab[0].host
                out.append([max(tk.t0, st - 0.05), "d", "A", h, "NOERROR", 0,
                            60.0 + len(h), 120.0, 2.0, stack])
        return out

    def _renewal(self, key: str, tk: Tick, period: float, jitter_abs: float,
                 r: np.random.Generator) -> List[float]:
        """Event times of a jittered periodic clock inside [t0, t1)."""
        nxt = self._state.get(key)
        if nxt is None or nxt < tk.t0 - 10 * period:
            nxt = tk.t0 + float(r.uniform(0.0, period))
        out = []
        while nxt < tk.t1:
            if nxt >= tk.t0:
                out.append(nxt)
            nxt += max(1.0, period + float(r.uniform(-jitter_abs, jitter_abs)))
        self._state[key] = nxt
        return out

    def _emit_periodic(self, mm: MachineModel, key: str, r: np.random.Generator, tk: Tick,
                       m: Mods) -> List[list]:
        """API clients and integration: polls of `batch` requests."""
        times = self._renewal(f"clk|{key}", tk, mm.period, mm.period * mm.jitter, r)
        if not times:
            return []
        out: List[list] = []
        load = 1.0
        if mm.archetype == "api":
            biz = float(np.interp(0.0, [0.0], [tk.biz.mean()]))
            load = 0.7 + 0.3 * biz
        lam = mm.batch * load * m.vol
        ep, cum, stack = mm.endpoints, mm.ep_cum, mm.stack
        for t in times:
            n = max(1, int(r.poisson(lam)))
            u = r.random(n)
            gaps = np.cumsum(r.uniform(0.02, 0.3, n))
            for j in range(n):
                e = ep[min(bisect.bisect_right(cum, float(u[j])), len(ep) - 1)]
                path = e.render(mm.id_lo + int(u[j] * 997) % 500, float(u[j]))
                status = 500 if r.random() < mm.err else 200
                up = e.base_up * float(r.lognormal(0, 0.2))
                down = e.base_down * 0.25 * float(r.lognormal(0, 0.4))
                out.append([t + float(gaps[j]), "h", e.method, e.host, path, status, up, down,
                            e.srv_ms * 0.3, stack])
        # resolver: library caches ~300 s
        for t in self._renewal(f"dns|{key}", tk, 300.0, 5.0, r):
            h = ep[0].host
            out.append([t, "d", "A", h, "NOERROR", 0, 60.0 + len(h), 120.0, 1.5, stack])
        return out

    def _emit_health(self, mm: MachineModel, key: str, r: np.random.Generator,
                     tk: Tick) -> List[list]:
        out: List[list] = []
        ep, stack = mm.endpoints, mm.stack
        for i, t in enumerate(self._renewal(f"clk|{key}", tk, 30.0, 1.0, r)):
            e = ep[(self._state.setdefault(f"rr|{key}", 0) + i) % len(ep)]
            out.append([t, "h", "GET", e.host, "/healthz", 200, 120.0, 800.0, 4.0, stack])
        self._state[f"rr|{key}"] = self._state.get(f"rr|{key}", 0) + len(out)
        for t in self._renewal(f"ntp|{key}", tk, 64.0, 0.5, r):
            out.append([t, "u", "ntp.corp.local", 123, "", 0, 76.0, 76.0, 1.0, stack])
        for t in self._renewal(f"probe|{key}", tk, 300.0, 2.0, r):
            out.append([t, "p", "reachable", "", "", 0, 0.0, 0.0, 0.0, stack])
        return out

    def _emit_backup(self, mm: MachineModel, key: str, r: np.random.Generator, tk: Tick,
                     m: Mods) -> List[list]:
        hour = m.backup_hour if m.backup_hour is not None else mm.hour
        out: List[list] = []
        e, stack = mm.endpoints[0], mm.stack
        for d in sorted(set(tk.dates)):
            jit = (crc(f"{key}|{d.isoformat()}") % 1201 - 600) / 60.0 / 60.0   # ±10 min
            a = self.clock.epoch(d, hour) + jit * 3600.0
            b = a + mm.duration
            lo, hi = max(a, tk.t0), min(b, tk.t1)
            if hi <= lo:
                continue
            n = int(r.poisson(mm.per_min * (hi - lo) / 60.0 * m.vol))
            ts = np.sort(r.uniform(lo, hi, n))
            for t in ts.tolist():
                oid = int(r.integers(1, 10 ** 6))
                out.append([t, "h", "PUT", e.host, f"/blob/{oid}", 200,
                            float(r.uniform(2e6, 6e6)), float(r.uniform(500, 2000)),
                            float(r.uniform(500, 3000)), stack])
            out.append([lo, "d", "A", e.host, "NOERROR", 0, 80.0, 120.0, 1.5, stack])
        return out

    def _emit_scanner(self, key: str, r: np.random.Generator, tk: Tick,
                      syn_per_min: float = 10.0, http_per_min: float = 3.0) -> List[list]:
        out: List[list] = []
        n = int(r.poisson(syn_per_min * tk.dt / 60.0))
        ts = r.uniform(tk.t0, tk.t1, n)
        a = r.integers(20, 41, n)
        b = r.integers(1, 10, n)
        c = r.integers(2, 255, n)
        ports = r.integers(1, 9000, n)
        for i in range(n):
            out.append([float(ts[i]), "s", f"10.{a[i]}.{b[i]}.{c[i]}", int(ports[i]), "", 0,
                        60.0, 0.0, 1.0, None])
        n = int(r.poisson(http_per_min * tk.dt / 60.0))
        st = fixed_stack("masscan", "Mozilla/5.0 zgrab/0.x", key)
        for t in r.uniform(tk.t0, tk.t1, n).tolist():
            lbl = "".join("abcdefghijklmnopqrstuvwxyz0123456789"[x] for x in r.integers(0, 36, 8))
            out.append([t, "h", "GET", "portal.corp.local", f"/{lbl}", 404, 80.0, 300.0, 5.0, st])
        return out

    def _emit_dns_tunnel(self, r: np.random.Generator, tk: Tick, zone: str, rate_s: float,
                         lmin: int, lmax: int, t_lo: Optional[float] = None,
                         t_hi: Optional[float] = None) -> List[list]:
        lo = tk.t0 if t_lo is None else max(tk.t0, t_lo)
        hi = tk.t1 if t_hi is None else min(tk.t1, t_hi)
        if hi <= lo:
            return []
        n = int(r.poisson(rate_s * (hi - lo)))
        out = []
        alpha = "abcdefghijklmnopqrstuvwxyz234567"
        for t in np.sort(r.uniform(lo, hi, n)).tolist():
            L = int(r.integers(lmin, lmax + 1))
            lbl = "".join(alpha[x] for x in r.integers(0, 32, L))
            q = f"{lbl}.{zone}"
            out.append([t, "d", "TXT", q, "NOERROR" if r.random() < 0.9 else "NXDOMAIN", 0,
                        60.0 + len(q), 200.0 + float(r.integers(0, 200)), 8.0, None])
        return out

    # ------------------------------------------------------------ modifiers
    @staticmethod
    def _post(evs: List[list], m: Mods, r: np.random.Generator) -> None:
        po, stack, status = m.path_override, m.stack, m.status
        up, down, up_rand = m.up, m.down, m.up_rand
        for e in evs:
            if e[EV_CH] != "h":
                if stack is not None and e[EV_CH] == "d":
                    e[EV_STACK] = stack
                continue
            if po is not None and r.random() < po[0]:
                t = po[1][int(r.integers(0, len(po[1])))]
                e[EV_A], e[EV_B] = t.method, t.host
                e[EV_C] = t.render(int(r.integers(1, 5000)), float(r.random()))
                e[EV_UP], e[EV_DOWN] = t.base_up, t.base_down * float(r.lognormal(0, 0.4))
            if up != 1.0:
                e[EV_UP] *= up
            if up_rand is not None:
                e[EV_UP] *= float(r.uniform(*up_rand))
            if down != 1.0:
                e[EV_DOWN] *= down
            if stack is not None:
                e[EV_STACK] = stack
            if status is not None and r.random() < status[0]:
                e[EV_ST] = status[1]
                e[EV_DOWN] = 300.0

    # --------------------------------------------------------- observations
    def _net(self, key: str) -> float:
        ip = key.partition("|")[2]
        subnet = ip.rsplit(".", 1)[0]
        base = 3.0 + (crc(f"rtt|{key.partition('|')[0]}|{subnet}") % 4200) / 100.0  # U(3,45)
        return base * (0.9 + (crc(f"rtt|{key}") % 200) / 1000.0)

    def _observations(self, key: str, evs: List[list], m: Mods) -> List[Observation]:
        system, _, entity = key.partition("|")
        r = self._rng(f"net|{key}")
        rtt0 = self._net(key) * m.rtt
        n = len(evs)
        jit = r.lognormal(0.0, 0.15, n)
        retr = r.poisson(0.02 * m.retrans, n)
        return [_make_obs(system, entity, e, e[EV_TS], rtt0 * float(jit[i]), int(retr[i]), None)
                for i, e in enumerate(evs)]

    def _aggregate(self, key: str, evs: List[list], m: Mods) -> List[Observation]:
        system, _, entity = key.partition("|")
        r = self._rng(f"net|{key}")
        rtt0 = self._net(key) * m.rtt
        groups: Dict[tuple, list] = {}
        for e in evs:
            st = e[EV_STACK]
            k = (e[EV_CH], e[EV_A], e[EV_B], e[EV_C], e[EV_ST], st.name if st else "")
            g = groups.get(k)
            if g is None:
                groups[k] = [e, [e[EV_TS]], e[EV_UP], e[EV_DOWN], e[EV_DUR]]
            else:
                g[1].append(e[EV_TS])
                g[2] += e[EV_UP]
                g[3] += e[EV_DOWN]
                g[4] += e[EV_DUR]
        out = []
        for g in groups.values():
            e, ts, up, down, dur = g
            w = len(ts)
            t0 = ts[0]
            if w > TS_SAMPLE_MAX:
                idx = np.linspace(0, w - 1, TS_SAMPLE_MAX).round().astype(int)
                sample = [round(ts[i] - t0, 3) for i in idx.tolist()]
            else:
                sample = [round(t - t0, 3) for t in ts]
            rep = list(e)
            rep[EV_UP], rep[EV_DOWN], rep[EV_DUR] = up / w, down / w, dur / w
            extra = {"count": w, "bytes_up_total": int(round(up)),
                     "bytes_down_total": int(round(down)), "ts_sample": sample}
            retr = int(r.poisson(0.02 * m.retrans * w))
            out.append(_make_obs(system, entity, rep, t0, rtt0 * float(r.lognormal(0, 0.1)),
                                 retr, extra))
        return out

    # --------------------------------------------------- scenario effects
    # Each _fx_<kind>(sc, tk, mods, extra) adds modifiers for the scenario's
    # entities and/or extra events (keyed 'sys|ip') for this tick.
    def _m(self, mods: Dict[str, Mods], key: str) -> Mods:
        m = mods.get(key)
        if m is None:
            m = mods[key] = Mods()
        return m

    def _span(self, sc: Scenario, tk: Tick) -> Tuple[float, float]:
        return max(tk.t0, sc.t_start), min(tk.t1, sc.t_end)

    def _sc_rng(self, sc: Scenario) -> np.random.Generator:
        return self._rng(f"scenario|{sc.scenario_id}|{sc.keys()[0] if sc.entities else ''}")

    def _periodic_http(self, sc: Scenario, key: str, tk: Tick, period: float, jit: float,
                       method: str, host: str, path: str, up: float, down: float,
                       stack: Optional[Stack], lo: Optional[float] = None,
                       hi: Optional[float] = None) -> List[list]:
        r = self._sc_rng(sc)
        a, b = self._span(sc, tk)
        if lo is not None:
            a = max(a, lo)
        if hi is not None:
            b = min(b, hi)
        out = []
        for t in self._renewal(f"sc|{sc.scenario_id}|{key}|{host}", tk, period, jit, r):
            if a <= t < b:
                out.append([t, "h", method, host, path, 200, up * float(r.lognormal(0, 0.1)),
                            down * float(r.lognormal(0, 0.2)), 40.0, stack])
        return out

    def _stack_of(self, key: str) -> Optional[Stack]:
        p = self.personas.get(key)
        if p and p.models:
            return getattr(p.models[0], "stack", None)
        return None

    def _human_activity_at(self, key: str, tk: Tick) -> np.ndarray:
        p = self.personas.get(key)
        if p is None or not p.models or not isinstance(p.models[0], HumanModel):
            return np.ones(tk.M)
        return p.models[0].activity(tk)

    def _fx_background(self, sc, tk, mods, extra) -> None:
        """Truth only (calendar, NAT, idle nights, DST, sanctioned automation)."""

    def _fx_beacon_replace(self, sc, tk, mods, extra) -> None:              # T1
        key = sc.keys()[0]
        self._m(mods, key).replace = True
        p = sc.params
        host = p.get("host", "c2.example.net")
        evs = self._periodic_http(sc, key, tk, p.get("period_s", 300.0), p.get("jitter_s", 3.0),
                                  "GET", host, p.get("path", "/ping"), 120.0, 800.0,
                                  self._stack_of(key))
        for t in self._renewal(f"scdns|{sc.scenario_id}", tk, 3600.0, 60.0, self._sc_rng(sc)):
            if sc.t_start <= t < sc.t_end:
                evs.append([t, "d", "A", host, "NOERROR", 0, 80.0, 120.0, 2.0, None])
        extra.setdefault(key, []).extend(evs)

    def _fx_exfil_business(self, sc, tk, mods, extra) -> None:              # T2
        key = sc.keys()[0]
        p = sc.params
        r = self._sc_rng(sc)
        a, b = self._span(sc, tk)
        act = self._human_activity_at(key, tk)
        days = int((tk.t0 - sc.t_start) // 86400.0)
        size = p.get("bytes", 200_000.0) * p.get("growth", 1.6) ** max(0, days)
        out = []
        for i, mid in enumerate(tk.mids.tolist()):
            lo, hi = max(a, mid - tk.sub / 2), min(b, mid + tk.sub / 2)
            if hi <= lo or act[i] < 0.3:
                continue
            n = int(r.poisson((hi - lo) / p.get("every_s", 900.0)))
            for t in r.uniform(lo, hi, n).tolist():
                out.append([t, "h", "POST", p.get("host", "ext-store.example.net"), "/upload",
                            200, size * float(r.lognormal(0, 0.1)), 600.0, 900.0,
                            self._stack_of(key)])
        extra.setdefault(key, []).extend(out)

    def _fx_bytes_ramp(self, sc, tk, mods, extra) -> None:                  # T3
        days = max(0.0, (tk.t0 + tk.dt / 2 - sc.t_start) / 86400.0)
        for key in sc.keys():
            self._m(mods, key).up *= sc.params.get("growth", 1.15) ** days

    def _fx_volume_ramp(self, sc, tk, mods, extra) -> None:                 # L5
        days = max(0.0, (tk.t0 + tk.dt / 2 - sc.t_start) / 86400.0)
        for key in sc.keys():
            self._m(mods, key).vol *= sc.params.get("growth", 1.02) ** days

    def _fx_duty_upload(self, sc, tk, mods, extra) -> None:                 # T4, T4b
        k = int((tk.t0 - sc.t_start) // sc.params.get("tick_s", REF_TICK_S))
        if sc.params.get("pattern") == "alternate":
            on = k % 2 == 0
        else:
            on = (crc(f"{sc.scenario_id}|{self.seed}|{k}") % 10_000) / 10_000.0 \
                < sc.params.get("p", 0.3)
        if on:
            for key in sc.keys():
                self._m(mods, key).up *= sc.params.get("factor", 20.0)

    def _fx_offhours(self, sc, tk, mods, extra) -> None:                    # T5
        wins = [(float(a), float(b)) for a, b in sc.params["windows"]]
        for key in sc.keys():
            self._m(mods, key).force.extend(w for w in wins if w[0] < tk.t1 and w[1] > tk.t0)

    def _fx_second_client(self, sc, tk, mods, extra) -> None:               # T6, T6b
        key = sc.keys()[0]
        p = self.personas[key]
        hm: HumanModel = p.models[0]
        st = sc.params.get("stack")
        if st is None:
            st = Stack("t6-python", PY_REQUESTS, ja3(_LIB_CIPH, _LIB_EXT, (64123,)), 64, 29200)
        r = self._sc_rng(sc)
        a, b = self._span(sc, tk)
        evs = self._emit_human(hm, r, tk, _NO_MODS)
        for e in evs:
            e[EV_STACK] = st
        extra.setdefault(key, []).extend(e for e in evs if a <= e[EV_TS] < b)

    def _fx_rare_resource(self, sc, tk, mods, extra) -> None:               # T7
        key = sc.keys()[0]
        if float(self._human_activity_at(key, tk).max()) < 0.3:
            return
        r = self._sc_rng(sc)
        a, b = self._span(sc, tk)
        n = max(1, int(round(tk.dt / REF_TICK_S * sc.params.get("per_tick", 1.0))))
        t = tpl("GET", SYSTEM_HOSTS[sc.system][0], sc.params.get("path", "/hr/salary/export"),
                "private")
        evs = [[float(x), "h", t.method, t.host, t.fmt, 200, t.base_up, t.base_down, t.srv_ms,
                self._stack_of(key)] for x in r.uniform(a, b, n)]
        extra.setdefault(key, []).extend(evs)

    def _fx_enumeration(self, sc, tk, mods, extra) -> None:                 # T8
        key = sc.keys()[0]
        st = self._state.setdefault(f"enum|{sc.scenario_id}", {
            "start": int(sc.params.get("start_id", 500_000)),
            "tpl": tpl("GET", SYSTEM_HOSTS[sc.system][0], "/orders/view/{id}", "view")})
        self._m(mods, key).enum = st

    def _fx_impersonate(self, sc, tk, mods, extra) -> None:                 # T9
        key = sc.keys()[0]
        src = self.personas[sc.params["looks_like_key"]].models[0]
        self._m(mods, key).swap = src

    def _fx_unknown_persona(self, sc, tk, mods, extra) -> None:             # T9b
        key = sc.keys()[0]
        mk = f"swap|{sc.scenario_id}|{key}"
        hm = self._state.get(mk)
        if hm is None:
            hm = self._state[mk] = build_human(key, tag=f"{sc.params.get('tag', 'new')}|{key}|"
                                                         f"{self.seed}")
        self._m(mods, key).swap = hm

    def _fx_jitter_beacon(self, sc, tk, mods, extra) -> None:               # T10
        key = sc.keys()[0]
        p = sc.params
        period = p.get("period_s", 300.0)
        evs = self._periodic_http(sc, key, tk, period, period * p.get("jitter", 0.3), "POST",
                                  p.get("host", "cdn-upd.example.net"), "/u", p.get("bytes", 1200.0),
                                  400.0, self._stack_of(key))
        extra.setdefault(key, []).extend(evs)

    def _fx_cold_start(self, sc, tk, mods, extra) -> None:                  # T11
        """Spawned attackers emit through their personas; the exfil one also
        uploads `bytes_per_tick` (per 900 s) to ext-drop."""
        p = sc.params
        key = p.get("exfil_key")
        if key:
            per = p.get("uploads_per_tick", 4)
            size = p.get("bytes_per_tick", 2e6) / per
            extra.setdefault(key, []).extend(self._periodic_http(
                sc, key, tk, REF_TICK_S / per, 20.0, "POST", p.get("host", "ext-drop.example.com"),
                "/put", size, 500.0, self._stack_of(key) or ORG_STANDARD))

    def _fx_cred_stuffing(self, sc, tk, mods, extra) -> None:               # T12
        key = sc.keys()[0]
        p = sc.params
        r = self._sc_rng(sc)
        a, b = self._span(sc, tk)
        n = int(r.poisson(p.get("per_tick", 6.0) * (b - a) / REF_TICK_S))
        host = SYSTEM_HOSTS[sc.system][0]
        st = self._stack_of(key)
        evs = [[float(t), "h", "POST", host, "/login",
                401 if r.random() < p.get("fail", 0.9) else 302, 600.0, 400.0, 30.0, st]
               for t in r.uniform(a, b, n)]
        extra.setdefault(key, []).extend(evs)

    def _fx_path_shift(self, sc, tk, mods, extra) -> None:                  # T13, T17
        tpls = [tpl(m, h, f, c) for m, h, f, c in sc.params["templates"]]
        for key in sc.keys():
            self._m(mods, key).path_override = (sc.params.get("share", 0.6), tpls)

    def _fx_dns_tunnel(self, sc, tk, mods, extra) -> None:                  # T14 (existing host)
        key = sc.keys()[0]
        p = sc.params
        evs = self._emit_dns_tunnel(self._sc_rng(sc), tk, p.get("zone", "tun.example.org"),
                                    p.get("per_tick", 2.0) / REF_TICK_S, p.get("lmin", 30),
                                    p.get("lmax", 48), sc.t_start, sc.t_end)
        extra.setdefault(key, []).extend(evs)

    def _fx_poison_ramp(self, sc, tk, mods, extra) -> None:                 # T15
        key = sc.keys()[0]
        p = sc.params
        tick_s = p.get("tick_s", REF_TICK_S)
        k = int((tk.t0 - sc.t_start) // tick_s)
        n_ramp = int(p.get("ramp_ticks", 200))
        base = p.get("bytes", 100_000.0) * p.get("rate", 1.03) ** min(k, n_ramp)
        if k >= n_ramp:
            base *= p.get("strike", 30.0)
        per = 3
        out = self._periodic_http(sc, key, tk, tick_s / per, 20.0, "PUT", "backup.corp.local",
                                  "/blob/inc", base / per, 400.0, self._stack_of(key))
        if k >= p.get("external_at", 50):
            out += self._periodic_http(sc, key, tk, tick_s / per, 20.0, "PUT",
                                       p.get("ext_host", "bk-mirror.example.net"), "/blob/inc",
                                       base / per / 2, 400.0, self._stack_of(key))
        extra.setdefault(key, []).extend(out)

    def _fx_weak_signals(self, sc, tk, mods, extra) -> None:                # T16
        key = sc.keys()[0]
        p = sc.params
        m = self._m(mods, key)
        t = tk.t0
        st0 = self._stack_of(key) or ORG_STANDARD
        if t >= p["t_ja3"]:
            m.stack = Stack(st0.name + "-minor", st0.ua, st0.ja3 + "-0", st0.ttl, st0.win) \
                if False else Stack(st0.name + "-minor", st0.ua,
                                    st0.ja3.replace(",29-23-24", "-57,29-23-24"), st0.ttl, st0.win)
        if t >= p["t_bytes"]:
            m.down *= p.get("bytes_mult", 1.3)
            m.up *= p.get("bytes_mult", 1.3)
        a, b = p["t_rare"], p["t_rare"] + 1.0
        if a < tk.t1 and b > tk.t0:
            h = SYSTEM_HOSTS[sc.system][0]
            extra.setdefault(key, []).append(
                [a, "h", "GET", h, p.get("rare_path", "/crm/lead/export"), 200, 500.0, 9000.0,
                 90.0, m.stack or st0])
        m.force.append((p["t_off"], p["t_off"] + p.get("off_len", 900.0)))

    def _fx_bytes_mult(self, sc, tk, mods, extra) -> None:                  # T18
        for key in sc.keys():
            self._m(mods, key).up_rand = tuple(sc.params.get("range", (3.0, 4.0)))

    def _fx_ip_hop(self, sc, tk, mods, extra) -> None:                      # T19
        p = sc.params
        for sp in sc.spawns:
            if sp.t_from < tk.t1 and sp.t_to > tk.t0:
                per = 4
                out = self._periodic_http(sc, sp.key, tk, REF_TICK_S / per, 20.0, "POST",
                                          p.get("host", "ext-sync.example.io"), "/sync",
                                          p.get("bytes_per_tick", 1e6) / per, 500.0,
                                          self._stack_of(sp.key), sp.t_from, sp.t_to)
                extra.setdefault(sp.key, []).extend(out)

    def _fx_lateral(self, sc, tk, mods, extra) -> None:                     # T20
        src = sc.keys()[0]
        ip = src.partition("|")[2]
        if float(self._human_activity_at(src, tk).max()) < 0.3:
            return
        r = self._sc_rng(sc)
        a, b = self._span(sc, tk)
        st = self._stack_of(src)
        for system, method, host, path in sc.params.get("targets", []):
            n = int(r.poisson(sc.params.get("per_tick", 3.0) * (b - a) / REF_TICK_S))
            for t in r.uniform(a, b, n).tolist():
                extra.setdefault(f"{system}|{ip}", []).append(
                    [t, "h", method, host, path, 200, 500.0, 4000.0, 40.0, st])

    def _fx_pool_upload(self, sc, tk, mods, extra) -> None:                 # T21
        p = sc.params
        for key in sc.keys():
            t_on = p["onsets"][key]
            if t_on >= tk.t1:
                continue
            extra.setdefault(key, []).extend(self._periodic_http(
                sc, key, tk, p.get("every_s", 900.0), 30.0, "POST",
                p.get("host", "telemetry-sync.example.org"), "/v2/ingest",
                p.get("bytes", 30_000.0), 400.0, self._stack_of(key), lo=t_on))

    def _fx_volume(self, sc, tk, mods, extra) -> None:                      # L1
        wins = sc.params.get("windows") or [(sc.t_start, sc.t_end)]
        if any(a < tk.t1 and b > tk.t0 for a, b in wins):
            for key in sc.keys():
                self._m(mods, key).vol *= sc.params.get("factor", 2.5)

    def _fx_stack_change(self, sc, tk, mods, extra) -> None:                # L2, L14
        p = sc.params
        for key in sc.keys():
            t_on = p.get("onsets", {}).get(key, sc.t_start)
            if tk.t0 < t_on:
                continue
            old = self._stack_of(key)
            if p.get("only_standard") and old is not ORG_STANDARD:
                continue
            new = p.get("stack")
            if new is None:
                st0 = old or ORG_STANDARD
                ua = re.sub(r"(\d+)\.0(\.0\.0)?", lambda mm: f"{int(mm.group(1)) + 1}.0"
                            + (mm.group(2) or ""), st0.ua, count=1)
                new = Stack(st0.name + "-upd", ua, st0.ja3.replace(",29-23-24", "-41,29-23-24"),
                            st0.ttl, st0.win)
            self._m(mods, key).stack = new

    def _fx_new_resource(self, sc, tk, mods, extra) -> None:                # L3
        p = sc.params
        t = tpl("GET", SYSTEM_HOSTS[sc.system][0], p.get("path", "/v2/orders"), "list")
        for key in sc.keys():
            if tk.t0 >= p.get("onsets", {}).get(key, sc.t_start):
                self._m(mods, key).extra_tpls.append((p.get("share", 0.1), t))

    def _fx_renumber(self, sc, tk, mods, extra) -> None:                    # L6
        self._m(mods, sc.params["from_key"]).entity_as = sc.params["to_key"]

    def _fx_schedule_shift(self, sc, tk, mods, extra) -> None:              # L9
        for key in sc.keys():
            self._m(mods, key).backup_hour = sc.params.get("hour", 3.0)

    def _fx_outage(self, sc, tk, mods, extra) -> None:                      # L10
        p = sc.params
        a, b = p["outage"]
        c, d = p["degrade"]
        for key in sc.keys():
            m = self._m(mods, key)
            if a < tk.t1 and b > tk.t0:
                m.status = (1.0, 503)
            if c < tk.t1 and d > tk.t0:
                m.rtt *= p.get("rtt_mult", 3.0)
                m.retrans *= p.get("retrans_mult", 5.0)

    def _fx_absence(self, sc, tk, mods, extra) -> None:                     # L12
        for key in sc.keys():
            self._m(mods, key).vol = 0.0

    def _fx_explorer(self, sc, tk, mods, extra) -> None:                    # L16
        key = sc.keys()[0]
        per_day = sc.params.get("per_day", 5)
        days = int((tk.t0 - sc.t_start) // 86400.0) + 1
        cand = self._state.get(f"explore|{key}")
        if cand is None:
            cand = self._state[f"explore|{key}"] = self._explore_candidates(key)
        added = cand[:per_day * days]
        if added:
            share = min(0.3, 0.02 * len(added))
            self._m(mods, key).extra_tpls.extend((share / len(added), t) for t in added)

    def _explore_candidates(self, key: str) -> List[Template]:
        """Templates common among the entity's role peers but rare for itself,
        most common first."""
        me = self.personas[key].models[0]
        peers = [p.models[0] for k, p in self.personas.items()
                 if k != key and p.system == me.system and p.archetype == "interactive"
                 and p.models and isinstance(p.models[0], HumanModel)]
        if not peers:
            return []
        avg = np.mean([pm.visit for pm in peers if pm.K == me.K] or [me.visit], axis=0)
        order = np.argsort(-avg)
        return [me.vocab[i] for i in order if me.visit[i] < 0.01 and avg[i] > 0.005]

    def _fx_exfil_bulk(self, sc, tk, mods, extra) -> None:                  # demo (v1 'exfil')
        key = sc.keys()[0]
        r = self._sc_rng(sc)
        a, b = self._span(sc, tk)
        n = int(r.poisson(6.0 * (b - a) / 60.0))
        extra.setdefault(key, []).extend(
            [float(t), "h", "POST", "ext-store.example.net", "/upload", 200,
             float(r.uniform(3e6, 8e6)), float(r.uniform(500, 1500)), 2000.0, self._stack_of(key)]
            for t in r.uniform(a, b, n))


def _ev_ts(e: list) -> float:
    return e[EV_TS]


def _make_obs(system: str, entity: str, e: list, ts: float, rtt: float, retr: int,
              extra: Optional[Dict[str, Any]]) -> Observation:
    ch = e[EV_CH]
    up, down = int(e[EV_UP]), int(e[EV_DOWN])
    st: Optional[Stack] = e[EV_STACK]
    ex = extra if extra is not None else {}
    if ch == "h":
        host = e[EV_B]
        return Observation(
            ts=ts, system=system, entity=entity, peer=host_ip(host), l3_proto="ip",
            l4_proto="tcp", dst_port=443, bytes_up=up, bytes_down=down,
            pkts_up=max(1, up // 1400 + 2), pkts_down=max(1, down // 1400 + 2),
            ttl=st.ttl if st else 64, win_size=st.win if st else 64240, retransmits=retr,
            rtt_ms=rtt, duration_ms=e[EV_DUR] + 2.0 * rtt, app_proto="http",
            http_method=e[EV_A], http_host=host, http_path=e[EV_C], http_status=e[EV_ST],
            user_agent=st.ua if st else "", content_type="text/html",
            tls_version="TLS1.3", tls_cipher="TLS_AES_128_GCM_SHA256", tls_sni=host,
            ja3=st.ja3 if st else "", ja3s="771,4865,0-43-51", extra=ex)
    if ch == "d":
        return Observation(
            ts=ts, system=system, entity=entity, peer=RESOLVER_IP, l3_proto="ip",
            l4_proto="udp", dst_port=53, app_proto="dns", dns_qname=e[EV_B], dns_qtype=e[EV_A],
            dns_rcode=e[EV_C], bytes_up=up, bytes_down=down, pkts_up=1, pkts_down=1,
            ttl=st.ttl if st else 64, rtt_ms=rtt * 0.2, duration_ms=e[EV_DUR], extra=ex)
    if ch == "u":
        return Observation(
            ts=ts, system=system, entity=entity, peer=host_ip(e[EV_A]), l3_proto="ip",
            l4_proto="udp", dst_port=int(e[EV_B]), bytes_up=up, bytes_down=down, pkts_up=1,
            pkts_down=1, ttl=st.ttl if st else 64, rtt_ms=rtt * 0.5, duration_ms=e[EV_DUR],
            app_proto="ntp", extra=ex)
    if ch == "s":
        return Observation(
            ts=ts, system=system, entity=entity, peer=e[EV_A], l3_proto="ip", l4_proto="tcp",
            dst_port=int(e[EV_B]), tcp_flags="SYN", bytes_up=up, bytes_down=0, pkts_up=1,
            pkts_down=0, ttl=64, rtt_ms=rtt, duration_ms=e[EV_DUR], extra=ex)
    reach = Reachability.REACHABLE
    return Observation(
        ts=ts, system=system, entity=entity, peer=entity, method=AcquisitionMethod.ACTIVE_PROBE,
        reachability=reach, rtt_ms=rtt, hop_count=3 + crc(entity) % 6, open_ports=(443, 80),
        extra=ex)


# --------------------------------------------------------------------------- #
# Demo scenario set (no pack): the v1 live anomaly schedule
# --------------------------------------------------------------------------- #
def demo_scenarios() -> List[Scenario]:
    return [
        Scenario("demo-exfil", "malicious", "erp-prod", ["10.20.1.13"], "exfil_bulk",
                 live_tick=3, truth={"expected_axes": ["exfil"], "perturbed_features": ["bytes_up"],
                                     "required_severity": "medium"}),
        Scenario("T11-scan", "malicious", "oa-portal", ["10.30.2.99"], "background", live_tick=5,
                 spawns=[Spawn("oa-portal|10.30.2.99", "scanner", archetype="scanner")],
                 truth={"expected_detectors": ["B02 new_entity_unmatched"]}),
        Scenario("T1-demo", "malicious", "api-gateway", ["10.40.4.52"], "beacon_replace",
                 mode="replace", live_tick=7, truth={"expected_detectors": ["B04 marg_int"]}),
        Scenario("T11-dns", "malicious", "erp-prod", ["10.20.7.77"], "background", live_tick=9,
                 spawns=[Spawn("erp-prod|10.20.7.77", "dns_tunnel", archetype="dns_tunnel")],
                 truth={"expected_detectors": ["B02 new_entity_unmatched"]}),
        Scenario("T12-demo", "malicious", "oa-portal", ["10.30.2.21"], "cred_stuffing",
                 live_tick=11, params={"per_tick": 50.0 * 15},
                 truth={"expected_axes": ["credential"]}),
    ]
