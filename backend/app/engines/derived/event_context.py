"""P01 EventContext (`derived.event_context`) — time, calendar and session
context of behaviour events, and metric-window events.

docs/lib3/progressive.md §6.2.3, card P01. Library 2 (derived).

evt.ctx (row-aligned with the tick's evt.batch of the same system):
    ctx.tod_min (local minute 0..1439), ctx.dow, ctx.daytype (workday |
    nonworkday), ctx.dayclass (workday | weekend | holiday | makeup),
    ctx.dom, ctx.mend (1 on the last 3 workdays of the month), ctx.when
    ((daytype, minute), the time hierarchy of lib/phier), ctx.sid, ctx.sess_pos,
    ctx.prev_route, ctx.think_s, ctx.sess_age_s.
    meta: {'day': local date ordinal, 'normal_prev_day': bool}.
Sessions: per system an LRU keyed (ip, sess.key) (cap S_sess) of
    (last_ts, sid, pos, last_route, start_ts); a new session starts when the
    gap exceeds G(s): P01's own valley of the decayed log10 inter-event gap
    histogram (Otsu between the two largest modes, clamped to [60 s, 2 h]),
    30 min until >= 200 gaps were seen. The session id is a hash of
    (ip, sess.key, start_ts), so any cadence gives the same ids and a gap
    across a tick boundary keeps one session.
Normal days: model.pcal@(s, '__system__') keeps the mass per local date and
    day class (60 d); a finished day is normal when it is not a holiday and
    its mass is within [0.25, 4] x the mean of the previous (up to 28 d) days
    of its class. Stale timers (P04) and schema-change detection (P02) count
    normal days only.
evt.win (kind win): at each H-grain decision tick, one event per IP active in
    the grain, attributes m.<metric> = the IP's fresh scalar value of raw /
    derived metric names: P05's kept `win` targets plus a rotating slice
    (crc32 phase) of the other names, A_win in total, so a new metric name
    enters the registry the day it appears. At most W_max events per system:
    priority sampling (weight 10 for earned and new IPs, 1 else) with HT mass.

Inert unless config['progressive']['enabled'].
"""
from __future__ import annotations

import calendar as _cal
import datetime as _dt
import hashlib
import math
import zlib
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import SYSTEM_ENTITY
from ..behavior.lib import grains as GR
from ..behavior.lib import m_ptree as MP
from ..behavior.lib import pevent as EV
from ..behavior.lib import psketch as PS
from ..behavior.lib import timebins as TB
from ..behavior.lib.combine import seeded_uniform

PCAL = "model.pcal"
GAP_BINS = 64
GAP_LO, GAP_HI = 0.0, 6.0             # log10 seconds
GAP_MIN_N = 200.0
GAP_CLAMP = (60.0, 7200.0)
BAND = (0.25, 4.0)
CAL_DAYS = 60
NEW_IP_WEIGHT = 10.0
NO_KEY = "∅"


def _sid(ip: str, key: str, start: float) -> str:
    h = hashlib.blake2b(f"{ip}|{key}|{start:.3f}".encode("utf-8", "surrogatepass"), digest_size=8)
    return h.hexdigest()


def _otsu(hist: np.ndarray) -> Optional[int]:
    """Otsu threshold index on a 1-D histogram (between-class variance max)."""
    tot = hist.sum()
    if tot <= 0:
        return None
    idx = np.arange(hist.size)
    w0 = np.cumsum(hist)
    w1 = tot - w0
    m0 = np.cumsum(hist * idx)
    mu_t = m0[-1]
    with np.errstate(divide="ignore", invalid="ignore"):
        between = (mu_t * w0 - m0 * tot) ** 2 / (w0 * w1)
    between[~np.isfinite(between)] = -1
    k = int(np.argmax(between[:-1])) if hist.size > 1 else None
    return k


class SessionGap:
    """G(s): valley of the decayed log10 inter-event gap histogram."""

    def __init__(self) -> None:
        self.h = PS.DecayedVector([PS.H_M] * GAP_BINS)
        self.n = 0.0

    def add(self, gaps: np.ndarray, t: float) -> None:
        g = np.asarray(gaps, dtype=np.float64)
        g = g[np.isfinite(g) & (g > 0)]
        if g.size == 0:
            return
        b = np.clip(((np.log10(g) - GAP_LO) / (GAP_HI - GAP_LO) * GAP_BINS).astype(int), 0,
                    GAP_BINS - 1)
        self.h.add(t, np.bincount(b, minlength=GAP_BINS).astype(float))
        self.n += g.size

    def gap(self, t: float, default: float) -> float:
        if self.n < GAP_MIN_N:
            return default
        k = _otsu(self.h.read(t))
        if k is None:
            return default
        g = 10 ** (GAP_LO + (k + 1) * (GAP_HI - GAP_LO) / GAP_BINS)
        return float(min(GAP_CLAMP[1], max(GAP_CLAMP[0], g)))


def day_class(d: _dt.date, cal: TB.Calendar) -> str:
    if d in cal.makeup_workdays:
        return "makeup"
    if d in cal.holidays:
        return "holiday"
    return "weekend" if d.weekday() >= 5 else "workday"


_MEND_CACHE: Dict[Tuple[int, int, int], frozenset] = {}


def month_end_days(y: int, m: int, cal: TB.Calendar) -> frozenset:
    """The last three workdays of month (y, m) under the calendar."""
    key = (y, m, id(cal))
    hit = _MEND_CACHE.get(key)
    if hit is not None:
        return hit
    last = _cal.monthrange(y, m)[1]
    out = []
    d = _dt.date(y, m, last)
    while len(out) < 3 and d.month == m:
        if TB.day_type(d, cal) == "workday":
            out.append(d.day)
        d -= _dt.timedelta(days=1)
    res = frozenset(out)
    if len(_MEND_CACHE) > 512:
        _MEND_CACHE.clear()
    _MEND_CACHE[key] = res
    return res


class EventContextEngine(Engine):
    name = "derived.event_context"
    layer = "derived"
    consumes = [EV.EVT_BATCH, "model.attrsel", "model.budget", "model.attr"]
    produces = [EV.EVT_CTX, EV.EVT_WIN, PCAL]
    description = "P01: event time / calendar / session context and metric-window events"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self._sess: Dict[str, PS.LRU] = {}
        self._gap: Dict[str, SessionGap] = {}
        self.last_stats: Dict[str, Any] = {}

    # ----------------------------------------------------------- calendar
    def _time_ctx(self, ts: float, tz: str, cal: TB.Calendar) -> Dict[str, Any]:
        dtl = TB.local_datetime(ts, tz)
        d = dtl.date()
        minute = dtl.hour * 60 + dtl.minute + dtl.second / 60.0
        dtype = TB.day_type(d, cal)
        return {"ctx.tod_min": float(round(minute, 3)), "ctx.dow": d.weekday(),
                "ctx.daytype": dtype, "ctx.dayclass": day_class(d, cal), "ctx.dom": d.day,
                "ctx.mend": 1 if d.day in month_end_days(d.year, d.month, cal) else 0,
                "ctx.when": ("wd" if dtype == "workday" else "nwd", int(minute))}

    def _calendar_update(self, store: Any, s: str, day: int, dclass: str, mass: float) -> bool:
        """Record mass per local date; return whether the previous day was normal."""
        m = store.get_model(s, SYSTEM_ENTITY, PCAL)
        if not isinstance(m, dict):
            m = {"days": {}, "normal": {}}
        days = m["days"]
        rec = days.setdefault(day, {"mass": 0.0, "class": dclass})
        rec["mass"] += float(mass)
        prev = day - 1
        if prev in days and prev not in m["normal"]:
            pr = days[prev]
            same = [v["mass"] for k, v in days.items()
                    if k < prev and k >= prev - 28 and v["class"] == pr["class"]]
            if pr["class"] == "holiday":
                ok = False
            elif len(same) >= 3:
                mu = float(np.mean(same))
                ok = mu > 0 and BAND[0] * mu <= pr["mass"] <= BAND[1] * mu
            else:
                ok = True
            m["normal"][prev] = bool(ok)
        for k in [k for k in days if k < day - CAL_DAYS]:
            days.pop(k, None)
            m["normal"].pop(k, None)
        store.put_model(s, SYSTEM_ENTITY, PCAL, m)
        return bool(m["normal"].get(prev, True))

    # ------------------------------------------------------------ sessions
    def _sessions(self, s: str, batch: EV.EventBatch, cap: int, default_gap: float,
                  cols: Dict[str, List[Any]]) -> None:
        lru = self._sess.get(s)
        if lru is None:
            lru = self._sess[s] = PS.LRU(cap)
        elif lru.cap != cap:
            lru.set_cap(cap)
        sg = self._gap.setdefault(s, SessionGap())
        G = sg.gap(batch.t1, default_gap)
        order = np.argsort(batch.ts, kind="stable")
        gaps = []
        for i in order:
            i = int(i)
            ip = batch.ip_of(i)
            key = batch.get("sess.key", i, NO_KEY)
            k = (ip, key)
            ts = float(batch.ts[i])
            route = batch.get("http.route", i)
            if route is EV.ABSENT:
                for alt in ("tls.sni", "dns.qname", "net.dst"):
                    route = batch.get(alt, i)
                    if route is not EV.ABSENT:
                        break
            st = lru.get(k)
            if st is not None and ts >= st[0]:
                gaps.append(ts - st[0])
            if st is None or ts - st[0] > G:
                start = ts
                sid = _sid(ip, key, start)
                pos, prev, think = 0, EV.ABSENT, EV.ABSENT
            else:
                sid, start = st[1], st[4]
                pos = st[2] + 1
                prev = st[3]
                think = max(0.0, ts - st[0])
            lru.put(k, (max(ts, st[0]) if st is not None and pos else ts, sid, pos, route, start))
            cols["ctx.sid"][i] = sid
            cols["ctx.sess_pos"][i] = pos
            cols["ctx.prev_route"][i] = prev
            cols["ctx.think_s"][i] = think
            cols["ctx.sess_age_s"][i] = ts - start
        sg.add(np.asarray(gaps), batch.t1)

    # ----------------------------------------------------------- windows
    def _window_events(self, ctx: Context, s: str, pc: Mapping[str, Any], tz: str,
                       cal: TB.Calendar) -> Optional[EV.EventBatch]:
        store = ctx.store
        now = float(ctx.now)
        mode = GR.mode_of(ctx.config)
        if not GR.decision(now, float(ctx.window_s), "h", mode):
            return None
        lo, hi = GR.window(now, "h", float(ctx.window_s), mode)
        active: Dict[str, float] = {}
        for _, b in store.batches_since(s, EV.EVT_BATCH, lo):
            for ip in b.ips:
                active[ip] = 1.0
        if not active:
            return None
        key = MP.tree_key(store, s)
        budget = MP.budget_for(store, key)
        w_max = int(budget.get("w_max", pc["defaults"]["w_max"]))
        a_win = int(budget.get("a_win", pc["defaults"]["a_win"]))
        earned = set(budget.get("earned") or ())
        ips = sorted(active)
        wts = []
        for ip in ips:
            fs = store.first_seen(s, ip)
            new = fs is not None and fs > lo
            wts.append(NEW_IP_WEIGHT if (ip in earned or new) else 1.0)
        u = [seeded_uniform(s, now, "win", ip) for ip in ips]
        kept, ht = EV.priority_sample(wts, w_max, u)
        wts_a = np.asarray(wts)
        # metric names: kept `win` targets first, then a rotating crc32 slice
        sel = MP.get_model(store, key, MP.ATTRSEL) or {}
        kept_names = []
        if isinstance(sel, Mapping):
            kept_names = [n[2:] for n in ((sel.get("targets_sys") or {}).get(EV.KIND_WIN) or [])
                          if str(n).startswith("m.")]
        grain_idx = int(math.floor(now / GR.GRAIN_S["h"])) if GR.canonical(mode) \
            else int(math.floor(now / max(1.0, float(ctx.window_s))))
        tctx = self._time_ctx(now, tz, cal)
        b = EV.BatchBuilder(s, EV.KIND_WIN)
        for j, i in enumerate(kept):
            ip = ips[int(i)]
            cand = sorted(set(store.raw_names(s, ip)) | set(store.derived_names(s, ip)))
            names = [n for n in kept_names if n in cand][:a_win]
            rest = [n for n in cand if n not in set(names)]
            slots = max(0, a_win - len(names))
            if rest and slots:
                K = max(1, math.ceil(len(rest) / slots))
                pick = [n for n in rest if (zlib.crc32(n.encode("utf-8")) + grain_idx) % K == 0]
                names += pick[:slots]
            snap = store.snapshot(s, ip, now, names=names) if names else {}
            attrs: Dict[str, Any] = {EV.attr_name("m", n): v for n, v in snap.items()}
            attrs["net.src"] = ip
            attrs["ev.ch"] = "win"
            attrs.update(tctx)
            # HT: w = 1 per window event, pi = w_i / max(w_i, tau)
            b.add(now, ip, attrs, 1.0, 0)
        if not len(b):
            return None
        batch = b.build(lo, now)
        pi = wts_a[kept] / np.maximum(ht, 1e-12)
        batch.pi[:] = np.clip(pi, 1e-12, 1.0).astype(np.float32)
        batch.learn[:] = True
        batch.meta["n_active"] = len(ips)
        return batch

    # ---------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        if not EV.enabled(ctx.config):
            return 0
        store = ctx.store
        now = float(ctx.now)
        pc = EV.pconfig(ctx.config)
        tz = ctx.config.get("tz") or TB.DEFAULT_TZ
        cal = TB.parse_calendar(ctx.config.get("calendar"))
        cap = int(pc["defaults"]["s_sess"])
        default_gap = float(pc["defaults"]["session_gap_s"])
        n = 0
        n_win = 0
        for s in store.batch_systems(EV.EVT_BATCH):
            batch = store.batch_at(s, EV.EVT_BATCH, now)
            if batch is not None and batch.n:
                budget = MP.budget_for(store, MP.tree_key(store, s))
                cols: Dict[str, List[Any]] = {k: [EV.ABSENT] * batch.n for k in (
                    "ctx.tod_min", "ctx.dow", "ctx.daytype", "ctx.dayclass", "ctx.dom",
                    "ctx.mend", "ctx.when", "ctx.sid", "ctx.sess_pos", "ctx.prev_route",
                    "ctx.think_s", "ctx.sess_age_s")}
                memo: Dict[int, Dict[str, Any]] = {}
                for i in range(batch.n):
                    ts = float(batch.ts[i])
                    mkey = int(ts // 60)
                    tc = memo.get(mkey)
                    if tc is None:
                        tc = memo[mkey] = self._time_ctx(float(mkey * 60), tz, cal)
                    for k, v in tc.items():
                        cols[k][i] = v
                self._sessions(s, batch, int(budget.get("s_sess", cap)), default_gap, cols)
                rows = [{k: cols[k][i] for k in cols if cols[k][i] is not EV.ABSENT}
                        for i in range(batch.n)]
                cb = batch.aligned(EV.cols_from_rows(batch.n, rows))
                d_now = TB.local_datetime(now, tz).date()
                day = d_now.toordinal()
                normal_prev = self._calendar_update(store, s, day, day_class(d_now, cal),
                                                    float(batch.w.sum()))
                cb.meta.update({"day": day, "normal_prev_day": normal_prev})
                store.add_batch(s, EV.EVT_CTX, now, cb)
                n += batch.n
            wb = self._window_events(ctx, s, pc, tz, cal)
            if wb is not None:
                store.add_batch(s, EV.EVT_WIN, now, wb)
                n_win += wb.n
        self.last_stats = {"events": n, "window_events": n_win}
        return n + n_win
