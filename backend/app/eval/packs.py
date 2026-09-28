"""Evaluation packs A-E (+ 'smoke', 'mini'): timelines, population and the
declarative T1-T21 / L1-L16 scenarios of docs/lib3/generator.md.

A Pack is plain data the runner and the generator read:

  name, tz, calendar {holidays, makeup_workdays}  per-run time context
  phases [(n_ticks, dt, aggregated)]              every phase but the last is
                                                  warm-up (training=True)
  start_epoch / scenario_start / end_epoch        virtual clock anchors
  population ['sys|ip']                           base personas (spawns add more)
  scenarios [generator.Scenario]                  the declarative scenarios
  fixtures {twins, nat, hr, ...}                  identification fixtures
  seeds, config, description

get_pack(name) always returns a *fresh* Pack (scenarios carry generator
state such as the demo live-tick activation, so packs are never shared).

Onsets. Scenario-phase tick k starts at scenario_start + k * dt; "day d HH:MM"
is local wall time on the d-th local day of the scenario phase (1-based,
DST-correct). Spec onsets written as a tick for a *human-driven* scenario
(T6, T6b, T7, T9, T9b, T13, L8) start at the first 15-min slot at or after
09:30 of that (or the next) workday in which the driving persona is actually
at work (Timeline.active_onset: activity >= 0.8 over the slot), so a 2-tick
deadline is not spent waiting for the user to arrive; the nominal tick is
kept in the truth row as `nominal_tick`.

Truth. Every scenario carries the generator.md §5 metadata (expected
detectors / axes, max_ttd, required_severity | max_allowed_severity,
perturbed_features) from SPEC below. Background legitimate conditions that
apply to everyone (L4 calendar, L11 idle nights, L13 DST) are recorded with
entities = [] and scope = 'all' so they do not remove every entity from the
control set; L7 (the NAT host) and L15 (sanctioned automation hosts) list
their entities.
"""
from __future__ import annotations

import copy
import datetime as _dt
import math
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Optional, Sequence, Tuple

from ..pipeline.generator import (BASE_KEYS, CHROME127, NAT_KEY, SMOKE_KEYS, SPOOF_T6B, TWINS,
                                  Clock, HumanModel, Scenario, Spawn, Tick, build_human,
                                  build_persona)

AGG_MIN_DT = 900.0
DEFAULT_SEEDS = [0, 1, 2, 3, 4]

ERP, OA, API = "erp-prod", "oa-portal", "api-gateway"
ERP_INTERACTIVE = ["10.20.1.11", "10.20.1.12", "10.20.1.13", "10.20.1.15", "10.20.1.16",
                   "10.20.1.17", "10.20.1.18", "10.20.1.21"]
OA_INTERACTIVE = ["10.30.2.21", "10.30.2.22", "10.30.2.24", "10.30.2.25", "10.30.2.26",
                  "10.30.2.27", "10.30.2.28", "10.30.2.29"]
API_GW_API = ["10.40.4.51", "10.40.4.52", "10.40.4.54", "10.40.4.55", "10.40.4.56",
              "10.40.4.57"]
API_GW_CLIENTS = API_GW_API + ["10.40.4.53", "10.40.9.9"]
AUTOMATION = ["erp-prod|10.20.9.9", "oa-portal|10.30.9.9", "api-gateway|10.40.9.9",
              "erp-prod|10.20.9.5", "api-gateway|10.40.9.6"]


# --------------------------------------------------------------------------- #
# Pack
# --------------------------------------------------------------------------- #
@dataclass
class Pack:
    name: str
    tz: str
    calendar: Dict[str, List[str]]
    phases: List[Tuple[int, float, bool]]
    start_epoch: float
    scenario_start: float
    end_epoch: float
    population: List[str]
    scenarios: List[Scenario] = field(default_factory=list)
    fixtures: Dict[str, Any] = field(default_factory=dict)
    seeds: List[int] = field(default_factory=lambda: list(DEFAULT_SEEDS))
    config: Dict[str, Any] = field(default_factory=dict)
    description: str = ""

    @property
    def n_ticks(self) -> int:
        return int(sum(p[0] for p in self.phases))

    @property
    def scenario_dt(self) -> float:
        return float(self.phases[-1][1])

    def scenario(self, sid: str) -> Scenario:
        for sc in self.scenarios:
            if sc.scenario_id == sid:
                return sc
        raise KeyError(sid)


class Timeline:
    """Local-time anchors of one pack: the scenario phase starts at local
    midnight of `scen_date` (+ `scen_hour`); warm-up phases precede it."""

    def __init__(self, tz: str, calendar: Dict[str, List[str]], scen_date: _dt.date,
                 phases: Sequence[Tuple[int, float, bool]], scen_hour: float = 0.0) -> None:
        self.tz, self.calendar, self.phases = tz, calendar, list(phases)
        self.clock = Clock(tz, calendar)
        self.date = scen_date
        self.scen_start = self.clock.epoch(scen_date, scen_hour)
        warm = sum(n * dt for n, dt, _ in self.phases[:-1])
        self.start_epoch = self.scen_start - warm
        n, dt, _ = self.phases[-1]
        self.dt = float(dt)
        self.n = int(n)
        self.end = self.scen_start + n * dt           # vt after the last tick

    def tick(self, k: float) -> float:
        return self.scen_start + k * self.dt

    def day_date(self, d: int) -> _dt.date:
        return self.date + _dt.timedelta(days=d - 1)

    def day(self, d: int, hh: float = 0.0) -> float:
        """Epoch of local hh:00 on scenario day d (1-based)."""
        return self.clock.epoch(self.day_date(d), hh)

    @property
    def n_days(self) -> int:
        return int(math.ceil((self.end - self.scen_start) / 86400.0 - 1e-9))

    def is_workday(self, d: _dt.date) -> bool:
        return self.clock.day_kind(d)[0]

    def workhours(self, t: float, hh: float = 9.5) -> float:
        """t if it is inside 09:00-17:00 of a workday, else hh on the next workday."""
        ld = self.clock.local(t)
        h = ld.hour + ld.minute / 60.0
        if self.is_workday(ld.date()) and 9.0 <= h < 17.0:
            return t
        d = ld.date() if h < 9.0 else ld.date() + _dt.timedelta(days=1)
        while not self.is_workday(d):
            d += _dt.timedelta(days=1)
        return self.clock.epoch(d, hh)

    def active_onset(self, t: float, who: Sequence[Any], min_act: float = 0.8,
                     grid_s: float = 900.0) -> float:
        """First grid slot at or after workhours(t) in which every persona in
        `who` ('sys|ip' keys or HumanModels) is at work (activity >= min_act
        over the whole slot, lunch excluded). Human-driven scenarios start
        there, so a 2-tick deadline is not spent waiting for the user to
        arrive."""
        models = [w if isinstance(w, HumanModel) else build_persona(w).models[0] for w in who]
        t = self.workhours(t)
        g = min(float(grid_s), self.dt)
        t = self.scen_start + math.ceil((t - self.scen_start) / g - 1e-9) * g
        for _ in range(int(7 * 86400 / g)):
            tk = Tick(t, g, self.clock, 0)
            if all(float(m.activity(tk).min()) >= min_act for m in models):
                return t
            t += g
        raise ValueError("no active onset within a week")

    def days_where(self, pred: Callable[[_dt.date], bool]) -> List[int]:
        return [d for d in range(1, self.n_days + 1) if pred(self.day_date(d))]

    def business_days(self) -> List[int]:
        return self.days_where(self.is_workday)


# --------------------------------------------------------------------------- #
# Truth metadata per scenario (generator.md THREAT SCENARIOS / LEGITIMATE)
# --------------------------------------------------------------------------- #
def _m(dets, axes, sev, ttd, pf, **kw) -> Dict[str, Any]:
    return dict(expected_detectors=list(dets), expected_axes=list(axes), required_severity=sev,
                max_ttd=ttd, perturbed_features=list(pf), **kw)


def _l(sev, pf=(), **kw) -> Dict[str, Any]:
    return dict(expected_detectors=list(kw.pop("dets", [])), expected_axes=list(kw.pop("axes", [])),
                max_allowed_severity=sev, perturbed_features=list(pf), **kw)


SPEC: Dict[str, Dict[str, Any]] = {
    "T1": _m(["B04 marg_int", "B08 novelty", "B16 identity (identity_mismatch)", "B14 bocpd"],
             ["volume", "categorical", "c2"], "high", "2 ticks",
             ["http_requests", "intensity", "new_peer_count", "sni_entropy", "periodicity"],
             title="beacon repurposing", baseline_unchanged=True),
    "T2": _m(["B08 novelty", "B13 budget_exfil", "B14 cusum", "B26 risk"],
             ["exfil", "exfiltration", "categorical"], "medium", "3d",
             ["bytes_up", "updown_log", "bytes_per_flow", "new_peer_count"],
             title="low-and-slow exfiltration", risk_high_by="3d"),
    "T3": _m(["B14 cusum", "B14 creep", "B13 budget_vol"], ["volume", "exfil", "change"],
             "low", "2d", ["bytes_up", "updown_log", "bytes_per_flow"],
             title="frog-boil", severity_by=[["low", "2d"], ["medium", "4d"]], never_accepted=True),
    "T4": _m(["B14 cusum", "B13 budget_vol", "B06 t2"], ["exfil", "volume"], "low", "12 ticks",
             ["bytes_up", "updown_log", "bytes_per_flow"], title="duty-cycled exfiltration"),
    "T4b": _m(["B14 cusum", "B13 budget_vol", "B06 t2"], ["exfil", "volume"], "low", "12 ticks",
              ["bytes_up", "updown_log", "bytes_per_flow"],
              title="duty-cycled exfiltration (alternate ticks)"),
    "T5": _m(["B07 offhours", "B04 marg_int"], ["temporal", "off_hours"], "medium", "4 ticks",
             ["intensity", "http_requests"], title="off-hours activity"),
    "T6": _m(["B09 client_impersonation", "B08 novelty"], ["identity"], "high", "2 ticks",
             ["ja3_diversity", "http_requests", "intensity"], title="concurrent client"),
    "T6b": _m(["B09 client_impersonation", "B09 client"], ["identity"], "high", "2 ticks",
              ["ja3_diversity"], title="spoofed UA"),
    "T7": _m(["B08 rare_access", "B10 seq"], ["categorical", "privilege"], "medium", "2 ticks",
             ["new_template_ratio", "distinct_templates"], title="rare sensitive resource"),
    "T8": _m(["B13 budget_breadth", "B11 timing"], ["breadth", "collection"], "low", "8h",
             ["objs", "path_entropy", "think_time"], title="slow record enumeration",
             per_tick_quiet=True),
    "T9": _m(["B16 identity (identity_mismatch)", "B10 seq (class_llr)"], ["identity"], "high",
             "4 ticks",
             ["path_entropy", "distinct_templates"],
             title="impersonation of a known individual", looks_like="10.30.2.21"),
    "T9b": _m(["B16 unknown_identity"], ["identity"], "medium", "8 ticks",
              ["path_entropy", "distinct_templates", "ja3_diversity"],
              title="unknown individual"),
    "T10": _m(["B12 beacon", "B08 novelty"], ["c2", "timing"], "medium", "16 ticks",
              ["periodicity", "timing_regularity", "new_peer_count"],
              title="slow jittered additive beacon"),
    "T11": _m(["B02 new_entity_unmatched", "B04 marg_int", "B08 novelty", "B13 budget_exfil"],
              ["exfil", "discovery", "c2", "categorical"], "medium", "8 ticks",
              ["syn_ratio", "distinct_peers", "dns_txt_ratio", "dns_qname_len", "bytes_up"],
              title="cold-start attackers", risk_high=True, quarantined_after_ticks=50),
    "T12": _m(["B04 marg_shape", "B10 dwell"], ["credential"], "medium", "3 ticks",
              ["http_4xx_rate", "http_write_ratio", "http_requests"],
              title="low-rate credential stuffing"),
    "T13": _m(["B08 novelty", "B06 spe", "B04 marg_shape"], ["privilege", "categorical", "shape"],
              "medium", "2 ticks", ["distinct_templates", "new_template_ratio", "path_entropy"],
              title="privilege misuse at unchanged volume", never_accepted=True),
    "T14": _m(["B08 novelty", "B13 budget_exfil", "B04 marg_shape"],
              ["dns", "exfil", "categorical", "c2"], "medium", "4 ticks",
              ["dns_txt_ratio", "dns_qname_len", "dns_name_entropy", "dns_queries"],
              title="DNS tunnel on an existing host"),
    "T15": _m(["B14 cusum", "B14 creep", "B08 novelty", "B28 regime"],
              ["exfil", "volume", "change"], "high", None,
              ["bytes_up", "updown_log", "new_peer_count"],
              title="baseline-poisoning ramp then strike", never_accepted=True),
    "T16": _m(["B25 evidence", "B26 risk"], ["identity", "categorical", "temporal", "volume"],
              "medium", "6h", ["ja3_diversity", "bytes_up", "new_template_ratio", "intensity"],
              title="several weak signals", risk_high_by="6h"),
    "T17": _m(["B02 class_transition", "B16 identity_mismatch", "B14 bocpd"],
              ["identity", "shape"], "medium", "12h",
              ["http_write_ratio", "http_get_ratio", "distinct_templates"],
              title="quiet role repurposing"),
    "T18": _m(["B06 spe"], ["shape", "exfil"], "low", "6 ticks",
              ["bytes_up", "updown_log", "bytes_per_flow", "req_bytes_avg"],
              title="correlation-break exfiltration"),
    "T19": _m(["B17 entity_resolution", "B13 budget_exfil", "B08 novelty"],
              ["exfil", "identity"], "low", "60 ticks", ["bytes_up", "new_peer_count"],
              title="IP-hopping actor"),
    "T20": _m(["B21 cross_system (first_access_system)"], ["lateral", "xsys"], "low", "1d",
              ["distinct_peers", "cross_system"], title="lateral access", priority="P2"),
    "T21": _m(["B18 class_novel", "B18 class_shape"], ["exfil", "categorical"], "medium", None,
              ["bytes_up", "new_peer_count"], title="compromised pool", level="class"),
    "L1": _l("low", ["intensity", "http_requests", "bytes_up"], level="class",
             title="month-end volume", max_class_incidents=1, max_member_severity="info"),
    "L2": _l("info", ["ja3_diversity"], level="class", title="client rollout"),
    "L3": _l("info", ["new_template_ratio", "distinct_templates"], level="class",
             title="new resource"),
    "L4": _l("info", [], title="calendar (weekends, holiday, make-up Saturday)", scope="all",
             background=True),
    "L5": _l("low", ["intensity", "http_requests", "bytes_up"], title="slow growth",
             accept_within_s=3 * 86400.0),
    "L6": _l("info", ["entity"], title="DHCP renumbering"),
    "L7": _l("info", [], title="NAT with two personas", background=True),
    "L8": _l("info", ["entity"], title="new employee"),
    "L9": _l("low", ["intensity"], title="backup schedule shift",
             accept_within_s=4 * 86400.0),
    "L10": _l("low", ["http_5xx_rate", "rtt", "retransmit_rate"], level="system",
              title="api-gateway outage and WAN degradation", max_class_incidents=2,
              max_member_severity="info"),
    "L11": _l("info", [], title="idle and near-idle nights", scope="all", background=True),
    "L12": _l("info", ["intensity"], title="absence and return"),
    "L13": _l("info", [], title="DST switch", scope="all", background=True),
    "L14": _l("info", ["ja3_diversity"], title="browser auto-update"),
    "L15": _l("info", [], title="sanctioned periodic automation", background=True),
    "L16": _l("info", ["new_template_ratio", "distinct_templates"], title="explorative user"),
}


def base_id(sid: str) -> str:
    s = str(sid).rstrip("'")
    return s.split("-")[0].split("_")[0]


def _label(sid: str) -> str:
    b = base_id(sid)
    if b.startswith("T"):
        return "malicious"
    return "system_change" if b == "L10" else "legit_change"


def _sc(sid: str, system: str, entities: List[str], kind: str, t0: float, t1: float,
        mode: str = "additive", params: Optional[Dict[str, Any]] = None,
        spawns: Optional[List[Spawn]] = None, **truth: Any) -> Scenario:
    meta = copy.deepcopy(SPEC[base_id(sid)])
    meta.update(truth)
    return Scenario(scenario_id=sid, label=_label(sid), system=system, entities=list(entities),
                    kind=kind, t_start=float(t0), t_end=float(t1), mode=mode,
                    params=dict(params or {}), spawns=list(spawns or []), truth=meta)


# --------------------------------------------------------------------------- #
# Scenario builders (one per spec entry; onsets supplied by the pack)
# --------------------------------------------------------------------------- #
def t1(tl: Timeline, t0: float, t1_: Optional[float] = None, sid: str = "T1") -> Scenario:
    return _sc(sid, API, ["10.40.4.52"], "beacon_replace", t0, t1_ or tl.end, mode="replace",
               params={"host": "c2.example.net", "path": "/ping", "period_s": 300.0,
                       "jitter_s": 3.0})


def t2(tl: Timeline, t0: float) -> Scenario:
    return _sc("T2", ERP, ["10.20.1.12"], "exfil_business", t0, t0 + 5 * 86400.0,
               params={"host": "ext-store.example.net", "bytes": 200_000.0, "growth": 1.6,
                       "every_s": 900.0})


def t3(tl: Timeline, t0: float) -> Scenario:
    return _sc("T3", ERP, ["10.20.4.30"], "bytes_ramp", t0, t0 + 10 * 86400.0,
               params={"growth": 1.15})


def t4(tl: Timeline, t0: float, alt: bool = False) -> Scenario:
    if alt:
        return _sc("T4b", ERP, ["10.20.4.33"], "duty_upload", t0, t0 + 2 * 86400.0,
                   params={"pattern": "alternate", "factor": 20.0, "tick_s": 900.0})
    return _sc("T4", OA, ["10.30.4.41"], "duty_upload", t0, t0 + 2 * 86400.0,
               params={"pattern": "random", "p": 0.3, "factor": 20.0, "tick_s": 900.0})


def t5(tl: Timeline, days: Sequence[int], sid: str = "T5",
       sunday: bool = True) -> Scenario:
    wins = [(tl.day(d, 2.0), tl.day(d, 4.0)) for d in days]
    if sunday:   # Sunday 14:00-16:00 slots inside the scenario phase (if any)
        for d in tl.days_where(lambda x: x.weekday() == 6 and not tl.is_workday(x)):
            wins.append((tl.day(d, 14.0), tl.day(d, 16.0)))
    wins.sort()
    return _sc(sid, OA, ["10.30.2.22"], "offhours", wins[0][0], wins[-1][1],
               params={"windows": wins}, windows=[list(w) for w in wins])


def t6(tl: Timeline, t0: float, nominal: Optional[int] = None) -> Scenario:
    return _sc("T6", ERP, ["10.20.1.11"], "second_client", t0, t0 + 86400.0,
               nominal_tick=nominal)


def t6b(tl: Timeline, t0: float, nominal: Optional[int] = None) -> Scenario:
    return _sc("T6b", ERP, ["10.20.1.18"], "stack_change", t0, t0 + 86400.0, mode="replace",
               params={"stack": SPOOF_T6B}, nominal_tick=nominal)


def t7(tl: Timeline, t0: float, nominal: Optional[int] = None) -> Scenario:
    return _sc("T7", ERP, ["10.20.1.13"], "rare_resource", t0, t0 + 86400.0,
               params={"path": "/hr/salary/export", "per_tick": 1.0}, nominal_tick=nominal)


def t8(tl: Timeline, day: int) -> Scenario:
    return _sc("T8", ERP, ["10.20.1.15"], "enumeration", tl.day(day, 9.0), tl.day(day, 17.0),
               mode="replace", params={"start_id": 500_000})


def t9(tl: Timeline, t0: float, nominal: Optional[int] = None) -> Scenario:
    return _sc("T9", OA, ["10.30.2.26"], "impersonate", t0, t0 + 86400.0, mode="replace",
               params={"looks_like_key": "oa-portal|10.30.2.21"}, nominal_tick=nominal)


def t9b(tl: Timeline, t0: float, nominal: Optional[int] = None) -> Scenario:
    return _sc("T9b", OA, ["10.30.2.29"], "unknown_persona", t0, t0 + 86400.0, mode="replace",
               params={"tag": "T9b-intruder"}, nominal_tick=nominal)


def t10(tl: Timeline, t0: float) -> Scenario:
    return _sc("T10", API, ["10.40.4.51"], "jitter_beacon", t0, min(tl.end, t0 + 3 * 86400.0),
               params={"host": "cdn-upd.example.net", "period_s": 300.0, "jitter": 0.3,
                       "bytes": 1200.0})


def t11(tl: Timeline, t0: float) -> Scenario:
    keys = ["oa-portal|10.30.2.99", "erp-prod|10.20.7.77", "erp-prod|10.20.1.66"]
    return _sc("T11", ERP, keys, "cold_start", t0, tl.end,
               params={"exfil_key": "erp-prod|10.20.1.66", "bytes_per_tick": 2e6,
                       "uploads_per_tick": 4, "host": "ext-drop.example.com"},
               spawns=[Spawn(keys[0], "scanner", t0, archetype="scanner"),
                       Spawn(keys[1], "dns_tunnel", t0, archetype="dns_tunnel"),
                       Spawn(keys[2], "prior", t0, tag="T11-exfil-actor")],
               match="all")


def t12(tl: Timeline, t0: float, t1_: float, sid: str = "T12") -> Scenario:
    return _sc(sid, OA, ["10.30.2.25"], "cred_stuffing", t0, t1_,
               params={"per_tick": 6.0, "fail": 0.9})


def t13(tl: Timeline, t0: float, nominal: Optional[int] = None) -> Scenario:
    return _sc("T13", ERP, ["10.20.1.16"], "path_shift", t0, t0 + 86400.0,
               params={"share": 0.6, "templates": [
                   ("GET", "erp.corp.local", "/admin/users/{id}", "private"),
                   ("GET", "erp.corp.local", "/admin/export", "private")]},
               nominal_tick=nominal)


def t14(tl: Timeline, t0: float) -> Scenario:
    return _sc("T14", ERP, ["10.20.1.13"], "dns_tunnel", t0, t0 + 2 * 86400.0,
               params={"zone": "tun.example.org", "per_tick": 2.0, "lmin": 30, "lmax": 48})


def t15(tl: Timeline, t0: float, ramp: int = 200, strike_ticks: int = 4) -> Scenario:
    tick_s = 900.0
    strike = t0 + ramp * tick_s
    return _sc("T15", ERP, ["10.20.9.5"], "poison_ramp", t0, strike + strike_ticks * tick_s,
               params={"bytes": 100_000.0, "rate": 1.03, "ramp_ticks": ramp, "strike": 30.0,
                       "external_at": 50, "ext_host": "bk-mirror.example.net",
                       "tick_s": tick_s},
               strike_ts=strike, max_ttd_s=(strike - t0) + tick_s)


def t16(tl: Timeline, t0: float) -> Scenario:
    return _sc("T16", OA, ["10.30.2.24"], "weak_signals", t0, t0 + 86400.0,
               params={"t_off": t0, "off_len": 900.0, "t_ja3": t0 + 2.5 * 3600.0,
                       "t_bytes": t0 + 3 * 3600.0, "bytes_mult": 2.0,
                       "t_rare": t0 + 4 * 3600.0})


def t17(tl: Timeline, t0: float) -> Scenario:
    return _sc("T17", API, ["10.40.4.53"], "path_shift", t0, tl.end,
               params={"share": 1.0, "templates": [
                   ("GET", "api.corp.local", "/v1/resource", "list")]})


def t18(tl: Timeline, t0: float, ticks: int = 48) -> Scenario:
    return _sc("T18", ERP, ["10.20.1.16"], "bytes_mult", t0, t0 + ticks * 900.0,
               params={"range": (3.0, 4.0)})


def t19(tl: Timeline, t0: float, hop_ticks: int = 30) -> Scenario:
    ips = ["10.20.1.70", "10.20.1.71", "10.20.1.72"]
    step = hop_ticks * 900.0
    spawns = [Spawn(f"{ERP}|{ip}", "prior", t0 + i * step, t0 + (i + 1) * step,
                    tag="T19-hopping-actor") for i, ip in enumerate(ips)]
    return _sc("T19", ERP, ips, "ip_hop", t0, t0 + len(ips) * step,
               params={"host": "ext-sync.example.io", "bytes_per_tick": 1e6}, spawns=spawns,
               chain=list(ips))


def t20(tl: Timeline, t0: float) -> Scenario:
    return _sc("T20", OA, ["10.30.2.26", f"{API}|10.30.2.26", f"{ERP}|10.30.2.26"], "lateral",
               t0, t0 + 3 * 86400.0,
               params={"per_tick": 3.0, "targets": [
                   (API, "GET", "api.corp.local", "/v1/resource"),
                   (ERP, "GET", "erp.corp.local", "/admin")]},
               match="any")


def t21(tl: Timeline, t0: float) -> Scenario:
    members = [f"{API}|{ip}" for ip in ("10.40.4.54", "10.40.4.55", "10.40.4.56", "10.40.4.57")]
    onsets = {k: t0 + i * 2400.0 for i, k in enumerate(members)}      # over 2 h
    third = onsets[members[2]]
    return _sc("T21", API, [k.partition("|")[2] for k in members], "pool_upload", t0,
               t0 + 3 * 86400.0,
               params={"onsets": onsets, "every_s": 900.0, "bytes": 30_000.0,
                       "host": "telemetry-sync.example.org"},
               class_members=list(API_GW_API), onsets={k: v for k, v in onsets.items()},
               max_ttd_s=(third - t0) + 4 * tl.dt)


def l1(tl: Timeline) -> Scenario:
    """x2.5 on the last 2 business days of the (first) month end in the phase."""
    days = tl.business_days()
    month_last: Dict[Tuple[int, int], List[int]] = {}
    for d in days:
        dd = tl.day_date(d)
        month_last.setdefault((dd.year, dd.month), []).append(d)
    pick: List[int] = []
    for (y, mth), ds in sorted(month_last.items()):
        nxt = _dt.date(y + (mth == 12), mth % 12 + 1, 1)
        if tl.day_date(1) <= nxt - _dt.timedelta(days=1) < tl.day_date(tl.n_days + 1):
            # last 2 business days of the month (holiday / make-up aware)
            d = nxt - _dt.timedelta(days=1)
            got = []
            while len(got) < 2:
                if tl.is_workday(d):
                    got.append(d)
                d -= _dt.timedelta(days=1)
            pick = sorted((x - tl.date).days + 1 for x in got)
            break
    wins = [(tl.day(d, 0.0), tl.day(d + 1, 0.0)) for d in pick]
    return _sc("L1", ERP, list(ERP_INTERACTIVE), "volume", wins[0][0], wins[-1][1],
               params={"factor": 2.5, "windows": wins}, class_members=list(ERP_INTERACTIVE),
               windows=[list(w) for w in wins])


def l2(tl: Timeline, day: int) -> Scenario:
    keys = [f"{OA}|{ip}" for ip in OA_INTERACTIVE]
    on = {k: tl.day(day, 9.0) + i * 3600.0 for i, k in enumerate(keys)}
    return _sc("L2", OA, list(OA_INTERACTIVE), "stack_change", tl.day(day, 9.0), tl.end,
               params={"stack": CHROME127, "only_standard": True, "onsets": on},
               class_members=list(OA_INTERACTIVE), onsets=on)


def l3(tl: Timeline, day: int) -> Scenario:
    keys = [f"{ERP}|{ip}" for ip in ERP_INTERACTIVE]
    on = {k: tl.day(day, 9.0) + i * 3000.0 for i, k in enumerate(keys)}
    return _sc("L3", ERP, list(ERP_INTERACTIVE), "new_resource", tl.day(day, 9.0), tl.end,
               params={"path": "/v2/orders", "share": 0.1, "onsets": on},
               class_members=list(ERP_INTERACTIVE), onsets=on)


def l4(tl: Timeline) -> Scenario:
    special = tl.days_where(lambda x: not tl.is_workday(x) or tl.clock.day_kind(x)[1])
    days = [tl.day_date(d).isoformat() for d in special]
    t0 = tl.day(special[0]) if special else tl.scen_start
    t1_ = tl.day(special[-1] + 1) if special else tl.end
    return _sc("L4", ERP, [], "background", t0, t1_, days=days,
               holidays=list(tl.calendar.get("holidays", [])),
               makeup_workdays=list(tl.calendar.get("makeup_workdays", [])))


def l5(tl: Timeline) -> Scenario:
    return _sc("L5", OA, ["10.30.4.40"], "volume_ramp", tl.scen_start, tl.end,
               params={"growth": 1.02})


def l6(tl: Timeline, t0: float) -> Scenario:
    return _sc("L6", ERP, ["10.20.1.12", "10.20.1.112", "10.20.1.114"], "renumber", t0, tl.end,
               mode="replace",
               params={"from_key": f"{ERP}|10.20.1.12", "to_key": f"{ERP}|10.20.1.112"},
               spawns=[Spawn(f"{ERP}|10.20.1.112", "alias", t0, alias_of=f"{ERP}|10.20.1.12"),
                       Spawn(f"{ERP}|10.20.1.114", "prior", t0, tag="L6-negative-control")],
               link_pairs=[["10.20.1.12", "10.20.1.112"]], negative_control=["10.20.1.114"],
               aliases=[["10.20.1.12", "10.20.1.112"]])


def l7(tl: Timeline) -> Scenario:
    return _sc("L7", OA, [NAT_KEY.partition("|")[2]], "background", tl.scen_start, tl.end,
               n_personas=2)


L8_TAG = "L8-new-employee"


def l8(tl: Timeline, t0: float, nominal: Optional[int] = None) -> Scenario:
    return _sc("L8", ERP, ["10.20.1.14"], "background", t0, tl.end,
               spawns=[Spawn(f"{ERP}|10.20.1.14", "prior", t0, tag=L8_TAG)],
               nominal_tick=nominal)


def l9(tl: Timeline, day: int) -> Scenario:
    return _sc("L9", ERP, ["10.20.9.5"], "schedule_shift", tl.day(day, 0.0), tl.end,
               params={"hour": 3.0}, old_hour=1.0, new_hour=3.0)


def l10(tl: Timeline, outage_day: int, degrade_day: int) -> Scenario:
    a = tl.day(outage_day, 11.0)
    c = tl.day(degrade_day, 11.0)
    return _sc("L10", API, list(API_GW_CLIENTS), "outage", a, c + 12 * 900.0,
               params={"outage": (a, a + 3 * 900.0), "degrade": (c, c + 12 * 900.0),
                       "rtt_mult": 3.0, "retrans_mult": 5.0},
               outage=[a, a + 3 * 900.0], degrade=[c, c + 12 * 900.0])


def l11(tl: Timeline) -> Scenario:
    return _sc("L11", ERP, [], "background", tl.scen_start, tl.end)


def l12(tl: Timeline, d0: int, d1: int) -> Scenario:
    return _sc("L12", OA, ["10.30.2.21"], "absence", tl.day(d0), tl.day(d1 + 1), mode="replace")


def l13(tl: Timeline, day: int) -> Scenario:
    return _sc("L13", ERP, [], "background", tl.day(day), tl.day(day + 1), tz=tl.tz)


def l14(tl: Timeline, t0: float, key_ip: str = "10.30.2.24") -> Scenario:
    system = OA if key_ip.startswith("10.30.") else ERP
    return _sc("L14", system, [key_ip], "stack_change", t0, tl.end, mode="replace",
               params={})


def l15(tl: Timeline, keys: Sequence[str]) -> Scenario:
    return _sc("L15", ERP, list(keys), "background", tl.scen_start, tl.end)


def l16(tl: Timeline) -> Scenario:
    return _sc("L16", OA, ["10.30.2.29"], "explorer", tl.scen_start, tl.end,
               params={"per_day": 5})


# --------------------------------------------------------------------------- #
# Packs
# --------------------------------------------------------------------------- #
WARMUP_528 = [(336, 3600.0, True), (192, 900.0, True)]      # v2 (kept for reference)
SHANGHAI, BERLIN = "Asia/Shanghai", "Europe/Berlin"
WARMUP_DAYS = 16


def warmup_phases(scen_date: _dt.date, calendar: Dict[str, List[str]], tz: str,
                  live_dt: float, total_days: int = WARMUP_DAYS
                  ) -> List[Tuple[int, float, bool]]:
    """Warm-up phases before a scenario phase at `live_dt` (spec v2.1,
    docs/lib3/cadence.md §10; a pack-definition rule, not seed tuning).

    live_dt <= 900: [((total - k) x 24, 3600), (k x 86400 / live_dt, live_dt)]
    where k >= 2 is the smallest number of local days before the scenario
    start whose span holds >= 1 full workday and >= 1 full non-workday under
    the pack's calendar (holidays and make-up workdays included), so the
    last warm-up phase runs at the live cadence over both day types (the Q
    grain and the cadence-class B24 rings are native before the scenario).
    live_dt = 3600: total_days at 3600."""
    live_dt = float(live_dt)
    if live_dt >= 3600.0:
        return _phases((int(total_days) * 24, 3600.0))
    clock = Clock(tz, calendar)
    kinds = []
    k = 0
    while True:
        k += 1
        kinds.append(clock.day_kind(scen_date - _dt.timedelta(days=k))[0])
        if k >= 2 and any(kinds) and not all(kinds):
            break
        if k >= total_days - 1:
            raise ValueError("warmup_phases: no workday / non-workday pair before the start")
    return _phases(((int(total_days) - k) * 24, 3600.0),
                   (int(round(k * 86400.0 / live_dt)), live_dt))


def _fixtures(population: Sequence[str]) -> Dict[str, Any]:
    pop = set(population)
    fx: Dict[str, Any] = {}
    tw = [list(k.split("|", 1)) for k in TWINS if k in pop]
    if tw:
        fx["twins"] = tw
    if NAT_KEY in pop:
        fx["nat"] = [NAT_KEY.split("|", 1)]
    if "erp-prod|10.20.1.21" in pop:
        fx["hr"] = [["erp-prod", "10.20.1.21"]]
    return fx


def _automation(pop: Sequence[str], used: Sequence[Scenario]) -> List[str]:
    taken = set()
    for sc in used:
        taken |= set(sc.keys())
    return [k for k in AUTOMATION if k in pop and k not in taken]


def _pack(name: str, tl: Timeline, population: Sequence[str], scenarios: List[Scenario],
          description: str, backgrounds: bool = True) -> Pack:
    pop = list(population)
    scs = list(scenarios)
    if backgrounds:
        if NAT_KEY in pop:
            scs.append(l7(tl))
        scs.append(l11(tl))
        auto = _automation(pop, scs)        # 'sys|ip' keys: L15 spans the three systems
        if auto:
            scs.append(l15(tl, auto))
    return Pack(name=name, tz=tl.tz, calendar=copy.deepcopy(tl.calendar),
                phases=[(int(n), float(dt), bool(agg)) for n, dt, agg in tl.phases],
                start_epoch=tl.start_epoch, scenario_start=tl.scen_start, end_epoch=tl.end,
                population=pop, scenarios=scs, fixtures=_fixtures(pop),
                config={"tz": tl.tz, "calendar": copy.deepcopy(tl.calendar)},
                description=description)


def _phases(*spec: Tuple[int, float]) -> List[Tuple[int, float, bool]]:
    return [(n, dt, dt >= AGG_MIN_DT) for n, dt in spec]


def pack_a() -> Pack:
    """Short / identity. Scenario phase Mon 2025-03-10 .. Thu (4 d at 900 s)."""
    cal = {"holidays": [], "makeup_workdays": []}
    d0 = _dt.date(2025, 3, 10)
    tl = Timeline(SHANGHAI, cal, d0, warmup_phases(d0, cal, SHANGHAI, 900.0)
                  + _phases((384, 900.0)))
    def on(k: int, *who: Any) -> float:
        return tl.active_onset(tl.tick(k), who)

    scs = [
        t1(tl, tl.tick(40)),
        t5(tl, [2, 3, 4]),
        t6(tl, on(20, f"{ERP}|10.20.1.11"), 20),
        t6b(tl, on(20, f"{ERP}|10.20.1.18"), 20),
        t7(tl, on(30, f"{ERP}|10.20.1.13"), 30),
        t9(tl, on(30, f"{OA}|10.30.2.21"), 30),          # the impersonated user is at work
        t9b(tl, on(30, f"{OA}|10.30.2.29"), 30),
        t11(tl, tl.tick(50)),
        t12(tl, tl.tick(60), tl.tick(60 + 32)),
        t13(tl, on(80, f"{ERP}|10.20.1.16"), 80),
        t16(tl, tl.tick(120)),
        t17(tl, tl.tick(140)),
        l6(tl, tl.tick(100)),
        l8(tl, on(70, build_human(f"{ERP}|10.20.1.14", tag=L8_TAG)), 70),
    ]
    return _pack("A", tl, BASE_KEYS, scs, "short / identity (Asia/Shanghai)")


def pack_b() -> Pack:
    """Long horizon. 12 d at 900 s from Thu 2025-06-12: day 3 is a 调休
    make-up Saturday, day 4 a Sunday, day 5 a holiday, days 10-11 a weekend."""
    cal = {"holidays": ["2025-06-16"], "makeup_workdays": ["2025-06-14"]}
    d0 = _dt.date(2025, 6, 12)
    tl = Timeline(SHANGHAI, cal, d0, warmup_phases(d0, cal, SHANGHAI, 900.0)
                  + _phases((1152, 900.0)))
    scs = [
        t2(tl, tl.day(2, 9.0)),
        t3(tl, tl.day(2, 0.0)),
        t4(tl, tl.day(5, 0.0)),
        t4(tl, tl.day(6, 0.0), alt=True),
        t8(tl, 3),
        t10(tl, tl.day(4, 0.0)),
        t14(tl, tl.day(3, 10.0)),
        t15(tl, tl.day(4, 1.0)),
        t18(tl, tl.day(7, 9.0)),
        t19(tl, tl.day(9, 9.0)),
        l4(tl),
    ]
    return _pack("B", tl, BASE_KEYS, scs, "long horizon (Asia/Shanghai, holiday + 调休)")


def pack_c() -> Pack:
    """Class-wide legitimate changes and DST. Europe/Berlin, 7 d at 900 s
    from Fri 2025-03-28; the DST switch (Sun 03-30) is scenario day 3."""
    cal = {"holidays": [], "makeup_workdays": []}
    d0 = _dt.date(2025, 3, 28)
    tl = Timeline(BERLIN, cal, d0, warmup_phases(d0, cal, BERLIN, 900.0) + _phases((672, 900.0)))
    scs = [l2(tl, 4), l3(tl, 5), l10(tl, 2, 3), l13(tl, 3)]
    return _pack("C", tl, BASE_KEYS, scs, "class-wide legitimate changes + DST (Europe/Berlin)")


def pack_d() -> Pack:
    """Slow legitimate changes, month end and class threat. 30 d at 3600 s
    from Thu 2025-09-04: 调休 Sunday 09-28, month end 09-29/30, National Day
    holiday from 10-01."""
    cal = {"holidays": [f"2025-10-0{d}" for d in range(1, 9)],
           "makeup_workdays": ["2025-09-28", "2025-10-11"]}
    d0 = _dt.date(2025, 9, 4)
    tl = Timeline(SHANGHAI, cal, d0, warmup_phases(d0, cal, SHANGHAI, 3600.0)
                  + _phases((720, 3600.0)))
    scs = [
        t20(tl, tl.day(12, 9.0)),
        t21(tl, tl.day(20, 10.0)),
        l1(tl),
        l4(tl),
        l5(tl),
        l9(tl, 8),
        l12(tl, 5, 12),
        l14(tl, tl.day(6, 10.0)),
        l16(tl),
    ]
    return _pack("D", tl, BASE_KEYS, scs, "slow legitimate changes, month end, class threat")


def pack_e() -> Pack:
    """Cadence switch: 7 d warm-up at 900 s, then 1 live day (Tue
    2025-03-11) at 60 s, non-aggregated, with T1', T5', T12' in wall time."""
    cal = {"holidays": [], "makeup_workdays": []}
    tl = Timeline(SHANGHAI, cal, _dt.date(2025, 3, 11), _phases((672, 900.0), (1440, 60.0)))
    wall = {"max_ttd": "30min"}
    scs = [
        _retime(t1(tl, tl.day(1, 10.0), sid="T1'"), **wall),
        _retime(t5(tl, [1], sid="T5'", sunday=False), max_ttd="1h"),
        _retime(t12(tl, tl.day(1, 15.0), tl.day(1, 21.0), sid="T12'"), max_ttd="45min"),
    ]
    return _pack("E", tl, BASE_KEYS, scs, "cadence switch 900 s -> 60 s (replays)")


def _retime(sc: Scenario, **truth: Any) -> Scenario:
    sc.truth.update(truth)
    sc.truth["replay_of"] = base_id(sc.scenario_id)
    return sc


def pack_smoke() -> Pack:
    """Smoke persona set (20) for the CPU gate: 120 warm-up ticks (spec v2.1:
    96 x 3600 s Wed 18:00 .. Sun 18:00 + 24 x 900 s Sun evening, so both day
    types are seen and the Q grain starts on the transfer path) + 60 x 900 s."""
    cal = {"holidays": [], "makeup_workdays": []}
    tl = Timeline(SHANGHAI, cal, _dt.date(2025, 3, 10),
                  _phases((96, 3600.0), (24, 900.0), (60, 900.0)))
    scs = [t1(tl, tl.tick(40)), t12(tl, tl.tick(44), tl.tick(60))]
    scs[1].entities = ["10.30.2.22"]
    return _pack("smoke", tl, SMOKE_KEYS, scs, "smoke (20 personas, 180 x 900 s)")


MINI_KEYS = ["erp-prod|10.20.1.11", "erp-prod|10.20.1.12", "erp-prod|10.20.1.13",
             "erp-prod|10.20.4.30", "erp-prod|10.20.9.9", "api-gateway|10.40.4.51",
             "api-gateway|10.40.4.52", "api-gateway|10.40.4.53"]


def pack_mini() -> Pack:
    """Fast CI pack: 8 entities, 72 x 3600 s warm-up + 128 x 900 s scenario
    phase (Mon 00:00 .. Tue 08:00) with T1, T6, T17 and L14."""
    cal = {"holidays": [], "makeup_workdays": []}
    tl = Timeline(SHANGHAI, cal, _dt.date(2025, 3, 10), _phases((72, 3600.0), (128, 900.0)))
    scs = [
        t1(tl, tl.tick(40)),
        t6(tl, tl.tick(44)),
        t17(tl, tl.tick(56)),
        l14(tl, tl.tick(54), "10.20.1.13"),
    ]
    scs[1].t_end = tl.end
    return _pack("mini", tl, MINI_KEYS, scs, "mini CI pack (8 entities, 200 ticks)")


PACKS: Dict[str, Callable[[], Pack]] = {"A": pack_a, "B": pack_b, "C": pack_c, "D": pack_d,
                                        "E": pack_e, "smoke": pack_smoke}
EXTRA_PACKS: Dict[str, Callable[[], Pack]] = {"mini": pack_mini}


def pack_names(include_extra: bool = False) -> List[str]:
    return list(PACKS) + (list(EXTRA_PACKS) if include_extra else [])


def _norm(name: str) -> str:
    n = str(name).strip()
    u = n.upper().replace("-", "_").replace(" ", "_")
    if u.startswith("PACK_"):
        u = u[5:]
    elif u.startswith("PACK") and len(u) == 5:
        u = u[4:]
    return u


def get_pack(name: Any) -> Pack:
    """A fresh Pack by name: 'A'..'E' (also 'pack_a', 'Pack A'), 'smoke', 'mini'."""
    if isinstance(name, Pack):
        return name
    n = _norm(str(name))
    for table in (PACKS, EXTRA_PACKS):
        for k, fn in table.items():
            if k.upper() == n:
                return fn()
    raise KeyError(f"unknown pack {name!r}; known: {pack_names(True)}")
