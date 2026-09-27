"""B18 ClassMonitorEngine (docs/lib3/engines.md '## B18'): spec unit tests
(a)-(d) plus edge cases. Member rows (feature.nat / active / expo, act.tokens,
behavior.z / pf) are written directly, as B01 / R2 / B04 would; model.class
uses the lib/m_class layout and the adoption ledgers the model.vocab@class
layout B08 writes (lib/m_vocab.adoption_records).

Spec test mapping:
  (a) test_a_coherent_volume_shift_class_low_no_member_alarm
      (class_int p < 1e-6; class B24 -> B25 alarm capped at LOW; the members
      carry no class-level score and get no alarm)
  (b) test_b_risky_adoption_class_novel_alarm
      (4 of 6 members adopt one new external upload SNI, 30 KB / 15 min each:
      the aggregate volume stays p > 0.01 -- the member-level view -- while
      class_novel p < 1e-6 and its accumulator alarm fire within 4 ticks of
      the 3rd adopter; B24 -> B25 grade the class alarm >= MEDIUM, and
      class_adoption_risky is emitted)
  (c) test_c_static_class_own_model_and_profile
  (d) test_d_internal_read_adoption_not_significant
B27 (incident) and B04 (member p) are separate engines; (a)/(b) check their
inputs at the class key: the calibrated class alarm severity from B25.
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional

import numpy as np
import pytest

from helpers import DT, make_store, run_engine

from app.engines.behavior.class_monitor import (AGG, CLASS, MODEL, ClassMonitorEngine,
                                                aggregate)
from app.engines.behavior.lib import emit
from app.engines.behavior.lib import features as F
from app.engines.behavior.lib import gating as G
from app.models.schema import RawMetric

S = "erp"
T = 1_699_999_200.0            # 900-aligned (Wed 2023-11-15 06:00 Asia/Shanghai)
I = F.FEATURE_INDEX
CK = "class:r1"
TOKS = {"GET erp.corp /orders/view/{num}|2xx": 50.0, "GET erp.corp /search|2xx": 30.0,
        "POST erp.corp /orders/save|2xx": 5.0}


# ------------------------------------------------------------------ helpers
def put_classes(store, roles: Dict[str, List[str]], statics: Optional[Dict[str, List[str]]] = None,
                pools: Optional[Dict[str, List[str]]] = None) -> None:
    assign: Dict[str, Dict] = {}

    def slot(ip: str) -> Dict:
        return assign.setdefault(f"{S}|{ip}", {"role": "unique", "sub": None, "prob": 1.0,
                                               "static": [], "pool": None, "super": "machine"})
    for rid, ips in roles.items():
        for ip in ips:
            slot(ip)["role"] = rid
    for name, ips in (statics or {}).items():
        for ip in ips:
            slot(ip)["static"].append(name)
    for cidr, ips in (pools or {}).items():
        for ip in ips:
            slot(ip)["pool"] = cidr
    store.put_model("__org__", "__org__", "model.class", {
        "assign": assign, "version": 1,
        "roles": {rid: {"name": rid, "members": [f"{S}|{ip}" for ip in ips], "version": 1}
                  for rid, ips in roles.items()}})


def nat_row(rng: np.random.Generator, scale: float = 1.0, up_extra: float = 0.0,
            p5xx: float = 0.005) -> np.ndarray:
    """One member's feature.nat row (natural units per 15-min tick)."""
    x = np.full(F.FEATURE_DIM, np.nan)
    for n in ("distinct_peers", "distinct_dports", "distinct_templates", "new_peer_count",
              "ja3_diversity"):
        x[I[n]] = 0.0
    fl, rq = rng.poisson(40 * scale), rng.poisson(100 * scale)
    x[I["flows"]], x[I["http_requests"]] = fl, rq
    x[I["dns_queries"]] = rng.poisson(20 * scale)
    x[I["tls_handshakes"]] = rng.poisson(10 * scale)
    x[I["intensity"]] = rng.poisson(120 * scale)
    up = 2e5 * scale * rng.lognormal(0.0, 0.2) + up_extra
    dn = 1e6 * scale * rng.lognormal(0.0, 0.2)
    x[I["bytes_up"]], x[I["bytes_down"]] = up, dn
    x[I["bytes_per_flow"]] = (up + dn) / max(fl, 1)
    x[I["updown_log"]] = math.log((up + 1.0) / (dn + 1.0))
    for n, p in (("http_write_ratio", 0.1), ("http_get_ratio", 0.85), ("http_4xx_rate", 0.02),
                 ("http_5xx_rate", p5xx), ("http_3xx_rate", 0.03), ("new_template_ratio", 0.01)):
        x[I[n]] = rng.binomial(rq, p) / rq if rq else np.nan
    x[I["syn_ratio"]] = rng.binomial(fl, 0.9) / fl if fl else np.nan
    x[I["http_latency"]] = 50.0 * rng.lognormal(0.0, 0.1)
    return x


def member_tick(store, ip: str, now: float, nat: Optional[np.ndarray], active: bool = True,
                tokens: Optional[Dict[str, float]] = None, z: Optional[np.ndarray] = None,
                dt: float = DT) -> None:
    store.register_entity(S, ip)
    if nat is not None:
        store.add_vec(S, ip, "feature.nat", now, np.asarray(nat, np.float32), window_s=int(dt))
    store.add_vec(S, ip, "feature.active", now, [1.0 if active else 0.0], window_s=int(dt))
    if active and tokens:
        store.add_raw(RawMetric(name="act.tokens", value=dict(tokens), ts=now, system=S, entity=ip))
    if z is not None:
        z = np.asarray(z, np.float64)
        from scipy.special import ndtr
        store.add_vec(S, ip, "behavior.z", now, z.astype(np.float32), window_s=int(dt))
        store.add_vec(S, ip, "behavior.pf", now, (2.0 * ndtr(-np.abs(z))).astype(np.float32),
                      window_s=int(dt))


def tick(store, eng, ips: Iterable[str], k: int, rng, training: bool, scale: float = 1.0,
         z_vol: Optional[float] = None, dt: float = DT, t0: float = T, now: Optional[float] = None,
         **kw) -> float:
    now = t0 + k * dt if now is None else now
    for ip in ips:
        z = rng.standard_normal(F.FEATURE_DIM)
        if z_vol is not None:
            z[F.GROUPS["volume"]] = z_vol
        member_tick(store, ip, now, nat_row(rng, scale, **kw), tokens=TOKS, z=z, dt=dt)
    run_engine(eng, store, now, training=training, dt=dt)
    return now


def warm(store, eng, ips: List[str], n: int = 96, seed: int = 1):
    rng = np.random.default_rng(seed)
    for k in range(n):
        tick(store, eng, ips, k, rng, training=True)
    return rng


def pm(store, key: str, now: float, d: str) -> float:
    return emit.read_row(store, S, key, emit.PM, now).get(d, math.nan)


def put_ledger(store, ck: str, records: Dict[str, Dict]) -> None:
    """model.vocab@(s, class) with B08's adoption ledger layout."""
    m = store.get_model(S, ck, "model.vocab") or {
        "fmt": 1, "kind": "class", "version": 1, "built": T, "H": 30 * 86400.0, "dims": {},
        "N": {}, "N1": {}, "n_ent": 0.0, "members": 1, "adoption": {}}
    m["adoption"].update(records)
    store.put_model(S, ck, "model.vocab", m)


def ledger_rec(dim: str, value: str, members: Dict[str, float], **flags) -> Dict:
    fl = {"external": False, "upload": False, "sensitive": False, "new_eTLD1_org": False}
    fl.update(flags)
    ts = list(members.values())
    return {"dim": dim, "value": value, "members": dict(members), "first_ts": min(ts),
            "last_ts": max(ts), "flags": fl, "adopted": False, "adopted_ts": None,
            "n_class": len(members), "tier": "class"}


def run_downstream(store, now: float, cal, fus) -> None:
    run_engine(cal, store, now)
    run_engine(fus, store, now)


# ================================================================ spec (a)
def test_a_coherent_volume_shift_class_low_no_member_alarm():
    from app.engines.behavior.calibration import CalibrationEngine
    from app.engines.behavior.fusion import FusionEngine
    store, eng = make_store(), ClassMonitorEngine()
    ips = [f"10.0.1.{i}" for i in range(1, 6)]
    put_classes(store, {"r1": ips})
    rng = warm(store, eng, ips)
    cal, fus = CalibrationEngine(), FusionEngine()
    sevs = []
    for k in range(96, 100):
        now = tick(store, eng, ips, k, rng, training=False, scale=2.5, z_vol=6.0)
        assert pm(store, CK, now, "class_int") < 1e-6
        assert pm(store, CK, now, "class_shape") > 1e-3          # same mix
        assert pm(store, CK, now, "class_coherence") < 1e-6
        ax = emit.read_dict(store, S, CK, emit.AXES, now)
        assert ax["class_int"] == ["volume"]
        assert set(ax["class_coherence"]) <= {"peer", "volume"}
        run_downstream(store, now, cal, fus)
        alarm = store.latest_derived(S, CK, "behavior.alarm")
        if alarm is not None and alarm.ts == now:
            sevs.append(alarm.value["severity"])
        for ip in ips:                                           # members: no class-level alarm
            a = store.latest_derived(S, ip, "behavior.alarm")
            assert a is None or a.ts != now
    assert sevs, "the coherent shift must reach fusion as a class alarm"
    assert set(sevs) == {"low"}                                   # intensity-only class: LOW
    ev = store.events(S, CK, kinds=["coherent_shift"])
    assert len(ev) == 1 and ev[0].severity.value == "info" and ev[0].axes == ["volume"]
    assert not store.events(S, CK, kinds=["class_shift", "class_adoption_risky"])


# ================================================================ spec (b)
def test_b_risky_adoption_class_novel_alarm():
    from app.engines.behavior.calibration import CalibrationEngine
    from app.engines.behavior.fusion import FusionEngine
    store, eng = make_store(), ClassMonitorEngine()
    ips = [f"10.0.2.{i}" for i in range(1, 7)]
    put_classes(store, {"r1": ips})
    rng = warm(store, eng, ips)
    cal, fus = CalibrationEngine(), FusionEngine()
    val = "exfil-drop.example.net"
    adopters = ips[:4]
    members: Dict[str, float] = {}
    third = None
    fired_at, sev = None, []
    for j, k in enumerate(range(96, 104)):
        now = T + k * DT
        if j < len(adopters):
            members[adopters[j]] = now
            put_ledger(store, CK, {f"sni={val}": ledger_rec(
                "sni", val, members, external=True, upload=True, new_eTLD1_org=True)})
            if j == 2:
                third = k
        for ip in ips:
            extra = 30_000.0 if ip in members else 0.0
            z = rng.standard_normal(F.FEATURE_DIM)
            member_tick(store, ip, now, nat_row(rng, up_extra=extra), tokens=TOKS, z=z)
        run_engine(eng, store, now)
        p_nov = pm(store, CK, now, "class_novel")
        assert pm(store, CK, now, "class_int") > 0.01           # 30 KB / 15 min is invisible
        acc = emit.read_dict(store, S, CK, emit.ACC_ALARM, now)
        if third is not None and fired_at is None and p_nov < 1e-6 and acc.get("class_novel"):
            fired_at = k
            assert emit.read_dict(store, S, CK, emit.AXES, now)["class_novel"] == ["exfil"]
        run_downstream(store, now, cal, fus)
        a = store.latest_derived(S, CK, "behavior.alarm")
        if a is not None and a.ts == now:
            sev.append(a.value["severity"])
    assert fired_at is not None and fired_at - third <= 4
    assert any(s in ("medium", "high", "critical") for s in sev)
    ev = store.events(S, CK, kinds=["class_adoption_risky"])
    assert len(ev) == 1                                          # once per value
    assert ev[0].extra["value"] == val and ev[0].extra["external"]
    assert ev[0].severity.value in ("low", "medium", "high")


# ================================================================ spec (c)
def test_c_static_class_own_model_and_profile():
    store, eng = make_store(), ClassMonitorEngine()
    dmz = ["10.9.0.1", "10.9.0.2", "10.9.0.3"]
    tiny = ["10.8.0.1", "10.8.0.2"]                             # static class < 3: not monitored
    put_classes(store, {}, statics={"dmz": dmz, "lab": tiny})
    rng = np.random.default_rng(3)
    for k in range(8):
        tick(store, eng, dmz + tiny, k, rng, training=True)
    key = "class:static:dmz"
    m = store.get_model(S, key, MODEL)
    assert m is not None and m["class_kind"] == "static" and m["n_members"] == 3
    assert m["current"].n_commit > 0                            # it learns (gated, D = 4 ticks)
    assert store.get_model(S, "class:static:lab", MODEL) is None
    prof = store.profile(S, key)
    assert prof is not None and "class_monitor" in prof.extra
    cm = prof.extra["class_monitor"]
    assert cm["kind"] == "static" and cm["n_members"] == 3
    assert set(cm["aggregate"]["features"]) >= {"flows", "http_requests", "bytes_up"}
    assert len(cm["active_frac_by_bin"]) == 48
    now = T + 7 * DT
    assert store.vec_at(S, key, AGG, now) is not None
    assert store.latest_derived(S, key, CLASS).value["m"] == 3
    assert math.isfinite(pm(store, key, now, "class_int"))


# ================================================================ spec (d)
def test_d_internal_read_adoption_not_significant():
    store, eng = make_store(), ClassMonitorEngine()
    ips = [f"10.0.4.{i}" for i in range(1, 9)]
    put_classes(store, {"r1": ips})
    rng = warm(store, eng, ips, n=24)
    members: Dict[str, float] = {}
    val = "GET erp.corp /v2/orders"
    ps = []
    for j, k in enumerate(range(24, 32)):
        now = T + k * DT
        if j < 6:
            members[ips[j]] = now
            put_ledger(store, CK, {f"tmpl={val}": ledger_rec("tmpl", val, members)})
        tick(store, eng, ips, k, rng, training=False, now=now)
        ps.append(pm(store, CK, now, "class_novel"))
    assert len(members) == 6
    assert min(ps) > 0.01
    assert not store.events(S, CK, kinds=["class_adoption_risky"])
    # the same adoption of an external upload host is significant (contrast)
    put_ledger(store, CK, {f"sni=x.example.net": ledger_rec(
        "sni", "x.example.net", {ip: T + 31 * DT for ip in ips[:3]}, external=True, upload=True)})
    now = tick(store, eng, ips, 32, rng, training=False)
    assert pm(store, CK, now, "class_novel") < 1e-6


# =============================================================== aggregate
def test_aggregate_sums_pools_and_weights():
    a = np.full(F.FEATURE_DIM, np.nan)
    b = np.full(F.FEATURE_DIM, np.nan)
    a[I["http_requests"]], b[I["http_requests"]] = 100, 300
    a[I["http_5xx_rate"]], b[I["http_5xx_rate"]] = 0.1, 0.0
    a[I["http_latency"]], b[I["http_latency"]] = 10.0, 50.0
    a[I["bytes_up"]], b[I["bytes_up"]] = 1e3, 2e3
    a[I["think_time"]] = 4.0                                    # b stale: NaN
    g = aggregate(np.vstack([a, b]))
    assert g[I["http_requests"]] == 400 and g[I["bytes_up"]] == 3e3
    assert g[I["http_5xx_rate"]] == pytest.approx(10 / 400)    # pooled sum k / sum n
    assert g[I["http_latency"]] == pytest.approx((10 * 100 + 50 * 300) / 400)
    assert g[I["think_time"]] == 4.0
    assert math.isnan(g[I["dns_fail_rate"]])                    # nobody had it: NaN
    assert g[I["flows"]] == 0.0                                 # stale count: a true 0


# ================================================================== edges
def test_empty_store_and_unobserved_class():
    store, eng = make_store(), ClassMonitorEngine()
    assert run_engine(eng, store, T) == 0
    put_classes(store, {"r1": ["10.0.0.1", "10.0.0.2"]})
    store.register_entity(S, "10.0.0.9")                        # system exists, class never seen
    assert run_engine(eng, store, T + DT) == 0
    assert store.get_model(S, CK, MODEL) is None


def test_silent_class_unscored_then_class_silence_alarm():
    store, eng = make_store(), ClassMonitorEngine()
    ips = [f"10.0.5.{i}" for i in range(1, 6)]
    put_classes(store, {"r1": ips})
    rng = warm(store, eng, ips)
    alarms = []
    for k in range(96, 104):
        now = T + k * DT
        for ip in ips:
            member_tick(store, ip, now, nat_row(rng, 0.0), active=False)
        run_engine(eng, store, now)
        row = emit.read_row(store, S, CK, emit.PM, now)
        assert "class_int" not in row and "class_shape" not in row   # silent: not scored (NaN)
        assert "class_coherence" not in row
        assert "class_rhythm" in row
        alarms.append(emit.read_dict(store, S, CK, emit.ACC_ALARM, now).get("class_rhythm"))
        assert emit.read_dict(store, S, CK, emit.DEGRADED, now) == {}
    assert alarms[-1] == 1 and alarms[0] == 0                   # class-wide silence accumulates
    assert store.latest_derived(S, CK, CLASS).value["active_frac"] == 0.0


def test_nan_inputs_never_p_one():
    store, eng = make_store(), ClassMonitorEngine()
    ips = ["10.0.6.1", "10.0.6.2"]
    put_classes(store, {"r1": ips})
    for k in range(3):
        now = T + k * DT
        for ip in ips:
            member_tick(store, ip, now, np.full(F.FEATURE_DIM, np.nan))
        run_engine(eng, store, now)
    row = emit.read_row(store, S, CK, emit.PM, now)
    assert "class_shape" not in row                             # nothing to score: NaN, not 1
    agg = store.vec_at(S, CK, AGG, now)
    assert agg[I["flows"]] == 0.0 and math.isnan(agg[I["http_5xx_rate"]])


def test_b01_failure_degrades_class_detectors():
    store, eng = make_store(), ClassMonitorEngine()
    ips = ["10.0.7.1", "10.0.7.2", "10.0.7.3"]
    put_classes(store, {"r1": ips})
    rng = warm(store, eng, ips, n=6)
    now = T + 6 * DT
    for ip in ips:                                              # B01 raised: no rows at now
        store.register_entity(S, ip)
    store.put_health("behavior.feature_vector", {"last_error_ts": now})
    run_engine(eng, store, now)
    deg = emit.read_dict(store, S, CK, emit.DEGRADED, now)
    assert set(deg) >= {"class_int", "class_shape", "class_rhythm"}
    assert all(v.startswith("producer_error") for v in deg.values() if v)
    row = emit.read_row(store, S, CK, emit.SCORE, now)
    assert "class_int" not in row
    assert store.vec_at(S, CK, AGG, now) is None               # nothing learned from the gap
    assert now not in store.get_model(S, CK, MODEL)["meta"]
    del rng


def test_training_emits_no_events():
    store, eng = make_store(), ClassMonitorEngine()
    ips = [f"10.0.8.{i}" for i in range(1, 6)]
    put_classes(store, {"r1": ips})
    rng = warm(store, eng, ips)
    members = {}
    for j, k in enumerate(range(96, 100)):
        now = T + k * DT
        members[ips[j]] = now
        put_ledger(store, CK, {"sni=evil.example": ledger_rec(
            "sni", "evil.example", members, external=True, upload=True)})
        tick(store, eng, ips, k, rng, training=True, scale=2.5, z_vol=6.0, now=now)
        assert pm(store, CK, now, "class_int") < 1e-6          # still scored
    assert store.events(S) == []


def test_app_error_class_shift_event():
    store, eng = make_store(), ClassMonitorEngine()
    ips = [f"10.0.9.{i}" for i in range(1, 6)]
    put_classes(store, {"r1": ips})
    rng = warm(store, eng, ips)
    for k in range(96, 99):
        now = tick(store, eng, ips, k, rng, training=False, p5xx=0.2)
        assert pm(store, CK, now, "class_shape") < 1e-6
        assert "app_error" in emit.read_dict(store, S, CK, emit.AXES, now)["class_shape"]
    ev = store.events(S, CK, kinds=["class_shift"])
    assert len(ev) == 1 and ev[0].severity.value == "low" and "app_error" in ev[0].axes
    assert not store.events(S, CK, kinds=["coherent_shift"])


def test_cadence_switch_900_to_60():
    store, eng = make_store(), ClassMonitorEngine()
    ips = [f"10.0.10.{i}" for i in range(1, 5)]
    put_classes(store, {"r1": ips})
    rng = warm(store, eng, ips, n=48)
    t_sw = T + 47 * DT
    m = store.get_model(S, CK, MODEL)
    n0, rh0 = m["current"].n_commit, float(m["aux"]["rh"]["v"][48, 0])
    ps = []
    for k in range(1, 46):                                      # 45 one-minute ticks
        now = t_sw + 60.0 * k
        for ip in ips:
            member_tick(store, ip, now, nat_row(rng, 60.0 / 900.0), tokens=TOKS, dt=60.0)
        run_engine(eng, store, now, training=True, dt=60.0)
        ps.append(pm(store, CK, now, "class_int"))
    m = store.get_model(S, CK, MODEL)
    assert m["current"].n_commit > n0 + 30                      # D = 10 ticks at 60 s
    assert m["meta"][t_sw + 60.0][0] == 60.0                    # row exposure = real dt
    assert all(math.isfinite(p) for p in ps)
    assert np.median(ps) > 0.01                                 # per-minute rates: no false shift
    assert float(m["aux"]["rh"]["v"][48, 0]) > rh0              # slots keep closing (every 15 min)


def test_frozen_control_stops_commits_and_rollback_restores():
    store, eng = make_store(), ClassMonitorEngine()
    ips = [f"10.0.11.{i}" for i in range(1, 4)]
    put_classes(store, {"r1": ips})
    rng = warm(store, eng, ips, n=24)
    n0 = store.get_model(S, CK, MODEL)["current"].n_commit
    store.put_model(S, CK, G.CONTROL_MODEL, {"version": 0, "frozen": True})
    for k in range(24, 30):
        tick(store, eng, ips, k, rng, training=True)
    m = store.get_model(S, CK, MODEL)
    assert m["current"].n_commit == n0 and m["gate"].frozen
    # rollback on a fresh class: the anchor is restored to the checkpoint state
    store2, eng2 = make_store(), ClassMonitorEngine()
    put_classes(store2, {"r1": ips})
    rng2 = warm(store2, eng2, ips, n=40)
    before = store2.get_model(S, CK, MODEL)["current"].n_commit
    store2.put_model(S, CK, G.CONTROL_MODEL, {"version": 0, "rollback_to": T + 10 * DT})
    tick(store2, eng2, ips, 40, rng2, training=True)
    m2 = store2.get_model(S, CK, MODEL)
    assert m2["gate"].held and m2["current"].n_commit < before


def test_perf_twelve_classes():
    store, eng = make_store(), ClassMonitorEngine()
    roles = {f"r{i}": [f"10.1.{i}.{j}" for j in range(1, 5)] for i in range(12)}
    put_classes(store, roles)
    ips = [ip for v in roles.values() for ip in v]
    rng = np.random.default_rng(5)
    rows = {ip: nat_row(rng) for ip in ips}
    spent = []
    for k in range(12):
        now = T + k * DT
        for ip in ips:
            member_tick(store, ip, now, rows[ip], tokens=TOKS)
        t0 = time.perf_counter()
        n = run_engine(eng, store, now, training=True)
        spent.append(time.perf_counter() - t0)
        assert n == 12
    assert float(np.median(spent[4:])) < 0.08                  # generous: ~1-2 ms per class


def test_pool_reads_member_role_ledgers_and_quiet_ledger_scores_zero():
    store, eng = make_store(), ClassMonitorEngine()
    ips = [f"10.2.0.{i}" for i in range(1, 7)]
    pool = "class:pool:10.2.0.0/29"
    put_classes(store, {"r1": ips}, pools={"10.2.0.0/29": ips[:3]})
    put_ledger(store, CK, {})                                   # B08 runs; nothing adopted yet
    rng = warm(store, eng, ips, n=12)
    now = T + 11 * DT
    assert pm(store, pool, now, "class_novel") == 1.0          # scored: no adoption, p = 1
    assert pm(store, CK, now, "class_novel") == 1.0
    now = T + 12 * DT
    put_ledger(store, CK, {"sni=c2.example.org": ledger_rec(
        "sni", "c2.example.org", {ips[0]: now, ips[1]: now, ips[4]: now}, external=True)})
    tick(store, eng, ips, 12, rng, training=False, now=now)
    p_pool = pm(store, pool, now, "class_novel")
    assert p_pool < 0.05                                        # 2 of the pool's 3 members
    assert emit.read_dict(store, S, pool, emit.AXES, now)["class_novel"] == ["c2"]
    assert store.latest_derived(S, pool, CLASS).value["new_ext"] == 1
    # no class vocabulary anywhere: class_novel is unscored (NaN), never p = 1
    store2, eng2 = make_store(), ClassMonitorEngine()
    put_classes(store2, {"r1": ips[:2]})
    now = tick(store2, eng2, ips[:2], 0, np.random.default_rng(0), training=True)
    assert "class_novel" not in emit.read_row(store2, S, CK, emit.PM, now)


def test_reference_anchor_delayed_and_excludes_class_incident():
    from app.models.schema import Incident
    store, eng = make_store(), ClassMonitorEngine()
    ips = [f"10.3.0.{i}" for i in range(1, 4)]
    put_classes(store, {"r1": ips})
    store.put_incident(Incident(system=S, entity=CK, entities=list(ips), status="closed",
                                opened=T + 100 * DT, last_seen=T + 102 * DT))
    rng = warm(store, eng, ips, n=100)
    m = store.get_model(S, CK, MODEL)
    assert m["reference"].n_commit == 0 or m["gate_ref"].last_ts <= T + 100 * DT - 86400.0 + 1
    for k in range(100, 140):
        tick(store, eng, ips, k, rng, training=True)
    m = store.get_model(S, CK, MODEL)
    el = m["ref_elig"]
    assert el and el[T + 2 * DT] is True                        # > 24 h before the incident
    assert el[T + 30 * DT] is False                             # within 24 h of it
    assert 0 < m["reference"].n_commit < m["current"].n_commit
