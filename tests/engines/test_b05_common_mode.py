"""B05 CommonModeEngine (docs/lib3/engines.md '## B05'): spec unit tests
(a)-(d) plus edge cases. B04 is not needed: tests write behavior.z rows
(52-dim float32 rings) and feature.active directly, and model.class via the
m_class layout.

Spec test mapping:
  (a) test_a_class_shift_removed_except_system_novelty
  (b) test_b_leave_one_out_single_shift_kept
  (c) test_c_composition_shift_never_removed
  (d) test_d_exactly_one_system_shift
"""
from __future__ import annotations

import math
import time
from typing import Dict, Iterable, List, Optional

import numpy as np

from helpers import DT, T0, make_store, run_engine, set_trust

from app.engines.behavior.common_mode import (ELIGIBLE, FLAG, GROUP_NAMES, ZI,
                                              CommonModeEngine, group_means, loo_median)
from app.engines.behavior.lib import emit
from app.engines.behavior.lib import gating as G
from app.engines.behavior.lib.classkeys import SYSTEM_KEY
from app.engines.behavior.lib.features import FEATURE_DIM, GROUPS
from app.models.schema import BehaviorEvent, Severity, SignatureMatch

S = "erp"
VOL = ELIGIBLE["volume"]
IVOL = GROUP_NAMES.index("volume")


# ------------------------------------------------------------------ helpers
def put_classes(store, roles: Dict[str, List[str]], system: str = S) -> None:
    assign = {f"{system}|{ip}": {"role": rid, "sub": None, "prob": 1.0, "static": [],
                                 "pool": None, "super": "machine"}
              for rid, ips in roles.items() for ip in ips}
    store.put_model("__org__", "__org__", "model.class", {
        "assign": assign,
        "roles": {rid: {"name": rid, "members": [f"{system}|{ip}" for ip in ips], "version": 1}
                  for rid, ips in roles.items()},
        "version": 1})


def zrow(**groups: float) -> np.ndarray:
    """52-dim z row with every feature of each named group set to a value
    (group names from features.GROUPS, or 'app_error')."""
    z = np.zeros(FEATURE_DIM)
    for g, v in groups.items():
        cols = ELIGIBLE["app_error"] if g == "app_error" else GROUPS[g]
        z[cols] = v
    return z


def tick(store, now: float, zs: Dict[str, Optional[np.ndarray]], system: str = S,
         dt: float = DT) -> None:
    """One B01/B04 tick: feature.active = 1 and behavior.z for each entity
    (None: active but unscored; use inactive() for silent entities)."""
    for ip, z in zs.items():
        store.register_entity(system, ip)
        store.add_vec(system, ip, "feature.active", now, [1.0], window_s=int(dt))
        if z is not None:
            store.add_vec(system, ip, "behavior.z", now, np.asarray(z, np.float32),
                          window_s=int(dt))


def inactive(store, now: float, ips: Iterable[str], system: str = S) -> None:
    for ip in ips:
        store.register_entity(system, ip)
        store.add_vec(system, ip, "feature.active", now, [0.0], window_s=int(DT))


def zi_of(store, ip: str, now: float, system: str = S) -> np.ndarray:
    r = store.vec_at(system, ip, ZI, now)
    assert r is not None, f"no zi for {ip}"
    return np.asarray(r, np.float64)


def flag_of(store, ip: str, now: float, system: str = S) -> Dict[str, int]:
    return emit.read_dict(store, system, ip, FLAG, now)


def vol_mean(z: np.ndarray) -> float:
    return float(np.mean(z[VOL]))


def shifts(store, system: str = S) -> List[BehaviorEvent]:
    return store.events(system, SYSTEM_KEY, kinds=("system_shift",), limit=100)


# ------------------------------------------------------------ spec tests
def test_a_class_shift_removed_except_system_novelty():
    store = make_store()
    ips = [f"10.0.0.{i}" for i in range(1, 7)]
    put_classes(store, {"r1": ips})
    eng = CommonModeEngine()
    now = T0 + 10 * DT
    # the member with a new system-tier SNI (B08 event at t-1)
    x = ips[0]
    store.add_event(BehaviorEvent(system=S, entity=x, ts=now - DT, kind="first_seen", score=1.0,
                                  extra={"tier": "system", "dim": "sni", "value": "evil.example"}))
    rng = np.random.default_rng(1)
    zs = {ip: zrow(volume=3.0) + np.r_[rng.normal(0, 0.1, 9), np.zeros(43)] for ip in ips}
    tick(store, now, zs)
    run_engine(eng, store, now)
    for ip in ips[1:]:
        zi = zi_of(store, ip, now)
        assert abs(vol_mean(zi)) < 0.5, (ip, vol_mean(zi))
        assert flag_of(store, ip, now)["volume"] == 1
    zi_x = zi_of(store, x, now)
    np.testing.assert_allclose(zi_x, np.asarray(zs[x], np.float32), atol=0)
    assert all(v == 0 for v in flag_of(store, x, now).values())


def test_b_leave_one_out_single_shift_kept():
    store = make_store()
    ips = [f"10.0.1.{i}" for i in range(1, 5)]
    put_classes(store, {"r1": ips})
    eng = CommonModeEngine()
    now = T0
    zs = {ip: zrow() for ip in ips}
    zs[ips[0]] = zrow(volume=3.0)
    tick(store, now, zs)
    run_engine(eng, store, now)
    zi = zi_of(store, ips[0], now)
    assert np.all(zi[VOL] >= 2.8)
    assert flag_of(store, ips[0], now)["volume"] == 0


def test_c_composition_shift_never_removed():
    store = make_store()
    ips = [f"10.0.2.{i}" for i in range(1, 7)]
    put_classes(store, {"r1": ips})
    eng = CommonModeEngine()
    now = T0
    # every non-eligible group (and the non-error app columns) shifts class-wide
    z = zrow(comp=3.0, breadth=3.0, dns=-3.0, tls=3.0, timing=3.0, app=3.0)
    z[ELIGIBLE["app_error"]] = 0.0
    tick(store, now, {ip: z for ip in ips})
    run_engine(eng, store, now)
    for ip in ips:
        zi = zi_of(store, ip, now)
        for g in ("comp", "breadth", "dns", "tls", "timing"):
            np.testing.assert_array_equal(zi[GROUPS[g]], z[GROUPS[g]])
        non_err = [i for i in GROUPS["app"] if i not in ELIGIBLE["app_error"]]
        np.testing.assert_array_equal(zi[non_err], z[non_err])
        assert "comp" not in flag_of(store, ip, now)


def test_c_app_error_is_eligible():
    store = make_store()
    ips = [f"10.0.2.{i}" for i in range(1, 7)]
    put_classes(store, {"r1": ips})
    eng = CommonModeEngine()
    tick(store, T0, {ip: zrow(app_error=4.0) for ip in ips})
    run_engine(eng, store, T0)
    for ip in ips:
        zi = zi_of(store, ip, T0)
        assert np.all(np.abs(zi[ELIGIBLE["app_error"]]) < 0.5)
        assert flag_of(store, ip, T0)["app_error"] == 1


def test_d_exactly_one_system_shift():
    store = make_store()
    ips = [f"10.0.3.{i}" for i in range(1, 9)]
    put_classes(store, {"r1": ips[:4], "r2": ips[4:]})
    eng = CommonModeEngine()
    for k in range(12):
        now = T0 + k * DT
        v = 3.0 if 3 <= k < 10 else 0.0
        tick(store, now, {ip: zrow(volume=v, transport=v) for ip in ips})
        run_engine(eng, store, now)
    ev = shifts(store)
    assert len(ev) == 1
    e = ev[0]
    assert e.ts == T0 + 4 * DT                          # the 2nd coherent tick
    assert e.severity == Severity.INFO
    assert set(e.extra["groups"]) == {"volume", "transport"}
    assert e.extra["groups"]["volume"]["dir"] == 1
    # the aggregates B18 reads
    sysv = store.latest_derived(S, SYSTEM_KEY, "behavior.common.volume")
    assert sysv is not None and sysv.value["n_scored"] == 8
    for ck in ("class:r1", "class:r2"):
        m = store.derived_series(S, ck, "behavior.common.volume")
        assert m and abs(m[5].value["L"] - 3.0) < 1e-6 and m[5].value["n_members"] == 4


def test_d_new_episode_after_quiet_emits_again_and_minority_does_not():
    store = make_store()
    ips = [f"10.0.4.{i}" for i in range(1, 9)]
    put_classes(store, {"r1": ips})
    eng = CommonModeEngine()
    pattern = [0, 3, 3, 3, 0, 0, 3, 3, 0]
    for k, v in enumerate(pattern):
        now = T0 + k * DT
        tick(store, now, {ip: zrow(volume=v) for ip in ips})
        run_engine(eng, store, now)
    assert len(shifts(store)) == 2
    # 3 of 8 (< 50 %) shifting for many ticks: nothing
    store2 = make_store()
    put_classes(store2, {"r1": ips})
    eng2 = CommonModeEngine()
    for k in range(6):
        now = T0 + k * DT
        tick(store2, now, {ip: zrow(volume=3.0 if i < 3 else 0.0) for i, ip in enumerate(ips)})
        run_engine(eng2, store2, now)
    assert shifts(store2) == []


# ---------------------------------------------------------- exclusions
def _class_shift_one_tick(setup) -> tuple:
    store = make_store()
    ips = [f"10.0.5.{i}" for i in range(1, 7)]
    put_classes(store, {"r1": ips})
    now = T0 + 5 * DT
    setup(store, ips[0], now)
    z = zrow(volume=3.0)
    tick(store, now, {ip: z for ip in ips})
    run_engine(CommonModeEngine(), store, now)
    return store, ips, now, z


def _is_excluded(store, ip, now, z) -> bool:
    return bool(np.array_equal(zi_of(store, ip, now), z.astype(np.float32))
                and not any(flag_of(store, ip, now).values()))


def test_exclusion_lib4_match_medium_but_not_low():
    def med(store, ip, now):
        store.add_match(SignatureMatch(system=S, entity=ip, ts=now - DT, signature_id="x",
                                       label="scan", category="recon", confidence=0.9,
                                       severity=Severity.MEDIUM))
    store, ips, now, z = _class_shift_one_tick(med)
    assert _is_excluded(store, ips[0], now, z)
    assert not _is_excluded(store, ips[1], now, z)

    def low(store, ip, now):
        store.add_match(SignatureMatch(system=S, entity=ip, ts=now - DT, signature_id="x",
                                       label="browse", category="browse", confidence=0.9,
                                       severity=Severity.LOW))
    store, ips, now, z = _class_shift_one_tick(low)
    assert not _is_excluded(store, ips[0], now, z)


def test_exclusion_client_score_identity_p_and_budget_exfil_alarm():
    def client(store, ip, now):
        emit.write_scores(store, S, ip, now - DT, {"client": 2.5})         # p ~ 3e-3
    store, ips, now, z = _class_shift_one_tick(client)
    assert _is_excluded(store, ips[0], now, z)

    def client_weak(store, ip, now):
        emit.write_scores(store, S, ip, now - DT, {"client": 1.0})         # p = 0.1
    store, ips, now, z = _class_shift_one_tick(client_weak)
    assert not _is_excluded(store, ips[0], now, z)

    def identity_calibrated(store, ip, now):
        # raw score looks weak, but B24's calibrated p is < 0.01
        emit.write_scores(store, S, ip, now - DT, {"identity": 0.5})
        emit.write_pvalues(store, S, ip, now - DT, {"identity": 0.004})
    store, ips, now, z = _class_shift_one_tick(identity_calibrated)
    assert _is_excluded(store, ips[0], now, z)

    def exfil(store, ip, now):
        emit.write_scores(store, S, ip, now - DT, {"budget_exfil": 0.3},
                          acc_alarm={"budget_exfil": 1})
    store, ips, now, z = _class_shift_one_tick(exfil)
    assert _is_excluded(store, ips[0], now, z)

    def stale(store, ip, now):                  # evidence 5 ticks old: not "this tick"
        emit.write_scores(store, S, ip, now - 5 * DT, {"client": 6.0})
    store, ips, now, z = _class_shift_one_tick(stale)
    assert not _is_excluded(store, ips[0], now, z)


def test_class_novelty_tier_does_not_exclude():
    def cls(store, ip, now):
        store.add_event(BehaviorEvent(system=S, entity=ip, ts=now - DT, kind="first_seen",
                                      score=1.0, extra={"tier": "class"}))
    store, ips, now, z = _class_shift_one_tick(cls)
    assert not _is_excluded(store, ips[0], now, z)


def test_quarantined_and_excluded_members_leave_the_pool():
    store = make_store()
    ips = [f"10.0.6.{i}" for i in range(1, 5)]
    put_classes(store, {"r1": ips})
    now = T0 + 3 * DT
    set_trust(store, S, ips[3], [now - DT], 0.0, quarantine=1.0)
    tick(store, now, {ip: zrow(volume=3.0) for ip in ips})
    run_engine(CommonModeEngine(), store, now)
    # 2 eligible others < 3 in the class and in the system -> L = 0
    assert vol_mean(zi_of(store, ips[0], now)) > 2.9
    # the quarantined entity itself is still de-meaned against its 3 others
    assert abs(vol_mean(zi_of(store, ips[3], now))) < 1e-6


def test_small_class_falls_back_to_system():
    store = make_store()
    big = [f"10.0.7.{i}" for i in range(1, 6)]
    small = ["10.0.8.1", "10.0.8.2"]
    put_classes(store, {"big": big, "small": small})
    tick(store, T0, {ip: zrow(volume=2.5) for ip in big + small})
    run_engine(CommonModeEngine(), store, T0)
    for ip in small:
        assert abs(vol_mean(zi_of(store, ip, T0))) < 1e-6
        assert flag_of(store, ip, T0)["volume"] == 1


def test_unclassified_system_tier_and_lonely_entity():
    store = make_store()
    ips = [f"10.0.9.{i}" for i in range(1, 5)]
    tick(store, T0, {ip: zrow(volume=3.0) for ip in ips})     # no model.class at all
    run_engine(CommonModeEngine(), store, T0)
    assert abs(vol_mean(zi_of(store, ips[0], T0))) < 1e-6
    store2 = make_store()
    tick(store2, T0, {"10.1.0.1": zrow(volume=3.0), "10.1.0.2": zrow(volume=3.0)})
    run_engine(CommonModeEngine(), store2, T0)
    assert vol_mean(zi_of(store2, "10.1.0.1", T0)) == 3.0      # L = 0


# ------------------------------------------------------------- loadings
def _factor_run(n_ticks: int, trust: Optional[float], quarantine: float = 0.0,
                training: bool = False, eng: Optional[CommonModeEngine] = None):
    """6 members follow a common volume factor f_t; the target ips[0]
    responds with 0.4 f_t (a half-loaded member). `quarantine` applies to the
    target only (a quarantined member leaves the others' pools)."""
    store = make_store()
    ips = [f"10.2.0.{i}" for i in range(1, 7)]
    put_classes(store, {"r1": ips})
    eng = eng or CommonModeEngine()
    rng = np.random.default_rng(7)
    for k in range(n_ticks):
        now = T0 + k * DT
        f = rng.normal(0.0, 1.5)
        zs = {ip: zrow(volume=f + rng.normal(0, 0.2)) for ip in ips[1:]}
        zs[ips[0]] = zrow(volume=0.4 * f + rng.normal(0, 0.2))
        if trust is not None:
            for ip in ips:
                set_trust(store, S, ip, [now], trust,
                          quarantine=quarantine if ip == ips[0] else 0.0)
        tick(store, now, zs)
        run_engine(eng, store, now, training=training)
    return store, eng, ips


def test_loading_learned_from_committed_rows_and_refit_every_16_ticks():
    store, eng, ips = _factor_run(200, trust=1.0)
    b = eng.beta(S, ips[0])[IVOL]
    assert 0.3 < b < 0.6, b
    # a fully loaded member stays near 1
    assert 0.8 < eng.beta(S, ips[1])[IVOL] < 1.2
    # beta only changes on refit ticks: count distinct values over 32 more ticks
    vals = []
    rng = np.random.default_rng(3)
    for k in range(200, 232):
        now = T0 + k * DT
        f = rng.normal(0, 1.5)
        zs = {ip: zrow(volume=f) for ip in ips[1:]}
        zs[ips[0]] = zrow(volume=0.4 * f)
        for ip in ips:
            set_trust(store, S, ip, [now], 1.0)
        tick(store, now, zs)
        run_engine(eng, store, now)
        vals.append(float(eng.beta(S, ips[0])[IVOL]))
    assert 2 <= len(set(vals)) <= 3


def test_untrusted_or_quarantined_rows_do_not_move_loading():
    _, eng, ips = _factor_run(120, trust=0.0)
    assert abs(eng.beta(S, ips[0])[IVOL] - 1.0) < 1e-12
    _, eng, ips = _factor_run(120, trust=1.0, quarantine=1.0)
    assert abs(eng.beta(S, ips[0])[IVOL] - 1.0) < 1e-12


def test_release_commits_held_rows_and_refits():
    store, eng, ips = _factor_run(80, trust=1.0, quarantine=1.0)
    assert abs(eng.beta(S, ips[0])[IVOL] - 1.0) < 1e-12
    now = T0 + 80 * DT
    assert len(eng._ents[(S, ips[0])].gate.held) > 50
    store.put_model(S, ips[0], "model.control", {"version": 0, "release": [T0, now]})
    tick(store, now, {ip: zrow() for ip in ips})
    run_engine(eng, store, now)
    assert eng.beta(S, ips[0])[IVOL] < 0.7            # refit forced by the control action


def test_rollback_restores_loading():
    """Loading 1 for 120 ticks, then the target decouples (loading 0) for 80
    ticks; model.control.rollback_to = the change point restores beta ~ 1."""
    store = make_store()
    ips = [f"10.2.1.{i}" for i in range(1, 7)]
    put_classes(store, {"r1": ips})
    eng = CommonModeEngine()
    rng = np.random.default_rng(11)
    for k in range(200):
        now = T0 + k * DT
        f = rng.normal(0.0, 1.5)
        zs = {ip: zrow(volume=f) for ip in ips[1:]}
        zs[ips[0]] = zrow(volume=f if k < 120 else 0.0)
        for ip in ips:
            # B28 quarantines at or before the tick it writes rollback_to
            set_trust(store, S, ip, [now], 1.0,
                      quarantine=1.0 if (ip == ips[0] and k == 199) else 0.0)
        tick(store, now, zs)
        run_engine(eng, store, now)
    assert eng.beta(S, ips[0])[IVOL] < 0.8
    tau = T0 + 119 * DT
    now = T0 + 200 * DT
    store.put_model(S, ips[0], "model.control", {"version": 0, "rollback_to": tau})
    tick(store, now, {ip: zrow() for ip in ips})
    run_engine(eng, store, now)
    st = eng._ents[(S, ips[0])]
    assert st.gate.applied.get("rollback_to") == tau
    assert all(r.ts <= tau for r in st.gate.journal)
    assert st.gate.held and min(r.ts for r in st.gate.held) > tau
    assert eng.beta(S, ips[0])[IVOL] > 0.95


def test_training_mode_learns_but_emits_no_event():
    store, eng, ips = _factor_run(120, trust=None, training=True)
    assert eng.beta(S, ips[0])[IVOL] < 0.7            # missing trust counts as 1 in training
    store = make_store()
    ips = [f"10.0.3.{i}" for i in range(1, 9)]
    eng = CommonModeEngine()
    for k in range(6):
        now = T0 + k * DT
        tick(store, now, {ip: zrow(volume=3.0) for ip in ips})
        run_engine(eng, store, now, training=True)
    assert shifts(store) == []
    assert flag_of(store, ips[0], T0 + 5 * DT)["volume"] == 1


# ----------------------------------------------------------- edge cases
def test_empty_store():
    store = make_store()
    assert run_engine(CommonModeEngine(), store, T0) == 0


def test_silent_and_degraded_entities():
    store = make_store()
    ips = [f"10.0.10.{i}" for i in range(1, 6)]
    put_classes(store, {"r1": ips})
    inactive(store, T0, [ips[0]])                       # silent: absence is data
    tick(store, T0, {ips[1]: None})                     # active, B04 did not score it
    tick(store, T0, {ip: zrow(volume=3.0) for ip in ips[2:]})
    run_engine(CommonModeEngine(), store, T0)
    assert store.vec_at(S, ips[0], ZI, T0) is None
    assert flag_of(store, ips[0], T0) == {}
    assert np.all(np.isnan(zi_of(store, ips[1], T0)))
    # only 2 scored others per entity -> L = 0
    assert vol_mean(zi_of(store, ips[2], T0)) == 3.0


def test_nan_inputs():
    store = make_store()
    ips = [f"10.0.11.{i}" for i in range(1, 7)]
    put_classes(store, {"r1": ips})
    zs = {ip: zrow(volume=3.0) for ip in ips}
    zs[ips[0]] = np.full(FEATURE_DIM, np.nan)            # all NaN: unscored everywhere
    zs[ips[1]][VOL[:4]] = np.nan                         # partial NaN
    tick(store, T0, zs)
    run_engine(CommonModeEngine(), store, T0)
    assert np.all(np.isnan(zi_of(store, ips[0], T0)))
    assert flag_of(store, ips[0], T0)["volume"] == 0
    zi1 = zi_of(store, ips[1], T0)
    assert np.all(np.isnan(zi1[VOL[:4]]))
    assert np.all(np.abs(zi1[VOL[4:]]) < 1e-6)
    for ip in ips[2:]:
        assert abs(vol_mean(zi_of(store, ip, T0))) < 1e-6


def test_cadence_900_to_60():
    store = make_store()
    ips = [f"10.0.12.{i}" for i in range(1, 9)]
    put_classes(store, {"r1": ips})
    eng = CommonModeEngine()
    now = T0
    for k in range(6):                               # 900 s ticks, coherent from k = 5
        v = 3.0 if k >= 5 else 0.0
        for ip in ips:
            set_trust(store, S, ip, [now], 1.0)
        tick(store, now, {ip: zrow(volume=v) for ip in ips}, dt=DT)
        run_engine(eng, store, now, dt=DT)
        now += DT
    now = now - DT + 60.0
    for k in range(40):                              # switch to 60 s ticks
        for ip in ips:
            set_trust(store, S, ip, [now], 1.0)
        tick(store, now, {ip: zrow(volume=3.0) for ip in ips}, dt=60.0)
        run_engine(eng, store, now, dt=60.0)
        now += 60.0
    ev = shifts(store)
    assert len(ev) == 1 and ev[0].ts == T0 + 5 * DT + 60.0    # run carried across the switch
    # rows keep committing at the new cadence (D = 10 ticks at 60 s; commits
    # are batched on the entity's refit tick, every 16 ticks = 960 s)
    st = eng._ents[(S, ips[0])]
    last = now - 60.0
    assert st.gate.last_ts <= last - G.commit_delay_ticks(60.0) * 60.0 + 1e-6
    assert st.gate.last_ts >= last - G.commit_delay_ticks(60.0) * 60.0 - 16 * 60.0 - 1e-6
    assert any(r.ts > T0 + 5 * DT for r in st.gate.journal)


def test_loo_median_matches_bruteforce():
    rng = np.random.default_rng(0)
    for n in range(1, 12):
        v = rng.normal(size=n)
        v[rng.random(n) < 0.2] = np.nan
        pool = np.isfinite(v) & (rng.random(n) < 0.8)
        got = loo_median(v, pool)
        for i in range(n):
            others = [v[j] for j in range(n) if pool[j] and j != i]
            want = float(np.median(others)) if len(others) >= 3 else math.nan
            if math.isnan(want):
                assert math.isnan(got[i])
            else:
                assert abs(got[i] - want) < 1e-12


def test_group_means_nan_aware():
    z = np.full((2, FEATURE_DIM), np.nan)
    z[0, VOL[0]] = 2.0
    z[0, VOL[1]] = 4.0
    m = group_means(z)
    assert m[0, IVOL] == 3.0 and np.isnan(m[1, IVOL])


def test_engine_declarations():
    eng = CommonModeEngine()
    assert eng.name == "behavior.common_mode" and eng.layer == "behavior"
    assert eng.interval == 1 and eng.period_s is None
    assert "behavior.z" in eng.consumes and ZI in eng.produces
    assert "event.system_shift" in eng.produces


def test_perf_40_entities():
    store = make_store()
    ips = [f"10.3.0.{i}" for i in range(40)]
    put_classes(store, {f"r{k}": ips[k * 8:(k + 1) * 8] for k in range(5)})
    eng = CommonModeEngine()
    rng = np.random.default_rng(5)
    times = []
    for k in range(40):
        now = T0 + k * DT
        for ip in ips:
            set_trust(store, S, ip, [now], 1.0)
        tick(store, now, {ip: rng.normal(size=FEATURE_DIM) for ip in ips})
        t0 = time.perf_counter()
        run_engine(eng, store, now)
        times.append(time.perf_counter() - t0)
    med_ms = 1000 * float(np.median(times[5:]))
    assert med_ms < 25.0, med_ms                 # spec target 1 ms; generous CI bound


def test_group_with_no_scored_member_is_written_as_undefined():
    """A class whose scored members all have NaN on one group (e.g. no
    transport exposure this tick) still gets a behavior.common.<g> point,
    undefined (L NaN, n 0, dir 0), so the series keeps its cadence instead
    of going silent (eval robustness gate: stale class:<id> series)."""
    store = make_store()
    ips = [f"10.0.12.{i}" for i in range(1, 5)]
    idle = ["10.0.12.8", "10.0.12.9"]
    put_classes(store, {"r1": ips, "r2": idle})
    inactive(store, T0, idle)
    zs = {}
    for ip in ips:
        z = zrow(volume=0.5)
        z[GROUPS["transport"]] = np.nan
        zs[ip] = z
    tick(store, T0, zs)
    run_engine(CommonModeEngine(), store, T0)
    for key in ("class:r1", SYSTEM_KEY):
        v = emit.read_dict(store, S, key, "behavior.common.transport", T0)
        assert v, key
        assert math.isnan(v["L"]) and v["n"] == 0 and v["n_scored"] == 0 and v["dir"] == 0
        assert emit.read_dict(store, S, key, "behavior.common.volume", T0)["n_scored"] == 4
    # a class none of whose members is scored this tick, and a system in
    # which nobody is scored, are written as undefined too
    v = emit.read_dict(store, S, "class:r2", "behavior.common.volume", T0)
    assert v and v["n_scored"] == 0 and math.isnan(v["L"]) and v["n_members"] == 2
    t1 = T0 + DT
    inactive(store, t1, ips + idle)
    run_engine(CommonModeEngine(), store, t1)
    for key in ("class:r1", "class:r2", SYSTEM_KEY):
        v = emit.read_dict(store, S, key, "behavior.common.volume", t1)
        assert v and v["n_scored"] == 0 and v["dir"] == 0, key
