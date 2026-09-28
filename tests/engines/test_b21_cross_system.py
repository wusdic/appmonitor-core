"""B21 CrossSystemEngine (engines/behavior/cross_system.py, lib/m_xsys.py).

Spec unit test (docs/lib3/engines.md B21): an IP active only in erp-prod for
30 d starts using oa-portal /admin: first_access_system at class tier; the
same access adopted class-wide is suppressed. Plus: the org tier (nobody
bridges the two systems), validity of the p-value on a stable multi-system
footprint, the 24-h spread NB, cadence invariance of the unit of
observation (canonical H windows), trust gating (a quarantined first access
stays new), training mode, precursor severity, B01 failure, the registry
position and the detector registry entry.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from helpers import T0, make_store, put_model, run_engine, set_trust

from app.engines.behavior import cross_system as X
from app.engines.behavior.lib import detectors as DET
from app.engines.behavior.lib import emit
from app.engines.behavior.lib import m_xsys as MX
from app.engines.behavior.lib import stages as STG
from app.engines.behavior.lib.classkeys import ORG
from app.models.schema import DerivedMetric, EntityProfile

ERP, OA, API = "erp-prod", "oa-portal", "api-gateway"
H = 3600.0
DAY = 86400.0
T_START = math.floor(T0 / DAY) * DAY          # a UTC midnight


def _class_model(roles):
    """model.class with {(system, ip): rid} assignments."""
    assign = {f"{s}|{ip}": {"role": rid, "prob": 1.0} for (s, ip), rid in roles.items()}
    return {"assign": assign, "roles": {}, "version": 1}


def _tick(st, eng, now, active, dt=H, training=True, config=None, known=()):
    """One tick: feature.active = 1 on the active (system, ip) keys, 0 on the
    other known keys (B01 writes every real key), trust 1 on the active keys
    live (as B28 on a trusted period), then B21."""
    keys = set(active) | set(known)
    for s, ip in sorted(keys):
        v = 1.0 if (s, ip) in active else 0.0
        st.add_vec(s, ip, "feature.active", now, np.array([v], dtype=np.float32),
                   window_s=int(dt))
        st.register_entity(s, ip)
        if st.profile(s, ip) is None:
            st.put_profile(EntityProfile(system=s, entity=ip, updated=now))
        if not training:
            set_trust(st, s, ip, [now], 1.0)
    return run_engine(eng, st, now, training=training, dt=dt, config=config)


def _history(st, eng, days, hours_active, footprint, t0=T_START, known=(), dt=H):
    """Warm-up (training) of `days` days at `dt`: every IP active in its
    footprint systems during `hours_active` local-ish hours of each day."""
    known = set(known) | {(s, ip) for ip, ss in footprint.items() for s in ss}
    n = int(days * DAY / dt)
    for i in range(1, n + 1):
        now = t0 + i * dt
        hour = ((now - dt / 2.0) % DAY) / H
        act = set()
        if hours_active[0] <= hour < hours_active[1]:
            act = {(s, ip) for ip, ss in footprint.items() for s in ss}
        _tick(st, eng, now, act, dt=dt, training=True, known=known)
    return t0 + n * dt, known


def _score_at(st, s, ip, now):
    row = emit.read_row(st, s, ip, emit.SCORE, now)
    pm = emit.read_row(st, s, ip, emit.PM, now)
    return row.get("cross_system"), pm.get("cross_system")


def _events(st, s=None, ip=None):
    return [e for e in st.events(s, ip, limit=1000) if e.kind == "first_access_system"]


ERP_CLASS = [f"10.20.1.{i}" for i in range(11, 17)]       # 6 members
IP = ERP_CLASS[0]


def _erp_world(bridge=False):
    st = make_store()
    roles = {(ERP, ip): "r1" for ip in ERP_CLASS}
    fp = {ip: [ERP] for ip in ERP_CLASS}
    fp.update({f"10.30.2.{i}": [OA] for i in range(21, 27)})
    roles.update({(OA, f"10.30.2.{i}"): "r2" for i in range(21, 27)})
    if bridge:                                   # an admin host of another role in both
        fp["10.99.0.1"] = [ERP, OA]
        roles[(ERP, "10.99.0.1")] = "r9"
    put_model(st, *ORG, "model.class", _class_model(roles))
    return st, fp


# ------------------------------------------------------------ spec test (a)
@pytest.mark.parametrize("bridge,tier", [(True, "class"), (False, "org")])
def test_first_access_to_a_new_system_is_an_event_at_class_or_org_tier(bridge, tier):
    st, fp = _erp_world(bridge)
    eng = X.CrossSystemEngine()
    now, known = _history(st, eng, 30, (9, 18), fp)
    # live: next workday morning, the IP reaches oa-portal (/admin)
    t = now + 10 * H
    _tick(st, eng, t, {(ERP, IP), (OA, IP)}, training=False, known=known)
    sc, pm = _score_at(st, OA, IP, t)
    assert pm is not None and pm < 1e-4, pm
    assert sc >= 3.75
    home_sc, home_pm = _score_at(st, ERP, IP, t)
    assert home_pm is not None and home_pm > 0.01    # the home system stays expected
    ev = _events(st, OA, IP)
    assert len(ev) == 1
    e = ev[0]
    assert e.extra["tier"] == tier and e.extra["home"] == ERP
    assert e.axes == ["lateral"] and "cross_system" in e.p_by_detector
    assert STG.stage_for_axis("lateral") == "lateral"
    assert emit.read_dict(st, OA, IP, emit.AXES, t)["cross_system"] == ["lateral"]
    assert (e.extra["support_aa"] > 0) == bridge
    # once per pair per 30 d, and never for the home key
    _tick(st, eng, t + H, {(ERP, IP), (OA, IP)}, training=False, known=known)
    assert len(_events(st, OA, IP)) == 1 and not _events(st, ERP, IP)


# ------------------------------------------------------------ spec test (b)
def test_the_same_access_adopted_class_wide_is_suppressed():
    st, fp = _erp_world(bridge=True)
    eng = X.CrossSystemEngine()
    now, known = _history(st, eng, 30, (9, 18), fp)
    # 4 of the 6 members start using oa-portal, one a day (staggered)
    t = now
    adopters = ERP_CLASS[1:5]
    for d in range(4):
        started = adopters[:d + 1]
        for h in range(9, 12):
            t = now + d * DAY + h * H
            act = {(ERP, ip) for ip in ERP_CLASS[1:]} | {(OA, ip) for ip in started}
            _tick(st, eng, t, act, training=False, known=known | {(OA, ip) for ip in started})
    n_before = len(_events(st))
    t2 = now + 4 * DAY + 10 * H
    _tick(st, eng, t2, {(ERP, IP), (OA, IP)}, training=False, known=known)
    assert not _events(st, OA, IP)
    sc, pm = _score_at(st, OA, IP, t2)
    # the adoption discount (p -> p^0.1) leaves no evidence
    assert pm is not None and pm > 0.05
    # the first adopter was reported (oa-portal class-rare: tier >= class);
    # once members use it, it is not class-rare (entity tier) and the 4th
    # adopter already joined an adoption (3 others within 7 d)
    assert len(_events(st, OA, adopters[0])) == 1
    assert 1 <= n_before <= 3
    assert not _events(st, OA, adopters[3])


def test_adoption_record_needs_max3_or_30pct():
    st, fp = _erp_world(bridge=True)
    eng = X.CrossSystemEngine()
    now, known = _history(st, eng, 30, (9, 18), fp)
    adopters = ERP_CLASS[1:3]                    # only 2 of 6: not an adoption
    for h in range(9, 12):
        t = now + h * H
        act = {(ERP, ip) for ip in ERP_CLASS[1:]} | {(OA, ip) for ip in adopters}
        _tick(st, eng, t, act, training=False, known=known | {(OA, ip) for ip in adopters})
    t2 = now + DAY + 10 * H
    _tick(st, eng, t2, {(ERP, IP), (OA, IP)}, training=False, known=known)
    assert len(_events(st, OA, IP)) == 1


# ------------------------------------------------------------- validity
def test_upper_p_is_valid_on_a_stable_multi_system_footprint():
    """IPs whose active hours fall in A / B / C with fixed probabilities:
    after 30 d, P(p <= alpha) <= alpha (+ Monte-Carlo slack) on live rows."""
    rng = np.random.default_rng(7)
    probs = {ERP: 0.8, OA: 0.15, API: 0.05}
    ips = [f"10.50.0.{i}" for i in range(1, 9)]
    st = make_store()
    put_model(st, *ORG, "model.class", _class_model({(ERP, ip): "r5" for ip in ips}))
    eng = X.CrossSystemEngine()
    known = {(s, ip) for s in probs for ip in ips}
    names, pv = list(probs), np.array(list(probs.values()))

    def draw():
        return {(names[int(rng.choice(3, p=pv))], ip) for ip in ips}

    t = T_START
    for i in range(30 * 24):
        t += H
        _tick(st, eng, t, draw(), training=True, known=known)
    ps = []
    for i in range(10 * 24):
        t += H
        act = draw()
        _tick(st, eng, t, act, training=False, known=known)
        for s, ip in act:
            ps.append(_score_at(st, s, ip, t)[1])
    ps = np.array(ps)
    assert len(ps) == 10 * 24 * len(ips)
    for a in (0.2, 0.1, 0.05):
        assert (ps <= a).mean() <= a + 0.02, (a, (ps <= a).mean())
    assert not _events(st)


def test_spread_nb_with_class_prior():
    hist = [(0.0, 1.0)] * 20                     # always one system a day
    p2, m, r = MX.spread_p(2, hist, 0.0)
    assert p2 < 0.01 and m < 0.05
    p2c, _, _ = MX.spread_p(2, [], 1.0)           # a class that routinely spans 2
    assert p2c > 5 * p2
    assert MX.spread_p(0, hist, 0.0)[0] == 1.0
    # more history of spreading -> less surprise
    assert MX.spread_p(2, [(1.0, 1.0)] * 20, 0.0)[0] > 0.1


def test_predictive_backoff_and_upper_p():
    systems = [API, ERP, OA]
    org = {ERP: 300.0, OA: 300.0, API: 300.0}
    cls = {ERP: 1000.0}
    pred = MX.predictive(systems, {ERP: 150.0}, 0.0, cls, 0.0, org)
    assert abs(sum(pred.values()) - 1.0) < 1e-12
    assert MX.upper_p(pred, ERP) == 1.0
    p_oa = MX.upper_p(pred, OA)
    assert 1e-8 < p_oa < 1e-5
    # an immature IP backs off to its class, a classless one to org
    assert MX.upper_p(MX.predictive(systems, {}, 0.0, cls, 0.0, org), OA) < 1e-3
    assert MX.upper_p(MX.predictive(systems, {}, 0.0, None, 0.0, org), OA) > 0.5
    # the score grid ties near-1 p at exactly 0
    assert MX.score_of(0.97) == 0.0 and MX.score_of(1e-6) == 6.0


# --------------------------------------------------------- cadence (canonical)
@pytest.mark.parametrize("dt", [60.0, 900.0, 3600.0])
def test_unit_of_observation_is_the_h_window(dt):
    """Canonical mode: one score and one learned count per active pair-hour
    at 60, 900 and 3600 s alike."""
    st = make_store()
    eng = X.CrossSystemEngine()
    cfg = {"grain_mode": "canonical"}
    ip = "10.20.1.99"
    t = T_START
    n = int(6 * H / dt)
    scored = 0
    for i in range(1, n + 1):
        t = T_START + i * dt
        _tick(st, eng, t, {(ERP, ip)}, dt=dt, training=True, config=cfg)
        if _score_at(st, ERP, ip, t)[0] is not None:
            scored += 1
    assert scored == 6
    # flush the commit delay, then compare the learned counts
    for j in range(1, max(6, int(2 * H / dt)) + 1):
        _tick(st, eng, t + j * dt, set(), dt=dt, training=True, config=cfg, known={(ERP, ip)})
    rec = st.get_model(*ORG, MX.MODEL)["pairs"][MX.pair_key(ERP, ip)]["state"]
    assert rec["n"] == 6
    assert len(rec["wins"]) == 6


# ---------------------------------------------------------------- gating
def test_quarantined_first_access_is_held_and_stays_new():
    st, fp = _erp_world(bridge=False)
    eng = X.CrossSystemEngine()
    now, known = _history(st, eng, 20, (9, 18), fp)
    pms = []
    for h in range(8):
        t = now + (10 + h) * H
        for s, ip in ((ERP, IP), (OA, IP)):
            st.add_vec(s, ip, "behavior.quarantine", t - H, np.array([1.0], dtype=np.float32))
        _tick(st, eng, t, {(ERP, IP), (OA, IP)}, training=False, known=known)
        for s, ip in ((ERP, IP), (OA, IP)):       # B28: incident open, quarantined
            st.add_vec(s, ip, "behavior.quarantine", t, np.array([1.0], dtype=np.float32))
        pms.append(_score_at(st, OA, IP, t)[1])
    assert max(pms) < 1e-4                       # never learned while quarantined
    rec = st.get_model(*ORG, MX.MODEL)["pairs"][MX.pair_key(OA, IP)]
    assert rec["gate"].held and rec["state"]["n"] == 0


def test_trusted_new_system_is_learned():
    st, fp = _erp_world(bridge=False)
    eng = X.CrossSystemEngine()
    now, known = _history(st, eng, 20, (9, 18), fp)
    t = now
    for d in range(12):                          # used daily and trusted (warm-up)
        for h in range(9, 18):
            t = now + d * DAY + h * H
            _tick(st, eng, t, {(ERP, IP), (OA, IP)}, training=True, known=known)
    t = now + 13 * DAY + 10 * H
    _tick(st, eng, t, {(ERP, IP), (OA, IP)}, training=False, known=known)
    assert _score_at(st, OA, IP, t)[1] > 0.05


# ------------------------------------------------------------ other paths
def test_training_mode_emits_no_events_and_precursor_raises_severity():
    st, fp = _erp_world(bridge=False)
    eng = X.CrossSystemEngine()
    now, known = _history(st, eng, 20, (9, 18), fp)
    t = now + 10 * H
    _tick(st, eng, t, {(ERP, IP), (OA, IP)}, training=True, known=known)
    assert not _events(st)
    # live, one day later, after an alarm on the home key
    t2 = t + DAY
    ip2 = ERP_CLASS[1]
    st.add_derived(DerivedMetric(
        name="behavior.alarm", value={"severity": "low", "axes": ["volume"]}, ts=t2 - 2 * H,
        system=ERP, entity=ip2, window_s=3600))
    _tick(st, eng, t2, {(ERP, ip2), (OA, ip2)}, training=False, known=known)
    ev = _events(st, OA, ip2)
    assert len(ev) == 1 and ev[0].severity.value == "medium"
    assert ev[0].extra["precursor"]["system"] == ERP


def test_b01_failure_writes_degraded():
    st, fp = _erp_world(bridge=False)
    eng = X.CrossSystemEngine()
    now, known = _history(st, eng, 2, (0, 24), fp)
    t = now + H
    st.put_health("behavior.feature_vector", {"engine": "behavior.feature_vector",
                                              "last_error_ts": t})
    run_engine(eng, st, t, training=False, dt=H)
    dg = emit.read_dict(st, ERP, IP, emit.DEGRADED, t)
    assert dg.get("cross_system", "").startswith("producer_error")
    assert _score_at(st, ERP, IP, t)[0] is None


def test_profile_lists_systems_accessed():
    st, fp = _erp_world(bridge=True)
    eng = X.CrossSystemEngine()
    _history(st, eng, 3, (9, 18), fp)
    ex = st.profile(ERP, "10.99.0.1").extra["systems_accessed"]
    assert set(ex) == {ERP, OA} and ex[ERP]["hours"] > 10


def test_registry_position_and_detector_entry():
    from app.pipeline.build import build_registry
    names = [e.name for e in build_registry().by_layer("behavior")]
    i = names.index("behavior.cross_system")
    assert names[i - 1] == "behavior.class_monitor" and names[i + 1] == "behavior.feedback"
    info = DET.DETECTOR_INFO["cross_system"]
    assert (info["family"], info["kind"], info["owner"], info["stream"]) == \
        ("xsys", "inst", "B21", "t")
    assert "cross_system" in DET.INSTANT_DETECTORS


def test_empty_store_and_idle_pairs():
    st = make_store()
    eng = X.CrossSystemEngine()
    assert run_engine(eng, st, T0, dt=H) == 0
    st.register_entity(ERP, IP)
    st.add_vec(ERP, IP, "feature.active", T0 + H, np.array([0.0], dtype=np.float32))
    assert run_engine(eng, st, T0 + H, dt=H) == 0
    assert _score_at(st, ERP, IP, T0 + H)[0] is None


def test_pair_checkpoint_roundtrip_and_replay_dedupe():
    st = MX.new_pair_state()
    rows = [(T0 + i * H, T0 + i * H, 20000 + i // 24) for i in range(60)]
    for r in rows:
        st = MX.pair_update(st, r, 1.0)
    back = MX.load_pair(MX.dump_pair(st))
    assert back["wins"] == st["wins"] and back["days"] == st["days"]
    assert back["c"] == st["c"] and back["n"] == 60
    # a replayed / released window is not counted twice
    again = MX.pair_update(back, rows[-1], 1.0)
    assert again["n"] == 60 and again["c"] == back["c"]
    # an older released row decays into the state
    st2 = MX.pair_update(MX.new_pair_state(), rows[10], 1.0)
    st2 = MX.pair_update(st2, rows[0], 1.0)
    assert st2["t"] == rows[10][0] and 1.0 < st2["c"] < 2.0


def test_class_tier_is_the_role_in_the_home_system():
    """B02 role ids are org-wide: one role 'human' holds the interactive users
    of erp-prod AND of oa-portal, each confined to its own system. The class
    tier of an oa-portal user must be that role IN oa-portal; pooled over
    systems it reads 'the class uses erp-prod half of the time' and an
    oa-portal user's first erp-prod access is not class-rare (pack D T20:
    only api-gateway was reported, erp-prod p 1.2e-3 instead of ~1e-6)."""
    st = make_store()
    oa = [f"10.30.2.{i}" for i in range(21, 27)]
    roles = {(ERP, ip): "human" for ip in ERP_CLASS}
    roles.update({(OA, ip): "human" for ip in oa})
    put_model(st, *ORG, "model.class", _class_model(roles))
    fp = {ip: [ERP] for ip in ERP_CLASS}
    fp.update({ip: [OA] for ip in oa})
    eng = X.CrossSystemEngine()
    now, known = _history(st, eng, 20, (9, 18), fp)
    user = oa[0]
    t = now + 10 * H
    _tick(st, eng, t, {(OA, user), (ERP, user)}, training=False, known=known)
    sc, pm = _score_at(st, ERP, user, t)
    assert pm is not None and pm < 1e-4, pm
    ev = _events(st, ERP, user)
    assert len(ev) == 1 and ev[0].extra["tier"] in ("class", "org")
    assert ev[0].extra["class"] == f"{OA}|class:human"
