"""B23 FeedbackEngine: spec unit tests (engines.md B23 a-c) plus edge cases.

The engine only talks to the store, so incidents, labels and the evidence
series B23 folds (behavior.p_family / p / z / e_day) are written directly.
"""
import math
import time

import numpy as np
import pytest

from helpers import DT, T0, make_store, run_engine

from app.engines.behavior.feedback import FeedbackEngine, fit_stacker, pav
from app.engines.behavior.lib import m_feedback as FB
from app.engines.behavior.lib.classkeys import ORG
from app.engines.behavior.lib.detectors import DETECTOR_INDEX, FAMILIES, N_DETECTORS
from app.engines.behavior.lib.features import FEATURE_DIM, FEATURE_INDEX
from app.models.schema import (BehaviorEvent, DerivedMetric, Incident, Label, MetricKind,
                               Severity)

S, E, E2 = "erp", "10.0.0.1", "10.0.0.2"
DAY = 86400.0


# ----------------------------------------------------------------- fixtures
def model(store):
    return store.get_model(ORG[0], ORG[1], FB.MODEL)


def put_z(store, ent, t, feats, system=S):
    row = np.zeros(FEATURE_DIM, dtype=np.float32)
    for n, z in feats.items():
        row[FEATURE_INDEX[n]] = z
    store.add_vec(system, ent, "behavior.z", t, row)


def put_pfam(store, ent, t, pf, dt=DT, system=S):
    store.add_derived(DerivedMetric(name="behavior.p_family", value=dict(pf), ts=t, system=system,
                                    entity=ent, window_s=int(dt), kind=MetricKind.CATEGORICAL))


def p_at(e_day, dt=DT):
    """p whose e_day at cadence dt is e_day."""
    return e_day * dt / DAY


def mk_inc(store, ent, t, kinds=("alarm",), axes=("volume",), e_day=1e-3, risk=10.0,
           sev=Severity.MEDIUM, opened=None, system=S, status="open"):
    inc = Incident(system=system, entity=ent, kinds=list(kinds), axes=list(axes),
                   opened=t if opened is None else opened, last_seen=t, e_day_min=e_day,
                   risk=risk, severity=sev, status=status)
    store.put_incident(inc)
    return inc


def label(store, target_id, verdict, scope="this", target_type="incident", ts=None,
          system=S, entity=E, **kw):
    lb = Label(system=system, entity=entity, target_type=target_type, target_id=target_id,
               verdict=verdict, scope=scope, ts=T0 if ts is None else ts, **kw)
    store.add_label(lb)
    return lb


BACKUP = {"bytes_up": 5.0, "flows": 4.0, "http_requests": 3.5}


def _labelled_pattern_store(scope="pattern", e_day=1e-3):
    """Incident A (volume alarm, backup-like z) labelled fp with `scope`."""
    st = make_store()
    st.register_entity(S, E)
    st.register_entity(S, E2)
    eng = FeedbackEngine()
    t = T0
    put_z(st, E, t, BACKUP)
    a = mk_inc(st, E, t, e_day=e_day)
    run_engine(eng, st, t + DT)                      # snapshot the live incident
    label(st, a.id, "fp", scope=scope, ts=t + DT)
    run_engine(eng, st, t + 2 * DT)
    return st, eng


def _new_incident(st, ent, t, e_day=1e-3, kinds=("alarm",), axes=("volume",), feats=BACKUP,
                  system=S):
    put_z(st, ent, t, feats, system=system)
    return mk_inc(st, ent, t, kinds=kinds, axes=axes, e_day=e_day, system=system)


# ======================================================== spec unit tests
def test_a_pattern_policy_suppresses_identical_but_not_rarer_or_other_kind():
    st, _ = _labelled_pattern_store()
    pols = model(st)["policies"]
    assert len(pols) == 1
    assert pols[0]["level"] == pytest.approx(1e-4)
    assert {"a:volume", "f:bytes_up+", "f:flows+", "f:http_requests+"} <= set(pols[0]["tokens"])

    t = T0 + 20 * DT
    same = _new_incident(st, E, t, e_day=1e-3)
    m = FB.suppression_match(st, same)
    assert m is not None and m["similarity"] >= FB.JACCARD_MIN
    assert FB.is_suppressed(st, same)

    rarer = _new_incident(st, E, t + DT, e_day=1e-3 / 10 ** 1.5)        # 1.5 decades rarer
    assert FB.suppression_match(st, rarer) is None

    other_kind = _new_incident(st, E, t + 2 * DT, kinds=("beacon",), axes=("c2",))
    assert FB.suppression_match(st, other_kind) is None
    # same kind but a new axis the analyst never saw also escapes
    new_axis = _new_incident(st, E, t + 3 * DT, axes=("volume", "exfil"))
    assert FB.suppression_match(st, new_axis) is None
    # same kind, unrelated features: Jaccard too low
    other_feats = _new_incident(st, E, t + 4 * DT,
                                feats={"dns_queries": 6.0, "dns_txt_ratio": 5.0,
                                       "dns_qname_len": 4.0})
    assert FB.suppression_match(st, other_feats) is None


def test_a_policy_level_is_one_decade_boundary_and_ttl_14d():
    st, _ = _labelled_pattern_store()
    t = T0 + 30 * DT
    edge = _new_incident(st, E, t, e_day=1.0001e-4)
    assert FB.suppression_match(st, edge) is not None
    below = _new_incident(st, E, t + DT, e_day=0.99e-4)
    assert FB.suppression_match(st, below) is None
    later = _new_incident(st, E, T0 + 15 * DAY)                           # beyond the 14 d TTL
    assert FB.suppression_match(st, later) is None
    # the hourly sweep drops the expired policy from the model
    run_engine(FeedbackEngine(), st, T0 + 15 * DAY + DT)
    assert model(st)["policies"] == []


def test_a_scope_pattern_is_system_wide_entity_scope_is_not():
    st, _ = _labelled_pattern_store(scope="pattern")
    other = _new_incident(st, E2, T0 + 10 * DT)
    assert FB.suppression_match(st, other) is not None
    elsewhere = _new_incident(st, E, T0 + 11 * DT, system="crm")
    assert FB.suppression_match(st, elsewhere) is None

    st2, _ = _labelled_pattern_store(scope="entity")
    assert FB.suppression_match(st2, _new_incident(st2, E, T0 + 10 * DT)) is not None
    assert FB.suppression_match(st2, _new_incident(st2, E2, T0 + 11 * DT)) is None

    st3, _ = _labelled_pattern_store(scope="this")
    assert model(st3)["policies"] == []


def test_a_unknown_level_never_suppresses_and_unknown_ref_is_low_level():
    st, _ = _labelled_pattern_store()
    t = T0 + 40 * DT
    put_z(st, E, t, BACKUP)
    no_eday = mk_inc(st, E, t, e_day=None)
    assert FB.suppression_match(st, no_eday) is None                      # cannot verify level
    # e_day read from behavior.e_day when the incident carries none
    st.add_vec(S, E, "behavior.e_day", t, np.array([2e-3], dtype=np.float32))
    assert FB.suppression_match(st, no_eday) is not None

    # a labelled incident without any e_day evidence gets level 0.03 / 10
    st2 = make_store()
    eng = FeedbackEngine()
    put_z(st2, E, T0, BACKUP)
    a = mk_inc(st2, E, T0, e_day=None)
    run_engine(eng, st2, T0 + DT)
    label(st2, a.id, "benign_known", scope="pattern", ts=T0 + DT)
    run_engine(eng, st2, T0 + 2 * DT)
    assert model(st2)["policies"][0]["level"] == pytest.approx(3e-3)


def _stack_store(n_seq_fp=12, n_tp=12, n_other_fp=6, seed=0):
    rng = np.random.default_rng(seed)
    st = make_store()
    eng = FeedbackEngine()
    t = T0
    incs = []
    specs = ([("seq", 0.0)] * n_seq_fp + [("tp", 1.0)] * n_tp + [("ofp", 0.0)] * n_other_fp)
    for i, (kind, _) in enumerate(specs):
        ent = f"10.1.0.{i}"
        st.register_entity(S, ent)
        if kind == "seq":
            pf = {"sequence": p_at(10 ** -rng.uniform(2, 5))}
            if rng.random() < 0.3:
                pf["shape"] = p_at(10 ** -rng.uniform(1.6, 2.5))
        elif kind == "tp":
            fams = rng.choice(["intensity", "categorical", "exfil", "c2"], size=2, replace=False)
            pf = {f: p_at(10 ** -rng.uniform(2, 5)) for f in fams}
        else:
            fams = rng.choice(["intensity", "shape", "temporal"], size=1)
            pf = {f: p_at(10 ** -rng.uniform(1.6, 3)) for f in fams}
        put_pfam(st, ent, t + i * DT, pf)
        incs.append((mk_inc(st, ent, t + i * DT, axes=["sequence" if kind == "seq" else "volume"],
                            e_day=1e-3), kind))
    now = t + (len(specs) + 1) * DT
    run_engine(eng, st, now)
    for inc, kind in incs:
        label(st, inc.id, "tp" if kind == "tp" else "fp", ts=now, entity=inc.entity)
    run_engine(eng, st, now + DT)
    return st, eng, now + DT


def test_b_sequence_always_fp_weight_below_half_uniform():
    st, _, _ = _stack_store()
    m = model(st)
    assert m["n_labelled"] == 30 and m["stacker"] is not None
    w = FB.family_weights(st)
    assert set(w) == set(FAMILIES)
    assert w["sequence"] < 0.5 * 1.0                                    # uniform = 1.0
    assert w["sequence"] < 0.5 * float(np.mean(list(w.values())))
    assert w["sequence"] >= FB.W_FLOOR
    assert max(w["intensity"], w["categorical"], w["exfil"], w["c2"]) > 1.0
    # Beta precision: sequence never tp -> pi at its floor; tp-heavy families
    # are precise but their pi stays at 1 (feedback never raises the evidence
    # of every entity above the calibrated null; evaluator round 4, gate 12)
    assert FB.risk_mult(st, "sequence") == pytest.approx(0.2)
    assert FB.precision(st, "sequence")[0] == pytest.approx(1 / 14)
    assert FB.precision(st, "exfil")[0] > 0.5
    assert FB.risk_mult(st, "exfil") == 1.0
    assert FB.risk_mult(st, "xsys") == 1.0                               # never labelled


def test_b_below_20_labels_weights_stay_uniform_but_precision_learns():
    st, _, _ = _stack_store(n_seq_fp=6, n_tp=6, n_other_fp=3)
    m = model(st)
    assert m["n_labelled"] == 15 and m["stacker"] is None
    assert all(v == 1.0 for v in FB.family_weights(st).values())
    assert FB.risk_mult(st, "sequence") < 0.5


def test_c_expected_change_in_accept_within_one_tick():
    st = make_store()
    st.register_entity(S, E)
    eng = FeedbackEngine()
    inc = mk_inc(st, E, T0)
    run_engine(eng, st, T0 + DT)
    label(st, inc.id, "expected_change", scope="entity", ts=T0 + DT, t0=T0 - 4 * DT)
    run_engine(eng, st, T0 + 2 * DT)                                     # the very next tick
    acc = model(st)["accept"]
    assert f"{S}|{E}" in acc and acc[f"{S}|{E}"][0]["t0"] == T0 - 4 * DT
    rec = FB.latest_accept(st, S, E)
    assert rec is not None and rec["ts"] == T0 + 2 * DT and rec["target_id"] == inc.id
    assert FB.accepts(st, S, E, since=T0 + 3 * DT) == []
    assert not FB.is_frozen(st, S, E)
    prof = st.profile(S, E)
    assert prof.extra["feedback"]["last_verdict"] == "expected_change"


# ============================================================ other behaviour
def test_tp_freezes_and_class_scope_reaches_members():
    st = make_store()
    for ip in ("10.0.0.1", "10.0.0.2", "10.0.0.3"):
        st.register_entity(S, ip)
    st.put_model(ORG[0], ORG[1], "model.class", {"assign": {
        f"{S}|10.0.0.{i}": {"role": "r1", "prob": 0.9} for i in (1, 2, 3)}})
    eng = FeedbackEngine()
    label(st, "", "tp", scope="entity", target_type="entity", ts=T0)
    label(st, "", "expected_change", scope="class", target_type="entity", ts=T0,
          entity="10.0.0.2")
    run_engine(eng, st, T0)
    assert FB.is_frozen(st, S, E)
    assert not FB.is_frozen(st, S, "10.0.0.3")
    assert f"{S}|class:r1" in model(st)["accept"]
    rec = FB.latest_accept(st, S, "10.0.0.3")                           # member, via class tier
    assert rec is not None and rec["key"] == f"{S}|class:r1"
    # a later expected_change on the frozen entity lifts the freeze
    label(st, "", "expected_change", scope="entity", target_type="entity", ts=T0 + DT)
    run_engine(eng, st, T0 + DT)
    assert not FB.is_frozen(st, S, E)


def test_benign_known_allowlists_new_values_by_scope():
    st = make_store()
    st.register_entity(S, E)
    st.register_entity(S, E2)
    eng = FeedbackEngine()
    ev = BehaviorEvent(system=S, entity=E, ts=T0, kind="first_seen", score=0.9,
                       extra={"dim": "sni", "value": "backup.example.com"}, e_day=1e-3)
    st.add_event(ev)
    label(st, ev.id, "benign_known", scope="entity", target_type="event", ts=T0)
    run_engine(eng, st, T0)
    assert FB.allowlisted(st, S, E, "sni", "backup.example.com")
    assert not FB.allowlisted(st, S, E2, "sni", "backup.example.com")
    assert not FB.allowlisted(st, S, E, "dns", "backup.example.com")
    assert FB.allowlist_values(st, S, E, "sni") == {"backup.example.com"}
    # event-level benign_known with scope entity also makes a pattern policy
    assert model(st)["policies"][0]["scope"] == "entity"

    ev2 = BehaviorEvent(system=S, entity=E, ts=T0 + DT, kind="rare_access", score=0.9,
                        extra={"dim": "template", "value": "GET erp /a?x={num}"})
    st.add_event(ev2)
    label(st, ev2.id, "benign_known", scope="system", target_type="event", ts=T0 + DT,
          ttl_s=3600.0)
    run_engine(eng, st, T0 + DT)
    assert FB.allowlisted(st, S, E2, "template", "GET erp /a?x={num}", now=T0 + 2 * DT)
    assert not FB.allowlisted(st, S, E2, "template", "GET erp /a?x={num}", now=T0 + 3 * 3600)


def test_alert_budget_alpha_daily_with_entity_cap_and_opt_in_ceiling():
    cfg = {"alert_budget": {"system_per_day": 2, "entity_per_hour": 3}}
    st = make_store()
    st.register_entity(S, E)
    eng = FeedbackEngine()
    run_engine(eng, st, T0, config=cfg)                                  # starts the daily clock
    for i in range(5):                                                   # 5 entities x 2
        for j in range(2):
            mk_inc(st, f"10.2.0.{i}", T0 + (1 + 2 * i + j) * DT, sev=Severity.LOW)
    for j in range(30):                                                  # one noisy entity: capped at 3
        mk_inc(st, "10.9.9.9", T0 + (20 + j) * DT, sev=Severity.MEDIUM)
    mk_inc(st, "10.3.0.1", T0 + 60 * DT, status="suppressed")            # not counted
    mk_inc(st, "10.3.0.2", T0 + 61 * DT, sev=Severity.INFO)              # not counted
    run_engine(eng, st, T0 + DAY - DT, config=cfg)
    assert FB.alpha_mult(st, S) == 1.0                                   # < 1 day: no update
    run_engine(eng, st, T0 + DAY, config=cfg)
    assert FB.alpha_mult(st, S) == pytest.approx(math.exp(0.1 * (2 - 13)))
    a1 = FB.alpha_mult(st, S)
    run_engine(eng, st, T0 + 2 * DAY, config=cfg)                        # quiet day: relaxes
    assert FB.alpha_mult(st, S) == pytest.approx(min(1.0, a1 * math.exp(0.2)))
    for _ in range(6):
        run_engine(eng, st, T0 + (3 + _) * DAY, config=cfg)
    assert FB.alpha_mult(st, S) == 1.0                                   # default ceiling
    cfg4 = {"alert_budget": {"system_per_day": 2, "entity_per_hour": 3, "alpha_max": 4.0}}
    run_engine(eng, st, T0 + 10 * DAY, config=cfg4)
    assert FB.alpha_mult(st, S) == pytest.approx(math.exp(0.2))
    for k in range(20):                                                  # flood -> floor 0.25
        for j in range(3):
            mk_inc(st, f"10.4.{k}.{j}", T0 + 10 * DAY + (1 + k) * DT)
    run_engine(eng, st, T0 + 11 * DAY, config=cfg4)
    assert FB.alpha_mult(st, S) == pytest.approx(0.25)


def test_label_queue_daily_risk_and_uncertainty_plus_held():
    st = make_store()
    eng = FeedbackEngine()
    run_engine(eng, st, T0)                                              # starts the queue clock
    t = T0 + DT
    # labelled history: intensity mostly tp, categorical mostly fp
    for i, (fam, v) in enumerate([("intensity", "tp")] * 3 + [("categorical", "fp")] * 3):
        ent = f"10.5.0.{i}"
        put_pfam(st, ent, t, {fam: p_at(1e-3)})
        inc = mk_inc(st, ent, t, risk=5.0)
        run_engine(eng, st, t + DT)
        label(st, inc.id, v, ts=t + DT, entity=ent)
    run_engine(eng, st, t + 2 * DT)
    # unlabelled pool
    pool = []
    for i in range(8):
        ent = f"10.6.0.{i}"
        put_pfam(st, ent, t + 3 * DT, {"intensity": p_at(1e-3)})
        pool.append(mk_inc(st, ent, t + 3 * DT, risk=40.0 + i))
    ent = "10.6.1.0"                                                     # P ~ 0.5, low risk
    put_pfam(st, ent, t + 3 * DT, {"intensity": p_at(1e-3), "categorical": p_at(1e-3)})
    unsure = mk_inc(st, ent, t + 3 * DT, risk=1.0)
    held = mk_inc(st, "10.7.0.1", t + 3 * DT, opened=t + 3 * DT - 15 * DAY, risk=2.0)
    run_engine(eng, st, t + 4 * DT)
    q = FB.label_queue(st)
    assert [it["incident_id"] for it in q] == [held.id]                  # held is immediate
    assert q[0]["reason"] == "held"
    run_engine(eng, st, T0 + DAY + DT)                                   # first daily batch
    q = FB.label_queue(st)
    by = {}
    for it in q:
        by.setdefault(it["reason"], []).append(it["incident_id"])
    assert by["held"] == [held.id]
    assert set(by["risk"]) == {p.id for p in pool[-4:]}                  # 80 %: 4 highest risk
    assert by["uncertain"] == [unsure.id]                                # 20 %: |P - 0.5| min
    assert len(by["risk"]) + len(by["uncertain"]) == 5
    run_engine(eng, st, T0 + DAY + 2 * DT)
    assert len(FB.label_queue(st)) == 6                                  # no second batch same day
    label(st, pool[-1].id, "fp", ts=T0 + DAY, entity=pool[-1].entity)
    run_engine(eng, st, T0 + DAY + 3 * DT)
    assert pool[-1].id not in {it["incident_id"] for it in FB.label_queue(st)}


def test_isotonic_after_100_labels_and_p_malicious_monotone():
    rng = np.random.default_rng(3)
    st = make_store()
    eng = FeedbackEngine()
    t = T0
    incs = []
    for i in range(110):
        ent = f"10.8.{i // 250}.{i % 250}"
        mal = rng.random() < 0.4
        s_ex = rng.uniform(2, 6) if mal else rng.uniform(0, 1.2)
        pf = {"exfil": p_at(10 ** -s_ex), "intensity": p_at(10 ** -rng.uniform(1.6, 4))}
        put_pfam(st, ent, t, pf)
        incs.append((mk_inc(st, ent, t), mal))
        t += DT
    run_engine(eng, st, t)
    for inc, mal in incs:
        label(st, inc.id, "tp" if mal else "fp", ts=t, entity=inc.entity)
    t0 = time.perf_counter()
    run_engine(eng, st, t + DT)
    refit_ms = (time.perf_counter() - t0) * 1000
    m = model(st)
    assert m["isotonic"] is not None and m["n_labelled"] == 110
    assert list(m["isotonic"]["x"]) == sorted(m["isotonic"]["x"])
    assert list(m["isotonic"]["y"]) == sorted(m["isotonic"]["y"])
    lo = {"fs": {"exfil": 0.2, "intensity": 2.0}, "stages": 1}
    hi = {"fs": {"exfil": 5.0, "intensity": 2.0}, "stages": 1}
    p_lo, p_hi = FB.p_malicious(st, lo), FB.p_malicious(st, hi)
    assert 0.0 < p_lo < 0.3 and 0.7 < p_hi < 1.0
    assert FB.family_weights(st)["exfil"] > 1.5
    assert refit_ms < 2000                                               # generous (refit only on labels)


# ================================================================ edge cases
def test_empty_store_and_neutral_accessors():
    st = make_store()
    run_engine(FeedbackEngine(), st, T0)
    m = model(st)
    assert m["version"] == 0 and m["policies"] == [] and m["queue"] == []
    assert st.last_write_ts(ORG[0], ORG[1], FB.MODEL) == T0
    fresh = make_store()                                                 # no model at all
    assert FB.family_weights(fresh) == {f: 1.0 for f in FAMILIES}
    assert FB.alpha_mult(fresh, S) == 1.0
    assert FB.risk_mult(fresh, "intensity") == 1.0
    assert FB.suppression_match(fresh, Incident(system=S, entity=E, e_day_min=1e-3)) is None
    assert not FB.allowlisted(fresh, S, E, "sni", "x")
    assert FB.label_queue(fresh) == [] and FB.accepts(fresh, S, E) == []
    assert FB.p_malicious(fresh, {"fs": {"exfil": 5.0}}) == 0.5
    assert FB.summary(fresh, S, E)["n_labelled"] == 0


def test_silent_entity_and_unresolvable_labels():
    st = make_store()
    st.register_entity(S, E)
    eng = FeedbackEngine()
    label(st, "no-such-incident", "fp", scope="pattern", ts=T0)          # target gone
    label(st, "no-such-event", "tp", target_type="event", ts=T0, system="", entity="")
    for i in range(3):
        run_engine(eng, st, T0 + i * DT)
    m = model(st)
    assert m["policies"] == [] and m["n_labelled"] == 0
    assert len(m["_state"]["labels_seen"]) == 2                          # consumed exactly once
    assert FB.is_frozen(st, S, E) is False
    assert FB.summary(st, S, "10.99.0.1")["policies"] == 0


def test_nan_inputs_are_unscored_not_evidence():
    st = make_store()
    eng = FeedbackEngine()
    put_pfam(st, E, T0, {"intensity": float("nan"), "shape": None, "c2": p_at(1e-4)})
    row = np.full(N_DETECTORS, np.nan, dtype=np.float32)
    row[DETECTOR_INDEX["beacon"]] = p_at(1e-4)
    st.add_vec(S, E, "behavior.p", T0, row)
    z = np.full(FEATURE_DIM, np.nan, dtype=np.float32)
    z[FEATURE_INDEX["bytes_up"]] = 4.0
    st.add_vec(S, E, "behavior.z", T0, z)
    st.add_vec(S, E, "behavior.e_day", T0, np.array([np.nan], dtype=np.float32))
    inc = mk_inc(st, E, T0, e_day=None, kinds=("beacon",), axes=("c2",))
    run_engine(eng, st, T0 + DT)
    case = model(st)["_cases"]["inc:" + inc.id]
    assert set(case["fs"]) == {"c2"} and case["fs"]["c2"] == pytest.approx(4.0)
    assert set(case["ds"]) == {"beacon"}
    assert case["z"] == {"bytes_up": 4.0}
    assert case["e_day"] is None                                         # NaN e_day never min()
    assert FB.surprise(float("nan"), DT) is None and FB.surprise(None, DT) is None
    assert FB.surprise(1.0, DT) == 0.0 and FB.surprise(0.0, DT) == FB.S_MAX


def test_training_learns_but_emits_no_events():
    st = make_store()
    st.register_entity(S, E)
    eng = FeedbackEngine()
    inc = mk_inc(st, E, T0)
    run_engine(eng, st, T0 + DT, training=True)
    label(st, inc.id, "expected_change", scope="entity", ts=T0 + DT)
    label(st, "", "tp", target_type="entity", ts=T0 + DT, entity=E2)
    for i in range(2, 200):
        run_engine(eng, st, T0 + i * DT, training=True)
    assert st.events() == []
    assert FB.latest_accept(st, S, E) is not None and FB.is_frozen(st, S, E2)
    assert FB.alpha_mult(st, S) == 1.0 and FB.label_queue(st) == []


def test_cadence_switch_900_to_60():
    st = make_store()
    st.register_entity(S, E)
    eng = FeedbackEngine()
    p = 1e-4
    inc = mk_inc(st, E, T0, e_day=None)
    put_pfam(st, E, T0, {"intensity": p}, dt=900)                         # e_day 9.6e-3
    run_engine(eng, st, T0 + 900, dt=900)
    case = model(st)["_cases"]["inc:" + inc.id]
    assert case["fs"]["intensity"] == pytest.approx(-math.log10(p * DAY / 900))
    # switch to 60 s ticks: the same p is 15x rarer per day at 60 s
    t = T0 + 900 + 60
    put_pfam(st, E, t, {"shape": p}, dt=60)
    inc.last_seen = t
    st.put_incident(inc)
    run_engine(eng, st, t + 60, dt=60)
    case = model(st)["_cases"]["inc:" + inc.id]
    assert case["fs"]["shape"] == pytest.approx(-math.log10(p * DAY / 60))
    assert case["fs"]["intensity"] == pytest.approx(-math.log10(p * DAY / 900))
    # wall-clock daily clocks: 96 ticks of 60 s are not a day
    for i in range(2, 100):
        run_engine(eng, st, t + i * 60, dt=60)
    assert model(st)["_state"]["alpha_ts"][S] == T0 + 900
    label(st, inc.id, "tp", ts=t)
    run_engine(eng, st, t + 101 * 60, dt=60)
    assert FB.is_frozen(st, S, E)


def test_relabel_latest_verdict_wins_and_scheduled_strict_run():
    st = make_store()
    eng = FeedbackEngine()
    put_pfam(st, E, T0, {"exfil": p_at(1e-4)})
    inc = mk_inc(st, E, T0)
    run_engine(eng, st, T0 + DT, scheduled=True)
    label(st, inc.id, "tp", ts=T0 + DT)
    run_engine(eng, st, T0 + 2 * DT, scheduled=True)
    assert model(st)["detector_prec"]["exfil|*"] == [1.0, 0.0]
    label(st, inc.id, "unsure", ts=T0 + 2 * DT)
    run_engine(eng, st, T0 + 3 * DT, scheduled=True)
    assert "exfil|*" not in model(st)["detector_prec"] and model(st)["n_labelled"] == 0
    assert eng.error_count == 0 and st.health()["behavior.feedback"]["ok"]


def test_deterministic_models():
    a, _, _ = _stack_store(seed=7)
    b, _, _ = _stack_store(seed=7)
    assert model(a)["family_w"] == model(b)["family_w"]
    assert model(a)["stacker"]["coef"] == model(b)["stacker"]["coef"]


def test_stacker_and_pav_maths():
    X = np.zeros((40, FB.N_STACK))
    y = np.r_[np.ones(20), np.zeros(20)]
    th = fit_stacker(X, y)                                              # no evidence: uniform
    beta = th[1:len(FAMILIES) + 1]
    assert np.allclose(np.logaddexp(0, beta), 1.0) and np.all(np.isfinite(th))
    th0 = fit_stacker(X, np.zeros(40))                                   # one-class: finite
    assert np.all(np.isfinite(th0)) and th0[0] < -3
    xs, ys = pav(np.array([0.1, 0.2, 0.2, 0.3, 0.9]), np.array([0, 1, 0, 0, 1.0]))
    assert xs == sorted(xs) and ys == sorted(ys) and len(xs) == len(set(xs))
    assert ys[-1] == 1.0


def test_perf_many_live_incidents_and_labels():
    st = make_store()
    eng = FeedbackEngine()
    rng = np.random.default_rng(1)
    incs = []
    for i in range(40):
        ent = f"10.10.0.{i}"
        st.register_entity(S, ent)
        incs.append(mk_inc(st, ent, T0))
    run_engine(eng, st, T0)
    for i, inc in enumerate(incs[:30]):                                  # 30 labels -> stacker
        label(st, inc.id, "tp" if i % 3 == 0 else "fp", scope="pattern", ts=T0,
              entity=inc.entity)
    times = []
    for k in range(1, 61):
        t = T0 + k * DT
        for inc in incs[20:]:                                            # 20 incidents stay live
            ent = inc.entity
            put_pfam(st, ent, t, {"intensity": p_at(10 ** -rng.uniform(0, 3))})
            row = np.full(N_DETECTORS, 0.5, dtype=np.float32)
            st.add_vec(S, ent, "behavior.p", t, row)
            put_z(st, ent, t, {"bytes_up": float(rng.normal(0, 2))})
            inc.last_seen = t
            st.put_incident(inc)
        t0 = time.perf_counter()
        run_engine(eng, st, t + 1)
        times.append((time.perf_counter() - t0) * 1000)
    steady = float(np.median(times[5:]))
    assert steady < 25.0, steady                                         # ~20 live incidents
    # the org-level steady state with no live incident is well under 1 ms
    quiet = []
    for k in range(20):
        t0 = time.perf_counter()
        run_engine(eng, st, T0 + (100 + k) * DT)
        quiet.append((time.perf_counter() - t0) * 1000)
    assert float(np.median(quiet)) < 2.0


def test_profile_feedback_summary_counts_the_label_just_applied():
    """W7: profile.extra.feedback was written before the refit, so after the
    first label its summary said n_labelled = 0 while n_labels = 1."""
    st = make_store()
    st.register_entity(S, E)
    eng = FeedbackEngine()
    t = T0
    put_z(st, E, t, BACKUP)
    a = mk_inc(st, E, t)
    run_engine(eng, st, t + DT)
    label(st, a.id, "tp", ts=t + DT)
    run_engine(eng, st, t + 2 * DT)
    fb = st.profile(S, E).extra["feedback"]
    assert fb["n_labels"] == 1 and fb["last_verdict"] == "tp"
    assert fb["n_labelled"] == 1 == model(st)["n_labelled"]
