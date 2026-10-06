"""Eval metrics on hand-built runs: TP / TTD / FAR edge cases, exclusion
windows, alias / actor / class matching, range-based P/R/F1, calibration
counters, gates and report rendering. No pipeline, no engines."""
import json
import math

import numpy as np
import pytest

from app.eval import metrics as M
from app.eval.report import build_report, render_html, write_report
from app.eval.runner import RunResult
from app.engines.behavior.lib.detectors import DETECTORS, FAMILIES

S = "erp-prod"
DT = 900.0
T0 = 1_700_000_000.0            # scenario phase start (first tick at T0 + DT)
N_TICKS = 96 * 2                # two days
TICKS = T0 + DT * np.arange(1, N_TICKS + 1)
T_END = TICKS[-1]


def k(e):
    return f"{S}|{e}"


def truth(sid, ents, t_start, t_end, label="malicious", **kw):
    row = {"scenario_id": sid, "pack": "A", "system": S, "entities": list(ents),
           "t_start": t_start, "t_end": t_end, "label": label,
           "expected_detectors": [], "expected_axes": [], "perturbed_features": []}
    if label == "malicious":
        row["required_severity"] = "medium"
    else:
        row["max_allowed_severity"] = "info"
    row.update(kw)
    return row


def inc(iid, ent, opened, sev="medium", axes=("volume",), history=None, **kw):
    d = {"id": iid, "system": S, "entity": ent, "entities": [], "kinds": [], "axes": list(axes),
         "status": "open", "opened": opened, "last_seen": opened, "severity": sev,
         "evidence": [], "explanation": {}, "narrative": ""}
    if history is not None:
        d["history"] = history
    d.update(kw)
    return d


def ev(eid, ent, ts, kind, **kw):
    d = {"id": eid, "system": S, "entity": ent, "ts": ts, "kind": kind, "severity": "info",
         "extra": {}, "axes": [], "p_by_detector": {}, "incident_id": ""}
    d.update(kw)
    return d


def personas(*ents, arch="interactive"):
    return {k(e): {"archetype": arch, "system": S, "entity": e} for e in ents}


def make_run(**kw):
    base = dict(pack="A", seed=0, scenario_dt=DT, scenario_window=(T0, T_END), tick_ts=TICKS,
                personas=personas("10.20.1.11", "10.20.1.12", "10.20.1.13", "10.20.1.15"))
    base.update(kw)
    return RunResult(**base)


# --------------------------------------------------------------------------- detection
def test_tp_ttd_counts_onset_tick():
    ts = TICKS[10]
    run = make_run(truth=[truth("T1", ["10.20.1.11"], ts, ts + 4 * DT, expected_axes=["volume"])],
                   incidents=[inc("i1", "10.20.1.11", ts + DT, "high")])
    o = M.score_run(run)["scenarios"][0]
    assert o["detected"] and o["tp_incident"] == "i1"
    assert o["ttd_s"] == DT and o["ttd_ticks"] == 2
    assert o["loudness"] == "loud" and o["within_deadline"]      # <= 2 ticks at 900 s


def test_loud_third_tick_misses_deadline_but_is_detected():
    ts = TICKS[10]
    run = make_run(truth=[truth("T1", ["10.20.1.11"], ts, ts + 8 * DT)],
                   incidents=[inc("i1", "10.20.1.11", ts + 2 * DT, "high")])
    o = M.score_run(run)["scenarios"][0]
    assert o["detected"] and o["ttd_ticks"] == 3 and not o["within_deadline"]


def test_severity_history_escalation_sets_detection_time():
    ts = TICKS[20]
    hist = [{"ts": ts, "severity": "low", "status": "open", "axes": ["volume"], "kinds": []},
            {"ts": ts + 3 * DT, "severity": "medium", "status": "open", "axes": ["volume"],
             "kinds": []}]
    run = make_run(truth=[truth("T3", ["10.20.1.12"], ts, ts + 40 * DT, max_ttd=7200)],
                   incidents=[inc("i1", "10.20.1.12", ts, "medium", history=hist)])
    o = M.score_run(run)["scenarios"][0]
    assert o["detected"] and o["t_detect"] == ts + 3 * DT and o["ttd_ticks"] == 4
    assert o["within_deadline"]                                   # 2700 s <= 7200 s
    run.truth[0]["max_ttd"] = "2 ticks"
    assert not M.score_run(run)["scenarios"][0]["within_deadline"]


def test_below_required_severity_is_not_tp():
    ts = TICKS[20]
    run = make_run(truth=[truth("T3", ["10.20.1.12"], ts, ts + 4 * DT)],
                   incidents=[inc("i1", "10.20.1.12", ts, "low")])
    assert not M.score_run(run)["scenarios"][0]["detected"]


@pytest.mark.parametrize("offset,expect", [(-DT, False), (0.0, True), (4 * DT, True),
                                           (4 * DT + 1.0, False)])
def test_tp_window_edges(offset, expect):
    # window = [t_start, t_end + max(4 ticks, 1 h)] = [t_start, t_end + 3600]
    ts, te = TICKS[20], TICKS[30]
    opened = (ts if offset <= 0 else te) + offset
    run = make_run(truth=[truth("T4", ["10.20.1.12"], ts, te)],
                   incidents=[inc("i1", "10.20.1.12", opened, "medium")])
    assert M.score_run(run)["scenarios"][0]["detected"] is expect


def test_axes_or_detectors_must_match_expected():
    ts = TICKS[20]
    row = truth("T7", ["10.20.1.13"], ts, ts + 10 * DT, expected_detectors=["B08 novelty"],
                expected_axes=["privilege"])
    run = make_run(truth=[row], incidents=[inc("i1", "10.20.1.13", ts, "medium", axes=["volume"])])
    assert not M.score_run(run)["scenarios"][0]["detected"]
    # detector via the incident's evidence p_by_detector
    run.incidents[0]["evidence"] = [{"ts": ts, "p_by_detector": {"novelty": 1e-6}}]
    o = M.score_run(run)["scenarios"][0]
    assert o["detected"] and o["matched"] == ["B08 novelty"]


def test_discrete_event_on_scenario_entity_counts_as_hit():
    ts = TICKS[20]
    row = truth("T9", ["10.20.1.13"], ts, ts + 10 * DT, expected_detectors=["B16"])
    run = make_run(truth=[row], incidents=[inc("i1", "10.20.1.13", ts + DT, "high",
                                                axes=["volume"])],
                   events=[ev("e1", "10.20.1.13", ts, "identity_mismatch")])
    assert M.score_run(run)["scenarios"][0]["detected"]      # B16 owns identity_mismatch
    run.events[0]["entity"] = "10.20.1.11"                   # event on another entity
    assert not M.score_run(run)["scenarios"][0]["detected"]


def test_continuity_alias_and_actor_chain_matching():
    ts = TICKS[20]
    row = truth("T2", ["10.20.1.12"], ts, ts + 10 * DT)
    run = make_run(truth=[row], incidents=[inc("i1", "10.20.1.112", ts + DT)])
    assert not M.score_run(run)["scenarios"][0]["detected"]
    run.profiles = {k("10.20.1.112"): {"extra": {"continuity": {"aliases": ["10.20.1.12"]}}}}
    assert M.score_run(run)["scenarios"][0]["detected"]

    row = truth("T19", ["10.20.1.70", "10.20.1.71", "10.20.1.72"], ts, ts + 90 * DT)
    run = make_run(truth=[row], incidents=[inc("i1", "10.20.1.99", ts + DT)],
                   models={"link": {S: {"actors": {"a1": {"members": ["10.20.1.99",
                                                                      "10.20.1.71"]}}}}})
    o = M.score_run(run)["scenarios"][0]
    assert o["detected"] and o["n_incidents"] == 1          # actor scenario: 'any'


def test_class_scenario_matches_class_key_and_multi_entity_requires_all():
    ts = TICKS[20]
    members = ["10.20.1.11", "10.20.1.12", "10.20.1.13"]
    assign = {k(e): {"role": "r1", "sub": "s", "prob": 1.0} for e in members}
    run = make_run(truth=[truth("T21", members, ts, ts + 10 * DT)],
                   incidents=[inc("c1", "class:r1", ts + DT)],
                   models={"class": {"assign": assign}})
    assert M.score_run(run)["scenarios"][0]["detected"]
    # T11-style independent attackers: every entity must be detected
    run = make_run(truth=[truth("T11", ["10.30.2.99", "10.20.7.77"], ts, ts + 10 * DT)],
                   incidents=[inc("i1", "10.30.2.99", ts)])
    o = M.score_run(run)["scenarios"][0]
    assert not o["detected"] and o["per_entity"] == {k("10.30.2.99"): True,
                                                      k("10.20.7.77"): False}
    run.incidents.append(inc("i2", "10.20.7.77", ts + DT))
    o = M.score_run(run)["scenarios"][0]
    assert o["detected"] and o["ttd_s"] == DT and o["incidents_per_episode"] == 1.0


def test_notifications_from_incident_events_and_history_fallback():
    ts = TICKS[20]
    hist = [{"ts": ts, "severity": "low"}, {"ts": ts + DT, "severity": "medium"},
            {"ts": ts + 2 * DT, "severity": "medium"}, {"ts": ts + 3 * DT, "severity": "high"}]
    run = make_run(truth=[truth("T3", ["10.20.1.12"], ts, ts + 10 * DT)],
                   incidents=[inc("i1", "10.20.1.12", ts, "high", history=hist)])
    assert M.score_run(run)["scenarios"][0]["notifications"] == 3
    run.events = [ev(f"e{j}", "10.20.1.12", ts + j * DT, "incident", incident_id="i1",
                     extra={"state": st}) for j, st in enumerate(["open", "update", "escalate"])]
    assert M.score_run(run)["scenarios"][0]["notifications"] == 2


# --------------------------------------------------------------------------- FAR
def test_far_control_entities_exclusions_and_thresholds():
    ts = TICKS[40]
    assign = {k(e): {"role": "r1", "prob": 1.0} for e in ("10.20.1.11", "10.20.1.12", "10.20.1.13")}
    assign[k("10.20.1.15")] = {"role": "r2", "prob": 1.0}
    rows = [truth("L1", ["10.20.1.11"], ts, ts + 4 * DT, label="legit_change")]
    incs = [inc("a", "10.20.1.12", ts + 2 * DT, "high"),        # inside L1 class exclusion
            inc("b", "10.20.1.13", TICKS[2], "low"),            # counted
            inc("c", "10.20.1.15", TICKS[3], "medium"),         # other class: counted
            inc("d", "10.20.1.11", TICKS[3], "critical"),       # scenario entity: not control
            inc("e", "class:r1", TICKS[3], "medium")]           # class incident: not entity FAR
    run = make_run(truth=rows, incidents=incs, models={"class": {"assign": assign}})
    far = M.score_run(run)["far"]
    assert far["n_control"] == 3
    assert (far["n_low"], far["n_medium"], far["n_high"], far["n_critical"]) == (2, 1, 0, 0)
    days = 3 * (T_END - T0) / 86400.0 - 2 * (4 * DT + 3600 + 86400) / 86400.0
    assert far["entity_days"] == pytest.approx(days)
    assert far["far_low"] == pytest.approx(2 / days)


def test_system_change_excludes_all_system_entities():
    ts = TICKS[40]
    rows = [truth("L10", ["10.20.1.11"], ts, ts + 3 * DT, label="system_change")]
    run = make_run(truth=rows, incidents=[inc("a", "10.20.1.13", ts + DT, "low")])
    assert M.score_run(run)["far"]["n_low"] == 0


def test_legit_outcome_severity_and_risk():
    ts = TICKS[40]
    row = truth("L5", ["10.20.1.12"], ts, ts + 10 * DT, label="legit_change",
                max_allowed_severity="low")
    run = make_run(truth=[row], incidents=[inc("a", "10.20.1.12", ts + DT, "low")])
    lg = M.score_run(run)["legit"][0]
    assert lg["ok"] and lg["max_severity"] == "low"
    run.incidents[0]["severity"] = "medium"
    assert not M.score_run(run)["legit"][0]["sev_ok"]
    run.incidents = []
    run.series = {k("10.20.1.12"): {"ts": TICKS, "risk": np.where(TICKS > ts, 35.0, 5.0)}}
    lg = M.score_run(run)["legit"][0]
    assert lg["sev_ok"] and not lg["risk_ok"] and lg["max_risk"] == 35.0


def test_legit_member_severity_for_class_changes():
    # L1: one class incident <= LOW is allowed, member incidents >= LOW are not
    ts = TICKS[40]
    row = truth("L1", ["10.20.1.11", "10.20.1.12"], ts, ts + 8 * DT, label="legit_change",
                max_allowed_severity="low", max_member_severity="info", level="class")
    run = make_run(truth=[row], incidents=[inc("c", "class:r1", ts + DT, "low")],
                   models={"class": {"assign": {k("10.20.1.11"): {"role": "r1"},
                                                k("10.20.1.12"): {"role": "r1"}}}})
    lg = M.score_run(run)["legit"][0]
    assert lg["ok"] and lg["n_class_incidents"] == 1 and lg["member_max_severity"] is None
    run.incidents.append(inc("m", "10.20.1.12", ts + 2 * DT, "low"))
    lg = M.score_run(run)["legit"][0]
    assert lg["sev_ok"] and not lg["member_ok"] and not lg["ok"]


# --------------------------------------------------------------------------- ranges
A = frozenset({"x"})
B = frozenset({"y"})


def test_range_prf_flat_and_cardinality():
    r = M.range_prf([(A, 0.0, 10.0)], [(A, 5.0, 15.0)])
    assert r["recall"] == pytest.approx(0.5) and r["precision"] == pytest.approx(0.5)
    assert r["f1"] == pytest.approx(0.5)
    # two predictions inside one real range: cardinality factor 1/2
    r = M.range_prf([(A, 0.0, 10.0)], [(A, 0.0, 4.0), (A, 6.0, 10.0)])
    assert r["recall"] == pytest.approx(0.5 * 0.8) and r["precision"] == pytest.approx(1.0)
    # different entities never overlap
    assert M.range_prf([(A, 0.0, 10.0)], [(B, 0.0, 10.0)])["recall"] == 0.0


def test_range_prf_existence_and_bias():
    real, pred = [(A, 0.0, 10.0)], [(A, 0.0, 5.0)]
    assert M.range_prf(real, pred, alpha=1.0)["recall"] == pytest.approx(1.0)
    assert M.range_prf(real, pred, bias="front")["recall"] == pytest.approx(0.75)
    assert M.range_prf(real, pred, bias="back")["recall"] == pytest.approx(0.25)
    assert M.range_prf(real, pred, bias="middle")["recall"] == pytest.approx(0.5)


def test_point_adjusted():
    pa = M.point_adjusted_prf([(A, 0.0, 10.0), (A, 20.0, 30.0)], [(A, 9.0, 15.0)])
    assert pa["recall"] == pytest.approx(0.5)
    assert pa["precision"] == pytest.approx(10.0 / 15.0)


# --------------------------------------------------------------------------- calibration
def test_ks_uniform_and_rising_edges():
    d, n = M.ks_uniform((np.arange(1000) + 0.5) / 1000)
    assert n == 1000 and d < 0.01
    d, _ = M.ks_uniform(np.full(100, 1e-6))
    assert d > 0.99
    assert M.ks_uniform(np.array([np.nan]))[0] is None
    assert M.rising_edges(np.array([1, 1, 0, 1, 0, 0, 1])) == 3


def test_calibration_gate_reads_the_anti_conservative_side():
    """Round 5 (§16.13): B24 issues p = 1 at a detector's atom (lower-atom
    rule), so a valid detector's issued p has an atom at 1. Gate 7 bounds
    the anti-conservative deviation D+ = max(F_n(x) - x) (validity); the
    two-sided D, which reads the atom's mass, is reported as D_two. An
    anti-conservative p (p / 2) still fails. Fails on the round-4 scorer,
    whose D was two-sided."""
    rng = np.random.default_rng(1)
    n = TICKS.size
    D = len(DETECTORS)
    p = rng.uniform(size=(n, D)).astype(np.float32)
    j = DETECTORS.index("marg_int")
    p[: n // 2, j] = 1.0                                  # atom: nothing unusual
    ser = {"ts": TICKS, "p": p, "e_day": rng.uniform(size=n) * 86400.0 / DT,
           "p_family": np.full((n, len(FAMILIES)), 0.9, dtype=np.float32),
           "alarm_path": [""] * n, "acc_alarm": np.zeros((n, D), dtype=np.int8), "risk": np.zeros(n)}
    c = M.score_run(make_run(series={k("10.20.1.11"): ser}))["calibration"]
    assert c["ks"]["marg_int"]["D"] < 0.15 and c["ks"]["marg_int"]["D_two"] > 0.4
    x = (np.arange(400) + 0.5) / 400
    assert M.ks_uniform(np.r_[x, np.ones(400)], side="upper")[0] == pytest.approx(0.0, abs=1e-9)
    assert M.ks_uniform(x / 2.0, side="upper")[0] > 0.45


def test_calibration_stats_counts_clean_control_ticks():
    rng = np.random.default_rng(0)
    n = TICKS.size
    D = len(DETECTORS)
    p = rng.uniform(size=(n, D)).astype(np.float32)
    e_day = rng.uniform(size=n) * 86400.0 / DT          # e_day = p * 86400 / dt
    paths = [""] * n
    paths[5] = paths[6] = "evidence_cusum"
    acc = np.zeros((n, D), dtype=np.int8)
    acc[10:13, DETECTORS.index("cusum")] = 1
    pf = np.full((n, len(FAMILIES)), 0.9, dtype=np.float32)
    p_bad = p.copy()
    p_bad[3, DETECTORS.index("marg_int")] = 1e-6     # masked by family p 0.9
    ser = {"ts": TICKS, "p": p_bad, "e_day": e_day, "p_family": pf, "alarm_path": paths,
           "acc_alarm": acc, "risk": np.zeros(n)}
    run = make_run(series={k("10.20.1.11"): ser},
                   incidents=[inc("i", "10.20.1.11", TICKS[3], "low")])
    c = M.score_run(run)["calibration"]
    assert c["ks"]["marg_int"]["n"] == n and c["ks"]["marg_int"]["D"] < 0.15
    assert c["evidence_cusum_alarms"] == 1
    assert c["acc_path_alarms"]["change"] == 1
    assert c["exceedance"]["0.03"]["n"] == n
    assert c["exceedance"]["0.03"]["expected"] == pytest.approx(n * 0.03 * DT / 86400.0)
    assert c["acat_ticks"] >= 1 and c["acat_incident_ticks"] == 1
    assert c["cc"] == 900


# --------------------------------------------------------------------------- misc
def test_deadline_parsing_and_base_id():
    assert M.deadline_s({"max_ttd": 1800}, DT) == 1800
    assert M.deadline_s({"max_ttd": "2 ticks"}, DT) == 1800
    assert M.deadline_s({"max_ttd": "6h"}, DT) == 21600
    assert M.deadline_s({"max_ttd_ticks": 4}, 60.0) == 240
    assert M.deadline_s({"max_ttd": {"ticks": 3}}, DT) == 2700
    assert M.deadline_s({}, DT) is None
    assert [M.base_id(x) for x in ("T1'", "T6b", "T12_e", "L10")] == ["T1", "T6b", "T12", "L10"]


def test_bootstrap_ci_deterministic():
    a = M.bootstrap_ci([0.9, 1.0, 0.95, 0.97, 0.92], seed=1)
    assert a == M.bootstrap_ci([0.9, 1.0, 0.95, 0.97, 0.92], seed=1)
    assert 0.9 <= a[0] <= a[1] <= 1.0
    assert M.bootstrap_ci([]) == (None, None)


def test_identification_and_classes_from_profiles_and_models():
    tw = ["10.30.2.27", "10.30.2.28"]
    pers = personas("10.20.1.11", "10.20.1.12", "10.20.1.13", "10.20.1.15")
    pers.update({f"oa-portal|{e}": {"archetype": "interactive"} for e in tw})
    pers.update({k("10.20.4.30"): {"archetype": "api"}, k("10.20.4.32"): {"archetype": "api"}})
    prof = {key: {"separability": 0.5 + 0.1 * i,
                  "extra": {"identity": {"recall1": 0.9 + 0.02 * i, "eer_hard": 0.01}}}
            for i, key in enumerate(sorted(personas("10.20.1.11", "10.20.1.12", "10.20.1.13",
                                                    "10.20.1.15")))}
    prof["oa-portal|10.30.2.27"] = {"extra": {"identity": {"eer_hard": 0.3,
                                                           "confusable_with": ["10.30.2.28"]}}}
    prof["oa-portal|10.30.2.28"] = {"extra": {"identity": {"eer_hard": 0.3,
                                                           "confusable_with": [{"entity": "10.30.2.27"}]}}}
    assign = {key: {"role": "human" if p["archetype"] == "interactive" else "m", "sub": key}
              for key, p in pers.items()}
    run = make_run(personas=pers, profiles=prof,
                   models={"class": {"assign": assign},
                           "class_history": [{"assign": assign}, {"assign": {
                               kk: {**v, "role": v["role"] + "2"} for kk, v in assign.items()}}]})
    sc = M.score_run(run)
    idn = sc["identification"]
    assert idn["twins_ok"] is True and idn["spearman"] == pytest.approx(1.0)
    assert idn["eer_frac_ok"] == 1.0
    cl = sc["classes"]
    assert cl["ari"] == pytest.approx(1.0) and cl["noise_frac"] == 0.0
    assert cl["refit_ari_min"] == pytest.approx(1.0) and cl["id_churn"] == 1


def test_explanation_hit3_and_natural_range():
    ts = TICKS[20]
    row = truth("T18", ["10.20.1.16"], ts, ts + 10 * DT,
                perturbed_features=["bytes_up", "updown_log"])
    i1 = inc("i1", "10.20.1.16", ts, "medium", explanation={
        "attributions": [{"feature": "flows", "range": [1, 2]}, {"feature": "feature.bytes_up"}],
        "counterfactual_valid": True})
    i2 = inc("i2", "10.20.1.11", TICKS[5], "low", narrative="usual 0.4–1.1 MB")
    run = make_run(truth=[row], incidents=[i1, i2])
    ex = M.score_run(run)["explanation"]
    assert ex["hit3"] == [True] and ex["cf_valid"] == [True]
    assert ex["natural_range"] == [True, True]


def test_hit3_skips_the_whole_entity_marker():
    # L6 / L8 rows list perturbed_features ['entity']: no attribution can rank
    # it, so the row has no hit@3 (it used to count as a miss); the
    # counterfactual check is unaffected
    ts = TICKS[20]
    row = truth("T18", ["10.20.1.16"], ts, ts + 10 * DT, perturbed_features=["entity"])
    i1 = inc("i1", "10.20.1.16", ts, "medium", explanation={
        "attributions": [{"feature": "flows"}], "counterfactual_valid": False})
    ex = M.score_run(make_run(truth=[row], incidents=[i1]))["explanation"]
    assert ex["hit3"] == [] and ex["cf_valid"] == [False]
    row["perturbed_features"] = ["entity", "flows"]
    ex = M.score_run(make_run(truth=[row], incidents=[i1]))["explanation"]
    assert ex["hit3"] == [True]


def _synthetic_scores():
    ts = TICKS[20]
    out = []
    for seed in range(3):
        rows = [truth("T1", ["10.20.1.11"], ts, ts + 4 * DT),
                truth("T3", ["10.20.1.12"], ts, ts + 40 * DT),
                truth("L5", ["10.20.1.13"], ts, ts + 4 * DT, label="legit_change")]
        incs = [inc("i1", "10.20.1.11", ts, "high")]
        if seed:
            incs.append(inc("i2", "10.20.1.12", ts + DT, "medium"))
        out.append(M.score_run(make_run(seed=seed, truth=rows, incidents=incs)))
    return out


def test_compute_gates_structure_and_values():
    scores = _synthetic_scores()
    g = M.compute_gates(json.loads(json.dumps(scores)))      # survives a JSON round trip
    assert len(g) == 16
    for gid, x in g.items():
        assert set(x) >= {"value", "target", "pass", "details"}, gid
        assert x["pass"] in (True, False, None)
    assert g["1_detection"]["value"] == pytest.approx(5 / 6)
    assert g["1_detection"]["pass"] is False                  # subtle recall 2/3 < 0.9
    assert g["3_far"]["details"]["checks"][0]["value"] == 0.0
    assert g["12_feedback"]["pass"] is None and g["13_ablation"]["pass"] is None
    assert g["16_report"]["pass"] is True
    fb = M.feedback_gate(scores, scores)
    assert fb["pass"] is None                                  # < 20 labels: n/a
    labelled = [dict(s, labels_added=10, far=dict(s["far"], n_low=0, n_low_notified=0)) for s in scores]
    base = [dict(s, far=dict(s["far"], n_low=4, n_low_notified=4)) for s in scores]
    fb = M.feedback_gate(base, labelled)
    assert fb["pass"] is True and fb["value"] == 1.0
    abl = M.ablation_table(scores, {"b14": [dict(s, scenarios=[dict(o, within_deadline=False)
                                                               for o in s["scenarios"]])
                                            for s in scores]})
    assert abl[0]["sole_detector_of"] == ["A/T1", "A/T3"]


def test_feedback_and_ablation_compare_the_same_pack_seeds():
    """Feedback / ablation runs usually cover a subset of the full runs; they
    used to be compared with ALL full runs (the cut and the FAR delta mixed
    packs and seeds)."""
    scores = _synthetic_scores()
    base = [dict(s, far=dict(s["far"], n_low=4, n_low_notified=4)) for s in scores]
    fb = [dict(base[0], labels_added=25, far=dict(base[0]["far"], n_low=2, n_low_notified=2))]
    g = M.feedback_gate(base, fb)
    assert g["value"] == pytest.approx(0.5)          # 4 -> 2 on the paired run, not 12 -> 2
    abl = M.ablation_table(base, {"b04": [dict(base[0], far=dict(base[0]["far"], n_low=4))]})
    assert abl[0]["delta_far_low"] == pytest.approx(0.0)


def test_report_json_and_self_contained_html(tmp_path):
    scores = _synthetic_scores()
    rep = write_report(scores, str(tmp_path), meta={"note": "test"})
    data = json.loads((tmp_path / "eval_report.json").read_text(encoding="utf-8"))
    assert data["summary"]["n_runs"] == 3 and len(data["gates"]) == 16
    assert any(r["scenario_id"] == "T1" for r in data["scenarios"])
    assert "| # | Gate |" in data["design_table_md"]
    html = (tmp_path / "eval_report.html").read_text(encoding="utf-8")
    assert html.startswith("<!doctype html>") and "<style>" in html
    assert "http://" not in html and "https://" not in html and "<script" not in html
    assert "T1" in html and "False alarms" in html and "prefers-color-scheme" in html
    assert render_html(build_report(scores)).count("<table") >= 5
    assert not math.isnan(rep["summary"]["gates_pass"])


def test_reopened_incident_is_a_new_opening_for_detection_but_one_far_incident():
    """B27 reuses an incident's id when it reopens within 24 h, so `opened`
    stays at the first opening. A threat that reopens an FP incident closed
    before its onset is detected by that reopening (split_episodes); for
    FAR the incident still counts once (with the max severity of its counted
    episodes), the reopenings are reported as n_reopened."""
    t_fp, ts = TICKS[4], TICKS[40]
    hist = [{"ts": t_fp, "severity": "low", "status": "open", "axes": ["identity"], "kinds": []},
            {"ts": t_fp + 4 * DT, "severity": "low", "status": "closed", "axes": ["identity"],
             "kinds": []},
            {"ts": ts + DT, "severity": "high", "status": "open", "axes": ["temporal"],
             "kinds": ["alarm"]}]
    run = make_run(truth=[truth("T5", ["10.20.1.11"], ts, ts + 8 * DT,
                                expected_axes=["temporal"])],
                   incidents=[inc("i1", "10.20.1.11", t_fp, "high", axes=("temporal",),
                                  history=hist)])
    o = M.score_run(run)["scenarios"][0]
    assert o["detected"] and o["t_detect"] == ts + DT and o["n_incidents"] == 1
    eps = M.split_episodes(run.incidents)
    assert [e["opened"] for e in eps] == [t_fp, ts + DT]
    assert eps[0]["status"] == "closed" and eps[0]["severity"] == "low"
    # the same incident on a control entity: one FAR incident, one reopening
    hist2 = [dict(h) for h in hist]
    run2 = make_run(incidents=[inc("i2", "10.20.1.12", t_fp, "high", history=hist2)])
    far = M.score_run(run2)["far"]
    assert (far["n_low"], far["n_high"], far["n_reopened"]) == (1, 1, 1)


def test_report_renders_ablation_and_meta_sections(tmp_path):
    scores = _synthetic_scores()
    abl = {"b14": [dict(s, scenarios=[dict(o, within_deadline=False) for o in s["scenarios"]])
                   for s in scores]}
    meta = {"sections": [{"title": "Before / after", "headers": ["gate", "before", "after"],
                          "rows": [["1_detection", 0.5, 0.75]], "note": "round 2"}]}
    write_report(scores, str(tmp_path), ablation=abl, meta=meta)
    html = (tmp_path / "eval_report.html").read_text(encoding="utf-8")
    assert "Ablation (one engine disabled)" in html and "b14" in html
    assert "Before / after" in html and "round 2" in html and "1_detection" in html


# ------------------------------------------------------ detection by escalation
def _esc_run(hist, events, axes_exp=("exfil",), req="medium", opened_off=-20):
    ts = TICKS[40]
    t_open = ts + opened_off * DT
    h = [dict(p, ts=t_open + p["ts"] * DT if p["ts"] < 0 else ts + p["ts"] * DT) for p in hist]
    run = make_run(truth=[truth("T4", ["10.20.1.11"], ts, ts + 16 * DT, max_ttd="6h",
                                expected_axes=list(axes_exp), required_severity=req)],
                   incidents=[inc("i1", "10.20.1.11", t_open, h[-1]["severity"],
                                  axes=tuple(h[-1]["axes"]), history=h)],
                   events=[ev(f"n{j}", "10.20.1.11", ts + o * DT, "incident", incident_id="i1",
                              extra={"state": st}) for j, (o, st) in enumerate(events)])
    return run, ts


def _p(t, sev, axes, status="open"):
    return {"ts": t, "severity": sev, "status": status, "axes": list(axes), "kinds": []}


def test_escalation_of_an_open_incident_with_new_expected_axis_is_a_detection():
    """Lead decision (round 4): an attack folded into an already-open (FP)
    incident is detected when, inside the window, the incident rises >= 1
    level to >= MEDIUM, gains an expected axis and emits 'escalate'. TTD is
    taken from that escalation; the opening-only rule stays a secondary
    field. Without the rule this scenario is missed (no incident opened in
    the window)."""
    hist = [_p(-1, "low", ["identity"]), _p(3, "high", ["identity", "exfil"])]
    run, ts = _esc_run(hist, [(-19, "open"), (3, "escalate"), (5, "escalate")])
    o = M.score_run(run)["scenarios"][0]
    assert o["detected"] and o["detected_by"] == "escalation"
    assert not o["detected_open"] and not o["within_deadline_open"]
    assert o["t_detect"] == ts + 3 * DT and o["ttd_ticks"] == 4 and o["within_deadline"]
    assert o["escalation"]["gained_axes"] == ["exfil"]
    # notifications from the onset on: the two escalations, not the FP's open
    assert o["notifications"] == 2
    g = M.gate_detection([M.score_run(run)])
    assert g["details"]["recall_open"] == 0.0 and g["details"]["n_by_escalation"] == 1


@pytest.mark.parametrize("case", ["no_new_axis", "no_level_rise", "below_medium",
                                  "no_notification", "closed_before", "unexpected_axis"])
def test_escalation_requirements(case):
    hist = [_p(-1, "low", ["identity"]), _p(3, "high", ["identity", "exfil"])]
    notes = [(3, "escalate")]
    axes_exp = ("exfil",)
    if case == "no_new_axis":
        hist = [_p(-1, "low", ["exfil"]), _p(3, "high", ["exfil"])]
    elif case == "no_level_rise":
        hist = [_p(-1, "high", ["identity"]), _p(3, "high", ["identity", "exfil"])]
    elif case == "below_medium":
        hist = [_p(-1, "info", ["identity"]), _p(3, "low", ["identity", "exfil"])]
    elif case == "no_notification":
        notes = [(3, "update")]
    elif case == "closed_before":
        hist = [_p(-2, "low", ["identity"]), _p(-1, "low", ["identity"], "closed"),
                _p(3, "high", ["identity", "exfil"])]
    elif case == "unexpected_axis":
        axes_exp = ("c2",)
    run, _ = _esc_run(hist, notes, axes_exp=axes_exp)
    o = M.score_run(run)["scenarios"][0]
    if case == "closed_before":
        # a REOPENING in the window is an opening (split_episodes), not an escalation
        assert o["detected"] and o["detected_by"] == "open"
    else:
        assert not o["detected"] and not o.get("detected_escalation")


def test_escalation_needs_the_required_severity():
    hist = [_p(-1, "low", ["identity"]), _p(3, "medium", ["identity", "exfil"])]
    run, _ = _esc_run(hist, [(3, "escalate")], req="high")
    assert not M.score_run(run)["scenarios"][0]["detected"]


def test_feedback_cut_counts_notified_control_incidents():
    """Gate 12 (round 4): an incident a pattern policy suppresses from its
    first tick never reaches the analyst; the cut counts the notifying
    control incidents and reports the raw count as a secondary check."""
    ts = TICKS[40]
    hist_sup = [{"ts": ts, "severity": "low", "status": "suppressed", "axes": ["volume"]}]
    hist_open = [{"ts": ts, "severity": "low", "status": "open", "axes": ["volume"]}]
    base = make_run(incidents=[inc("i1", "10.20.1.12", ts, "low", history=hist_open),
                               inc("i2", "10.20.1.13", ts, "low", history=hist_open)])
    fb = make_run(labels_added=25,
                  incidents=[inc("i1", "10.20.1.12", ts, "low", history=hist_sup),
                             inc("i2", "10.20.1.13", ts, "low", history=hist_open)])
    sb, sf = M.score_run(base), M.score_run(fb)
    assert (sb["far"]["n_low"], sb["far"]["n_low_notified"]) == (2, 2)
    assert (sf["far"]["n_low"], sf["far"]["n_low_notified"]) == (2, 1)
    g = M.feedback_gate([sb], [sf])
    assert g["value"] == pytest.approx(0.5)
    raw = [c for c in g["details"]["checks"] if "suppressed incidents included" in c["name"]][0]
    assert raw["value"] == pytest.approx(0.0)
    # an open notification event makes it notifying whatever its status history
    fb.events = [ev("n1", "10.20.1.12", ts, "incident", incident_id="i1",
                    extra={"state": "open"})]
    assert M.score_run(fb)["far"]["n_low_notified"] == 2


def test_spearman_uses_the_graded_cv_recall_and_the_twins():
    """Round 4, gate 8: recall@1 saturates at 1.0 for individuated personas
    (A / B) and left Spearman undefined; the graded CV recall (B15's median
    held-out margin) over the individuated personas plus the twins defines it."""
    tw = ["10.30.2.27", "10.30.2.28"]
    ind = ("10.20.1.11", "10.20.1.12", "10.20.1.13", "10.20.1.15")
    pers = personas(*ind)
    pers.update({f"oa-portal|{e}": {"archetype": "interactive"} for e in tw})
    prof = {key: {"separability": 1.0,
                  "extra": {"identity": {"recall1": 1.0, "eer_hard": 0.0, "margin": 20.0 + i}}}
            for i, key in enumerate(sorted(personas(*ind)))}
    for j, e in enumerate(tw):
        prof[f"oa-portal|{e}"] = {"separability": 0.4 + 0.1 * j, "extra": {"identity": {
            "recall1": 0.8, "eer_hard": 0.3 - 0.05 * j, "margin": 1.0 + j}}}
    idn = M.score_run(make_run(personas=pers, profiles=prof))["identification"]
    assert idn["spearman"] is not None and idn["spearman"] > 0.8
    # without margins (old B15 profiles) recall@1 is the fallback grade
    for v in prof.values():
        v["extra"]["identity"].pop("margin")
    idn = M.score_run(make_run(personas=pers, profiles=prof))["identification"]
    assert idn["spearman"] is not None and idn["spearman"] > 0.5
