"""Round 4 B29: attribution ranked by the calibrated per-feature evidence of the
rows that cover the incident's opening, and a counterfactual search that also
neutralises the detectors / findings that drove the decision.

Why: pack A seed 0 had hit@3 0.12 (the Garthwaite-Koch shares of the LATEST
explained tick, often an escalation days after the opening) and
counterfactual validity 0.11 (decisions driven by held detectors - seq,
client, dwell, novelty - by B09's client_impersonation finding or by B27's
risk opening could never flip: 'no candidate set flips the decision').
"""
from __future__ import annotations

import math
from types import SimpleNamespace

import numpy as np

from helpers import make_store

from app.engines.behavior import explain as EX
from app.engines.behavior.lib import features as F
from app.engines.behavior.lib.detectors import DETECTOR_INDEX, N_DETECTORS

S, E = "erp", "10.0.0.1"
NF = F.FEATURE_DIM
IDX = F.FEATURE_INDEX
T_OPEN = 1_741_600_800.0                       # an H decision tick (epoch-aligned hour)


def _pf_rows(store, name: str, ts, rows):
    for t, r in zip(ts, rows):
        store.add_vec(S, E, name, float(t), np.asarray(r, dtype=np.float32), window_s=900)


def _world():
    store = make_store()
    rng = np.random.default_rng(0)
    pre = T_OPEN - 5 * 3600.0 + 900.0 * np.arange(16)      # before opened - 1 h
    base = rng.uniform(0.05, 1.0, (16, NF))
    base[:, IDX["rtt"]] = 1e-5                              # chronically extreme for this IP
    after = [T_OPEN, T_OPEN + 900.0, T_OPEN + 1800.0]
    r0 = rng.uniform(0.2, 1.0, NF)
    r0[IDX["rtt"]] = 1e-5
    r0[IDX["bytes_up"]] = 1e-7                              # what opened the incident
    r0[IDX["updown_log"]] = 1e-4
    late = rng.uniform(0.2, 1.0, NF)
    late[IDX["http_5xx_rate"]] = 1e-12                      # an unrelated later escalation
    rows = [r0, rng.uniform(0.2, 1.0, NF), late]
    for name in ("behavior.pf.q",):
        _pf_rows(store, name, list(pre) + after, list(base) + rows)
    return store


def test_opening_evidence_ranks_the_opening_rows_calibrated_by_own_history():
    store = _world()
    inc = SimpleNamespace(opened=T_OPEN, explanation={})
    oe = EX.ExplainEngine._opening_evidence(store, S, E, inc, T_OPEN + 1800.0, {})
    assert oe and "q" in oe["grains"] and oe["grains"]["q"]["ts"] == T_OPEN
    ev = np.array([np.nan if x is None else x for x in oe["grains"]["q"]["ev"]])
    order = [F.FEATURE_NAMES_V2[i] for i in np.argsort(-np.nan_to_num(ev, nan=-9))]
    assert order[:2] == ["bytes_up", "updown_log"]
    assert ev[IDX["rtt"]] < 1.0                               # its routine level is removed
    assert ev[IDX["http_5xx_rate"]] < 1.0                     # later rows are not the opening


def test_opening_evidence_is_carried_once_the_rows_expired():
    store = make_store()
    inc = SimpleNamespace(opened=T_OPEN, explanation={})
    prev = {"opening_evidence": {"t_opened": T_OPEN,
                                 "grains": {"q": {"ts": T_OPEN, "ev": [1.0] * NF}}}}
    oe = EX.ExplainEngine._opening_evidence(store, S, E, inc, T_OPEN + 86400.0, prev)
    assert oe["grains"]["q"]["ev"] == [1.0] * NF
    # a later reopening (another first opening time) is not carried
    inc2 = SimpleNamespace(opened=T_OPEN + 7200.0, explanation={})
    assert EX.ExplainEngine._opening_evidence(store, S, E, inc2, T_OPEN + 86400.0, prev) is None


def test_due_refreshes_at_the_first_h_tick_after_the_opening():
    inc = SimpleNamespace(opened=T_OPEN - 900.0, last_seen=T_OPEN - 900.0, evidence=[],
                          severity="medium", axes=[],
                          explanation={"ts": T_OPEN - 900.0, "severity": "medium", "axes": [],
                                       "opening_evidence": {"t_opened": T_OPEN - 900.0,
                                                            "grains": {"q": {}}}})
    assert EX.ExplainEngine._due(inc, T_OPEN, 900.0)
    assert not EX.ExplainEngine._due(inc, T_OPEN - 450.0, 450.0)   # not an H tick
    inc.explanation["opening_evidence"]["grains"]["h"] = {}
    assert not EX.ExplainEngine._due(inc, T_OPEN, 900.0)


# ------------------------------------------------------------ counterfactual
def _cf():
    cf = EX.Counterfactual.__new__(EX.Counterfactual)
    cf.neutralised = set()
    return cf


def test_detector_neutralisation_only_raises_p():
    cf = _cf()
    p = np.full(N_DETECTORS, 0.9)
    p[DETECTOR_INDEX["seq"]] = 1e-9
    p[DETECTOR_INDEX["dwell"]] = np.nan
    nz = EX._Neutral(["detector:seq", "detector:dwell", "detector:client"])
    cf._neutralise(p, nz)
    assert p[DETECTOR_INDEX["seq"]] == EX.NEUTRAL_P
    assert math.isnan(p[DETECTOR_INDEX["dwell"]])            # not scored: stays NaN
    assert p[DETECTOR_INDEX["client"]] == 0.9                # never lowered
    assert cf.neutralised == {"seq", "client"}


def test_findings_are_removed_by_their_tokens_or_their_detector():
    cf = _cf()
    imp = SimpleNamespace(kind="client_impersonation", extra={"stack": "abc|curl/8|linux|64|w"})
    fs = SimpleNamespace(kind="first_seen", extra={"dim": "sni", "value": "evil.example"})
    assert EX.finding_tokens(imp) == ["stack=abc|curl/8|linux|64|w"]
    assert cf._finding_removed(imp, EX._Neutral(["token:stack=abc|curl/8|linux|64|w"]))
    assert cf._finding_removed(imp, EX._Neutral(["detector:client"]))
    assert not cf._finding_removed(imp, EX._Neutral(["detector:seq"]))
    assert cf._finding_removed(fs, EX._Neutral(["token:sni=evil.example"]))
    assert cf._finding_removed(fs, EX._Neutral(["detector:novelty"]))


def test_decisive_finding_at_the_opening_names_its_feature():
    from app.models.schema import BehaviorEvent, Severity
    store = _world()
    store.add_event(BehaviorEvent(system=S, entity=E, ts=T_OPEN, kind="client_impersonation",
                                  score=0.9, severity=Severity.HIGH, description="new stack",
                                  extra={"stack": "abc"}))
    store.add_event(BehaviorEvent(system=S, entity=E, ts=T_OPEN, kind="first_seen", score=0.5,
                                  severity=Severity.LOW, description="low: not decisive",
                                  extra={"dim": "tmpl", "value": "GET x /y"}))
    inc = SimpleNamespace(opened=T_OPEN, explanation={})
    oe = EX.ExplainEngine._opening_evidence(store, S, E, inc, T_OPEN + 900.0, {})
    assert oe["findings"] == {"ja3_diversity": "client_impersonation"}
