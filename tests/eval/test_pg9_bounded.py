"""PG9 bounded mode (round-7 lead decision, docs/lib3/eval.md): on packs A / E
(nearly every IP earns its models) bounded mode must cost no accuracy (gates
1, 3, 8) and no lib-3 CPU; the <= 0.5 x full cost target applies to large
populations dominated by unearned IPs. The lib-3 cost of a run is the thread
CPU of its behaviour-layer engines without the progressive core's learners
(P15 counted), live ticks only."""
from __future__ import annotations

import numpy as np

from app.eval import pmetrics as PM
from app.eval import pscale as S


def _timings():
    names = ["raw.http", "behavior.feature_vector", "behavior.resource_governor",
             "behavior.pattern_tree", "behavior.risk", "signature.rule_match"]
    layers = ["raw", "behavior", "behavior", "behavior", "behavior", "signature"]
    cpu = np.array([[5, 10, 1, 100, 2, 7],
                    [5, 20, 1, 100, 4, 7],
                    [5, 30, 1, 100, 6, 7]], dtype=float)
    ticks = np.zeros((3, 11))
    ticks[0, 3] = 1                                           # a training tick
    return {"engine_names": names, "engine_layers": layers, "engine_cpu_ms": cpu,
            "ticks": ticks, "columns": ["ts", "dt", "phase", "training", "gen_ms", "pipeline_ms",
                                        "raw_ms", "derived_ms", "behavior_ms", "signature_ms",
                                        "collect_ms"]}


def test_lib3_tick_cpu_counts_lib3_engines_and_p15_on_live_ticks():
    t = S.lib3_tick_cpu(_timings())
    # feature_vector + resource_governor + risk; not P04, raw or signature; tick 0 is training
    assert t.tolist() == [25.0, 37.0]
    c = S.lib3_cost(_timings())
    assert c["n_ticks"] == 2 and c["mean_ms"] == 31.0
    assert S.lib3_tick_cpu({"engine_names": ["a"], "engine_layers": ["behavior"]}) is None


def _summ(recall, med, high, cost, t9=True):
    return {"recall": recall, "n_medium": med, "n_high": high, "twins_ok": True, "t9": t9,
            "t19": True, "cost": {"p95_ms": cost}}


def test_pg9_bounded_pairs_and_large_points():
    pairs = {"A seed 0": {"full": _summ(1.0, 7, 2, 2000.0), "bounded": _summ(1.0, 7, 2, 1900.0)},
             "E seed 0": {"full": _summ(1.0, 1, 0, 600.0), "bounded": _summ(1.0, 0, 0, 500.0)}}
    large = {"O-scale 5000 IPs": {"full": {"p95_ms": 1000.0}, "bounded": {"p95_ms": 400.0}}}
    g = PM.pg9_bounded(pairs, large)
    assert g["pass"] is True and g["value"] == 0.4
    # a bounded run that loses T9, adds a MEDIUM incident or costs more fails
    bad = {"A seed 0": {"full": _summ(1.0, 7, 2, 2000.0), "bounded": _summ(1.0, 8, 2, 2100.0, t9=False)}}
    g = PM.pg9_bounded(bad, large)
    failed = {c["name"].split(":")[1].strip().split(" (")[0] for c in g["details"]["checks"]
              if c["pass"] is False}
    assert g["pass"] is False
    assert failed == {"gate 3 incidents >= MEDIUM", "gate 8 t9 kept",
                      "lib-3 CPU p95 bounded / full"}
    # the large-population target is 0.5 x full
    g = PM.pg9_bounded(pairs, {"O-scale 20000 IPs": {"full": {"p95_ms": 1000.0},
                                                      "bounded": {"p95_ms": 600.0}}})
    assert g["pass"] is False and g["value"] == 0.6
    # a point compared over less than 48 h is reported, not judged (day 1: every
    # known IP is still in the active set)
    g = PM.pg9_bounded(pairs, {"short": {"full": {"p95_ms": 1000.0, "span_h": 14.0},
                                         "bounded": {"p95_ms": 900.0, "span_h": 14.0}},
                               "long": {"full": {"p95_ms": 1000.0, "span_h": 84.0},
                                        "bounded": {"p95_ms": 440.0, "span_h": 84.0}}})
    assert g["pass"] is True and g["value"] == 0.44
    assert [c["pass"] for c in g["details"]["checks"] if "short" in c["name"]] == [None]
    # recall drop beyond 0.03 fails gate 1
    g = PM.pg9_bounded({"A": {"full": _summ(1.0, 1, 0, 10.0), "bounded": _summ(11 / 12, 1, 0, 9.0)}})
    assert g["details"]["checks"][0]["pass"] is False
    assert PM.pg9_bounded({}, {})["pass"] is None


def test_pg9_run_summary_from_a_score():
    score = {"pack": "A", "seed": 0,
             "scenarios": [{"detected": True, "within_deadline": True}, {"detected": False}],
             "far": {"n_low": 3, "n_medium": 1, "n_high": 0},
             "identification": {"twins_ok": True, "t9": [True, True], "t9b": [False], "t19": []}}
    s = PM.pg9_run_summary(score, {"p95_ms": 12.0})
    assert s["recall"] == 0.5 and s["detected"] == 1 and s["within_deadline"] == 1
    assert (s["t9"], s["t9b"], s["t19"], s["twins_ok"]) == (True, False, None, True)
    assert s["cost"]["p95_ms"] == 12.0 and s["n_medium"] == 1
