"""Evaluator round 7: PG6 anomaly matching follows the truth family of the
anomaly's system. O-real's R9 load-balances oa onto the replica oa-r2, the
injected anomaly events included, so an anomaly of 'oa' can be observed and
detected on oa-r2 (seed 1: A4 detected there, MEDIUM, counted as missed). The
IPs and the time window must still match; another system never counts."""
from __future__ import annotations

from types import SimpleNamespace

from app.eval import pmetrics as M

T = 1_790_000_000.0


def _pt(families):
    return M.PTruth({"pattern_truth": [], "system_truth": {"systems": {}, "families": families}})


def _run(system):
    row = {"scenario_id": "A4", "label": "malicious", "system": "oa", "entities": ["10.0.0.9"],
           "t_start": T, "t_first": T, "expected_types": ["when"], "required_severity": "medium"}
    ev = {"kind": "pattern_violation", "system": system, "entity": "10.0.0.9", "ts": T + 10.0,
          "severity": "medium", "extra": {"type": "when"}}
    inc = {"system": system, "entity": "10.0.0.9", "opened": T + 900.0, "severity": "medium",
           "history": [{"ts": T + 900.0, "severity": "medium"}]}
    return SimpleNamespace(truth=[row], events=[ev], incidents=[inc], scenario_dt=900.0)


def test_anomaly_detected_on_a_replica_of_its_system_counts():
    fam = {"oa": ["oa", "oa-r2"]}
    assert M.pg6_anomalies(_run("oa-r2"), _pt(fam), {})["A4"]["detected"] is True
    assert M.pg6_anomalies(_run("oa"), _pt(fam), {})["A4"]["detected"] is True
    # without the family (pack O) or on another system: not detected
    assert M.pg6_anomalies(_run("oa-r2"), _pt({}), {})["A4"]["detected"] is False
    assert M.pg6_anomalies(_run("finance"), _pt(fam), {})["A4"]["detected"] is False


def test_pg2_depth_is_judged_on_the_statements_confirmed_on_both_days():
    """(round 7, evaluator; tree owner's OI-b) new route nodes confirmed at
    depth 1 lower the MEAN depth although no statement lost depth (pack O seed
    1, round 6: 1.424 -> 1.421 on days 5 -> 7): the trend is judged per step on
    the statements of both days; a statement that lost depth still fails it."""
    from app.eval import pmetrics as PM

    class _PT:
        rows = []
        n_days = 21

    def snap(depths):
        return {"recall": 0.8, "recall_by_period": {}, "ece": 0.1,
                "mean_depth": sum(depths.values()) / len(depths), "stmt_depth": dict(depths)}
    pg1 = {3: snap({"a": 1, "b": 2}), 5: snap({"a": 1, "b": 2}),
           7: snap({"a": 1, "b": 2, "c": 1, "d": 1}), 10: snap({"a": 2, "b": 2, "c": 1, "d": 1})}
    out = PM.pg2_convergence(pg1, _PT())
    assert out["depth_mean_nondecreasing"] is False          # 1.5 -> 1.25 on days 5 -> 7
    assert out["depth_nondecreasing"] is True
    pg1[10] = snap({"a": 1, "b": 1})                          # b lost a level
    assert PM.pg2_convergence(pg1, _PT())["depth_nondecreasing"] is False
