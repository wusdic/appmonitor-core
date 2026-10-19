"""P03 round 8 (OC7-6): §6.9.2 (iii) - an operator's benign verdict absorbs a
source into a system's who sets.

A who-closed node learns a foreign source damped (0.1) until it qualifies;
persistence alone never qualifies it (A9). The spec's third way in, "an
operator labels it (B23 fp with scope pattern, or expected_change)", was not
implemented: a host whose only findings were who findings, resolved benign by
the operator (round 7: a health-check host, 29 LOW findings on O-real seed 1),
stayed damped and flagged, and accrued standing at the damped rate (> 40 days).
"""
from __future__ import annotations

import pytest

import test_p03_conformity as T3

from app.engines.behavior import conformity as CF
from app.models.schema import Incident, Label, Severity

HOST = "192.168.9.9"                      # no P11 group: foreign to the closed node


def _days():
    return T3.workdays(33)[22:28]


def _score_days(fx, days):
    out = []
    for d in days:
        _, asg = fx.score("finance", [T3._approve(HOST, d + 10.5 * 3600),
                                      T3._approve("192.168.2.10", d + 10.6 * 3600)])
        out.append((asg.get("damp", 0), asg.get("p_who", 0)))
    return out


def _who_days(fx):
    return {e.dedupe_key for e in fx.violations(HOST) if e.extra["type"] == "who"}


def test_unlabelled_foreign_source_stays_damped_and_flagged():
    """A9 is unchanged: no label, no absorption."""
    fx, _ = T3.finance_fixture()
    res = _score_days(fx, _days())
    assert all(dm == pytest.approx(0.1) and pw < 1.0 for dm, pw in res)
    assert len(_who_days(fx)) == len(_days())


@pytest.mark.parametrize("verdict,scope,axes", [
    ("fp", "pattern", ["privilege"]),      # benign who finding, pattern scope
    ("benign_known", "entity", []),        # the entity as a whole
    ("expected_change", "this", []),       # an expected change of access
])
def test_benign_label_absorbs_the_source(verdict, scope, axes):
    fx, _ = T3.finance_fixture()
    days = _days()
    before = _score_days(fx, days[:2])
    assert all(dm == pytest.approx(0.1) for dm, _ in before)
    inc = Incident(id="inc-h", system="finance", entity=HOST, axes=list(axes),
                   severity=Severity.LOW, opened=fx.now, last_seen=fx.now)
    fx.st.put_incident(inc)
    fx.st.add_label(Label(system="finance", entity=HOST, target_type="incident", target_id="inc-h",
                          verdict=verdict, scope=scope, analyst="t", ts=fx.now))
    n_who = len(_who_days(fx))
    after = _score_days(fx, days[2:])
    assert all(dm == 1.0 and pw == 1.0 for dm, pw in after), after   # learned at full weight
    assert len(_who_days(fx)) == n_who                                # no further who findings


def test_pattern_label_on_a_non_who_incident_does_not_absorb():
    fx, _ = T3.finance_fixture()
    fx.st.put_incident(Incident(id="inc-c", system="finance", entity=HOST, axes=["content"],
                                severity=Severity.LOW, opened=fx.now, last_seen=fx.now))
    fx.st.add_label(Label(system="finance", entity=HOST, target_type="incident", target_id="inc-c",
                          verdict="fp", scope="pattern", analyst="t", ts=fx.now))
    assert all(dm == pytest.approx(0.1) for dm, _ in _score_days(fx, _days()))


def test_later_tp_label_and_ttl_end_the_absorption():
    fx, _ = T3.finance_fixture()
    days = _days()
    fx.st.add_label(Label(system="finance", entity=HOST, verdict="benign_known", scope="entity",
                          target_type="entity", analyst="t", ts=fx.now))
    assert all(dm == 1.0 for dm, _ in _score_days(fx, days[:2]))
    fx.st.add_label(Label(system="finance", entity=HOST, verdict="tp", target_type="entity",
                          analyst="t", ts=fx.now))
    assert all(dm == pytest.approx(0.1) for dm, _ in _score_days(fx, days[2:4]))
    ab = CF.label_absorbed(fx.st, fx.now)
    assert HOST not in ab.get("finance", {})
    fx.st.add_label(Label(system="finance", entity=HOST, verdict="expected_change",
                          target_type="entity", analyst="t", ts=fx.now, ttl_s=3600.0))
    assert CF.label_absorbed(fx.st, fx.now + 60.0)["finance"][HOST][1] == fx.now + 3600.0
    assert all(dm == pytest.approx(0.1) for dm, _ in _score_days(fx, days[4:]))   # expired


def test_other_systems_are_not_absorbed():
    fx, _ = T3.finance_fixture()
    fx.st.add_label(Label(system="oa", entity=HOST, verdict="benign_known", scope="entity",
                          target_type="entity", analyst="t", ts=fx.now))
    assert all(dm == pytest.approx(0.1) for dm, _ in _score_days(fx, _days()))
