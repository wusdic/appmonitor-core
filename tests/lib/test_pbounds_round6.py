"""Round 6 (content / time / views owner): P06 publishes a numeric record only
when it states something (lib/pbounds.stated). Fails on the round-5 code."""
from __future__ import annotations

from types import SimpleNamespace

from app.engines.behavior import content_bounds as CBE
from app.engines.behavior.lib import pbounds as PB
from app.engines.behavior.lib import pnode as PN

T0 = 1_780_000_000.0
DAY = 86400.0


def _num(values, days=4):
    s = PN.NumSummary(log=False)
    for i, v in enumerate(values):
        d = i % days
        s.update(float(v), T0 + d * DAY + i * 60.0, 1.0, 1.0, day=700000 + d)
    return s, T0 + days * DAY, 700000 + days - 1


def test_a_record_without_evidence_states_nothing():
    """Pack O seed 0, day 21: mail's TLS node 1 published net.pkts_down from
    n_eff 0.6 - band coverage 0.006, range 23-26 at P(next outside) <= 76 % -
    and P04's held-out tests failed its range at the nominal 0.24 (p_hold 0.62,
    the evaluator's checks held 0.94)."""
    s, t, day = _num([23, 24, 26, 25])
    rec = PB.fit_numeric(s, t, day, n_c=0.6, n_eff=0.6)
    assert rec is not None and rec["coverage"] < PB.STATED_MIN
    assert not PB.stated(rec)
    # with evidence the same values make a claim
    s2, t2, day2 = _num([23 + (i % 4) for i in range(400)], days=10)
    rec2 = PB.fit_numeric(s2, t2, day2, n_c=400, n_eff=400)
    assert PB.stated(rec2)


class _Node:
    def __init__(self, summ, mass):
        self.targets = {"net.pkts_down": summ}
        self._mass = mass
        self.rate_iph = None
        self.state = "confirmed"

    def mass_at(self, t):
        return self._mass

    def n_c(self, t):
        return 0.6

    def n_m(self, t):
        return 0.6


def test_content_bounds_leaves_out_records_that_state_nothing():
    s, t, day = _num([23, 24, 26, 25])
    node = _Node(s, s.td.total(t))
    store = SimpleNamespace(get_model=lambda *a, **k: None)
    eng = CBE.ContentBoundsEngine()
    ent = eng.fit_node(store, "mail", node, t, day, None, {}, (), None)
    assert "net.pkts_down" not in ent["attrs"]
    assert ent["status"] == "none"
