"""P11 who groups, round 10 (tree / groups owner): P03's damping of a source
it finds foreign weighs that source's signature EVIDENCE, not its MASS (what
the address does). O-real R3 (seed 0, round 9): 财务部's clerk moved from
192.168.2.11 to .51 on day 9; P03 damped every finance row of .51 as a foreign
source, so its signature held its (undamped) OA reading only; P11 put it with
综合部's approver on day 12 and never with 财务部, and P03 - which asks P11 who
.51's colleagues are - kept damping it: the finance statements never named
.51 (days 9-35, recall @35 lost the finance @1 rows on seeds 0-2)."""
from __future__ import annotations

import numpy as np

import test_p11_who_groups as T
from test_p11_who_groups import OrgSim, dept_ips

from app.engines.behavior.lib import pevent as EV

NEW = "192.168.2.51"


class DampSim(OrgSim):
    """OrgSim whose P03 stand-in damps every row of `damped` sources on
    `damp_sys` (pat.assign 'damp' = 0.1, as conformity.DAMP)."""

    def __init__(self, *a, damped=(), damp_sys=("finance",), **kw):
        super().__init__(*a, **kw)
        self.damped = set(damped)
        self.damp_sys = set(damp_sys)

    def tick(self) -> None:
        t1 = self.now + T.DT
        for s, evs in list(self.pending.items()):
            cur = sorted((e for e in evs if self.now < e[0] <= t1), key=lambda e: e[0])
            self.pending[s] = [e for e in evs if e[0] > t1]
            b = EV.BatchBuilder(s, EV.KIND_TXN)
            for ts, ip, attrs in cur:
                b.add(ts, ip, dict(attrs, **{"net.src": ip}))
            bt = b.build(self.now, t1)
            self.st.add_batch(s, EV.EVT_BATCH, t1, bt)
            if s in self.damp_sys and bt.n:
                d = np.asarray([0.1 if bt.ip_of(i) in self.damped else 1.0 for i in range(bt.n)])
                cols = {"damp": EV.Col(np.arange(bt.n, dtype=np.int32), d)}
                self.st.add_batch(s, EV.PAT_ASSIGN, t1, bt.aligned(cols))
        self.now = t1
        self.eng.safe_run(T.ctx(self.st, t1, window_s=T.DT, config=self.cfg), None)


def test_a_damped_source_doing_a_groups_work_joins_that_group(monkeypatch):
    """The moved clerk does 财务部's finance work (damped: P03 finds it foreign
    at the who-closed finance nodes) and 财务部's OA reading (undamped). Its
    action mix is 财务部's: it joins 财务部's group within days. Before round
    10 its signature was its OA reading alone (a tenth of its finance work):
    it was never grouped with 财务部 (it stayed ungrouped here; on O-real it
    joined 综合部's approver)."""
    prog = dict(T.PROGRAM)
    prog["FIN"] = list(prog["FIN"]) + [("oa", "GET oa /docs", 3)]
    monkeypatch.setattr(T, "PROGRAM", prog)
    sim = DampSim(damped={NEW})
    sim.days(5)
    ips = dict(dept_ips())
    ips["FIN"] = ["192.168.2.10", NEW, "192.168.2.12"]
    sim.days(7, ips=ips)
    ip2g = sim.model().get("ip2g") or {}
    assert ip2g.get("192.168.2.12") is not None
    assert ip2g.get(NEW) == ip2g.get("192.168.2.12")


def test_damping_still_slows_the_evidence_of_a_foreign_source(monkeypatch):
    """Damping keeps its weight on the EVIDENCE: a source all of whose rows
    are damped accumulates a tenth of an undamped one's evidence units (it
    needs days of sustained work before it is clustered or joins)."""
    sim = DampSim(damped={NEW})
    ips = dict(dept_ips())
    ips["FIN"] = ["192.168.2.10", "192.168.2.11", "192.168.2.12", NEW]
    sim.days(2, ips=ips)
    st = sim.st.get_model(T.ORG, T.ORG, T.WG.STATE)
    a, b = st.sigs.get(NEW), st.sigs.get("192.168.2.11")
    assert a is not None and b is not None
    assert a.ev < 0.2 * b.ev
    # the mix is the clerk's: its finance share equals an undamped colleague's
    pa, pb = st.sigs.profile(NEW, sim.now), st.sigs.profile("192.168.2.11", sim.now)
    fa = sum(v for k, v in pa.items() if (st.sigs.items.key_of(int(k)) or "").startswith("finance|")) / sum(pa.values())
    fb = sum(v for k, v in pb.items() if (st.sigs.items.key_of(int(k)) or "").startswith("finance|")) / sum(pb.values())
    assert abs(fa - fb) < 0.15
