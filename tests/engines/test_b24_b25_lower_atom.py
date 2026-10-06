"""Round 5: B24 / B25 at the lowest level of a ring (lib/calib "Lower atom").

PG5 D4 (pack O seeds 3 / 4; §16.12.15 item 4): the health monitor's detector
scores sat at their atom (score 0, pm = 1) on every tick, B24 issued the
seeded coin flip p = U on its all-zero ring and B25's single-tick path opened
an incident when U happened to be ~1e-4 (oa, 1758132000, conf_who p = 7.9e-5,
on every seed). Now a tick at the atom issues p = 1 (B24) and q = 1 (B25),
while the calibration monitors - B24's health KS / rate and live power
correction, B25's ACI and CUSUM-input calibration - observe the randomised p,
so a calibrated detector is not read as conservative and no threshold is
relaxed to spend the null budget elsewhere.
"""
from __future__ import annotations

import numpy as np

from app.engines.behavior import fusion as F
from app.engines.behavior.lib import calib, m_calib
from app.engines.behavior.lib.classkeys import SYSTEM_KEY
from app.engines.behavior.lib.detectors import DETECTORS

from test_b24_calibration import E, S, Rig as CalRig, ks
from test_b25_fusion import Rig as FusRig


def test_b24_a_source_at_its_atom_on_every_tick_issues_one():
    """The health monitor: score 0, pm = 1 on 400 ticks. Every issued p is 1
    (it was U: min over 400 ticks ~ 1/400, alarming at ~1e-4 now and then);
    the health check sees the randomised p, uniform."""
    rig = CalRig(daypart="wd_day")
    ps = []
    for _ in range(400):
        ts = rig.step({E: {"novelty": 0.0}}, pm={E: {"novelty": 1.0}})
        ps.append(rig.p(E, "novelty", ts))
    assert np.all(np.asarray(ps) == 1.0)
    st = calib.stratum_key("wd_day", 900)
    assert m_calib.ring_size(rig.model(), "novelty", st) >= 256    # an all-zero ring
    hs = rig.store.get_model(S, SYSTEM_KEY, m_calib.MODEL)[m_calib.HEALTH]
    hp = np.asarray(list(hs["ks"][DETECTORS.index("novelty")]), dtype=np.float64)
    assert hp.size >= 300 and hp.max() < 1.0 and ks(hp) < 0.1
    # PRAND holds only the rows not yet committed (popped at commit, pruned at 1 d)
    assert len(rig.model().get("prand", {})) <= 8
    # then a real deviation is still evidence: above every zero, p <= 1/(n+1)
    ts = rig.step({E: {"novelty": 3.0}}, pm={E: {"novelty": 1e-3}})
    assert rig.p(E, "novelty", ts) <= 1.0 / 257


def test_b24_pcal_observes_the_randomised_p():
    """The live power correction counts the share of p <= 0.05 among trusted
    live rows; on an atom-only detector that share is the randomised p's
    (~0.05), not 0 (the issued 1)."""
    rig = CalRig(daypart="wd_day")
    for _ in range(300):
        rig.step({E: {"novelty": 0.0}}, pm={E: {"novelty": 1.0}})
    pc = rig.store.get_model(S, SYSTEM_KEY, m_calib.MODEL)[m_calib.PCAL]
    st = pc[m_calib.pcal_key("novelty", 900)]
    share = st[0] / st[1]
    assert 0.02 < share < 0.09


def _fusion_atom_run(n_keys: int = 10, ticks: int = 300):
    keys = [f"10.0.0.{i + 1}" for i in range(n_keys)]
    rig = FusRig(entities=keys)
    q = []
    for k in range(ticks):
        rig.step({e: {"marg_int": 1.0, "novelty": 1.0, "t2": 1.0} for e in keys})
        q.extend(rig.v(F.Q_ALL, e) for e in keys)
        for e in keys:
            assert rig.alarm(e) is None
    sysm = rig.store.get_model(S, SYSTEM_KEY, m_calib.MODEL)
    aci = (sysm or {}).get(F.ACI) or {}
    return np.asarray(q), aci, aci.get(F.QCAL) or {}


def test_b25_all_p_one_issues_q_one_and_aci_does_not_relax():
    q, aci, qcal = _fusion_atom_run()
    # every tick of a key whose detectors are all at their atom: q_all = 1
    # (the meta ring holds only -log10 1 = 0: its lowest level)
    assert np.all(q == 1.0)
    # ACI observed the randomised q: its system shift stays near 0 at every
    # level (observing the issued 1 it never sees an error and walks to -1)
    th = aci["th"]["tick|900"]
    assert aci["n"] > 2000
    assert all(abs(v) < 0.4 for v in th if v is not None), th
    # the CUSUM-input calibration saw the randomised q (share ~0.05)
    st = qcal["inst|900"]
    assert 0.02 < st[0] / st[1] < 0.09


def test_b24_prand_survives_a_json_restore():
    """A model restored from JSON (checkpoint / export) keeps the pending
    randomised p (str keys -> float ts / int detector index) and the health
    check reads it at the row's commit."""
    import json

    rig = CalRig(daypart="wd_day")
    for _ in range(80):
        rig.step({E: {"novelty": 0.0}}, pm={E: {"novelty": 1.0}})
    m = rig.model()
    pend = dict(m.get("prand") or {})
    assert pend and all(isinstance(t, float) for t in pend)
    blob = m_calib.to_json(m)
    blob["profile_ts"] = None
    blob["dt"] = None
    rig.store.put_model(S, E, m_calib.MODEL, json.loads(json.dumps(blob, allow_nan=False)))
    hs = rig.store.get_model(S, SYSTEM_KEY, m_calib.MODEL)[m_calib.HEALTH]
    i = DETECTORS.index("novelty")
    n0 = len(hs["ks"][i])
    for _ in range(6):
        rig.step({E: {"novelty": 0.0}}, pm={E: {"novelty": 1.0}})
    m = rig.model()
    assert all(isinstance(t, float) for t in m.get("prand", {}))
    assert all(isinstance(k, int) for c in m.get("prand", {}).values() for k in c)
    new = list(hs["ks"][i])[n0:]
    assert len(new) >= 5 and max(new) < 1.0          # the randomised p, not the issued 1



def test_b25_decides_on_the_issued_row_and_learns_the_randomised_one():
    """B24 issues p = 1 at a detector's atom and keeps its randomised p in
    model.calib['prand'][ts]. B25 must DECIDE on the issued row (q_all = 1:
    no coin flip reaches the single-tick path) and LEARN its meta rings from
    the randomised row (rand_row), so the rings hold the null they held
    before round 5 and q_dec >= q_rand pointwise."""
    rng = np.random.default_rng(12)
    rig = FusRig()
    idx = [DETECTORS.index(d) for d in ("marg_int", "novelty", "t2")]
    model = {}
    rig.store.put_model(S, E, m_calib.MODEL, model, version=0, ts=rig.t)
    qs, scores = [], []
    for _ in range(300):
        alt = {i: float(rng.random()) for i in idx}
        model.setdefault("prand", {})[rig.t] = alt
        rig.step({E: {"marg_int": 1.0, "novelty": 1.0, "t2": 1.0}})
        qs.append(rig.v(F.Q_ALL))
        assert rig.alarm() is None
    assert np.all(np.asarray(qs) == 1.0)                     # decisions: no evidence
    meta = rig.store.get_model(S, E, m_calib.MODEL)["meta"]
    ring = next(r for k, r in F.meta_rings(meta).items() if k.startswith(F.META_ALL))
    assert len(ring) >= 60
    assert np.mean(ring.scores > 0.0) > 0.95                 # learned the randomised scores
    # q of the randomised row against those rings is uniform; the issued one is 1
    assert F.rand_row(model, np.ones(len(DETECTORS)), -1.0) is None
    sysm = rig.store.get_model(S, SYSTEM_KEY, m_calib.MODEL)
    th = sysm[F.ACI]["th"]["tick|900"]
    assert all(abs(v) < 0.4 for v in th if v is not None), th
