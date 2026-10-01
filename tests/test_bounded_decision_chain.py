"""Bounded decision chain (docs/lib3/progressive.md §10.1-§10.3): in
lib3.resource_mode = bounded the B24-B29 state is kept per IP only where it
is earned; unearned IPs calibrate against their class's pooled rings, and
the per-IP state of an IP idle for 7 days is released by P15."""
from __future__ import annotations

from collections import OrderedDict

from app.core.store import MetricStore
from app.engines.behavior import calibration as CAL
from app.engines.behavior.lib import calib, m_calib
from app.engines.behavior.lib import pactive as PA
from app.engines.behavior.resource_governor import (RELEASE_IDLE_S, GovState,
                                                    ResourceGovernorEngine)
from app.models.schema import ORG
from app.pipeline import build

T0 = 1_790_000_000.0
BOUNDED = {"lib3": {"resource_mode": "bounded", "pool_unearned": True}}


def test_bounded_mode_registers_the_resource_governor():
    """Without P15 the active / earned sets are never published and
    pactive falls back to every entity: 'bounded' was 'full'."""
    names = [e.name for e in build.build_registry(config=BOUNDED)._engines]
    assert "behavior.resource_governor" in names
    assert names.index("behavior.resource_governor") < names.index("behavior.calibration")
    assert "behavior.resource_governor" not in [e.name for e in build.build_registry()._engines]


def _store_with_sets(earned, active):
    st = MetricStore()
    st.put_model(ORG, ORG, PA.BUDGET, {"systems": {"oa": {"ts": T0, "earned": list(earned),
                                                          "active": list(active)}}}, ts=T0)
    return st


def test_unearned_ips_share_their_class_pool_and_earned_keep_their_own():
    st = _store_with_sets(earned=["10.0.0.1"], active=["10.0.0.1", "10.0.0.2", "10.0.0.3"])
    eng = CAL.CalibrationEngine()
    eng._config = BOUNDED
    pool = CAL._PoolCache(st, "oa")
    models = {}
    for ip in ("10.0.0.1", "10.0.0.2", "10.0.0.3"):
        m = CAL.new_model()
        eng._bounded_rings(st, "oa", ip, m, pool, T0)
        models[ip] = m
    # the two unearned IPs share ONE rings dict, the pool's (system pool: no class, no group)
    assert models["10.0.0.2"][CAL.RINGS] is models["10.0.0.3"][CAL.RINGS]
    pm = st.get_model("oa", PA.POOL_ENTITY + "*", CAL.MODEL)
    assert pm is not None and pm[CAL.RINGS] is models["10.0.0.2"][CAL.RINGS]
    assert models["10.0.0.1"][CAL.RINGS] is not pm[CAL.RINGS]
    # admission by one unearned IP is seen by the other (class null)
    r = models["10.0.0.2"][CAL.RINGS].setdefault("x@d", calib.Ring())
    r.add(1.0, T0)
    assert len(models["10.0.0.3"][CAL.RINGS]["x@d"]) == 1
    # promotion: own rings warm-started from a copy of the pool (later pool
    # admissions are not the promoted IP's)
    st2 = _store_with_sets(earned=["10.0.0.1", "10.0.0.2"], active=[])
    PA._CACHE.clear()
    eng._bounded_rings(st2, "oa", "10.0.0.2", models["10.0.0.2"], CAL._PoolCache(st2, "oa"), T0)
    own = models["10.0.0.2"][CAL.RINGS]
    assert own is not pm[CAL.RINGS] and len(own["x@d"]) == 1
    pm[CAL.RINGS]["x@d"].add(2.0, T0 + 1)
    assert len(own["x@d"]) == 1
    assert CAL.POOLED not in models["10.0.0.2"]


def test_full_mode_never_pools():
    st = _store_with_sets(earned=[], active=["10.0.0.2"])
    eng = CAL.CalibrationEngine()
    eng._config = {}
    m = CAL.new_model()
    eng._bounded_rings(st, "oa", "10.0.0.2", m, CAL._PoolCache(st, "oa"), T0)
    assert CAL.POOLED not in m and not st.model_names("oa", PA.POOL_ENTITY + "*")


def test_idle_unearned_ip_state_is_released_after_seven_days():
    """P15 releases model.calib / model.governor / model.control of an IP not
    seen for 7 days unless it is earned or active (open incident, regime
    event); the work follows the tick's sources (an LRU), not the known IPs."""
    st = MetricStore()
    gov, eng = GovState(), ResourceGovernorEngine()
    for ip in ("10.0.0.1", "10.0.0.2", "10.0.0.3"):
        for n in ("model.calib", "model.governor", "model.control"):
            st.put_model("oa", ip, n, {"x": 1}, ts=T0)
    eng._release_idle(st, gov, {"oa": {"10.0.0.1", "10.0.0.2", "10.0.0.3"}}, {}, T0)
    t = T0 + RELEASE_IDLE_S - 3600.0
    eng._release_idle(st, gov, {"oa": {"10.0.0.3"}}, {}, t)
    assert all(st.get_model("oa", ip, "model.calib") is not None for ip in ("10.0.0.1", "10.0.0.2"))
    t2 = T0 + RELEASE_IDLE_S + 3600.0
    n = eng._release_idle(st, gov, {"oa": set()},
                          {"oa": {"earned": ["10.0.0.1"], "active": []}}, t2)
    assert n == 3
    assert st.get_model("oa", "10.0.0.2", "model.calib") is None
    assert st.get_model("oa", "10.0.0.2", "model.control") is None
    assert st.get_model("oa", "10.0.0.1", "model.calib") is not None      # earned: kept
    assert st.get_model("oa", "10.0.0.3", "model.calib") is not None      # seen 1 h before the window
    assert isinstance(gov.keep["oa"], OrderedDict) and "10.0.0.2" not in gov.keep["oa"]


def test_unearned_ips_share_class_meta_rings_in_fusion():
    from app.engines.behavior import fusion as FU
    st = _store_with_sets(earned=["10.0.0.1"], active=["10.0.0.1", "10.0.0.2", "10.0.0.3"])
    PA._CACHE.clear()
    eng = FU.FusionEngine()
    metas = {}
    for ip in ("10.0.0.2", "10.0.0.3", "10.0.0.1"):
        model = {}
        meta = FU._ensure_meta(model)
        eng._bounded_meta(st, "oa", ip, meta, T0, BOUNDED)
        metas[ip] = (model, meta)
    r2, _ = FU._mr(metas["10.0.0.2"][1])
    r3, _ = FU._mr(metas["10.0.0.3"][1])
    r1, _ = FU._mr(metas["10.0.0.1"][1])
    assert r2 is r3 and r1 is metas["10.0.0.1"][1]
    # learning by one unearned IP lands in the class pool; readers (B29 via
    # m_calib.ring) find it for the other
    row = FU._MRow("oa", "10.0.0.2", T0, 1.5, 2.0, "day|900", None, None)
    eng._update(metas["10.0.0.2"][1], row, 1.0)
    key = eng._ring_key(FU.META_ALL, "day|900")
    assert key in r3 and FU.meta_rings(metas["10.0.0.3"][1]) == {}
    assert m_calib.ring(metas["10.0.0.3"][0], FU.META_ALL, "day|900") is r3[key]
    # promotion: own rings from a copy of the pool
    st2 = _store_with_sets(earned=["10.0.0.1", "10.0.0.3"], active=[])
    PA._CACHE.clear()
    eng._bounded_meta(st2, "oa", "10.0.0.3", metas["10.0.0.3"][1], T0, BOUNDED)
    own = FU.meta_rings(metas["10.0.0.3"][1])
    assert key in own and own[key] is not r2[key]
    assert FU._mr(metas["10.0.0.3"][1])[0] is metas["10.0.0.3"][1]


def test_pooling_is_off_by_default_in_bounded_mode():
    """Measured (pack E seed 0, T5' lost its CRITICAL under the class pool):
    bounded mode keeps per-IP rings for active IPs unless lib3.pool_unearned."""
    st = _store_with_sets(earned=[], active=["10.0.0.2"])
    PA._CACHE.clear()
    assert not PA.pooled(st, "oa", "10.0.0.2", {"lib3": {"resource_mode": "bounded"}})
    assert PA.pooled(st, "oa", "10.0.0.2", BOUNDED)
