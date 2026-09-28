"""B25 FusionEngine: meta-ring learning through lib/gating (contract H), the
evidence audit and cadence handling (900 s -> 60 s).

The meta rings (model.calib['meta'], keys 'meta_inst@<stratum>' /
'meta_all@<stratum>') must learn late (row t - D), from trusted rows only,
hold while quarantined, and obey release / rollback_to / rebase_from /
frozen and link seeding from model.control / model.link.
"""
from __future__ import annotations

import math
from typing import Dict, List

import numpy as np
import pytest

from helpers import DT, put_model, set_trust

from app.engines.behavior import fusion as F
from app.engines.behavior.lib import calib, m_calib

from test_b25_fusion import E, S, Rig, p_at

NULL = {"marg_int": 0.4, "novelty": 0.6}


def meta(rig: Rig, e: str = E) -> Dict:
    return rig.store.get_model(S, e, m_calib.MODEL)["meta"]


def ring_sizes(rig: Rig, e: str = E) -> Dict[str, int]:
    return {k: len(r) for k, r in F.meta_rings(meta(rig, e)).items()}


def ring_ts(rig: Rig, e: str = E) -> np.ndarray:
    rs = F.meta_rings(meta(rig, e)).values()
    return np.concatenate([r.ts for r in rs]) if rs else np.empty(0)


def run_null(rig: Rig, n: int, **kw) -> List[float]:
    rng = np.random.default_rng(int(rig.t) % 1000)
    out = []
    for _ in range(n):
        out.append(rig.step({E: {"marg_int": float(rng.random()),
                                 "novelty": float(rng.random())}}, **kw))
    return out


def test_commit_delay_and_trust_weight():
    rig = Rig()
    ts = run_null(rig, 10)
    st = meta(rig)["state"]
    assert st["n_admit"] == 10 - 4                   # D = 4 ticks at 900 s
    assert ring_ts(rig).max() == ts[-5]
    for k, n in ring_sizes(rig).items():
        assert k.startswith(("meta_inst@", "meta_all@"))
    # the model is shared with B24: its own keys are never created here
    assert set(rig.store.get_model(S, E, m_calib.MODEL)) == {"meta"}


def test_live_meta_admission_is_the_period_trust_not_the_row_trust():
    """Round 4 (lib/gating.period_weight): behavior.trust carries the row's
    own evidence (its q_inst, its alarm), so a meta ring admitting by it
    truncates its own tail. Live rows are admitted whatever their trust; a
    missing or NaN governor tick admits nothing; quarantine holds."""
    rig = Rig()
    run_null(rig, 12, trust=0.0)
    assert meta(rig)["state"]["n_admit"] == 8                  # D = 4: every committed row
    rig = Rig()
    run_null(rig, 104, trust=0.5)
    assert meta(rig)["state"]["n_admit"] == 100
    rig = Rig()
    run_null(rig, 12, trust=float("nan"))
    assert meta(rig)["state"]["n_admit"] == 0 and sum(ring_sizes(rig).values()) == 0
    assert len(meta(rig)["state"]["gate"].journal) == 8       # committed with w = 0
    rig = Rig()
    run_null(rig, 12, trust=None)                              # no governor at all: fail safe
    assert meta(rig)["state"]["n_admit"] == 0


def test_training_trusts_missing_rows_but_not_explicit_zero():
    rig = Rig()
    run_null(rig, 8, trust=None, training=True)
    assert meta(rig)["state"]["n_admit"] == 4
    rig = Rig()
    run_null(rig, 8, trust=0.0, training=True)
    assert meta(rig)["state"]["n_admit"] == 0


def test_quarantine_holds_and_release_commits():
    rig = Rig()
    ts = run_null(rig, 8, quarantine=1.0)
    g = meta(rig)["state"]["gate"]
    assert meta(rig)["state"]["n_admit"] == 0 and len(g.held) == 4
    put_model(rig.store, S, E, "model.control", {"version": 0, "release": [ts[0], ts[3]]})
    run_null(rig, 1, quarantine=1.0)
    st = meta(rig)["state"]
    assert st["n_admit"] == 4
    assert sorted(set(ring_ts(rig).tolist())) == ts[:4]


def test_rollback_deletes_entries_after_onset():
    rig = Rig()
    ts = run_null(rig, 23)
    ts += run_null(rig, 1, quarantine=1.0)           # B28: quarantine at/before rollback_to
    assert ring_ts(rig).max() == ts[-5]
    tau = ts[8]
    put_model(rig.store, S, E, "model.control", {"version": 0, "rollback_to": tau})
    run_null(rig, 1, quarantine=1.0)                 # governor quarantines at the rollback
    assert ring_ts(rig).max() <= tau
    g = meta(rig)["state"]["gate"]
    assert g.applied["_last_rollback"]["removed"] > 0
    assert g.held and min(r.ts for r in g.held) > tau
    # release afterwards puts them back
    put_model(rig.store, S, E, "model.control",
              {"version": 0, "rollback_to": tau, "release": [tau + 1, ts[-1]]})
    run_null(rig, 1)
    assert ring_ts(rig).max() > tau


def test_frozen_stops_commits():
    rig = Rig()
    run_null(rig, 8)
    n0 = meta(rig)["state"]["n_admit"]
    put_model(rig.store, S, E, "model.control", {"version": 0, "frozen": True})
    run_null(rig, 6)
    assert meta(rig)["state"]["n_admit"] == n0


def test_rebase_and_version_change_reset_rings():
    rig = Rig()
    ts = run_null(rig, 12)
    assert sum(ring_sizes(rig).values()) > 0
    put_model(rig.store, S, E, "model.control", {"version": 1, "rebase_from": ts[-1]})
    run_null(rig, 1)
    st = meta(rig)["state"]
    assert st["resets"] == 1
    assert ring_ts(rig).size == 0 or ring_ts(rig).min() >= ts[-1] - 4 * DT
    # a bare version change (no rebase_from) also resets
    rig = Rig()
    put_model(rig.store, S, E, "model.control", {"version": 0})
    run_null(rig, 10)
    assert sum(ring_sizes(rig).values()) > 0
    put_model(rig.store, S, E, "model.control", {"version": 2})
    run_null(rig, 1)
    assert meta(rig)["state"]["resets"] == 1
    assert sum(ring_sizes(rig).values()) == 0        # reset after this tick's commit
    run_null(rig, 1)
    assert sum(ring_sizes(rig).values()) == 2        # learning resumes in version 2


def test_link_seeding_merges_half_of_the_source():
    A, B = "10.0.0.8", "10.0.0.9"
    rig = Rig(entities=(A, B))
    rng = np.random.default_rng(5)
    for _ in range(40):
        rig.step({A: {"marg_int": float(rng.random())}, B: {"marg_int": float(rng.random())}})
    before = ring_sizes(rig, B)
    put_model(rig.store, S, "__system__", "model.link",
              {"version": 1, "links": [{"from": A, "to": B, "ts": rig.t}]})
    rig.step({A: {"marg_int": 0.5}, B: {"marg_int": 0.5}})
    after = ring_sizes(rig, B)
    src = ring_sizes(rig, A)
    for k, n in before.items():
        expect = min(calib.RING_M, n + 1 + min(src[k], calib.RING_M // 2))
        assert after[k] in (expect, expect - 1, expect + 1)


def test_meta_tail_uses_predictive_xi_floor():
    rng = np.random.default_rng(2)
    r = calib.Ring()
    for i, x in enumerate(rng.exponential(1.0, 256)):
        r.add(float(x), float(i))
    t = F.meta_tail(r)
    n_u = 256 - 1 - int(math.floor(calib.TAIL_Q * 255))
    assert t is not None and t.xi >= 1.0 / n_u - 1e-12
    assert F.meta_tail(calib.Ring()) is None


def test_evidence_audit_raises_h_when_rate_too_high():
    """Round 4: the audit solves h from the entity's own period-trusted q_inst
    history (fusion.solve_h) and moves h_mult towards it by at most
    H_MULT_STEP_UP per audit, never above H_MULT_MAX; quarantined ticks are
    not part of the null stream."""
    rig = Rig()
    now = rig.t + 3 * 86400
    rng = np.random.default_rng(4)
    t = rig.t
    ts_hist = []
    while t < now:                                    # 3 days of skewed q_inst history
        rig.store.add_vec(S, E, F.Q_INST, t, np.asarray([rng.random() ** 4], np.float32),
                          window_s=900)
        ts_hist.append(t)
        t += DT
    set_trust(rig.store, S, E, ts_hist, 0.0, quarantine=0.0)   # row trust 0: still null rows
    rig.t = now
    rig.step({E: NULL})
    st = meta(rig)["state"]
    assert st["audit"] is not None and st["audit"]["rate"] > 2.0 / F.EVIDENCE_ARL_DAYS
    assert st["audit"]["h_star"] > F.evidence_h(900)
    assert st["h_mult"] == pytest.approx(1.0 + F.H_MULT_STEP_UP)
    # the audit is hourly: the next tick does not audit again
    rig.step({E: NULL})
    assert meta(rig)["state"]["h_mult"] == pytest.approx(1.0 + F.H_MULT_STEP_UP)
    # never more than H_MULT_MAX
    st["h_mult"] = F.H_MULT_MAX
    rig.t += 3600
    rig.step({E: NULL})
    assert meta(rig)["state"]["h_mult"] <= F.H_MULT_MAX + 1e-12
    # the raised h is the one the CUSUM uses
    for _ in range(12):
        rig.step({E: {"novelty": 1e-4}}, trust=None)
    a = rig.alarm()
    assert a["h"] == pytest.approx(meta(rig)["state"]["h_mult"] * F.evidence_h(900))


def test_evidence_audit_ignores_quarantined_history():
    rig = Rig()
    now = rig.t + 3 * 86400
    rng = np.random.default_rng(4)
    t = rig.t
    ts_hist = []
    while t < now:
        rig.store.add_vec(S, E, F.Q_INST, t, np.asarray([rng.random() ** 4], np.float32),
                          window_s=900)
        ts_hist.append(t)
        t += DT
    set_trust(rig.store, S, E, ts_hist, 1.0, quarantine=1.0)
    rig.t = now
    rig.step({E: NULL})
    st = meta(rig)["state"]
    assert st.get("audit") is None and float(st.get("h_mult", 1.0)) == 1.0


def test_audit_rate_on_null_is_near_target():
    rng = np.random.default_rng(11)
    q = rng.random(int(7 * 86400 / 900))
    rate = F.audit_rate(q, F.evidence_h(900), 900.0, np.random.default_rng(0), days=2000)
    # a 7-day bootstrap cannot produce q beyond the sample's own extremes, so
    # it under-estimates the null rate: the audit never raises h on a null
    assert 0.0 <= rate <= 2.0 / F.EVIDENCE_ARL_DAYS
    assert math.isnan(F.audit_rate(q[:10], 5.0, 900.0, rng))


def test_cadence_switch_900_to_60():
    rig = Rig()
    for _ in range(3):
        rig.step({E: {"novelty": 0.01}}, trust=None)
    S900 = rig.v(F.EVIDENCE)
    assert S900 == pytest.approx(3 * (-math.log(0.01) - 3.0), rel=1e-5)
    # switch: the CUSUM state carries over, h and e_day follow the new dt
    rig.step({E: {"novelty": 0.01}}, trust=None, dt=60.0)
    a = rig.alarm()
    assert rig.v(F.EVIDENCE) == pytest.approx(4 * (-math.log(0.01) - 3.0), rel=1e-5)
    assert a is None                                  # 6.42 < h(60) = 8.19

    rig.step({E: {"novelty": 0.01}}, trust=None, dt=60.0)
    assert rig.alarm() is None                        # 8.03 < 8.19
    rig.step({E: {"novelty": 0.01}}, trust=None, dt=60.0)
    a = rig.alarm()
    assert a["path"] == F.PATH_EVIDENCE and a["h"] == pytest.approx(F.evidence_h(60.0))
    assert rig.v(F.E_DAY) == pytest.approx(rig.v(F.Q_ALL) * 86400 / 60.0, rel=1e-5)
    # meta rings of the new cadence class are separate strata
    rig2 = Rig()
    run_null(rig2, 8)
    for _ in range(14):
        rig2.step({E: NULL}, dt=60.0)
    keys = set(ring_sizes(rig2))
    assert any(k.endswith("|900") for k in keys) and any(k.endswith("|60") for k in keys)
    # e_day severity is cadence free: the same e_day gives the same level
    rig3 = Rig(dt=60.0)
    rig3.step({E: {"jsd": p_at(2e-3, 60.0)}})
    assert rig3.alarm()["severity"] == "medium"


def test_canonical_60s_t_meta_strata_are_thinned_and_start_on_a_conservative_prior():
    """Round 4, canonical mode below 900 s: the per-tick-type meta strata
    (meta_all at tau = t, meta_inst_t) admit one row per 900-s slot (a rule
    on ts), and while young their prior is the raw p to the power 1/v with v
    learned from the stratum's own committed rows (prior v = 2)."""
    rig = Rig()
    cfg = {"grain_mode": "canonical"}
    rig.t = (int(rig.t) // 3600 + 1) * 3600.0
    rng = np.random.default_rng(3)
    for _ in range(8):
        rig.step({E: {"novelty": float(rng.random()), "timing": float(rng.random())}},
                 dt=900.0, training=True, config=cfg)
    n_ticks = 600
    for _ in range(n_ticks):
        rig.step({E: {"novelty": float(rng.random()), "timing": float(rng.random())}},
                 dt=60.0, config=cfg)
    m = meta(rig)
    sizes = {k: len(r) for k, r in F.meta_rings(m).items() if "|t:t|60" in k}
    assert sizes, sizes
    for k, n in sizes.items():
        assert n <= n_ticks // 15 + 2, (k, n)                 # one per 900-s slot
    xs = m["state"].get(F.XFER) or {}
    assert any("|t:t|60" in k and v[1] >= 0.8 * n_ticks for k, v in xs.items() if "meta_all" in k
               or "meta_inst_t" in k)
    # the young-stratum prior: raw p^(1/v)
    st = m["state"]
    key = next(k for k in sizes if k.startswith(F.META_ALL))
    v = m_calib.xfer_v(xs.get(key))
    assert rig.eng._t_prior(st, F.META_ALL, key.split("@", 1)[1], None, 1e-4) == \
        pytest.approx(1e-4 ** (1.0 / v))
