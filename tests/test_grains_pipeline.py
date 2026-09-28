"""spec v2.1 canonical grain mode end to end (docs/lib3/cadence.md §9, §11).

One mini-pack run (seed 0, strict, the full registry, grain_mode canonical,
the default since M8) is shared by the checks of the downstream consumers:
B04 / B06 score H rows on H decision ticks only and Q rows on Q ticks,
B14's chart moves on H ticks only, B24 keeps grain strata, B25 writes the
H-stream evidence on H ticks, B29 recomputes the opening decision exactly
(decision and q_all match the stored ones) and names the grains it
recomputed, B30 publishes per-grain workload bands, and the eval runner keeps
per-grain baseline snapshots and held-out grain rows.
"""
from __future__ import annotations

import math

import numpy as np
import pytest

from app.engines.behavior.lib import grains as GR
from app.engines.behavior.lib import m_calib
from app.engines.behavior.lib.detectors import DETECTOR_INDEX, Q_DETECTORS
from app.eval import packs as P
from app.eval.runner import run_pack

CANON = GR.CANONICAL


@pytest.fixture(scope="module")
def mini():
    pk = P.get_pack("mini")
    pk.config["grain_mode"] = "canonical"
    res = run_pack(pk, seed=0, strict=True, on_error="record", keep_store=True)
    return pk, res


def _keys(res):
    st = res.store
    return [(s, e) for s in st.systems() for e in st.entities(s)]


def test_runs_clean(mini):
    _, res = mini
    assert res.aborted is None
    assert res.exceptions == []
    assert res.stale == []
    assert res.config.get("grain_mode") == "canonical"


def test_grain_scores_only_on_decision_ticks(mini):
    pk, res = mini
    st = res.store
    n_h = n_q = 0
    for s, e in _keys(res):
        ts, Z = st.vec_since(s, e, "behavior.z", 0.0)
        for t in ts.tolist():
            dt = 3600.0 if t <= pk.scenario_start else 900.0
            assert GR.decision(t, dt, "h", CANON), (e, t)
            n_h += 1
        tq, _ = st.vec_since(s, e, "behavior.z.q", 0.0)
        for t in tq.tolist():
            assert t > pk.scenario_start and GR.decision(t, 900.0, "q", CANON)
            n_q += 1
        # B14's state ring moves on H ticks only
        tc, _ = st.vec_since(s, e, "behavior.cusum_state", 0.0)
        for t in tc.tolist():
            assert float(t) % 3600.0 == 0.0, (e, t)
        # the H-stream evidence statistic is written on H ticks only
        th, _ = st.vec_since(s, e, "behavior.evidence.h", 0.0)
        for t in th.tolist():
            assert float(t) % 3600.0 == 0.0, (e, t)
    assert n_h > 0 and n_q > 0


def test_q_provisional_weights_and_grain_strata(mini):
    _, res = mini
    st = res.store
    seen_prov = seen_strata = False
    for s, e in _keys(res):
        m = st.latest_derived(s, e, "behavior.prov")
        if m is not None and isinstance(m.value, dict):
            for d in Q_DETECTORS:
                v = m.value.get(d)
                if v is not None:
                    assert 0.0 <= float(v) <= 1.0
                    seen_prov = True
        cal = st.get_model(s, e, m_calib.MODEL)
        rings = (cal or {}).get(m_calib.RINGS) or {}
        if any("|g:h" in k for k in rings) and any("|g:q|p:" in k for k in rings):
            seen_strata = True
    assert seen_prov and seen_strata


def test_explanations_recompute_the_stored_decision(mini):
    _, res = mini
    st = res.store
    n = 0
    for inc in st.incidents():
        ex = inc.explanation or {}
        cf = ex.get("counterfactual") or {}
        sc = cf.get("scope") or {}
        fid = sc.get("fidelity") or {}
        if not fid:
            continue
        n += 1
        assert fid["decision_match"], (inc.entity, fid)
        qs, qr = fid.get("q_all_stored"), fid.get("q_all_recomputed")
        if qs is not None and qr is not None:
            assert math.isclose(qs, qr, rel_tol=1e-3, abs_tol=1e-12), (inc.entity, qs, qr)
        # stateless recomputes are exact up to the models that learn right
        # after scoring on the same tick (B06 densities, B18 class anchors:
        # B29 anchors those on the stored score) and the float32 rings
        learn_after = {"t2", "spe", "t2_q", "spe_q", "class_int", "class_shape"}
        if fid["p_log10_err"] > 0.1:
            assert set(fid.get("inexact") or {}) <= learn_after, (inc.entity, fid)
        assert set(sc.get("grains") or []) <= {"h", "q"} and sc.get("grains"), inc.entity
        if cf.get("valid"):
            assert not cf["neutralised"]["trigger"]
        for a in ex.get("attributions") or ():
            if a.get("grain") == "h" and a.get("observed_text", "").endswith("/h"):
                break
    assert n >= 3


def test_portrait_bands_per_grain(mini):
    _, res = mini
    st = res.store
    found = 0
    for s, e in _keys(res):
        por = ((st.profile(s, e) or None) and st.profile(s, e).extra.get("portrait")) or {}
        wl = (por.get("json") or {}).get("workload") or {}
        blk = wl.get("http_requests") or {}
        g = blk.get("grains") or {}
        if "h" in g and "q" in g:
            found += 1
            assert g["h"]["exposure_s"] == 3600.0 and g["q"]["exposure_s"] == 900.0
            if g["h"].get("p50") is not None and g["q"].get("p50") is not None:
                # an hour holds more requests than a quarter of it (typical)
                assert g["h"]["p95"] >= g["q"]["p95"]
            assert blk.get("exposure_s") in (None, 900.0)      # legacy band = the Q band
            assert isinstance(g["q"].get("provisional"), (bool, type(None)))
    assert found >= 3


def test_model_state_per_grain(mini):
    _, res = mini
    st = res.store
    n = 0
    for s, e in _keys(res):
        prof = st.profile(s, e)
        ms = (prof.extra or {}).get("model_state") if prof is not None else None
        if not ms or "grains" not in ms:
            continue
        n += 1
        assert ms["grains"]["h"]["dt_s"] == 3600.0
        if "q" in ms["grains"]:
            assert ms["grains"]["q"]["dt_s"] == 900.0
            assert "provisional" in ms["grains"]["q"]
            assert ms["dt_s"] == 900.0                   # legacy fields: Q when observable
    assert n >= 3


def test_holdout_grain_rows(mini):
    _, res = mini
    g = (res.holdout or {}).get("grains") or {}
    assert g.get("h"), "held-out H rows expected (gate 11 per grain)"
    for k, rows in g["h"].items():
        assert all(float(t) % 3600.0 == 0.0 for t in rows["ts"])


def test_runner_snapshots_per_grain(mini):
    _, res = mini
    snaps = (res.models or {}).get("baseline_snaps") or {}
    assert snaps, "the mini pack has malicious scenarios: snapshots expected"
    got = False
    for rec in snaps.values():
        for side in ("pre", "post"):
            for m in ((rec.get(side) or {}).get("models") or {}).values():
                mm = m.get("model") or {}
                if mm.get("grain_mode") == "canonical":
                    assert "grains" in mm and "h" in mm["grains"]
                    got = True
    assert got


def test_p_rows_hold_q_detectors_only_on_q_ticks(mini):
    pk, res = mini
    st = res.store
    iq = [DETECTOR_INDEX[d] for d in Q_DETECTORS]
    for s, e in _keys(res):
        ts, Pm = st.vec_since(s, e, "behavior.p", pk.scenario_start)
        for t, row in zip(ts.tolist(), Pm):
            if not GR.decision(t, 900.0, "q", CANON):
                assert not np.isfinite(np.asarray(row)[iq]).any(), (e, t)


def test_b14_latch_is_held_between_h_ticks(mini):
    """A B14 accumulator alarm raised on an H tick is reported on the Q / T
    ticks up to the next H tick (one continuous alarm, not an on-off train)."""
    pk, res = mini
    st = res.store
    checked = 0
    for s, e in _keys(res):
        rows = {m.ts: m.value for m in st.derived_tail(s, e, "behavior.acc_alarm", 10 ** 6)
                if isinstance(m.value, dict) and m.ts > pk.scenario_start}
        for t, v in rows.items():
            if float(t) % 3600.0 != 0.0 or not any(v.get(d) for d in ("cusum", "mcusum")):
                continue
            for k in (1, 2, 3):
                nxt = rows.get(t + k * 900.0)
                if t + k * 900.0 > pk.end_epoch:
                    break
                assert nxt is not None and any(nxt.get(d) for d in ("cusum", "mcusum")), (e, t, k)
                checked += 1
    assert checked > 0
