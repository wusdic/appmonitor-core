"""B29 ExplainEngine (docs/lib3/engines.md '## B29'): the spec unit test plus
the replay helpers it relies on.

Spec test: an incident where bytes_up and updown_log are perturbed by about
+4 sigma for 12 ticks (so the CUSUM alarms) and a new SNI appears.
  * the top-3 attributions include {bytes_up, updown_log};
  * new_tokens contains the SNI at class tier;
  * the counterfactual set is a subset of the perturbed features and the
    counterfactual is valid: the replayed CUSUM stays below h and the fused
    decision is recomputed as no incident;
  * the narrative contains the natural-unit range.

The incident is produced by the real decision chain, not hand-written: B03
baseline -> B04 likelihood -> B05 common mode -> B06 multivariate -> B14
changepoint -> B24 calibration -> B25 fusion -> B26 risk -> B27 incident ->
B28 governor -> B29 explain, on synthetic feature.nat rows of three
entities (5 days of 900-s warm-up in training mode, then live ticks). Only
B08 is played by the test (the first_seen finding of the new SNI at class
tier), since its input would need the raw TLS sets and a peer class.
"""
from __future__ import annotations

import copy
import math
import re

import numpy as np
import pytest

from helpers import DT, T0, make_store, run_engine

from app.core.engine import default_config
from app.engines.behavior.baseline import BaselineEngine
from app.engines.behavior.calibration import CalibrationEngine
from app.engines.behavior.changepoint import ChangepointEngine
from app.engines.behavior.common_mode import CommonModeEngine
from app.engines.behavior.explain import ExplainEngine, bh_flags, fmt_range, gk_shares
from app.engines.behavior.fusion import FusionEngine
from app.engines.behavior.governor import GovernorEngine
from app.engines.behavior.incident import IncidentEngine
from app.engines.behavior.lib import features as F
from app.engines.behavior.lib import m_calib, m_cp
from app.engines.behavior.lib import replay as RP
from app.engines.behavior.lib import timebins as TB
from app.engines.behavior.likelihood import LikelihoodEngine
from app.engines.behavior.multivariate import MultivariateEngine
from app.engines.behavior.risk import RiskEngine
from app.eval.metrics import has_natural_range
from app.models.schema import BehaviorEvent, DerivedMetric, MetricKind, Severity

S = "erp"
ENTS = ("10.0.0.1", "10.0.0.2", "10.0.0.3")
E = ENTS[0]
IDX = F.FEATURE_INDEX
NF = F.FEATURE_DIM
CFG = default_config({"strict": True})
TRAIN_TICKS = 5 * 96
CLEAN_TICKS = 8
PERTURB_TICKS = 12
TRAIN_T0 = T0 - 6 * 86400.0
SD_LOG = 0.3                                   # log-sd of the byte volumes
SHIFT = 4.0 * math.sqrt(2.0) * SD_LOG          # +4 sigma of updown_log (bytes_up +5.7 sigma)
SNI = "evil-cdn.example"
PERTURBED = {"bytes_up", "updown_log"}


def base_row(rng: np.random.Generator, shift: float = 0.0) -> np.ndarray:
    """feature.nat of one 900-s tick (natural units; unused features NaN)."""
    x = np.full(NF, np.nan)
    down = 1.0e6 * math.exp(SD_LOG * rng.standard_normal())
    up = 2.0e5 * math.exp(SD_LOG * rng.standard_normal())
    up_obs = up * math.exp(shift)
    flows = float(rng.poisson(40))
    req = float(rng.poisson(60))
    x[IDX["bytes_up"]] = round(up_obs)
    x[IDX["bytes_down"]] = round(down)
    x[IDX["flows"]] = flows
    x[IDX["http_requests"]] = req
    x[IDX["dns_queries"]] = float(rng.poisson(10))
    x[IDX["tls_handshakes"]] = float(rng.poisson(8))
    x[IDX["intensity"]] = req
    # bytes_per_flow is kept at its unperturbed value: only bytes_up and
    # updown_log are the perturbed features of this test
    x[IDX["bytes_per_flow"]] = (up + down) / max(flows, 1.0)
    x[IDX["updown_log"]] = math.log((up_obs + 1.0) / (down + 1.0))
    x[IDX["distinct_peers"]] = float(rng.poisson(5))
    x[IDX["distinct_templates"]] = float(rng.poisson(6))
    x[IDX["http_write_ratio"]] = float(rng.binomial(int(req), 0.1)) / max(req, 1.0)
    x[IDX["http_4xx_rate"]] = float(rng.binomial(int(req), 0.02)) / max(req, 1.0)
    return x


def feed(store, t: float, nat: np.ndarray, e: str) -> None:
    store.add_vec(S, e, "feature.nat", t, np.asarray(nat, dtype=np.float32), window_s=int(DT))
    store.add_vec(S, e, "feature.active", t, np.array([1.0], dtype=np.float32),
                  window_s=int(DT))
    store.register_entity(S, e)
    store.add_derived(DerivedMetric(name="feature.tctx", value=TB.tctx_from_config(t, CFG, DT),
                                    ts=t, system=S, entity=e, window_s=int(DT),
                                    kind=MetricKind.CATEGORICAL))


def engines():
    return [BaselineEngine(), LikelihoodEngine(), CommonModeEngine(), MultivariateEngine(),
            ChangepointEngine(), CalibrationEngine(), FusionEngine(), RiskEngine(),
            IncidentEngine(), GovernorEngine(), ExplainEngine()]


class Run:
    """The chain run once for the module; explanations are snapshotted
    every time B29 rewrites one."""

    def __init__(self) -> None:
        self.store = make_store()
        self.eng = engines()
        self.explain = self.eng[-1]
        rng = np.random.default_rng(7)
        self.history = []                         # [(tick index, ts, incident id, expl copy)]
        t = TRAIN_T0
        for _ in range(TRAIN_TICKS):
            self._tick(t, rng, training=True)
            t += DT
        self.t_live = t
        for _ in range(CLEAN_TICKS):
            self._tick(t, rng)
            t += DT
        self.t_perturb = t
        for i in range(PERTURB_TICKS):
            self._tick(t, rng, shift=SHIFT, sni=(i == 2), idx=i)
            t += DT
        self.t_end = t - DT

    def _tick(self, t: float, rng, training: bool = False, shift: float = 0.0,
              sni: bool = False, idx: int = -1) -> None:
        for e in ENTS:
            feed(self.store, t, base_row(rng, shift if e == E else 0.0), e)
        if sni:                                   # B08's finding (played, see module doc)
            self.store.add_event(BehaviorEvent(
                system=S, entity=E, ts=t, kind="first_seen", score=0.5, severity=Severity.LOW,
                description=f"first seen: sni {SNI} (new to the class)",
                extra={"dim": "sni", "value": SNI, "tier": "class", "bits": 14.0,
                       "adopted": False, "flags": {"external": True}},
                axes=["categorical"], p_by_detector={"novelty": 1e-4}))
        for eng in self.eng:
            run_engine(eng, self.store, t, training=training, config=CFG)
        for inc in self.store.incidents(system=S, entity=E):
            ex = inc.explanation or {}
            if ex.get("ts") == t:
                self.history.append((idx, t, inc.id, copy.deepcopy(ex), inc.narrative))


@pytest.fixture(scope="module")
def run() -> Run:
    return Run()


def _latest(run: Run):
    assert run.history, "B29 wrote no explanation"
    return run.history[-1]


def _with_cusum(run: Run):
    """The latest explanation; its counterfactual concerns the opening
    decision, which the CUSUM accumulator raised."""
    ex = _latest(run)
    assert "cusum" in ex[3]["counterfactual"]["factual"]["acc"], ex[3]["counterfactual"]
    return ex


# ================================================================== spec test
def test_an_incident_opens_and_is_explained(run):
    incs = run.store.incidents(system=S, entity=E)
    assert len(incs) == 1
    inc = incs[0]
    assert run.t_perturb <= inc.opened <= run.t_perturb + 2 * DT
    assert inc.explanation and inc.narrative
    # the other entities stay quiet
    for e in ENTS[1:]:
        assert run.store.incidents(system=S, entity=e) == []


def test_top3_attributions_include_the_perturbed_features(run):
    for _i, _t, _iid, ex, _n in run.history:
        top3 = {a["feature"] for a in ex["attributions"][:3]}
        assert PERTURBED <= top3, [(a["feature"], a["share"]) for a in ex["attributions"][:5]]
        a0 = ex["attributions"][0]
        assert 0.0 < a0["share"] <= 1.0 and a0["range"] and a0["observed_text"]
        assert sum(a["share"] for a in ex["attributions"] if a["share"]) <= 1.0 + 1e-9


def test_new_tokens_contain_the_sni_at_class_tier(run):
    _i, _t, _iid, ex, _n = _latest(run)
    toks = {t["token"]: t for t in ex["new_tokens"]}
    hit = [t for k, t in toks.items() if SNI in k]
    assert hit and hit[0]["tier"] == "class" and hit[0]["dim"] == "sni"
    assert hit[0]["first_seen"] == run.t_perturb + 2 * DT


def test_counterfactual_is_a_valid_subset_recomputed_through_the_cusum(run):
    _i, t, _iid, ex, _n = _with_cusum(run)
    cf = ex["counterfactual"]
    assert ex["counterfactual_valid"] is True, cf
    assert set(ex["counterfactual_set"]) and set(ex["counterfactual_set"]) <= PERTURBED, cf
    scope = ex["counterfactual_scope"]
    # stateful detectors are in scope: the CUSUM bank and MCUSUM were replayed
    assert {"cusum", "mcusum"} <= set(scope["replayed"])
    assert {"marg_int", "t2"} <= set(scope["recomputed"])
    assert scope["ticks"]["cusum"] >= 2
    # factually the CUSUM alarmed and the decision triggers; neutralised, the
    # replayed CUSUM stays below h and the fused decision is no incident
    assert cf["factual"]["trigger"] and "accumulator" in cf["factual"]["paths"]
    assert cf["factual"]["cusum_max"] >= 1.0
    assert cf["neutralised"]["cusum_max"] < 1.0
    assert not cf["neutralised"]["trigger"] and cf["neutralised"]["paths"] == []
    assert cf["neutralised"]["acc"] == []
    # fidelity: the factual recompute reproduces what B24 / B25 / B14 stored
    fid = scope["fidelity"]
    assert fid["decision_match"] is True
    assert fid["p_log10_err"] <= 1e-6
    assert fid["cusum"]["max_err"] <= 1e-6 and fid["cusum"]["n_resync"] == 0


def test_narrative_contains_the_natural_unit_range(run):
    _i, _t, _iid, ex, narrative = _latest(run)
    rng = ex["natural_range"]
    assert rng and rng["usual"]
    for text in (ex["narrative_en"], ex["narrative_zh"], narrative):
        assert rng["usual"] in text
    assert re.search(r"MB/15 min|KB/15 min", ex["narrative_en"])
    assert has_natural_range({"explanation": ex, "narrative": narrative})
    assert len(ex["bullets_en"]) == 3 and len(ex["bullets_zh"]) == 3


def test_explanations_are_written_only_on_open_and_escalation(run):
    inc = run.store.incidents(system=S, entity=E)[0]
    ticks = [t for _i, t, *_ in run.history]
    assert ticks[0] == inc.opened                  # explained on the opening tick
    assert len(ticks) < PERTURB_TICKS              # not every tick of the episode
    assert set(inc.explanation["axes"]) == set(inc.axes)
    # every later explanation still reports the counterfactual of the opening decision
    assert {ex["counterfactual"]["decision_ts"] for *_x, ex, _n in run.history} == {inc.opened}


def test_perf_per_incident(run):
    for *_x, ex, _n in run.history:
        assert ex["counterfactual_scope"]["ms"] < 1000.0


# ============================================================ replay helpers
def test_replay_generic_and_minimal_set():
    step = lambda s, x: (s + x, s + x)                      # noqa: E731
    res = RP.replay(step, 0.0, [(3.0, 1.0), (1.0, 2.0), (2.0, 5.0)], threshold=7.0)
    assert res.ts == [1.0, 2.0, 3.0] and res.scores == [2.0, 7.0, 8.0]
    assert res.alarmed_at == 2.0 and res.final_state == 8.0
    res2 = RP.replay(step, 0.0, [(1.0, 2.0), (2.0, 5.0)],
                     perturb=lambda t, x: 0.0 if t == 2.0 else x)
    assert res2.scores == [2.0, 2.0]
    # minimal_set: forward selection then backward elimination
    need = {"b", "d"}
    sub, n = RP.minimal_set(["a", "b", "c", "d"], lambda c: need <= set(c))
    assert sub == ["b", "d"] and n >= 4
    assert RP.minimal_set(["a"], lambda c: False)[0] == []
    # counterfactual_features on a scalar CUSUM
    inputs = [(float(i), np.array([3.0 if i >= 2 else 0.0, 0.5, 0.0])) for i in range(6)]

    def cstep(s, x):
        s2 = max(0.0, s + float(np.sum(x)) - 1.0)
        return s2, s2

    def neutral(x, sub):
        y = np.array(x, copy=True)
        y[list(sub)] = 0.0
        return y

    assert RP.counterfactual_features(cstep, 0.0, inputs, [1, 0, 2], 4.0, neutral) == [0]


def test_m_cp_replay_rows_reproduce_the_bank(run):
    """replay_rows + replay_step reproduce B14's recorded states bit for bit
    (the inputs come from the state ring itself, after B14's support mask)."""
    st = run.store
    params = m_cp.replay_params(st, S, E, dt=DT)
    ts0, s0 = m_cp.replay_state(st, S, E, run.t_perturb - 4 * DT)
    rows = m_cp.replay_rows(st, S, E, ts0, run.t_end, dt=DT)
    marked, info = m_cp.mark_resets(params, s0, rows)
    assert info["n_resync"] == 0 and info["max_err"] == 0.0 and info["n_reset"] == 0
    step = m_cp.replay_step(params)
    res = RP.replay(step, m_cp.replay_state0(s0), marked)
    assert np.array_equal(res.final_state["S"].astype(np.float32),
                          rows[-1][1]["S"].astype(np.float32))
    # the MCUSUM follows its recorded whitened inputs (B06 refits changed the
    # whitener inside this window; the input is recovered per tick)
    for (_ts, inp), mc in zip(marked, _mc_path(step, s0, marked), strict=True):
        assert np.allclose(mc, inp["mc"], rtol=1e-5, atol=1e-6)
    assert res.max_score >= 1.0
    k = [IDX["bytes_up"], IDX["updown_log"]]
    cf = RP.replay(step, m_cp.replay_state0(s0), marked,
                   perturb=lambda t, x: dict(x, x=m_cp.neutralize_key(x["x"], k)))
    assert cf.max_score < 1.0


def _mc_path(step, s0, rows):
    st = m_cp.replay_state0(s0)
    out = []
    for _ts, inp in rows:
        st, _ = step(st, inp)
        out.append(st["mc"].copy())
    return out


def test_m_cp_mc_input_inverts_the_crosier_step():
    rng = np.random.default_rng(3)
    S = rng.standard_normal(12)
    for scale in (0.01, 1.0, 5.0):
        w = scale * rng.standard_normal(12)
        S2, _ = m_cp.seq.mcusum_step(S, w, m_cp.MC_K)
        w2 = m_cp.mc_input(S, S2)
        S3, _ = m_cp.seq.mcusum_step(S, w2, m_cp.MC_K)
        assert np.allclose(S3, S2, atol=1e-12)
        if np.any(S2 != 0):
            assert np.allclose(w2, w, atol=1e-9)


def test_m_cp_mark_resets_recovers_restarts_and_latch_resets():
    """Recorded rows produced with a chart restart and a bank latch reset
    are replayed exactly once mark_resets has found them (no resync)."""
    rng = np.random.default_rng(5)
    params = {"h": m_cp.bank_h(DT), "h_mc": m_cp.mcusum_h(DT), "W": np.eye(12),
              "Sigma": np.eye(12)}
    step = m_cp.replay_step(params)
    st = {"S": np.zeros(48), "mc": np.zeros(12), "prev": np.full(12, np.nan),
          "phi": np.full(12, 0.2), "mc_ratio": 0.0}
    s0 = {k: np.array(v, copy=True) if isinstance(v, np.ndarray) else v for k, v in st.items()}
    rows = []
    for i in range(12):
        x = rng.standard_normal(12) + (1.5 if i >= 3 else 0.0)
        inp = {"x": x, "phi": np.full(12, 0.2), "adjacent": True,
               "mode": ("restart",) if i == 6 else ("zero_S",) if i == 9 else ()}
        st, _ = step(st, inp)
        rows.append((T0 + i * DT, {"x": x, "phi": inp["phi"], "adjacent": True,
                                   "S": st["S"].copy(), "mc": st["mc"].copy()}))
    marked, info = m_cp.mark_resets(params, s0, rows)
    assert info["n_resync"] == 0 and info["n_reset"] == 2
    assert "restart" in marked[6][1]["mode"] and "zero_S" in marked[9][1]["mode"]
    res = RP.replay(step, s0, marked)
    assert np.allclose(res.final_state["S"], rows[-1][1]["S"])
    assert np.allclose(res.final_state["mc"], rows[-1][1]["mc"], atol=1e-6)


def test_m_cp_replay_rows_judge_adjacency_at_each_ticks_own_cadence(run):
    st = run.store
    t0 = run.t_live - 3 * DT
    a = m_cp.replay_rows(st, S, E, t0, run.t_live + DT, dt=DT)
    assert all(r["adjacent"] for _t, r in a)
    b = m_cp.replay_rows(st, S, E, t0, run.t_live + DT, dt=DT / 4.0)
    assert not any(r["adjacent"] for _t, r in b)          # 900-s gaps at a 225-s cadence
    c = m_cp.replay_rows(st, S, E, t0, run.t_live + DT, dt=DT / 4.0, dt_of=lambda ts: DT)
    assert all(r["adjacent"] for _t, r in c)


def test_m_calib_p_replay_matches_issued_p(run):
    """B24's p recomputed from the snapshot equals the stored p (the prior
    order of the small-sample blend included)."""
    st = run.store
    model = st.get_model(S, E, m_calib.MODEL)
    t = run.t_end
    dp, terc, dt = m_calib.issued_stratum(model, t)
    score = st.vec_at(S, E, "behavior.score", t)
    pm = st.vec_at(S, E, "behavior.pm", t)
    p = st.vec_at(S, E, "behavior.p", t)
    from app.engines.behavior.lib.detectors import DETECTORS
    n = 0
    for i, d in enumerate(DETECTORS):
        if not math.isfinite(float(score[i])):
            continue
        pr = m_calib.p_replay(model, d, dp, TB.cadence_class(dt), float(score[i]),
                              m_calib.uniform(S, E, d, t), tercile=terc, pm=float(pm[i]))
        assert pr == pytest.approx(float(p[i]), rel=1e-6), d
        n += 1
    assert n >= 5


def test_gk_shares_bh_and_ranges():
    # Sigma = I: shares are z^2 / sum z^2 on the modelled dims
    assert gk_shares(None, np.zeros(NF))[0] is None
    p = np.full(NF, np.nan)
    p[:4] = [1e-6, 0.01, 0.2, 0.9]
    assert bh_flags(p, 0.05)[:4].tolist() == [True, True, False, False]
    r = fmt_range("bytes_up", 38.2e6, 0.4e6, 0.7e6, 1.1e6)
    assert r["observed_text"] == "38.2 MB/15 min" and r["range"] == "0.4–1.1 MB"
    assert r["ratio"] == pytest.approx(38.2 / 0.7)
    r = fmt_range("http_4xx_rate", 0.25, 0.01, 0.02, 0.05)
    assert r["observed_text"] == "25%" and r["range"] == "1–5%"
