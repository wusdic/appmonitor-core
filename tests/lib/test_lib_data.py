"""Contract tests for the data / simple helper modules of engines/behavior/lib
(features, detectors, stages, classkeys, names, priors) plus an import smoke
test of every stub module, so the frozen contract cannot drift silently."""
from __future__ import annotations

import importlib
import math
import os
import sys

import numpy as np
import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from app.engines.behavior.lib import classkeys as ck  # noqa: E402
from app.engines.behavior.lib import detectors as det  # noqa: E402
from app.engines.behavior.lib import features as F  # noqa: E402
from app.engines.behavior.lib import names  # noqa: E402
from app.engines.behavior.lib import priors  # noqa: E402
from app.engines.behavior.lib import stages  # noqa: E402

LIB_MODULES = ["features", "detectors", "stages", "classkeys", "names", "priors", "bayes",
               "combine", "calib", "evt", "seq", "gating", "timebins", "sketch", "ppm",
               "robustcov", "template", "stack", "replay", "featcache"]


# ------------------------------------------------------------------ imports
@pytest.mark.parametrize("mod", LIB_MODULES)
def test_every_lib_module_imports(mod):
    m = importlib.import_module(f"app.engines.behavior.lib.{mod}")
    assert m.__doc__ and len(m.__doc__) > 40          # WHY docstring present


def test_lib_does_not_import_engines():
    lib_dir = os.path.dirname(F.__file__)
    for fn in os.listdir(lib_dir):
        if fn.endswith(".py"):
            with open(os.path.join(lib_dir, fn), encoding="utf-8") as fh:
                src = fh.read()
            for eng in ("feature_vector", "baseline", "anomaly", "clustering", "drift",
                        "sequence", "fingerprint"):
                assert f"import {eng}" not in src and f".{eng} import" not in src, (fn, eng)


# ----------------------------------------------------------------- features
def test_feature_spec_shape_and_exports():
    assert F.FEATURE_DIM == 52 == len(F.FEATURE_SPEC_V2) == len(F.FEATURE_NAMES_V2)
    assert len(set(F.FEATURE_NAMES_V2)) == 52
    for name, src, kind, nsrc, group in F.FEATURE_SPEC_V2:
        assert kind in F.KINDS
        assert group in F.GROUP_ORDER
        F.source_kind(src)                                   # well-formed source
        assert F.FEATURE_INDEX[name] == F.FEATURE_NAMES_V2.index(name)
        assert F.FEATURE_KIND[name] == kind and F.FEATURE_GROUP[name] == group
        assert F.FEATURE_NSRC[name] == nsrc
        if kind in ("ratio", "bounded"):
            assert nsrc is not None, name                    # exposure-gated kinds
    assert "fanout" not in F.FEATURE_INDEX and "peer_novelty" not in F.FEATURE_INDEX


def test_group_partition_covers_all_indices():
    idx = sorted(i for g in F.GROUPS.values() for i in g)
    assert idx == list(range(52))
    sizes = {g: len(v) for g, v in F.GROUPS.items()}
    assert sizes == {"volume": 9, "breadth": 5, "app": 10, "dns": 5, "tls": 4,
                     "timing": 5, "transport": 4, "probe": 2, "comp": 8}
    assert F.features_in(["probe"]) == F.GROUPS["probe"]


def test_key_features_and_exposure():
    assert len(F.KEY_FEATURES) == 12 and set(F.KEY_FEATURES) <= set(F.FEATURE_NAMES_V2)
    assert F.KEY_FEATURE_IDX == [F.FEATURE_INDEX[n] for n in F.KEY_FEATURES]
    assert F.EXPOSURE_CHANNELS == ["http", "dns", "tls", "flows", "probe"]
    assert set(F.EXPOSURE_SOURCE) == set(F.EXPOSURE_CHANNELS)


def test_formula_sources_encoded_explicitly():
    assert F.FEATURE_SOURCE["http_write_ratio"] == ("http.write_count", "http.requests")
    assert F.FEATURE_SOURCE["bytes_per_flow"] == ("l3.bytes_total", "l4.flows")
    assert F.source_kind(F.FEATURE_SOURCE["updown_log"]) == "logratio"
    assert F.FEATURE_NSRC["dest_concentration"] == ("tls.handshakes", "dns.queries")
    assert F.CLR_FEATURES == ["comp_get", "comp_write", "comp_4xx", "comp_5xx",
                              "comp_dns", "comp_tls", "comp_flows", "comp_syn"]


def test_transform_count_bytes():
    vec, nat = F.transform("count", 10, None, 60)
    assert vec == pytest.approx(math.log1p(10)) and nat == 10
    assert F.transform("count", None, None, 900) == (0.0, 0.0)       # stale -> true 0
    assert F.transform("bytes", float("nan"), None, 900) == (0.0, 0.0)
    assert F.transform("bytes", 1500, None, 900)[0] == pytest.approx(math.log1p(100))


def test_transform_ratio_nan_rules():
    vec, nat = F.transform("ratio", 3, 10, 60)
    assert vec == pytest.approx(math.log(3.5 / 11) - math.log(7.5 / 11))
    assert nat == pytest.approx(0.3)
    assert all(math.isnan(x) for x in F.transform("ratio", 3, 0, 60))     # n = 0
    assert all(math.isnan(x) for x in F.transform("ratio", None, 10, 60))  # stale
    assert all(math.isnan(x) for x in F.transform("ratio", 3, None, 60))
    assert F.transform("ratio", 12, 10, 60)[1] == 1.0                    # k clipped to n


def test_transform_avg_bounded_window_gauge():
    assert F.transform("avg", 100.0, 5, 60) == (pytest.approx(math.log(100.0)), 100.0)
    assert math.isnan(F.transform("avg", 100.0, 0, 60)[0])               # n = 0
    assert math.isnan(F.transform("avg", None, 5, 60)[0])                # stale
    assert math.isfinite(F.transform("avg", 0.0, 5, 60)[0])              # floored, no -inf
    assert F.transform("avg", 8.0, None, 60)[0] == pytest.approx(math.log(8.0))  # fresh-only
    assert F.transform("avg", -0.7, 5, 60, tx="identity") == (-0.7, -0.7)
    assert math.isnan(F.transform("bounded", 0.5, 4, 60)[0])             # n < 5
    assert F.transform("bounded", 0.5, 5, 60)[0] == pytest.approx(0.0)
    hi = F.transform("bounded", 1.0, 50, 60)[0]
    assert math.isfinite(hi) and hi == pytest.approx(math.log(0.999 / 0.001))
    assert F.transform("window", 0.25, None, 60) == (0.25, 0.25)
    assert math.isnan(F.transform("window", None, None, 60)[0])
    assert math.isnan(F.transform("gauge", None, None, 60)[0])
    with pytest.raises(ValueError):
        F.transform("clr", 1.0, None, 60)


def test_clr_rates_and_nan():
    v, n = F.clr([10, 0, 5, 0, 20, 3, 7, 1], 60)
    assert v.sum() == pytest.approx(0.0, abs=1e-12)
    assert n.sum() == pytest.approx(1.0)
    v0, n0 = F.clr([0] * 8, 60)
    assert np.isnan(v0).all() and np.isnan(n0).all()
    v15, _ = F.clr([150, 0, 75, 0, 300, 45, 105, 15], 900)
    np.testing.assert_allclose(v, v15, atol=1e-12)


def _getter(d):
    return lambda name: d.get(name)


def _raw(scale):
    base = {
        "l4.bytes_up": 2000, "l4.bytes_down": 40000, "l4.flows": 12, "http.requests": 30,
        "dns.queries": 8, "tls.handshakes": 6, "act.events": 44, "l3.bytes_total": 42000,
        "l4.distinct_peers": 4, "l4.distinct_dports": 2, "act.distinct_templates": 9,
        "derived.new_peer_count": 1, "http.get_count": 25, "http.write_count": 5,
        "http.status_4xx": 2, "http.status_5xx": 0, "l4.syn_count": 12,
        "derived.ja3_diversity": 2,
    }
    out = {k: v * scale for k, v in base.items()}
    return out


def test_cadence_invariance_counts_bytes_clr():
    """15x the counts at dt = 900 equal the counts at dt = 60 (B01 test d) for
    every count / bytes / clr column."""
    v60, _ = F.compute_features(_getter(_raw(1)), 60.0)
    v900, _ = F.compute_features(_getter(_raw(15)), 900.0)
    cols = [i for i, n in enumerate(F.FEATURE_NAMES_V2)
            if F.FEATURE_KIND[n] in ("count", "bytes", "clr")]
    assert len(cols) == 20   # volume 7, breadth 4, tls 1, clr 8
    np.testing.assert_allclose(v60[cols], v900[cols], atol=1e-12)
    # bytes_per_flow is a mean, so also invariant
    i = F.FEATURE_INDEX["bytes_per_flow"]
    assert v60[i] == pytest.approx(v900[i])


def test_compute_features_staleness():
    """Only http.requests fresh -> counts 0 elsewhere, ratios NaN (B01 test a)."""
    vec, nat = F.compute_features(_getter({"http.requests": 10}), 900.0)
    assert vec.shape == (52,) and nat.shape == (52,)
    assert vec[F.FEATURE_INDEX["bytes_up"]] == 0.0
    assert vec[F.FEATURE_INDEX["http_requests"]] == pytest.approx(math.log1p(10 * 60 / 900))
    assert np.isnan(vec[F.FEATURE_INDEX["http_write_ratio"]])     # write_count stale
    assert np.isnan(vec[F.FEATURE_INDEX["dest_concentration"]])
    empty, _ = F.compute_features(_getter({}), 900.0)
    for n in F.FEATURE_NAMES_V2:
        k = F.FEATURE_KIND[n]
        x = empty[F.FEATURE_INDEX[n]]
        if k in ("count", "bytes"):
            assert x == 0.0, n
        else:
            assert np.isnan(x), n


def test_feature_inputs_formulas():
    d = {"http.write_count": 5, "http.requests": 20, "l4.bytes_up": 99, "l4.bytes_down": 9,
         "l4.flows": 3, "tls.weak_version_ratio": 0.25, "tls.handshakes": 8,
         "derived.dest_concentration": 0.7, "dns.queries": 2}
    g = _getter(d)
    assert F.feature_inputs(F.FEATURE_INDEX["http_write_ratio"], g) == (5.0, 20.0)
    v, n = F.feature_inputs(F.FEATURE_INDEX["updown_log"], g)
    assert v == pytest.approx(math.log(100 / 10)) and n == 3
    assert F.feature_inputs(F.FEATURE_INDEX["tls_weak_ratio"], g) == (2.0, 8.0)
    assert F.feature_inputs(F.FEATURE_INDEX["dest_concentration"], g) == (0.7, 10.0)
    vec, _ = F.compute_features(g, 60.0)
    assert vec[F.FEATURE_INDEX["updown_log"]] == pytest.approx(math.log(10.0))
    # declared exposure stale -> avg gated to NaN even if the value is fresh
    vec2, _ = F.compute_features(_getter({"http.latency_ms_avg": 40.0}), 60.0)
    assert np.isnan(vec2[F.FEATURE_INDEX["http_latency"]])
    vec3, _ = F.compute_features(_getter({"derived.think_time_s_avg": 8.0}), 60.0)
    assert vec3[F.FEATURE_INDEX["think_time"]] == pytest.approx(math.log(8.0))
    ex = F.exposure(g)
    assert ex == {"http": 20.0, "dns": 2.0, "tls": 8.0, "flows": 3.0, "probe": 0.0}


# ---------------------------------------------------------------- detectors
def test_detector_registry():
    # spec v2.1 (docs/lib3/cadence.md §7.3, deliberate): the four Q-grain
    # detectors are appended, so D = 35 and the v2 prefix is unchanged
    assert det.N_DETECTORS == 35 == len(det.DETECTORS)
    assert det.DETECTORS[:3] == ["marg_int", "marg_shape", "peer"]
    assert det.DETECTORS[30] == "cross_system" and det.N_V2_DETECTORS == 31
    assert det.DETECTORS[-4:] == ["marg_int_q", "marg_shape_q", "t2_q", "spe_q"]
    assert det.DETECTORS[-1] == "spe_q"
    assert {d for d in det.DETECTORS if det.DETECTOR_INFO[d]["stream"] == "q"} == set(det.Q_DETECTORS)
    assert det.DETECTOR_INFO["identity"]["overlap"] is True
    assert det.DETECTOR_INFO["marg_int"]["stream"] == "h" and det.DETECTOR_INFO["novelty"]["stream"] == "t"
    assert all(det.DETECTOR_INDEX[d] == i for i, d in enumerate(det.DETECTORS))
    for d in det.DETECTORS:
        info = det.DETECTOR_INFO[d]
        assert info["family"] in det.FAMILIES
        assert info["kind"] in ("inst", "acc")
        assert info["owner"].startswith("B") and info["axes"]
        if info["kind"] == "acc":
            assert 0 < info["budget_per_day"] <= 0.03
            assert det.arl_days(d) == pytest.approx(1 / info["budget_per_day"])
        else:
            assert "budget_per_day" not in info
    assert set(det.P2_DETECTORS) == {"mixture", "session", "cross_system"}
    s = det.new_score_vector()
    assert s.shape == (35,) and np.isnan(s).all()


def test_detector_families_partition():
    members = [d for f in det.FAMILIES for d in det.family_members(f)]
    assert sorted(members) == sorted(det.DETECTORS)
    # spec v2.1 (deliberate): the appended Q detectors join intensity / shape
    assert det.family_members("intensity") == ["marg_int", "t2", "budget_vol", "class_int",
                                               "marg_int_q", "t2_q"]
    assert det.family_members("intensity", "inst") == ["marg_int", "t2", "class_int",
                                                       "marg_int_q", "t2_q"]
    assert det.family_members("shape")[-2:] == ["marg_shape_q", "spe_q"]
    assert det.family_members("change") == ["cusum", "mcusum", "bocpd", "creep"]
    assert det.family_members("temporal", "inst") == []
    assert set(det.INSTANT_FAMILIES) == {"intensity", "shape", "peer", "categorical",
                                         "sequence", "identity", "xsys"}
    assert det.DETECTOR_INFO["identity"]["strata"] == "daypart_regime"
    assert det.detectors_of("B14") == ["cusum", "mcusum", "bocpd", "creep"]
    # path budgets are split exactly
    for path, total in det.BUDGET_PATHS.items():
        got = sum(det.DETECTOR_INFO[d]["budget_per_day"] for d in det.ACC_DETECTORS
                  if det.DETECTOR_INFO[d]["budget_path"] == path)
        if got:
            assert got == pytest.approx(total)
    with pytest.raises(ValueError):
        det.arl_days("marg_int")


# ------------------------------------------------------------------- stages
def test_stage_map():
    sa = stages.stage_for_axis
    for ax in ("volume", "shape", "peer"):
        assert sa(ax) == "behavior"
    assert sa("temporal") == "off_hours"
    assert sa("categorical", sensitive=True) == "privilege"
    assert sa("categorical", admin=True) == "privilege"
    assert sa("categorical", external=True, upload_dominant=True) == "exfiltration"
    assert sa("categorical", new_external_domain=True) == "c2"
    assert sa("categorical", low_prevalence=True) == "c2"
    assert sa("exfil") == "exfiltration"
    assert sa("breadth", internal=True) == "discovery"
    assert sa("breadth", object_ids=True) == "collection"
    assert sa("sequence", auth=True) == "credential"
    assert sa("sequence", login_4xx=True) == "credential"
    assert sa("sequence") == "behavior"
    assert sa("identity") == "identity"
    assert sa("c2") == "c2"
    assert sa("xsys") == "lateral"
    assert sa("nonsense") is None
    assert stages.stage_for_event("client_impersonation") == "identity"
    assert stages.stage_for_event("possible_impersonation") == "identity"
    assert stages.stage_for_event("beacon") == "c2"
    cat = stages.stage_for_category
    assert cat("recon") == cat("scan") == "discovery"
    assert cat("auth") == cat("bruteforce") == "credential"
    assert cat("tunnel") == cat("beacon") == cat("c2") == "c2"
    assert cat("transfer") == cat("exfil") == "exfiltration"
    assert cat("admin") == "privilege"
    assert cat("browse") is None
    assert stages.stages_for(["volume", "temporal", "categorical"], {"sensitive": True}) == \
        {"behavior", "off_hours", "privilege"}
    for ax in [a for d in det.DETECTORS for a in det.DETECTOR_INFO[d]["axes"]]:
        assert sa(ax) in stages.STAGES, ax                  # every default axis maps
    for g in F.GROUP_ORDER:
        assert sa(g) in stages.STAGES, g                     # feature groups are axes too


# --------------------------------------------------------------- class keys
def test_classkeys():
    assert ck.role_key(7) == "class:7" and ck.role_key("r3") == "class:r3"
    assert ck.static_key("dmz") == "class:static:dmz"
    assert ck.pool_key("10.1.0.0/24") == "class:pool:10.1.0.0/24"
    for k in ("class:7", "class:static:dmz", "__system__", "__org__"):
        assert ck.is_pseudo(k)
    assert not ck.is_pseudo("10.0.0.1")
    assert ck.is_class("class:pool:10.1.0.0/24") and not ck.is_class("__system__")
    assert ck.class_kind("class:static:dmz") == "static"
    assert ck.class_kind("class:pool:x") == "pool" and ck.class_kind("class:9") == "role"
    assert ck.class_id("class:static:dmz") == "dmz" and ck.class_id("10.0.0.1") is None
    assert ck.SYSTEM_KEY == "__system__" and ck.ORG == ("__org__", "__org__")
    with pytest.raises(ValueError):
        ck.role_key("static:x")


# -------------------------------------------------------------------- names
@pytest.mark.parametrize("host,expected", [
    ("www.example.com", "example.com"),
    ("a.b.cdn.example.com.", "example.com"),
    ("mail.corp.example.com.cn", "example.com.cn"),
    ("x.y.gov.cn", "y.gov.cn"),
    ("pku.edu.cn", "pku.edu.cn"),
    ("www.news.org.cn", "news.org.cn"),
    ("a.b.net.cn", "b.net.cn"),
    ("www.bbc.co.uk", "bbc.co.uk"),
    ("shop.example.com.au", "example.com.au"),
    ("www.example.co.jp", "example.co.jp"),
    ("EXAMPLE.COM:443", "example.com"),
    ("svc.corp.local", "corp.local"),
    ("localhost", "localhost"),
    ("10.1.2.3", "10.1.2.3"),
    ("", ""),
])
def test_etld1(host, expected):
    assert names.etld1(host) == expected


def test_split_host_and_is_external():
    assert names.split_host("a1b2.x.evil.com") == (["a1b2", "x"], "evil.com")
    assert names.split_host("example.com") == ([], "example.com")
    org = ["corp.example.com", "example.cn"]
    assert not names.is_external("erp.corp.example.com", org)
    assert not names.is_external("www.example.cn", org)
    assert not names.is_external("svc.corp.local", org)
    assert not names.is_external("fileserver", org)
    assert not names.is_external("10.2.3.4", org)
    assert names.is_external("8.8.8.8", org)
    assert names.is_external("dropbox.com", org)
    assert names.is_external("upload.evil.com.cn", org)
    assert names.is_external("dropbox.com", None)
    assert not names.is_external("", org)


# ------------------------------------------------------------------- priors
def test_priors_per_kind():
    gp = priors.prior_for_kind("count")
    assert isinstance(gp, priors.GammaPoissonPrior)
    assert gp.mu0 == 1.0 and gp.a0 == 0.5 and gp.b0 == pytest.approx(0.5)
    bp = priors.prior_for_kind("ratio")
    assert (bp.a0, bp.b0) == (0.5, 0.5)
    nb = priors.prior_for_kind("bytes")
    assert nb.m0 == pytest.approx(math.log(1e4))
    assert (nb.kappa0, nb.alpha0, nb.beta0) == (0.01, 1.0, 4.0)
    assert nb.df == 2.0 and nb.predictive_scale() == pytest.approx(math.sqrt(4 * 1.01 / 0.01))
    for n in F.FEATURE_NAMES_V2:
        p = priors.prior_for_feature(n)
        assert p is not None
        assert priors.likelihood_family(F.FEATURE_KIND[n]) in ("nb", "bb", "t")
    with pytest.raises(ValueError):
        priors.prior_for_kind("nope")


def test_cold_start_prior_is_wide():
    """Under the hyperprior the first predictive is finite and wide: 10 req/min
    against mu0 = 1/min is unusual but nowhere near 'garbage' territory.

    NOTE: with the spec values (mu0 = 1/min, a0 = 0.5) the upper tail
    P(X >= 10 e) is ~1.7e-3 (e = 15 min), i.e. two-sided mid-p ~3.4e-3, so
    engines.md B03 test (c) ('p(10 req/min) > 0.05' on an empty store) cannot
    hold on the hyperprior alone; see priors.COLD_START_NOTE."""
    from scipy.special import betainc
    gp = priors.COUNT_PRIOR
    for e in (1.0, 15.0, 60.0):
        mean, r = gp.mu0 * e, gp.a0
        q = r / (r + mean)
        sf = 1.0 - betainc(r, 10 * e, q)      # P(X >= 10e) = 1 - P(X <= 10e - 1)
        assert 1e-3 < sf < 0.01
