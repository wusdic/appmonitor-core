"""Sanity check that the shared test helpers drive the v2 lib-3 engines
(ported from the v1 feature.<name> version at integration)."""
import numpy as np

from helpers import DT, add_feature_rows, ctx, make_store

from app.engines.behavior.baseline import BaselineEngine
from app.engines.behavior.lib import features as F
from app.engines.behavior.lib import m_baseline as MB

REQ = F.FEATURE_INDEX["http_requests"]
COUNTS = [i for i, n in enumerate(F.FEATURE_NAMES_V2) if F.FEATURE_KIND[n] == "count"]


def _rows(rng, n):
    out = np.full((n, F.FEATURE_DIM), np.nan)
    out[:, COUNTS] = 0.0
    out[:, REQ] = rng.poisson(10.0 * DT / 60.0, size=n)          # 10 requests / min
    return out


def test_baseline_from_feature_rows():
    """feature.nat + feature.active rings (what B01 writes) are enough for B03
    to learn (training: trust = 1) and publish the v2 profile fields."""
    store = make_store()
    rng = np.random.default_rng(0)
    ts = add_feature_rows(store, "sys", "10.0.0.1", _rows(rng, 40))
    eng = BaselineEngine()
    for t in ts:
        eng.run(ctx(store, t, training=True, window_s=DT))
    m = store.get_model("sys", "10.0.0.1", MB.MODEL)
    assert m is not None and MB.n_eff(m) > 20
    prof = store.profile("sys", "10.0.0.1")
    assert prof is not None
    assert len(prof.baseline_median) == F.FEATURE_DIM
    # baseline_median is in feature.vec space: log1p(rate per minute) for counts
    assert abs(prof.baseline_median[REQ] - np.log1p(10.0)) < 0.3
    assert not prof.stable                  # n_eff < 96: B01 owns 'stable', and it is not yet
