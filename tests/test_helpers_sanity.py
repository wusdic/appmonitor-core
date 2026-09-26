"""Sanity check that the helpers drive existing lib-3 engines correctly."""
import numpy as np

from helpers import T0, DT, add_feature_rows, ctx, make_store

from app.engines.behavior.baseline import BaselineEngine
from app.engines.behavior.features import FEATURE_DIM


def test_baseline_from_feature_rows():
    store = make_store()
    rng = np.random.default_rng(0)
    rows = rng.normal(5.0, 0.5, size=(40, FEATURE_DIM))
    ts = add_feature_rows(store, "sys", "10.0.0.1", rows)
    BaselineEngine().run(ctx(store, ts[-1]))
    prof = store.profile("sys", "10.0.0.1")
    assert prof is not None and prof.stable
    assert len(prof.baseline_median) == FEATURE_DIM
    assert abs(np.mean(prof.baseline_median) - 5.0) < 0.3
