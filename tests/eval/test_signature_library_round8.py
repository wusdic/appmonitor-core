"""(round 8, harness owner) The lib-4 signature library is never missing silently.

Round 7: build.DATA_DIR finds data/signatures relative to the code, and the
evaluator's code snapshots held only backend/ and scripts/, so the first pack A
runs ran every lib-4 engine on an empty library (12 / 7 / 1 incidents instead
of 18 / 7 / 2) and nothing said so.
"""
from __future__ import annotations

import os
import warnings

import pytest

from app.eval import runner as R
from app.pipeline import build


def _empty_dir(tmp_path):
    d = tmp_path / "nosigs"
    d.mkdir()
    return str(d)


def test_load_signatures_raises_in_strict_mode_when_the_library_is_missing(tmp_path):
    d = _empty_dir(tmp_path)
    with pytest.raises(build.SignatureLibraryMissing):
        build.load_signatures(strict=True, directory=d)


def test_load_signatures_warns_and_marks_the_store_otherwise(tmp_path, monkeypatch):
    monkeypatch.delenv("APPMON_REQUIRE_SIGNATURES", raising=False)
    d = _empty_dir(tmp_path)
    with pytest.warns(build.SignatureLibraryWarning):
        store, comp = build.load_signatures(directory=d)
    assert store.signatures == [] and comp == []
    assert sorted(store.library_missing) == ["composite.yaml", "primitives.yaml"]
    monkeypatch.setenv("APPMON_REQUIRE_SIGNATURES", "1")
    with pytest.raises(build.SignatureLibraryMissing):
        build.load_signatures(directory=d)


def test_the_repository_library_loads_without_warning():
    with warnings.catch_warnings():
        warnings.simplefilter("error", build.SignatureLibraryWarning)
        store, comp = build.load_signatures(strict=True)
    assert len(store.signatures) > 0 and store.library_missing == []
    assert build.signature_status()["ok"]


def test_strict_registry_build_refuses_a_missing_library(tmp_path, monkeypatch):
    """An eval run (ctx.config['strict']) on a snapshot without data/ fails at
    registry build instead of running lib-4 on nothing."""
    monkeypatch.setattr(build, "DATA_DIR", _empty_dir(tmp_path))
    with pytest.raises(build.SignatureLibraryMissing):
        build.build_registry(config={"strict": True})
    # the lib-4-free registries do not need it
    build.build_registry(config={"strict": True}, progressive="progressive_decision")


def test_runner_marks_the_run_and_rejects_an_empty_library_in_strict_mode(tmp_path):
    from app.engines.signature.store import SignatureStore
    reg = build.build_registry(SignatureStore(), [], config={"strict": False})
    with pytest.raises(build.SignatureLibraryMissing):
        R.check_signatures(reg, True)
    with pytest.warns(build.SignatureLibraryWarning):
        st = R.check_signatures(reg, False)
    assert st["engines"] and st["n_signatures"] == 0
    full = build.build_registry(config={"strict": True})
    st = R.check_signatures(full, True)
    assert st["engines"] and st["n_signatures"] > 0 and st["missing"] == []
    assert os.path.samefile(st["dir"], build.signature_status()["dir"])
    assert R.check_signatures(build.build_registry(progressive="progressive_only"), True) == {"engines": False}
