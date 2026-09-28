import os, sys

# Single-threaded BLAS / OpenMP (integration notes R15.1 / R19.3): the lib-3
# fits are many tiny eigendecompositions and C-steps, which multithreaded
# OpenBLAS makes 3-10x slower under load. Set before numpy is imported.
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")

sys.path.insert(0, os.path.dirname(__file__))
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))


# spec v2.1 (docs/lib3/cadence.md §12 M8): the default grain mode is
# 'canonical' for pipeline / runtime / eval / API tests; engine, lib and core
# unit tests keep v2 'tick' (their maths is cadence-agnostic). Module-level
# default_config() calls of those unit-test modules run at import (collection)
# time, so the tick default is also applied while they are imported; at run
# time tests/{engines,lib,core}/conftest.py applies it per test.
import pytest  # noqa: E402

_TICK_DIRS = ("engines", "lib", "core")


def _unit_module(path) -> bool:
    parts = str(path).replace("\\", "/").split("/")
    return "tests" in parts and any(d in parts[parts.index("tests") + 1:-1]
                                    for d in _TICK_DIRS)


@pytest.hookimpl(hookwrapper=True)
def pytest_make_collect_report(collector):
    tick = isinstance(collector, pytest.Module) and _unit_module(collector.path)
    prev = None
    if tick:
        from app.core import engine as _eng
        prev = _eng.DEFAULT_CONFIG.get("grain_mode")
        _eng.DEFAULT_CONFIG["grain_mode"] = "tick"
    try:
        yield
    finally:
        if tick:
            _eng.DEFAULT_CONFIG["grain_mode"] = prev
