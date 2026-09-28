"""Engine / lib unit tests run in v2 tick grain mode by default (spec v2.1,
docs/lib3/cadence.md §12 M8): their maths is cadence-agnostic, and a test
that exercises the canonical grain mode passes config={'grain_mode':
'canonical'} explicitly. Pipeline, runtime, eval and API tests run the
canonical default."""
import pytest

from helpers import tick_mode_default


@pytest.fixture(autouse=True)
def _tick_grain_mode(monkeypatch):
    tick_mode_default(monkeypatch)
