"""pipeline.orchestrator.configure_gc: the pipeline raises the cyclic-GC
thresholds only when the process kept the interpreter defaults."""
import gc

from app.pipeline import orchestrator as O


def test_configure_gc_sets_only_over_the_defaults():
    saved = gc.get_threshold()
    try:
        gc.set_threshold(*O._GC_DEFAULTS)
        assert O.configure_gc() is True
        assert gc.get_threshold() == O.GC_THRESHOLDS
        gc.set_threshold(1234, 5, 6)                     # the process chose its own
        assert O.configure_gc() is False
        assert gc.get_threshold() == (1234, 5, 6)
    finally:
        gc.set_threshold(*saved)
