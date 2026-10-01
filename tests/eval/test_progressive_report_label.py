"""scripts/progressive_report.py: the top-level registry label is the
registry the runs were made with (a report assembled from
progressive_decision runs said 'pack default (full+progressive)')."""
import importlib.util
import os

_P = os.path.join(os.path.dirname(__file__), "..", "..", "scripts", "progressive_report.py")


def _mod():
    spec = importlib.util.spec_from_file_location("progressive_report", _P)
    m = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(m)
    return m


def test_label_is_the_registry_the_runs_used_not_the_command_line():
    m = _mod()
    runs = [{"seed": s, "registry": "progressive_decision"} for s in range(3)]
    assert m.registry_label(runs, None, False) == "progressive_decision"
    lab = m.registry_label(runs + [{"seed": 3, "registry": "full+progressive"}], None, False)
    assert lab.startswith("MIXED") and "progressive_decision" in lab and "full+progressive" in lab


def test_label_carries_the_resource_mode_runs_recorded():
    m = _mod()
    runs = [{"seed": 0, "registry": "progressive_decision", "resource_mode": "bounded"}]
    assert m.registry_label(runs) == "progressive_decision, lib3.resource_mode = bounded"
    assert m.registry_label([], "progressive_only", True) == "progressive_only, lib3.resource_mode = bounded"
