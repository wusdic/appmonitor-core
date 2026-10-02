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


def test_checklist_reads_the_most_specific_statement_of_the_department():
    """Evaluator round 3: when two statements name exactly 综合部's three
    addresses, the example checklist reads the address-level node's own
    statement, not the department's part of a mixed node (seed 3 read the
    part, whose window spanned the D1 change: IoU 0.34)."""
    m = _mod()
    ga = ["192.168.1.21", "192.168.1.23", "10.168.7.121"]
    part = {"route": "POST oa /login", "who": {"level": "grp", "members": ga, "items": ["grp:G10"]},
            "when": {"workday": [[514, 560]]}, "context": [["http.route", 0, ["x"], False],
                                                          ["net.src", 3, ["grp:G10"], False]],
            "content": {}, "bindings": {}, "text_zh": "part"}
    node = {"route": "POST oa /login", "who": {"level": "ip", "members": ga, "items": ga},
            "when": {"workday": [[513, 531]]}, "context": [["http.route", 0, ["x"], False]],
            "content": {}, "bindings": {}, "text_zh": "node"}
    truth = [{"activity": "GA.oa.login", "step": 0, "valid_from_day": 12, "valid_to_day": 99,
              "who": {"level": "ip", "value": ga}, "windows": {"workday": [[510, 531]]},
              "content": {}, "bindings": {}}]
    rows = m.checklist({"oa_statements": [part, node], "finance_statements": []}, truth, 21)
    assert rows[0]["statement"] == "node"
    assert rows[1]["pass"]                                       # the node's window, IoU >= 0.7
