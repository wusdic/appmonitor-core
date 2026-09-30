"""The temporal engines become more precise with observation time on the
generator's own example (pack O's OA system; requirement S3/S5/S10). See
tests/eval/temporal_convergence.py for the measures."""
from __future__ import annotations

import temporal_convergence as T


def test_windows_and_workflows_sharpen_with_observation_time():
    res = T.run(days=(2, 10), dt=3600.0, seed=0)
    w2, w10 = res[2]["windows"], res[10]["windows"]
    f2, f10 = res[2]["workflow"], res[10]["workflow"]
    # time windows: every OA pattern with a workday window is found by day 10, and the
    # learned windows move towards the truth (IoU) while staying inside it (precision)
    assert w10["found"] >= w2["found"] and w10["found"] >= w10["n"] - 1, (w2, w10)
    assert w10["iou"] >= w2["iou"] + 0.08, (w2["iou"], w10["iou"])
    assert w10["iou"] >= 0.85 and w10["precision"] >= 0.9, w10
    gl = w10["per"]["GA.oa.login#0"]                              # the requirement's example
    assert gl is not None and gl[0] >= 0.9, gl                    # 09:00-09:21 learned
    # workflows: more truth edges are recovered (dep >= 0.8, band overlapping the truth)
    # and what is kept is part of the persona program
    assert f10["recall"] >= f2["recall"] + 0.4 and f10["recall"] >= 0.8, (f2, f10)
    assert f10["correct"] > f2["correct"]
    assert f10["program_precision"] >= 0.8, f10
