"""The progressive lattice becomes more precise with observation time on the
generator's own example (pack O's OA system, docs/lib3/progressive.md §6.22,
requirement S2/S3). See tests/eval/lattice_convergence.py for the measure."""
from __future__ import annotations

import lattice_convergence as L


def test_precision_rises_with_observation_time():
    res = L.run(days=(2, 7, 14), dt=3600.0, seed=0)
    early, mid, late = res[2], res[7], res[14]
    assert late["patterns"] >= 8
    # the tree grows from the root into specific patterns ...
    assert early["nodes"] < late["nodes"] and early["depth"] < late["depth"]
    # ... and the patterns of the example's IP lists become isolated to their IPs
    assert late["who_precision"] >= early["who_precision"] + 0.15, res
    assert late["who_precision"] >= mid["who_precision"] - 0.05, res
    assert late["route_specific"] > early["route_specific"], res
