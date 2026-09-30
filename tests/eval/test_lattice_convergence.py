"""The progressive lattice becomes more precise with observation time on the
generator's own example (pack O's OA system, docs/lib3/progressive.md §6.22,
requirement S2/S3 "用的时间越长越精准"). See tests/eval/lattice_convergence.py
for the measure: every truth pattern with an IP-level who-set is scored against
its covering confident pattern (the node P03 would score it against).

Measured (seeds 0 / 1, 3600-s ticks, P02 + P05 + P04 only, no P11 groups):
day 2 -> 21 who precision 0.175 -> 0.24 / 0.14 -> 0.31, route specific 0 ->
0.5 / 0.36, covering depth 0 -> 3.1 / 4.1, nodes 1 -> 23 / 37."""
from __future__ import annotations

import lattice_convergence as L


def test_precision_rises_with_observation_time():
    res = L.run(days=(2, 21), dt=3600.0, seed=0)
    early, late = res[2], res[21]
    assert late["patterns"] >= 10
    # the tree grows from the root into specific, confirmed patterns (the route
    # partition gives every recurring action a node from day 2, but none is a
    # confirmed pattern before its third date) ...
    assert early["depth"] == 0.0 and late["nodes"] >= 10, res
    assert late["depth"] >= 1.5, res
    # ... whose contexts isolate the example's actions (routes) ...
    assert early["route_specific"] == 0.0 and late["route_specific"] >= 0.3, res
    # ... and whose who-sets move toward the truth IP lists
    assert late["who_precision"] >= early["who_precision"] + 0.05, res
