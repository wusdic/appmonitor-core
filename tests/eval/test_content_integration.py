"""The content fitters P06-P08 on the real progressive lattice (P00-P05, pack O's
OA system; docs/lib3/progressive.md §6.10-§6.12). See
tests/eval/content_integration.py for the measure.

Measured (seeds 0 / 1, days 3, 7, 11, 14, 18): the username grammar is on the
login path from day 7 ([a-z]{3,8}, every truth username accepted; unseen-shape
mass 0.041 -> 0.017), 财务部's three bindings are published 3/3 with LB >= 0.97
by day 14 (seed 0) / day 18 (seed 1) and none before day 11, the login key set
equals the truth from day 7. The login node still mixes departments at day 18,
so its size band is the mixture's, not 综合部's (lattice specialisation, W-P3)."""
from __future__ import annotations

import content_integration as CI


def test_content_constraints_appear_and_tighten_on_the_real_lattice():
    res = CI.run(days=(7, 14), dt=3600.0, seed=0)
    d7, d14 = res[7], res[14]
    fin7, fin14 = d7["FIN.oa.login#0"], d14["FIN.oa.login#0"]
    # the grammar exists early and accepts every truth username ...
    assert fin7.get("grammar_ok") and fin14.get("grammar_ok")
    # ... and its unseen mass falls with observation time
    assert fin14["U_s"] < fin7["U_s"]
    # bindings are earned, not assumed: none at day 7, all three by day 14
    assert fin7["bindings"] == 0
    assert fin14["bindings"] == 3 and fin14["lb"] >= 0.9
    # the login form's required keys equal the truth's
    assert fin14.get("keys_ok")
