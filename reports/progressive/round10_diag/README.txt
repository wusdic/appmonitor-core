Round 10 diagnostics (docs/lib3/progressive.md §16.18.11). All runs on code10 (= 4065855 + EV10-1).

hold_O2.json    pack O seed 2, days 5-9 (23:45): portal ptree nodes whose P04 p_hold or
                hold-without-when is < 0.3, with HoldRecord T (passes, tests), F (key -> failed
                tests), facet_tests, c (key -> [n, hits, nominal]) and node meta. NOTE: meta
                'hold_fp' is shared by reference in the hook, so every day shows the day-9 fingerprint.
                Script: diagHold.py <codedir> O 2 <out> portal 5 9
low_O2.json     pack O seed 2, days 7-9: portal statements with confidence < 0.3 (full evidence).
                Script: diagLow.py
oreal0_r10.json O-real seed 0, days 8-35 (23:45): finance / oa (fam:1) statements (pattern_id,
                route, state, confidence, who, text_zh), P11 groups and 'readdressed', and the
                number of ptree nodes per system. Script: diagFin10.py <codedir> O-real 0 <out> finance,oa 8 35
calbins.py      per-bin breakdown of PG2's pooled debiased calibration error from a run JSON.
cmp10.py, tables10.py, gates.py, pg4tab10.py, cost10.py
                comparison tables round 10 / 9 / 8 (paths point at the evaluator's scratch copy).
