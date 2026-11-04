Round 11 evaluator diagnostics (progressive.md §16.19). JSON / text only; no pickles or checkpoints.

Final code: code11 = code10 + the tree owner's pattern_tree.py / views.py (their snapshot t5)
+ the bounds owner's conformity.py, lib/pfd.py, lib/pwindows.py, time_window.py (their snapshot fin5)
+ EV11-1 (EV11-1_pfd_shared_terminal.patch on lib/pfd.py). EV11-2 reverted T11-5 / T11-6
(reverted_pattern_tree_t7.py.txt is the file with them; reverted_T11-5_T11-6_tests.py.txt their two tests).
code11_owned_files.md5: md5 of the six changed files (the working tree equals code11).

bisect11.json      key metrics (recall / precision by day, CE, FAR, anomalies, D1-D5, checklist, bindings @14,
                   PG11 items) of every final run, round 10, and every bisection point:
                   fin5_bounds_only (code10 + bounds files), t5_tree_only / t7_tree_only_with_T11-5_6 (code10 +
                   tree files, before / after T11-5/6), bis_c11t7_merged_with_T11-5_6 (code11 + T11-5/6),
                   bis_t5_tree_only (O-real seed 0 on t5). Built by bisect11.py.
A_0_full.json, E_0_full.json   packs A / E through scripts/evaluate.py (full mode, seed 0, code11).
ex_O0_21.txt, ex_Oreal0_35.txt the user's example in both views (P14 text) for pack O seed 0 day 21 and O-real
                   seed 0 day 35 (ex9.py).
Scripts: cmp11.py (one run against rounds 10 / 9 / 8 and bisection runs), tables11.py / tables2_11.py (the §16.19
tables), finsum11.py (finance after R3's move), cost11.py (engine time), pg4tab11.py (PG4 table), jdiff.py
(leaf differences, timings excluded), evq.py (the restartable queue that ran everything).
