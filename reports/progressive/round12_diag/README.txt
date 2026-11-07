Round 12 diagnostics (evaluator, 2026-11-07; docs/lib3/progressive.md §16.20). JSON / text only, no pickles.

bisect12.json          key metrics (cmp12.py's summary) of every round-12 run: main (code12, final), b6 (EV12-2's B6 variant),
                       det_h1 (PYTHONHASHSEED=1), bis_LV / bis_VB / bis_V / bis_L (code11 + the named owners' files).
tables12.txt           tables12.py output: pack O per seed, medians, progress with time, pooled claims, O-real, O-red, O60
                       against rounds 11, 10, 9, 8.
ex_O0_21.txt           the user's example in both views, pack O seed 0 day 21 (ex9.py on runs/O_0.json).
ex_Oreal0_35.txt       the same for O-real seed 0 day 35.
pg4tab.md              PG4 points r10 -> r11 -> r12 (pg4tab12.py).
A_0_full.json, E_0_full.json   scripts/evaluate.py on packs A / E (code12); identical to round11_diag in every non-timing leaf.
code12_owned_files.md5 md5 of the ten files that differ from code11 (= the working tree at the final runs).
evq12.py, run.sh, extra.json   the restartable evaluation queue and its extra (bisection) jobs.
cmp12.py, tables12.py, jdiff.py, jdiff12.py, finrows.py, cost12.py, pg4tab12.py, ex9.py, bisect12.py   analysis scripts.

Every run JSON in ../runs/ carries the scorer's round-12 lists: score.pg1[day].failed = [pattern_id, confidence,
observable, [[failed constraint, nominal, held], ...]] (runs O_0, O-red_0 and O-real_2 have the first three fields only:
they were scored before the constraint field was added) and score.pg6.far.incidents (the incidents FAR counts).
