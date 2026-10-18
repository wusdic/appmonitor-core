# Round 7, evaluation-harness owner: raw results

- `pg9/pair_{A,E}_0_{full,bounded}.json`: PG9 pairs (scripts/progressive_pg9.py pairs), seed 0, working tree of 2026-10-17 16:00.
- `pg9/large_<IPs>_<pack days>d_<hours>h_0_{full,bounded}.json`: PG9 large-population runs (scripts/progressive_pg9.py large); runs stop at 8.5 GB RSS (`aborted`), `tick_cpu_ms` = lib-3 thread CPU per live tick.
- `pg9/pg9.json`: `progressive_pg9.py summary` (pmetrics.pg9_bounded) over the files above.
- `input_cost/np_{before,after}_m{0,300}.json`: thread CPU per input record of P00 / P01 / P02 on days 3-4 of a 4-day O-scale-500 run with 0 / 300 synthetic attributes (40 / 340 attributes). `before` = HEAD dda0658; `after` = HEAD + the P00 changes (flatten_meta, apply_all skip) **and the rejected P02 column-sampling variant**: its P00 / P01 numbers are those of the final code, its P02 numbers are not (P02 was left unchanged; progressive.md §16.15 item 3).
