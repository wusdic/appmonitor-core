"""lib-3 evaluation harness (docs/lib3/eval.md).

  packs    packs A-E, 'smoke' and 'mini': timelines + declarative scenarios
  truth    query helpers over gen.truth (windows, controls, aliases)
  runner   one (pack, seed) through the full pipeline -> RunResult
  metrics  RunResult -> per-run scores -> the 16 acceptance gates
  report   gates -> eval_report.json / eval_report.html

Submodules are imported on demand (runner pulls in the whole pipeline), so
`import app.eval` stays cheap.
"""
__all__ = ["packs", "truth", "runner", "metrics", "report"]
