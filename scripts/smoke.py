"""Smoke test: build the v2 runtime (full lib-3 registry), warm up, run live
ticks, print a summary of what the behaviour library produced.

    python scripts/smoke.py [--warmup N | --plan 120x3600,192x900] [--live 16] [--strict]

Warm-up runs with training=True over the Runtime's warm-up plan (spec v2.1
default 120 x 3600 s + 192 x 900 s; `--warmup N` keeps the v2 plan N x 900 s),
the live phase at the runtime's 60 s window (a 900 -> 60 s cadence switch, as
in eval pack E); every tick is stamped with the generator's virtual clock.
Exits non-zero on an engine error.
"""
import argparse
import math
import os
import sys
import time

# single-threaded BLAS / OpenMP before numpy loads (integration R15.1 / R19.3)
for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.pipeline.build import Runtime  # noqa: E402


def _latest(store, s, e, name):
    ts, M = store.vec_tail(s, e, name, 1)
    return float(M[-1][0]) if len(ts) else math.nan


def _sev(v):
    return getattr(v, "value", v)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warmup", type=int, default=None,
                    help="v2 plan: N warm-up ticks at 900 s")
    ap.add_argument("--plan", default=None,
                    help="warm-up plan 'ticks x dt, ...', e.g. 120x3600,192x900")
    ap.add_argument("--live", type=int, default=16)
    ap.add_argument("--strict", action="store_true", help="engines re-raise (ctx.config strict)")
    args = ap.parse_args()

    plan = None
    if args.plan:
        plan = [(int(a), float(b)) for a, b in
                (x.strip().lower().split("x") for x in args.plan.split(",") if x.strip())]
    rt = Runtime(warmup_ticks=args.warmup, live_period_s=0.0, strict=args.strict,
                 warmup_plan=plan)
    st = rt.store
    print(f"grain mode {rt.config.get('grain_mode')}; warm-up plan {rt.plan_text()}, training")
    for w in rt.plan_warnings(time.time()):
        print(f"  warning: {w}")
    t0 = time.perf_counter()
    rt.warmup()
    t1 = time.perf_counter()
    for _ in range(args.live):
        rt.step_once()
    t2 = time.perf_counter()
    print(f"warm-up {t1 - t0:.1f} s ({(t1 - t0) / max(rt.warmup_ticks, 1) * 1e3:.0f} ms/tick), "
          f"live {args.live} x {rt.window_s} s in {t2 - t1:.1f} s; "
          f"pipeline ticks {rt.pipeline.tick_count}")

    print("\n== engines (last tick) ==")
    n_err = 0
    for info in rt.pipeline.engine_info():
        n_err += int(info["error_count"])
        err = f"  ERR x{info['error_count']}: {info['last_error']}" if info["error_count"] else ""
        print(f"  [{info['layer']:9}] {info['name']:28} out={info['last_count']:4} "
              f"{info['duration_ms']:7.2f} ms{err}")

    print("\n== systems / entities ==")
    for s in st.systems():
        ents = st.entities(s)
        classes = [p for p in st.pseudo_entities(s) if p.startswith("class:")]
        print(f"  {s}: {len(ents)} entities, classes {classes}")

    print("\n== profiles (class path / separability / maturity) ==")
    for p in sorted(st.all_profiles(), key=lambda p: (p.system, p.entity)):
        if p.entity.startswith("__"):
            continue
        ex = p.extra or {}
        path = (ex.get("peer_group") or {}).get("class_path") or "-"
        stage = (ex.get("maturity") or {}).get("stage", "-")
        sep = p.separability if isinstance(p.separability, float) else float("nan")
        print(f"  {p.system:12} {p.entity:18} path={path:28} sep={sep:.2f} "
              f"stable={p.stable} n={p.sample_count} maturity={stage}")

    print("\n== risk top-10 (behavior.risk, entities and classes) ==")
    rows = []
    for s in st.systems():
        for e in st.entities(s) + [p for p in st.pseudo_entities(s) if p.startswith("class:")]:
            r = _latest(st, s, e, "behavior.risk")
            if r == r:
                tier = ((st.profile(s, e) or None) and (st.profile(s, e).extra.get("risk") or {})
                        .get("tier")) or "-"
                rows.append((r, s, e, tier))
    for r, s, e, tier in sorted(rows, reverse=True)[:10]:
        print(f"  {r:6.1f}  {tier:8} {s}/{e}")

    print("\n== incidents ==")
    incs = sorted(st.incidents(), key=lambda i: i.opened)
    for inc in incs[-20:]:
        print(f"  {inc.id:10} {inc.system}/{inc.entity:18} {_sev(inc.severity):8} {inc.status:9} "
              f"kinds={sorted(inc.kinds or [])} axes={sorted(inc.axes or [])}")
    print(f"  ({len(incs)} total, {sum(1 for i in incs if i.status != 'closed')} not closed)")

    print("\n== recent behaviour events ==")
    for e in st.events(limit=15):
        print(f"  {e.kind:22} {e.system}/{e.entity} sev={_sev(e.severity)} :: {e.description[:90]}")

    print("\n== separability (B15) ==")
    for s in st.systems():
        m = st.get_model(s, "__system__", "model.identity")
        stats = (m or {}).get("stats") or {}
        seps = [v.get("separability") for v in stats.values() if isinstance(v, dict)]
        seps = [x for x in seps if isinstance(x, (int, float)) and x == x]
        if seps:
            print(f"  {s}: {len(seps)} enrolled, separability median "
                  f"{sorted(seps)[len(seps) // 2]:.2f} min {min(seps):.2f}")
        else:
            print(f"  {s}: identity model not fitted yet")

    assert st.systems(), "no systems"
    assert st.all_profiles(), "no profiles"
    assert rows, "no behavior.risk written"
    if n_err:
        print(f"\nSMOKE FAILED: {n_err} engine errors")
        sys.exit(1)
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
