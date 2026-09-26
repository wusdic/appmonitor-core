"""Smoke test: build runtime, warm up, run live ticks, print a summary."""
import os
import sys

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "backend"))

from app.pipeline.build import Runtime  # noqa: E402


def main():
    rt = Runtime(warmup_ticks=120, live_period_s=0.0)
    print("warming up...")
    rt.warmup()
    for i in range(16):
        stats = rt.step_once()
    print("live ticks:", rt.live_ticks, "pipeline ticks:", rt.pipeline.tick_count)
    print("\n== engines (last tick counts) ==")
    for info in rt.pipeline.engine_info():
        err = f"  ERR={info['last_error']}" if info["last_error"] else ""
        print(f"  [{info['layer']:9}] {info['name']:28} produced={info['last_count']}{err}")

    print("\n== systems / entities ==")
    for s in rt.store.systems():
        ents = rt.store.entities(s)
        print(f"  {s}: {len(ents)} entities -> {ents}")

    print("\n== profiles (archetype / separability) ==")
    for p in rt.store.all_profiles():
        print(f"  {p.system:12} {p.entity:14} arch={p.archetype:24} "
              f"sep={p.separability:.2f} stable={p.stable} n={p.sample_count}")

    print("\n== recent behaviour events ==")
    for e in rt.store.events(limit=15):
        print(f"  {e.kind:8} {e.system}/{e.entity} score={e.score} sev={e.severity.value} :: {e.description}")

    print("\n== recent signature matches ==")
    seen = set()
    for m in rt.store.matches(limit=60):
        k = (m.system, m.entity, m.signature_id)
        if k in seen:
            continue
        seen.add(k)
        print(f"  {m.system}/{m.entity} [{m.category}] {m.label} conf={m.confidence} ({m.signature_id})")

    # sanity assertions
    assert rt.store.systems(), "no systems"
    assert rt.store.all_profiles(), "no profiles"
    print("\nSMOKE OK")


if __name__ == "__main__":
    main()
