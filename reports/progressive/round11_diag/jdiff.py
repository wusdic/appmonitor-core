"""jdiff.py a.json b.json: leaf differences, timing/memory keys excluded."""
import json, sys
SKIP = {"wall_s", "job_s", "engine_s", "peak_rss_mb", "cpu_mem", "timings", "rss", "wall", "cpu_s", "elapsed"}
n = [0]; diffs = []
def walk(a, b, p):
    if isinstance(a, dict) and isinstance(b, dict):
        for k in sorted(set(a) | set(b), key=str):
            if k in SKIP or "wall" in str(k) or "rss" in str(k) or str(k).endswith("_ms") or str(k).endswith("_s"):
                continue
            walk(a.get(k), b.get(k), p + "/" + str(k))
    elif isinstance(a, list) and isinstance(b, list) and len(a) == len(b):
        for i, (x, y) in enumerate(zip(a, b)):
            walk(x, y, f"{p}[{i}]")
    else:
        n[0] += 1
        if a != b:
            diffs.append((p, str(a)[:80], str(b)[:80]))
walk(json.load(open(sys.argv[1])), json.load(open(sys.argv[2])), "")
print(f"{n[0]} leaves, {len(diffs)} differences")
for d in diffs[:30]:
    print(" ", d)
