"""jdiff12.py a.json b.json: jdiff with pg1[day].failed entries cut to their first three fields
(main O_0 was scored before item 0 added the failed constraints; information only)."""
import json, sys, runpy
def norm(r):
    def walk(x):
        if isinstance(x, dict):
            if isinstance(x.get("failed"), list):
                x["failed"] = [f[:3] if isinstance(f, list) else f for f in x["failed"]]
            for v in x.values():
                walk(v)
        elif isinstance(x, list):
            for v in x:
                walk(v)
    walk(r)
    return r
a, b = (norm(json.load(open(x))) for x in sys.argv[1:3])
import tempfile, os
fa, fb = (tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) for _ in range(2))
json.dump(a, fa); json.dump(b, fb); fa.close(); fb.close()
sys.argv = ["jdiff.py", fa.name, fb.name]
runpy.run_path(os.path.join(os.path.dirname(os.path.abspath(__file__)), "jdiff.py"), run_name="__main__")
os.unlink(fa.name); os.unlink(fb.name)
