"""evaluator round-7 driver (resumable): pack P (progressive_decision), one seed, from code CODE.
Writes OUT/<P>_<seed>.json (scored like scripts/progressive_report.py) and OUT/<P>_<seed>.res.pkl.
usage: drv6.py CODE SEED OUT [segment_s] [pack]"""
from __future__ import annotations
import json, os, pickle, resource, sys, time
for _v in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_v, "1")
CODE = sys.argv[1]
sys.path.insert(0, os.path.join(CODE, "backend"))
sys.path.insert(0, os.path.join(CODE, "scripts"))


def main():
    seed = int(sys.argv[2]); out = sys.argv[3]
    seg = float(sys.argv[4]) if len(sys.argv) > 4 else 900.0
    pname = sys.argv[5] if len(sys.argv) > 5 else "O"
    os.makedirs(out, exist_ok=True)
    import progressive_report as PR
    from app.eval.packs import get_pack
    from app.eval.resumable import run_pack_resumable
    from app.eval.report import _clean
    print("code", PR.__file__, "hashseed", os.environ.get("PYTHONHASHSEED"), flush=True)
    t0 = time.time()
    pack = get_pack(pname)
    pack.registry_mode = "progressive_decision"
    ck = os.path.join(out, f"{pname}_{seed}.ckpt")
    res = run_pack_resumable(pack, seed, ck, segment_s=seg, record_series=False,
                             on_progress=lambda m: print(f"[{pname} {seed}] {m}", flush=True))
    print("run done", time.time() - t0, "exceptions", len(res.exceptions), flush=True)
    for e in res.exceptions[:5]:
        print(str(e)[:1000])
    res.store = None
    with open(os.path.join(out, f"{pname}_{seed}.res.pkl.tmp"), "wb") as f:
        pickle.dump(res, f, protocol=pickle.HIGHEST_PROTOCOL)
    os.replace(os.path.join(out, f"{pname}_{seed}.res.pkl.tmp"), os.path.join(out, f"{pname}_{seed}.res.pkl"))
    sc = PR._score(res, seed, {"registry": pack.registry_mode, "resource_mode": "full"})
    sc["job_s"] = time.time() - t0
    sc["peak_rss_mb"] = round(resource.getrusage(resource.RUSAGE_SELF).ru_maxrss / 1024.0, 1)
    with open(os.path.join(out, f"{pname}_{seed}.json"), "w") as f:
        json.dump(_clean(sc), f, default=str, ensure_ascii=False, indent=1)
    for r in sc["example"].get("checklist") or []:
        print(r["clause"], "|", r["measured"], "|", r["pass"])
    if os.path.exists(ck):
        os.remove(ck)


if __name__ == "__main__":
    main()
