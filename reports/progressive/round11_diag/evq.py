"""Evaluator round-11 queue: 2 workers, restartable (re-launch after a reboot).
Rules: <= 2 heavy jobs; only one O-real at a time; the PG9 large full run never
beside an O-real. Done = output file present without 'error'."""
import json, os, subprocess, sys, threading, time, shutil
E = os.path.dirname(os.path.abspath(__file__))
SP = os.path.dirname(E)
C = os.path.join(E, "code11")
PY = "/home/user/appmonitor-core/.venv/bin/python"
LOG = os.path.join(SP, "evaluate.LOG.md")
OUT = os.path.join(E, "out"); CK = os.path.join(E, "ckpt")
os.makedirs(OUT, exist_ok=True); os.makedirs(CK, exist_ok=True)

def log(msg):
    with open(LOG, "a") as f:
        f.write(f"- {time.strftime('%Y-%m-%dT%H:%M:%SZ', time.gmtime())} [queue] {msg}\n")

def ok_json(fn):
    try:
        with open(fn) as f:
            return "error" not in json.load(f)
    except Exception:
        return False

def prun(pack, seed, out, hs="0", code=None):
    tag = "O" if pack == "O" else pack
    ck = os.path.join(CK, os.path.basename(out))
    cmd = [PY, "scripts/progressive_report.py", "--seeds", str(seed), "--workers", "1", "--out", out,
           "--registry", "progressive_decision", "--pack", pack, "--checkpoint", ck,
           "--segment-s", "600", "--skip-existing", "--no-series"]
    done = os.path.join(out, "runs", f"{tag}_{seed}.json")
    if pack.startswith("O-real"):          # shared machine-wide O-real lock (brief)
        cmd = ["flock", os.path.join(SP, "oreal.lock")] + cmd
    return dict(name=f"{os.path.basename(out)}:{tag}_{seed}", cmd=cmd, env={"PYTHONHASHSEED": hs},
                done=done, ckpt=ck, oreal=pack.startswith("O-real"), cwd=code or C)

JOBS = []
R = os.path.join(OUT, "main")
# O-real seed 2 first: it validates EV11-1 (pfd shared terminal: D2 / bindings @14) early
JOBS += [prun("O-real", 2, R), prun("O", 0, R), prun("O", 1, R), prun("O", 2, R), prun("O-real", 0, R),
         prun("O", 3, R), prun("O", 4, R), prun("O-real", 1, R),
         prun("O", 0, os.path.join(OUT, "det_h1"), hs="1"),
         prun("O-red", 0, R), prun("O60", 0, R)]
PG9 = os.path.join(OUT, "pg9")
for p in ("A", "E"):
    for m in ("full", "bounded"):
        JOBS.append(dict(name=f"pg9:pair_{p}_{m}", cmd=[PY, os.path.join(E, "pairdrv.py"), C, p, m, PG9],
                         env={}, done=os.path.join(PG9, f"pair_{p}_0_{m}.json"), cwd=C))
SC = os.path.join(OUT, "scale")
for k, n in (("ips", 500), ("attrs", 300), ("servers", 20), ("servers", 100), ("servers", 300), ("ips", 20000)):
    JOBS.append(dict(name=f"pg4:{k}_{n}", cmd=[PY, os.path.join(E, "pg4drv.py"), C, k, str(n), SC, os.path.join(CK, "scale")],
                     env={}, done=os.path.join(SC, f"scale_{k}_{n}.json"), cwd=C))
for m in ("full", "bounded"):
    JOBS.append(dict(name=f"pg9:large_{m}", cmd=[PY, "scripts/progressive_pg9.py", "large", "--ips", "1000", "--days", "7",
                                                  "--hours", "84", "--modes", m, "--sequential", "--workers", "1",
                                                  "--skip-existing", "--out", PG9],
                     env={}, done=os.path.join(PG9, f"large_1000_7d_84h_0_{m}.json"), cwd=C, bigmem=(m == "full")))

for p in ("A", "E"):
    JOBS.append(dict(name=f"ae:{p}_full", cmd=[PY, "scripts/evaluate.py", "--packs", p, "--seeds", "0", "--workers", "1",
                                              "--out", os.path.join(OUT, f"ae_{p}")],
                     env={"PYTHONHASHSEED": "0"}, done=os.path.join(OUT, f"ae_{p}", "eval_report.json"), cwd=C))

JOBS.append(dict(name="suite", cmd=["bash", "-c", f"cd /home/user/appmonitor-core && OMP_NUM_THREADS=1 {PY} -m pytest -q -p no:cacheprovider tests > {OUT}/suite.txt.part 2>&1; mv {OUT}/suite.txt.part {OUT}/suite.txt"],
                 env={}, done=os.path.join(OUT, "suite.txt"), cwd="/home/user/appmonitor-core"))

# extra jobs (bisects), read at every pick: ev11/extra.json = [[pack, seed, out, hs, code, prio], ...]
# prio 0 = before every queued job, 1 = after them
BASE = list(JOBS)
def load_extra():
    try:
        ex = json.load(open(os.path.join(E, "extra.json")))
    except Exception:
        ex = []
    def mk(e):
        if isinstance(e, dict):          # generic: {name, cmd, done, cwd, oreal, prio}
            return dict(name=e["name"], cmd=e["cmd"], env={"PYTHONHASHSEED": "0"}, done=e["done"],
                        cwd=e.get("cwd", C), oreal=bool(e.get("oreal")), bigmem=bool(e.get("bigmem")))
        p, s, o, h, c, pr = e
        return prun(p, s, os.path.join(OUT, o), hs=h, code=c)
    prio = lambda e: e.get("prio", 0) if isinstance(e, dict) else e[5]
    pre = [mk(e) for e in ex if prio(e) == 0]
    post = [mk(e) for e in ex if prio(e) != 0]
    JOBS[:] = pre + BASE + post
lock = threading.Lock(); running = {}; failed = set()
only = set(sys.argv[1:])

def eligible(j):
    if j["name"] in running or j["name"] in failed or ok_json(j["done"]) or (j["done"].endswith(".txt") and os.path.exists(j["done"])):
        return False
    if only and j["name"] not in only:
        return False
    rj = list(running.values())
    if j.get("oreal") and any(x.get("oreal") for x in rj): return False
    if j.get("bigmem") and any(x.get("oreal") or x.get("bigmem") for x in rj): return False
    if j.get("oreal") and any(x.get("bigmem") for x in rj): return False
    return True

def worker(wi):
    while True:
        # (round 11) worker 1 waits while another agent's heavy run is still on the
        # machine (the tree owner's dq queue: dhook11.py), so <= 2 heavy processes
        if wi > 0 and subprocess.call(["pgrep", "-f", "dhook11.py"], stdout=subprocess.DEVNULL) == 0:
            time.sleep(60); continue
        with lock:
            load_extra()
            pend = [j for j in JOBS if not (ok_json(j["done"]) or (j["done"].endswith(".txt") and os.path.exists(j["done"]))) and j["name"] not in failed and (not only or j["name"] in only)]
            if not pend:
                return
            j = next((x for x in pend if eligible(x)), None)
            if j is not None:
                running[j["name"]] = j
        if j is None:
            time.sleep(30); continue
        t0 = time.time(); log(f"start {j['name']} (worker {wi})")
        env = dict(os.environ, OMP_NUM_THREADS="1", **j["env"])
        lf = os.path.join(OUT, "logs"); os.makedirs(lf, exist_ok=True)
        with open(os.path.join(lf, j["name"].replace(":", "_").replace("/", "_") + ".log"), "a") as fo:
            rc = subprocess.call(j["cmd"], cwd=j["cwd"], env=env, stdout=fo, stderr=subprocess.STDOUT)
        okd = ok_json(j["done"]) or (j["done"].endswith(".txt") and os.path.exists(j["done"]))
        if okd and j.get("ckpt"):
            cf = os.path.join(j["ckpt"], os.path.basename(j["done"])[:-5] + ".ckpt")
            if os.path.exists(cf):
                os.remove(cf); log(f"removed checkpoint {cf}")
        log(f"end {j['name']} rc={rc} done={okd} {time.time()-t0:.0f}s")
        with lock:
            running.pop(j["name"], None)
            if not okd:
                failed.add(j["name"])

log(f"queue launched pid {os.getpid()} jobs={len(JOBS)}")
ts = [threading.Thread(target=worker, args=(i,)) for i in range(2)]
for t in ts: t.start(); time.sleep(5)
for t in ts: t.join()
log("queue finished (failed: %s)" % sorted(failed))
