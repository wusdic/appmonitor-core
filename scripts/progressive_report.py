"""Progressive-core evaluation report (docs/lib3/progressive.md §12, §16).

Runs pack O (the requirement's organisation) for the given seeds through the
production registry (build.build_registry; the pack asks for
'full+progressive': B01-B30 + P00-P15 + lib-4), scores every run with
eval/pmetrics (PG1-PG11), and extracts the requirement's example in BOTH views
from the day snapshots:

  system view   OA: every statement that names a 综合部 IP (login, approvals,
                reports); finance: the approval-list / approve statements
  group view    the learned group(s) holding 综合部's IPs: header, statements
                and negative statements ("... 在财务系统中从未执行写操作")
  checklist     each clause of the requirement's example, measured against the
                generator's truth (who, window, size band / range, username
                grammar, bindings, approvals, reports, finance approver)
  anomalies     A1-A10: pattern_violation type and incident severity per seed

Writes <out>/progressive_report.json and <out>/progressive_report.html; with
--scale <peval dir>, the PG4 points of scripts/evaluate.py --scale are merged.

  .venv/bin/python scripts/progressive_report.py --seeds 0,1,2 --workers 3 --out reports/progressive
  .venv/bin/python scripts/progressive_report.py --render reports/progressive   # HTML from saved JSON
"""
from __future__ import annotations

import argparse
import html
import json
import math
import os
import sys
import time
import traceback
from concurrent.futures import ProcessPoolExecutor, as_completed
from typing import Any, Dict, List, Mapping, Optional, Sequence

for _var in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
    os.environ.setdefault(_var, "1")
sys.path.insert(0, os.path.join(os.path.dirname(os.path.abspath(__file__)), "..", "backend"))

GA_IPS = ("192.168.1.21", "192.168.1.23", "10.168.7.121")
FIN_APPROVER = "192.168.2.10"


# ------------------------------------------------------------------ helpers
def _stmts(snap: Mapping[str, Any], system: str) -> List[Dict[str, Any]]:
    pv = ((snap.get("systems") or {}).get(system) or {}).get("model.pviews") or {}
    return list(pv.get("statements") or [])


def _who_ips(st: Mapping[str, Any]) -> List[str]:
    who = ((st.get("evidence") or {}).get("who") or {})
    out = [str(x) for x in (who.get("members") or who.get("items") or []) if "/" not in str(x)]
    return out


def _route(st: Mapping[str, Any]) -> str:
    return str((st.get("evidence") or {}).get("route") or "")


def _brief(st: Mapping[str, Any]) -> Dict[str, Any]:
    ev = st.get("evidence") or {}
    return {"pattern_id": st.get("pattern_id"), "route": ev.get("route"), "state": st.get("state"),
            "confidence": st.get("confidence"), "support": st.get("support"),
            "who": {k: (ev.get("who") or {}).get(k) for k in ("level", "items", "members", "group")},
            "when": ev.get("when"), "content": ev.get("content"), "bindings": ev.get("bindings"),
            "workflow": ev.get("workflow"), "context": ev.get("context"),
            "text_zh": st.get("text_zh"), "text_en": st.get("text_en")}


def _jacc(a: Sequence[str], b: Sequence[str]) -> float:
    A, B = set(a), set(b)
    return len(A & B) / len(A | B) if (A or B) else 0.0


def _iou(ws: Sequence[Sequence[float]], truth: Sequence[Sequence[float]]) -> float:
    def cover(iv):
        s = set()
        for a, b in iv or []:
            s |= set(range(int(a), int(math.ceil(b))))
        return s
    A, B = cover(ws), cover(truth)
    return len(A & B) / len(A | B) if (A or B) else 0.0


def example_views(res: Any) -> Dict[str, Any]:
    """The requirement's example, read from the day snapshots (both views)."""
    snaps = {int(k): v for k, v in (res.psnaps or {}).items()}
    if not snaps:
        return {}
    days = sorted(snaps)
    pick = sorted({d for d in (7, 11, 14, 21, days[-1]) if d in snaps})
    truth = [r for r in (res.ptruth or {}).get("pattern_truth") or []]
    out: Dict[str, Any] = {"days": {}}
    for d in pick:
        snap = snaps[d]
        oa = [s for s in _stmts(snap, "oa") if set(_who_ips(s)) & set(GA_IPS)
              or any(ip in json.dumps((s.get("evidence") or {}).get("who") or {}) for ip in GA_IPS)]
        fin = [s for s in _stmts(snap, "finance") if "/fin/approval" in _route(s)]
        wg = (snap.get("org") or {}).get("model.who_groups") or {}
        ip2g = wg.get("ip2g") or {}
        gids = sorted({ip2g[ip] for ip in GA_IPS if ip in ip2g})
        gviews = []
        for g in gids:
            gv = (snap.get("group_views") or {}).get(f"class:grp:{g}") or {}
            gr = (wg.get("groups") or {}).get(g) or {}
            gviews.append({"group": g, "name": gr.get("name"), "name_source": gr.get("name_source"),
                           "members": gr.get("members"),
                           "header_zh": (gv.get("header") or {}).get("text_zh"),
                           "header_en": (gv.get("header") or {}).get("text_en"),
                           "statements": [{"text_zh": s.get("text_zh"), "text_en": s.get("text_en"),
                                           "negative": bool((s.get("evidence") or {}).get("negative")
                                                            or "从未" in str(s.get("text_zh")))}
                                          for s in gv.get("statements") or []]})
        out["days"][d] = {"oa_statements": [_brief(s) for s in oa],
                          "finance_statements": [_brief(s) for s in fin],
                          "group_views": gviews,
                          "ga_groups": {ip: ip2g.get(ip) for ip in GA_IPS}}
    out["checklist"] = checklist(out["days"].get(days[-1]) or {}, truth, days[-1])
    return out


def checklist(day: Mapping[str, Any], truth: Sequence[Mapping[str, Any]], d: int) -> List[Dict[str, Any]]:
    """Each clause of the requirement's example against the truth valid on day d."""
    def valid(act: str, step: int = 0) -> Optional[Mapping[str, Any]]:
        for r in truth:
            if r.get("activity") == act and int(r.get("step", 0)) == step and \
                    int(r.get("valid_from_day", 0)) <= d < int(r.get("valid_to_day", 99)):
                return r
        return None

    oa = day.get("oa_statements") or []
    fin = day.get("finance_statements") or []
    rows: List[Dict[str, Any]] = []

    def best(stmts, route_part, who):
        sc = []
        for s in stmts:
            if route_part not in str(s.get("route") or ""):
                continue
            ips = [str(x) for x in ((s.get("who") or {}).get("members") or (s.get("who") or {}).get("items") or [])
                   if "/" not in str(x)]
            sc.append((_jacc(ips, who), s))
        return max(sc, key=lambda x: x[0]) if sc else (0.0, None)

    t_login = valid("GA.oa.login")
    if t_login:
        j, s = best(oa, "/login", t_login["who"]["value"])
        w = (s or {}).get("when") or {}
        rows.append({"clause": "综合部 3 个 IP 登录 OA（who = 192.168.1.21、192.168.1.23、10.168.7.121）",
                     "measured": f"who Jaccard {j:.2f}", "pass": j >= 0.8,
                     "statement": (s or {}).get("text_zh")})
        iou = _iou(w.get("workday") or [], t_login["windows"]["workday"]) if s else 0.0
        rows.append({"clause": "工作日登录时间窗 " + "–".join(f"{int(a)//60:02d}:{int(a)%60:02d}"
                                                     for a in t_login["windows"]["workday"][0]),
                     "measured": f"IoU {iou:.2f}", "pass": iou >= 0.7})
        bl = (((s or {}).get("content") or {}).get("body.len") or {})
        band = bl.get("band90")
        rows.append({"clause": "提交数据量 90 % 在 1–2 KB", "measured": f"band {band}",
                     "pass": bool(band) and abs(band[0] - 1024) <= 205 and abs(band[1] - 2048) <= 410})
        rng = bl.get("range")
        rows.append({"clause": "100 % 在 0.5–3 KB", "measured": f"range {rng}",
                     "pass": bool(rng) and abs(rng[0] - 512) <= 128 and abs(rng[1] - 3072) <= 768})
        un = (((s or {}).get("content") or {}).get("body.kv.username") or {})
        g = un.get("grammar")
        rows.append({"clause": "提交内容含 username=，取值不超过 10 个字符", "measured": f"grammar {g}",
                     "pass": bool(g)})
        b = (((s or {}).get("bindings") or {}).get("body.kv.username") or {}).get("table") or {}
        want = t_login.get("bindings", {}).get("body.kv.username") or {}
        hit = sum(1 for ip, u in want.items() if str(u) in (b.get(ip) or []) or b.get(ip) == u)
        rows.append({"clause": "绑定 " + "、".join(f"{ip}→{u}" for ip, u in want.items()),
                     "measured": f"{hit}/{len(want)}", "pass": hit == len(want) and hit > 0})
    for act, route, label in (("GA.oa.approvals", "/approval", "192.168.1.21 访问业务审批页面"),
                              ("GA.oa.report", "/report/generate", "下午 5 点提交报告（.23、.121）")):
        r = next((x for x in truth if x.get("activity") == act and "generate" in str(x.get("route")) or
                  (x.get("activity") == act and route in str(x.get("route")))), None)
        if r is None:
            continue
        j, s = best(oa, route, r["who"]["value"])
        rows.append({"clause": label, "measured": f"who Jaccard {j:.2f}", "pass": j >= 0.8,
                     "statement": (s or {}).get("text_zh")})
    r = valid("FIN.finance.approval")
    if r:
        j, s = best(fin, "/fin/approval", r["who"]["value"])
        rows.append({"clause": "财务系统审批只有财务部 192.168.2.10 访问", "measured": f"who Jaccard {j:.2f}",
                     "pass": j >= 0.8, "statement": (s or {}).get("text_zh")})
    neg = any(st.get("negative") and ("finance" in str(st.get("text_zh")) or "财务" in str(st.get("text_zh")))
              for gv in day.get("group_views") or [] for st in gv.get("statements") or [])
    rows.append({"clause": "用户视角：综合部在财务系统中从未执行写操作（否定陈述）",
                 "measured": "present" if neg else "absent", "pass": neg})
    return rows


def anomaly_table(res: Any, sc: Mapping[str, Any]) -> Dict[str, Any]:
    an = ((sc.get("pg6") or {}).get("anomalies") or {})
    return {k: {"violation": v.get("violation"), "types": v.get("types"), "flags": v.get("flags"),
                "incident": v.get("incident"), "detected": v.get("detected")} for k, v in an.items()}


# ------------------------------------------------------------------ worker
def _job(seed: int, opts: Mapping[str, Any]) -> Dict[str, Any]:
    from app.eval.pmetrics import score_prun
    from app.eval.report import _clean
    from app.eval.runner import run_pack
    from app.eval.packs import get_pack
    t0 = time.perf_counter()
    try:
        pack = get_pack("O")
        if opts.get("registry"):
            pack.registry_mode = opts["registry"]
        if opts.get("bounded"):
            pack.config.setdefault("lib3", {})["resource_mode"] = "bounded"
        res = run_pack(pack, seed, strict=True, record_series=True)
        sc = score_prun(res)
        sc["exceptions"] = len(res.exceptions)
        sc["exception_samples"] = [str(e)[:300] for e in res.exceptions[:5]]
        ex = example_views(res)
        tm = res.timings or {}
        eng = {}
        if tm.get("engine_names") is not None:
            import numpy as np
            em = np.asarray(tm["engine_ms"])
            for i, n in enumerate(tm["engine_names"]):
                eng[n] = round(float(em[:, i].sum()) / 1000.0, 1)
        out = {"seed": seed, "score": sc, "example": ex, "anomalies": anomaly_table(res, sc),
               "engine_s": eng, "wall_s": res.wall_s, "registry": pack.registry_mode,
               "incidents": len(res.incidents), "events": len(res.events)}
    except Exception as exc:
        return {"seed": seed, "error": f"{type(exc).__name__}: {exc}", "traceback": traceback.format_exc()[-4000:]}
    out["job_s"] = time.perf_counter() - t0
    return _clean(out)


# ------------------------------------------------------------------ render
def _fmt(v: Any) -> str:
    if isinstance(v, float):
        return "—" if not math.isfinite(v) else f"{v:.3g}"
    if v is None:
        return "—"
    return html.escape(str(v))


def render_html(rep: Mapping[str, Any]) -> str:
    css = """
    :root{--bg:#fff;--fg:#1d2127;--mut:#5b6573;--line:#dde2e8;--ok:#1a7f37;--bad:#b42318;--card:#f6f8fa}
    @media (prefers-color-scheme: dark){:root{--bg:#111418;--fg:#e6e9ee;--mut:#9aa4b2;--line:#2b3138;--ok:#4ac26b;--bad:#ff7b72;--card:#171b21}}
    body{background:var(--bg);color:var(--fg);font:14px/1.55 -apple-system,Segoe UI,'PingFang SC','Microsoft YaHei',sans-serif;margin:0;padding:24px 16px;max-width:1200px;margin:auto}
    h1{font-size:22px}h2{font-size:18px;margin-top:32px;border-bottom:1px solid var(--line);padding-bottom:4px}
    h3{font-size:15px;margin-top:20px}table{border-collapse:collapse;width:100%;margin:8px 0;font-size:13px}
    th,td{border:1px solid var(--line);padding:4px 8px;text-align:left;vertical-align:top}th{background:var(--card)}
    .ok{color:var(--ok);font-weight:600}.bad{color:var(--bad);font-weight:600}.mut{color:var(--mut)}
    .stmt{background:var(--card);border:1px solid var(--line);border-radius:6px;padding:8px 10px;margin:6px 0;font-size:13px}
    .wrap{overflow-x:auto}"""
    p: List[str] = [f"<!doctype html><html lang='zh'><head><meta charset='utf-8'>"
                    f"<meta name='viewport' content='width=device-width,initial-scale=1'>"
                    f"<title>渐进画像评估</title><style>{css}</style></head><body>"]
    p.append("<h1>渐进画像内核评估 · Progressive profile core evaluation (pack O)</h1>")
    p.append(f"<p class='mut'>generated {html.escape(str(rep.get('generated')))} · seeds "
             f"{html.escape(str(rep.get('seeds')))} · registry {html.escape(str(rep.get('registry')))}</p>")
    p.append("<h2>Gates PG1–PG11</h2><div class='wrap'><table><tr><th>gate</th><th>pass</th><th>value</th>"
             "<th>target</th></tr>")
    for name, g in (rep.get("gates") or {}).items():
        ps = g.get("pass")
        cls = "ok" if ps else ("bad" if ps is False else "mut")
        p.append(f"<tr><td>{_fmt(name)} {_fmt(g.get('name'))}</td><td class='{cls}'>{_fmt(ps)}</td>"
                 f"<td>{_fmt(g.get('value'))}</td><td>{_fmt(g.get('target'))}</td></tr>")
    p.append("</table></div>")
    p.append("<h2>Per seed</h2><div class='wrap'><table><tr><th>seed</th><th>recall@14</th><th>precision@14</th>"
             "<th>ECE</th><th>who</th><th>when</th><th>content</th><th>bindings</th><th>workflow</th>"
             "<th>GA+FIN bindings</th><th>ARI</th><th>pv FAR</th><th>wall s</th></tr>")
    for r in rep.get("runs") or []:
        if "error" in r:
            p.append(f"<tr><td>{r['seed']}</td><td colspan=12 class='bad'>{_fmt(r['error'])}</td></tr>")
            continue
        sc = r["score"]
        d14 = sc.get("pg1_day14") or {}
        c = d14.get("components") or {}
        p.append(f"<tr><td>{r['seed']}</td><td>{_fmt(d14.get('recall'))}</td><td>{_fmt(d14.get('precision'))}</td>"
                 f"<td>{_fmt(d14.get('ece'))}</td>" + "".join(f"<td>{_fmt(c.get(k))}</td>" for k in
                                                               ("who", "when", "content", "bindings", "workflow"))
                 + f"<td>{_fmt(sc.get('bindings_ga_fin_day14'))}</td><td>{_fmt((sc.get('pg3') or {}).get('ari'))}</td>"
                 f"<td>{_fmt(((sc.get('pg6') or {}).get('far') or {}).get('pv_far_low'))}</td>"
                 f"<td>{_fmt(r.get('wall_s'))}</td></tr>")
    p.append("</table></div>")
    p.append("<h2>Recall over observation time (PG2)</h2><div class='wrap'><table><tr><th>seed</th>")
    days = sorted({int(d) for r in rep.get("runs") or [] if "score" in r
                   for d in ((r["score"].get("pg2") or {}).get("recall") or {})})
    p.append("".join(f"<th>d{d}</th>" for d in days) + "</tr>")
    for r in rep.get("runs") or []:
        if "score" not in r:
            continue
        rc = (r["score"].get("pg2") or {}).get("recall") or {}
        p.append(f"<tr><td>{r['seed']}</td>" + "".join(f"<td>{_fmt(rc.get(str(d), rc.get(d)))}</td>"
                                                          for d in days) + "</tr>")
    p.append("</table></div>")
    p.append("<h2>The requirement's example · 需求示例（两种视角）</h2>")
    for r in rep.get("runs") or []:
        ex = r.get("example") or {}
        if not ex:
            continue
        p.append(f"<h3>seed {r['seed']} — checklist (day {max(map(int, ex.get('days') or {0: 0}))})</h3>")
        p.append("<div class='wrap'><table><tr><th>clause</th><th>measured</th><th>pass</th></tr>")
        for c in ex.get("checklist") or []:
            cls = "ok" if c.get("pass") else "bad"
            p.append(f"<tr><td>{_fmt(c['clause'])}</td><td>{_fmt(c['measured'])}</td>"
                     f"<td class='{cls}'>{_fmt(c.get('pass'))}</td></tr>")
        p.append("</table></div>")
        last = max(ex.get("days") or {}, key=lambda k: int(k))
        dd = ex["days"][last]
        p.append(f"<h3>seed {r['seed']} · 业务系统视角 OA（day {last}）</h3>")
        for s in dd.get("oa_statements") or []:
            p.append(f"<div class='stmt'>{_fmt(s.get('text_zh'))}</div>")
        if not dd.get("oa_statements"):
            p.append("<p class='mut'>no OA statement names a 综合部 IP</p>")
        p.append(f"<h3>seed {r['seed']} · 业务系统视角 财务系统审批（day {last}）</h3>")
        for s in dd.get("finance_statements") or []:
            p.append(f"<div class='stmt'>{_fmt(s.get('text_zh'))}</div>")
        if not dd.get("finance_statements"):
            p.append("<p class='mut'>no finance approval statement</p>")
        p.append(f"<h3>seed {r['seed']} · 用户视角（含综合部 IP 的学习群组，day {last}）</h3>")
        for gv in dd.get("group_views") or []:
            p.append(f"<div class='stmt'><b>{_fmt(gv.get('header_zh'))}</b><br><span class='mut'>members "
                     f"{_fmt(gv.get('members'))}</span></div>")
            for s in (gv.get("statements") or [])[:12]:
                p.append(f"<div class='stmt'>{_fmt(s.get('text_zh'))}</div>")
    p.append("<h2>Anomalies A1–A10 (PG6)</h2><div class='wrap'><table><tr><th>seed</th>"
             + "".join(f"<th>A{i}</th>" for i in range(1, 11)) + "</tr>")
    for r in rep.get("runs") or []:
        an = r.get("anomalies") or {}
        cells = []
        for i in range(1, 11):
            a = an.get(f"A{i}") or {}
            txt = ("pv " + ",".join(a.get("types") or []) if a.get("violation") else "no pv") + \
                  (" · inc" if a.get("incident") else "")
            cls = "ok" if a.get("detected") else ("mut" if a.get("violation") else "bad")
            cells.append(f"<td class='{cls}'>{_fmt(txt)}</td>")
        p.append(f"<tr><td>{r['seed']}</td>{''.join(cells)}</tr>")
    p.append("</table></div>")
    if rep.get("scale"):
        p.append("<h2>Scaling (PG4)</h2><pre>" + html.escape(json.dumps(rep["scale"], indent=1,
                                                                         ensure_ascii=False)[:6000]) + "</pre>")
    p.append("</body></html>")
    return "".join(p)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--seeds", default="0,1,2")
    ap.add_argument("--workers", type=int, default=3)
    ap.add_argument("--out", default="reports/progressive")
    ap.add_argument("--registry", default=None, help="override the pack's registry mode")
    ap.add_argument("--bounded", action="store_true", help="lib3.resource_mode = bounded")
    ap.add_argument("--scale", default=None, help="directory with scale_*.json points (pscale.run_point)")
    ap.add_argument("--render", default=None, help="only re-render the HTML of a saved report dir")
    args = ap.parse_args()
    if args.render:
        with open(os.path.join(args.render, "progressive_report.json"), encoding="utf-8") as f:
            rep = json.load(f)
        with open(os.path.join(args.render, "progressive_report.html"), "w", encoding="utf-8") as f:
            f.write(render_html(rep))
        return
    from app.eval.pmetrics import compute_pgates
    from app.eval.pscale import pg4_summary
    from app.eval.report import _clean
    seeds = [int(s) for s in args.seeds.split(",") if s.strip()]
    os.makedirs(os.path.join(args.out, "runs"), exist_ok=True)
    opts = {"registry": args.registry, "bounded": args.bounded}
    runs: List[Dict[str, Any]] = []
    t0 = time.perf_counter()
    with ProcessPoolExecutor(max_workers=max(1, args.workers)) as ex:
        futs = {ex.submit(_job, s, opts): s for s in seeds}
        for fut in as_completed(futs):
            r = fut.result()
            runs.append(r)
            with open(os.path.join(args.out, "runs", f"O_{r['seed']}.json"), "w", encoding="utf-8") as f:
                json.dump(r, f, indent=1, ensure_ascii=False)
            print(f"[{time.perf_counter() - t0:7.1f}s] seed {r['seed']}: {r.get('error') or 'ok'}", flush=True)
    runs.sort(key=lambda r: r["seed"])
    scale_pts = []
    if args.scale and os.path.isdir(args.scale):
        for fn in sorted(os.listdir(args.scale)):
            if fn.startswith("scale_") and fn.endswith(".json"):
                with open(os.path.join(args.scale, fn), encoding="utf-8") as f:
                    pt = json.load(f)
                if "error" not in pt:
                    scale_pts.append(pt)
    scores = [dict(r["score"], pack="O") for r in runs if "score" in r]
    gates = compute_pgates(scores, scale=pg4_summary(scale_pts) if scale_pts else None)
    rep = _clean({"generated": time.strftime("%Y-%m-%d %H:%M:%S"), "seeds": seeds,
                  "registry": args.registry or "pack default (full+progressive)",
                  "gates": gates, "runs": runs,
                  "scale": {"points": scale_pts, "pg4": pg4_summary(scale_pts)} if scale_pts else None})
    with open(os.path.join(args.out, "progressive_report.json"), "w", encoding="utf-8") as f:
        json.dump(rep, f, indent=1, ensure_ascii=False)
    with open(os.path.join(args.out, "progressive_report.html"), "w", encoding="utf-8") as f:
        f.write(render_html(rep))
    for name, g in gates.items():
        print(f"{name}: pass={g.get('pass')} value={g.get('value')}")


if __name__ == "__main__":
    main()
