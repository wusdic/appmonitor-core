"""eval_report.json and a self-contained eval_report.html (docs/lib3/eval.md).

Why both: the JSON is the single source for every accuracy number (gate 16
regenerates the design doc's section 3.3 table from it, and CI diffs it),
while the HTML is what GET /api/eval/report serves to a human. The HTML has
no external dependencies (inline CSS, no scripts, no fonts or CDNs) so it
renders the same offline, inside the API, and as a CI artifact.
"""
from __future__ import annotations

import datetime as _dt
import html
import json
import math
from typing import Any, Dict, List, Mapping, Optional, Sequence

import numpy as np

from .metrics import compute_gates, ttd_table


def _clean(obj: Any) -> Any:
    """JSON-safe copy: numpy scalars/arrays -> Python, NaN/inf -> None."""
    if isinstance(obj, dict):
        return {str(k): _clean(v) for k, v in obj.items()}
    if isinstance(obj, (list, tuple, set)):
        return [_clean(v) for v in obj]
    if isinstance(obj, np.ndarray):
        return _clean(obj.tolist())
    if isinstance(obj, (np.floating, float)):
        x = float(obj)
        return x if math.isfinite(x) else None
    if isinstance(obj, (np.integer,)):
        return int(obj)
    if isinstance(obj, np.bool_):
        return bool(obj)
    return obj


def build_report(scores: Sequence[Dict[str, Any]],
                 feedback: Optional[Sequence[Dict[str, Any]]] = None,
                 ablation: Optional[Mapping[str, Sequence[Dict[str, Any]]]] = None,
                 smoke: Optional[Dict[str, Any]] = None,
                 meta: Optional[Dict[str, Any]] = None) -> Dict[str, Any]:
    """The full report dict: gates, per-scenario table, FAR / legit tables,
    run list and the regenerated design table."""
    full = [s for s in scores if not s.get("disabled_engines")]
    gates = compute_gates(full, feedback=feedback, ablation=ablation, smoke=smoke)
    passed = [g["pass"] for g in gates.values()]
    report = {
        "generated": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "meta": meta or {},
        "summary": {
            "n_runs": len(full),
            "packs": sorted({s["pack"] for s in full}),
            "seeds": sorted({int(s["seed"]) for s in full}),
            "gates_pass": sum(1 for p in passed if p is True),
            "gates_fail": sum(1 for p in passed if p is False),
            "gates_na": sum(1 for p in passed if p is None),
            "all_pass": all(p is True for p in passed),
        },
        "gates": gates,
        "scenarios": ttd_table(full),
        "far": gates["3_far"]["details"].get("table", []),
        "legit": gates["4_legit"]["details"].get("table", []),
        "runs": [{"pack": s["pack"], "seed": s["seed"], "aborted": s.get("aborted"),
                  "wall_s": s["cpu_mem"].get("wall_s"),
                  "exceptions": s["robustness"]["exceptions"]} for s in full],
        "design_table_md": gates["16_report"]["details"].get("table_md", ""),
    }
    return _clean(report)


def write_json(report: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        json.dump(_clean(report), f, indent=1, ensure_ascii=False, sort_keys=False)


# --------------------------------------------------------------------------- #
# HTML
# --------------------------------------------------------------------------- #
_CSS = """
:root{--bg:#fbfbfa;--fg:#1d1d1b;--muted:#6b6b66;--line:#e3e2de;--card:#ffffff;
--ok:#1f7a45;--ok-bg:#e6f4ea;--bad:#b3261e;--bad-bg:#fbe9e7;--na:#6b6b66;--na-bg:#efefec}
@media (prefers-color-scheme:dark){:root:not([data-theme="light"]){--bg:#151514;--fg:#ecebe7;
--muted:#9c9b95;--line:#2f2f2c;--card:#1d1d1b;--ok:#7fd49b;--ok-bg:#16301f;--bad:#f2a49b;
--bad-bg:#3a1a17;--na:#a7a6a0;--na-bg:#2a2a27}}
:root[data-theme="dark"]{--bg:#151514;--fg:#ecebe7;--muted:#9c9b95;--line:#2f2f2c;--card:#1d1d1b;
--ok:#7fd49b;--ok-bg:#16301f;--bad:#f2a49b;--bad-bg:#3a1a17;--na:#a7a6a0;--na-bg:#2a2a27}
*{box-sizing:border-box}
body{margin:0;background:var(--bg);color:var(--fg);
font:14px/1.5 system-ui,-apple-system,"Segoe UI",Roboto,"PingFang SC","Microsoft YaHei",sans-serif}
main{max-width:1180px;margin:0 auto;padding:24px 16px 64px}
h1{font-size:22px;margin:0 0 4px}h2{font-size:17px;margin:32px 0 10px}
.muted{color:var(--muted)}.tiles{display:flex;flex-wrap:wrap;gap:12px;margin:16px 0}
.tile{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:10px 14px;min-width:120px}
.tile b{display:block;font-size:20px;font-variant-numeric:tabular-nums}
.wrap{overflow-x:auto;border:1px solid var(--line);border-radius:8px;background:var(--card)}
table{border-collapse:collapse;width:100%;font-variant-numeric:tabular-nums}
th,td{padding:6px 10px;border-bottom:1px solid var(--line);text-align:left;vertical-align:top;white-space:nowrap}
th{font-weight:600;color:var(--muted);font-size:12px;text-transform:uppercase;letter-spacing:.03em}
td.num{text-align:right}tr:last-child td{border-bottom:0}
.pill{display:inline-block;padding:1px 8px;border-radius:999px;font-size:12px;font-weight:600}
.pass{color:var(--ok);background:var(--ok-bg)}.fail{color:var(--bad);background:var(--bad-bg)}
.na{color:var(--na);background:var(--na-bg)}
details{margin:0}summary{cursor:pointer}
.checks td{white-space:normal}
pre{background:var(--card);border:1px solid var(--line);border-radius:8px;padding:12px;overflow-x:auto;font-size:12px}
"""


def _e(v: Any) -> str:
    return html.escape(str(v))


def _num(v: Any, digits: int = 4) -> str:
    if v is None:
        return "–"
    if isinstance(v, bool):
        return "yes" if v else "no"
    if isinstance(v, float):
        if abs(v) >= 1e5 or (v != 0 and abs(v) < 1e-3):
            return f"{v:.3g}"
        return f"{v:.{digits}g}"
    if isinstance(v, (list, tuple)):
        return "[" + ", ".join(_num(x, digits) for x in v) + "]"
    return _e(v)


def _pill(p: Optional[bool]) -> str:
    if p is True:
        return '<span class="pill pass">pass</span>'
    if p is False:
        return '<span class="pill fail">fail</span>'
    return '<span class="pill na">n/a</span>'


def _ci(c: Mapping[str, Any]) -> str:
    ci = c.get("ci95")
    if not ci or ci[0] is None:
        return ""
    return f' <span class="muted">[{_num(ci[0])}, {_num(ci[1])}]</span>'


def _table(headers: List[str], rows: List[List[str]], num_cols: Sequence[int] = ()) -> str:
    th = "".join(f"<th>{_e(h)}</th>" for h in headers)
    body = []
    for r in rows:
        tds = "".join(f'<td class="num">{c}</td>' if i in num_cols else f"<td>{c}</td>"
                      for i, c in enumerate(r))
        body.append(f"<tr>{tds}</tr>")
    return f'<div class="wrap"><table><thead><tr>{th}</tr></thead><tbody>{"".join(body)}' \
           f"</tbody></table></div>"


def render_html(report: Dict[str, Any]) -> str:
    s = report.get("summary", {})
    gates = report.get("gates", {})
    parts = [f"<h1>Behavior library evaluation</h1>"
             f'<div class="muted">generated {_e(report.get("generated", ""))} · packs '
             f'{_e(", ".join(s.get("packs", [])))} · seeds '
             f'{_e(", ".join(str(x) for x in s.get("seeds", [])))}</div>',
             '<div class="tiles">'
             f'<div class="tile">runs<b>{s.get("n_runs", 0)}</b></div>'
             f'<div class="tile">gates passed<b>{s.get("gates_pass", 0)}</b></div>'
             f'<div class="tile">gates failed<b>{s.get("gates_fail", 0)}</b></div>'
             f'<div class="tile">not computable<b>{s.get("gates_na", 0)}</b></div></div>']

    # gates
    rows = []
    for gid, g in gates.items():
        checks = (g.get("details") or {}).get("checks") or []
        inner = "".join(
            f"<tr><td>{_e(c.get('name'))}</td><td class=\"num\">{_num(c.get('value'))}{_ci(c)}</td>"
            f"<td class=\"num\">{_num(c.get('target'))}</td><td>{_pill(c.get('pass'))}</td></tr>"
            for c in checks)
        det = (f'<details><summary>{_e(g.get("name"))}</summary><table class="checks">'
               f"<tbody>{inner}</tbody></table></details>" if checks else _e(g.get("name")))
        rows.append([_e(gid.split("_")[0]), det, _num(g.get("value")), _num(g.get("target")),
                     _pill(g.get("pass"))])
    parts.append("<h2>Gates</h2>" + _table(["#", "Gate (expand for checks, 95% CI over seeds)",
                                             "Value", "Target", "Result"], rows, (2, 3)))

    # scenarios
    rows = []
    for r in report.get("scenarios", []):
        rows.append([_e(r["pack"]), _e(r["scenario_id"]), _e(r["loudness"]),
                     f'{r["n_detected"]}/{r["n_runs"]}', _num(r["within_deadline"], 3),
                     _num(r["ttd_ticks_median"], 3), _num(r["ttd_ticks_p90"], 3),
                     _num(None if r["ttd_s_median"] is None else r["ttd_s_median"] / 60.0, 3),
                     _num(None if r["ttd_s_p90"] is None else r["ttd_s_p90"] / 60.0, 3),
                     _e(r.get("max_severity") or "–"), _e(r.get("required_severity") or "–")])
    parts.append("<h2>Threat scenarios</h2>" + _table(
        ["Pack", "Scenario", "Class", "Detected", "In deadline", "TTD ticks p50", "p90",
         "TTD min p50", "p90", "Max severity", "Required"], rows, (3, 4, 5, 6, 7, 8)))

    # FAR
    rows = [[_e(r["pack"]), _e(r["seed"]), _num(r["entity_days"], 4), _num(r["n_low"]),
             _num(r["n_medium"]), _num(r["n_high"]), _num(r["n_critical"]),
             _num(r["far_low"], 3), _num(r["far_medium"], 3)] for r in report.get("far", [])]
    parts.append("<h2>False alarms on control entities</h2>" + _table(
        ["Pack", "Seed", "Entity-days", "≥ LOW", "≥ MEDIUM", "≥ HIGH", "CRITICAL",
         "FAR ≥ LOW / e-day", "FAR ≥ MEDIUM / e-day"], rows, (1, 2, 3, 4, 5, 6, 7, 8)))

    # legit
    rows = [[_e(r["pack"]), _e(r["seed"]), _e(r["scenario_id"]), _e(r.get("max_severity") or "–"),
             _e(r.get("allowed")), _num(r.get("max_risk"), 3), _pill(r.get("ok"))]
            for r in report.get("legit", [])]
    parts.append("<h2>Legitimate changes</h2>" + _table(
        ["Pack", "Seed", "Scenario", "Max severity", "Allowed", "Max risk", "Result"], rows,
        (1, 5)))

    # ablation (gate 13): per engine, the scenarios it is the sole / main
    # detector of and the FAR change with it disabled
    abl = ((gates.get("13_ablation") or {}).get("details") or {}).get("table") or []
    if abl:
        rows = []
        for r in abl:
            worse = sorted(k for k, d in (r.get("delta_recall") or {}).items()
                           if d is not None and d < 0)
            better = sorted(k for k, d in (r.get("delta_recall") or {}).items()
                            if d is not None and d > 0)
            rows.append([_e(r.get("engine")), _e(", ".join(r.get("sole_detector_of") or []) or "–"),
                         _e(", ".join(r.get("main_detector_of") or []) or "–"),
                         _e(", ".join(worse) or "–"), _e(", ".join(better) or "–"),
                         _num(r.get("delta_far_low"), 3)])
        parts.append("<h2>Ablation (one engine disabled)</h2>" + _table(
            ["Engine off", "Sole detector of", "Main detector of", "Recall lost",
             "Recall gained", "ΔFAR ≥ LOW / e-day"], rows, (5,)))

    # free-form sections the caller adds through meta['sections'] (e.g. the
    # before / after comparison of an evaluation round)
    for sec in (report.get("meta") or {}).get("sections") or []:
        hdr = [str(h) for h in sec.get("headers") or []]
        body = [[_e(c) if not isinstance(c, (int, float)) or isinstance(c, bool) else _num(c, 4)
                 for c in row] for row in sec.get("rows") or []]
        note = f'<p class="muted">{_e(sec.get("note"))}</p>' if sec.get("note") else ""
        parts.append(f"<h2>{_e(sec.get('title', ''))}</h2>{note}" + _table(hdr, body))

    rows = [[_e(r["pack"]), _e(r["seed"]), _num(r.get("wall_s"), 4), _num(r.get("exceptions")),
             _e(r.get("aborted") or "")] for r in report.get("runs", [])]
    parts.append("<h2>Runs</h2>" + _table(["Pack", "Seed", "Wall s", "Engine exceptions",
                                            "Aborted"], rows, (1, 2, 3)))
    parts.append("<h2>Design table (section 3.3, generated)</h2><pre>"
                 f"{_e(report.get('design_table_md', ''))}</pre>")
    return ("<!doctype html><html lang=\"en\"><head><meta charset=\"utf-8\">"
            "<meta name=\"viewport\" content=\"width=device-width,initial-scale=1\">"
            f"<title>Eval Report</title><style>{_CSS}</style></head><body><main>"
            + "".join(parts) + "</main></body></html>")


def write_html(report: Dict[str, Any], path: str) -> None:
    with open(path, "w", encoding="utf-8") as f:
        f.write(render_html(report))


def write_report(scores: Sequence[Dict[str, Any]], out_dir: str, **kw: Any) -> Dict[str, Any]:
    """Build and write eval_report.json + eval_report.html into out_dir."""
    import os
    os.makedirs(out_dir, exist_ok=True)
    rep = build_report(scores, **kw)
    write_json(rep, os.path.join(out_dir, "eval_report.json"))
    write_html(rep, os.path.join(out_dir, "eval_report.html"))
    return rep
