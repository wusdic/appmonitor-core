"""Portrait signatures, diffs and value masking (owner: B30 PortraitEngine).

Why a module: B30 cuts a new portrait version when two signatures differ
materially (diff_signatures), and the API's portrait-diff route must compare
two kept versions with exactly those semantics; the class page masks
adoption-record templates with the same re-masking B30 applies to every
template it publishes (safe_token: no path or query VALUE survives). Both
therefore import these pure functions from here instead of reaching into
the engine's private names.

Signatures (pure; JSON-like inputs):
    signature(portrait_json) -> {window, workload, cats, class_path}
    diff_signatures(old_sig, new_sig) -> [{field, label_zh, label_en, kind, ...}]
    safe_token(token) -> str      HTTP token / template with values re-masked
    safe_path(path) -> str;  safe_family(family) -> str
    jsd_dicts(a, b) -> float      JSD (bits) of two top-k share maps
    top_items(d, k) -> [(value, share)]
    hdist(a_h, b_h) -> float      signed circular hour distance in [-12, 12)
Thresholds (engines.md B30): DIFF_JSD 0.1 bits, DIFF_QSHIFT 25 % relative
quantile shift, DIFF_WINDOW_H 1 h of an active-window edge.
"""
from __future__ import annotations

import math
import re
from typing import Any, Dict, List, Mapping, Tuple

from .m_timing import jsd_bits
from .template import channel_of, mask_segment

DIFF_JSD = 0.1
DIFF_QSHIFT = 0.25
DIFF_WINDOW_H = 1.0
MEMBER_MIN_SHARE = 0.01
PLACEHOLDER = re.compile(r"^\{[a-z0-9_]+\}$")
LABELS = {  # diff field -> (zh, en)
    "rhythm.window": ("活跃时段", "active window"),
    "workload": ("工作负载", "workload"),
    "top_templates": ("常用模板", "top templates"),
    "activity_mix": ("活动构成", "activity mix"),
    "client": ("终端栈", "client stacks"),
    "members": ("成员", "members"),
    "class_path": ("类别", "class path"),
}


def signature(js: Mapping[str, Any]) -> Dict[str, Any]:
    """Compact state the diffs compare (see module doc)."""
    rh = js.get("rhythm") or {}
    win = {}
    for dtp in ("workday", "nonworkday"):
        b = rh.get(dtp)
        if isinstance(b, Mapping) and b.get("start_h") is not None and b.get("len_h") is not None:
            win[dtp] = [float(b["start_h"]), float(b["len_h"]), b.get("window")]
    wl = {}
    for f, b in (js.get("workload") or {}).items():
        wl[f] = {dtp: [b[dtp]["p10"], b[dtp]["p50"], b[dtp]["p90"]]
                 for dtp in ("workday", "nonworkday") if isinstance(b.get(dtp), Mapping)}
    cats: Dict[str, Dict[str, float]] = {
        "top_templates": {d["template"]: d["share"] for d in js.get("top_templates") or []
                          if d.get("share") is not None},
        "client": dict((js.get("client") or {}).get("shares") or {}),
    }
    role = js.get("role") or {}
    if role.get("activity_mix"):
        cats["activity_mix"] = {f: v for f, v in role["activity_mix"]}
    if js.get("kind") == "class":
        n = max(1, len(js.get("members") or []))
        cats["members"] = {m["ip"]: 1.0 / n for m in js.get("members") or []}
    return {"window": win, "workload": wl, "cats": cats,
            "class_path": (js.get("identity") or {}).get("class_path")
            if js.get("kind") == "entity" else None}


def diff_signatures(old: Mapping[str, Any], new: Mapping[str, Any]) -> List[Dict[str, Any]]:
    """Material changes between two signatures (engines.md B30 thresholds)."""
    out: List[Dict[str, Any]] = []
    ow, nw = old.get("window") or {}, new.get("window") or {}
    for dtp in ("workday", "nonworkday"):
        a, b = ow.get(dtp), nw.get(dtp)
        if a is None and b is None:
            continue
        if a is None or b is None:
            sh = math.inf
        else:
            d0 = abs(hdist(a[0], b[0]))
            d1 = abs(hdist(a[0] + a[1], b[0] + b[1]))
            sh = max(d0, d1)
        if sh > DIFF_WINDOW_H:
            out.append(diff_item("rhythm.window", kind="window", day_type=dtp,
                                 before=a[2] if a else None, after=b[2] if b else None,
                                 shift_h=None if math.isinf(sh) else round(sh, 2)))
    for f, nb in (new.get("workload") or {}).items():
        ob = (old.get("workload") or {}).get(f) or {}
        for dtp, qn in nb.items():
            qo = ob.get(dtp)
            if qo is None:
                continue
            rel = max(abs(x - y) / max(abs(y), 1.0) for x, y in zip(qn, qo))
            if rel > DIFF_QSHIFT:
                out.append(diff_item("workload", kind="quantile", feature=f, day_type=dtp,
                                     before=list(qo), after=list(qn),
                                     max_rel_shift=round(rel, 3)))
    for key, nd in (new.get("cats") or {}).items():
        od = (old.get("cats") or {}).get(key)
        if od is None or (not od and not nd):
            continue
        j = jsd_dicts(od, nd)
        if j > DIFF_JSD:
            added = [v for v, _ in top_items(nd, 5) if od.get(v, 0.0) < MEMBER_MIN_SHARE]
            removed = [v for v, _ in top_items(od, 5) if nd.get(v, 0.0) < MEMBER_MIN_SHARE]
            inc = sorted(((x - od.get(v, 0.0), v) for v, x in nd.items()), reverse=True)
            gained = [v for dx, v in inc[:3] if dx > 0.05]
            out.append(diff_item(key, kind="categorical", jsd=round(j, 4), added=added,
                                 removed=removed, gained=gained,
                                 before=[v for v, _ in top_items(od, 3)],
                                 after=[v for v, _ in top_items(nd, 3)]))
    if old.get("class_path") != new.get("class_path") and new.get("class_path") is not None \
            and old.get("class_path") is not None:
        out.append(diff_item("class_path", kind="categorical", before=old.get("class_path"),
                             after=new.get("class_path")))
    return out


def diff_item(field: str, **kw: Any) -> Dict[str, Any]:
    """One diff entry with its zh / en field label (LABELS)."""
    zh, en = LABELS[field]
    return {"field": field, "label_zh": zh, "label_en": en, **kw}


def safe_token(tok: Any) -> str:
    """Re-mask an HTTP token / template so no path or query VALUE survives
    (placeholders kept, other segments through mask_segment, query -> names)."""
    if not isinstance(tok, str):
        return "" if tok is None else str(tok)
    if not tok or tok[:1] == "{" or channel_of(tok) != "http":
        return tok
    parts = tok.split(" ", 2)
    if len(parts) < 3:
        return tok
    meth, host, rest = parts
    path, sep, status = rest.partition("|")
    return f"{meth} {host} {safe_path(path)}" + (sep + status if sep else "")


def safe_path(path: str) -> str:
    p, _, query = path.partition("?")
    segs = []
    for seg in p.split("/"):
        if not seg:
            continue
        segs.append(seg if PLACEHOLDER.match(seg) else mask_segment(seg)[0])
    out = "/" + "/".join(segs)
    if query:
        names = sorted({n if PLACEHOLDER.match(n) else mask_segment(n)[0]
                        for n in (x.split("=", 1)[0] for x in query.split("&")) if n})
        if names:
            out += "?" + "&".join(names)
    return out


def safe_family(fam: str) -> str:
    """'http|read|host|seg0': seg0 is the first template segment; re-masked."""
    ps = str(fam).split("|")
    if len(ps) == 4 and ps[0] == "http" and ps[3] and not PLACEHOLDER.match(ps[3]):
        ps[3] = mask_segment(ps[3])[0]
    return "|".join(ps)


def jsd_dicts(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    """JSD (bits) of two top-k share maps, each completed with an '__other__' mass."""
    keys = sorted(set(a) | set(b))
    pa = [max(0.0, float(a.get(k, 0.0))) for k in keys]
    pb = [max(0.0, float(b.get(k, 0.0))) for k in keys]
    pa.append(max(0.0, 1.0 - sum(pa)))
    pb.append(max(0.0, 1.0 - sum(pb)))
    j = jsd_bits(pa, pb)
    return j if j == j else (0.0 if not (sum(pa) or sum(pb)) else 1.0)


def top_items(d: Mapping[str, float], k: int) -> List[Tuple[str, float]]:
    return sorted(((str(v), float(x)) for v, x in d.items() if x == x),
                  key=lambda t: (-t[1], t[0]))[:k]


def hdist(a: float, b: float) -> float:
    return ((float(b) - float(a) + 12.0) % 24.0) - 12.0
