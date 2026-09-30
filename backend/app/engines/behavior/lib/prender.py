"""Statement rendering of the progressive core (docs/lib3/progressive.md §6.17.2, P14).

STATUS: implemented (W-P7, P14 maths). Pure functions; no store access.
P14 (`behavior.views`) builds statement objects from pattern nodes and the
fitted models; this module turns each constraint into (a) the machine-readable
`evidence` block of the statement contract (eval/pmetrics header) and (b) its
zh / en phrase, and assembles the sentence:

  【OA · 192.168.100.100:8080】工作日 09:00–09:21，综合部（192.168.1.21、192.168.1.23、
  10.168.7.121）访问 POST /login：提交数据量 90 % 在 1–2 KB，全部在 0.5–3 KB（n = 183，
  下次越界概率 ≤ 1.1 %）；…。置信 0.97 · 首次 2026-09-01 · 最近 2026-09-28 · v7.3

Rules (§6.17.2):
  who   <= 8 IPs covering >= 95 % of the node's mass with unseen mass U <= 0.05
        -> the IP list (prefixed with the group name when one group holds them
        all); else one group covering >= 90 % -> the group name (members listed
        when <= 5); else <= 4 prefixes (/24, /16) or regions covering >= 90 % ->
        the prefix list (a learned P11 region is rendered as its CIDR covers);
        else "任意 IP（约 N 个，分散）". Closed = U <= 0.05 and >= 5 active days.
  numbers  P06's display rounding (1-2-5 grid keeping the coverage); sizes in
        B / KB / MB; percentages without decimals unless < 1 %.
  confidence  min over the statement's constraints of: 1 - U (closed who),
        P09 coverage x stability (when), P06 band confidence, c_g (1 - U_s)
        (grammar), 1 - U (closed value set), min LB_x (bindings), the minimum
        edge confidence (workflow); x 0.5 for stale patterns.
Attribute display names come from ATTR_NAMES (namespace globs, a rendering
vocabulary only — every attribute renders, unknown ones under their own
name); config progressive.attr_names extends it.
"""
from __future__ import annotations

import datetime as _dt
import fnmatch
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

from . import pbounds as PB
from . import pwindows as PW

NAN = float("nan")
IP_LIST_MAX = 8
IP_COVER = 0.95
U_CLOSED = 0.05
GROUP_COVER = 0.9
MEMBERS_LISTED = 5
PREFIX_MAX = 4
PREFIX_COVER = 0.9
CLOSED_DAYS = 5
STALE_FACTOR = 0.5
WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})

# display vocabulary: (glob, zh, en); first match wins
ATTR_NAMES: List[Tuple[str, str, str]] = [
    ("body.len", "提交数据量", "submitted size"),
    ("net.bytes_up", "上行字节", "bytes up"),
    ("net.bytes_down", "下行字节", "bytes down"),
    ("net.dur_ms", "时长", "duration"),
    ("http.status", "状态码", "status"),
    ("http.sclass", "状态类别", "status class"),
    ("http.method", "方法", "method"),
    ("body.keys", "表单键", "form keys"),
    ("q.keys", "查询参数", "query keys"),
    ("client.stack", "客户端栈", "client stack"),
    ("rate.ip_h", "每 IP 每小时次数", "events per IP-hour"),
    ("body.kv.*", "{key}=", "{key}="),
    ("q.kv.*", "查询参数 {key}=", "query {key}="),
    ("hdr.*", "请求头 {key}", "header {key}"),
    ("meta.*", "{key}", "{key}"),
]


def attr_label(a: str, lang: str = "zh", extra: Optional[Sequence[Tuple[str, str, str]]] = None) -> str:
    for g, zh, en in list(extra or ()) + ATTR_NAMES:
        if fnmatch.fnmatchcase(a, g):
            key = a[len(g.split("*", 1)[0]):] if "*" in g else a
            return (zh if lang == "zh" else en).format(key=key)
    return a


def local_date(ts: Optional[float], tz_off: float) -> str:
    if ts is None or not math.isfinite(float(ts)):
        return "?"
    return _dt.datetime.utcfromtimestamp(float(ts) + tz_off).strftime("%Y-%m-%d")


def pct(x: float, up: bool = False) -> str:
    """Percentages without decimals unless < 1 % (§6.17.2). An upper bound
    (up=True) is rounded UP, with one decimal below 10 %, so '≤ 1.1 %' is never
    rendered as a smaller '≤ 1 %'."""
    if not math.isfinite(x):
        return "?"
    v = 100.0 * x
    if up:
        if v < 10.0:
            return f"{math.ceil(v * 10.0 - 1e-9) / 10.0:.1f} %"
        return f"{math.ceil(v - 1e-9):.0f} %"
    return f"{v:.1f} %" if v < 1.0 else f"{v:.0f} %"


def fnum(x: float, nd: int = 2) -> str:
    if not math.isfinite(float(x)):
        return "?"
    return f"{float(x):.{nd}f}"


def route_parts(route: Any) -> Tuple[str, str]:
    """('POST', '/login') from 'POST oa.corp.local /login'; ('', 'TLS mail.x') for others."""
    s = str(route or "")
    p = s.split()
    if len(p) >= 2 and p[0].isupper() and p[0] not in ("TLS", "DNS", "DST"):
        return p[0], p[-1]
    return "", s


def route_text(route: Any) -> str:
    m, r = route_parts(route)
    return f"{m} {r}".strip()


def join_zh(xs: Iterable[str]) -> str:
    return "、".join(x for x in xs if x)


def join_en(xs: Iterable[str]) -> str:
    return ", ".join(x for x in xs if x)


# ===================================================================== who
def _ip_sort(s: str) -> Tuple[int, int, str]:
    import ipaddress
    x = s[7:] if s.startswith("shared:") else s
    try:
        a = ipaddress.ip_address(x)
        return (a.version, int(a), s)
    except ValueError:
        return (9, 0, s)


def who_block(who: Any, t: float, n_days: int, ip2g: Mapping[str, str],
              groups: Mapping[str, Mapping[str, Any]], regions_cfg: Iterable[str] = (),
              mode: Optional[str] = None) -> Tuple[Dict[str, Any], str, str, float]:
    """(evidence who, zh, en, confidence) of a pnode.WhoSummary (§6.17.2 who
    rule). `regions_cfg` = configured region names (rendered 'reg:<name>');
    a learned region (a P11 group id) renders as that group's CIDR covers."""
    lv0 = who.levels[0]
    tot = lv0.total(t)
    distinct = int(round(who.distinct(t))) if tot > 0 else 0
    if tot <= 0:
        return ({"level": "any", "items": ["*"], "closed": False, "U": NAN, "confidence": 0.0},
                "（尚无来源）", "(no source yet)", 0.0)
    U0 = float(lv0.unseen(t))
    heavy, cov = who.heavy_set(0, t)
    closed0 = bool(U0 <= U_CLOSED and n_days >= CLOSED_DAYS)
    regions_cfg = set(regions_cfg)
    if mode != "none" and len(heavy) <= IP_LIST_MAX and cov >= IP_COVER - 1e-9 and U0 <= U_CLOSED:
        ips = sorted((str(x) for x in heavy), key=_ip_sort)
        plain = [x[7:] if x.startswith("shared:") else x for x in ips]
        gs = {ip2g.get(x) for x in plain}
        gname = ""
        if len(gs) == 1 and None not in gs:
            g = next(iter(gs))
            gr = groups.get(g) or {}
            if set(gr.get("members") or []) and set(plain) <= set(gr.get("members") or []):
                gname = str(gr.get("name") or g)
        lst_zh, lst_en = join_zh(ips), join_en(ips)
        zh = f"{gname}（{lst_zh}）" if gname else lst_zh
        en = f"{gname} ({lst_en})" if gname else lst_en
        if not closed0:
            zh = f"目前观测到 {zh}（来源集合尚未封闭）"
            en = f"so far {en} (source set not yet closed)"
        ev = {"level": "ip", "items": ips, "members": plain, "U": U0, "closed": closed0,
              "confidence": 1.0 - U0 if closed0 else 0.0, "distinct": distinct}
        return ev, zh, en, (1.0 - U0) if closed0 else 1.0
    # one group covering >= 90 %
    lv3 = who.levels[3] if len(who.levels) > 3 else None
    if mode != "none" and lv3 is not None and lv3.total(t) > 0:
        hg, cg = who.heavy_set(3, t, GROUP_COVER)
        if len(hg) == 1 and str(hg[0]).startswith("grp:") and cg >= GROUP_COVER - 1e-9 and \
                str(hg[0]) != "grp:∅":
            g = str(hg[0])[4:]
            gr = groups.get(g) or {}
            mem = [str(m) for m in gr.get("members") or []]
            name = str(gr.get("name") or g)
            U3 = float(lv3.unseen(t))
            closed = bool(U3 <= U_CLOSED and n_days >= CLOSED_DAYS)
            if mem and len(mem) <= MEMBERS_LISTED:
                zh, en = f"{name}（{join_zh(mem)}）", f"{name} ({join_en(mem)})"
            else:
                zh, en = f"{name}（{len(mem)} 个 IP）", f"{name} ({len(mem)} IPs)"
            ev = {"level": "grp", "items": [f"grp:{g}"], "members": mem, "U": U3, "closed": closed,
                  "confidence": 1.0 - U3 if closed else 0.0, "distinct": distinct}
            return ev, zh, en, (1.0 - U3) if closed else 1.0
    # <= 4 prefixes or regions covering >= 90 %
    for l in (1, 2, 4):
        if l >= len(who.levels) or who.levels[l].total(t) <= 0:
            continue
        hp, cp = who.heavy_set(l, t, PREFIX_COVER)
        if not hp or len(hp) > PREFIX_MAX or cp < PREFIX_COVER - 1e-9:
            continue
        items: List[str] = []
        ok = True
        for x in hp:
            x = str(x)
            if l == 4:
                if not x.startswith("reg:") or x == "reg:∅":
                    ok = False
                    break
                nm = x[4:]
                if nm in regions_cfg:
                    items.append(x)
                else:
                    cv = (groups.get(nm) or {}).get("covers") or []
                    if not cv:
                        ok = False
                        break
                    items.extend(str(c) for c in cv)
            else:
                items.append(x)
        if not ok or not items:
            continue
        Ul = float(who.levels[l].unseen(t))
        closed = bool(Ul <= U_CLOSED and n_days >= CLOSED_DAYS)
        level = "reg" if l == 4 and all(i.startswith("reg:") for i in items) else "prefix"
        shown = [i[4:] if i.startswith("reg:") else i for i in items]
        zh = f"来自 {join_zh(shown)}（约 {distinct} 个 IP）"
        en = f"from {join_en(shown)} (about {distinct} IPs)"
        ev = {"level": level, "items": items, "U": Ul, "closed": closed,
              "confidence": 1.0 - Ul if closed else 0.0, "distinct": distinct}
        return ev, zh, en, (1.0 - Ul) if closed else 1.0
    ev = {"level": "any", "items": ["*"], "U": U0, "closed": False, "confidence": 0.0,
          "distinct": distinct}
    return ev, f"任意 IP（约 {distinct} 个，分散）", f"any IP (about {distinct}, dispersed)", 1.0


# ==================================================================== when
def when_block(entry: Optional[Mapping[str, Any]]) -> Tuple[Optional[Dict[str, Any]], str, str, float]:
    """P09 node entry -> (evidence when, zh, en, confidence)."""
    if not entry or entry.get("status") != "fitted":
        return None, "", "", 1.0
    w = entry.get("when") or {}
    if not (w.get("workday") or w.get("nonworkday")):
        return None, "", "", 1.0
    zh, en = [], []
    for dt in ("wd", "nwd"):
        rec = (entry.get("by_daytype") or {}).get(dt)
        if rec and (rec.get("windows") or rec.get("all_day")):
            zh.append(rec.get("text_zh") or PW.render_zh(dt, rec))
            en.append(rec.get("text_en") or PW.render_en(dt, rec))
    conf = float(w.get("confidence", w.get("coverage", NAN)))
    ev = {"workday": [list(x) for x in w.get("workday") or []],
          "nonworkday": [list(x) for x in w.get("nonworkday") or []],
          "coverage": w.get("coverage"), "confidence": conf}
    return ev, "；".join(zh), "; ".join(en), conf if math.isfinite(conf) else 1.0


# ================================================================= content
def _num_phrase(a: str, rec: Mapping[str, Any], labels: Any) -> Tuple[Dict[str, Any], str, str]:
    d90 = rec.get("disp90") or {}
    lo, hi = d90.get("lo", NAN), d90.get("hi", NAN)
    if not (isinstance(lo, (int, float)) and math.isfinite(lo) and math.isfinite(hi)):
        lo, hi = rec["band90"]
    cov = rec.get("coverage", NAN)
    dc = d90.get("coverage", NAN)
    if isinstance(dc, (int, float)) and math.isfinite(dc) and math.isfinite(float(cov)):
        cov = min(float(cov), float(dc))
    ev: Dict[str, Any] = {"band90": [float(lo), float(hi)], "coverage": cov,
                          "band90_raw": list(rec.get("band90") or []), "n_eff": rec.get("n_eff"),
                          "n_c": rec.get("n_c"), "approx": rec.get("approx", 0.0),
                          "confidence": rec.get("confidence")}
    unit = rec.get("unit", "")
    band_txt = d90.get("text") or PB._text(lo, hi, unit)
    name_zh, name_en = attr_label(a, "zh", labels), attr_label(a, "en", labels)
    zh = f"{name_zh} 90 % 在 {band_txt}"
    en = f"90 % of {name_en} within {band_txt}"
    if rec.get("range") is not None:
        dr = rec.get("disp_range") or {}
        rlo, rhi = dr.get("lo", rec["range"][0]), dr.get("hi", rec["range"][1])
        ev.update({"range": [float(rlo), float(rhi)], "range_raw": list(rec["range"]),
                   "n_rng": rec.get("n_rng"), "cover": rec.get("cover"), "hard": rec.get("hard")})
        rt = dr.get("text") or PB._text(rlo, rhi, unit)
        n_rng = int(round(float(rec.get("n_rng") or 0)))
        if rec.get("hard"):
            zh += f"，全部在 {rt}（n = {n_rng}，下次越界概率 ≤ {pct(float(rec.get('cover', NAN)), up=True)}）"
            en += f", all within {rt} (n = {n_rng}, P(next outside) ≤ {pct(float(rec.get('cover', NAN)), up=True)})"
        else:
            zh += f"，观测范围 {rt}（n = {n_rng}）"
            en += f", observed range {rt} (n = {n_rng})"
    elif rec.get("observed") is not None:
        ev["observed"] = list(rec["observed"])
        zh += "（近似：聚合记录）"
        en += " (approximate: aggregated records)"
    return ev, zh, en


def _text_phrase(a: str, rec: Mapping[str, Any], labels: Any) -> Tuple[Dict[str, Any], str, str, float]:
    ev: Dict[str, Any] = {"n": rec.get("n")}
    name_zh, name_en = attr_label(a, "zh", labels), attr_label(a, "en", labels)
    zh, en, conf = [], [], []
    if rec.get("grammar"):
        ev.update({"grammar": rec["grammar"], "c_g": rec.get("c_g"), "U_s": rec.get("U_s"),
                   "len": rec.get("len")})
        zh.append(f"{name_zh} 取值 `{rec['grammar']}`")
        en.append(f"{name_en} matching `{rec['grammar']}`")
        cg, us = float(rec.get("c_g", NAN)), float(rec.get("U_s", NAN))
        if math.isfinite(cg) and math.isfinite(us):
            conf.append(cg * (1.0 - us))
    if rec.get("closed") is not None:
        vals = [str(v) for v in rec["closed"]]
        ev.update({"closed": vals, "U": rec.get("U")})
        shown = vals if len(vals) <= 8 else vals[:8] + ["…"]
        zh.append(f"{name_zh} 取值集合封闭 {{{join_en(shown)}}}")
        en.append(f"{name_en} in the closed set {{{join_en(shown)}}}")
        u = float(rec.get("U", NAN))
        if math.isfinite(u):
            conf.append(1.0 - u)
    return ev, "，".join(zh), ", ".join(en), min(conf) if conf else 1.0


def _set_phrase(a: str, rec: Mapping[str, Any], labels: Any) -> Tuple[Dict[str, Any], str, str, float]:
    req = [str(k) for k in rec.get("required") or []]
    ev = {"required": req, "optional": list(rec.get("optional") or []),
          "p_new_key": rec.get("p_new_key"), "n": rec.get("n")}
    if not req:
        return ev, "", "", 1.0
    keys_zh = join_zh(f"{k}=" for k in req)
    keys_en = join_en(f"{k}=" for k in req)
    zh = f"{attr_label(a, 'zh', labels)}必含 {keys_zh}"
    en = f"{attr_label(a, 'en', labels)} always carry {keys_en}"
    return ev, zh, en, float(rec.get("confidence", 1.0))


def content_block(bounds: Optional[Mapping[str, Any]], grammar: Optional[Mapping[str, Any]],
                  labels: Any = None, skip: Iterable[str] = ()
                  ) -> Tuple[Dict[str, Any], List[str], List[str], float]:
    """Fitted P06 / P07 node entries -> (evidence content, zh phrases, en phrases, confidence)."""
    content: Dict[str, Any] = {}
    zh: List[str] = []
    en: List[str] = []
    confs: List[float] = []
    skip = set(skip)
    for a, rec in sorted(((bounds or {}).get("attrs") or {}).items()):
        if a in skip or not rec or not rec.get("band90"):
            continue
        ev, z, e = _num_phrase(a, rec, labels)
        content[a] = ev
        zh.append(z)
        en.append(e)
        c = rec.get("confidence")
        if c is not None and math.isfinite(float(c)):
            confs.append(float(c))
    for a, rec in sorted(((grammar or {}).get("attrs") or {}).items()):
        if a in skip or not rec:
            continue
        k = rec.get("kind")
        if k == "set":
            ev, z, e, c = _set_phrase(a, rec, labels)
        elif k == "cat":
            if rec.get("closed") is None:
                continue
            ev, z, e, c = _text_phrase(a, rec, labels)
        else:
            ev, z, e, c = _text_phrase(a, rec, labels)
        if not z:
            continue
        content.setdefault(a, {}).update(ev)
        zh.append(z)
        en.append(e)
        confs.append(c)
    return content, zh, en, (min(confs) if confs else 1.0)


# ================================================================ bindings
def binding_block(bind: Optional[Mapping[str, Any]], labels: Any = None
                  ) -> Tuple[Dict[str, Any], str, str, float]:
    """P08 node entry -> (evidence bindings, zh, en, min LB). Forward pairs
    render 'x → y' for bound sources when the dependency holds and set bindings
    always; reverse pairs render 'y ← {x...}' for closed source sets."""
    out: Dict[str, Any] = {}
    zh, en, lbs = [], [], []
    for pk, rec in sorted(((bind or {}).get("pairs") or {}).items()):
        if not rec:
            continue
        X, Y = rec.get("x"), rec.get("y")
        if not X or not Y:
            X, _, Y = pk.partition("->")
        holds = bool((rec.get("fd") or {}).get("holds"))
        table, LB, rev = {}, {}, {}
        for x, ent in sorted((rec.get("table") or {}).items()):
            if ent.get("bound") and holds:
                table[x] = ent.get("top")
                if ent.get("LB") is not None:
                    LB[x] = float(ent["LB"])
            elif ent.get("set"):
                table[x] = [str(v) for v in ent["set"]]
        if not table:
            continue
        if rec.get("dir") == "rev":
            # Y -> X fitted on the pair (Y, X): table {y: x | [x...]}
            for y, xs in table.items():
                rev[str(y)] = xs if isinstance(xs, list) else [xs]
            b = out.setdefault(str(X), {"x": str(Y)})
            b["reverse"] = rev
            name = attr_label(str(X), "zh", labels)
            zh.append("；".join(f"{name}{y} 只来自 {join_zh(v)}" for y, v in list(rev.items())[:8]))
            en.append("; ".join(f"{attr_label(str(X), 'en', labels)}{y} only from {join_en(v)}"
                                for y, v in list(rev.items())[:8]))
        else:
            b = out.setdefault(str(Y), {"x": str(X)})
            b["table"] = {str(k): v for k, v in table.items()}
            b["LB"] = LB
            fd = rec.get("fd") or {}
            b["g3"] = fd.get("g3")
            b["one_to_one"] = fd.get("one_to_one")
            name = attr_label(str(Y), "zh", labels)
            items_zh = [f"{x} → {name}{v if not isinstance(v, list) else '{' + ','.join(v) + '}'}"
                        for x, v in list(table.items())[:8]]
            items_en = [f"{x} → {attr_label(str(Y), 'en', labels)}"
                        f"{v if not isinstance(v, list) else '{' + ','.join(v) + '}'}"
                        for x, v in list(table.items())[:8]]
            ns = [float(e.get("n", 0)) for e in (rec.get("table") or {}).values() if e.get("bound")]
            tail = f"（g3 = {fnum(float(fd.get('g3', NAN)))}，各 ≥ {int(min(ns))} 次）" if ns else ""
            tail_en = f" (g3 = {fnum(float(fd.get('g3', NAN)))}, each ≥ {int(min(ns))} times)" if ns else ""
            zh.append("绑定：" + join_zh(items_zh) + tail)
            en.append("bound values " + join_en(items_en) + tail_en)
        lbs.extend(LB.values())
    return out, "；".join(zh), "; ".join(en), (min(lbs) if lbs else 1.0)


# ================================================================ workflow
def workflow_block(edges: Sequence[Mapping[str, Any]]) -> Tuple[List[Dict[str, Any]], str, str, float]:
    ev = []
    zh, en, confs = [], [], []
    for e in edges:
        band = e.get("band") or [NAN, NAN]
        ev.append({"from": e.get("from"), "to": e.get("to"), "dep": e.get("dep"), "band": list(band),
                   "confidence": e.get("confidence")})
        bz = ""
        if band and all(isinstance(x, (int, float)) and math.isfinite(x) for x in band):
            bz = f"（间隔 {_dur_zh(band[0])}–{_dur_zh(band[1])}）"
        zh.append(f"{route_text(e.get('from'))} → {route_text(e.get('to'))}{bz}")
        en.append(f"{route_text(e.get('from'))} → {route_text(e.get('to'))}")
        c = e.get("confidence")
        if c is not None and math.isfinite(float(c)):
            confs.append(float(c))
    return ev, ("流程：" + "；".join(zh)) if zh else "", ("workflow: " + "; ".join(en)) if en else "", \
        (min(confs) if confs else 1.0)


def _dur_zh(s: float) -> str:
    s = float(s)
    if s < 90:
        return f"{s:.0f} 秒"
    if s < 5400:
        return f"{s / 60:.0f} 分钟"
    return f"{s / 3600:.1f} 小时"


# ================================================================ sentence
def sentence(sys_label: str, addr: str, when_zh: str, when_en: str, who_zh: str, who_en: str,
             route: str, content_zh: Sequence[str], content_en: Sequence[str],
             bind_zh: str, bind_en: str, flow_zh: str, flow_en: str, conf: float,
             first: str, last: str, version: int, cver: int, state: str
             ) -> Tuple[str, str]:
    head_zh = f"【{sys_label}" + (f" · {addr}" if addr else "") + "】"
    head_en = f"[{sys_label}" + (f" · {addr}" if addr else "") + "] "
    verb_zh = "访问" if not route.startswith(("TLS", "DNS", "DST")) else "连接"
    body_zh = (f"{when_zh}，" if when_zh else "") + f"{who_zh}{verb_zh} {route}"
    body_en = (f"On {when_en}, " if when_en else "") + f"{who_en} {'opens' if verb_zh == '访问' else 'connects to'} {route}"
    parts_zh = [p for p in list(content_zh) + [bind_zh] if p]
    parts_en = [p for p in list(content_en) + [bind_en] if p]
    if parts_zh:
        body_zh += "：" + "；".join(parts_zh)
        body_en += ": " + "; ".join(parts_en)
    body_zh += "。"
    body_en += "."
    if flow_zh:
        body_zh += flow_zh + "。"
        body_en += " " + flow_en[:1].upper() + flow_en[1:] + "."
    st_zh = {"stale": "（近期未出现）", "evolving": "（正在变化）"}.get(state, "")
    st_en = {"stale": " (not seen recently)", "evolving": " (changing)"}.get(state, "")
    tail_zh = f"置信 {fnum(conf)} · 首次 {first} · 最近 {last} · v{version}.{cver}{st_zh}"
    tail_en = f" Confidence {fnum(conf)}, first seen {first}, last seen {last}, v{version}.{cver}{st_en}."
    return head_zh + body_zh + tail_zh, head_en + body_en + tail_en


def negative_sentence(group: str, system: str, n_days: int, what_zh: str = "写操作",
                      what_en: str = "a write action") -> Tuple[str, str]:
    return (f"{group} 在 {system} 中从未执行{what_zh}（{n_days} 天、0 次）",
            f"{group} has never performed {what_en} on {system} ({n_days} days, 0 times)")


def is_write(route: Any) -> bool:
    return route_parts(route)[0] in WRITE_METHODS
