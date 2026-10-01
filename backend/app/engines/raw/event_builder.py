"""P00 EventBuilder (`raw.event`) — observations -> open-attribute behaviour events.

docs/lib3/progressive.md §5.1, §5.2, §6.2.1-§6.2.2, card P00. Library 1 (raw).

Per tick, per system, one lib/pevent.EventBatch (kind txn) written to the
store's batch series `evt.batch`. Every event carries an OPEN attribute map
(no attribute list in code, PPC-4):

  * every non-default Observation field, under its namespace (net.*, http.*,
    tls.*, dns.*): net.src (the resolved client, §5.1.4), net.peer_src (the
    transport source when a trusted proxy forwarded the request), net.dst
    (peer:dport), sizes, durations, http.method / host / path (masked) /
    status / sclass / ua / ctype, tls.sni / ver / ja3, dns.qname / qtype / rcode;
  * http.route: R2's route template, read-only (Templater.apply_path; R2 runs
    first and owns the templater);
  * client.stack (R3's stack token), ev.ch (http | tls | dns | l4);
  * every scalar leaf of extra['meta'] (meta.*), so a new adapter / WAF field
    becomes an attribute with no code change;
  * the payload view extra['l7'] (§5.1.1): body.fmt / len / keys / kv.<key>,
    q.keys / kv.<key>, hdr.<name>, sess.key (HMAC), parsed by lib/pparse;
    a text attribute the registry found to be structured (parse_as) is
    re-parsed into <a>.keys / <a>.kv.<key>.
The value retention policy (§5.1.3) is applied before anything is stored.

Aggregated records: extra['ev_sample'] rows (§5.1.2) expand into one event
each with mass count / len(rows); without them one event with the record means,
flagged approx. extra['sample_rate'] multiplies mass only (PPC-9). A tick with
more than R_tick rows subsamples ev_sample rows per record, keeping >= 1 row
per record, mass-preserving.

Learning sample (§6.2.2): stratified threshold sampling with E_learn =
e_rate x dt learned events per system; strata = (ev.ch, the value of the
attribute the system's txn tree root splits on), bootstrap (ev.ch, route |
sni | qname | dst) before the root has split. Scoring (P03) sees every event.

Inert unless config['progressive']['enabled'].
"""
from __future__ import annotations

import math
from typing import Any, Dict, List, Mapping, Optional, Tuple

import numpy as np

from ...core.engine import Context, Engine
from ...models.schema import AcquisitionMethod, Observation, is_pseudo_entity
from ..behavior.lib import m_ptree as MP
from ..behavior.lib import m_template as MT
from ..behavior.lib import pevent as EV
from ..behavior.lib import pparse as PP
from ..behavior.lib.combine import seeded_uniform
from ..behavior.lib.stack import stack_token
from ..behavior.lib.template import mask_segment, status_class

_ACTIVE = frozenset({AcquisitionMethod.ACTIVE_PROBE, AcquisitionMethod.ACTIVE_DNS,
                     AcquisitionMethod.ACTIVE_TLS})
_ACTIVE_S = frozenset(m.value for m in _ACTIVE)

# Observation field -> attribute name (numeric fields are kept when non-zero,
# strings when non-empty). Fields not listed here and added to the schema
# later are still picked up generically as 'obs.<field>'.
FIELD_ATTR: Dict[str, str] = {
    "l3_proto": "net.l3", "l4_proto": "net.l4", "src_port": "net.sport", "dst_port": "net.dport",
    "bytes_up": "net.bytes_up", "bytes_down": "net.bytes_down", "pkts_up": "net.pkts_up",
    "pkts_down": "net.pkts_down", "ttl": "net.ttl", "tcp_flags": "net.tcp_flags",
    "win_size": "net.win", "retransmits": "net.retrans", "rtt_ms": "net.rtt_ms",
    "duration_ms": "net.dur_ms", "app_proto": "net.app", "http_method": "http.method",
    "http_host": "http.host", "http_status": "http.status", "user_agent": "http.ua",
    "content_type": "http.ctype", "tls_version": "tls.ver", "tls_cipher": "tls.cipher",
    "tls_sni": "tls.sni", "ja3": "tls.ja3", "ja3s": "tls.ja3s", "dns_qname": "dns.qname",
    "dns_qtype": "dns.qtype", "dns_rcode": "dns.rcode", "banner": "net.banner",
}
_SKIP_FIELDS = frozenset({"ts", "system", "entity", "peer", "method", "extra", "http_path",
                          "reachability", "open_ports", "hop_count"})
_EXTRA_KNOWN = frozenset({"count", "ts_sample", "bytes_up_total", "bytes_down_total",
                          "retransmits_total", "l7", "meta", "ev_sample", "sample_rate", "ja4"})


def _weight(ex: Optional[Mapping[str, Any]]) -> float:
    if not ex:
        return 1.0
    c = ex.get("count", 1)
    try:
        v = float(c)
    except (TypeError, ValueError):
        return 1.0
    if not math.isfinite(v):
        return 1.0
    return v if v >= 1.0 else 0.0


def _num(x: Any) -> Optional[float]:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _mask_path(path: str) -> str:
    """Raw path with id / token segments masked (lib/template rules) and the
    query string removed (query values go to q.kv.* under the value policy)."""
    p = str(path).split("?", 1)[0]
    segs = p.split("/")
    return "/".join(mask_segment(s)[0] if s else s for s in segs)


def channel(o: Observation) -> str:
    if o.http_method or o.http_path or o.http_status:
        return "http"
    if o.dns_qname:
        return "dns"
    if o.tls_sni or o.ja3 or (o.app_proto or "").lower() in ("tls", "https", "ssl"):
        return "tls"
    return "l4"


class EventBuilderEngine(Engine):
    name = "raw.event"
    layer = "raw"
    consumes = ["observations", "model.template", "model.attr", "model.sysprof", "model.budget",
                "model.ptree"]
    produces = [EV.EVT_BATCH]
    description = "P00: observations -> open-attribute behaviour events (progressive core)"

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.last_stats: Dict[str, Any] = {}

    # ------------------------------------------------------------ helpers
    def _base(self, o: Observation, count: float, trusted: PP.TrustedNets,
              who_headers: Tuple[str, ...], headers: Optional[Mapping[str, Any]],
              tpl: Any) -> Dict[str, Any]:
        a: Dict[str, Any] = {}
        for f in o.__slots__:
            if f in _SKIP_FIELDS:
                continue
            v = getattr(o, f)
            if v is None or v == "" or v == 0 or v == 0.0:
                continue
            nm = FIELD_ATTR.get(f, "obs." + f)
            if isinstance(v, (int, float)) and not isinstance(v, bool):
                a[nm] = float(v)
            else:
                a[nm] = str(v.value if hasattr(v, "value") else v)
        if count > 1:                                   # aggregated record: per-event means
            ex = o.extra or {}
            for tot_key, nm in (("bytes_up_total", "net.bytes_up"),
                                ("bytes_down_total", "net.bytes_down")):
                if tot_key in ex:
                    x = _num(ex[tot_key])
                    if x is not None:
                        a[nm] = x / count
        src, peer_src = PP.resolve_who(o.entity, headers, trusted, who_headers)
        a["net.src"] = src
        if peer_src is not None:
            a["net.peer_src"] = peer_src
        if o.peer:
            a["net.dst"] = f"{o.peer}:{int(o.dst_port or 0)}"
        ch = channel(o)
        a["ev.ch"] = ch
        if ch == "http":
            if o.http_path:
                a["http.path"] = _mask_path(o.http_path)
            if tpl is not None:
                a["http.route"] = tpl.apply_path(o.http_host or o.peer, o.http_method,
                                                 o.http_path or "/")
            if o.http_status:
                a["http.sclass"] = status_class(o.http_status)
        if o.ja3 or o.user_agent or o.ttl or o.win_size:
            ja4 = (o.extra or {}).get("ja4")
            a["client.stack"] = stack_token(o.ja3, o.user_agent, o.ttl, o.win_size,
                                            ja4 if isinstance(ja4, str) else None)
        ex = o.extra or {}
        meta = ex.get("meta")
        if isinstance(meta, Mapping):
            EV.flatten("meta", meta, a, max_leaves=64)
        for k, v in ex.items():                        # unknown scalar extras are open too
            if k not in _EXTRA_KNOWN and isinstance(v, (str, int, float)) and not isinstance(v, bool):
                a[EV.attr_name("x", k)] = v
        return a

    # ---------------------------------------------------------------- run
    def run(self, ctx: Context, observations: Optional[List[Observation]] = None) -> int:
        if not EV.enabled(ctx.config):
            return 0
        pc = EV.pconfig(ctx.config)
        dflt = pc["defaults"]
        vp = PP.ValuePolicy(pc["value_policy"])
        trusted = PP.TrustedNets(pc.get("trusted_proxies") or ())
        who_headers = tuple(pc.get("client_ip_headers") or ())
        cookies = tuple(pc.get("session_cookies") or ())
        hkey = str(pc["value_policy"].get("hmac_key", ""))
        k_body = int(dflt["k_body"])
        body_cap = int(dflt["body_cap"])
        ev_max = int(dflt["ev_sample_max"])
        r_tick = int(dflt["r_tick"])
        store = ctx.store
        now = float(ctx.now)
        t0 = now - float(ctx.window_s)

        # pass 1: records per system with their row budget
        recs: Dict[str, List[Tuple[Observation, float, List[Any]]]] = {}
        skipped = 0
        total_rows = 0
        for o in observations or ():
            m = o.method
            if m in _ACTIVE or m in _ACTIVE_S:
                skipped += 1
                continue
            if not o.system or not o.entity or is_pseudo_entity(o.entity) or is_pseudo_entity(o.system):
                skipped += 1
                continue
            ex = o.extra or {}
            count = _weight(ex)
            if count <= 0:
                continue
            rows = ex.get("ev_sample")
            rows = list(rows[:ev_max]) if isinstance(rows, (list, tuple)) and rows else [None]
            recs.setdefault(o.system, []).append((o, count, rows))
            total_rows += len(rows)
        keep_frac = 1.0 if total_rows <= r_tick else r_tick / float(total_rows)

        n_out = 0
        stats = {"events": 0, "learned": 0, "approx": 0, "parse_errors": 0, "skipped": skipped,
                 "systems": 0}
        for s, items in recs.items():
            key = MP.tree_key(store, s)
            tpl = MT.templater(store, s)
            reg = MP.get_registry(store, key)
            sysprof = MP.get_model(store, key, MP.SYSPROF) or {}
            chosen = (sysprof.get("chosen") or {}) if isinstance(sysprof, Mapping) else {}
            budget = MP.budget_for(store, key)
            parse_on = not bool(budget.get("skip_body_parsing")) and \
                chosen.get("content", "on") != "off"
            parse_as = {}
            if reg is not None:
                parse_as = {n: r.parse_as for n, r in reg.records.items() if r.parse_as}
            b = EV.BatchBuilder(s, EV.KIND_TXN)
            modes: Dict[str, str] = {}
            for o, count, rows in items:
                ex = o.extra or {}
                l7_rec = ex.get("l7") if isinstance(ex.get("l7"), Mapping) else None
                headers = (l7_rec or {}).get("headers") if l7_rec else None
                if rows[0] is not None and keep_frac < 1.0:
                    k = max(1, int(len(rows) * keep_frac))
                    rows = rows[:k]
                sr = _num(ex.get("sample_rate", 1.0)) or 1.0
                w_row = count * max(sr, 1.0) / len(rows)
                try:
                    base = self._base(o, count, trusted, who_headers, headers, tpl)
                except Exception:
                    stats["parse_errors"] += 1
                    continue
                for r in rows:
                    e = dict(base)
                    ts = float(o.ts)
                    flags = 0
                    l7 = l7_rec
                    if isinstance(r, Mapping):
                        off = _num(r.get("o"))
                        ts = ts + (off or 0.0)
                        for rk, nm in (("up", "net.bytes_up"), ("down", "net.bytes_down"),
                                       ("st", "http.status")):
                            x = _num(r.get(rk))
                            if x is not None:
                                e[nm] = x
                        if "st" in r and _num(r.get("st")):
                            e["http.sclass"] = status_class(int(float(r["st"])))
                        if isinstance(r.get("l7"), Mapping):
                            l7 = r["l7"]
                        # per-event adapter / WAF fields (§5.1.2): an aggregated
                        # record carries them on its ev_sample rows, not on the record
                        if isinstance(r.get("meta"), Mapping):
                            EV.flatten("meta", r["meta"], e, max_leaves=64)
                    elif count > 1:
                        flags |= EV.FLAG_APPROX
                    if l7:
                        try:
                            pa, trunc = PP.parse_l7(l7, cookies, hkey, k_body, body_cap, parse_on)
                            e.update(pa)
                            if trunc:
                                flags |= EV.FLAG_TRUNC
                        except Exception:
                            stats["parse_errors"] += 1
                    for nm, how in parse_as.items():
                        v = e.get(nm)
                        if isinstance(v, str):
                            e.update(PP.parse_structured_text(nm, v, how, k_body))
                    vp.apply_all(e, modes)
                    b.add(ts, str(e.get("net.src", o.entity)), e, w_row, flags)
                    if flags & EV.FLAG_APPROX:
                        stats["approx"] += 1
            if not len(b):
                continue
            b.meta["policy"] = modes
            batch = b.build(t0, now)
            self._learning_sample(store, key, batch, ctx, pc)
            store.add_batch(s, EV.EVT_BATCH, now, batch)
            n_out += batch.n
            stats["events"] += batch.n
            stats["learned"] += int(batch.learn.sum())
            stats["systems"] += 1
        # batches are retained D + 1 ticks (raise-only for slow cadences)
        store.ensure_retention("evt.", max_age_s=EV.learn_delay_s(ctx.window_s, ctx.config)
                               + 2.0 * float(ctx.window_s))
        self.last_stats = stats
        return n_out

    def _learning_sample(self, store: Any, key: str, batch: EV.EventBatch, ctx: Context,
                         pc: Mapping[str, Any]) -> None:
        budget = MP.budget_for(store, key)
        e_rate = float(budget.get("e_rate", pc["defaults"]["e_rate"]))
        e_learn = max(1.0, e_rate * float(ctx.window_s))
        root_attr = None
        m = MP.get_ptree(store, key)
        tree = m.kinds.get(EV.KIND_TXN) if m is not None else None
        hier = None
        if tree is not None:
            sp = tree.nodes[tree.root].split
            if sp is not None:
                root_attr = (sp.attr, sp.level)
                hier = MP.hierarchies(store, key, ctx.config)
        strata = []
        for i in range(batch.n):
            if root_attr is not None:
                strata.append((batch.get("ev.ch", i, ""),
                               repr(hier.gen(root_attr[0], root_attr[1],
                                             batch.get(root_attr[0], i)))))
            else:
                strata.append(EV.bootstrap_stratum(batch, i))
        u = [seeded_uniform(batch.system, batch.t1, int(i)) for i in range(batch.n)]
        EV.select_learning_sample(batch, strata, e_learn, u)
