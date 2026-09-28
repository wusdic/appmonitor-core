"""Read accessors and shared maths for model.link (owner: B17 EntityLinkEngine;
contract C).

Why a module: an IP is not a person. DHCP and VPN re-addressing move a person
to a new address, an IP-hopping actor walks through several, and a NAT puts
several people behind one. B17 decides which addresses are the same actor;
everybody else only has to ASK: learners seed B := B_own + 0.5 A (lib/gating
reads the links directly), B08 / B13 / B26 / B27 merge an actor's evidence,
B16 treats aliases as "self", B28 undoes a retracted link's seed through
rollback_to, B30 narrates continuity. They call these pure functions instead
of reading the layout, so it can evolve in one place. Nothing here mutates a
model or touches the store except get().

model.link@(s, '__system__') layout (JSON-like; stored by reference):
    {'fmt': 1, 'version': int,        # +1 on every link or retraction: learners
                                      # seed on a version increase (contract H)
     'updated': ts,
     'links': [{'id': 'A>B@ts', 'from': A, 'to': B, 'ts': t_link,
                'lo': float, 'conf': sigma(lo - 5), 'margin': float,
                'terms': {'behaviour', 'vocab', 'device', 'topology', 'time'},
                'device_lr': float | None, 'gap_s': float, 'n_ticks': int,
                'status': 'active' | 'retracted', 'retracted': bool,
                'retracted_ts': ts | None, 'rollback_to': t_link | None,
                'reason': str | None}],
     'actors': [{'id': 'actor:<first member>@<first ts>', 'members': [e, ...]
                 (appearance order), 'links': [link id], 'first_ts', 'last_ts'}],
     'shared': {e: {'last_run': ts, 'streak': int, 'bic_delta': [last 3],
                    'hour_overlap': float, 'w_min': float, 'overlap': [ts],
                    'flag': bool, 'since': ts | None, 'neg': int}},
     'pending': {B: B17 private trigger bookkeeping (first_seen, evaluations,
                 candidate log-odds, impersonation pairs)}}
    A retracted link keeps its record (status 'retracted', retracted True,
    rollback_to = t_link) so B28 can write model.control.rollback_to and
    lib/gating never seeds it again ('retracted': True is what
    gating._retracted checks).

Actors: active links chained within ACTOR_CHAIN_S (24 h) of each other through
a shared entity form one actor; every link belongs to exactly one actor (a
single DHCP move is a 2-member actor: it IS one actor).

Scoring maths (used by B17, reusable for portraits / what-if views):
    device LR of B's stack mix under "same device as A" vs "someone from the
        system" (ε = DEVICE_CHANGE_P, the chance a device's stack changes across
        a re-addressing; ρ = NOVEL_STACK_P, the chance a random member shows a
        stack the system has never seen):
            LR(t) = [(1 - ε) p_A(t) + ε f(t)] / f(t),   f(t) = max(f_sys(t), ρ)
            device_llr = Σ_t share_B(t) ln LR(t)   in [ln ε, ln(1/ρ)]
        One device is one draw, so the mix is share-weighted, not count-weighted
        (connections within a tick are not independent).
    topology prior: +2 same /24 (/64) or same DHCP scope, -2 otherwise, 0 when
        either key is not an IP address.
    time prior: -gap / lease (log of the exponential survival; 0 at gap 0).
    conf = sigma(lo - LINK_LO).

Public API:
    MODEL, FMT, LINK_LO, LINK_MARGIN, IMPERSONATION_LR, ACTOR_CHAIN_S,
    DEVICE_CHANGE_P, NOVEL_STACK_P, TOPO_SAME, TOPO_OTHER, DEFAULT_LEASE_S
    get(store, s) -> dict | None;  version(model) -> int;  empty() -> dict
    is_retracted(link) -> bool
    links(model, active_only=True) -> [link]  (copies of the records)
    link_between(model, a, b, active_only=True) -> link | None  (either direction)
    linked_from(model, e) -> A | None;  linked_to(model, e) -> B | None
    seeds_for(model, e) -> [(A, t_link)]   active links INTO e (learner seeding)
    retractions(model, e=None, since=None) -> [link]  retracted links (B28 rollback_to)
    aliases(model, e) -> [e']   transitive over active links, e excluded
    chain(model, e) -> [e, ...] the continuity chain in appearance order
    continuity_id(model, e) -> 'cid:<root>' | None (None when e has no link)
    actors(model) -> [actor];  actor_of(model, e) -> actor | None
    actor_members(model, e) -> [e, ...] (e alone when it has no actor)
    shared_ip(model, e) -> bool;  entity_kind(model, e) -> 'ip' | 'ip-class'
    continuity(model, e) -> {continuity_id, aliases, linked_from, linked_to,
                             entity_kind, shared_ip}   (profile.extra.continuity)
    descriptors(model, e) -> JSON-safe portrait dict (B30)
    build_actors(links, chain_s=ACTOR_CHAIN_S) -> [actor]
    parse_scopes(cfg_scopes) -> [(network, lease_s)]
    scope_of(ip, scopes) -> (network, lease_s) | None
    topology_prior(a, b, scopes) -> float;  lease_for(a, b, scopes) -> seconds
    time_prior(gap_s, lease_s) -> float
    device_llr(shares_a, sys_shares, stacks_b, eps=DEVICE_CHANGE_P, rho=NOVEL_STACK_P)
        -> nats | NaN (NaN: no stacks for B, or no model for A / the system)
    conf(lo) -> sigma(lo - LINK_LO)
"""
from __future__ import annotations

import ipaddress
import math
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

from .classkeys import SYSTEM_KEY

MODEL = "model.link"
FMT = 1
LINK_LO = 5.0                   # Fellegi-Sunter log-odds to link
LINK_MARGIN = 2.0               # over the runner-up (row and column of the assignment)
IMPERSONATION_LR = 0.1          # device LR at or below this: possible_impersonation
ACTOR_CHAIN_S = 86400.0         # links within 24 h chain into one actor
DEVICE_CHANGE_P = 0.05          # eps: stack change across a re-addressing
NOVEL_STACK_P = 0.02            # rho: a random member shows a never-seen stack
TOPO_SAME, TOPO_OTHER = 2.0, -2.0
DEFAULT_LEASE_S = 86400.0
_NAN = math.nan


# ================================================================= basics
def empty() -> Dict[str, Any]:
    return {"fmt": FMT, "version": 0, "updated": None, "links": [], "actors": [],
            "shared": {}, "pending": {}}


def get(store, s: str) -> Optional[Dict[str, Any]]:
    m = store.get_model(s, SYSTEM_KEY, MODEL, default=None)
    return m if isinstance(m, dict) else None


def version(model: Any) -> int:
    if not isinstance(model, Mapping):
        return 0
    try:
        return int(model.get("version") or 0)
    except (TypeError, ValueError):
        return 0


def is_retracted(lk: Mapping) -> bool:
    """The same test lib/gating applies before seeding."""
    return bool(lk.get("retracted")) or lk.get("status") == "retracted" \
        or lk.get("state") == "retracted" or lk.get("active") is False


def _iter(model: Any) -> Iterable[Mapping]:
    if not isinstance(model, Mapping):
        return ()
    lks = model.get("links")
    if isinstance(lks, Mapping):
        lks = list(lks.values())
    return [lk for lk in (lks or ()) if isinstance(lk, Mapping) and lk.get("from")
            and lk.get("to")]


def links(model: Any, active_only: bool = True) -> List[Dict[str, Any]]:
    return [dict(lk) for lk in _iter(model) if not (active_only and is_retracted(lk))]


def link_between(model: Any, a: str, b: str, active_only: bool = True
                 ) -> Optional[Dict[str, Any]]:
    best = None
    for lk in _iter(model):
        if active_only and is_retracted(lk):
            continue
        if {lk["from"], lk["to"]} == {a, b}:
            if best is None or _ts(lk) > _ts(best):
                best = lk
    return dict(best) if best is not None else None


def _ts(lk: Mapping) -> float:
    try:
        v = float(lk.get("ts"))
    except (TypeError, ValueError):
        return _NAN
    return v


def _latest(cands: List[Mapping]) -> Optional[Mapping]:
    return max(cands, key=lambda lk: (_ts(lk) if _ts(lk) == _ts(lk) else -math.inf)) \
        if cands else None


def linked_from(model: Any, e: str) -> Optional[str]:
    lk = _latest([lk for lk in _iter(model) if lk["to"] == e and not is_retracted(lk)])
    return str(lk["from"]) if lk is not None else None


def linked_to(model: Any, e: str) -> Optional[str]:
    lk = _latest([lk for lk in _iter(model) if lk["from"] == e and not is_retracted(lk)])
    return str(lk["to"]) if lk is not None else None


def seeds_for(model: Any, e: str) -> List[Tuple[str, float]]:
    return [(str(lk["from"]), _ts(lk)) for lk in _iter(model)
            if lk["to"] == e and not is_retracted(lk)]


def retractions(model: Any, e: Optional[str] = None, since: Optional[float] = None
                ) -> List[Dict[str, Any]]:
    out = []
    for lk in _iter(model):
        if not is_retracted(lk):
            continue
        if e is not None and e not in (lk["from"], lk["to"]):
            continue
        rt = lk.get("retracted_ts")
        if since is not None and (rt is None or float(rt) < since):
            continue
        out.append(dict(lk))
    return out


def _adjacency(model: Any) -> Dict[str, List[str]]:
    adj: Dict[str, List[str]] = {}
    for lk in _iter(model):
        if is_retracted(lk):
            continue
        a, b = str(lk["from"]), str(lk["to"])
        adj.setdefault(a, []).append(b)
        adj.setdefault(b, []).append(a)
    return adj


def aliases(model: Any, e: str) -> List[str]:
    """Every entity connected to e through active links (transitive), e excluded."""
    adj = _adjacency(model)
    seen, stack = {e}, [e]
    while stack:
        x = stack.pop()
        for y in adj.get(x, ()):
            if y not in seen:
                seen.add(y)
                stack.append(y)
    seen.discard(e)
    return sorted(seen)


def chain(model: Any, e: str) -> List[str]:
    """e's continuity chain in appearance order: the root (no active link
    into it) first, then along the active links by link time."""
    comp = set(aliases(model, e)) | {e}
    if len(comp) == 1:
        return [e]
    act = [lk for lk in _iter(model) if not is_retracted(lk) and lk["from"] in comp]
    act.sort(key=lambda lk: (_ts(lk) if _ts(lk) == _ts(lk) else math.inf, lk["from"], lk["to"]))
    targets = {lk["to"] for lk in act}
    roots = sorted(x for x in comp if x not in targets)
    out: List[str] = list(roots)
    for lk in act:
        for x in (lk["from"], lk["to"]):
            if x not in out:
                out.append(x)
    return out


def continuity_id(model: Any, e: str) -> Optional[str]:
    ch = chain(model, e)
    return f"cid:{ch[0]}" if len(ch) > 1 else None


def build_actors(lks: Sequence[Mapping], chain_s: float = ACTOR_CHAIN_S) -> List[Dict[str, Any]]:
    """Actors from ACTIVE link records: two links that share an entity and lie
    within chain_s of each other belong to the same actor (union-find over
    links). Deterministic: ids from the first member and first link time."""
    act = [lk for lk in lks if not is_retracted(lk) and lk.get("from") and lk.get("to")]
    act.sort(key=lambda lk: (_ts(lk) if _ts(lk) == _ts(lk) else math.inf,
                             str(lk["from"]), str(lk["to"])))
    n = len(act)
    parent = list(range(n))

    def find(i: int) -> int:
        while parent[i] != i:
            parent[i] = parent[parent[i]]
            i = parent[i]
        return i

    by_ent: Dict[str, List[int]] = {}
    for i, lk in enumerate(act):
        for x in (lk["from"], lk["to"]):
            for j in by_ent.get(x, ()):
                ti, tj = _ts(act[i]), _ts(act[j])
                if ti == ti and tj == tj and abs(ti - tj) <= chain_s:
                    parent[find(i)] = find(j)
            by_ent.setdefault(x, []).append(i)
    groups: Dict[int, List[int]] = {}
    for i in range(n):
        groups.setdefault(find(i), []).append(i)
    out = []
    for idx in groups.values():
        idx.sort()
        members: List[str] = []
        for i in idx:
            for x in (str(act[i]["from"]), str(act[i]["to"])):
                if x not in members:
                    members.append(x)
        ts = [_ts(act[i]) for i in idx]
        out.append({"id": f"actor:{members[0]}@{int(ts[0]) if ts[0] == ts[0] else 0}",
                    "members": members,
                    "links": [str(act[i].get("id") or f"{act[i]['from']}>{act[i]['to']}")
                              for i in idx],
                    "first_ts": ts[0], "last_ts": ts[-1]})
    out.sort(key=lambda a: (a["first_ts"], a["id"]))
    return out


def actors(model: Any) -> List[Dict[str, Any]]:
    if not isinstance(model, Mapping):
        return []
    acts = model.get("actors")
    if isinstance(acts, Mapping):
        acts = list(acts.values())
    return [dict(a) for a in (acts or ()) if isinstance(a, Mapping)]


def actor_of(model: Any, e: str) -> Optional[Dict[str, Any]]:
    best = None
    for a in actors(model):
        if e in (a.get("members") or ()):
            if best is None or float(a.get("last_ts") or 0.0) > float(best.get("last_ts") or 0.0):
                best = a
    return best


def actor_members(model: Any, e: str) -> List[str]:
    a = actor_of(model, e)
    return list(a["members"]) if a is not None else [e]


def _shared(model: Any, e: str) -> Mapping:
    sh = model.get("shared") if isinstance(model, Mapping) else None
    rec = sh.get(e) if isinstance(sh, Mapping) else None
    return rec if isinstance(rec, Mapping) else {}


def shared_ip(model: Any, e: str) -> bool:
    return bool(_shared(model, e).get("flag"))


def entity_kind(model: Any, e: str) -> str:
    return "ip-class" if shared_ip(model, e) else "ip"


def continuity(model: Any, e: str) -> Dict[str, Any]:
    """profile.extra.continuity (contract G)."""
    return {"continuity_id": continuity_id(model, e), "aliases": aliases(model, e),
            "linked_from": linked_from(model, e), "linked_to": linked_to(model, e),
            "entity_kind": entity_kind(model, e), "shared_ip": shared_ip(model, e)}


def _j(v: Any, nd: int = 4) -> Any:
    try:
        f = float(v)
    except (TypeError, ValueError):
        return None
    return round(f, nd) if math.isfinite(f) else None


def descriptors(model: Any, e: str) -> Dict[str, Any]:
    """Portrait / profile view of e's continuity (JSON-safe)."""
    c = continuity(model, e)
    lk_in = link_between(model, c["linked_from"], e) if c["linked_from"] else None
    lk_out = link_between(model, e, c["linked_to"]) if c["linked_to"] else None
    sh = _shared(model, e)
    a = actor_of(model, e)

    def brief(lk: Optional[Mapping]) -> Optional[Dict[str, Any]]:
        if lk is None:
            return None
        return {"from": lk["from"], "to": lk["to"], "ts": _j(lk.get("ts"), 1),
                "conf": _j(lk.get("conf")), "lo": _j(lk.get("lo"), 2),
                "gap_s": _j(lk.get("gap_s"), 1)}

    return {**c, "chain": chain(model, e), "link_in": brief(lk_in), "link_out": brief(lk_out),
            "actor": ({"id": a["id"], "members": list(a["members"]),
                       "first_ts": _j(a.get("first_ts"), 1), "last_ts": _j(a.get("last_ts"), 1)}
                      if a else None),
            "retracted": [brief(lk) for lk in retractions(model, e)][-3:],
            "shared": ({"since": _j(sh.get("since"), 1), "hour_overlap": _j(sh.get("hour_overlap")),
                        "bic_delta": [_j(x, 1) for x in (sh.get("bic_delta") or [])]}
                       if sh.get("flag") else None)}


# ============================================================ scoring maths
def parse_scopes(cfg_scopes: Any) -> List[Tuple[Any, float]]:
    """ctx.config['dhcp_scopes']: CIDR strings or {'cidr', 'lease_s'} dicts
    (bad entries are skipped). Returns [(ip_network, lease_s)]."""
    out: List[Tuple[Any, float]] = []
    for c in cfg_scopes or ():
        cidr, lease = c, DEFAULT_LEASE_S
        if isinstance(c, Mapping):
            cidr = c.get("cidr")
            try:
                lease = float(c.get("lease_s") or c.get("lease") or DEFAULT_LEASE_S)
            except (TypeError, ValueError):
                lease = DEFAULT_LEASE_S
        try:
            net = ipaddress.ip_network(str(cidr), strict=False)
        except (ValueError, TypeError):
            continue
        out.append((net, lease if lease > 0 and math.isfinite(lease) else DEFAULT_LEASE_S))
    return out


def _ip(e: str) -> Optional[Any]:
    try:
        return ipaddress.ip_address(str(e))
    except ValueError:
        return None


def scope_of(e: str, scopes: Sequence[Tuple[Any, float]]) -> Optional[Tuple[Any, float]]:
    ip = _ip(e)
    if ip is None:
        return None
    best = None
    for net, lease in scopes:
        if net.version == ip.version and ip in net:
            if best is None or net.prefixlen > best[0].prefixlen:   # most specific
                best = (net, lease)
    return best


def _block(ip: Any) -> Any:
    return ipaddress.ip_network(f"{ip}/{24 if ip.version == 4 else 64}", strict=False)


def topology_prior(a: str, b: str, scopes: Sequence[Tuple[Any, float]] = ()) -> float:
    ia, ib = _ip(a), _ip(b)
    if ia is None or ib is None:
        return 0.0
    if ia.version == ib.version and _block(ia) == _block(ib):
        return TOPO_SAME
    sa, sb = scope_of(a, scopes), scope_of(b, scopes)
    if sa is not None and sb is not None and sa[0] == sb[0]:
        return TOPO_SAME
    return TOPO_OTHER


def lease_for(a: str, b: str, scopes: Sequence[Tuple[Any, float]] = ()) -> float:
    sc = scope_of(b, scopes) or scope_of(a, scopes)
    return float(sc[1]) if sc is not None else DEFAULT_LEASE_S


def time_prior(gap_s: float, lease_s: float = DEFAULT_LEASE_S) -> float:
    g = float(gap_s)
    if not math.isfinite(g):
        return _NAN
    return -max(0.0, g) / max(float(lease_s), 1.0)


def device_llr(shares_a: Mapping[str, float], sys_shares: Mapping[str, float],
               stacks_b: Mapping[str, float], eps: float = DEVICE_CHANGE_P,
               rho: float = NOVEL_STACK_P) -> float:
    """Share-weighted ln LR of B's stack mix (module doc). NaN without B
    stacks or without A's stack model (no evidence, never a guess)."""
    tot = sum(float(n) for n in stacks_b.values() if n > 0 and math.isfinite(n))
    if tot <= 0.0 or not shares_a:
        return _NAN
    out = 0.0
    for t, n in stacks_b.items():
        if not (n > 0 and math.isfinite(n)):
            continue
        f = max(float(sys_shares.get(t, 0.0)), rho)
        p = float(shares_a.get(t, 0.0))
        lr = ((1.0 - eps) * p + eps * f) / f
        out += (n / tot) * math.log(max(lr, 1e-300))
    return out


def conf(lo: float) -> float:
    x = float(lo) - LINK_LO
    if not math.isfinite(x):
        return _NAN
    return 1.0 / (1.0 + math.exp(-x)) if x >= 0 else math.exp(x) / (1.0 + math.exp(x))


# ================================================== actor evidence (round 4)
# Fellegi-Sunter agreement fields for "same actor" beyond the modality LLRs,
# for B17's pairs (A silent predecessor, B new entity):
#   'tok'   B shares a RARE categorical value with A (dim=value used by at most
#           FS_RARE_ENT other entities of the system: an IP-hopping actor's
#           private destination / template follows it from address to address);
#   'hand'  temporal handoff: A's last activity within FS_HANDOFF_S before B's
#           first (the actor moves; a DHCP renewal or a hop is immediate);
#   'stk'   B's dominant client stack is one A was seen with.
# The m / u probabilities are learned by EM (conditional independence, latent
# match status) over every pair B17 has compared in the system, with Beta
# priors (FS_PRIOR) so that a system with a handful of pairs keeps sane
# weights; weights are capped at +-FS_CAP nats like every modality.
FS_FIELDS = ("tok", "hand", "stk")
FS_HANDOFF_S = 2 * 3600.0
FS_RARE_ENT = 1.5
FS_CAP = 4.0
FS_KEEP = 512
# prior (mean, strength) of m (agreement among matches) and u (among non-matches)
FS_PRIOR = {"tok": ((0.8, 10.0), (0.02, 10.0)), "hand": ((0.8, 10.0), (0.1, 10.0)),
            "stk": ((0.9, 10.0), (0.5, 10.0))}
FS_PI_PRIOR = (0.05, 20.0)


def fs_em(obs: Sequence[Sequence[Any]], iters: int = 50) -> Dict[str, Any]:
    """EM for the two-class Fellegi-Sunter model on binary agreement vectors
    (rows: one value per FS_FIELDS, 1 / 0 / None = missing, skipped). MAP
    estimates under Beta priors. Returns {'m': {f: .}, 'u': {f: .}, 'pi': ., 'n': .}."""
    X = [[(None if v is None else float(v)) for v in row] for row in obs
         if len(row) == len(FS_FIELDS)]
    m = {f: FS_PRIOR[f][0][0] for f in FS_FIELDS}
    u = {f: FS_PRIOR[f][1][0] for f in FS_FIELDS}
    pi = FS_PI_PRIOR[0]
    for _ in range(iters if X else 0):
        g = []
        for row in X:
            lm, lu = math.log(pi), math.log(1.0 - pi)
            for f, v in zip(FS_FIELDS, row):
                if v is None:
                    continue
                lm += math.log(m[f] if v >= 0.5 else 1.0 - m[f])
                lu += math.log(u[f] if v >= 0.5 else 1.0 - u[f])
            mx = max(lm, lu)
            g.append(math.exp(lm - mx) / (math.exp(lm - mx) + math.exp(lu - mx)))
        a0, s0 = FS_PI_PRIOR
        pi = (sum(g) + a0 * s0) / (len(g) + s0)
        for k, f in enumerate(FS_FIELDS):
            (mm, ms), (um, us) = FS_PRIOR[f]
            nm = am = nu = au = 0.0
            for gi, row in zip(g, X):
                v = row[k]
                if v is None:
                    continue
                nm += gi
                am += gi * v
                nu += 1.0 - gi
                au += (1.0 - gi) * v
            m[f] = min(0.999, max(0.001, (am + mm * ms) / (nm + ms)))
            u[f] = min(0.999, max(0.001, (au + um * us) / (nu + us)))
    return {"m": m, "u": u, "pi": pi, "n": len(X)}


def fs_weights(params: Optional[Mapping[str, Any]]) -> Dict[str, Tuple[float, float]]:
    """{field: (agree weight, disagree weight)} in nats, capped at +-FS_CAP."""
    p = params if isinstance(params, Mapping) and params.get("m") else fs_em([], iters=0)
    out = {}
    for f in FS_FIELDS:
        m, u = float(p["m"][f]), float(p["u"][f])
        wa = max(-FS_CAP, min(FS_CAP, math.log(m / u)))
        wd = max(-FS_CAP, min(FS_CAP, math.log((1.0 - m) / (1.0 - u))))
        out[f] = (wa, wd)
    return out


def fs_score(gamma: Mapping[str, Any], weights: Mapping[str, Tuple[float, float]],
             params: Optional[Mapping[str, Any]] = None) -> float:
    """Sum of the agreement weights of the observed fields (None = missing: 0).
    An agreement carrying its own frequency-based u (gamma['u_<field>'], the
    chance that a random non-matching pair shares that very value: Winkler's
    frequency-based FS weights) is weighted ln(m / u) with m from `params`
    (the EM fit), capped at +-FS_CAP; otherwise the field's EM weight."""
    p = params if isinstance(params, Mapping) and params.get("m") else fs_em([], iters=0)
    tot = 0.0
    for f in FS_FIELDS:
        v = gamma.get(f)
        if v is None:
            continue
        wa, wd = weights[f]
        u = gamma.get(f"u_{f}")
        if v and isinstance(u, (int, float)) and u > 0.0:
            wa = max(-FS_CAP, min(FS_CAP, math.log(float(p["m"][f]) / float(u))))
        tot += wa if v else wd
    return tot
