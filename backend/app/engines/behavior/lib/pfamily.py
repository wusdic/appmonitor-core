"""System families: servers are not enumerated either (docs/lib3/progressive.md
§6.20, PPC-10; card P12).

STATUS: implemented (W-P6). Pure maths and a small state machine; no store access.

Many servers run one application (a load-balanced cluster, branch servers of
one ERP). They share ONE pattern tree under the tree key 'fam:<id>', and
`net.dst` (which server) becomes an ordinary split attribute: "一类服务器 →
某一台服务器", decided by the same evidence rule as "一类用户 → 某一个用户".

Signature (per system, daily): a weighted feature set over route prefixes
(route template generalised to 'host /a/b/*'), HTTP hosts, SNI eTLD+1,
destination ports, top client stacks and registered attribute names. Each
group is normalised to a fixed share of the total weight (routes 0.45, hosts
0.15, SNI 0.1, ports 0.05, stacks 0.1, names 0.15), so generic features
shared by every system (the attribute names net.* / http.*, port 8080) cannot
by themselves make two systems similar.

    weighted MinHash, k = 64 (improved consistent weighted sampling, Ioffe 2010)
    LSH 16 bands x 4 rows proposes candidate pairs  -> O(#systems . k), no all-pairs scan
    weighted Jaccard J = sum min / sum max, exact on the (bounded) feature sets of a pair
    pair matches on a day   <=> J >= 0.6 and |payload_vis difference| <= 0.2
                                and both signatures are informative (>= 3 route prefixes)
    family                   = connected component of pairs matched on 2 consecutive days
    config system_families   = [{name, systems | cidrs, force: bool}] forces (force=True)
                               or forbids (force=False) membership

A signature with fewer than 3 route prefixes (a health-check endpoint, an idle
system with one '/status' route) says nothing about the application behind it
and never matches: several unrelated idle systems would otherwise look alike.

Family ids are stable: a component keeps the id of the existing family that
holds most of its members; new ids are 'fam:<n>' from a counter. A member
detaches (its subtree is copied into its own tree by the caller, lineage kept)
when its own branch under the family tree's `net.dst` split holds >= 20 % of
the family's nodes on 7 consecutive days (sharing no longer saves anything).
"""
from __future__ import annotations

import hashlib
import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

K_SIG = 64
BANDS, ROWS = 16, 4
J_MIN = 0.6
PAYLOAD_TOL = 0.2
MATCH_DAYS = 2
MATCH_GAP_DAYS = 7               # a match older than this no longer continues a streak
MIN_ROUTES = 3
DETACH_SHARE = 0.2
DETACH_DAYS = 7
REJOIN_DAYS = 30                 # a detached member does not rejoin by similarity for 30 d
GROUP_WEIGHTS = {"route": 0.45, "host": 0.15, "sni": 0.10, "port": 0.05, "stack": 0.10, "name": 0.15}
PREFIX = "fam:"


# ========================================================= feature sets
def route_prefix(route: str, depth: int = 2) -> str:
    """'GET oa /approval/{num}/approve' -> 'oa /approval/{num}/*' (method dropped,
    first `depth` path segments kept): the ℓ2 route level of lib/phier."""
    s = str(route)
    parts = s.split(" ")
    if len(parts) >= 3 and parts[0].isupper():
        host, path = parts[1], " ".join(parts[2:])
    elif len(parts) >= 2:
        host, path = parts[0], " ".join(parts[1:])
    else:
        host, path = "", s
    segs = [x for x in path.split("?")[0].split("/") if x]
    tail = "/*" if len(segs) > depth else ""
    return f"{host} /" + "/".join(segs[:depth]) + tail


def etld1(host: str) -> str:
    h = str(host).strip(".").lower()
    p = h.split(".")
    return ".".join(p[-2:]) if len(p) >= 2 else h


def feature_set(groups: Mapping[str, Mapping[str, float]]) -> Dict[str, float]:
    """Normalise each group {feature: mass} to its share GROUP_WEIGHTS[g] of
    the total and return one flat {'g:feature': weight} set (weights > 0)."""
    out: Dict[str, float] = {}
    for g, items in groups.items():
        gw = GROUP_WEIGHTS.get(g, 0.05)
        tot = float(sum(max(0.0, float(v)) for v in items.values()))
        if tot <= 0:
            continue
        for f, v in items.items():
            w = gw * max(0.0, float(v)) / tot
            if w > 0:
                out[f"{g}:{f}"] = w
    return out


def informative(groups: Mapping[str, Mapping[str, float]]) -> bool:
    return len([k for k, v in (groups.get("route") or {}).items() if v > 0]) >= MIN_ROUTES


def weighted_jaccard(a: Mapping[str, float], b: Mapping[str, float]) -> float:
    keys = set(a) | set(b)
    num = sum(min(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    den = sum(max(a.get(k, 0.0), b.get(k, 0.0)) for k in keys)
    return float(num / den) if den > 0 else 0.0


# ================================================ weighted MinHash (ICWS)
def _feature_randoms(f: str, k: int, seed: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Per-feature ICWS randoms r, c ~ Gamma(2, 1), beta ~ U(0, 1) for k hashes,
    seeded by the feature name (stable across processes)."""
    h = int.from_bytes(hashlib.blake2b(f"{seed}|{f}".encode("utf-8", "surrogatepass"),
                                       digest_size=8).digest(), "big")
    rng = np.random.default_rng(h)
    r = rng.gamma(2.0, 1.0, k)
    c = rng.gamma(2.0, 1.0, k)
    beta = rng.uniform(0.0, 1.0, k)
    return r, c, beta


def weighted_minhash(fs: Mapping[str, float], k: int = K_SIG, seed: int = 0) -> List[Tuple[str, int]]:
    """Ioffe's improved consistent weighted sampling: for each of k hashes the
    sample (feature, t) minimising a_f = c / exp(r (t - beta + 1)) with
    t = floor(ln w / r + beta). P[sample equal] = weighted Jaccard."""
    best_a = np.full(k, np.inf)
    best: List[Tuple[str, int]] = [("", 0)] * k
    for f, w in fs.items():
        if w <= 0:
            continue
        r, c, beta = _feature_randoms(f, k, seed)
        t = np.floor(math.log(w) / r + beta)
        y = np.exp(r * (t - beta))
        a = c / (y * np.exp(r))
        m = a < best_a
        if m.any():
            best_a[m] = a[m]
            for j in np.flatnonzero(m).tolist():
                best[j] = (f, int(t[j]))
    return best


def minhash_similarity(s1: Sequence[Tuple[str, int]], s2: Sequence[Tuple[str, int]]) -> float:
    n = min(len(s1), len(s2))
    if n == 0:
        return 0.0
    return sum(1 for i in range(n) if s1[i] == s2[i] and s1[i][0]) / float(n)


def lsh_buckets(sig: Sequence[Tuple[str, int]], bands: int = BANDS, rows: int = ROWS) -> List[str]:
    out = []
    for b in range(bands):
        part = sig[b * rows:(b + 1) * rows]
        if not part or not all(x[0] for x in part):
            continue
        h = hashlib.blake2b(repr((b, tuple(part))).encode("utf-8", "surrogatepass"), digest_size=8)
        out.append(f"{b}:{h.hexdigest()}")
    return out


def candidate_pairs(sigs: Mapping[str, Sequence[Tuple[str, int]]]) -> Set[Tuple[str, str]]:
    """Pairs of systems sharing at least one LSH bucket (O(#systems . bands))."""
    buckets: Dict[str, List[str]] = {}
    for s, sig in sigs.items():
        for key in lsh_buckets(sig):
            buckets.setdefault(key, []).append(s)
    pairs: Set[Tuple[str, str]] = set()
    for mem in buckets.values():
        if len(mem) < 2 or len(mem) > 256:          # a giant bucket is a generic signature
            continue
        mem = sorted(mem)
        for i in range(len(mem)):
            for j in range(i + 1, len(mem)):
                pairs.add((mem[i], mem[j]))
    return pairs


# ====================================================== family state
def _components(nodes: Iterable[str], edges: Iterable[Tuple[str, str]]) -> List[Set[str]]:
    parent: Dict[str, str] = {n: n for n in nodes}

    def find(x: str) -> str:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x

    for a, b in edges:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    comp: Dict[str, Set[str]] = {}
    for n in parent:
        comp.setdefault(find(n), set()).add(n)
    return list(comp.values())


def _cfg_rules(config_families: Sequence[Mapping[str, Any]]) -> Tuple[List[Tuple[str, Set[str]]], Set[str]]:
    forced: List[Tuple[str, Set[str]]] = []
    forbidden: Set[str] = set()
    for r in config_families or ():
        mem = set(map(str, r.get("systems") or ()))
        if not mem:
            continue
        if r.get("force", True):
            forced.append((str(r.get("name") or sorted(mem)[0]), mem))
        else:
            forbidden |= mem
    return forced, forbidden


def update_families(state: Dict[str, Any], day: int, sigs: Mapping[str, Mapping[str, Any]],
                    config_families: Sequence[Mapping[str, Any]] = (),
                    detach_shares: Optional[Mapping[str, float]] = None) -> Dict[str, Any]:
    """One daily family pass (mutates `state`).

    sigs  {system: {'fs': feature set, 'sig': minhash, 'informative': bool,
                    'payload_vis': float}} for the systems with a fresh
          signature today (idle systems are simply absent: they keep their
          membership and cost nothing).
    detach_shares {system: share of its family's nodes in its own net.dst branch}.
    Returns {'member': {system: 'fam:<id>'}, 'families': {fid: sorted members},
             'joined': [(s, fid)], 'left': [(s, fid, why)], 'pairs': n candidates}."""
    st = state
    st.setdefault("matches", {})          # "a|b" -> last day matched
    st.setdefault("streak", {})           # "a|b" -> consecutive days matched
    st.setdefault("member", {})
    st.setdefault("next_id", 1)
    st.setdefault("detach", {})           # system -> consecutive days over the detach share
    st.setdefault("forbid_pairs", [])
    forced, forbidden = _cfg_rules(config_families)
    informative = {s for s, v in sigs.items() if v.get("informative")}
    pairs = candidate_pairs({s: sigs[s]["sig"] for s in informative})
    streak = st["streak"]
    # pairs already on a streak are judged today whenever both signatures are
    # informative, whether or not LSH proposed them again (an LSH miss is not
    # evidence of dissimilarity)
    judged = set(pairs) | {tuple(k.split("|", 1)) for k in streak}
    judged = {(a, b) for a, b in judged if a in informative and b in informative}
    matched_today: Set[Tuple[str, str]] = set()
    for a, b in sorted(judged):
        if a in forbidden or b in forbidden:
            continue
        va, vb = sigs[a], sigs[b]
        if abs(float(va.get("payload_vis", 0.0)) - float(vb.get("payload_vis", 0.0))) > PAYLOAD_TOL:
            continue
        if weighted_jaccard(va["fs"], vb["fs"]) < J_MIN:
            continue
        matched_today.add((a, b))
    # MATCH_DAYS matches on days on which BOTH were observed with an informative
    # signature, with no contrary day in between: a day on which either side is
    # idle or uninformative neither counts nor resets (round 4, §16.12: a
    # replica of a load-balanced family sees ~1 session a day and is idle or
    # uninformative on many days, so calendar-consecutive matches left 24 MB of
    # unjoined replicas at 300 systems); evidence older than MATCH_GAP_DAYS lapses
    for a, b in matched_today:
        key = f"{a}|{b}"
        last = st["matches"].get(key)
        if last == day:
            streak[key] = int(streak.get(key, 1))
        elif last is not None and day - int(last) <= MATCH_GAP_DAYS and key in streak:
            streak[key] = int(streak.get(key, 0)) + 1
        else:
            streak[key] = 1
        st["matches"][key] = day
    # a pair judged today (both informative) that did not match loses its streak
    for key in list(streak):
        a, b = key.split("|", 1)
        if (a, b) in judged and (a, b) not in matched_today:
            streak.pop(key, None)
            st["matches"].pop(key, None)
    recent = {x for x, d0 in (st.get("detached") or {}).items() if day - int(d0) < REJOIN_DAYS}
    edges = [tuple(k.split("|", 1)) for k, v in streak.items() if v >= MATCH_DAYS]
    edges = [e for e in edges if e[0] not in recent and e[1] not in recent]
    old_member: Dict[str, str] = dict(st["member"])
    nodes = set(old_member) | {x for e in edges for x in e}
    # existing families stay connected (members that were idle today keep their family)
    fam_edges: List[Tuple[str, str]] = []
    by_fam: Dict[str, List[str]] = {}
    for s, f in old_member.items():
        by_fam.setdefault(f, []).append(s)
    for f, mem in by_fam.items():
        mem = sorted(mem)
        fam_edges += [(mem[0], m) for m in mem[1:]]
    comps = _components(nodes, list(edges) + fam_edges)
    new_member: Dict[str, str] = {}
    for comp in comps:
        comp = {x for x in comp if x not in forbidden}
        if len(comp) < 2:
            continue
        olds = [old_member[x] for x in comp if x in old_member]
        if olds:
            fid = max(set(olds), key=lambda f: (olds.count(f), -int(f[len(PREFIX):] or 0)
                                              if f[len(PREFIX):].isdigit() else 0))
        else:
            fid = f"{PREFIX}{st['next_id']}"
            st["next_id"] += 1
        for x in comp:
            new_member[x] = fid
    for name, mem in forced:
        fid = f"{PREFIX}{name}"
        for x in mem:
            new_member[x] = fid
    # detach: a member whose own branch holds >= 20 % of the family's nodes for 7 days
    left: List[Tuple[str, str, str]] = []
    for s, share in (detach_shares or {}).items():
        if s not in new_member or any(s in m for _, m in forced):
            continue
        if float(share) >= DETACH_SHARE:
            st["detach"][s] = int(st["detach"].get(s, 0)) + 1
        else:
            st["detach"].pop(s, None)
        if int(st["detach"].get(s, 0)) >= DETACH_DAYS:
            fid = new_member.pop(s)
            left.append((s, fid, f"branch>={DETACH_SHARE:.0%} of nodes for {DETACH_DAYS}d"))
            st["detach"].pop(s, None)
            for key in list(streak):
                if s in key.split("|"):
                    streak.pop(key, None)
            st.setdefault("detached", {})[s] = day
    # a family reduced to one member dissolves
    cnt: Dict[str, int] = {}
    for f in new_member.values():
        cnt[f] = cnt.get(f, 0) + 1
    for s in [s for s, f in new_member.items() if cnt[f] < 2]:
        left.append((s, new_member.pop(s), "family dissolved"))
    joined = [(s, f) for s, f in sorted(new_member.items()) if old_member.get(s) != f]
    left += [(s, f, "no longer similar") for s, f in sorted(old_member.items())
             if s not in new_member and not any(x[0] == s for x in left)]
    st["member"] = new_member
    fams: Dict[str, List[str]] = {}
    for s, f in new_member.items():
        fams.setdefault(f, []).append(s)
    return {"member": dict(new_member), "families": {f: sorted(m) for f, m in fams.items()},
            "joined": joined, "left": left, "pairs": len(pairs), "matched": len(matched_today)}
