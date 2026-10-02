"""Communities, stable ids and prefix covers for who-discovery
(docs/lib3/progressive.md §6.15 items 3, 4 and 6, P11).

STATUS: implemented (W-P5, P11 maths). Pure functions; no store access.

    louvain(n, edges, resolution=1)   Louvain modularity optimisation (Blondel,
                                      Guillaume, Lambiotte & Lefebvre 2008) with a
                                      deterministic node order (the caller passes
                                      nodes sorted by IP as integer). Returns the
                                      labels of every level (level 0 = first pass
                                      = the display sub-groups, last = the groups).
    merge_similar(labels, M, rho, adj) Deviation (documented): adjacent
                                      communities whose members are as similar
                                      across as within (MinHash agreement J(A, B) >=
                                      rho x max(J(A, A), J(B, B))) are one group. A
                                      mutual-kNN graph of a homogeneous population is
                                      sparse, and modularity cuts it into pieces
                                      although nothing separates them.
    merge_local(labels, M, src, eph)  Deviation (documented): ephemeral-address
                                      communities (a DHCP / VPN pool) inside one
                                      pure prefix (<= /16) are one group.
    merge_profiles(labels, prof, thr) profile-overlap merge (kept for callers
                                      that hold profiles, not MinHash samples).
    match_ids(prev, new, j_min=0.3)   Hungarian matching on 1 - Jaccard(members)
                                      (scipy linear_sum_assignment); inherit an id at
                                      J >= j_min, else a new id.
    match_names(groups, named, j_min) the same for configured / imported names
                                      (Jaccard >= 0.5).
    prefix_covers(members, active)    greedy smallest CIDR set covering >= 90 % of
                                      the members with purity (members / active IPs
                                      inside) >= 0.8, each prefix the most specific
                                      one with that coverage (IPv4 and IPv6).
"""
from __future__ import annotations

import ipaddress
from typing import Any, Dict, Hashable, Iterable, List, Mapping, Optional, Sequence, Set, Tuple

import numpy as np

COVER = 0.9
COHESION_RATIO = 0.5        # merge_prefix: comparable within-similarity
PURITY = 0.8
MAX_PREFIXES = 8
V4_LENGTHS = tuple(range(32, 7, -1))
V6_LENGTHS = (128, 112, 96, 80, 64, 56, 48, 40, 32)


# ================================================================ Louvain
def _one_level(n: int, adj: List[Dict[int, float]], deg: np.ndarray, m2: float, res: float,
               max_passes: int = 32) -> Tuple[np.ndarray, bool]:
    comm = np.arange(n)
    tot = deg.astype(np.float64).copy()
    moved_any = False
    for _ in range(max_passes):
        moved = 0
        for i in range(n):
            ci = int(comm[i])
            ki = float(deg[i])
            links: Dict[int, float] = {}
            for j, w in adj[i].items():
                if j == i:
                    continue
                cj = int(comm[j])
                links[cj] = links.get(cj, 0.0) + w
            tot[ci] -= ki
            best_c = ci
            best_g = links.get(ci, 0.0) - res * tot[ci] * ki / m2
            for c in sorted(links):
                g = links[c] - res * tot[c] * ki / m2
                if g > best_g + 1e-12:
                    best_g, best_c = g, c
            tot[best_c] += ki
            if best_c != ci:
                comm[i] = best_c
                moved += 1
        if moved == 0:
            break
        moved_any = True
    # relabel 0..k-1 in order of first appearance (deterministic)
    lab: Dict[int, int] = {}
    out = np.empty(n, dtype=np.int64)
    for i in range(n):
        c = int(comm[i])
        if c not in lab:
            lab[c] = len(lab)
        out[i] = lab[c]
    return out, moved_any


def louvain(n: int, edges: Mapping[Tuple[int, int], float], resolution: float = 1.0,
            max_levels: int = 16) -> List[np.ndarray]:
    """Labels of the original nodes after each Louvain level (at least one
    level: the identity labelling when nothing moves). Isolated nodes keep
    their own singleton community."""
    if n == 0:
        return [np.zeros(0, dtype=np.int64)]
    adj: List[Dict[int, float]] = [dict() for _ in range(n)]
    for (a, b), w in edges.items():
        if w <= 0:
            continue
        adj[a][b] = adj[a].get(b, 0.0) + float(w)
        if a != b:
            adj[b][a] = adj[b].get(a, 0.0) + float(w)
    levels: List[np.ndarray] = []
    node_of = np.arange(n)                     # original node -> current super node
    cur_n, cur_adj = n, adj
    for _ in range(max_levels):
        deg = np.asarray([sum(v for v in a.values()) + a.get(i, 0.0) for i, a in enumerate(cur_adj)])
        m2 = float(deg.sum())
        if m2 <= 0:
            if not levels:
                levels.append(node_of.copy())
            break
        lab, moved = _one_level(cur_n, cur_adj, deg, m2, resolution)
        k = int(lab.max()) + 1
        node_of = lab[node_of]
        levels.append(node_of.copy())
        if not moved or k == cur_n:
            break
        new_adj: List[Dict[int, float]] = [dict() for _ in range(k)]
        for i, a in enumerate(cur_adj):
            ci = int(lab[i])
            for j, w in a.items():
                cj = int(lab[j])
                if ci == cj and i != j:
                    new_adj[ci][ci] = new_adj[ci].get(ci, 0.0) + w / 2.0   # each internal edge seen twice
                elif ci == cj:
                    new_adj[ci][ci] = new_adj[ci].get(ci, 0.0) + w
                else:
                    new_adj[ci][cj] = new_adj[ci].get(cj, 0.0) + w
        cur_n, cur_adj = k, new_adj
    return levels


def modularity(labels: np.ndarray, edges: Mapping[Tuple[int, int], float], resolution: float = 1.0) -> float:
    n = labels.size
    deg = np.zeros(n)
    m = 0.0
    inside: Dict[int, float] = {}
    for (a, b), w in edges.items():
        deg[a] += w
        deg[b] += w
        m += w
        if labels[a] == labels[b]:
            inside[int(labels[a])] = inside.get(int(labels[a]), 0.0) + w
    if m <= 0:
        return 0.0
    q = 0.0
    for c in set(labels.tolist()):
        dc = float(deg[labels == c].sum())
        q += inside.get(c, 0.0) / m - resolution * (dc / (2 * m)) ** 2
    return float(q)


def _wj(p: Mapping[Any, float], q: Mapping[Any, float]) -> float:
    keys = set(p) | set(q)
    num = sum(min(p.get(k, 0.0), q.get(k, 0.0)) for k in keys)
    den = sum(max(p.get(k, 0.0), q.get(k, 0.0)) for k in keys)
    return num / den if den > 0 else 0.0


def merge_profiles(labels: np.ndarray, profiles: Mapping[int, Mapping[Any, float]], thr: float,
                   adjacent: Optional[Iterable[Tuple[int, int]]] = None) -> np.ndarray:
    """Merge communities whose normalised profiles have weighted Jaccard >= thr
    (union-find; only adjacent pairs when `adjacent` is given, else all pairs of
    communities). profiles: {community: {item: weight}} (mean member vector)."""
    comms = sorted(profiles)
    norm = {}
    for c in comms:
        s = sum(profiles[c].values())
        norm[c] = {k: v / s for k, v in profiles[c].items()} if s > 0 else {}
    parent = {c: c for c in comms}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    pairs = adjacent if adjacent is not None else [(a, b) for i, a in enumerate(comms) for b in comms[i + 1:]]
    for a, b in pairs:
        if a == b or a not in norm or b not in norm:
            continue
        if _wj(norm[a], norm[b]) >= thr:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    return np.asarray([find(int(c)) if int(c) in parent else int(c) for c in labels], dtype=np.int64)


def _mean_j(M: np.ndarray, a: Sequence[int], b: Optional[Sequence[int]] = None) -> float:
    """Mean MinHash agreement between the rows a and b (within a when b is None)."""
    A = M[list(a)]
    if b is None:
        if len(a) < 2:
            return 1.0
        eq = (A[:, None, :] == A[None, :, :]).mean(axis=2)
        k = len(a)
        return float((eq.sum() - k) / (k * (k - 1)))
    B = M[list(b)]
    return float((A[:, None, :] == B[None, :, :]).mean())


def _sample(ix: Sequence[int], k: int) -> List[int]:
    ix = list(ix)
    if len(ix) <= k:
        return ix
    return [ix[int(j)] for j in np.linspace(0, len(ix) - 1, k).round().astype(int)]


def merge_similar(labels: np.ndarray, M: np.ndarray, rho: float,
                  adjacent: Iterable[Tuple[int, int]], sample: int = 16, passes: int = 8,
                  min_size: int = 2, mult: Optional[np.ndarray] = None
                  ) -> Tuple[np.ndarray, Dict[int, float]]:
    """Merge adjacent communities A, B whose members are as similar across as
    within: J(A, B) >= rho * max(J(A, A), J(B, B)) (MinHash agreement on
    evenly spaced samples of <= `sample` members). Communities smaller than
    min_size are left alone (a singleton has no within-similarity). The tighter
    community sets the bar: a loose community would otherwise absorb anything
    moderately similar. `mult`
    gives the multiplicity of each row (rows that stand for several identical
    signatures). Returns the merged labels and the within-similarity of every
    final community."""
    lab = np.asarray(labels, dtype=np.int64).copy()
    mult = np.ones(lab.size, dtype=np.int64) if mult is None else np.asarray(mult, dtype=np.int64)
    members: Dict[int, List[int]] = {}
    for i, c in enumerate(lab.tolist()):
        members.setdefault(int(c), []).append(i)
    intra = {c: _mean_j(M, _sample(m, sample)) for c, m in members.items()
             if int(mult[m].sum()) >= min_size}
    parent = {c: c for c in members}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    adj = sorted({(min(a, b), max(a, b)) for a, b in adjacent if a != b})
    for _ in range(passes):
        merged = False
        for a0, b0 in adj:
            a, b = find(a0), find(b0)
            if a == b or a not in intra or b not in intra:
                continue
            sa, sb = _sample(members[a], sample), _sample(members[b], sample)
            jab = _mean_j(M, sa, sb)
            if jab >= rho * max(intra[a], intra[b]):
                lo, hi = min(a, b), max(a, b)
                parent[hi] = lo
                members[lo] = members[lo] + members.pop(hi)
                intra[lo] = _mean_j(M, _sample(members[lo], sample))
                intra.pop(hi, None)
                merged = True
        if not merged:
            break
    out = np.asarray([find(int(c)) for c in lab.tolist()], dtype=np.int64)
    return out, {c: v for c, v in intra.items() if find(c) == c}


def merge_local(labels: np.ndarray, M: np.ndarray, sources: Sequence[str], ephemeral: Sequence[bool],
                active: "ActiveIndex", j_min: float = 0.0, sample: int = 16,
                min_size: int = 3, max_prefixes: int = 16, min_len: Tuple[int, int] = (16, 48),
                nets: Optional[Sequence[Any]] = None) -> np.ndarray:
    """Address-locality merge for re-addressed populations (a DHCP / VPN pool,
    §6.21). A pool whose members get a new address every day produces one-day
    signatures that differ only by which of the pool's actions fell on that
    day; neither modularity nor the within / across rule can join them, the
    address plan can: the pure prefixes (purity >= 0.8 over the active
    addresses) covering the union of the EPHEMERAL communities (most members
    active on one day only) are found once, and the ephemeral communities
    inside one such prefix are merged (MinHash agreement >= j_min with the
    growing union; default 0: an address that lives one day shows a random
    slice of its pool's behaviour, so slices need not overlap).

    `nets` (round 3): the pool prefixes when the caller knows them (configured
    DHCP scopes, prefixes whose judged addresses turn over, who_groups
    _pool_nets); the purity-over-active-addresses covers are then not
    computed. Measured on pack O: the 研发 pool's active addresses include its
    young leases (first seen < 2 days ago, not judged ephemeral yet) and its
    low-evidence ones, so 10.50.0.0/22 was never "pure" over the union of the
    ephemeral communities and the pool stayed 5-10 groups (dev_pool_grouped
    0/5)."""
    lab = np.asarray(labels, dtype=np.int64).copy()
    members: Dict[int, List[int]] = {}
    for i, c in enumerate(lab.tolist()):
        members.setdefault(int(c), []).append(i)
    eph = [c for c, m in sorted(members.items())
           if len(m) >= min_size and float(np.mean([bool(ephemeral[i]) for i in m])) >= 0.5]
    if len(eph) < 2:
        return lab
    if nets is None:
        union = [sources[i] for c in eph for i in members[c]]
        covers = prefix_covers(union, active, cover=1.0, max_prefixes=max_prefixes, partial=True,
                               min_len=min_len)
        nets = []
        for cv in covers:
            try:
                nets.append(ipaddress.ip_network(cv))
            except ValueError:
                continue
    else:
        nets = [n if not isinstance(n, str) else ipaddress.ip_network(n, strict=False) for n in nets]
    remap: Dict[int, int] = {}
    for net in nets:
        inside = []
        for c in eph:
            addrs = []
            for i in members[c]:
                try:
                    addrs.append(ipaddress.ip_address(sources[i]))
                except ValueError:
                    continue
            if addrs and sum(1 for a in addrs if a.version == net.version and a in net) >= 0.6 * len(addrs):
                inside.append(c)
        inside = [c for c in inside if c not in remap]
        if len(inside) < 2:
            continue
        inside.sort(key=lambda c: -len(members[c]))
        head = inside[0]
        grown = list(members[head])
        for c in inside[1:]:
            if _mean_j(M, _sample(grown, sample), _sample(members[c], sample)) >= j_min:
                remap[c] = head
                grown += members[c]
    if not remap:
        return lab
    return np.asarray([remap.get(int(c), int(c)) for c in lab.tolist()], dtype=np.int64)


def merge_prefix(labels: np.ndarray, M: np.ndarray, sources: Sequence[str], active: "ActiveIndex",
                 rho: float, sample: int = 16, min_size: int = 2, cover: float = COVER,
                 purity: float = PURITY, min_len: Tuple[int, int] = (16, 48)) -> np.ndarray:
    """Address-plan corroborated merge of static communities (§6.15.2 prefix
    generalisation read back into the grouping): two communities A, B are one
    group when BOTH lines of evidence agree,
      behaviour  J(A, B) >= rho x min(J(A, A), J(B, B))  (as similar across as
                 the looser one is within; merge_similar asks the tighter one),
                 between communities of comparable cohesion (the looser one's
                 within-similarity >= 1/2 of the tighter one's: a loose
                 population in the subnet, a printer fleet, is no sub-team)
      address    one pure prefix (>= /16, purity >= 0.8 over the active
                 addresses) covers >= 90 % of A u B.
    A department whose members split by one optional activity (a sub-team that
    never comments on documents) stays one group whose base communities are
    its sub-groups; two departments in different subnets, or a public
    population, never merge this way. Communities are bucketed by their
    majority /16 (/48), so only pairs that could share a prefix are tested."""
    lab = np.asarray(labels, dtype=np.int64).copy()
    members: Dict[int, List[int]] = {}
    for i, c in enumerate(lab.tolist()):
        members.setdefault(int(c), []).append(i)
    buckets: Dict[Tuple[int, int], List[int]] = {}
    for c, m in members.items():
        if len(m) < min_size:
            continue
        keys: Dict[Tuple[int, int], int] = {}
        for i in m:
            r = _ip_int(sources[i])
            if r is None:
                continue
            ver, x = r
            k = (ver, x >> (16 if ver == 4 else 80))
            keys[k] = keys.get(k, 0) + 1
        if keys:
            k, n = max(keys.items(), key=lambda kv: kv[1])
            if n >= cover * len(m):
                buckets.setdefault(k, []).append(c)
    parent = {c: c for c in members}

    def find(x: int) -> int:
        while parent[x] != x:
            parent[x] = parent[parent[x]]
            x = parent[x]
        return x
    intra: Dict[int, float] = {}

    def within(c: int) -> float:
        v = intra.get(c)
        if v is None:
            v = intra[c] = _mean_j(M, _sample(members[c], sample))
        return v
    for _, cs in sorted(buckets.items()):
        cs = sorted(cs, key=lambda c: (-len(members[c]), c))
        merged = True
        while merged:
            merged = False
            for x in range(len(cs)):
                for y in range(x + 1, len(cs)):
                    a, b = find(cs[x]), find(cs[y])
                    if a == b:
                        continue
                    wa, wb = within(a), within(b)
                    if min(wa, wb) < COHESION_RATIO * max(wa, wb):
                        continue                  # a loose population is not a sub-team
                    jab = _mean_j(M, _sample(members[a], sample), _sample(members[b], sample))
                    if jab < rho * min(wa, wb):
                        continue
                    union = [sources[i] for i in members[a] + members[b]]
                    if not prefix_covers(union, active, cover=cover, purity=purity, max_prefixes=1,
                                         min_len=min_len):
                        continue
                    lo, hi = min(a, b), max(a, b)
                    parent[hi] = lo
                    members[lo] = members[lo] + members.pop(hi)
                    intra.pop(lo, None)
                    intra.pop(hi, None)
                    merged = True
    return np.asarray([find(int(c)) for c in lab.tolist()], dtype=np.int64)


# ============================================================ id matching
def _jac(a: Set[Any], b: Set[Any]) -> float:
    u = len(a | b)
    return len(a & b) / u if u else 0.0


def assignment(cost: np.ndarray) -> List[Tuple[int, int]]:
    """Minimum-cost assignment (Hungarian / Jonker-Volgenant via scipy)."""
    if cost.size == 0:
        return []
    from scipy.optimize import linear_sum_assignment
    r, c = linear_sum_assignment(cost)
    return list(zip(r.tolist(), c.tolist()))


def match_ids(prev: Mapping[str, Set[Any]], new: Sequence[Set[Any]], j_min: float = 0.3
              ) -> Dict[int, Optional[str]]:
    """{index of new group: inherited previous id or None} (§6.15 item 4)."""
    pk = sorted(prev)
    out: Dict[int, Optional[str]] = {i: None for i in range(len(new))}
    if not pk or not new:
        return out
    J = np.asarray([[_jac(set(prev[p]), set(g)) for p in pk] for g in new])
    for i, j in assignment(1.0 - J):
        if J[i, j] >= j_min:
            out[i] = pk[j]
    return out


def match_names(groups: Mapping[str, Set[Any]], named: Sequence[Tuple[str, Set[Any]]],
                j_min: float = 0.5) -> Dict[str, Tuple[str, float]]:
    """{group id: (name, Jaccard)} for configured names (§6.15 item 5)."""
    gk = sorted(groups)
    if not gk or not named:
        return {}
    J = np.asarray([[_jac(set(groups[g]), set(s)) for _, s in named] for g in gk])
    out: Dict[str, Tuple[str, float]] = {}
    for i, j in assignment(1.0 - J):
        if J[i, j] >= j_min:
            out[gk[i]] = (named[j][0], float(J[i, j]))
    return out


# ============================================================ prefix covers
def _ip_int(s: str) -> Optional[Tuple[int, int]]:
    try:
        a = ipaddress.ip_address(str(s))
    except ValueError:
        return None
    return a.version, int(a)


class ActiveIndex:
    """Sorted integer addresses of the active population (per IP version), so
    'active IPs inside a prefix' is two binary searches."""

    def __init__(self, ips: Iterable[str]) -> None:
        v4: List[int] = []
        v6: List[int] = []
        for s in ips:
            r = _ip_int(s)
            if r is None:
                continue
            (v4 if r[0] == 4 else v6).append(r[1])
        self.v4 = np.sort(np.asarray(v4, dtype=np.uint64))
        self.v6 = sorted(v6)

    def count(self, version: int, lo: int, hi: int) -> int:
        if version == 4:
            return int(np.searchsorted(self.v4, np.uint64(hi), "right")
                       - np.searchsorted(self.v4, np.uint64(lo), "left"))
        import bisect
        return bisect.bisect_right(self.v6, hi) - bisect.bisect_left(self.v6, lo)


def prefix_covers(members: Iterable[str], active: ActiveIndex, cover: float = COVER,
                  purity: float = PURITY, max_prefixes: int = MAX_PREFIXES,
                  partial: bool = False, min_len: Tuple[int, int] = (8, 32)) -> List[str]:
    """Greedy smallest set of prefixes covering >= cover of the members, each
    with purity >= purity; among prefixes covering the same members the most
    specific one. [] when no such set of <= max_prefixes prefixes exists
    (partial=True returns the prefixes chosen so far instead)."""
    mem = [r for r in (_ip_int(s) for s in set(members)) if r is not None]
    if not mem:
        return []
    need = cover * len(mem)
    cand: Dict[Tuple[int, int, int], Set[Tuple[int, int]]] = {}
    for ver, x in mem:
        bits = 32 if ver == 4 else 128
        for L in (V4_LENGTHS if ver == 4 else V6_LENGTHS):
            if L < (min_len[0] if ver == 4 else min_len[1]):
                continue
            base = (x >> (bits - L)) << (bits - L)
            cand.setdefault((ver, L, base), set()).add((ver, x))
    ok: List[Tuple[Tuple[int, int, int], Set[Tuple[int, int]]]] = []
    for (ver, L, base), ms in cand.items():
        bits = 32 if ver == 4 else 128
        hi = base + (1 << (bits - L)) - 1
        act = max(active.count(ver, base, hi), len(ms))
        if len(ms) / act >= purity:
            ok.append(((ver, L, base), ms))
    chosen: List[Tuple[int, int, int]] = []
    covered: Set[Tuple[int, int]] = set()
    while len(covered) < need and len(chosen) < max_prefixes:
        best, gain = None, 0
        for key, ms in ok:
            g = len(ms - covered)
            if g > gain or (g == gain and g > 0 and best is not None and key[1] > best[1]):
                best, gain = key, g
        if best is None or gain == 0:
            break
        chosen.append(best)
        covered |= cand[best]
    if len(covered) < need and not partial:
        return []
    out = []
    for ver, L, base in chosen:
        net = ipaddress.ip_network((base, L)) if ver == 4 else ipaddress.IPv6Network((base, L))
        out.append(str(net))
    return out
