"""Read accessors and shared maths for model.vocab (owner: B08 NoveltyEngine; contract C, L).

Why a module: the categorical vocabulary of an IP (which templates, SNIs, DNS
domains, ports, peers, content types and lib-4 categories it touches, and how
often) is evidence for far more than novelty. B02 types roles from its
template-family mix, B12 whitelists destinations by prevalence, B13 finds
novel exfil destinations by first-seen age, B15/B16/B17 score a window of
values under each candidate's vocabulary, B18 scores class adoptions and B30
narrates it. They must all use the SAME hierarchical smoothing B08 scores
with, so the maths lives here once, as pure functions over the model dicts.
Consumers never mutate a model.

Dimensions (DIMS; the `dim` of every value, also the allowlist / event dim):
    tmpl   HTTP method + host + path template  (template_key of act.tokens HTTP tokens,
           e.g. 'GET erp.corp /orders/view/{num}')
    ctype  HTTP content type (http.content_types, parameters stripped, lower case)
    sni    TLS SNI at eTLD+1 (tls.sni_etld1_set)
    dns    DNS query domain at eTLD+1 (dns.qname_etld1_set)
    qtype  DNS query type (dns.qtype_set)
    dport  destination port (l4.dport_set, as a string)
    peer   peer /24 (/64, or service host) (l4.peer_set)
    cat    lib-4 signature category (store.matches, one tick late)

Layouts (stored by reference with put_model; objects are live):
  entity  model.vocab@(s, ip):
    {'fmt': 1, 'kind': 'entity', 'version': int (gate version), 'ts': float,
     'class_key': 'class:<rid>' | None,
     'state': {                                  # the gated learner state
         'H': half-life s (30 d), 't_ref': float, 'g': float,
         'dims': {dim: {value: [c, n_acc, first_ts, last_ts]}},
         'N':  {dim: sum c},  'N1': {dim: sum c over values with n_acc == 1},
         'clock': newest committed row ts, 'first': oldest committed row ts,
         'days': {utc_day: class-rare novelties}, 'jsd': {'h1': Ring, 'h24': Ring},
         'n_rows': int},
     'gate' / 'run' / 'rows' / 'held_rows' / 'n_entries': B08 private}
    Entity counts are lazily decayed: c and N are stored at scale
    2^((t_clock - t_ref) / H); the count at time `now` is c * 2^(-(now - t_ref)/H).
    n_acc is the undecayed number of committed accesses (Good-Turing singletons
    are n_acc == 1). Space-Saving cap ENTITY_CAP values per dim.
  class   model.vocab@(s, 'class:<rid>')  and  system  model.vocab@(s, '__system__'):
    {'fmt': 1, 'kind': 'class' | 'system', 'version': int, 'built': ts, 'H': s,
     'dims': {dim: {value: [c, df, first_ts, last_ts, n_ent, n_acc]}},
     'N': {dim: total c}, 'N1': {dim: singleton mass}, 'n_ent': decayed member count,
     'members': int (members with a model at the build),
     class only:  'adoption': {'dim=value': record}  (see adoption_records)
     system only: 'sightings': {'dim=value': [first_ts, entity]}  (values first seen
                  by some entity but not yet in the rebuilt system counts)}
    Rebuilt hourly by B08 from the member entity models (so the tiers inherit
    the members' trust gating and rollbacks): c = sum of member counts decayed
    to 'built'; df = sum over members of 2^(-(built - member last_ts)/H) (the
    decayed document frequency); first/last = min/max over members; n_ent =
    members holding the value. Space-Saving cap TIER_CAP values per dim (the
    dropped mass stays in N).

Hierarchical Dirichlet backoff (DIRICHLET_BACKOFF = 5, U = 1 / universe(dim)):
    p_s(v) = (c_s + U) / (N_s + 1)
    p_c(v) = (c_c + 5 p_s) / (N_c + 5)      (class tier skipped when absent)
    p_e(v) = (c_e + 5 p_c) / (N_e + 5)      (entity tier skipped when absent)
    surprisal I_v = -log2 p_e(v).  Each tier is exactly normalised over the
    dimension's universe of UNIVERSE[dim] (or 2x the system's distinct values
    if larger) values.
Class rarity: IDF = ln((N_class + 1)/(df + 1)) + 1 with N_class the decayed
member count (idf()).

Accessor signatures (pure reads; missing data gives the documented default):
    DIMS, MODEL, HALF_LIFE_S, UNIVERSE, BACKOFF, ENTITY_CAP, TIER_CAP
    get(store, s, key) -> dict | None                 key: ip | 'class:<rid>' | '__system__'
    kind(model) -> 'entity' | 'class' | 'system' | None
    value_key(dim, value) -> 'dim=value';  split_key(key) -> (dim, value)
    factor(model, now=None) -> float                  stored -> true count multiplier
    counts(model, dim, now=None) -> {value: c}        decayed to now (entity) / to 'built'
    total(model, dim, now=None) -> float
    entry(model, dim, value, now=None) -> dict | None {count, n_acc, first_ts, last_ts[, df, n_ent]}
    universe(dim, sys_model=None) -> int
    Backoff(ent, cls, sys, now=None)   .p(dim, v) -> p_e;  .tiers(dim, v) -> (p_e, p_c, p_s)
                                       .bits(dim, v) -> -log2 p_e
                                       .probs(dim, values) -> [p_e]
    backoff_models(store, s, e) -> (ent, cls, sys)    class via m_class.class_key
    prob(store, s, e, dim, value, now=None) -> p_e     (e may be a class key / '__system__')
    surprisal(store, s, e, dim, value, now=None) -> bits
    loglik(store, s, e, obs, now=None) -> nats         obs {dim: {value: n}} | [(dim, value, n)]
    loglik_models(ent, cls, sys, obs, now=None) -> nats
    llr_vs_system(store, s, e, obs, now=None) -> nats  loglik under e minus under the system tier
    idf(model, dim, value) -> float                    tier model; NaN without one
    prevalence(store, s, dim, value, now=None) -> share of the system's entities (NaN: no model)
    prevalence_n(store, s, dim, value) -> decayed number of entities using value
    dest_prevalence(store, s, name, now=None) -> max share over sni / dns / peer
    first_seen_ts(store, s, e, dim, value) -> ts | NaN
    first_seen_age(store, s, e, dim, value, now) -> s; 0.0 when never seen (new now,
        including values only sighted, not yet committed); NaN without a model
    novelty_rate(model, dim=None) -> Good-Turing missing mass N1/N (1.0 when empty)
    maturity(model, dim=None) -> 1 - novelty_rate
    top_values(model, dim, k=10, now=None) -> [(value, share)]
    family_distribution(model_or_store, s=None, e=None, now=None) -> {family: share}
        family = 'channel|method class|host|first path segment' (lib/template.token_family)
        for tmpl; 'tls|read|<sni>|', 'dns|read|<domain>|', 'l4|other|<dport>|' otherwise
    adoption_records(store, s, class_key, since=None, min_members=1) -> [record]
        record: {'dim', 'value', 'members': {ip: first_ts}, 'first_ts', 'last_ts',
                 'flags': {'external', 'upload', 'sensitive', 'new_eTLD1_org'},
                 'adopted': bool, 'adopted_ts', 'n_class', 'tier', 'n_recent'}
        (n_recent = members whose first_ts >= since, or all members)
    descriptors(model, k=8, now=None) -> dict          portrait / profile descriptor (B30)
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple, Union

from . import m_class
from .classkeys import SYSTEM_KEY, is_class
from .template import token_family

MODEL = "model.vocab"
DIMS: Tuple[str, ...] = ("tmpl", "ctype", "sni", "dns", "qtype", "dport", "peer", "cat")
HALF_LIFE_S = 30 * 86400.0
BACKOFF = 5.0                       # priors.DIRICHLET_BACKOFF
ENTITY_CAP = 256
TIER_CAP = 2048
# Size of each dimension's value universe: the base measure U = 1/V of the
# system tier. Templates are capped at 4000 per system by R2; ports and /24s
# are 16-bit-ish spaces; names are open, 65536 stands for "many".
UNIVERSE: Dict[str, int] = {"tmpl": 4096, "ctype": 1024, "sni": 65536, "dns": 65536,
                            "qtype": 256, "dport": 65536, "peer": 65536, "cat": 256}
DEST_DIMS = ("sni", "dns", "peer")
_LN2 = math.log(2.0)

Obs = Union[Mapping[str, Mapping[str, float]], Iterable[Tuple[str, str, float]]]


# ------------------------------------------------------------------ basics
def get(store, system: str, key: str) -> Optional[Dict[str, Any]]:
    m = store.get_model(system, key, MODEL, default=None)
    return m if isinstance(m, dict) else None


def kind(model: Optional[Mapping]) -> Optional[str]:
    return model.get("kind") if isinstance(model, dict) else None


def value_key(dim: str, value: Any) -> str:
    return f"{dim}={value}"


def split_key(key: str) -> Tuple[str, str]:
    d, _, v = str(key).partition("=")
    return d, v


def _state(model: Optional[Mapping]) -> Optional[Mapping]:
    """The count-holding dict: model['state'] for entities, the model itself for tiers."""
    if not isinstance(model, dict):          # models are plain dicts (fast check)
        return None
    if model.get("kind") == "entity":
        st = model.get("state")
        return st if isinstance(st, dict) else None
    return model if "dims" in model else None


def factor(model: Optional[Mapping], now: Optional[float] = None) -> float:
    """Multiplier from stored counts to true counts (entity: decayed to now or to
    its clock; tiers: counts are already true at 'built')."""
    st = _state(model)
    if st is None or model.get("kind") != "entity":
        return 1.0
    t_ref = float(st.get("t_ref", math.nan))
    if not math.isfinite(t_ref):
        return 1.0
    if now is None or not math.isfinite(float(now)):
        return 2.0 ** (-float(st.get("g", 0.0)))
    return 2.0 ** (-(float(now) - t_ref) / float(st.get("H", HALF_LIFE_S)))


def _dim(model: Optional[Mapping], dim: str) -> Mapping[str, list]:
    st = _state(model)
    if st is None:
        return {}
    return st.get("dims", {}).get(dim) or {}


def counts(model: Optional[Mapping], dim: str, now: Optional[float] = None) -> Dict[str, float]:
    f = factor(model, now)
    return {v: float(x[0]) * f for v, x in _dim(model, dim).items()}


def total(model: Optional[Mapping], dim: str, now: Optional[float] = None) -> float:
    st = _state(model)
    if st is None:
        return 0.0
    return float((st.get("N") or {}).get(dim, 0.0)) * factor(model, now)


def entry(model: Optional[Mapping], dim: str, value: Any,
          now: Optional[float] = None) -> Optional[Dict[str, float]]:
    x = _dim(model, dim).get(value)
    if x is None:
        return None
    out = {"count": float(x[0]) * factor(model, now)}
    if model.get("kind") == "entity":
        out.update(n_acc=float(x[1]), first_ts=float(x[2]), last_ts=float(x[3]))
    else:
        out.update(df=float(x[1]), first_ts=float(x[2]), last_ts=float(x[3]),
                   n_ent=float(x[4]), n_acc=float(x[5]) if len(x) > 5 else math.nan)
    return out


def universe(dim: str, sys_model: Optional[Mapping] = None) -> int:
    v = UNIVERSE.get(dim, 4096)
    if sys_model is not None:
        v = max(v, 2 * len(_dim(sys_model, dim)))
    return v


# ---------------------------------------------------------------- backoff
class Backoff:
    """Hierarchical Dirichlet predictive entity -> class -> system for one
    (entity, class, system) triple at time `now`. Per-dimension tables and
    totals are resolved once, so scoring many values costs a few dict lookups
    each (B08 and B16 score every value of a tick through one instance)."""

    __slots__ = ("ent", "cls", "sys", "now", "_cache", "k")

    def __init__(self, ent: Optional[Mapping], cls: Optional[Mapping], sys: Optional[Mapping],
                 now: Optional[float] = None, k: float = BACKOFF) -> None:
        self.ent = ent if _state(ent) is not None else None
        self.cls = cls if _state(cls) is not None else None
        self.sys = sys if _state(sys) is not None else None
        self.now = now
        self.k = float(k)
        self._cache: Dict[str, tuple] = {}

    def _tables(self, dim: str) -> tuple:
        t = self._cache.get(dim)
        if t is None:
            U = 1.0 / universe(dim, self.sys)
            parts = []
            for m in (self.ent, self.cls, self.sys):
                if m is None:
                    parts.append(None)
                else:
                    st = _state(m)
                    f = factor(m, self.now)
                    parts.append(((st.get("dims") or {}).get(dim) or {}, f,
                                  float((st.get("N") or {}).get(dim, 0.0)) * f))
            t = self._cache[dim] = (U, parts[0], parts[1], parts[2])
        return t

    def tiers(self, dim: str, value: Any) -> Tuple[float, float, float]:
        U, e, c, s = self._tables(dim)
        k = self.k
        if s is None:
            p_s = U
        else:
            x = s[0].get(value)
            p_s = ((x[0] * s[1] if x is not None else 0.0) + U) / (s[2] + 1.0)
        if c is None:
            p_c = p_s
        else:
            x = c[0].get(value)
            p_c = ((x[0] * c[1] if x is not None else 0.0) + k * p_s) / (c[2] + k)
        if e is None:
            p_e = p_c
        else:
            x = e[0].get(value)
            p_e = ((x[0] * e[1] if x is not None else 0.0) + k * p_c) / (e[2] + k)
        return p_e, p_c, p_s

    def p(self, dim: str, value: Any) -> float:
        return self.tiers(dim, value)[0]

    def probs(self, dim: str, values: Iterable[Any]) -> List[float]:
        """p_e of many values of one dimension (the table lookups hoisted)."""
        U, e, c, s = self._tables(dim)
        k = self.k
        out = []
        for v in values:
            if s is None:
                p = U
            else:
                x = s[0].get(v)
                p = ((x[0] * s[1] if x is not None else 0.0) + U) / (s[2] + 1.0)
            if c is not None:
                x = c[0].get(v)
                p = ((x[0] * c[1] if x is not None else 0.0) + k * p) / (c[2] + k)
            if e is not None:
                x = e[0].get(v)
                p = ((x[0] * e[1] if x is not None else 0.0) + k * p) / (e[2] + k)
            out.append(p)
        return out

    def bits(self, dim: str, value: Any) -> float:
        return -math.log2(max(self.tiers(dim, value)[0], 1e-300))


def backoff_models(store, system: str, entity: str
                   ) -> Tuple[Optional[Dict], Optional[Dict], Optional[Dict]]:
    """(entity model, class model, system model) for scoring under `entity`.
    A class key scores under (class, system), '__system__' under the system."""
    sys = get(store, system, SYSTEM_KEY)
    if entity == SYSTEM_KEY:
        return None, None, sys
    if is_class(entity):
        return None, get(store, system, entity), sys
    ck = m_class.class_key(store, system, entity)
    return get(store, system, entity), (get(store, system, ck) if ck else None), sys


def prob(store, system: str, entity: str, dim: str, value: Any,
         now: Optional[float] = None) -> float:
    return Backoff(*backoff_models(store, system, entity), now=now).p(dim, value)


def surprisal(store, system: str, entity: str, dim: str, value: Any,
              now: Optional[float] = None) -> float:
    return Backoff(*backoff_models(store, system, entity), now=now).bits(dim, value)


def _iter_obs(obs: Obs) -> Iterable[Tuple[str, Any, float]]:
    if isinstance(obs, Mapping):
        for dim, vals in obs.items():
            for v, n in (vals or {}).items():
                yield dim, v, n
    else:
        for dim, v, n in obs:
            yield dim, v, n


def loglik_models(ent: Optional[Mapping], cls: Optional[Mapping], sys: Optional[Mapping],
                  obs: Obs, now: Optional[float] = None) -> float:
    """Multinomial log-likelihood (nats) of the counts under the backoff chain.
    Non-finite or non-positive counts are skipped; no observation gives 0.0."""
    b = Backoff(ent, cls, sys, now=now)
    ll = 0.0
    for dim, v, n in _iter_obs(obs):
        try:
            n = float(n)
        except (TypeError, ValueError):
            continue
        if n > 0.0 and math.isfinite(n):
            ll += n * math.log(max(b.p(dim, v), 1e-300))
    return ll


def loglik(store, system: str, entity: str, obs: Obs, now: Optional[float] = None) -> float:
    return loglik_models(*backoff_models(store, system, entity), obs, now=now)


def llr_vs_system(store, system: str, entity: str, obs: Obs,
                  now: Optional[float] = None) -> float:
    """Multinomial naive-Bayes LLR of the counts: under the entity's chain minus
    under the system tier alone (B17's 'vocab MNB LLR against the system')."""
    ent, cls, sys = backoff_models(store, system, entity)
    return loglik_models(ent, cls, sys, obs, now) - loglik_models(None, None, sys, obs, now)


# ------------------------------------------------------------ tier queries
def idf(model: Optional[Mapping], dim: str, value: Any, extra_df: float = 0.0) -> float:
    """ln((N_ent + 1)/(df + 1)) + 1 on a class / system model; NaN without one."""
    if not isinstance(model, dict) or model.get("kind") not in ("class", "system"):
        return math.nan
    x = _dim(model, dim).get(value)
    df = (float(x[1]) if x is not None else 0.0) + max(0.0, float(extra_df))
    n = float(model.get("n_ent", 0.0))
    return math.log((n + 1.0) / (df + 1.0)) + 1.0


def prevalence_n(store, system: str, dim: str, value: Any) -> float:
    """Decayed number of the system's entities using `value` (0.0 unknown value,
    NaN without a system model)."""
    sm = get(store, system, SYSTEM_KEY)
    if sm is None or "dims" not in sm:
        return math.nan
    x = _dim(sm, dim).get(value)
    return float(x[1]) if x is not None else 0.0


def prevalence(store, system: str, dim: str, value: Any, now: Optional[float] = None) -> float:
    """Share of the system's (decayed) entities using `value`, in [0, 1];
    NaN without a system model or with no entity mass. `now` is accepted for
    signature symmetry (df and the entity count decay alike)."""
    sm = get(store, system, SYSTEM_KEY)
    if sm is None or "dims" not in sm:
        return math.nan
    n = float(sm.get("n_ent", 0.0))
    if not n > 0.0:
        return math.nan
    x = _dim(sm, dim).get(value)
    return min(1.0, (float(x[1]) if x is not None else 0.0) / n)


def dest_prevalence(store, system: str, name: str, now: Optional[float] = None) -> float:
    """Prevalence of a destination name (eTLD+1 or '/24', m_template.dest_name_of)
    over the sni / dns / peer dimensions (the largest share); NaN without a model."""
    vals = [prevalence(store, system, d, name, now) for d in DEST_DIMS]
    vals = [v for v in vals if v == v]
    return max(vals) if vals else math.nan


def first_seen_ts(store, system: str, entity: str, dim: str, value: Any) -> float:
    """Tier-level first-seen ts of a value (entity: committed or sighted), NaN
    when the model is absent or the value was never seen."""
    m = get(store, system, entity)
    if m is None:
        return math.nan
    x = _dim(m, dim).get(value)
    best = float(x[2]) if x is not None else math.inf
    k = value_key(dim, value)
    if m.get("kind") == "entity":
        seen = (m.get("run") or {}).get("seen") or {}
        if k in seen:
            best = min(best, float(seen[k]))
    elif m.get("kind") == "system":
        sg = (m.get("sightings") or {}).get(k)
        if sg:
            best = min(best, float(sg[0]))
    return best if math.isfinite(best) else math.nan


def first_seen_age(store, system: str, entity: str, dim: str, value: Any, now: float) -> float:
    """Seconds since the tier (entity / class / '__system__') first saw `value`;
    0.0 when never seen (it is new now); NaN when the model does not exist."""
    m = get(store, system, entity)
    if m is None:
        return math.nan
    t = first_seen_ts(store, system, entity, dim, value)
    if t != t:
        return 0.0
    return max(0.0, float(now) - t)


def _n1_n(model: Optional[Mapping], dim: Optional[str]) -> Tuple[float, float]:
    st = _state(model)
    if st is None:
        return 0.0, 0.0
    dims = [dim] if dim else list((st.get("N") or {}).keys())
    n1 = sum(float((st.get("N1") or {}).get(d, 0.0)) for d in dims)
    n = sum(float((st.get("N") or {}).get(d, 0.0)) for d in dims)
    return n1, n


def novelty_rate(model: Optional[Mapping], dim: Optional[str] = None) -> float:
    """Good-Turing missing mass N1/N (decayed singleton mass over total mass),
    pooled over dimensions when dim is None; 1.0 for an empty model."""
    n1, n = _n1_n(model, dim)
    if not n > 0.0:
        return 1.0
    return min(1.0, max(0.0, n1 / n))


def maturity(model: Optional[Mapping], dim: Optional[str] = None) -> float:
    return 1.0 - novelty_rate(model, dim)


def top_values(model: Optional[Mapping], dim: str, k: int = 10,
               now: Optional[float] = None) -> List[Tuple[str, float]]:
    c = counts(model, dim, now)
    tot = total(model, dim, now)
    if not c or not tot > 0.0:
        return []
    best = sorted(c.items(), key=lambda kv: -kv[1])[:k]
    return [(v, x / tot) for v, x in best]


def _family_of(dim: str, value: str) -> str:
    if dim == "tmpl":
        return token_family(value)
    if dim == "sni":
        return f"tls|read|{value}|"
    if dim == "dns":
        return f"dns|read|{value}|"
    if dim == "dport":
        return f"l4|other|{value}|"
    return ""


def family_distribution(src: Any, system: Optional[str] = None, entity: Optional[str] = None,
                        now: Optional[float] = None) -> Dict[str, float]:
    """{family: share} over tmpl / sni / dns / dport mass (B02 role descriptor).
    `src` is a model dict, or a store together with (system, entity)."""
    model = src if isinstance(src, dict) else get(src, system, entity)
    acc: Dict[str, float] = {}
    for dim in ("tmpl", "sni", "dns", "dport"):
        for v, c in counts(model, dim, now).items():
            f = _family_of(dim, v)
            if f and c > 0.0:
                acc[f] = acc.get(f, 0.0) + c
    tot = sum(acc.values())
    return {f: c / tot for f, c in acc.items()} if tot > 0.0 else {}


def adoption_records(store, system: str, class_key: str, since: Optional[float] = None,
                     min_members: int = 1) -> List[Dict[str, Any]]:
    """Class adoption records (B08 writes one per value some member first saw,
    always, whatever the per-entity discount). Copies; newest last_ts first."""
    m = get(store, system, class_key)
    if m is None:
        return []
    out = []
    for key, rec in (m.get("adoption") or {}).items():
        mem = rec.get("members") or {}
        recent = [t for t in mem.values() if since is None or float(t) >= float(since)]
        if len(recent) < int(min_members):
            continue
        r = dict(rec)
        r["members"] = dict(mem)
        r["flags"] = dict(rec.get("flags") or {})
        r["n_recent"] = len(recent)
        out.append(r)
    out.sort(key=lambda r: -float(r.get("last_ts", 0.0)))
    return out


def descriptors(model: Optional[Mapping], k: int = 8,
                now: Optional[float] = None) -> Dict[str, Any]:
    """Portrait descriptor: per dimension the number of values and the top-k
    shares, the Good-Turing novelty rate and the template families."""
    dims = {}
    for d in DIMS:
        tv = top_values(model, d, k, now)
        if tv:
            dims[d] = {"n_values": len(_dim(model, d)),
                       "top": [[v, round(s, 4)] for v, s in tv]}
    fam = family_distribution(model, now=now) if model is not None else {}
    top_fam = sorted(fam.items(), key=lambda kv: -kv[1])[:k]
    return {"dims": dims, "novelty_rate": round(novelty_rate(model), 4),
            "families": [[f, round(s, 4)] for f, s in top_fam]}
