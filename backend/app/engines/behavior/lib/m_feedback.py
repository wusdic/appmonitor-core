"""Read accessors for model.feedback (owner: B23 feedback; contract C).

Why a module and not direct dict reads: five engines act on analyst feedback
(B25 family weights and alpha_mult, B26 the risk multiplier and the 0.25
weight of suppressed incidents, B27 suppression, B28 accept / freeze, B08 and
B12 allowlists) and the UI shows the label queue. A suppression decision is
only safe if the pattern that was labelled and the pattern that is matched
are tokenised by the SAME code, so the tokeniser, the Jaccard / level test
and the stacker features live here and B23 builds its policies with them.

Every accessor takes `src`: a MetricStore (the model is read from
model.feedback@('__org__', '__org__')) or the model dict itself. An absent
model is neutral: uniform family weights, alpha_mult 1, no policies,
nothing allowlisted, precision 0.5 (pi = 1), p_malicious 0.5.

Layout of model.feedback@('__org__', '__org__') (JSON-safe; written by B23):
    {
      "version": int,                         # bumped on every content change
      "family_w": {family: w},                # all 12 families, mean-normalised
                                              # (uniform = 1.0), floored at W_FLOOR
      "detector_prec": {"<family>|<key>": [tp, fp]},
                                              # key '*' = the family aggregate,
                                              # otherwise a detector name
      "policies": [{id, label_id, verdict, scope, system, entity|None,
                    class_key|None, tokens[], gate[], e_day_ref, level,
                    created, expires}],
      "allowlist": {"<system>|<entity>": {dim: {value: expires|None}}},
      "alpha_mult": {system: a},              # a in [ALPHA_MIN, alpha_max]
      "accept": {"<system>|<entity>": [{ts, seq, t0, t1, label_id, target_id, scope}]},
      "freeze": {"<system>|<entity>": [{ts, seq, t0, t1, label_id, target_id, scope}]},
      "queue": [{incident_id, system, entity, reason, risk, p_malicious,
                 e_day, opened, added, severity}],
      "stacker": {"intercept", "coef": {feature: b}, "lam", "n", "n_pos",
                  "fitted_ts"} | None,
      "isotonic": {"x": [...], "y": [...], "n"} | None,
      "n_labelled": int,
      "_cases": {...}, "_state": {...}        # B23-private bookkeeping
    }
The contract writes accept[(s, e, ts)] / freeze[(s, e, ts)]; JSON has no
tuple keys, so the (s, e) pair is the "s|e" key and ts is the record's `ts`
(the pipeline time B23 processed the label; `seq` orders labels processed in
the same tick; t0 / t1 are the label's window, else the incident's).

Pattern tokens (engines.md B23 step 3). sigma(x) =
    {'k:<kind>' for each event kind of x} | {'a:<axis>' for each axis}
  | {'f:<feature>+|-' for the top-5 |z| features with |z| >= Z_MIN}
  | {'n:<dim>=<value>' for up to MAX_NEW new categorical tokens ('*' = no dim)}.
The kind / axis tokens are the GATE: an incident is suppressed by a policy iff
  * every gate token of the incident is in the policy's gate (it shows no
    kind or axis the analyst did not see: an exfil axis appearing on a
    suppressed backup pattern escapes), and
  * Jaccard(sigma_incident, sigma_policy) >= JACCARD_MIN (0.6), and
  * e_day(incident) >= policy.level = e_day_ref / 10 (one order of magnitude
    of headroom: a 1.5-decade rarer recurrence escapes), and
  * the policy is not expired (TTL 14 d unless the label set ttl_s) and the
    incident lies in its scope (pattern / system: same system; entity: same
    entity; class: the class key or one of its members).
An incident whose e_day is unknown is never suppressed (the level cannot be
verified). Features come from the object itself (a 'features' mapping,
BehaviorEvent.contributors, explanation['attributions'], evidence entries)
or, when it carries none and a store is given, from behavior.z at the
incident's ts; new tokens from the object and from the discrete novelty
events of the entity inside [opened, ts]. Give producers' events
extra = {'dim': ..., 'value': ...} (or 'token' / 'new_tokens') so their new
values tokenise, and B29 explanation['new_tokens'] as [{'dim', 'value'}].

Accessors (all read-only):
    get(src) -> dict
    family_weights(src) -> {family: w}                 # B25: w_f = this * weight_mult
    family_weight(src, family) -> float
    precision(src, family, key=None) -> (mean, n)      # Beta(1+TP, 1+FP)
    risk_mult(src, family, key=None) -> float          # B26 pi = clip(E[prec]/0.5, 0.2, 1)
    alpha_mult(src, system) -> float                   # B25: e_day thresholds x this
    allowlisted(store, system, entity, dim, value, now=None) -> bool   # B08 / B12
    allowlist_values(store, system, entity, dim, now=None) -> set[str]
    suppression_match(store, incident_like, now=None) -> policy dict | None   # B27
    is_suppressed(store, incident_like, now=None) -> bool
    SUPPRESSED_RISK_WEIGHT = 0.25                      # B26
    accepts(store, system, entity, since=None) -> [record]    # B28 (entity, class, system tier)
    freezes(store, system, entity, since=None) -> [record]
    latest_accept / latest_freeze(store, system, entity, since=None) -> record | None
    is_frozen(store, system, entity) -> bool
    label_queue(src, system=None) -> [item]            # GET /api/label-queue
    p_malicious(src, case_or_vector) -> float          # stacker (+ isotonic)
    summary(src, system=None, entity=None) -> dict     # portraits / profile.extra.feedback
Pure helpers shared with B23:
    describe(obj, store=None, dt_s=None) -> dict       # normalised incident-like view
    explicit_evidence(obj, evidence_from=0) -> ({feature: z}, {tokens})
    top_features(features) / build_tokens(kinds, axes, features, new) -> (tokens, gate)
    pattern_tokens(obj, store=None) -> (tokens, gate)
    jaccard(a, b), surprise(p, dt_s), surprise_e(e_day), stack_vector(case),
    token_str(dim, value) / split_token(tok), event_new_tokens(ev), involved_families(case)
"""
from __future__ import annotations

import math
from collections.abc import Mapping
from typing import Any, Dict, FrozenSet, Iterable, List, Optional, Sequence, Set, Tuple

import numpy as np

from . import m_class
from .classkeys import ORG, SYSTEM_KEY, is_class
from .detectors import FAMILIES
from .features import FEATURE_NAMES_V2
from .stages import stage_for_category, stage_for_event, stages_for

MODEL = "model.feedback"

# ---------------------------------------------------------------- constants
# pi = clip(E[prec] / 0.5, 0.2, 1): feedback may LOWER a family's risk weight,
# never raise it above the calibrated null (evaluator round 4, gate 12). The
# spec's upper clip 2 let tp labels on the attack incidents an analyst reviews
# first (change 12 tp / 3 fp, shape 7 / 1.5, intensity 5 / 1 on pack A seed 0)
# multiply that family's evidence on EVERY entity: the labelled set is
# selected (newest / loudest first), so its precision is not the family's
# precision on unlabelled traffic, and L_ref = 60 is fixed on clean replays.
# Measured: mean control-entity risk 30.9 -> 36.3 and 12 -> 15 risk
# reopenings of control incidents with the feedback on (pack A seed 0).
PREC_CLIP = (0.2, 1.0)
KEY_PRIOR = 2.0                 # key-level Beta shrinks to the family mean with this strength
W_FLOOR = 0.1                   # no family is ever weighted out of fusion entirely
ALPHA_MIN, ALPHA_MAX = 0.25, 4.0
SUPPRESSED_RISK_WEIGHT = 0.25   # a suppressed incident still feeds risk at this weight
JACCARD_MIN = 0.6
LEVEL_HEADROOM = 10.0           # policy level = labelled e_day / 10
E_DAY_UNKNOWN_REF = 0.03        # labelled e_day unknown -> treat as a LOW-level incident
Z_MIN = 2.0                     # a feature enters a pattern only at |z| >= 2
TOP_K = 5
MAX_NEW = 16
S_MAX = 8.0                     # stacker surprise cap: min(8, log10(1/e_day))
SIG_S = -math.log10(0.03)       # surprise of a family at e_day = 0.03 (LOW): "involved"
SECONDS_PER_DAY = 86400.0

SEVERITY_RANK = {"info": 0, "low": 1, "medium": 2, "high": 3, "critical": 4}
# e_day equivalent of a discrete finding's severity (architecture section 4 ladder)
SEVERITY_E_DAY = {"low": 0.03, "medium": 3e-3, "high": 3e-4, "critical": 3e-6}

STACK_FEATURES: List[str] = [f"s:{f}" for f in FAMILIES] + ["stages", "sig"]
N_STACK = len(STACK_FEATURES)
STAGES_CAP = 10

GENERIC_KINDS = frozenset({"incident", "alarm", ""})
# Discrete event kinds whose extra carries a (dim, value) new categorical token.
NEW_TOKEN_KINDS = ("first_seen", "rare_access", "class_adopted", "client_change",
                   "client_impersonation", "beacon", "class_adoption_risky", "shared_ip")
# Detector family a discrete finding counts for (precision / stacker features).
EVENT_FAMILY: Dict[str, str] = {
    "first_seen": "categorical", "rare_access": "categorical", "class_adopted": "categorical",
    "class_adoption_risky": "categorical",
    "client_change": "identity", "client_impersonation": "identity",
    "identity_mismatch": "identity", "unknown_identity": "identity",
    "low_identifiability": "identity", "entity_resolution": "identity",
    "possible_impersonation": "identity", "shared_ip": "identity",
    "identity_moved": "identity", "link_retracted": "identity",
    "new_entity_matched": "peer", "new_entity_unmatched": "peer", "class_transition": "peer",
    "class_split": "peer", "class_merge": "peer", "peer_outlier": "peer",
    "system_shift": "peer", "coherent_shift": "peer", "class_shift": "peer",
    "schedule_shift": "temporal", "beacon": "c2", "budget_exceeded": "exfil",
    "baseline_creep": "change", "regime": "change", "first_access_system": "xsys",
    "pattern_violation": "conformity",
}
DISCRETE_KINDS: Tuple[str, ...] = tuple(sorted(set(EVENT_FAMILY) | set(NEW_TOKEN_KINDS)
                                               | {"incident"}))
# Fallback when a labelled case has no p-value evidence at all: axis -> family.
AXIS_FAMILY: Dict[str, str] = {
    "volume": "intensity", "shape": "shape", "peer": "peer", "temporal": "temporal",
    "categorical": "categorical", "privilege": "categorical", "breadth": "breadth",
    "discovery": "breadth", "collection": "breadth", "exfil": "exfil",
    "sequence": "sequence", "credential": "sequence", "identity": "identity",
    "change": "change", "c2": "c2", "xsys": "xsys", "lateral": "xsys",
    "content": "conformity",
}

_FAMILY_SET = frozenset(FAMILIES)


# ================================================================ model access
def get(src: Any) -> Dict[str, Any]:
    """The model dict ({} when absent). `src` is a store or the model itself."""
    if src is None:
        return {}
    if isinstance(src, Mapping):
        return src  # type: ignore[return-value]
    m = src.get_model(ORG[0], ORG[1], MODEL, default=None)
    return m if isinstance(m, Mapping) else {}


def _store_of(src: Any) -> Any:
    return None if (src is None or isinstance(src, Mapping)) else src


def _key(system: str, entity: str) -> str:
    return f"{system}|{entity}"


def _finite(x: Any) -> bool:
    try:
        return math.isfinite(float(x))
    except (TypeError, ValueError):
        return False


# ============================================================ family weights
def family_weights(src: Any) -> Dict[str, float]:
    """{family: w} over every detector family; 1.0 (uniform) where absent.
    B25 multiplies these by the calibration-health weight_mult."""
    fw = get(src).get("family_w") or {}
    out = {}
    for f in FAMILIES:
        w = fw.get(f, 1.0)
        out[f] = float(w) if _finite(w) and float(w) > 0.0 else 1.0
    return out


def family_weight(src: Any, family: str) -> float:
    return family_weights(src)[family] if family in _FAMILY_SET else 1.0


# ================================================================= precision
def precision(src: Any, family: str, key: Optional[str] = None) -> Tuple[float, float]:
    """(E[prec], n labels) for a family, or for a (family, contributor key).

    Family level: Beta(1 + TP, 1 + FP). Key level (a detector of the family):
    Beta prior centred on the family mean with strength KEY_PRIOR = 2, which is
    exactly Beta(1 + TP, 1 + FP) while the family is still at 0.5, and lets a
    rarely labelled detector borrow from its family instead of sitting at 0.5.
    """
    prec = get(src).get("detector_prec") or {}
    tp_f, fp_f = _counts(prec.get(f"{family}|*"))
    mean_f = (1.0 + tp_f) / (2.0 + tp_f + fp_f)
    if key is None or key == "*":
        return mean_f, tp_f + fp_f
    tp_k, fp_k = _counts(prec.get(f"{family}|{key}"))
    return (tp_k + KEY_PRIOR * mean_f) / (tp_k + fp_k + KEY_PRIOR), tp_k + fp_k


def _counts(v: Any) -> Tuple[float, float]:
    if not v:
        return 0.0, 0.0
    tp, fp = float(v[0]), float(v[1])
    return (tp if tp > 0 else 0.0), (fp if fp > 0 else 0.0)


def risk_mult(src: Any, family: str, key: Optional[str] = None) -> float:
    """B26 feedback multiplier pi = clip(E[prec] / 0.5, 0.2, 1); 1.0 unlabelled."""
    mean, _ = precision(src, family, key)
    return min(PREC_CLIP[1], max(PREC_CLIP[0], mean / 0.5))


# ================================================================ alert budget
def alpha_mult(src: Any, system: str) -> float:
    """Per-system multiplier of the e_day severity thresholds (B25 passes it to
    combine.e_day_severity). 1.0 when no daily budget update has run yet."""
    a = (get(src).get("alpha_mult") or {}).get(system, 1.0)
    if not _finite(a) or float(a) <= 0.0:
        return 1.0
    return min(ALPHA_MAX, max(ALPHA_MIN, float(a)))


# =================================================================== allowlist
def token_str(dim: Any, value: Any) -> str:
    """Canonical new-token string '<dim>=<value>' ('*' when there is no dim).
    A dim never contains '=', so split_token() recovers (dim, value) exactly
    even when the value itself contains '='."""
    d = "*" if dim is None or str(dim) == "" else str(dim).replace("=", "_")
    return f"{d}={value}"


def split_token(tok: str) -> Tuple[str, str]:
    """Inverse of token_str: (dim, value); a bare legacy token gives ('*', tok)."""
    d, sep, v = str(tok).partition("=")
    return (d, v) if sep else ("*", str(tok))


def _entity_keys(store: Any, system: str, entity: str) -> List[str]:
    """The entity itself, its class keys (role, static, pool) and __system__."""
    keys = [_key(system, entity)]
    if store is not None and not is_class(entity) and not entity.startswith("__"):
        ck = m_class.class_key(store, system, entity)
        if ck:
            keys.append(_key(system, ck))
        a = m_class.assignment(store, system, entity) or {}
        for name in a.get("static") or []:
            keys.append(_key(system, f"class:static:{name}"))
        if a.get("pool"):
            keys.append(_key(system, f"class:pool:{a['pool']}"))
    if entity != SYSTEM_KEY:
        keys.append(_key(system, SYSTEM_KEY))
    return keys


def _alive(expires: Any, now: Optional[float]) -> bool:
    return expires is None or now is None or float(expires) > float(now)


def allowlisted(store: Any, system: str, entity: str, dim: Any, value: Any,
                now: Optional[float] = None) -> bool:
    """True when (dim, value) is allowlisted for the entity, one of its
    classes, or the whole system (benign_known labels). dim '*' entries match
    any dim. Values compare as str(value). `now` enforces TTLs (None: ignore)."""
    al = get(store).get("allowlist") or {}
    if not al:
        return False
    d, v = str(dim), str(value)
    for k in _entity_keys(_store_of(store), system, entity):
        dims = al.get(k)
        if not dims:
            continue
        for dd in (d, "*"):
            vals = dims.get(dd)
            if vals and v in vals and _alive(vals[v], now):
                return True
    return False


def allowlist_values(store: Any, system: str, entity: str, dim: Any,
                     now: Optional[float] = None) -> Set[str]:
    """Every allowlisted value for `dim` (plus '*' entries) across the entity,
    class and system tiers: one lookup for a whole token set (B08)."""
    al = get(store).get("allowlist") or {}
    out: Set[str] = set()
    if not al:
        return out
    for k in _entity_keys(_store_of(store), system, entity):
        dims = al.get(k) or {}
        for dd in (str(dim), "*"):
            for v, exp in (dims.get(dd) or {}).items():
                if _alive(exp, now):
                    out.add(v)
    return out


# ============================================================== accept/freeze
def _records(src: Any, field: str, system: str, entity: str,
             since: Optional[float]) -> List[Dict[str, Any]]:
    tab = get(src).get(field) or {}
    out = []
    for k in _entity_keys(_store_of(src), system, entity):
        for r in tab.get(k) or []:
            if since is None or float(r.get("ts", -math.inf)) >= since:
                rr = dict(r)
                rr["key"] = k
                out.append(rr)
    out.sort(key=_order)
    return out


def accepts(src: Any, system: str, entity: str,
            since: Optional[float] = None) -> List[Dict[str, Any]]:
    """expected_change records for the entity (and its class / system tier),
    oldest first, with ts >= since. Each: {ts, t0, t1, label_id, target_id,
    scope, key}. B28 turns them into ACCEPT (rebase_from = t0)."""
    return _records(src, "accept", system, entity, since)


def freezes(src: Any, system: str, entity: str,
            since: Optional[float] = None) -> List[Dict[str, Any]]:
    """tp (malicious) records: B28 rejects and freezes the learners."""
    return _records(src, "freeze", system, entity, since)


def latest_accept(src: Any, system: str, entity: str,
                  since: Optional[float] = None) -> Optional[Dict[str, Any]]:
    r = accepts(src, system, entity, since)
    return r[-1] if r else None


def latest_freeze(src: Any, system: str, entity: str,
                  since: Optional[float] = None) -> Optional[Dict[str, Any]]:
    r = freezes(src, system, entity, since)
    return r[-1] if r else None


def _order(r: Mapping[str, Any]) -> Tuple[float, int]:
    return float(r.get("ts", 0.0)), int(r.get("seq", 0))


def is_frozen(src: Any, system: str, entity: str) -> bool:
    """A tp record exists for the entity (or its class / system tier) that no
    LATER expected_change at the SAME tier has superseded: a class-wide
    acceptance never lifts a freeze on one member labelled malicious."""
    fz = freezes(src, system, entity)
    if not fz:
        return False
    ac = accepts(src, system, entity)
    for f in fz:
        if not any(a["key"] == f["key"] and _order(a) > _order(f) for a in ac):
            return True
    return False


# ================================================================ label queue
def label_queue(src: Any, system: Optional[str] = None) -> List[Dict[str, Any]]:
    """The analyst label queue: 'held' (> 14 d unresolved) first, then the
    daily picks ('risk', 'uncertain') by risk. Copies; never the model's own."""
    q = [dict(it) for it in (get(src).get("queue") or [])
         if system is None or it.get("system") == system]
    order = {"held": 0, "risk": 1, "uncertain": 2}
    q.sort(key=lambda it: (order.get(it.get("reason"), 3), -float(it.get("risk") or 0.0),
                           str(it.get("incident_id"))))
    return q


# ====================================================== incident-like parsing
def _attr(obj: Any, name: str, default: Any = None) -> Any:
    if isinstance(obj, Mapping):
        return obj.get(name, default)
    return getattr(obj, name, default)


def _sev_name(sev: Any) -> str:
    v = getattr(sev, "value", sev)
    return str(v).lower() if v is not None else ""


def surprise(p: Any, dt_s: float) -> Optional[float]:
    """Excess surprise min(8, max(0, log10(1 / e_day(p)))) with e_day =
    p * 86400 / dt: 0 at once-a-day, 1.52 at e_day 0.03, capped at 8. The floor
    at 0 makes it cadence-free (an unremarkable p gives 0 at 60 s and 900 s
    alike). NaN / None p -> None (unscored is not 'normal')."""
    if p is None or not _finite(p) or not (dt_s and dt_s > 0):
        return None
    pv = min(1.0, max(1e-300, float(p)))
    s = -math.log10(pv * SECONDS_PER_DAY / float(dt_s))
    return min(S_MAX, max(0.0, s))


def surprise_e(e_day: Any) -> Optional[float]:
    """Same excess surprise from an e_day value."""
    if e_day is None or not _finite(e_day):
        return None
    e = max(1e-300, float(e_day))
    return min(S_MAX, max(0.0, -math.log10(e)))


def _feature_items(val: Any) -> Iterable[Tuple[str, float]]:
    """(name, signed z) pairs from a mapping, a list of pairs, or a list of
    dicts {feature|name, z|value|sign|direction}."""
    if not val:
        return
    if isinstance(val, Mapping):
        for n, z in val.items():
            if _finite(z):
                yield str(n), float(z)
        return
    for it in val:
        if isinstance(it, Mapping):
            n = it.get("feature", it.get("name"))
            z = it.get("z", it.get("value"))
            if z is None:
                s = it.get("sign", it.get("direction"))
                if s in ("up", "+", "high"):
                    z = Z_MIN
                elif s in ("down", "-", "low"):
                    z = -Z_MIN
                elif _finite(s) and float(s) != 0.0:
                    z = math.copysign(Z_MIN, float(s))
            if n is not None and _finite(z):
                yield str(n), float(z)
        elif isinstance(it, (list, tuple)) and len(it) >= 2 and _finite(it[1]):
            yield str(it[0]), float(it[1])


def _merge_features(dst: Dict[str, float], items: Iterable[Tuple[str, float]]) -> None:
    for n, z in items:
        if n not in dst or abs(z) > abs(dst[n]):
            dst[n] = z


def _new_items(val: Any) -> Iterable[str]:
    """New categorical tokens from a list of str / {dim, value} / {token} /
    (dim, value) entries."""
    if not val:
        return
    if isinstance(val, (str, bytes)):
        yield _as_token(val)
        return
    if isinstance(val, Mapping):
        if "value" in val or "token" in val:
            yield token_str(val.get("dim"), val.get("value", val.get("token")))
        return
    for it in val:
        if isinstance(it, Mapping):
            if "value" in it or "token" in it:
                yield token_str(it.get("dim"), it.get("value", it.get("token")))
        elif isinstance(it, (list, tuple)) and len(it) >= 2:
            yield token_str(it[0], it[1])
        elif it is not None:
            yield _as_token(it)


def _as_token(x: Any) -> str:
    """A bare string that is already 'dim=value' stays as is; else '*=x'."""
    t = x.decode("utf-8", "replace") if isinstance(x, bytes) else str(x)
    d, sep, _ = t.partition("=")
    return t if sep and d and " " not in d and len(d) <= 32 else token_str("*", t)


def event_new_tokens(ev: Any) -> List[str]:
    """New categorical tokens a discrete event carries in its extra:
    {'dim', 'value'} | {'token'} | {'tokens' | 'new_tokens': [...]}."""
    ex = _attr(ev, "extra") or {}
    out: List[str] = []
    if not isinstance(ex, Mapping):
        return out
    if "value" in ex or "token" in ex:
        v = ex.get("value", ex.get("token"))
        if isinstance(v, (list, tuple, set)):
            out.extend(token_str(ex.get("dim"), x) for x in v)
        elif v is not None:
            out.append(token_str(ex.get("dim"), v))
    for k in ("tokens", "new_tokens", "values"):
        if ex.get(k):
            if k == "values" and not isinstance(ex[k], Mapping):
                out.extend(token_str(ex.get("dim"), x) for x in ex[k])
            else:
                out.extend(_new_items(ex[k]))
    return out


def explicit_evidence(obj: Any, evidence_from: int = 0) -> Tuple[Dict[str, float], Set[str]]:
    """({feature: z}, {new tokens}) the object itself carries: a 'features'
    mapping, BehaviorEvent.contributors, explanation['attributions'] /
    ['new_tokens'], 'new_tokens', a novelty event's extra, and the evidence
    entries from index `evidence_from` on (B23 parses an append-only
    evidence list incrementally)."""
    feats: Dict[str, float] = {}
    new: Set[str] = set()
    _merge_features(feats, _feature_items(_attr(obj, "features", None)))
    _merge_features(feats, _feature_items(_attr(obj, "contributors", None)))
    expl = _attr(obj, "explanation", None) or {}
    if isinstance(expl, Mapping) and expl:
        _merge_features(feats, _feature_items(expl.get("attributions")))
        new.update(_new_items(expl.get("new_tokens")))
    evidence = _attr(obj, "evidence", None) or []
    for ev in evidence[evidence_from:] if evidence_from else evidence:
        if isinstance(ev, Mapping):
            _merge_features(feats, _feature_items(ev.get("contributors")))
            _merge_features(feats, _feature_items(ev.get("features")))
            new.update(_new_items(ev.get("new_tokens")))
    new.update(_new_items(_attr(obj, "new_tokens", None)))
    if _attr(obj, "kind", None) in NEW_TOKEN_KINDS:        # a novelty event itself
        new.update(event_new_tokens(obj))
    return feats, new


def describe(obj: Any, store: Any = None, dt_s: Optional[float] = None) -> Dict[str, Any]:
    """Normalised view of an Incident, a BehaviorEvent or a dict:
    {system, entity, kinds, axes, ts, opened, e_day, features {name: z},
    new [tokens]}. With a store, missing features are read from behavior.z at
    ts, novelty events of the entity in [opened, ts] add new tokens, and a
    missing e_day is the min of behavior.e_day over [opened, ts]."""
    system = str(_attr(obj, "system", "") or "")
    entity = str(_attr(obj, "entity", "") or "")
    kinds_v = _attr(obj, "kinds", None)
    if kinds_v is None:
        k = _attr(obj, "kind", None)
        kinds_v = [k] if isinstance(k, str) else (k or [])
    kinds = sorted({str(k) for k in kinds_v if k is not None})
    axes = sorted({str(a) for a in (_attr(obj, "axes", None) or []) if a})
    ts = _attr(obj, "last_seen", None)
    if ts is None or not _finite(ts) or float(ts) <= 0:
        ts = _attr(obj, "ts", None)
    ts = float(ts) if _finite(ts) else math.nan
    opened = _attr(obj, "opened", None)
    opened = float(opened) if _finite(opened) and float(opened) > 0 else ts
    e = _attr(obj, "e_day_min", None)
    if e is None:
        e = _attr(obj, "e_day", None)
    e_day = float(e) if _finite(e) else None

    feats, new = explicit_evidence(obj)

    if store is not None and system and entity and math.isfinite(ts):
        if not feats:
            feats.update(z_features(store, system, entity, ts, opened))
        if _attr(obj, "kinds", None) is not None:          # incident-like: merged novelty events
            for ev in store.events(system, entity, since=opened, kinds=NEW_TOKEN_KINDS, limit=64):
                if ev.ts <= ts:
                    new.update(event_new_tokens(ev))
        if e_day is None:
            _, M = store.vec_range(system, entity, "behavior.e_day", opened, ts)
            if M.size:
                v = M[:, 0].astype(np.float64)
                v = v[np.isfinite(v)]
                if v.size:
                    e_day = float(v.min())
    return {"system": system, "entity": entity, "kinds": kinds, "axes": axes, "ts": ts,
            "opened": opened, "e_day": e_day, "features": feats, "new": sorted(new)}


def z_features(store: Any, system: str, entity: str, ts: float,
               opened: Optional[float] = None) -> Dict[str, float]:
    """{feature: z} of behavior.z (B04, 52 columns in FEATURE_NAMES_V2 order) at
    ts, else the newest row in [opened, ts]; {} when there is none."""
    row = store.vec_at(system, entity, "behavior.z", ts)
    if row is None:
        lo = ts if opened is None or not math.isfinite(opened) else min(opened, ts)
        tss, M = store.vec_range(system, entity, "behavior.z", lo, ts)
        if not len(tss):
            return {}
        row = M[-1]
    row = np.asarray(row, dtype=np.float64)
    if row.shape[0] != len(FEATURE_NAMES_V2):
        return {}
    return {FEATURE_NAMES_V2[i]: float(row[i]) for i in np.flatnonzero(np.isfinite(row))}


# ===================================================================== tokens
def top_features(features: Any, k: int = TOP_K, z_min: float = Z_MIN) -> List[Tuple[str, float]]:
    """The k largest |z| features with |z| >= z_min, ties broken by name."""
    items = [(n, z) for n, z in _feature_items(features) if abs(z) >= z_min]
    items.sort(key=lambda t: (-abs(t[1]), t[0]))
    return items[:k]


def build_tokens(kinds: Iterable[str], axes: Iterable[str], features: Any,
                 new: Iterable[str]) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    """(sigma, gate): the pattern token set and its kind / axis subset."""
    gate = {f"k:{k}" for k in kinds if k not in GENERIC_KINDS}
    gate |= {f"a:{a}" for a in axes if a}
    toks = set(gate)
    toks |= {f"f:{n}{'+' if z > 0 else '-'}" for n, z in top_features(features)}
    toks |= {f"n:{t}" for t in sorted(set(new))[:MAX_NEW]}
    return frozenset(toks), frozenset(gate)


def pattern_tokens(obj: Any, store: Any = None) -> Tuple[FrozenSet[str], FrozenSet[str]]:
    d = obj if (isinstance(obj, Mapping) and "new" in obj and "features" in obj
                and "kinds" in obj) else describe(obj, store)
    return build_tokens(d["kinds"], d["axes"], d["features"], d["new"])


def jaccard(a: Iterable[str], b: Iterable[str]) -> float:
    """|a & b| / |a | b|; 0 when both are empty (an empty pattern matches nothing)."""
    sa, sb = set(a), set(b)
    u = len(sa | sb)
    return len(sa & sb) / u if u else 0.0


# ================================================================ suppression
# One-entry cache of the frozenset form of the policies, keyed by the policy
# list object itself (held, so its id cannot be recycled) and the model
# version, which B23 bumps on every change.
_POLICY_CACHE: Dict[str, Any] = {"ref": None, "ver": None, "compiled": []}


def _compiled_policies(model: Mapping) -> List[Tuple[Dict[str, Any], FrozenSet[str], FrozenSet[str]]]:
    pol = model.get("policies") or []
    ver = model.get("version")
    if (isinstance(ver, int) and _POLICY_CACHE["ref"] is pol
            and _POLICY_CACHE["ver"] == (ver, len(pol))):
        return _POLICY_CACHE["compiled"]
    comp = [(p, frozenset(p.get("tokens") or ()), frozenset(p.get("gate") or ())) for p in pol]
    if isinstance(ver, int):
        _POLICY_CACHE.update(ref=pol, ver=(ver, len(pol)), compiled=comp)
    return comp


def _in_scope(store: Any, pol: Mapping, system: str, entity: str) -> bool:
    if pol.get("system") and pol["system"] != system:
        return False
    scope = pol.get("scope")
    if scope == "entity":
        return pol.get("entity") == entity
    if scope == "class":
        ck = pol.get("class_key")
        if not ck:
            return pol.get("entity") == entity
        if entity == ck:
            return True
        return store is not None and entity in m_class.class_members(store, system, ck)
    return True                                  # pattern / system: the whole system


def suppression_match(store: Any, incident_like: Any,
                      now: Optional[float] = None) -> Optional[Dict[str, Any]]:
    """The first live pattern policy that suppresses this incident, else None.

    `incident_like`: an Incident (B27 calls this when it opens or updates one),
    a BehaviorEvent or a dict (see describe). The returned dict is a copy of
    the policy plus 'similarity'. B27 then stores the incident with status =
    suppressed; B26 still counts it at SUPPRESSED_RISK_WEIGHT.
    """
    model = get(store)
    comp = _compiled_policies(model)
    if not comp:
        return None
    d = describe(incident_like, _store_of(store))
    e_day = d["e_day"]
    if e_day is None:
        return None
    t_now = now if now is not None else (d["ts"] if math.isfinite(d["ts"]) else None)
    toks, gate = build_tokens(d["kinds"], d["axes"], d["features"], d["new"])
    if not toks:
        return None
    st = _store_of(store)
    for pol, ptoks, pgate in comp:
        if not ptoks or not _alive(pol.get("expires"), t_now):
            continue
        if not gate <= pgate:
            continue
        lvl = pol.get("level")
        if lvl is None or not _finite(lvl) or e_day < float(lvl):
            continue
        sim = jaccard(toks, ptoks)
        if sim < JACCARD_MIN:
            continue
        if not _in_scope(st, pol, d["system"], d["entity"]):
            continue
        out = dict(pol)
        out["similarity"] = sim
        return out
    return None


def is_suppressed(store: Any, incident_like: Any, now: Optional[float] = None) -> bool:
    return suppression_match(store, incident_like, now) is not None


# ==================================================================== stacker
def stack_vector(case: Mapping[str, Any]) -> np.ndarray:
    """Stacker input [s_family x 12, stage count, signature max severity] from a
    B23 case {fs: {family: surprise}, stages, sig}. Absent families are 0
    (no excess surprise), never NaN."""
    x = np.zeros(N_STACK)
    fs = case.get("fs") or {}
    for i, f in enumerate(FAMILIES):
        v = fs.get(f)
        if v is not None and _finite(v):
            x[i] = min(S_MAX, max(0.0, float(v)))
    x[len(FAMILIES)] = min(STAGES_CAP, max(0.0, float(case.get("stages") or 0)))
    x[len(FAMILIES) + 1] = min(4.0, max(0.0, float(case.get("sig") or 0)))
    return x


def _sigmoid(z: float) -> float:
    if z >= 0.0:
        return 1.0 / (1.0 + math.exp(-z))
    ez = math.exp(z)
    return ez / (1.0 + ez)


def stacker_prob(stacker: Mapping[str, Any], x: np.ndarray) -> float:
    coef = stacker.get("coef") or {}
    z = float(stacker.get("intercept", 0.0))
    for i, name in enumerate(STACK_FEATURES):
        z += float(coef.get(name, 0.0)) * float(x[i])
    return _sigmoid(z)


def isotonic_apply(iso: Mapping[str, Any], p: float) -> float:
    xs, ys = iso.get("x") or [], iso.get("y") or []
    if not xs:
        return p
    return float(np.interp(p, xs, ys))


def involved_families(case: Mapping[str, Any]) -> Dict[str, float]:
    """{family: surprise} of families at or beyond the LOW level in a case;
    falls back to the case's axes (weight SIG_S) when it has no p evidence."""
    fs = case.get("fs") or {}
    out = {f: float(s) for f, s in fs.items() if f in _FAMILY_SET and _finite(s) and s >= SIG_S}
    if not out:
        for a in case.get("axes") or []:
            f = AXIS_FAMILY.get(a)
            if f:
                out[f] = SIG_S
    return out


def p_malicious(src: Any, case: Any) -> float:
    """P(malicious) of a case (dict with fs / stages / sig) or a stack vector.

    >= 20 labels: the stacking logistic regression, isotonic-calibrated once
    >= 100 labels. Before that: the surprise-weighted mean Beta precision of
    the involved families (0.5 with no labels or no involved family)."""
    model = get(src)
    x = case if isinstance(case, np.ndarray) else stack_vector(case)
    st = model.get("stacker")
    if st:
        p = stacker_prob(st, x)
        iso = model.get("isotonic")
        return isotonic_apply(iso, p) if iso else p
    if isinstance(case, np.ndarray):
        fam = {f: float(x[i]) for i, f in enumerate(FAMILIES) if x[i] >= SIG_S}
    else:
        fam = involved_families(case)
    if not fam:
        return 0.5
    num = sum(precision(model, f)[0] * s for f, s in fam.items())
    return num / sum(fam.values())


def stage_count(axes: Iterable[str], kinds: Iterable[str],
                categories: Iterable[str] = ()) -> int:
    """Distinct kill-chain stages over axes, event kinds and lib-4 categories."""
    st = set(stages_for(axes))
    st |= {s for s in (stage_for_event(k) for k in kinds) if s}
    st |= {s for s in (stage_for_category(c) for c in categories) if s}
    return len(st)


# ==================================================================== summary
def summary(src: Any, system: Optional[str] = None,
            entity: Optional[str] = None) -> Dict[str, Any]:
    """Compact descriptor for portraits, profile.extra.feedback and the UI."""
    m = get(src)
    pol = m.get("policies") or []
    if system is not None:
        pol = [p for p in pol if p.get("system") in (None, system)]
    if entity is not None:
        pol = [p for p in pol if p.get("scope") not in ("entity", "class")
               or p.get("entity") == entity or p.get("class_key") == entity]
    out: Dict[str, Any] = {
        "n_labelled": int(m.get("n_labelled") or 0),
        "policies": len(pol),
        "stacker": bool(m.get("stacker")),
        "isotonic": bool(m.get("isotonic")),
        "queue": len(label_queue(m, system)),
    }
    if system is not None:
        out["alpha_mult"] = alpha_mult(m, system)
    if system is not None and entity is not None:
        al = (m.get("allowlist") or {}).get(_key(system, entity)) or {}
        out["allowlist"] = sum(len(v) for v in al.values())
        a = (m.get("accept") or {}).get(_key(system, entity)) or []
        f = (m.get("freeze") or {}).get(_key(system, entity)) or []
        out["accepted_ts"] = a[-1]["ts"] if a else None
        out["frozen_ts"] = f[-1]["ts"] if f else None
    return out
