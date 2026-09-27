"""Read accessors and shared maths for model.identity / model.idwin (owner: B15
IdentityModelEngine; contract C). B16 (attribution), B17 (entity linking),
B30 (portraits) and B02 (role descriptors) call these functions instead of
reading model internals, so the layout can evolve in one place.

Why a module: identity is ABSOLUTE (architecture principle 7). A window is
scored under every candidate's OWN model, on absolute data (feature.vec,
feature.sketch, behavior.timing, raw tokens / streams / stacks), never on
behavior.z / zr / zi, which are normalised to each entity itself and would
make every entity look like "itself". The window representation, the metric
(PCA -> WCCN -> LDA), the per-modality LLR calibration and the modality
log-likelihood helpers must be byte-identical between the engine that fits
them (B15) and the engines that apply them (B16 / B17), so they live here.

Window representation (K_WIN = 4 committed active ticks, non-overlapping):
    tick_row(store, s, e, ts)      -> float64[TICK_DIM = 138] | None
        [feature.vec 52 | feature.sketch 80 | behavior.timing (B, M, think_mu) 3 |
         sin, cos of the local hour, workday flag 3]
        think_mu is already the mean of ln(gap) (B11), i.e. the spec's
        "log think_mu".
    window_vector(rows[k, 138])    -> float32[RAW_DIM = 146]
        [median vec 52 (NaN where every tick is NaN) | IQR of IQR_FEATURES 8 |
         mean sketch 80 | median timing 3 | circular-mean hour sin, cos and the
         workday fraction at the window centre 3]
    NaN in the median block is imputed at FIT/transform time with the
    role-class median (else the system median) and a missing mask is appended,
    so the stored vector is a pure function of its rows (replay is exact).

model.identity@(s, '__system__') layout (JSON-like, arrays as lists):
    {'fmt': 1, 'version': int, 'fitted_ts': float, 'run': int,
     'entities': [e, ...], 'roles': {e: rid | None},
     'pca': {'fill': [146], 'fill_cls': {ck: [146]}, 'mask_cols': [idx < 52],
             'keep': [col idx of the augmented vector], 'center': [p], 'scale': [p],
             'P': [[p x r]]  (standardise -> PCA(d <= 48) -> WCCN -> LDA, fused),
             'd_pca': int, 'r': int},
     'W': [[d_pca x r]] (LDA directions in WCCN-whitened PCA space, diagnostics),
     'means': {e: [r]},  'class_means': {ck: [r]}, 'class_var': {ck: [r]},
     'bg': {'mu': [r], 'prec': [[r x r]], 'logdet': float},
     'llr_calib': {m: [a, b]}, 'llr_n': {m: [n_genuine, n_impostor]},
     'confusion': {e: {j: share}}, 'anonymity_sets': [[e, ...], ...],
     'stats': {e: {'recall1', 'recallK', 'eer_hard', 'eer_pair': {j: eer},
                   't99', 'separability', 'near': [3 nearest impostors],
                   'n_windows', 'confusable_with': [..]}},
     'classes': {ck: {'recall1', 'eer_hard', 'separability', 'identifiability',
                      'confusable_with', 'n_members', 'n_windows'}},
     'distinctive': {e: {'features': [...], 'vocab': [...]}},
     'modality_share': {e: {block: share}}, 'modality_share_ts': float}

Gaussian modality (LDA space, within-entity covariance = I by construction):
    l_j(z)  = -0.5 ||z - m_j||^2 - 0.5 r ln 2 pi            (entity j)
    l_c(z)  = diagonal Gaussian around the class mean       (class c)
    l_bg(z) = full Gaussian of all windows (the "somebody" background)
    llr_gauss(z, j) = l_j(z) - l_bg(z)
Non-gauss modalities (nats, candidate's own model through its accessor):
    vocab  m_vocab chain (entity -> class -> system) minus the system tier
    rhythm ln p_j(slot active) (m_rhythm) minus the system / pooled p
    seq    PPM surprisal under the system tier minus under j's chain (m_seq), x ln 2
    client m_client chain minus the system tier
    timing gap-histogram loglik (m_timing) minus the pooled system pmf
Calibration (fitted by B15, logistic regression of genuine vs impostor
window LLRs with balanced classes, so a*llr + b is a prior-free LLR):
    calibrate(model, m, llr) = clip(a_m * llr + b_m, -LLR_CAP, LLR_CAP)
    Defaults (1, 0) until MIN_CAL samples of each label exist.

Public API (all pure reads; nothing here mutates a model):
    get(store, s) -> dict | None;  is_fitted(model) -> bool;  version(model) -> int
    tick_row(store, s, e, ts, *, tctx=None, timing=None, depth=64) -> ndarray | None
    window_vector(rows) -> float32[146];  window_from_store(store, s, e, ts_list)
    transform(model, x, class_key=None) -> float64[r] (NaN when unfitted)
    transform_many(model, X, class_keys=None) -> float64[n, r]
    entities(model) -> [e];  entity_mean(model, e);  class_mean(model, ck)
    gauss_loglik(model, z, cand) -> nats;  bg_loglik(model, z) -> nats
    gauss_llr(model, z, cand) -> nats;  scores(model, z) -> {e: l_e(z)}
    top_k(model, z, k=5, exclude=()) -> [(e, l_e)] best first
    llr_calib(model, m) -> (a, b);  calibrate(model, m, llr, cap=LLR_CAP) -> nats
    confusion(model, e, j=None);  confusable_with(model, e) -> [j]
    anonymity_set(model, e) -> [e, ...];  anonymity_sets(model) -> [[...]]
    stats(model, e) -> dict;  t99(model, e);  eer_hard(model, e);  separability(model, e)
    nearest(model, e) -> [j];  class_stats(model, ck) -> dict
    descriptors(model, e) -> JSON-safe dict for profile.extra.identity / portraits
    distinctive(model, e) -> {'features': [...], 'vocab': [...]}
    tick_modal_data(store, s, e, ts, *, tctx=None, smap=None) -> ModalData
    Background(store, s, now)  (per-tick cache of the system tiers)
    modality_logliks(store, s, cand, data, now, bg) -> {m: nats (NaN = no evidence)}
    modality_llrs(store, s, cands, data, now, bg=None) -> {cand: {m: nats}}
    eer(genuine, impostor) -> float;  logistic_calibration(llr, label) -> (a, b)
    union_find_sets(nodes, pairs) -> [[...]];  bhattacharyya_diag(m1, v1, m2, v2)
    mcq_log_odds(y_self, y_peer, prior) -> (delta, z);  to_natural(idx, v) -> float
"""
from __future__ import annotations

import math
import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np

from . import features as F
from . import m_client, m_rhythm, m_seq, m_template, m_timing, m_vocab
from .classkeys import SYSTEM_KEY
from .names import etld1
from .template import channel_of
from .sketch import SKETCH_BLOCK, SKETCH_NAMESPACES

MODEL = "model.identity"
IDWIN = "model.idwin"
FMT = 1

K_WIN = 4                                   # committed active ticks per window
VEC_DIM = F.FEATURE_DIM                     # 52
SKETCH_DIM = 80
TIMING_KEYS = ("B", "M", "think_mu")
IQR_FEATURES = ("bytes_up", "bytes_down", "flows", "http_requests", "dns_queries",
                "distinct_peers", "distinct_templates", "http_write_ratio")
IQR_IDX = tuple(F.FEATURE_INDEX[n] for n in IQR_FEATURES)

# tick row layout
T_VEC = slice(0, 52)
T_SK = slice(52, 132)
T_TIM = slice(132, 135)
T_CLK = slice(135, 138)
TICK_DIM = 138
# window vector layout
W_MED = slice(0, 52)
W_IQR = slice(52, 60)
W_SK = slice(60, 140)
# sketch namespace 'client.stack_set' (lib/sketch.SKETCH_NAMESPACES[1]) inside W_SK
_SK_CLIENT = SKETCH_NAMESPACES.index("client.stack_set")
W_SK_CLIENT = slice(60 + SKETCH_BLOCK * _SK_CLIENT, 60 + SKETCH_BLOCK * (_SK_CLIENT + 1))
W_TIM = slice(140, 143)
W_CLK = slice(143, 146)
RAW_DIM = 146

# modality-drop blocks (raw window columns; a feature's missing mask follows it)
_G = F.GROUPS
_IQR_POS = {F.FEATURE_INDEX[n]: 52 + i for i, n in enumerate(IQR_FEATURES)}


def _feat_block(groups: Sequence[str]) -> Tuple[int, ...]:
    cols: List[int] = []
    for g in groups:
        for i in _G[g]:
            cols.append(i)
            if i in _IQR_POS:
                cols.append(_IQR_POS[i])
    return tuple(sorted(cols))


BLOCKS: Dict[str, Tuple[int, ...]] = {
    "volume": _feat_block(["volume"]),
    "breadth": _feat_block(["breadth"]),
    "app": _feat_block(["app"]),
    "dns_tls": _feat_block(["dns", "tls"]),
    "timing": _feat_block(["timing"]) + tuple(range(140, 143)),
    "transport": _feat_block(["transport", "probe"]),
    "comp": _feat_block(["comp"]),
    "sketch": tuple(range(60, 140)),
    "clock": tuple(range(143, 146)),
}

MODALITIES = ("gauss", "vocab", "rhythm", "seq", "client", "timing")
VOCAB_N = 20.0                  # vocab LLR x min(n_tok, VOCAB_N) / VOCAB_N
LLR_CAP = 4.0                    # nats per modality per window (B16)
MIN_CAL = 20                     # samples of EACH label before a fit replaces (1, 0)
DEFAULT_CALIB = (1.0, 0.0)
CONFUSABLE_MIN = 0.05            # confusion share listed in confusable_with
PAIR_EER_CONFUSABLE = 0.2        # pairwise EER listed in confusable_with
ANON_CONFUSION = 0.2             # union-find edge (spec)
SEQ_MAX_TOKENS = 64              # per tick, PPM scoring cost bound
_LN2 = math.log(2.0)
_LN2PI = math.log(2.0 * math.pi)
_NAN = math.nan


# ================================================================= basics
def get(store, s: str) -> Optional[Dict[str, Any]]:
    m = store.get_model(s, SYSTEM_KEY, MODEL, default=None)
    return m if isinstance(m, dict) and m.get("fmt") == FMT else None


def is_fitted(model: Any) -> bool:
    return (isinstance(model, Mapping) and bool(model.get("means"))
            and isinstance(model.get("pca"), Mapping) and model["pca"].get("P") is not None)


def version(model: Any) -> int:
    return int(model.get("version", 0)) if isinstance(model, Mapping) else 0


def entities(model: Any) -> List[str]:
    return list(model.get("entities") or []) if isinstance(model, Mapping) else []


def _f(x: Any) -> float:
    try:
        v = float(x)
    except (TypeError, ValueError):
        return _NAN
    return v


# ================================================================= windows
def _dict_at(store, s: str, e: str, name: str, ts: float, depth: int) -> Optional[Mapping]:
    """Dict series point written at exactly ts, scanning back <= depth points."""
    for m in reversed(store.derived_tail(s, e, name, max(1, int(depth)))):
        if m.ts == ts:
            return m.value if isinstance(m.value, Mapping) else None
        if m.ts < ts:
            break
    return None


def clock_features(tctx: Optional[Mapping]) -> np.ndarray:
    """(sin, cos) of the local hour and a workday flag; NaN without a context."""
    if not isinstance(tctx, Mapping):
        return np.full(3, _NAN)
    h = _f(tctx.get("hour_local"))
    if not math.isfinite(h):
        return np.full(3, _NAN)
    a = 2.0 * math.pi * h / 24.0
    dtp = tctx.get("day_type")
    wd = 1.0 if dtp == "workday" else 0.0 if dtp == "nonworkday" else _NAN
    return np.array([math.sin(a), math.cos(a), wd])


def tick_row(store, s: str, e: str, ts: float, *, tctx: Optional[Mapping] = None,
             timing: Optional[Mapping] = None, depth: int = 64) -> Optional[np.ndarray]:
    """float64[138] absolute tick row, None when feature.vec has no row at ts
    (not written, or pruned: feature.vec is retained 1 d). A missing sketch or
    timing row gives NaN in its block (absence of a descriptor is not a zero).
    tctx / timing may be passed in; otherwise they are looked up at exactly ts
    (feature.tctx, behavior.timing), scanning back at most `depth` points."""
    v = store.vec_at(s, e, "feature.vec", ts)
    if v is None:
        return None
    out = np.full(TICK_DIM, _NAN)
    v = np.asarray(v, dtype=np.float64).reshape(-1)
    if v.size == VEC_DIM:
        out[T_VEC] = v
    sk = store.vec_at(s, e, "feature.sketch", ts)
    if sk is not None:
        sk = np.asarray(sk, dtype=np.float64).reshape(-1)
        if sk.size == SKETCH_DIM:
            out[T_SK] = sk
    tm = timing if timing is not None else _dict_at(store, s, e, "behavior.timing", ts, depth)
    if isinstance(tm, Mapping):
        out[T_TIM] = [_f(tm.get(k)) for k in TIMING_KEYS]
    tc = tctx if tctx is not None else _dict_at(store, s, e, "feature.tctx", ts, depth)
    out[T_CLK] = clock_features(tc)
    return out


def _nanmedian(a: np.ndarray, axis: int = 0) -> np.ndarray:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", RuntimeWarning)
        return np.nanmedian(a, axis=axis)


def _col_quantiles(R: np.ndarray, qs: Sequence[float]) -> List[np.ndarray]:
    """NaN-aware per-column quantiles (linear interpolation, numpy's default)
    by one sort: NaN sorts last, so the finite values of a column are its
    first n entries. All-NaN columns give NaN, without numpy's warnings. For
    the 4-row windows this is ~20x cheaper than nanpercentile."""
    fin = np.isfinite(R)
    S = np.sort(np.where(fin, R, np.nan), axis=0)          # +-inf count as missing
    n = np.sum(fin, axis=0)
    cols = np.arange(R.shape[1])
    out = []
    for q in qs:
        h = (np.maximum(n, 1) - 1) * float(q)
        lo = np.floor(h).astype(int)
        hi = np.minimum(lo + 1, np.maximum(n - 1, 0))
        a, b = S[lo, cols], S[hi, cols]
        v = a + (h - lo) * (b - a)
        v = np.where(h == lo, a, v)
        out.append(np.where(n > 0, v, np.nan))
    return out


def _col_nanmean(R: np.ndarray) -> np.ndarray:
    ok = np.isfinite(R)
    n = ok.sum(axis=0)
    s = np.where(ok, R, 0.0).sum(axis=0)
    return np.where(n > 0, s / np.maximum(n, 1), np.nan)


def window_vector(rows: Any) -> np.ndarray:
    """float32[146] window vector of k tick rows (k = K_WIN in the engine)."""
    R = np.asarray(rows, dtype=np.float64).reshape(-1, TICK_DIM)
    out = np.full(RAW_DIM, _NAN)
    if not R.shape[0]:
        return out.astype(np.float32)
    vec = R[:, T_VEC]
    out[W_MED] = _col_quantiles(vec, (0.5,))[0]
    q25, q75 = _col_quantiles(vec[:, list(IQR_IDX)], (0.25, 0.75))
    out[W_IQR] = q75 - q25
    out[W_SK] = _col_nanmean(R[:, T_SK])
    # the client.stack_set namespace block of the sketch is zeroed (integration
    # R20.2): client stacks are their own capped modality (B16's +-4 nats),
    # and leaving them in z too would let a browser upgrade move the gauss
    # block without bound
    out[W_SK_CLIENT] = 0.0
    out[W_TIM] = _col_quantiles(R[:, T_TIM], (0.5,))[0]
    clk = R[:, T_CLK]
    sc = _col_nanmean(clk[:, :2])
    nrm = math.hypot(sc[0], sc[1]) if np.all(np.isfinite(sc)) else _NAN
    if nrm > 1e-9:
        out[143:145] = sc / nrm
    out[145] = _col_nanmean(clk[:, 2:3])[0]
    return out.astype(np.float32)


def window_from_store(store, s: str, e: str, ts_list: Sequence[float]) -> Optional[np.ndarray]:
    """Window vector of the tick rows at ts_list (B16: the last K active ticks);
    None when no row is available."""
    rows = [r for r in (tick_row(store, s, e, float(t)) for t in ts_list) if r is not None]
    return window_vector(np.vstack(rows)) if rows else None


# ================================================================ transform
def _pca(model: Mapping) -> Mapping:
    return model.get("pca") or {}


def augment(model: Mapping, X: np.ndarray, class_keys: Optional[Sequence[Optional[str]]] = None
            ) -> np.ndarray:
    """Impute NaN (class fill, else system fill, else 0), append the missing
    mask of model['pca']['mask_cols'] and select the kept columns."""
    pc = _pca(model)
    X = np.array(X, dtype=np.float64).reshape(-1, RAW_DIM)
    fill = np.asarray(pc.get("fill", np.zeros(RAW_DIM)), dtype=np.float64)
    fcls = pc.get("fill_cls") or {}
    mcols = [int(c) for c in pc.get("mask_cols") or []]
    miss = ~np.isfinite(X)
    mask = miss[:, mcols].astype(np.float64) if mcols else np.zeros((X.shape[0], 0))
    if miss.any():
        Fm = np.repeat(fill.reshape(1, -1), X.shape[0], axis=0)
        if class_keys is not None and fcls:
            ks = np.asarray([k if k is not None else "" for k in class_keys], dtype=object)
            for ck, f in fcls.items():
                sel = ks == ck
                if sel.any():
                    Fm[sel] = np.asarray(f, dtype=np.float64)
        X = np.where(miss, Fm, X)
        X[~np.isfinite(X)] = 0.0
    A = np.hstack([X, mask])
    keep = pc.get("keep")
    return A[:, [int(k) for k in keep]] if keep is not None else A


def transform_many(model: Any, X: Any, class_keys: Optional[Sequence[Optional[str]]] = None
                   ) -> np.ndarray:
    """[n, r] LDA-space coordinates of raw window vectors (NaN when unfitted)."""
    X = np.asarray(X, dtype=np.float64).reshape(-1, RAW_DIM)
    if not is_fitted(model):
        return np.full((X.shape[0], 0), _NAN)
    pc = _pca(model)
    A = augment(model, X, class_keys)
    U = (A - np.asarray(pc["center"])) / np.asarray(pc["scale"])
    return U @ np.asarray(pc["P"], dtype=np.float64)


def transform(model: Any, x: Any, class_key: Optional[str] = None) -> np.ndarray:
    """LDA-space coordinates of one raw window vector (float64[r])."""
    z = transform_many(model, np.asarray(x, dtype=np.float64).reshape(1, -1),
                       [class_key] if class_key is not None else None)
    return z[0] if z.shape[0] else np.full(0, _NAN)


# ============================================================ gauss scores
def entity_mean(model: Any, e: str) -> Optional[np.ndarray]:
    m = (model.get("means") or {}).get(e) if isinstance(model, Mapping) else None
    return np.asarray(m, dtype=np.float64) if m is not None else None


def class_mean(model: Any, ck: str) -> Optional[np.ndarray]:
    m = (model.get("class_means") or {}).get(ck) if isinstance(model, Mapping) else None
    return np.asarray(m, dtype=np.float64) if m is not None else None


def bg_loglik(model: Any, z: Any) -> float:
    """Log-density of z under the pooled 'somebody in this system' Gaussian."""
    bg = model.get("bg") if isinstance(model, Mapping) else None
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if not bg or not z.size or not np.all(np.isfinite(z)):
        return _NAN
    d = z - np.asarray(bg["mu"])
    q = float(d @ np.asarray(bg["prec"]) @ d)
    return -0.5 * (q + float(bg["logdet"]) + z.size * _LN2PI)


def gauss_loglik(model: Any, z: Any, cand: str) -> float:
    """l_cand(z) in nats: an entity (identity within covariance) or a class
    key (diagonal covariance of its members' windows). NaN if unknown."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    if not z.size or not np.all(np.isfinite(z)):
        return _NAN
    m = entity_mean(model, cand)
    if m is not None and m.size == z.size:
        return -0.5 * (float(np.sum((z - m) ** 2)) + z.size * _LN2PI)
    m = class_mean(model, cand)
    if m is not None and m.size == z.size:
        v = np.maximum(np.asarray((model.get("class_var") or {}).get(cand, np.ones(z.size)),
                                  dtype=np.float64), 1e-6)
        return -0.5 * (float(np.sum((z - m) ** 2 / v)) + float(np.sum(np.log(v)))
                       + z.size * _LN2PI)
    return _NAN


def gauss_llr(model: Any, z: Any, cand: str) -> float:
    return gauss_loglik(model, z, cand) - bg_loglik(model, z)


def scores(model: Any, z: Any) -> Dict[str, float]:
    """{entity: l_e(z)} for every fitted entity (vectorised)."""
    z = np.asarray(z, dtype=np.float64).reshape(-1)
    ms = model.get("means") if isinstance(model, Mapping) else None
    if not ms or not z.size or not np.all(np.isfinite(z)):
        return {}
    names = list(ms.keys())
    M = np.asarray([ms[n] for n in names], dtype=np.float64)
    if M.shape[1] != z.size:
        return {}
    ll = -0.5 * (np.sum((M - z) ** 2, axis=1) + z.size * _LN2PI)
    return dict(zip(names, ll.tolist()))


def top_k(model: Any, z: Any, k: int = 5, exclude: Iterable[str] = ()) -> List[Tuple[str, float]]:
    ex = set(exclude)
    sc = [(e, v) for e, v in scores(model, z).items() if e not in ex]
    sc.sort(key=lambda t: (-t[1], t[0]))
    return sc[:max(0, int(k))]


# ============================================================= calibration
def llr_calib(model: Any, m: str) -> Tuple[float, float]:
    c = (model.get("llr_calib") or {}).get(m) if isinstance(model, Mapping) else None
    if c is None or len(c) != 2:
        return DEFAULT_CALIB
    a, b = _f(c[0]), _f(c[1])
    return (a, b) if math.isfinite(a) and math.isfinite(b) else DEFAULT_CALIB


def calibrate(model: Any, m: str, llr: float, cap: float = LLR_CAP) -> float:
    """clip(a_m * llr + b_m, -cap, cap); NaN stays NaN (no evidence)."""
    x = _f(llr)
    if not math.isfinite(x):
        return _NAN
    a, b = llr_calib(model, m)
    return float(min(cap, max(-cap, a * x + b)))


def logistic_calibration(llr: Any, label: Any, ridge: float = 1e-3, iters: int = 50,
                         min_n: int = MIN_CAL) -> Tuple[float, float]:
    """(a, b) of P(genuine | llr) = sigmoid(a llr + b), each label weighted to
    total 0.5 (prior-free, so a llr + b is a calibrated LLR). Newton / IRLS
    with a small ridge on a. a is floored at 0 (a modality that anti-predicts
    identity is ignored, never inverted). DEFAULT_CALIB below min_n samples of
    either label."""
    x = np.asarray(llr, dtype=np.float64).reshape(-1)
    y = np.asarray(label, dtype=np.float64).reshape(-1)
    ok = np.isfinite(x) & np.isfinite(y)
    x, y = x[ok], (y[ok] > 0.5).astype(np.float64)
    n1, n0 = float(y.sum()), float((1.0 - y).sum())
    if n1 < min_n or n0 < min_n:
        return DEFAULT_CALIB
    w = np.where(y > 0.5, 0.5 / n1, 0.5 / n0)
    sd = float(np.std(x)) or 1.0
    xs = x / sd                                   # conditioning; undone below
    a, b = 0.0, 0.0
    for _ in range(iters):
        t = np.clip(a * xs + b, -30.0, 30.0)
        p = 1.0 / (1.0 + np.exp(-t))
        g_a = float(np.sum(w * (p - y) * xs)) + ridge * a
        g_b = float(np.sum(w * (p - y)))
        h = w * p * (1.0 - p)
        H = np.array([[float(np.sum(h * xs * xs)) + ridge, float(np.sum(h * xs))],
                      [float(np.sum(h * xs)), float(np.sum(h)) + 1e-12]])
        try:
            da, db = np.linalg.solve(H, [g_a, g_b])
        except np.linalg.LinAlgError:
            break
        a, b = a - da, b - db
        if abs(da) + abs(db) < 1e-10:
            break
    a = a / sd
    if not (math.isfinite(a) and math.isfinite(b)):
        return DEFAULT_CALIB
    if a < 0.0:
        a = 0.0
        b = 0.0
    return float(a), float(b)


# ========================================================== CV statistics
def eer(genuine: Any, impostor: Any) -> float:
    """Equal error rate of 'higher score = genuine': the crossing of
    FRR(t) = P(g < t) and FAR(t) = P(i >= t), linearly interpolated between
    adjacent thresholds. NaN when either side is empty."""
    g = np.sort(np.asarray(genuine, dtype=np.float64)[np.isfinite(genuine)])
    im = np.sort(np.asarray(impostor, dtype=np.float64)[np.isfinite(impostor)])
    if not g.size or not im.size:
        return _NAN
    t = np.unique(np.concatenate([g, im]))
    t = np.concatenate([t, [np.inf]])
    frr = np.searchsorted(g, t, side="left") / g.size
    far = 1.0 - np.searchsorted(im, t, side="left") / im.size
    d = frr - far                                  # non-decreasing in t
    k = int(np.searchsorted(d, 0.0, side="left"))
    if k == 0:
        return float((frr[0] + far[0]) / 2.0)
    if k >= d.size:
        return float((frr[-1] + far[-1]) / 2.0)
    d0, d1 = d[k - 1], d[k]
    u = (0.0 - d0) / (d1 - d0) if d1 != d0 else 0.5
    e0 = (frr[k - 1] + far[k - 1]) / 2.0
    e1 = (frr[k] + far[k]) / 2.0
    return float(e0 + u * (e1 - e0))


def union_find_sets(nodes: Sequence[str], pairs: Iterable[Tuple[str, str]]) -> List[List[str]]:
    """Connected components (size >= 2) of the pair graph, sorted."""
    parent = {n: n for n in nodes}

    def find(a: str) -> str:
        while parent[a] != a:
            parent[a] = parent[parent[a]]
            a = parent[a]
        return a

    for a, b in pairs:
        if a in parent and b in parent:
            ra, rb = find(a), find(b)
            if ra != rb:
                parent[max(ra, rb)] = min(ra, rb)
    comp: Dict[str, List[str]] = {}
    for n in nodes:
        comp.setdefault(find(n), []).append(n)
    return sorted((sorted(c) for c in comp.values() if len(c) >= 2), key=lambda c: c[0])


def bhattacharyya_diag(m1: Any, v1: Any, m2: Any, v2: Any, floor: float = 1e-6) -> float:
    """Bhattacharyya distance of two diagonal Gaussians."""
    m1, m2 = np.asarray(m1, dtype=np.float64), np.asarray(m2, dtype=np.float64)
    v1 = np.maximum(np.asarray(v1, dtype=np.float64), floor)
    v2 = np.maximum(np.asarray(v2, dtype=np.float64), floor)
    v = 0.5 * (v1 + v2)
    return float(0.125 * np.sum((m1 - m2) ** 2 / v)
                 + 0.5 * np.sum(np.log(v) - 0.5 * (np.log(v1) + np.log(v2))))


def mcq_log_odds(y_self: Mapping[str, float], y_peer: Mapping[str, float],
                 prior: Mapping[str, float], alpha0: float = 50.0
                 ) -> Dict[str, Tuple[float, float]]:
    """Monroe-Colaresi-Quinn weighted log-odds with an informative Dirichlet
    prior (alpha_w = alpha0 * prior share of w): {w: (delta, z)}, z > 0 means
    w is over-used by `self` relative to its peers."""
    ns, npr = float(sum(y_self.values())), float(sum(y_peer.values()))
    pt = float(sum(prior.values()))
    if not (ns > 0.0 and npr > 0.0 and pt > 0.0):
        return {}
    a0 = float(alpha0)
    out: Dict[str, Tuple[float, float]] = {}
    for w in set(y_self) | set(y_peer):
        aw = a0 * max(float(prior.get(w, 0.0)), 0.0) / pt + 1e-3
        ys, yp = float(y_self.get(w, 0.0)), float(y_peer.get(w, 0.0))
        ls = math.log((ys + aw) / max(ns + a0 - ys - aw, 1e-9))
        lp = math.log((yp + aw) / max(npr + a0 - yp - aw, 1e-9))
        var = 1.0 / (ys + aw) + 1.0 / (yp + aw)
        out[w] = (ls - lp, (ls - lp) / math.sqrt(var))
    return out


def to_natural(idx: int, v: float) -> float:
    """Inverse FEATURE_SPEC vec transform of one value: per-minute rate for
    counts / bytes, share for ratios / bounded, the average itself for avg,
    identity for gauge / window / clr (req_per_session: log1p inverted)."""
    x = _f(v)
    if not math.isfinite(x):
        return _NAN
    name = F.FEATURE_NAMES_V2[int(idx)]
    kind, tx = F.FEATURE_KIND[name], F.VEC_TX.get(name)
    if tx == "identity":
        return x
    if tx == "log1p":
        return math.expm1(x)
    if kind in ("count", "bytes"):
        return math.expm1(x)
    if kind in ("ratio", "bounded"):
        return 1.0 / (1.0 + math.exp(-min(50.0, max(-50.0, x))))
    if kind == "avg":
        return math.exp(min(700.0, x))
    return x


# ============================================================= per entity
def _stats(model: Any) -> Mapping[str, Any]:
    return (model.get("stats") or {}) if isinstance(model, Mapping) else {}


def stats(model: Any, e: str) -> Dict[str, Any]:
    return dict(_stats(model).get(e) or {})


def t99(model: Any, e: str) -> float:
    return _f(stats(model, e).get("t99"))


def eer_hard(model: Any, e: str) -> float:
    return _f(stats(model, e).get("eer_hard"))


def separability(model: Any, e: str) -> float:
    return _f(stats(model, e).get("separability"))


def nearest(model: Any, e: str) -> List[str]:
    return list(stats(model, e).get("near") or [])


def confusion(model: Any, e: str, j: Optional[str] = None) -> Any:
    """Row {j: share of e's held-out windows attributed to j}, or one share
    (0.0 for a pair never confused; NaN when e was not scored)."""
    conf = model.get("confusion") if isinstance(model, Mapping) else None
    row = (conf or {}).get(e)
    if j is None:
        return dict(row or {})
    if row is None:
        return _NAN
    return float(row.get(j, 0.0))


def confusable_with(model: Any, e: str) -> List[str]:
    return list(stats(model, e).get("confusable_with") or [])


def anonymity_sets(model: Any) -> List[List[str]]:
    return [list(x) for x in (model.get("anonymity_sets") or [])] \
        if isinstance(model, Mapping) else []


def anonymity_set(model: Any, e: str) -> List[str]:
    for s_ in anonymity_sets(model):
        if e in s_:
            return s_
    return [e]


def class_stats(model: Any, ck: str) -> Dict[str, Any]:
    return dict(((model.get("classes") or {}).get(ck) or {})) if isinstance(model, Mapping) else {}


def distinctive(model: Any, e: str) -> Dict[str, Any]:
    d = (model.get("distinctive") or {}).get(e) if isinstance(model, Mapping) else None
    return dict(d or {})


def modality_share(model: Any, e: str) -> Dict[str, float]:
    d = (model.get("modality_share") or {}).get(e) if isinstance(model, Mapping) else None
    return dict(d or {})


def descriptors(model: Any, e: str) -> Dict[str, Any]:
    """profile.extra.identity (contract G) for one entity; {} when unscored."""
    st = stats(model, e)
    if not st:
        return {}
    return {
        "recall1": st.get("recall1"), "recallK": st.get("recallK"),
        "eer_hard": st.get("eer_hard"), "t99": st.get("t99"),
        "separability": st.get("separability"),
        "confusable_with": list(st.get("confusable_with") or []),
        "anonymity_set": anonymity_set(model, e),
        "distinctive": distinctive(model, e),
        "modality_share": modality_share(model, e),
        "n_windows": st.get("n_windows"),
        "version": version(model), "fitted_ts": model.get("fitted_ts"),
    }


# ======================================================= modality scoring
@dataclass
class ModalData:
    """Raw evidence of one tick (or a window of ticks) for the non-gauss
    modalities. Fields left empty carry no evidence (NaN LLR)."""
    vocab: Dict[str, Dict[str, float]] = field(default_factory=dict)
    tokens: List[str] = field(default_factory=list)
    gaps: np.ndarray = field(default_factory=lambda: np.zeros(0))
    stacks: Dict[str, float] = field(default_factory=dict)
    cells: List[Tuple[int, int]] = field(default_factory=list)   # active (c48, c168)

    def extend(self, other: "ModalData") -> None:
        for dim, vals in other.vocab.items():
            d = self.vocab.setdefault(dim, {})
            for v, n in vals.items():
                d[v] = d.get(v, 0.0) + n
        self.tokens.extend(other.tokens)
        self.gaps = np.concatenate([self.gaps, other.gaps])
        for t, n in other.stacks.items():
            self.stacks[t] = self.stacks.get(t, 0.0) + n
        self.cells.extend(other.cells)


def _pos(n: Any) -> float:
    if isinstance(n, Mapping):
        n = n.get("n", 0.0)
    if isinstance(n, bool):
        return 0.0
    v = _f(n)
    return v if (v > 0.0 and math.isfinite(v)) else 0.0


def tick_modal_data(store, s: str, e: str, ts: float, *, tctx: Optional[Mapping] = None,
                    smap: Optional[m_seq.SymbolMap] = None) -> ModalData:
    """The raw modality evidence written at tick ts (act.tokens, TLS/DNS/port
    sets, act.stream, client.stack_set; raw sets are retained 1 h, so call it
    at or shortly after ts). tctx gives the rhythm cell (looked up at ts when
    not passed)."""
    md = ModalData()

    def put(dim: str, v: Any, n: Any) -> None:
        c = _pos(n)
        if c > 0.0 and v is not None and str(v) and str(v) != "__other__":
            d = md.vocab.setdefault(dim, {})
            d[str(v)] = d.get(str(v), 0.0) + c

    def fresh(name: str) -> Any:
        m = store.latest_raw_at(s, e, name, ts)
        return m.value if m is not None else None

    toks = fresh("act.tokens")
    if isinstance(toks, Mapping):
        for t, n in toks.items():
            if isinstance(t, str) and t and t[0] != "{" and t != "__other__" \
                    and channel_of(t) == "http":
                put("tmpl", m_template.template_key(t), n)
    for dim, name in (("sni", "tls.sni_etld1_set"), ("dns", "dns.qname_etld1_set"),
                      ("dport", "l4.dport_set")):
        x = fresh(name)
        if isinstance(x, Mapping):
            for v, n in x.items():
                put(dim, v, n)
    if not md.vocab.get("sni"):
        x = fresh("tls.sni_set")
        if isinstance(x, Mapping):
            for v, n in x.items():
                if isinstance(v, str) and v != "__other__":
                    put("sni", etld1(v), n)
    t, syms, _fam, _auth = m_seq.stream_symbols(store, s, e, ts, smap)
    md.tokens = list(syms[:SEQ_MAX_TOKENS])
    if t.size >= 2:
        frac = m_template.stream_frac(store, s, e, ts)
        md.gaps = m_timing.gaps_from_times(t, frac if math.isfinite(frac) else 1.0)
    md.stacks = m_client.stack_counts(fresh("client.stack_set"))
    tc = tctx if tctx is not None else _dict_at(store, s, e, "feature.tctx", ts, 4)
    if isinstance(tc, Mapping):
        try:
            md.cells.append(m_rhythm.cells_of_tctx(tc))
        except (KeyError, TypeError, ValueError):
            pass
    return md


class Background:
    """Per-(system, tick) cache of the background tiers every candidate's
    modality log-likelihood is compared against, plus per-candidate model
    lookups. Build one per tick and pass it to modality_logliks / _llrs."""

    def __init__(self, store, s: str, now: float) -> None:
        self.store, self.s, self.now = store, s, float(now)
        self.vocab_sys = m_vocab.get(store, s, SYSTEM_KEY)
        self.seq_sys = m_seq.get(store, s, SYSTEM_KEY)
        self.seq_V = m_seq.vocab_size(store, s)
        self.client_sys = m_client.get(store, s, SYSTEM_KEY)
        self._rhythm_sys = store.get_model(s, SYSTEM_KEY, m_rhythm.MODEL)
        self._timing_pmf: Optional[np.ndarray] = None
        self._rhythm_models: Optional[List[Mapping]] = None
        self._p_bg: Dict[Tuple[int, int], float] = {}
        self._cache: Dict[Tuple[str, str], Any] = {}

    def _model(self, name: str, cand: str) -> Any:
        key = (name, cand)
        if key not in self._cache:
            self._cache[key] = self.store.get_model(self.s, cand, name)
        return self._cache[key]

    def timing_pmf(self) -> np.ndarray:
        if self._timing_pmf is None:
            acc, n = np.zeros(m_timing.N_BINS), 0
            for e in self.store.entities(self.s):
                p = m_timing.pmf(self._model(m_timing.MODEL, e))
                if np.all(np.isfinite(p)):
                    acc += p
                    n += 1
            self._timing_pmf = acc / n if n else np.full(m_timing.N_BINS, _NAN)
        return self._timing_pmf

    def rhythm_p(self, c48: int, c168: int) -> float:
        key = (int(c48), int(c168))
        hit = self._p_bg.get(key)
        if hit is not None:
            return hit
        p = _NAN
        if isinstance(self._rhythm_sys, Mapping) and "state" in self._rhythm_sys:
            p = m_rhythm.p_cell(self._rhythm_sys, *key)
        if not math.isfinite(p):
            if self._rhythm_models is None:
                self._rhythm_models = [
                    m for m in (self._model(m_rhythm.MODEL, e) for e in self.store.entities(self.s))
                    if isinstance(m, Mapping) and m.get("kind") == "entity" and "state" in m]
            ps = [m_rhythm.p_cell(m, *key) for m in self._rhythm_models]
            ps = [x for x in ps if math.isfinite(x)]
            p = float(np.mean(ps)) if ps else _NAN
        self._p_bg[key] = p
        return p


def _rhythm_ll(model: Any, cells: Sequence[Tuple[int, int]]) -> float:
    if not (isinstance(model, Mapping) and "state" in model) or not cells:
        return _NAN
    return m_rhythm.loglik(model, [1.0] * len(cells), list(cells))


def vocab_factor(data: ModalData) -> float:
    """min(n_tok, 20) / 20 over the window's vocabulary tokens (engines.md
    B16): a few tokens carry little vocabulary evidence."""
    n_tok = sum(float(c) for vals in data.vocab.values() for c in vals.values())
    return min(n_tok, VOCAB_N) / VOCAB_N


def modality_logliks(store, s: str, cand: str, data: ModalData, now: float,
                     bg: Optional[Background] = None) -> Dict[str, float]:
    """Per-modality LLR (nats) of `data` under candidate `cand`'s own models
    against the background tiers; NaN = no evidence for that modality (no
    data, or no model at any tier). `cand` may be an entity or a class key.

    The vocab LLR already carries vocab_factor(data) (integration R20.1), so
    B15's llr_calib['vocab'] is fitted on the same scaled LLR that B16 and
    B17 calibrate."""
    bg = bg if bg is not None else Background(store, s, now)
    out = {m: _NAN for m in MODALITIES if m != "gauss"}
    if data.vocab:
        ent, cls, sys_ = m_vocab.backoff_models(store, s, cand)
        if ent is not None or cls is not None or sys_ is not None:
            out["vocab"] = (m_vocab.loglik_models(ent, cls, sys_, data.vocab, now)
                            - m_vocab.loglik_models(None, None, bg.vocab_sys, data.vocab, now)
                            ) * vocab_factor(data)
    if data.cells:
        ll = _rhythm_ll(bg._model(m_rhythm.MODEL, cand), data.cells)
        pb = [bg.rhythm_p(*c) for c in data.cells]
        if math.isfinite(ll) and all(math.isfinite(p) for p in pb):
            lb = sum(math.log(min(1.0 - 1e-6, max(1e-6, p))) for p in pb)
            out["rhythm"] = ll - lb
    if data.tokens:
        own = bg._model(m_seq.MODEL, cand)
        tiers = m_seq.backoff(store, s, cand) if cand != SYSTEM_KEY else []
        if m_seq.ppm(own) is not None or tiers:
            b_c = m_seq.loglik(own, data.tokens, tiers, bg.seq_V)
            b_0 = m_seq.loglik(bg.seq_sys, data.tokens, (), bg.seq_V)
            out["seq"] = float(np.nansum(b_0) - np.nansum(b_c)) * _LN2
    if data.stacks:
        ent, cls, sys_ = m_client.backoff_models(store, s, cand)
        if ent is not None or cls is not None or sys_ is not None:
            out["client"] = (m_client.loglik(ent, data.stacks, sys_, cls, now)
                             - m_client.loglik(None, data.stacks, bg.client_sys, None, now))
    if data.gaps.size:
        ll = m_timing.loglik(bg._model(m_timing.MODEL, cand), data.gaps)
        p0 = bg.timing_pmf()
        if math.isfinite(ll) and np.all(np.isfinite(p0)):
            b = m_timing.bin_index(data.gaps)
            b = b[b >= 0]
            out["timing"] = ll - float(np.sum(np.log(p0[b]))) if b.size else _NAN
    return out


def modality_llrs(store, s: str, cands: Sequence[str], data: ModalData, now: float,
                  bg: Optional[Background] = None) -> Dict[str, Dict[str, float]]:
    bg = bg if bg is not None else Background(store, s, now)
    return {c: modality_logliks(store, s, c, data, now, bg) for c in cands}
