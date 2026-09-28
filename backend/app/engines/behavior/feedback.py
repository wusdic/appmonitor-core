"""FeedbackEngine (B23) — the analyst loop.

Why: the statistical layers are calibrated against the NULL (how rare is
this for the entity), not against what analysts care about (is it
malicious). Verdicts close that gap without touching calibration:

  * precision per (family, contributor) ~ Beta(1 + TP, 1 + FP) gives B26 a
    bounded risk multiplier pi = clip(E[prec] / 0.5, 0.2, 1) (feedback lowers a
    family's weight, never raises it: lib/m_feedback.PREC_CLIP);
  * after >= 20 labelled incidents a stacking logistic regression on the
    per-family excess surprise re-weights the families in fusion (B25), with
    an L2 penalty toward UNIFORM weights (lambda = 5) so a handful of noisy
    labels cannot switch a family off; isotonic calibration of P(malicious)
    once >= 100 labels (fitted on out-of-fold scores, not in-sample);
  * fp / benign_known labels with a widened scope become pattern policies
    (token Jaccard >= 0.6 AND a level headroom of one decade AND a TTL), so a
    known-benign recurrence is suppressed but a rarer or differently-shaped
    one escapes; benign_known also allowlists its new values for B08 / B12;
  * expected_change -> accept, tp -> freeze, consumed by the governor (B28);
  * a per-system alert budget alpha_mult rescales the e_day SEVERITY
    thresholds only (never the p-values);
  * an active-learning label queue (<= 5 a day: 80 % highest risk, 20 % most
    uncertain, plus every incident held > 14 d).

Learning signal. B23 learns from analyst verdicts, which are ground truth
supplied from outside, not from the entity's own traffic; it is therefore not
a contract-H learner (trust / delay / quarantine gating protects models from
learning an attacker's ticks and does not apply to labels). Its per-incident
evidence snapshots are taken while incidents are live, because behavior.z
(6 h) and behavior.p / p_family (1 d) are pruned long before an analyst
labels a case.

Timing within a tick: B23 (order 23) runs before B24-B27, so rows those
engines write at `now` are folded on the next tick; fold windows are
half-open [lo, now) so each row is folded exactly once.

Everything written lives in model.feedback@('__org__', '__org__') (read it
through lib/m_feedback.py) plus profile.extra.feedback of labelled entities.
B23 emits no events, in training or live.
"""
from __future__ import annotations

import math
from collections import Counter
from collections.abc import Mapping
from typing import Any, Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np
from scipy.special import expit

from ...core.engine import Context, Engine
from ...models.schema import EntityProfile, Incident, Label
from .lib import m_class
from .lib import m_feedback as FB
from .lib import m_governor as MG
from .lib.classkeys import ORG, SYSTEM_KEY, is_class
from .lib.combine import seeded_uniform
from .lib.detectors import DETECTORS, FAMILIES, family_of

MODEL = FB.MODEL
P_FAMILY = "behavior.p_family"
P_VEC = "behavior.p"
E_DAY = "behavior.e_day"
Z_VEC = "behavior.z"

DAY = 86400.0
HOUR = 3600.0

# ---- learning
LAMBDA = 5.0                    # L2 toward uniform family weights
MIN_STACK = 20                  # labelled incidents before the stacker is used
MIN_ISO = 100                   # labels before isotonic calibration
ISO_FOLDS = 5
C0 = math.log(math.e - 1.0)     # softplus(C0) = 1: the uniform point of a family coefficient
Y_OF = {"tp": 1.0, "fp": 0.0, "benign_known": 0.0, "expected_change": 0.0}
# (TP, FP) credit per verdict. expected_change was a true detection of a
# change that turned out legitimate: half an FP for "is it malicious" precision.
PREC_CREDIT = {"tp": (1.0, 0.0), "fp": (0.0, 1.0), "benign_known": (0.0, 1.0),
               "expected_change": (0.0, 0.5)}

# ---- policies, records, cases
POLICY_TTL_S = 14 * DAY
MAX_POLICIES = 500
RECORD_KEEP_S = 30 * DAY        # accept / freeze records
MAX_RECORDS = 16                # per (system, entity) key
CASE_KEEP_S = 30 * DAY          # unlabelled case snapshots after their last activity
MAX_UNLABELLED = 1000
MAX_LABELLED = 2000
Z_KEEP = 12                     # features kept per case snapshot (tokens use the top 5)
MAX_FOLD_ROWS = 1500
SWEEP_EVERY_S = HOUR

# ---- alert budget
ALPHA_GAIN = 0.1
ALPHA_PERIOD_S = DAY
ALPHA_MAX_DEFAULT = 1.0         # see _update_alpha: raising sensitivity is opt-in

# ---- label queue
QUEUE_PER_DAY = 5
QUEUE_UNCERTAIN_FRAC = 0.2
QUEUE_PERIOD_S = DAY
QUEUE_LOOKBACK_S = 7 * DAY
HELD_S = 14 * DAY
HELD_SCAN_S = HOUR
QUEUE_MAX = 50

_FAM = frozenset(FAMILIES)
_DET = frozenset(DETECTORS)


def _f(x: Any) -> Optional[float]:
    """float(x) when finite, else None."""
    try:
        v = float(x)
    except (TypeError, ValueError):
        return None
    return v if math.isfinite(v) else None


def _sev(x: Any) -> str:
    v = getattr(x, "value", x)
    return str(v).lower() if v is not None else ""


# =============================================================== stacker maths
def fit_stacker(X: np.ndarray, y: np.ndarray, lam: float = LAMBDA,
                max_iter: int = 100) -> np.ndarray:
    """Stacking logistic regression with an L2 penalty toward UNIFORM family
    weights; returns theta = [b0, beta_family (12), gamma_stages, gamma_sig].

    Objective (NLL summed over labelled cases):
        NLL + lam/2 * sum_f (softplus(beta_f) - 1)^2 + lam/2 * |gamma|^2
            + lam/2 * (b0 - logit(ybar))^2,     ybar = (sum y + 0.5) / (n + 1).
    * The penalty is on the WEIGHT scale family_w = softplus(beta), whose
      uniform point is 1 (beta = C0). It is quadratic near uniform (a few
      noisy labels barely move a family) and bounded by lam/2 as a weight goes
      to 0, so a family that is consistently fp can be down-weighted. Measured
      on synthetic label sets: a family fp in all of its 12 appearances among
      30 labels ends at 0.40 x uniform; the same penalty on beta itself
      leaves it at 0.72 x.
    * Incidents never carry zero evidence, so a free intercept is confounded
      with the common level of the betas (measured: b0 = -11.8 with every
      beta near 2.2, which flattens relative weights). The intercept is
      therefore anchored at the empirical label base rate ("no excess
      surprise anywhere -> malicious at the labelled rate") with the same lam.
    Newton steps with the Gauss-Newton (PSD) curvature of the penalty and a
    backtracking line search on the exact objective: deterministic.
    """
    X = np.asarray(X, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    n, p = X.shape
    nf = len(FAMILIES)
    k = p + 1
    ybar = (float(y.sum()) + 0.5) / (n + 1.0)
    b_ref = math.log(ybar / (1.0 - ybar))
    A = np.hstack([np.ones((n, 1)), X])
    fam = np.arange(1, nf + 1)
    rest = np.arange(nf + 1, k)

    def obj(th: np.ndarray) -> float:
        z = A @ th
        sp = np.logaddexp(0.0, th[fam])
        return float(np.sum(np.logaddexp(0.0, z) - y * z)
                     + 0.5 * lam * (np.sum((sp - 1.0) ** 2) + np.sum(th[rest] ** 2)
                                    + (th[0] - b_ref) ** 2))

    theta = np.zeros(k)
    theta[0] = b_ref
    theta[fam] = C0
    f0 = obj(theta)
    for _ in range(max_iter):
        mu = expit(A @ theta)
        g = A.T @ (mu - y)
        H = (A * (mu * (1.0 - mu))[:, None]).T @ A
        sp = np.logaddexp(0.0, theta[fam])
        sg = expit(theta[fam])
        g[fam] += lam * (sp - 1.0) * sg
        H[fam, fam] += lam * sg * sg
        g[rest] += lam * theta[rest]
        H[rest, rest] += lam
        g[0] += lam * (theta[0] - b_ref)
        H[0, 0] += lam
        H[np.diag_indices(k)] += 1e-9
        step = np.linalg.solve(H, g)
        t = 1.0
        while True:
            cand = theta - t * step
            f1 = obj(cand)
            if f1 <= f0 + 1e-12 or t < 1e-8:
                break
            t *= 0.5
        theta, f0 = cand, f1
        if float(np.max(np.abs(t * step))) < 1e-9:
            break
    return theta


def softplus(x: np.ndarray) -> np.ndarray:
    return np.logaddexp(0.0, x)


def family_weights_from(theta: np.ndarray) -> Dict[str, float]:
    """family_w = softplus(beta) / mean(softplus(beta)), floored at W_FLOOR."""
    beta = theta[1:len(FAMILIES) + 1]
    sp = softplus(beta)
    m = float(sp.mean())
    w = sp / m if m > 0 else np.ones_like(sp)
    return {f: max(FB.W_FLOOR, float(w[i])) for i, f in enumerate(FAMILIES)}


def pav(x: np.ndarray, y: np.ndarray) -> Tuple[List[float], List[float]]:
    """Pool-adjacent-violators on (x, y), ties in x pooled first. Returns the
    block x-means (strictly increasing) and block y-means (non-decreasing)."""
    x = np.asarray(x, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    ux, inv = np.unique(x, return_inverse=True)
    sw = np.bincount(inv).astype(np.float64)
    sy = np.bincount(inv, weights=y)
    sx = ux * sw
    blocks: List[List[float]] = []            # [sum_y, sum_w, sum_x]
    for i in range(len(ux)):
        blocks.append([sy[i], sw[i], sx[i]])
        while len(blocks) > 1 and blocks[-2][0] / blocks[-2][1] >= blocks[-1][0] / blocks[-1][1]:
            b = blocks.pop()
            blocks[-1][0] += b[0]
            blocks[-1][1] += b[1]
            blocks[-1][2] += b[2]
    return [b[2] / b[1] for b in blocks], [b[0] / b[1] for b in blocks]


def _design(theta: np.ndarray, X: np.ndarray) -> np.ndarray:
    return expit(theta[0] + X @ theta[1:X.shape[1] + 1])


# ====================================================================== engine
class FeedbackEngine(Engine):
    name = "behavior.feedback"
    layer = "behavior"
    consumes = ["store.labels", "store.incidents", "store.events", "store.matches",
                "behavior.p_family", "behavior.p", "behavior.e_day", "behavior.z", "model.class"]
    produces = ["model.feedback", "profile.extra.feedback"]
    description = ("Analyst loop: Beta precision per family/detector, stacked family weights "
                   "(L2 toward uniform, isotonic P(malicious)), pattern suppression policies, "
                   "allowlist, accept/freeze, alert-budget alpha_mult and the label queue.")
    interval = 1

    def __init__(self, **params: Any) -> None:
        super().__init__(**params)
        self.lam = float(params.get("lam", LAMBDA))
        self.min_stack = int(params.get("min_stack", MIN_STACK))
        self.min_iso = int(params.get("min_iso", MIN_ISO))
        self.queue_per_day = int(params.get("queue_per_day", QUEUE_PER_DAY))
        self._seen: Optional[set] = None
        self._seen_for: Optional[int] = None
        self._tick_systems: set = set()
        self._profile_todo: List[Tuple[str, str, Label]] = []

    # ------------------------------------------------------------------ run
    def run(self, ctx: Context, observations: Optional[List] = None) -> int:
        store = ctx.store
        now = float(ctx.now)
        dt = float(ctx.window_s)
        model = self._load(store)
        st = model["_state"]
        last = st.get("last_run")
        if last is not None and now < last:           # clock went back: new run / replay
            last = None
            st.update(alpha_ts={}, queue_ts=None, held_ts=None, sweep_ts=None)
        changed = False

        n = self._fold_live(store, model, last, now, dt)

        labels = self._new_labels(store, model)
        relearn = False
        self._profile_todo = []
        for lb in labels:
            relearn |= self._apply_label(ctx, model, lb)
        if labels:
            changed = True
        if relearn:
            self._relearn(model, now)
        # profile.extra.feedback after the refit: its summary (n_labelled,
        # stacker, isotonic) must describe the model this label produced
        for system, entity, lb in self._profile_todo:
            self._write_profile(store, model, system, entity, lb, now)
        self._profile_todo = []
        n += len(labels)

        if st.get("sweep_ts") is None or now - st["sweep_ts"] >= SWEEP_EVERY_S:
            st["sweep_ts"] = now
            changed |= self._sweep(model, now)

        changed |= self._update_alpha(ctx, model, now)
        changed |= self._update_queue(ctx, model, now)

        if changed:
            model["version"] = int(model.get("version", 0)) + 1
        st["last_run"] = now
        store.put_model(ORG[0], ORG[1], MODEL, model, version=model["version"], ts=now)
        return n

    # ---------------------------------------------------------------- model
    def _load(self, store) -> Dict[str, Any]:
        m = store.get_model(ORG[0], ORG[1], MODEL, default=None)
        if not isinstance(m, dict):
            m = {}
        m.setdefault("version", 0)
        m.setdefault("family_w", {f: 1.0 for f in FAMILIES})
        for key, default in (("detector_prec", {}), ("policies", []), ("allowlist", {}),
                             ("alpha_mult", {}), ("accept", {}), ("freeze", {}), ("queue", []),
                             ("_cases", {})):
            if not isinstance(m.get(key), type(default)):
                m[key] = default
        m.setdefault("stacker", None)
        m.setdefault("isotonic", None)
        m.setdefault("n_labelled", 0)
        st = m.get("_state")
        if not isinstance(st, dict):
            st = m["_state"] = {}
        for key, default in (("last_run", None), ("labels_seen", []), ("alpha_ts", {}),
                             ("queue_ts", None), ("held_ts", None), ("sweep_ts", None),
                             ("policy_seq", 0), ("rec_seq", 0)):
            st.setdefault(key, default)
        if self._seen_for != id(m):
            self._seen = set(st["labels_seen"])
            self._seen_for = id(m)
        return m

    # ----------------------------------------------------- live snapshots
    def _fold_live(self, store, model: Dict[str, Any], last: Optional[float], now: float,
                   dt: float) -> int:
        """Refresh the evidence snapshot of every incident B27 touched since the
        previous run (last_seen >= last)."""
        cases = model["_cases"]
        n = 0
        self._tick_systems = set()
        for inc in store.incidents(since=last):
            self._tick_systems.add(inc.system)
            key = "inc:" + inc.id
            case = cases.get(key)
            if case is None:
                case = cases[key] = self._case_from_incident(inc)
            self._refresh(case, inc)
            self._fold(store, case, float(case["ts_fold"]), now, dt)
            n += 1
        return n

    @staticmethod
    def _case_from_incident(inc: Incident) -> Dict[str, Any]:
        opened = _f(inc.opened) or _f(inc.last_seen) or 0.0
        return {"sys": inc.system, "ent": inc.entity, "opened": opened,
                "last": _f(inc.last_seen) or opened, "ts_fold": opened,
                "kinds": [], "axes": [], "fs": {}, "ds": {}, "z": {}, "new": [], "ek": [],
                "cats": [], "stages": 0, "sig": 0, "e_day": None, "risk": 0.0,
                "status": inc.status, "severity": _sev(inc.severity), "verdict": None}

    @staticmethod
    def _refresh(case: Dict[str, Any], inc: Incident) -> None:
        """Fields the incident object itself carries (B27 / B29 fill them).
        The evidence list is append-only, so only new entries are parsed."""
        kinds = {str(k) for k in (inc.kinds or []) if k}
        axes = {str(a) for a in (inc.axes or []) if a}
        if not kinds <= set(case["kinds"]):
            case["kinds"] = sorted(kinds | set(case["kinds"]))
        if not axes <= set(case["axes"]):
            case["axes"] = sorted(axes | set(case["axes"]))
        case["last"] = max(float(case["last"]), _f(inc.last_seen) or 0.0)
        case["status"] = inc.status
        case["severity"] = _sev(inc.severity)
        r = _f(inc.risk)
        if r is not None:
            case["risk"] = max(float(case.get("risk") or 0.0), r)
        e = _f(inc.e_day_min)
        if e is not None and (case["e_day"] is None or e < case["e_day"]):
            case["e_day"] = e
        n_ev = len(inc.evidence or [])
        start = int(case.get("ev_n", 0))
        if n_ev < start:                                      # rewritten: parse it all again
            start = 0
        if inc.explanation or n_ev > start:
            feats, new = FB.explicit_evidence(inc, evidence_from=start)
            _merge_z(case, feats)
            if new and not new <= set(case["new"]):
                case["new"] = sorted(new | set(case["new"]))
        case["ev_n"] = n_ev
        case["stages"] = FB.stage_count(case["axes"], list(case["kinds"]) + list(case["ek"]),
                                        case["cats"])

    def _fold(self, store, case: Dict[str, Any], lo: float, hi: float, dt: float) -> None:
        """Fold store evidence with ts in [lo, hi) into the case: per-family
        and per-detector max excess surprise, signed z at max |z|, min e_day,
        novelty tokens and event families, lib-4 severity and categories."""
        if not hi > lo:
            return
        s, e = case["sys"], case["ent"]
        fs, ds = case["fs"], case["ds"]
        n_rows = int(min(MAX_FOLD_ROWS, math.ceil((hi - lo) / dt) + 2))

        for m in store.derived_tail(s, e, P_FAMILY, n_rows):
            if lo <= m.ts < hi and isinstance(m.value, Mapping):
                dtm = float(m.window_s) if m.window_s and m.window_s > 0 else dt
                for fam, p in m.value.items():
                    sv = FB.surprise(p, dtm)
                    if sv and fam in _FAM and sv > fs.get(fam, -1.0):
                        fs[fam] = sv

        ts, M = store.vec_range(s, e, P_VEC, lo, hi)
        keep = ts < hi
        if keep.any() and M.shape[1] == len(DETECTORS):
            P = M[keep].astype(np.float64)
            P[~np.isfinite(P)] = np.inf                      # NaN = unscored, never p = 1
            sv = _surprise_vec(P.min(axis=0), dt)
            for j in np.flatnonzero(sv > 0.0):               # only excess surprise is kept
                d = DETECTORS[j]
                if sv[j] > ds.get(d, -1.0):
                    ds[d] = float(sv[j])

        ts, Z = store.vec_range(s, e, Z_VEC, lo, hi)
        keep = ts < hi
        if keep.any() and Z.shape[1] == len(FB.FEATURE_NAMES_V2):
            Zk = Z[keep].astype(np.float64)
            az = np.where(np.isfinite(Zk), np.abs(Zk), -1.0)
            idx = az.argmax(axis=0)
            cols = np.flatnonzero(az[idx, np.arange(Zk.shape[1])] >= FB.Z_MIN)
            if cols.size:
                best = Zk[idx[cols], cols]
                _merge_z(case, {FB.FEATURE_NAMES_V2[j]: float(v) for j, v in zip(cols, best)})

        ts, M = store.vec_range(s, e, E_DAY, lo, hi)
        keep = ts < hi
        if keep.any():
            v = M[keep, 0].astype(np.float64)
            v = v[np.isfinite(v)]
            if v.size and (case["e_day"] is None or float(v.min()) < case["e_day"]):
                case["e_day"] = float(v.min())

        new, ek = set(case["new"]), set(case["ek"])
        for ev in store.events(s, e, since=lo, kinds=FB.DISCRETE_KINDS, limit=200):
            if ev.ts >= hi:
                continue
            fam = FB.EVENT_FAMILY.get(ev.kind)
            if fam is not None:
                ek.add(ev.kind)
                sv = FB.surprise_e(ev.e_day)
                if sv is None:
                    sv = FB.surprise_e(FB.SEVERITY_E_DAY.get(_sev(ev.severity)))
                if sv is not None and sv > fs.get(fam, -1.0):
                    fs[fam] = sv
            for d, p in (ev.p_by_detector or {}).items():
                sv = FB.surprise(p, dt)
                if d in _DET and sv is not None and sv > ds.get(d, -1.0):
                    ds[d] = sv
            if ev.kind in FB.NEW_TOKEN_KINDS:
                new.update(FB.event_new_tokens(ev))
        case["new"] = sorted(new)
        case["ek"] = sorted(ek)

        cats = set(case["cats"])
        for mt in store.matches(s, e, since=lo, limit=100):
            if mt.ts >= hi:
                continue
            case["sig"] = max(int(case["sig"]), FB.SEVERITY_RANK.get(_sev(mt.severity), 0))
            if mt.category:
                cats.add(str(mt.category))
        case["cats"] = sorted(cats)
        case["stages"] = FB.stage_count(case["axes"], list(case["kinds"]) + list(case["ek"]),
                                        case["cats"])
        case["ts_fold"] = max(float(case["ts_fold"]), hi)

    @staticmethod
    def _case_from_event(ev: Any, dt: float) -> Dict[str, Any]:
        """A labelled discrete event is a one-event case: only its own
        evidence (the entity's other families at that tick were not judged)."""
        d = FB.describe(ev)
        case = {"sys": ev.system, "ent": ev.entity, "opened": float(ev.ts), "last": float(ev.ts),
                "ts_fold": float(ev.ts), "kinds": d["kinds"], "axes": d["axes"], "fs": {},
                "ds": {}, "z": {}, "new": d["new"], "ek": [], "cats": [], "stages": 0, "sig": 0,
                "e_day": d["e_day"], "risk": 0.0, "status": ev.status,
                "severity": _sev(ev.severity), "verdict": None}
        _merge_z(case, d["features"])
        fam = FB.EVENT_FAMILY.get(ev.kind)
        if fam is not None:
            case["ek"] = [ev.kind]
            sv = FB.surprise_e(ev.e_day)
            if sv is None:
                sv = FB.surprise_e(FB.SEVERITY_E_DAY.get(_sev(ev.severity)))
            if sv is not None:
                case["fs"][fam] = sv
        for det, p in (ev.p_by_detector or {}).items():
            sv = FB.surprise(p, dt)
            if det in _DET and sv is not None:
                case["ds"][det] = max(sv, case["ds"].get(det, -1.0))
                fam_d = family_of(det)
                case["fs"][fam_d] = max(sv, case["fs"].get(fam_d, -1.0))
        case["stages"] = FB.stage_count(case["axes"], case["kinds"], ())
        return case

    # --------------------------------------------------------------- labels
    def _new_labels(self, store, model: Dict[str, Any]) -> List[Label]:
        """Labels not processed yet, oldest first. Labels are never pruned
        (contract B), so the processed-id set is exact whatever order or ts
        the labels were added with."""
        st = model["_state"]
        seen = self._seen if self._seen is not None else set()
        allv = store.labels()
        if len(allv) == len(st["labels_seen"]) and all(lb.id in seen for lb in allv[:1]):
            return []
        new = [lb for lb in reversed(allv) if lb.id not in seen]
        new.sort(key=lambda lb: _f(lb.ts) or 0.0)             # stable: insertion order on ties
        for lb in new:
            seen.add(lb.id)
            st["labels_seen"].append(lb.id)
        self._seen = seen
        return new

    def _apply_label(self, ctx: Context, model: Dict[str, Any], lb: Label) -> bool:
        """Apply one verdict; returns True when precision / stacker must refit."""
        store, now, dt = ctx.store, float(ctx.now), float(ctx.window_s)
        cases = model["_cases"]
        case_key: Optional[str] = None
        case: Optional[Dict[str, Any]] = None
        if lb.target_type == "incident" and lb.target_id:
            case_key = "inc:" + lb.target_id
            case = cases.get(case_key)
            inc = store.get_incident(lb.target_id)
            if inc is not None:
                if case is None:
                    case = self._case_from_incident(inc)
                self._refresh(case, inc)
                self._fold(store, case, float(case["ts_fold"]), now, dt)
        elif lb.target_type == "event" and lb.target_id:
            case_key = "ev:" + lb.target_id
            case = cases.get(case_key)
            if case is None:
                ev = store.get_event(lb.target_id)
                if ev is not None:
                    case = self._case_from_event(ev, dt)
        system = lb.system or (case["sys"] if case else "")
        entity = lb.entity or (case["ent"] if case else "")
        if not system or not entity:
            return False                                        # unresolvable target

        relearn = False
        if case is not None and case_key is not None:
            prev = case.get("verdict")
            case["verdict"] = lb.verdict
            case["label_ts"] = now
            case["label_id"] = lb.id
            cases[case_key] = case
            relearn = lb.verdict in Y_OF or prev in Y_OF

        tier = self._scope_entity(store, lb.scope, system, entity)
        t0 = _f(lb.t0) if lb.t0 is not None else (case["opened"] if case else None)
        t1 = _f(lb.t1) if lb.t1 is not None else (case["last"] if case else None)
        st = model["_state"]
        st["rec_seq"] = int(st.get("rec_seq", 0)) + 1
        rec = {"ts": now, "seq": st["rec_seq"], "label_ts": _f(lb.ts), "t0": t0, "t1": t1,
               "label_id": lb.id, "target_id": lb.target_id, "target_type": lb.target_type,
               "scope": lb.scope}
        if lb.verdict == "expected_change":
            _append_record(model["accept"], f"{system}|{tier}", rec)
        elif lb.verdict == "tp":
            _append_record(model["freeze"], f"{system}|{tier}", rec)

        ttl = _f(lb.ttl_s)
        if lb.verdict in ("fp", "benign_known") and lb.scope != "this" and case is not None:
            self._add_policy(store, model, lb, case, system, entity, now,
                             ttl if ttl and ttl > 0 else POLICY_TTL_S)
        if lb.verdict == "benign_known" and case is not None and case["new"]:
            exp = now + ttl if ttl and ttl > 0 else None       # known-benign: no TTL unless given
            al = model["allowlist"].setdefault(f"{system}|{tier}", {})
            for tok in case["new"]:
                dim, val = FB.split_token(tok)
                al.setdefault(dim, {})[val] = exp

        if lb.target_type == "incident":
            model["queue"] = [it for it in model["queue"] if it.get("incident_id") != lb.target_id]
        self._profile_todo.append((system, entity, lb))      # written after _relearn (run)
        return relearn

    @staticmethod
    def _scope_entity(store, scope: str, system: str, entity: str) -> str:
        """The key a widened verdict is recorded under: the class key for
        scope 'class' (the entity itself when it has none), __system__ for
        scope 'system', else the entity."""
        if scope == "system":
            return SYSTEM_KEY
        if scope == "class":
            if is_class(entity):
                return entity
            return m_class.class_key(store, system, entity) or entity
        return entity

    def _add_policy(self, store, model: Dict[str, Any], lb: Label, case: Dict[str, Any],
                    system: str, entity: str, now: float, ttl: float) -> None:
        toks, gate = FB.build_tokens(case["kinds"], case["axes"], case["z"], case["new"])
        if not toks:
            return
        e_ref = _f(case.get("e_day"))
        level = (e_ref if e_ref is not None and e_ref > 0 else FB.E_DAY_UNKNOWN_REF) \
            / FB.LEVEL_HEADROOM
        st = model["_state"]
        st["policy_seq"] = int(st.get("policy_seq", 0)) + 1
        ckey = None
        if lb.scope == "class":
            ckey = entity if is_class(entity) else m_class.class_key(store, system, entity)
        model["policies"].append({
            "id": f"pol{st['policy_seq']:06d}", "label_id": lb.id, "verdict": lb.verdict,
            "scope": lb.scope, "system": system, "entity": entity, "class_key": ckey,
            "target_id": lb.target_id, "tokens": sorted(toks), "gate": sorted(gate),
            "e_day_ref": e_ref, "level": level, "created": now, "expires": now + ttl,
        })
        if len(model["policies"]) > MAX_POLICIES:
            model["policies"] = model["policies"][-MAX_POLICIES:]

    def _write_profile(self, store, model: Dict[str, Any], system: str, entity: str,
                       lb: Label, now: float) -> None:
        """profile.extra.feedback (contract G) of the labelled entity."""
        prof = store.profile(system, entity)
        if prof is None:
            prof = EntityProfile(system=system, entity=entity, updated=now)
        fb = dict(prof.extra.get("feedback") or {})
        verdicts = dict(fb.get("verdicts") or {})
        verdicts[lb.verdict] = int(verdicts.get(lb.verdict, 0)) + 1
        fb.update(FB.summary(model, system, entity))
        fb.update(n_labels=int(fb.get("n_labels", 0)) + 1, verdicts=verdicts,
                  last_verdict=lb.verdict, last_scope=lb.scope, last_label_ts=now)
        prof.extra["feedback"] = fb
        store.put_profile(prof)

    # ------------------------------------------------------------- learning
    def _relearn(self, model: Dict[str, Any], now: float) -> None:
        cases = model["_cases"]
        lab = [(k, c) for k, c in cases.items() if c.get("verdict") in Y_OF]
        if len(lab) > MAX_LABELLED:                             # keep the most recent labels
            lab.sort(key=lambda kc: (float(kc[1].get("label_ts") or 0.0), kc[0]))
            for k, _ in lab[:len(lab) - MAX_LABELLED]:
                del cases[k]
            lab = lab[len(lab) - MAX_LABELLED:]
        lab.sort(key=lambda kc: kc[0])                          # deterministic fit order

        prec: Dict[str, List[float]] = {}
        for _, c in lab:
            tp, fp = PREC_CREDIT[c["verdict"]]
            keys = [f"{f}|*" for f in FB.involved_families(c)]
            keys += [f"{family_of(d)}|{d}" for d, s in (c.get("ds") or {}).items()
                     if d in _DET and s >= FB.SIG_S]
            for k in keys:
                a = prec.setdefault(k, [0.0, 0.0])
                a[0] += tp
                a[1] += fp
        model["detector_prec"] = {k: [round(v[0], 6), round(v[1], 6)] for k, v in sorted(prec.items())}
        model["n_labelled"] = len(lab)

        if len(lab) < self.min_stack:
            model["stacker"] = None
            model["isotonic"] = None
            model["family_w"] = {f: 1.0 for f in FAMILIES}
            return
        X = np.vstack([FB.stack_vector(c) for _, c in lab])
        y = np.array([Y_OF[c["verdict"]] for _, c in lab])
        theta = fit_stacker(X, y, self.lam)
        model["stacker"] = {
            "intercept": float(theta[0]),
            "coef": {name: float(theta[1 + i]) for i, name in enumerate(FB.STACK_FEATURES)},
            "lam": self.lam, "n": int(len(y)), "n_pos": int(y.sum()),
            "fitted_ts": now,
        }
        model["family_w"] = family_weights_from(theta)
        if len(lab) >= self.min_iso:
            oof = self._oof(X, y, [k for k, _ in lab], theta)
            xs, ys = pav(oof, y)
            lo = 0.5 / (len(y) + 1.0)
            model["isotonic"] = {"x": [float(v) for v in xs],
                                 "y": [float(min(1.0 - lo, max(lo, v))) for v in ys],
                                 "n": int(len(y)), "fitted_ts": now}
        else:
            model["isotonic"] = None

    def _oof(self, X: np.ndarray, y: np.ndarray, keys: Sequence[str],
             theta_all: np.ndarray) -> np.ndarray:
        """Out-of-fold stacker probabilities (deterministic folds by case key),
        so the isotonic map is fitted on scores the stacker did not train on."""
        fold = np.array([min(ISO_FOLDS - 1, int(seeded_uniform("b23-fold", k) * ISO_FOLDS))
                         for k in keys])
        oof = _design(theta_all, X)
        for f in range(ISO_FOLDS):
            te = fold == f
            tr = ~te
            if not te.any() or tr.sum() < self.min_stack // 2:
                continue
            th = fit_stacker(X[tr], y[tr], self.lam)
            oof[te] = _design(th, X[te])
        return oof

    # ------------------------------------------------------------ expiry
    @staticmethod
    def _sweep(model: Dict[str, Any], now: float) -> bool:
        changed = False
        pol = [p for p in model["policies"] if p.get("expires") is None or p["expires"] > now]
        if len(pol) != len(model["policies"]):
            model["policies"] = pol
            changed = True
        for k in list(model["allowlist"]):
            dims = model["allowlist"][k]
            for d in list(dims):
                vals = dims[d]
                dead = [v for v, exp in vals.items() if exp is not None and exp <= now]
                for v in dead:
                    del vals[v]
                    changed = True
                if not vals:
                    del dims[d]
            if not dims:
                del model["allowlist"][k]
        for field in ("accept", "freeze"):
            tab = model[field]
            for k in list(tab):
                recs = [r for r in tab[k] if now - float(r.get("ts", now)) <= RECORD_KEEP_S]
                if len(recs) != len(tab[k]):
                    changed = True
                    if recs:
                        tab[k] = recs
                    else:
                        del tab[k]
        cases = model["_cases"]
        unl = [(k, c) for k, c in cases.items() if not c.get("verdict")]
        for k, c in unl:
            if now - float(c.get("last", now)) > CASE_KEEP_S:
                del cases[k]
        unl = [(k, c) for k, c in cases.items() if not c.get("verdict")]
        if len(unl) > MAX_UNLABELLED:
            unl.sort(key=lambda kc: (float(kc[1].get("last", 0.0)), kc[0]))
            for k, _ in unl[:len(unl) - MAX_UNLABELLED]:
                del cases[k]
        return changed

    # ---------------------------------------------------------- alert budget
    def _update_alpha(self, ctx: Context, model: Dict[str, Any], now: float) -> bool:
        """Daily per system: alpha <- alpha * exp(0.1 * (target - observed)),
        target = alert_budget.system_per_day notifying incidents, observed the
        system's non-suppressed root incidents >= LOW opened over the elapsed
        interval (per day). Two guards, both about poisoning and the null
        budget: each entity counts at most alert_budget.entity_per_hour
        incidents a day (one noisy or hostile entity cannot desensitise its
        whole system), tp-labelled incidents never count (they are the point);
        and the upper clip is alert_budget.alpha_max (default 1.0, the spec's
        4.0 is opt-in) because raising thresholds above the calibrated ladder
        on a quiet system spends false alarms the eval gates do not allow."""
        st = model["_state"]
        store = ctx.store
        ats: Dict[str, float] = st["alpha_ts"]
        systems = sorted(set(store.systems()) | set(ats) | self._tick_systems)
        if ctx.training:                                      # no incidents open in training
            for s in systems:
                ats[s] = now
            return False
        cfg = ctx.config.get("alert_budget") or {}
        target = float(cfg.get("system_per_day", 20))
        cap_e = max(1, int(cfg.get("entity_per_hour", 3)))
        a_max = min(FB.ALPHA_MAX, max(FB.ALPHA_MIN, float(cfg.get("alpha_max", ALPHA_MAX_DEFAULT))))
        cases = model["_cases"]
        changed = False
        for s in systems:
            t0 = ats.get(s)
            if t0 is None:
                ats[s] = now
                continue
            span = now - float(t0)
            if span < ALPHA_PERIOD_S:
                continue
            per_ent: Counter = Counter()
            for inc in store.incidents(system=s, since=t0):
                if not (t0 <= inc.opened < now) or inc.status == "suppressed" or inc.parent_id:
                    continue
                if FB.SEVERITY_RANK.get(_sev(inc.severity), 0) < 1:
                    continue
                c = cases.get("inc:" + inc.id)
                if c is not None and c.get("verdict") == "tp":
                    continue
                per_ent[inc.entity] += 1
            observed = sum(min(v, cap_e) for v in per_ent.values()) * DAY / span
            a0 = FB.alpha_mult(model, s)
            a = a0 * math.exp(max(-50.0, min(50.0, ALPHA_GAIN * (target - observed))))
            model["alpha_mult"][s] = float(min(a_max, max(FB.ALPHA_MIN, a)))
            ats[s] = now
            changed = True
        return changed

    # ----------------------------------------------------------- label queue
    def _update_queue(self, ctx: Context, model: Dict[str, Any], now: float) -> bool:
        store = ctx.store
        st = model["_state"]
        cases = model["_cases"]
        q = model["queue"]
        keep = []
        gov = None                     # governor label queue {(s, e): record}, read lazily
        for it in q:
            if it.get("source") == "governor":
                if gov is None:
                    gov = {(r["system"], r["entity"]): r for r in MG.label_queue(store)}
                if (it.get("system"), it.get("entity")) in gov:
                    keep.append(it)                               # still held DRIFTING
                continue
            iid = it.get("incident_id")
            c = cases.get("inc:" + str(iid))
            if c is not None and c.get("verdict"):
                continue                                          # labelled
            inc = store.get_incident(iid)
            if inc is None or (it.get("reason") == "held" and inc.status == "closed"):
                continue
            keep.append(it)
        changed = len(keep) != len(q)
        if ctx.training:
            st["queue_ts"] = now
            model["queue"] = keep
            return changed
        queued = {it["incident_id"] for it in keep if it.get("incident_id") is not None}

        if st.get("held_ts") is None or now - st["held_ts"] >= HELD_SCAN_S:
            st["held_ts"] = now
            # keys B28 has held DRIFTING > 14 d (m_governor.label_queue): B28
            # cannot write model.feedback, so B23 queues them (integration R14.3)
            have = {(it.get("system"), it.get("entity")) for it in keep
                    if it.get("source") == "governor"}
            if gov is None:
                gov = {(r["system"], r["entity"]): r for r in MG.label_queue(store)}
            for (gs, ge), r in sorted(gov.items()):
                if (gs, ge) in have:
                    continue
                keep.append({"incident_id": None, "system": gs, "entity": ge,
                             "reason": "held", "source": "governor", "risk": 0.0,
                             "p_malicious": 0.5, "e_day": None,
                             "opened": _f(r.get("since")), "added": now,
                             "severity": "info", "status": "drifting",
                             "type": r.get("type"), "p_legit": r.get("p_legit")})
                changed = True
            for inc in store.incidents(status=("open", "acked")):
                if inc.id in queued or now - float(inc.opened) <= HELD_S:
                    continue
                c = cases.get("inc:" + inc.id)
                if c is not None and c.get("verdict"):
                    continue
                keep.append(self._item(model, inc, c, "held", now))
                queued.add(inc.id)
                changed = True

        if st.get("queue_ts") is None:
            st["queue_ts"] = now                                 # first batch after a day of cases
        elif now - st["queue_ts"] >= QUEUE_PERIOD_S:
            st["queue_ts"] = now
            pool = []
            for k, c in cases.items():
                if not k.startswith("inc:") or c.get("verdict"):
                    continue
                iid = k[4:]
                if iid in queued or now - float(c.get("last", 0.0)) > QUEUE_LOOKBACK_S:
                    continue
                inc = store.get_incident(iid)
                if inc is not None:
                    pool.append((k, c, inc))
            if pool:
                n_unc = int(round(self.queue_per_day * QUEUE_UNCERTAIN_FRAC))
                n_risk = self.queue_per_day - n_unc
                by_risk = sorted(pool, key=lambda t: (-float(t[1].get("risk") or 0.0),
                                                      _f(t[1].get("e_day")) or math.inf, t[0]))
                picks = [(t, "risk") for t in by_risk[:n_risk]]
                rest = by_risk[n_risk:]
                unc = sorted(rest, key=lambda t: (abs(FB.p_malicious(model, t[1]) - 0.5),
                                                  -float(t[1].get("risk") or 0.0), t[0]))
                picks += [(t, "uncertain") for t in unc[:n_unc]]
                used = {t[0] for t, _ in picks}
                for t in rest:                                   # fill unused uncertainty slots
                    if len(picks) >= self.queue_per_day:
                        break
                    if t[0] not in used:
                        picks.append((t, "risk"))
                for (k, c, inc), reason in picks:
                    keep.append(self._item(model, inc, c, reason, now))
                changed = True

        if len(keep) > QUEUE_MAX:                                # never drop held items first
            keep.sort(key=lambda it: (it["reason"] != "held", -float(it.get("risk") or 0.0),
                                      -float(it.get("added") or 0.0)))
            keep = keep[:QUEUE_MAX]
            changed = True
        model["queue"] = keep
        return changed

    @staticmethod
    def _item(model: Dict[str, Any], inc: Incident, case: Optional[Dict[str, Any]],
              reason: str, now: float) -> Dict[str, Any]:
        risk = max(float(case.get("risk") or 0.0) if case else 0.0, _f(inc.risk) or 0.0)
        e_day = _f(case.get("e_day")) if case else None
        if e_day is None:
            e_day = _f(inc.e_day_min)
        return {"incident_id": inc.id, "system": inc.system, "entity": inc.entity,
                "reason": reason, "risk": risk,
                "p_malicious": float(FB.p_malicious(model, case)) if case else 0.5,
                "e_day": e_day, "opened": float(inc.opened), "added": now,
                "severity": _sev(inc.severity), "status": inc.status}


def _surprise_vec(p: np.ndarray, dt: float) -> np.ndarray:
    """Vectorised FB.surprise: min(8, max(0, -log10(p * 86400 / dt))); a
    non-finite p (unscored) gives 0, i.e. no excess surprise recorded."""
    out = np.zeros(p.shape)
    ok = np.isfinite(p)
    if ok.any():
        e = np.clip(p[ok], 1e-300, 1.0) * (DAY / dt)
        out[ok] = np.clip(-np.log10(e), 0.0, FB.S_MAX)
    return out


def _merge_z(case: Dict[str, Any], feats: Mapping[str, float]) -> None:
    """Keep, per feature, the signed z of largest |z| (only |z| >= Z_MIN can
    ever become a pattern token); retain the top Z_KEEP."""
    if not feats:
        return
    z = case["z"]
    for n, v in feats.items():
        fv = _f(v)
        if fv is not None and abs(fv) >= FB.Z_MIN and (n not in z or abs(fv) > abs(z[n])):
            z[n] = fv
    if len(z) > Z_KEEP:
        top = sorted(z.items(), key=lambda t: (-abs(t[1]), t[0]))[:Z_KEEP]
        case["z"] = dict(top)


def _append_record(tab: Dict[str, List[Dict[str, Any]]], key: str, rec: Dict[str, Any]) -> None:
    recs = tab.setdefault(key, [])
    recs.append(rec)
    if len(recs) > MAX_RECORDS:
        del recs[:len(recs) - MAX_RECORDS]
