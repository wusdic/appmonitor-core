"""Habituation of recurring HIGH lib-4 matches (owner: B26 risk; readers
B27 incident, B28 governor). Lead decision, evaluation round 4.

Why. lib-4 grades activities, not intent: `bulk_upload` (HIGH) is what a
sanctioned nightly backup looks like, every night. B26 counted it at full
weight on every match and B28 zeroed the warm-up trust of every such tick,
so B13's committed ring never held a trusted backup hour and fell back to
the peers' level: the backup hosts (pack L15) sat at risk 80-100 and were
CRITICAL in every run; the NAT host (L7) likewise. The rule for <= MEDIUM
matches (lib/stages.habit_step: any match seen on >= 4 ticks over >= 24 h)
is far too loose for HIGH. A HIGH match habituates only per (entity, rule),
only from TRUSTED history, and only inside the envelope that history
learnt, so the same rule firing at a new time, towards a new peer or with
more volume than the entity ever moved is counted in full again.

Rule, per (system, entity, signature) with severity HIGH (CRITICAL never
habituates; lower severities keep the stages rule):
  history   the matches on TRUSTED ticks: warm-up ticks (the reference
            period) and live ticks at which the key was not quarantined
            (behavior.quarantine = 0: no live incident, regime normal /
            returned / accepted). Kept per local day (ctx.config tz) for
            FORGET_DAYS days: the local hours matched, the day's amount
            (sum of the per-tick totals of the rule's additive counters)
            and the peak per-grain value of those counters, plus the union
            of the named peers (raw l4.peer_set) of the matched ticks.
  habituated  >= MIN_DAYS distinct trusted days.
  envelope  schedule: the local hour of the match (the hour holding the
                tick's last second) is within +-1 h of an hour matched on
                >= 2 distinct trusted days (a rhythm, not a one-off);
            budget: peak <= max trusted peak x slack and the day's running
                amount (this match included, whatever its trust) <= max
                trusted daily amount x slack, slack = exp(max(ln 1.5,
                2 sd)) with sd the sample SD of the log trusted values (a
                new day of a lognormal volume exceeds max-of-n x e^(2 sd)
                with probability < 1/(n+1) x 2.3 %);
            peers: every named peer of the tick was seen on a trusted match.
            A rule without additive-counter terms, or a tick without peer
            data, skips that part (no data is not an exceedance).
  verdict   'in'  habituated and inside the envelope -> B26 weight 0; B27
                  treats it as habitual (joins, never restarts the quiet
                  clock); B28 does not treat it as a HIGH match (training
                  trust, the malicious-source flag).
            'out' habituated but outside (reasons listed) -> full weight.
            'learning' not habituated yet -> full weight.
  learning  a trusted match is added to the history unless its verdict is
            'out': a live match outside the envelope never widens it (an
            unquarantined slow exfiltration at a new hour could otherwise
            habituate itself in two nights). A legitimate schedule change
            is learnt after the governor ACCEPTS it: a model.control version
            bump of the key restarts the history (MIN_DAYS again).

Layout of model.lib4_habit@(system, entity) (JSON-safe, written by B26):
    {"ctl": control version seen, "sigs": {signature_id: {
        "days": {"<local day>": {"h": [hours], "amt": float, "peak": float}},
        "peers": [peer, ...], "last": ts of the last processed match,
        "today": [local day, running amount],
        "v": {"<ts>": verdict}}}}           # verdicts of the last 2 days
"""
from __future__ import annotations

import math
from typing import Any, Dict, Iterable, List, Mapping, Optional, Tuple

from . import timebins as TB

MODEL = "model.lib4_habit"
SEVERITY = "high"
MIN_DAYS = 5
FORGET_DAYS = 30
HOUR_MIN_DAYS = 2               # an hour belongs to the rhythm once seen on 2 days
HOUR_TOL = 1                    # +- 1 h around a rhythm hour
SLACK_MIN = 1.5
SLACK_SD = 2.0
MAX_PEERS = 64
VERDICT_KEEP_S = 2 * 86400.0
# graded weight of a HIGH match that is not habituated (round 4, evaluator;
# grade()): the prior of a rule's per-day match probability has mean PI_REF
# (a once-a-month activity weighs in full) and PRIOR_DAYS pseudo-days
PI_REF = 1.0 / 30.0
PRIOR_DAYS = 0.5 / PI_REF                   # Beta(1/2, PRIOR_DAYS - 1/2): 15 pseudo-days
SCHEDULE_WHY = "outside the learnt schedule"
PEER_SET = "l4.peer_set"
OTHER = "__other__"
IN, OUT, LEARNING = "in", "out", "learning"


# ======================================================================= pure
def local_day_hour(ts: float, tz: str) -> Tuple[int, int]:
    """(local day index, local hour) of the tick ending at ts (its last second)."""
    slot = TB.slot_of(float(ts) - 1.0, tz)
    return slot // 96, (slot % 96) // 4


def slack(values: Iterable[float]) -> float:
    """exp(max(ln SLACK_MIN, SLACK_SD x sd(log v))) over the positive values."""
    lv = [math.log(v) for v in values if v > 0.0]
    if len(lv) < 2:
        return SLACK_MIN
    m = sum(lv) / len(lv)
    sd = math.sqrt(sum((x - m) ** 2 for x in lv) / (len(lv) - 1))
    return math.exp(max(math.log(SLACK_MIN), SLACK_SD * sd))


def rhythm_hours(days: Mapping[str, Mapping[str, Any]]) -> set:
    """Hours matched on >= HOUR_MIN_DAYS distinct days, dilated by +- HOUR_TOL."""
    cnt: Dict[int, int] = {}
    for d in days.values():
        for h in set(int(x) for x in d.get("h") or ()):
            cnt[h] = cnt.get(h, 0) + 1
    core = [h for h, c in cnt.items() if c >= HOUR_MIN_DAYS]
    return {(h + k) % 24 for h in core for k in range(-HOUR_TOL, HOUR_TOL + 1)}


def judge(rec: Mapping[str, Any], day: int, hour: int, peak: Optional[float],
          amount: Optional[float], peers: Optional[Iterable[str]]) -> Tuple[str, List[str]]:
    """Verdict of one match against the trusted history `rec` (before it is
    learnt): (verdict, reasons)."""
    days = rec.get("days") or {}
    if len(days) < MIN_DAYS:
        return LEARNING, []
    why: List[str] = []
    if hour not in rhythm_hours(days):
        why.append(f"hour {hour:02d} {SCHEDULE_WHY}")
    if peak is not None:
        peaks = [float(d.get("peak") or 0.0) for d in days.values()]
        mx = max(peaks, default=0.0)
        if mx > 0.0 and peak > mx * slack(peaks):
            why.append(f"peak {peak:.3g} > budget {mx * slack(peaks):.3g}")
    if amount is not None:
        amts = [float(d.get("amt") or 0.0) for d in days.values()]
        mx = max(amts, default=0.0)
        run = amount
        td = rec.get("today")
        if td and int(td[0]) == day:
            run += float(td[1])
        if mx > 0.0 and run > mx * slack(amts):
            why.append(f"daily amount {run:.3g} > budget {mx * slack(amts):.3g}")
    if peers is not None:
        known = set(rec.get("peers") or ())
        new = sorted(p for p in peers if p != OTHER and p not in known)
        if new:
            why.append("new peer " + ",".join(new[:3]))
    return (OUT, why) if why else (IN, [])


def note_day(model: Dict[str, Any], day: int) -> bool:
    """Record a local day on which the key had TRUSTED activity (the
    denominator of grade()); keeps FORGET_DAYS days. True when added."""
    obs = model.setdefault("obs", [])
    if day in obs:
        return False
    obs.append(int(day))
    obs.sort()
    del obs[:max(0, len(obs) - FORGET_DAYS - 1)]
    model["obs"] = [d for d in obs if d >= day - FORGET_DAYS]
    return True


def grade(model: Mapping[str, Any], signature_id: Any, day: int) -> float:
    """Weight factor in (0, 1] of a HIGH match that is not habituated
    (verdict 'learning', or 'out' for the schedule alone), from how often
    the entity itself matched the rule on its trusted days before `day`
    (round 4, evaluator).

    B26 turns every other evidence into EXCESS SURPRISE, log10(1 / e_day);
    a lib-4 match counted its full severity weight whatever the entity's own
    history, so a user who uploads a document batch on a third of her
    trusted days (pack A seed 1: oa-portal 10.30.2.23, bulk_upload on 4 of
    ~18 trusted days, at 11, 14 and 16 h) sat at risk 60-80 from lib-4 alone
    and every null family hit opened a risk incident (control risk openings:
    A1 10 of 21, E0 8-11 of 13-16, B0 21 of 57). The binary envelope of the
    lead's HIGH habituation needs 5 days AND a schedule, which an irregular
    human activity never has. Here: k trusted days matched out of n trusted
    active days, per-day match probability pi = (k + 1/2) / (n + PRIOR_DAYS)
    (Beta prior with mean PI_REF = 1/30 and 15 pseudo-days, so no history or
    a first match gives pi = 1/30 and the full weight), and the factor is
    the surprise of a match today relative to a once-a-month one,
    log10(1 / pi) / log10(1 / PI_REF), capped at 1. A rule matched on 4 of
    18 days weighs 0.58, on 10 of 18 0.34; an entity that never matched it
    in 30 days weighs 1. Only the entity's own TRUSTED days count, so an
    attacker cannot lower it by being noisy under quarantine, and volume or
    peers outside the trusted envelope still count in full (B26)."""
    rec = (model.get("sigs") or {}).get(str(signature_id)) or {}
    k = sum(1 for d in (rec.get("days") or {}) if int(d) < day)
    n = sum(1 for d in (model.get("obs") or ()) if int(d) < day)
    n = max(n, k)
    pi = (k + 0.5) / (n + PRIOR_DAYS)
    return min(1.0, math.log10(1.0 / pi) / math.log10(1.0 / PI_REF)) if pi < 1.0 else 0.0


def schedule_only(why: Iterable[str]) -> bool:
    """True when every reason of an 'out' verdict is the schedule."""
    w = list(why)
    return bool(w) and all(SCHEDULE_WHY in x for x in w)


def irregular(days: Mapping[str, Mapping[str, Any]]) -> bool:
    """A rule's trusted history has no schedule to be off: at least half of
    its matched days had a match outside the rhythm hours (hours matched on
    HOUR_MIN_DAYS days, +- HOUR_TOL). A nightly backup (every day at 01 h)
    is regular; a user uploading at 9, 14, 16, 11, 17 h is not."""
    if not days:
        return True
    rh = rhythm_hours(days)
    off = sum(1 for d in days.values() if any(int(h) not in rh for h in (d.get("h") or ())))
    return off >= 0.5 * len(days)


def graded(model: Mapping[str, Any], signature_id: Any, verdict: str,
           why: Iterable[str]) -> bool:
    """Does grade() apply to this verdict (round 4, evaluator)? 'learning'
    always; 'out' only when the schedule is its sole reason AND the rule's
    trusted history is irregular (irregular()): an activity without a
    schedule cannot be off it. A scheduled activity (a nightly backup) at a
    new hour keeps the full weight (lead decision), as does more volume
    than trusted or a new peer."""
    if verdict == LEARNING:
        return True
    if verdict != OUT or not schedule_only(why):
        return False
    rec = (model.get("sigs") or {}).get(str(signature_id)) or {}
    return irregular(rec.get("days") or {})


def learn(rec: Dict[str, Any], day: int, hour: int, peak: Optional[float],
          amount: Optional[float], peers: Optional[Iterable[str]]) -> None:
    days = rec.setdefault("days", {})
    d = days.setdefault(str(day), {"h": [], "amt": 0.0, "peak": 0.0})
    if hour not in d["h"]:
        d["h"].append(hour)
    if amount is not None:
        d["amt"] = float(d["amt"]) + float(amount)
    if peak is not None:
        d["peak"] = max(float(d["peak"]), float(peak))
    if peers is not None:
        known = rec.setdefault("peers", [])
        for p in sorted(peers):
            if p != OTHER and p not in known and len(known) < MAX_PEERS:
                known.append(p)
    for k in [k for k in days if int(k) < day - FORGET_DAYS]:
        del days[k]


# ================================================================= accessors
def get(store: Any, s: str, e: str) -> Dict[str, Any]:
    m = store.get_model(s, e, MODEL)
    return m if isinstance(m, dict) else {}


def verdict(store: Any, s: str, e: str, signature_id: Any, ts: float) -> Optional[str]:
    """B26's verdict for the match (s, e, signature_id) at ts: 'in' | 'out' |
    'learning', or None (not a HIGH match, or B26 has not seen it)."""
    rec = (get(store, s, e).get("sigs") or {}).get(str(signature_id))
    if not rec:
        return None
    return (rec.get("v") or {}).get(f"{float(ts):.3f}")


def habituated(store: Any, s: str, e: str, signature_id: Any, ts: float) -> bool:
    """True iff the HIGH match is inside its learnt envelope (B27 / B28)."""
    return verdict(store, s, e, signature_id, ts) == IN


# ==================================================================== writer
def match_inputs(store: Any, mt: Any, additive: Iterable[str]) -> Tuple[
        Optional[float], Optional[float], Optional[List[str]]]:
    """(peak per grain, tick amount, named peers) of a match: the peak is the
    largest additive-counter value in its evidence (per 15-min grain in
    canonical mode), the amount the sum of those counters' raw per-tick
    totals, the peers the raw l4.peer_set keys of the tick."""
    add = set(additive)
    peak: Optional[float] = None
    amount: Optional[float] = None
    seen = set()
    for term, v in (getattr(mt, "evidence", None) or {}).items():
        metric = str(term).split(" ")[0]
        if metric not in add or metric in seen:
            continue
        seen.add(metric)
        try:
            fv = float(v)
        except (TypeError, ValueError):
            continue
        if math.isfinite(fv):
            peak = fv if peak is None else max(peak, fv)
        raw = store.latest_raw_at(mt.system, mt.entity, metric, float(mt.ts))
        if raw is not None and isinstance(raw.value, (int, float)) \
                and math.isfinite(float(raw.value)):
            amount = (amount or 0.0) + float(raw.value)
    peers: Optional[List[str]] = None
    ps = store.latest_raw_at(mt.system, mt.entity, PEER_SET, float(mt.ts))
    if ps is not None and isinstance(ps.value, Mapping):
        peers = sorted(str(p) for p in ps.value if str(p) != OTHER)
    return peak, amount, peers


def observe(model: Dict[str, Any], mt: Any, trusted: bool, tz: str,
            inputs: Tuple[Optional[float], Optional[float], Optional[List[str]]]
            ) -> Tuple[str, List[str]]:
    """Judge one HIGH match against its history, learn it when trusted (and
    not 'out'), record the verdict. Idempotent per (signature, ts)."""
    sigs = model.setdefault("sigs", {})
    sid = str(mt.signature_id)
    rec = sigs.setdefault(sid, {"days": {}, "peers": [], "last": None, "today": None, "v": {}})
    ts = float(mt.ts)
    key = f"{ts:.3f}"
    if key in rec["v"]:
        return rec["v"][key], []
    if rec.get("last") is not None and ts <= float(rec["last"]):
        return LEARNING, []                   # an older match seen late: never re-learnt
    day, hour = local_day_hour(ts, tz)
    peak, amount, peers = inputs
    v, why = judge(rec, day, hour, peak, amount, peers)
    if trusted and v != OUT:
        learn(rec, day, hour, peak, amount, peers)
    td = rec.get("today")
    if amount is not None:
        rec["today"] = [day, (float(td[1]) if td and int(td[0]) == day else 0.0) + amount]
    rec["last"] = ts
    rec["v"][key] = v
    for k in [k for k in rec["v"] if float(k) < ts - VERDICT_KEEP_S]:
        del rec["v"][k]
    return v, why


def reset_on_control(model: Dict[str, Any], ctl_version: int) -> bool:
    """A governor ACCEPT / rebase (model.control version bump) restarts the
    history: the accepted behaviour is learnt afresh. True if it reset."""
    old = model.get("ctl")
    model["ctl"] = int(ctl_version)
    if old is not None and int(old) != int(ctl_version) and model.get("sigs"):
        for rec in model["sigs"].values():
            rec["days"], rec["peers"], rec["today"] = {}, [], None
        return True
    return False
