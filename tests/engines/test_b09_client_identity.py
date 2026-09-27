"""B09 ClientIdentityEngine (docs/lib3/engines.md '## B09'): spec unit tests
(a)-(c). Raw inputs come from the real R3 ClientStackEngine fed with
Observations, so the tokens, episodes and OS/UA/TTL pairs are exactly what the
pipeline produces; a B01 stand-in writes the feature.active clock the gated
learner steps on, and B28's trust rings are written with set_trust.

Spec test mapping:
  (a) test_a_python_interleaved_in_one_tick_is_impersonation
  (b) test_b_class_rollout_is_only_client_change
  (c) test_c_copied_ua_with_linux_ja3_and_ttl64_is_inconsistent
Edge cases live in test_b09_client_identity_edges.py (they import Feed).
"""
from __future__ import annotations

import math
from typing import Dict, Iterable, List, Optional, Sequence, Tuple

import numpy as np

from helpers import DT, T0, make_store, obs, put_model, run_engine, set_trust

from app.engines.behavior.client_identity import DETECTOR, ClientIdentityEngine
from app.engines.behavior.lib import emit
from app.engines.behavior.lib import m_client as MC
from app.engines.raw.client_stack import ClientStackEngine

S = "erp"
E = "10.0.0.1"

UA_CHROME = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like "
             "Gecko) Chrome/{v}.0.0.0 Safari/537.36")
UA_PY = "python-requests/2.31.0"
JA3_WIN = ("771,4865-4866-4867-49195-49199-49196-49200-52393-52392,"
           "0-23-65281-10-11-35-16-5-13-18-51-45-43-27-21,29-23-24,0")
JA3_WIN127 = ("771,4865-4866-4867-49195-49199-49196-49200-52393-52392-49171,"
              "0-23-65281-10-11-35-16-5-13-18-51-45-43-27-21-17513,29-23-24,0")
JA3_LINUX = "771,4866-4867-4865-49196-49200-159-52393,0-11-10-35-22-23-13-43-45-51,29-23-30-25-24,0-1-2"

# (ja3, ua, ttl, win)
CHROME126 = (JA3_WIN, UA_CHROME.format(v=126), 128, 64240)
CHROME127 = (JA3_WIN127, UA_CHROME.format(v=127), 128, 64240)
PYTHON = (JA3_LINUX, UA_PY, 64, 29200)
FORGED = (JA3_LINUX, UA_CHROME.format(v=126), 64, 29200)     # copied UA, Linux stack


def times(now: float, dt: float = DT, step: float = 60.0, offset: float = 0.0,
          start: float = 0.0, end: Optional[float] = None) -> np.ndarray:
    """Event times inside the tick (now - dt, now]: every `step` s from
    now - dt + offset + start up to now - dt + end."""
    lo = now - dt
    hi = now if end is None else lo + end
    t = np.arange(lo + start + offset, hi, step)
    return t[t > lo]


class Feed:
    """R3 + B01 + B28 stand-in for one system."""

    def __init__(self, store, system: str = S) -> None:
        self.store, self.s = store, system
        self.r3 = ClientStackEngine()
        self.entities: set = set()

    def tick(self, now: float, traffic: Dict[str, Sequence[Tuple[tuple, Iterable[float]]]],
             dt: float = DT, trust: bool = True) -> None:
        """traffic: {entity: [(stack, event times), ...]}."""
        ob = []
        for e, parts in traffic.items():
            self.entities.add(e)
            for (ja3, ua, ttl, win), ts in parts:
                for t in ts:
                    ob.append(obs(self.s, e, float(t), ja3=ja3, user_agent=ua, ttl=ttl,
                                  win_size=win, bytes_up=500, bytes_down=5000))
        if ob:
            run_engine(self.r3, self.store, now, dt=dt, observations=ob)
        for e in sorted(self.entities):
            active = 1.0 if traffic.get(e) else 0.0
            self.store.add_vec(self.s, e, "feature.active", now,
                               np.array([active], np.float32), window_s=int(dt))
            if trust:
                set_trust(self.store, self.s, e, [now], 1.0)


def pm_at(store, e: str, now: float, s: str = S) -> float:
    return float(emit.read_array(store, s, e, emit.PM, now)[emit.DETECTOR_INDEX[DETECTOR]])


def score_at(store, e: str, now: float, s: str = S) -> float:
    return float(emit.read_array(store, s, e, emit.SCORE, now)[emit.DETECTOR_INDEX[DETECTOR]])


def events(store, kind: str, e: Optional[str] = None, s: str = S) -> List:
    return store.events(system=s, entity=e, kinds=[kind], limit=10_000)


def warm(feed: Feed, eng: ClientIdentityEngine, ents: Dict[str, tuple], n: int,
         t_start: float = T0, dt: float = DT, training: bool = True) -> float:
    """n ticks of every entity on its own stack; returns the last tick ts."""
    now = t_start
    for i in range(n):
        now = t_start + i * dt
        feed.tick(now, {e: [(st, times(now, dt, offset=(j % 7) * 3.0))]
                        for j, (e, st) in enumerate(sorted(ents.items()))}, dt=dt)
        run_engine(eng, feed.store, now, training=training, dt=dt)
    return now


def class_model(system: str, ips: Sequence[str], rid: str = "r1") -> dict:
    return {"assign": {f"{system}|{ip}": {"role": rid, "sub": None, "prob": 0.9,
                                          "static": [], "pool": None, "super": "human"}
                       for ip in ips},
            "roles": {rid: {"name": "office", "members": [f"{system}|{ip}" for ip in ips],
                            "medoid": f"{system}|{ips[0]}", "lineage": [], "version": 1}},
            "subs": {}, "statics": {}, "pools": {}, "version": 1}


# ============================================================ spec test (a)
def test_a_python_interleaved_in_one_tick_is_impersonation():
    store, eng = make_store(), ClientIdentityEngine()
    feed = Feed(store)
    peers = {f"10.0.0.{i}": CHROME126 for i in range(2, 6)}
    ents = {E: CHROME126, **peers}
    now = warm(feed, eng, ents, 100)
    assert not events(store, "client_impersonation")
    # a quiet, mature entity scores low
    assert pm_at(store, E, now) > 0.5

    fired_at = None
    for k in range(1, 4):
        now = T0 + (99 + k) * DT
        traffic = {e: [(st, times(now))] for e, st in peers.items()}
        # chrome every 60 s, python-requests interleaved 30 s later, same tick
        traffic[E] = [(CHROME126, times(now)), (PYTHON, times(now, offset=30.0))]
        feed.tick(now, traffic)
        run_engine(eng, store, now)
        if events(store, "client_impersonation", E) and fired_at is None:
            fired_at = k
    assert fired_at is not None and fired_at <= 2, fired_at
    ev = events(store, "client_impersonation", E)[0]
    assert ev.severity.value == "high"
    assert ev.extra["C"] == 1
    assert MC.parse(ev.extra["stack"]).family == "python-requests"
    assert MC.parse(ev.extra["dominant"]).ua == "chrome/126"
    assert ev.extra["risk"] > 0.8
    assert ev.axes == ["identity"]
    # the score and pm agree with the event's risk
    t_ev = ev.ts
    assert score_at(store, E, t_ev) > -math.log10(0.2 + 1e-6) - 1e-9
    assert pm_at(store, E, t_ev) < 0.2
    # one alert per (entity, stack) per day, and no alert for the peers
    assert len(events(store, "client_impersonation", E)) == 1
    assert len(events(store, "client_impersonation")) == 1


# ============================================================ spec test (b)
def test_b_class_rollout_is_only_client_change():
    store, eng = make_store(), ClientIdentityEngine()
    feed = Feed(store)
    ips = [f"10.1.0.{i}" for i in range(1, 11)]
    put_model(store, "__org__", "__org__", "model.class", class_model(S, ips))
    now = warm(feed, eng, {ip: CHROME126 for ip in ips}, 100)
    t_warm = now

    movers = ips[:8]                     # 80 % of the class, one every 2.5 h
    move_tick = {ip: 101 + 10 * k for k, ip in enumerate(movers)}
    pms: Dict[str, List[float]] = {ip: [] for ip in movers}
    for i in range(101, 101 + 96):       # one day
        now = T0 + (i - 1) * DT
        traffic = {}
        for ip in ips:
            moved = ip in move_tick and i >= move_tick[ip]
            if ip in move_tick and i == move_tick[ip]:
                # restart mid-tick: chrome126 until 10:00, chrome127 from 10:20
                traffic[ip] = [(CHROME126, times(now, end=600.0)),
                               (CHROME127, times(now, start=620.0))]
            else:
                traffic[ip] = [(CHROME127 if moved else CHROME126, times(now))]
        feed.tick(now, traffic)
        run_engine(eng, store, now)
        for ip in movers:
            if i >= move_tick[ip]:
                p = pm_at(store, ip, now)
                if p == p:
                    pms[ip].append(p)

    assert not events(store, "client_impersonation")
    changes = events(store, "client_change")
    assert changes and all(ev.severity.value == "info" for ev in changes)
    assert {ev.entity for ev in changes} == set(movers)
    for ev in changes:
        assert MC.parse(ev.extra["from"]).ua == "chrome/126"
        assert MC.parse(ev.extra["to"]).ua == "chrome/127"
        assert ev.extra["upgrade"] is True
    # later movers see the rollout itself
    late = [ev for ev in changes if ev.entity in movers[4:]]
    assert late and all(ev.extra["R"] >= 0.3 for ev in late)
    # the client p stays above 0.01 for every mover on every tick
    for ip in movers:
        assert pms[ip], ip
        assert min(pms[ip]) > 0.01, (ip, min(pms[ip]))
    assert t_warm < now


# ============================================================ spec test (c)
def test_c_copied_ua_with_linux_ja3_and_ttl64_is_inconsistent():
    store, eng = make_store(), ClientIdentityEngine()
    feed = Feed(store)
    peers = {f"10.0.0.{i}": CHROME126 for i in range(2, 6)}
    now = warm(feed, eng, {E: CHROME126, **peers}, 100)
    sysm = MC.get(store, S, "__system__")
    ja3_linux = MC.parse(_token(FORGED)).ja3n
    p, n_ua = MC.p_ja3n_given_ua(sysm, ja3_linux, "chrome/126")
    assert p == 0.0 and n_ua > 16.0

    fired = []
    for k in range(1, 6):
        now = T0 + (99 + k) * DT
        traffic = {e: [(st, times(now))] for e, st in peers.items()}
        traffic[E] = [(FORGED, times(now))]            # the legit browser is gone
        feed.tick(now, traffic)
        run_engine(eng, store, now)
        fired += events(store, "client_impersonation", E)
        if fired:
            break
    assert fired, MC.recent(MC.get(store, S, E))
    ev = fired[0]
    assert ev.extra["I"] == 1 and ev.extra["I_os"] and ev.extra["I_ja3n"]
    assert ev.extra["C"] == 0                          # no concurrency needed
    assert ev.extra["p_ja3n_given_ua"] < 0.01
    tok = ev.extra["stack"]
    assert MC.parse(tok).ua == "chrome/126" and MC.parse(tok).ttl == "64"


def _token(stack: tuple) -> str:
    from app.engines.behavior.lib.stack import stack_token
    ja3, ua, ttl, win = stack
    return stack_token(ja3, ua, ttl, win)
