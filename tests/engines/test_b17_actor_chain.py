"""Round 4: B17 actor chains (T19 IP hopping) through Fellegi-Sunter actor
evidence: a rare value the new address shares with a silent YOUNG predecessor
(frequency-based weight), the temporal handoff and the client stack, with
m / u learned by EM over every compared pair (lib/m_link.fs_em).

Why: an IP-hopping actor's previous address is itself new, so it has no
committed vocabulary / client model and is not enrolled in model.identity;
every modality term backed off to its class and the three hops of pack B's
T19 were never linked (T19 actor chain recovered in 0 runs). The hops share
the actor's private channel (a destination nobody else uses) and its device.
"""
from __future__ import annotations

import math

import numpy as np

from app.engines.behavior.lib import m_link as ML

import test_b17_entity_link as T17
from test_b17_entity_link import S, World

H1, H2, H3 = "10.0.0.70", "10.0.0.71", "10.0.0.72"
SYNC = {"POST ext-sync.example.io /sync|2xx": 4.0}


def _run_chain(share_token: bool):
    w = World().setup()
    rng = np.random.default_rng(5)
    base = w.p[T17.D1].mean
    # three addresses, three behaviours (a person by day, a script at night):
    # only the private channel and the device are common
    hops = [T17.Persona(np.random.default_rng(40 + i),
                        base + np.random.default_rng(60 + i).normal(0.0, 2.0, 52), 20 + i, [],
                        rate=12.0) for i in range(3)]
    dev = {T17.stack_tok(30, 0): 20.0}                     # the actor's one device
    for i, (e, p) in enumerate(zip((H1, H2, H3), hops)):
        for _ in range(8):
            w.now += T17.DT
            for x in T17.ENTS:
                w.feed(x, w.now, w.p[x])
            v, toks, _, sk = w.draw(p, rng)
            if share_token:
                toks = {**toks, **SYNC}
            w.feed(e, w.now, p, stacks=dev,
                   extra_raw={"act.tokens": toks})
            T17.run_engine(w.eng, w.store, w.now, dt=T17.DT)
    return w


def test_hops_sharing_a_private_channel_form_one_actor():
    w = _run_chain(share_token=True)
    links = ML.links(w.link())
    pairs = {(lk["from"], lk["to"]) for lk in links}
    assert (H1, H2) in pairs and (H2, H3) in pairs, [(lk["from"], lk["to"], lk["terms"])
                                                     for lk in links]
    act = ML.actor_of(w.link(), H3)
    assert act is not None and set(act["members"]) >= {H1, H2, H3}
    for lk in links:
        if lk["to"] in (H2, H3):
            g = lk["terms"]["gamma"]
            assert g["tok"] is True and any("ext-sync" in x for x in g["shared"])
            assert math.isfinite(lk["terms"]["lo_actor"]) and lk["terms"]["lo_actor"] >= ML.LINK_LO


def test_hops_without_a_shared_rare_value_are_not_chained():
    w = _run_chain(share_token=False)
    pairs = {(lk["from"], lk["to"]) for lk in ML.links(w.link())}
    assert (H1, H2) not in pairs and (H2, H3) not in pairs


def test_em_learns_that_common_agreements_carry_little_weight():
    # 200 compared pairs where the handoff and the stack agree for most non-matches
    obs = [[0, 1, 1]] * 120 + [[0, 0, 1]] * 60 + [[0, 1, 0]] * 15 + [[1, 1, 1]] * 5
    p = ML.fs_em(obs)
    w = ML.fs_weights(p)
    assert w["tok"][0] > 3.0                                  # rare shared value: strong
    assert abs(w["hand"][0]) < 1.0 and abs(w["stk"][0]) < 1.0  # common agreements: weak
    # a frequency-based u for a value nobody else uses outweighs the average u
    g = {"tok": True, "hand": True, "stk": True, "u_tok": 1e-3}
    assert ML.fs_score(g, w, p) > ML.fs_score({**g, "u_tok": None}, w, p) - 1e-9
