"""P07 round 9 (OI8-b): closure with one member seen once.

The round-4 closure test U < p_min (Good-Turing's unseen mass below the rarest
member's share) can never pass while ONE member is a singleton: U >= 1.5 / (N
+ 1) > 1 / N. Pack O, day 14: finance's approval opinions (0.8 / 0.1 / 0.1,
~20 arrivals on the approver's node; evidence 16.9 / 1.8 / 1.2 on seed 1)
stayed open on 4 of 5 seeds; the 60 研发 usernames (each user on 40 % of the
days) kept a singleton or two to day 21 (DEV.oa.login#0 missed on 5 of 5
seeds). A set now also closes when Chao1's estimate of the values not yet
seen (f1^2 / (2 f2)) is at most one half. Fails on round 8."""
from __future__ import annotations

import numpy as np

from app.engines.behavior.lib import pgrammar as PG
from app.engines.behavior.lib import pnode as PN

T0 = 1_788_220_800.0
DAY = 86400.0
OPIN = ["同意", "退回", "同意，请尽快办理"]


def test_opinions_with_one_rare_value_seen_once_close():
    ts = PN.TextSummary()
    seq = ["同意"] * 9 + ["退回"] + ["同意"] * 8 + ["退回", "同意，请尽快办理"]   # 17 / 2 / 1
    for i, v in enumerate(seq):
        ts.update(v, T0 + i * 6 * 3600.0, 1.0, 1.0)
    r = PG.fit_text(ts, T0 + len(seq) * 6 * 3600.0)
    assert r["U"] > r["p_min"]                       # the round-4 test alone: open
    assert r["closed"] == sorted(OPIN), r
    assert r["unseen_values"] <= 0.5
    # the stated U is Good-Turing's, unchanged
    assert abs(r["U"] - (1 + 0.5) / (len(seq) + 1.0)) < 0.01


def test_sixty_users_with_one_occasional_user_close():
    """59 研发 users seen on several days, one seen once (its first login)."""
    ts = PN.TextSummary()
    rng = np.random.default_rng(9)
    names = [f"dev{k:02d}" for k in range(60)]
    vcap = None
    for d in range(14):
        t = T0 + d * DAY + 12 * 3600
        today = [n for n in names[:59] if rng.random() < 0.5]
        if d == 12:
            today.append(names[59])
        for i, v in enumerate(today):
            ts.update(v, t - 3600 + i * 20.0, 1.0, 1.0)
        vcap = PG.adapt_values(ts, t, vcap)
    rec = PG.fit_text(ts, t, vcap=vcap)
    assert rec.get("closed") == sorted(names), (rec.get("U"), rec.get("p_min"), rec.get("unseen_values"))


def test_two_rare_values_each_seen_once_stay_open():
    ts = PN.TextSummary()
    seq = ["同意"] * 16 + ["退回", "同意，请尽快办理"]
    for i, v in enumerate(seq):
        ts.update(v, T0 + i * 6 * 3600.0, 1.0, 1.0)
    r = PG.fit_text(ts, T0 + len(seq) * 6 * 3600.0)
    assert "closed" not in r and r["unseen_values"] > 0.5
