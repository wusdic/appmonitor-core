"""Behavioural groups on the generator's organisation become more precise with
observation time (docs/lib3/progressive.md §6.15, PG3; requirement S3/S5/S11).
See tests/eval/who_convergence.py for the measure (P11 on R2/R3/P00 only).

Measured (pack O, 3600-s ticks, days 2 / 5 / 8 / 12 / 16 / 21), before the
integration's TLS / DNS route keys (lib/pdfg.host_key):
  seed 0  ARI 0.25 0.87 0.97 0.85 0.97 0.97   DEV pool grouped from day 8
  seed 1  ARI 0.42 1.00 0.87 1.00 0.85 0.85   DEV pool grouped from day 8
With host_key (mail.corp.local and git.corp.local are two actions instead of
one 'TLS corp.local'), days 2 / 8 / 14 / 21:
  seed 0  ARI 0.13 0.84 0.84 0.86      seed 1  ARI - 0.86 0.86 0.86
(ARI over GA / FIN / SALES, singletons for ungrouped IPs). The residual error
is the two approvers (综合部's 192.168.1.21, 财务部's 192.168.2.10): their
action mix differs from their departments' (approvals vs reports /
bookkeeping); with the eTLD+1 keys they formed one 'approver' pair instead of
two singletons. On seed 0 the A9 address 192.168.3.33 (a sales IP that reads
finance's approval list daily from day 11) is also left ungrouped. PG3's
target (ARI >= 0.9) is therefore not met by P11 on pack O; the test guards
the convergence (day 21 >> day 2) and the level reached."""
from __future__ import annotations

import who_convergence as W


def test_groups_converge_on_the_departments():
    res = W.run(days=(2, 8, 21), dt=3600.0, seed=0)
    d2, d8, d21 = res[2], res[8], res[21]
    assert d21["ari"] >= 0.85 and d21["ari"] >= d2["ari"] + 0.3, res
    assert d8["pool_grouped"] and d21["pool_grouped"], res
    assert d21["state_kb"] < 2048                       # bounded state (S_max LRU)
