"""Round 7 (evaluator): P14 group-view numbers must not depend on Python's
per-process str-hash salt (PYTHONHASHSEED).

Pack O seed 0 run under PYTHONHASHSEED 0 and 1 (docs/lib3/progressive.md
§16.15): the scored JSON was identical, but 8 group-view statements of
class:grp:G69 (portal POST /comment) differed in the last digit of their
support / n_obs on days 11 and 19. views._group_mass summed the member IPs'
counts while iterating a set of member strings, whose order follows the hash
salt; a float sum depends on its order. The test runs the same computation in
fresh interpreters with different hash seeds and requires identical output
(it fails on the round-7 checkpoint code)."""
from __future__ import annotations

import os
import subprocess
import sys
import textwrap

BACKEND = os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..", "backend"))
SEEDS = ("0", "1", "2", "3", "77", "1234")


def _outputs(code: str) -> set:
    outs = set()
    for seed in SEEDS:
        env = dict(os.environ, PYTHONHASHSEED=seed, OMP_NUM_THREADS="1")
        env["PYTHONPATH"] = BACKEND
        r = subprocess.run([sys.executable, "-c", textwrap.dedent(code)], env=env,
                           capture_output=True, text=True, timeout=300)
        assert r.returncode == 0, r.stderr[-2000:]
        outs.add(r.stdout)
    return outs


def test_group_mass_from_member_ips_does_not_depend_on_hash_seed():
    """The member-IP fallback of _group_mass (the group key is not in the
    node's grp-level summary): its sum must be taken in a fixed order."""
    code = """
    from types import SimpleNamespace
    from app.engines.behavior import views as V
    from app.engines.behavior.lib.psketch import DecayedSpaceSaving
    lv = [DecayedSpaceSaving(64) for _ in range(4)]
    ips = ["192.168.%d.%d" % (a, b) for a in (1, 2, 3) for b in range(10, 17)]
    for i, ip in enumerate(ips):
        # weights whose float sum depends on the order of the additions
        lv[0].add(ip, 0.0, w=0.1 * (1 + i % 3) + 1e-13 * i)
    lv[0].add("10.9.9.9", 0.0, w=7.0)
    lv[3].add("grp:Gother", 0.0, w=1.0)
    nd = SimpleNamespace(who=SimpleNamespace(levels=lv))
    for n in (5, 9, 14, 21):
        print(V._group_mass(nd, "G1", set(ips[:n]), 0.0).hex())
    """
    outs = _outputs(code)
    assert len(outs) == 1, outs
