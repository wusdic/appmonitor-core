"""lib/m_ptree, round 7 (evaluator; the conformity owner's OC7-2, cost):
hierarchies() reuses the Regions of the previous call while the config object
and P11's covers dict are the same objects and who_groups' version is
unchanged, and rebuilds them when any of the three changes. Same answers as a
fresh build."""
from __future__ import annotations

from app.engines.behavior.lib import m_ptree as MP
from app.engines.behavior.lib.phier import Regions
from app.models.schema import ORG
from helpers import make_store


def _wg(store, covers, version):
    store.put_model(ORG, ORG, MP.WHO_GROUPS, {"covers": covers, "version": version, "ip2g": {}},
                    version=version, ts=0.0)


def test_regions_are_reused_until_the_config_or_the_covers_change():
    st = make_store()
    cfg = {"progressive": {"enabled": True}}
    covers = {"g1": ["10.1.0.0/16"], "g2": ["10.2.0.0/16"]}
    _wg(st, covers, 1)
    h1 = MP.hierarchies(st, "oa", cfg)
    h2 = MP.hierarchies(st, "portal", cfg)
    assert h1.regions is h2.regions
    fresh = Regions([(g, c) for g, c in covers.items()])
    for ip in ("10.1.3.4", "10.2.0.1", "10.3.0.1"):
        assert h1.regions.of(ip) == fresh.of(ip)
    # P11 re-clusters: a new covers dict and a new version
    covers2 = {"g1": ["10.1.0.0/16", "10.3.0.0/16"]}
    _wg(st, covers2, 2)
    h3 = MP.hierarchies(st, "oa", cfg)
    assert h3.regions is not h1.regions
    assert h3.regions.of("10.3.0.1") == Regions([(g, c) for g, c in covers2.items()]).of("10.3.0.1")
    # a different config object with configured classes wins over the covers
    cfg2 = {"progressive": {"enabled": True}, "ip_classes": [{"name": "lab", "cidrs": ["10.3.0.0/16"]}]}
    h4 = MP.hierarchies(st, "oa", cfg2)
    assert h4.regions.of("10.3.0.1") == Regions.from_config(cfg2).of("10.3.0.1")
    assert MP.hierarchies(st, "oa", cfg).regions.of("10.3.0.1") == h3.regions.of("10.3.0.1")
