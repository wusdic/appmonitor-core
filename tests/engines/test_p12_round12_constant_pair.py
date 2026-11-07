"""P12, round 12 (learning owner's proposal for system_profile._constant_pair)."""
from types import SimpleNamespace as NS


def test_a_one_source_one_value_pair_is_no_measurement_of_p08():
    """Pack O seed 0 (round 12): finance's first judged P08 records were
    one-source 'net.src -> body.fmt' pairs of young nodes (one IP, value
    'form', gain 0). _constant_pair needed two sources holding the same value,
    so these counted: P12 measured P08 at 0 bits for ~200 us/event from day
    4-5 and switched it off on day 10 when a P04 change made one such record
    judged a day earlier - before the users' username pair reached n_bind, so
    FIN.finance.login#0's bindings were never stated. One source with one
    value binds nothing either; a pair whose sources hold different values is
    a measurement as before."""
    from app.engines.behavior.system_profile import _constant_pair, fitted_gain
    node = NS(parent=None, mass_at=lambda t: 100.0)
    leaf = NS(parent=0, mass_at=lambda t: 10.0)
    ptm = NS(kinds={0: NS(root=0, nodes={0: node, 1: leaf})})
    one = {"gain": 0.0, "fd": {"judged": 1}, "table": {"192.168.2.10": {"top": "form", "n": 9.0}}}
    assert _constant_pair(one)
    assert fitted_gain({"nodes": {0: {1: {"pairs": {"net.src->body.fmt": one}}}}}, ptm, 0.0) is None
    two = {"gain": 0.0, "fd": {"judged": 2},
           "table": {"192.168.2.10": {"top": "lucy", "n": 9.0}, "192.168.2.11": {"top": "tom", "n": 9.0}}}
    assert not _constant_pair(two)
    assert fitted_gain({"nodes": {0: {1: {"pairs": {"net.src->body.kv.username": two}}}}}, ptm, 0.0) == 0.0
    done = dict(two, gain=1.5)
    assert fitted_gain({"nodes": {0: {1: {"pairs": {"a": one, "b": done}}}}}, ptm, 0.0) == 0.15
