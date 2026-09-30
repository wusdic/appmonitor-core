"""lib/pparse.py: body / query / header parsing, the value retention policy,
who resolution behind trusted proxies and session keys (§5.1, §6.2.1)."""
from __future__ import annotations

import json

import pytest

from app.engines.behavior.lib import pevent as EV
from app.engines.behavior.lib import pparse as PP

POLICY = PP.ValuePolicy(EV.pconfig({})["value_policy"])


def _apply(attrs):
    modes = {}
    POLICY.apply_all(attrs, modes)
    return attrs, modes


def test_form_body_keys_values_and_secret_shape():
    """P00 (a): username=jack&password=x&captcha=1234."""
    out, trunc = PP.parse_l7({"body": "username=jack&password=x&captcha=1234",
                              "headers": {"Content-Type": "application/x-www-form-urlencoded"}})
    assert not trunc
    assert out["body.fmt"] == "form"
    assert out["body.keys"] == frozenset({"captcha", "password", "username"})
    assert out["body.len"] == len("username=jack&password=x&captcha=1234")
    attrs, modes = _apply(out)
    assert attrs["body.kv.username"] == "jack"
    assert attrs["body.kv.password"] == "L1" and modes["body.kv.password"] == "shape"
    assert attrs["body.kv.captcha"] == "D4"
    assert attrs["hdr.content-type"] == "application/x-www-form-urlencoded"


def test_json_body_flattens_arrays():
    """P00 (b): nested arrays flatten to items[].name."""
    body = json.dumps({"order": {"id": 17, "items": [{"name": "pen", "qty": 2},
                                                     {"name": "ink", "qty": 1}]},
                       "tags": ["a", "b"]})
    out, _ = PP.parse_l7({"body": body})
    assert out["body.fmt"] == "json"
    assert out["body.kv.order.items[].name"] == frozenset({"pen", "ink"})
    assert out["body.kv.order.items[].n"] == 2
    assert out["body.kv.tags[]"] == frozenset({"a", "b"})
    assert "order.items[].name" in out["body.keys"] and "order.id" in out["body.keys"]


def test_multipart_xml_text_and_key_cap():
    mp = ("--XyZ\r\nContent-Disposition: form-data; name=\"title\"\r\n\r\nreport q3\r\n"
          "--XyZ\r\nContent-Disposition: form-data; name=\"file\"; filename=\"r.pdf\"\r\n"
          "Content-Type: application/pdf\r\n\r\n%PDF-1.4 binary\r\n--XyZ--\r\n")
    out, _ = PP.parse_l7({"body": mp, "headers": {"content-type": "multipart/form-data; boundary=XyZ"}})
    assert out["body.fmt"] == "multipart" and out["body.kv.title"] == "report q3"
    assert out["body.kv.file.len"] == len("%PDF-1.4 binary")
    assert out["body.kv.file.ctype"] == "application/pdf"
    x, _ = PP.parse_l7({"body": "<req><user>jack</user><op/></req>"})
    assert x["body.paths"] == frozenset({"req", "req/user", "req/op"})
    t, _ = PP.parse_l7({"body": "please approve order 12345 today"})
    assert t["body.tpl"] == "please approve order {num} today"
    many = "&".join(f"k{i}=v" for i in range(80))
    c, _ = PP.parse_l7({"body": many}, k_max=64)
    assert len(c["body.keys"]) == 64 and c["body.keys_extra"] == 16


def test_query_headers_session_key():
    out, trunc = PP.parse_l7({"query": "page=2&q=report", "body_trunc": True,
                              "headers": {"Cookie": "JSESSIONID=abc123; lang=zh",
                                          "X-Client-Ver": "3.1"}},
                             session_cookies=("JSESSIONID",), hmac_key="k")
    assert trunc and out["q.keys"] == frozenset({"page", "q"}) and out["q.kv.q"] == "report"
    assert "hdr.cookie" not in out and out["hdr.x-client-ver"] == "3.1"
    assert out["sess.key"] == PP.hmac12("k", "abc123") and len(out["sess.key"]) == 12


def test_value_policy_random_long_hmac_rules():
    """P00 (h): a 40-character random value under an unknown key and a
    200-character comment are kept as shape (+ length), username=jack clear."""
    rnd = "aZ3kP9qX2mL7vB4nR8tY1wE6uI0oS5dF3gH7jK2l"
    attrs, modes = _apply({"body.kv.zz": rnd, "body.kv.comment": "word " * 40,
                           "body.kv.username": "jack", "tls.ja3": "e7d705a3286e19ea42f587b344ee6865",
                           "hdr.authorization": "Bearer abc"})
    assert modes["body.kv.zz"] == "shape" and attrs["body.kv.zz.len"] == 40
    assert modes["body.kv.comment"] == "shape" and attrs["body.kv.comment.len"] == 200
    assert attrs["body.kv.username"] == "jack" and "body.kv.username" not in modes
    assert attrs["tls.ja3"] == "e7d705a3286e19ea42f587b344ee6865"      # structural: untouched
    assert modes["hdr.authorization"] == "shape"
    pol = PP.ValuePolicy({"rules": [("body.kv.username", "hmac")], "hmac_key": "k"})
    v, m = pol.apply("body.kv.username", "jack")
    assert m == "hmac" and v == "h:" + PP.hmac12("k", "jack")
    assert pol.apply("body.kv.tags[]", frozenset({"a"}))[1] == "clear"


def test_resolve_who_behind_trusted_proxy():
    """P00 (g): x-forwarded-for: 10.1.2.3, 192.168.0.9 (the latter trusted)."""
    tn = PP.TrustedNets(["192.168.0.0/24"])
    ip, peer = PP.resolve_who("192.168.0.5", {"X-Forwarded-For": "10.1.2.3, 192.168.0.9"}, tn)
    assert (ip, peer) == ("10.1.2.3", "192.168.0.5")
    assert PP.resolve_who("172.16.0.1", {"x-forwarded-for": "6.6.6.6"}, tn) == ("172.16.0.1", None)
    assert PP.resolve_who("192.168.0.5", {"forwarded": 'for="[2001:db8::7]:4711";proto=https'},
                          tn) == ("2001:db8::7", "192.168.0.5")
    assert PP.resolve_who("192.168.0.5", {}, tn) == ("192.168.0.5", None)
    assert PP.resolve_who("192.168.0.5", {"x-forwarded-for": "10.0.0.1"},
                          PP.TrustedNets()) == ("192.168.0.5", None)


def test_structured_text_reparse():
    out = PP.parse_structured_text("hdr.x-prefs", "lang=zh;tz=8", "form")
    assert out["hdr.x-prefs.keys"] == frozenset({"lang", "tz"})
    assert out["hdr.x-prefs.kv.lang"] == "zh"
