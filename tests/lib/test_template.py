"""Tests for engines/behavior/lib/template.py: segment masks, token builders
(HTTP / TLS / DNS / L4), token inspection, the Drain-lite prefix tree with
provisional literals, crowded merge and {var} collapse, the Space-Saving
vocabulary and serialisation.

The R2 behaviour targets from docs/lib3/engines.md are checked directly:
3000 /orders/view/<id> + 2000 random 8-char 404 paths give <= 60 templates
and /orders/view/{num} holds >= 95 % of the orders mass (whatever the
maintenance schedule); DNS random labels become {rnd}; the vocabulary is
capped with Space-Saving eviction to {rare}. Space-Saving is compared step
by step against a brute-force reference (argmin scan) so the lazy heap
cannot drift. Everything is seeded and small (< 1 s for the file).
"""
from __future__ import annotations

import copy
import json
import math
import os
import random
import string
import sys
import time
from collections import Counter

import pytest

sys.path.insert(0, os.path.join(os.path.dirname(__file__), "..", "..", "backend"))

from app.engines.behavior.lib import template as T  # noqa: E402

NAN = float("nan")
INF = float("inf")


# ------------------------------------------------------------------ helpers
def _rand_str(rng: random.Random, alphabet: str, n: int) -> str:
    return "".join(rng.choice(alphabet) for _ in range(n))


def _warm(t: T.Templater, path: str, host: str = "erp.corp", method: str = "GET", n: int = 3):
    """Hit a path n times so its literals are established (count > 2)."""
    out = None
    for _ in range(n):
        out = t.template_path(host, method, path)
    return out


def _root(t: T.Templater, host: str = "erp.corp", method: str = "GET") -> T._Node:
    return t.trees[(host, method)]


def _n_nodes(node: T._Node) -> int:
    return 1 + sum(_n_nodes(c) for c in node.children.values())


def _ref_space_saving(stream, cap):
    """Brute-force Space-Saving: evict argmin (count, id); new = c_min + w, err = c_min."""
    vocab, next_id = {}, 1
    for tok, w in stream:
        if tok in vocab:
            vocab[tok][1] += w
            continue
        c_min = 0.0
        if len(vocab) >= cap:
            victim = min(vocab, key=lambda k: (vocab[k][1], vocab[k][0]))
            c_min = vocab.pop(victim)[1]
        vocab[tok] = [next_id, c_min + w, c_min]
        next_id += 1
    return vocab, next_id


# ------------------------------------------------------------------ constants
def test_frozen_constants():
    assert T.MASK_TOKENS == ("{uuid}", "{email}", "{date}", "{hex}", "{num}", "{id}", "{tok}")
    assert T.OBJ_MASKS == {"{num}", "{id}", "{uuid}"}
    assert (T.VAR_TOKEN, T.RND_TOKEN, T.RARE_TOKEN) == ("{var}", "{rnd}", "{rare}")
    assert (T.MAX_DEPTH, T.VAR_MIN_CHILDREN, T.VAR_MAX_TOP_SHARE, T.MERGE_MAX_COUNT,
            T.VOCAB_CAP) == (6, 40, 0.5, 2, 4000)
    assert (T.DNS_RND_ENTROPY, T.DNS_RND_LEN, T.DNS_RND_DIGIT_RATIO, T.TOK_ENTROPY) == \
        (3.2, 20, 0.3, 3.5)
    assert T.WRITE_METHODS <= T.HTTP_METHODS
    assert T.SERVICE_PORTS[22] == "ssh" and T.SERVICE_PORTS[443] == "https"


# ------------------------------------------------------------------ masks
@pytest.mark.parametrize("seg, mask", [
    ("550e8400-e29b-41d4-a716-446655440000", "{uuid}"),
    ("550E8400-E29B-41D4-A716-446655440000", "{uuid}"),
    ("john.doe+x@example.com", "{email}"),
    ("john%40example.com", "{email}"),
    ("2024-01-15", "{date}"),
    ("2024_01_15", "{date}"),
    ("2024.1.5", "{date}"),
    ("2024-01", "{date}"),
    ("15-01-2024", "{date}"),
    ("01.15.2024", "{date}"),
    ("2024-01-15T10:30:00Z", "{date}"),
    ("2024-01-15T10:30:00.123+08:00", "{date}"),
    ("20240115", "{date}"),
    ("20240115T103000Z", "{date}"),
    ("deadbeefcafebabe0123", "{hex}"),
    ("d41d8cd98f00b204e9800998ecf8427e", "{hex}"),
    ("DEADBEEF01234567", "{hex}"),
    ("7", "{num}"),
    ("123456", "{num}"),
    ("12345678", "{num}"),               # year 1234: not a date
    ("20241399", "{num}"),               # month 13: not a date
    ("1234567890123456789", "{num}"),     # all digits: a snowflake id, not a digest
    ("ORD20240115", "{id}"),
    ("a8x9k2m1", "{id}"),
    ("office365", "{id}"),               # the spec rule, verbatim
    ("E16Q16_5yhgKuKH1kCr49A", "{tok}"),
    ("eyJhbGciOiJIUzI1NiJ9.eyJzdWIiOiIxMjM0NTY3ODkwIn0.Sfl-KxwRJ_SMeKKF2QT4", "{tok}"),
    ("x7Kp2Qm9Lz4W-Rt", "{tok}"),
])
def test_mask_segment_rules(seg, mask):
    m, v = T.mask_segment(seg)
    assert m == mask, (seg, m)
    assert v is not None


@pytest.mark.parametrize("seg", [
    "orders", "view", "v2", "api2", "login", "index.html", "favicon.ico",
    "getUserProfileByEmail", "ProductCatalogService", "GetCustomerOrderHistory",
    "customer-relationship-management", "employee-onboarding-checklist",
    "how-to-configure-the-server-for-production-use", "main.bundle.js",
    "jquery-3.6.0.min.js", "report_2024-01-15.pdf", "bootstrap.min.css",
    "XMLHttpRequestHandler", "user_preferences_v2", "deadbeefdeadbeef",
    "abcdefabcdefab1", "2024-13-01", "abc@", "v1.2.3", ".", "..", "listOrdersByCustomerId",
])
def test_mask_segment_keeps_literals(seg):
    m, v = T.mask_segment(seg)
    assert v is None and m == seg


def test_mask_segment_extension_and_values():
    assert T.mask_segment("12345.jpg") == ("{num}.jpg", "12345")
    assert T.mask_segment("2024-01-15.csv") == ("{date}.csv", "2024-01-15")
    assert T.mask_segment("zYxWzYxW-zYxWzYxW.png") == ("{tok}.png", "zYxWzYxW-zYxWzYxW")
    # whole-segment rules come first: a high-entropy name is a token as a whole
    assert T.mask_segment("E16Q16_5yhgKuKH1kCr49A.png")[0] == "{tok}"
    assert T.mask_segment("photo.jpg") == ("photo.jpg", None)
    assert T.mask_segment("12345.678") == ("12345.678", None)      # numeric "extension" is not one
    # the value is the percent-decoded original
    assert T.mask_segment("a%40b.com") == ("{email}", "a@b.com")


def test_mask_segment_sanitises_literals():
    # a literal can never alias a mask token nor break the token grammar
    assert T.mask_segment("{num}") == ("%7Bnum%7D", None)
    assert T.mask_segment("{var}")[0] == "%7Bvar%7D"
    m, _ = T.mask_segment("a b|c&d")
    assert " " not in m and "|" not in m and "&" not in m
    long = "abc-" * 40
    m, v = T.mask_segment(long)
    assert v is None and len(m) == 64 and m.endswith("...")
    assert T.mask_segment("") == ("", None)
    assert T.mask_segment(None) == ("", None)


def test_tok_detection_rates():
    """Random base64url / base62 ids are masked ({tok} or {id}); identifiers are not."""
    rng = random.Random(7)
    b64 = string.ascii_letters + string.digits + "-_"
    masked = 0
    n = 400
    for _ in range(n):
        m, _ = T.mask_segment(_rand_str(rng, b64, 22))
        masked += m in ("{tok}", "{id}")
    assert masked / n >= 0.90                           # measured 0.94 over 20k
    b62 = string.ascii_letters + string.digits
    masked = sum(T.mask_segment(_rand_str(rng, b62, 32))[0] in ("{tok}", "{id}") for _ in range(n))
    assert masked / n >= 0.95                           # measured 0.98 over 20k


# ------------------------------------------------------------------ mask_path
def test_mask_path_basic_and_query_names():
    assert T.mask_path("/orders/view/123") == (["orders", "view", "{num}"], [], [("{num}", "123")])
    segs, params, masked = T.mask_path("/search?q=x&page=2&q=y")
    assert segs == ["search"] and params == ["page", "q"] and masked == []
    assert T.mask_path("/search?")[1] == []
    assert T.mask_path("/x?=v&&a")[1] == ["a"]
    assert T.mask_path("/x?1699999999")[1] == ["{num}"]          # cache-buster name masked
    many = "&".join(f"p{i:02d}=1" for i in range(40))
    assert len(T.mask_path("/x?" + many)[1]) == 16


def test_mask_path_normalisation():
    assert T.mask_path("/a//b/")[0] == ["a", "b"]
    assert T.mask_path("/a/b#frag?x=1")[0:2] == (["a", "b"], [])
    assert T.mask_path("http://erp.corp:8080/a/7?x=1")[0:2] == (["a", "{num}"], ["x"])
    assert T.mask_path("/app;jsessionid=ABCDEF0123456789/page")[0] == ["app", "page"]
    assert T.mask_path("") == ([], [], [])
    assert T.mask_path("/") == ([], [], [])
    assert len(T.mask_path("/a" * 100)[0]) == 32
    segs, _, masked = T.mask_path("/u/550e8400-e29b-41d4-a716-446655440000/f/12.jpg")
    assert segs == ["u", "{uuid}", "f", "{num}.jpg"]
    assert masked == [("{uuid}", "550e8400-e29b-41d4-a716-446655440000"), ("{num}", "12")]


def test_mask_path_returns_fresh_lists():
    a = T.mask_path("/static/app.css?v=1")
    a[0].append("junk")
    a[1].append("junk")
    b = T.mask_path("/static/app.css?v=1")
    assert b == (["static", "app.css"], ["v"], [])


# ------------------------------------------------------------------ small tokens
@pytest.mark.parametrize("status, cls", [
    (200, "2xx"), (204, "2xx"), (101, "1xx"), (301, "3xx"), (404, "4xx"), (503, "5xx"),
    (599, "5xx"), (0, "0xx"), (None, "0xx"), (-1, "0xx"), (99, "0xx"), (600, "0xx"),
    (NAN, "0xx"), (INF, "0xx"), ("302", "3xx"), ("abc", "0xx"), (True, "0xx"), (404.0, "4xx"),
])
def test_status_class(status, cls):
    assert T.status_class(status) == cls


def test_size_class_exact_floor_log2():
    assert T.size_class(0, 0) == "u0/d0"
    assert T.size_class(1, 1) == "u0/d0"
    assert T.size_class(1024, 1023) == "u10/d9"
    assert T.size_class(NAN, INF) == "u0/d0"
    assert T.size_class(-5, "x") == "u0/d0"
    assert T.size_class(None, 3) == "u0/d1"
    for k in range(1, 54):                            # 2^k - 1 is exact in float64 up to k = 53
        assert T.size_class(2.0 ** k, 2 ** k - 1) == f"u{k}/d{k - 1}"
    assert T.size_class(2 ** 200, 0) == "u200/d0"
    assert T.size_class(1e300, 2) == f"u{math.floor(math.log2(1e300))}/d1"


@pytest.mark.parametrize("label, rnd", [
    ("www", False), ("mail", False), ("googleapis", False), ("accounts", False),
    ("a" * 20, True),                     # length rule
    ("a" * 19, False),
    ("aaaaaaa111", True),                 # digit ratio exactly 0.3
    ("aaaaaaaa11", False),                # 0.2, low entropy
    ("web01", True),
    ("qwertyuiop", True),                 # 10 distinct chars: 3.32 bits
    ("safebrowsing", True),               # 3.42 bits: the spec rule, verbatim
    ("", False),
])
def test_dns_label_is_random(label, rnd):
    assert T.dns_label_is_random(label) is rnd


def test_dns_template_and_token():
    assert T.dns_token("TXT", "a8f3k2l9d0s8f7g6h5j4k3.x.evil.com") == "TXT {rnd}.x.evil.com"
    # consecutive random labels collapse into one {rnd}
    q = "a8f3k2l9d0s8f7g6h5j4k3.b7c2x9q4w8e1r5t6y7u.x.evil.com"
    assert T.dns_template(q) == "{rnd}.x.evil.com"
    assert T.dns_template("WWW.Example.COM.") == "www.example.com"
    assert T.dns_template("example.com") == "example.com"
    assert T.dns_template("login.portal.example.co.uk") == "login.portal.example.co.uk"
    assert T.dns_template("10.0.0.1") == "10.0.0.1"
    assert T.dns_template("") == "."
    assert T.dns_template("4.3.2.10.in-addr.arpa") == "{rnd}.in-addr.arpa"
    assert T.dns_token("a", "www.example.com") == "A www.example.com"
    assert T.dns_token("16", "x.com") == "TXT x.com"
    assert T.dns_token("65", "x.com") == "HTTPS x.com"
    assert T.dns_token("99", "x.com") == "TYPE99 x.com"
    assert T.dns_token("", "x.com") == "UNK x.com"
    assert T.dns_token(None, "x.com") == "UNK x.com"


def test_tls_and_l4_tokens():
    assert T.tls_token("cdn.a.example.com", 443, 1024, 16384) == "example.com:443 u10/d14"
    assert T.tls_token("x.example.com.cn", 8443, 0, 1) == "example.com.cn:8443 u0/d0"
    assert T.tls_token("", 443, 1, 1) == "-:443 u0/d0"
    assert T.tls_token("2001:db8::1", 443, 2, 2) == "[2001:db8::1]:443 u1/d1"
    assert T.tls_token("x.com", "bad", 2, 2) == "x.com:0 u1/d1"
    assert T.l4_token("tcp", 22, 4096, 512) == "tcp/ssh u12/d9"
    assert T.l4_token("UDP", 53, 60, 120) == "udp/dns u5/d6"
    assert T.l4_token("tcp", 31337, 1, 1) == "tcp/31337 u0/d0"
    assert T.l4_token("", 0, 1, 1) == "ip/0 u0/d0"


# ------------------------------------------------------------------ inspection
@pytest.mark.parametrize("token, ch, fam", [
    ("GET erp.corp /orders/view/{num}|2xx", "http", "http|read|erp.corp|orders"),
    ("POST erp.corp /api/v1/orders?id&q|4xx", "http", "http|write|erp.corp|api"),
    ("OTHER erp.corp /dav/{var}|2xx", "http", "http|other|erp.corp|dav"),
    ("HEAD erp.corp /|2xx", "http", "http|read|erp.corp|"),
    ("TXT {rnd}.x.evil.com", "dns", "dns|txt|evil.com|x"),
    ("A www.example.com|NXDOMAIN", "dns", "dns|addr|example.com|www"),
    ("MX example.com", "dns", "dns|other|example.com|"),
    ("example.com:443 u10/d14", "tls", "tls|conn|example.com|443"),
    ("[2001:db8::1]:443 u1/d1|ok", "tls", "tls|conn|[2001:db8::1]|443"),
    ("tcp/ssh u12/d9", "l4", "l4|tcp||ssh"),
    ("udp/5353 u1/d1|reset", "l4", "l4|udp||5353"),
    ("{rare}", "rare", "rare|||"),
    ("{rare:http}", "http", "http|||"),
    ("{rare:bogus}", "rare", "rare|||"),
    ("", "rare", "rare|||"),
    ("hello", "rare", "rare|||"),
])
def test_channel_of_and_token_family(token, ch, fam):
    assert T.channel_of(token) == ch
    assert T.token_family(token) == fam


def test_channel_of_round_trips_builders():
    t = T.Templater()
    for m in ("GET", "post", "PROPFIND", "BREW"):
        tok, _ = t.http_token(m, "Erp.Corp:8080", "/a/b?x=1", 200)
        assert T.channel_of(tok) == "http"
    assert T.channel_of(T.tls_token("a.b.example.org", 443, 5, 5)) == "tls"
    assert T.channel_of(T.dns_token("TXT", "abcdefghij0123456789xyz.t.evil.com")) == "dns"
    assert T.channel_of(T.dns_token("TYPE65534", "x.com")) == "dns"
    assert T.channel_of(T.l4_token("tcp", 3389, 1e6, 1e3)) == "l4"


# ------------------------------------------------------------------ templater
@pytest.mark.parametrize("maintain_every", [None, 1, 250])
@pytest.mark.parametrize("alphabet", [string.ascii_lowercase,
                                      string.ascii_lowercase + string.digits])
def test_orders_flood_target(maintain_every, alphabet):
    """R2 unit test (a): 3000 /orders/view/<id> + 2000 random 8-char 404 paths
    -> <= 60 templates and /orders/view/{num} >= 95 % of the orders mass."""
    rng = random.Random(42)
    reqs = [("orders", f"/orders/view/{rng.randrange(1, 10 ** 7)}", 200) for _ in range(3000)]
    reqs += [("junk", "/" + _rand_str(rng, alphabet, 8), 404) for _ in range(2000)]
    rng.shuffle(reqs)
    t = T.Templater()
    templates = Counter()
    orders = Counter()
    for i, (kind, path, st) in enumerate(reqs):
        tok, masked = t.http_token("GET", "erp.corp", path, st)
        tpl = tok.split(" ", 2)[2].rsplit("|", 1)[0]
        templates[tpl] += 1
        if kind == "orders":
            orders[tpl] += 1
            assert masked and masked[0][0] == "{num}"
        if maintain_every and (i + 1) % maintain_every == 0:
            t.maintain()
    assert len(templates) <= 60, templates.most_common(10)
    assert orders["/orders/view/{num}"] / 3000 >= 0.95
    t.maintain()
    assert _n_nodes(_root(t)) <= 20                    # the junk was merged away


def test_provisional_literal_promotes_on_third_hit():
    t = T.Templater()
    assert t.template_path("h", "GET", "/newpage")[0] == "/{var}"
    assert t.template_path("h", "GET", "/newpage")[0] == "/{var}"
    assert t.template_path("h", "GET", "/newpage")[0] == "/newpage"
    # a weighted (aggregated) observation establishes it at once
    assert t.template_path("h", "GET", "/other", w=5)[0] == "/other"
    # masks are never provisional
    assert T.Templater().template_path("h", "GET", "/{0}".format(42))[0] == "/{num}"
    # separate trees per (host, method)
    assert t.template_path("h", "POST", "/newpage")[0] == "/{var}"
    assert t.template_path("h2", "GET", "/newpage")[0] == "/{var}"


def test_templates_do_not_depend_on_uncrowded_maintenance():
    """Without crowding the template stream is identical whatever the schedule."""
    rng = random.Random(3)
    words = ["orders", "view", "list", "hr", "leave", "fin", "ledger", "crm", "lead", "report"]
    paths = ["/" + "/".join(rng.choice(words) for _ in range(rng.randint(1, 4)))
             + (f"/{rng.randrange(1000)}" if rng.random() < 0.3 else "") for _ in range(1500)]
    a, b = T.Templater(), T.Templater()
    for i, p in enumerate(paths):
        ta = a.template_path("h", "GET", p)[0]
        b.maintain()
        tb = b.template_path("h", "GET", p)[0]
        assert ta == tb, (i, p)


def test_uncrowded_sparse_path_accumulates_across_passes():
    """An endpoint hit once per tick (maintain every tick) still gets its name:
    rare siblings are merged only in crowded nodes."""
    t = T.Templater()
    _warm(t, "/reports/daily", n=10)
    seen = []
    for _ in range(4):
        seen.append(t.template_path("erp.corp", "GET", "/reports/monthly")[0])
        t.maintain()
    assert seen == ["/reports/{var}", "/reports/{var}", "/reports/monthly", "/reports/monthly"]


def test_crowded_merge_keeps_established_and_reports_same():
    t = T.Templater()
    for name in ("dashboard", "orders", "profile"):
        _warm(t, f"/{name}", n=10)
    rng = random.Random(1)
    junk = ["/" + _rand_str(rng, string.ascii_lowercase, 10) for _ in range(60)]
    before = [t.template_path("erp.corp", "GET", p)[0] for p in junk]
    assert set(before) == {"/{var}"}
    assert len(_root(t).children) > T.VAR_MIN_CHILDREN
    t.maintain()
    root = _root(t)
    assert set(root.children) == {"dashboard", "orders", "profile", "{var}"}
    assert not root.is_var
    assert root.children["{var}"].count == 60
    assert t.template_path("erp.corp", "GET", "/orders")[0] == "/orders"
    assert t.template_path("erp.corp", "GET", junk[0])[0] == "/{var}"
    # counts are conserved by the merge
    assert root.count == sum(c.count for c in root.children.values())


def test_var_collapse_of_variable_position():
    t = T.Templater()
    rng = random.Random(5)
    users = [_rand_str(rng, string.ascii_lowercase, 6) for _ in range(60)]
    for u in users:
        _warm(t, f"/users/{u}/profile", n=4)
    # established literals are reported literally until the pass
    assert t.template_path("erp.corp", "GET", f"/users/{users[0]}/profile")[0] == \
        f"/users/{users[0]}/profile"
    t.maintain()
    node = _root(t).children["users"]
    assert node.is_var and set(node.children) == {"{var}"}
    assert node.children["{var}"].children["profile"].count == 60 * 4 + 1
    for p in (f"/users/{users[3]}/profile", "/users/brand-new/profile", "/users/12345/profile"):
        assert t.template_path("erp.corp", "GET", p)[0] == "/users/{var}/profile"
    # masked values are still returned under a collapsed position
    assert t.template_path("erp.corp", "GET", "/users/777/profile")[1] == [("{num}", "777")]


def test_no_collapse_when_top_child_dominates():
    t = T.Templater()
    _warm(t, "/users/me", n=400)
    rng = random.Random(9)
    for _ in range(50):
        _warm(t, "/users/" + _rand_str(rng, string.ascii_lowercase, 7), n=3)
    t.maintain()
    node = _root(t).children["users"]
    assert not node.is_var and len(node.children) == 51
    assert t.template_path("erp.corp", "GET", "/users/me")[0] == "/users/me"


def test_depth_cap_tail():
    t = T.Templater()
    for p in ("/a/b/c/d/e/f/g", "/a/b/c/d/e/f/g/h/i"):
        tpl, _ = _warm(t, p)
        assert tpl == "/a/b/c/d/e/f/{var}"
    assert t.template_path("erp.corp", "GET", "/a/b/c/d/e/f")[0] == "/a/b/c/d/e/f"


def test_http_token_format_and_privacy():
    t = T.Templater()
    for _ in range(3):
        tok, masked = t.http_token("get", "ERP.corp:8080", "/orders/view/9876", 200)
    assert tok == "GET erp.corp /orders/view/{num}|2xx"
    assert masked == [("{num}", "9876")]
    for _ in range(3):
        tok, masked = t.http_token("GET", "erp.corp", "/search?q=secret&page=2", None)
    assert tok == "GET erp.corp /search?page&q|0xx"
    assert "secret" not in tok
    for _ in range(3):
        tok, masked = t.http_token("POST", "erp.corp", "/users/bob%40corp.com/reset", 302)
    assert tok == "POST erp.corp /users/{email}/reset|3xx" and "bob" not in tok
    assert masked == [("{email}", "bob@corp.com")]
    tok, _ = t.http_token("BREW", "", "/", 418)
    assert tok == "OTHER - /|4xx"
    tok, _ = t.http_token("PROPFIND", "dav.corp", "/", 207)
    assert tok.startswith("PROPFIND dav.corp /|2xx")


def test_template_path_returns_fresh_masked_list():
    t = T.Templater()
    _, m1 = t.template_path("h", "GET", "/static/app.js")
    m1.append(("{num}", "1"))
    assert t.template_path("h", "GET", "/static/app.js")[1] == []


@pytest.mark.parametrize("w", [0.0, -1.0, NAN, INF, "x", None])
def test_invalid_weight_is_a_lookup(w):
    t = T.Templater()
    _warm(t, "/known/path")
    snap = json.dumps(t.to_dict(), sort_keys=True)
    assert t.template_path("erp.corp", "GET", "/known/path", w)[0] == "/known/path"
    assert t.template_path("erp.corp", "GET", "/unknown/7", w)[0] == "/{var}/{num}"
    assert t.template_path("other.host", "GET", "/x", w)[0] == "/{var}"
    assert json.dumps(t.to_dict(), sort_keys=True) == snap


def test_hard_caps_bound_memory_without_maintenance():
    t = T.Templater()
    for i in range(400):
        t.template_path("h", "GET", f"/p{i:04d}x")
    root = t.trees[("h", "GET")]
    assert len(root.children) <= 256 + 1
    assert root.children["{var}"].count == 400 - 256
    t2 = T.Templater()
    for i in range(700):
        t2.template_path(f"host{i}.corp", "GET", "/a")
    assert len(t2.trees) <= 512
    assert t2.template_path("host699.corp", "GET", "/a")[0] == "/{var}"
    t2.maintain()                                       # > 256 trees: count <= 2 roots dropped
    assert len(t2.trees) == 0


def test_dirty_maintenance_equals_full_pass():
    """Incremental passes (dirty nodes + merged subtrees) give the same tree as
    a full traversal every tick, on a stream that crowds, merges and collapses."""
    rng = random.Random(12)
    words = [_rand_str(rng, string.ascii_lowercase, 5) for _ in range(140)]
    a, b = T.Templater(), T.Templater()
    for tick in range(30):
        for _ in range(150):
            depth = rng.randint(1, 4)
            p = "/" + "/".join(rng.choice(words[: 20 + 4 * tick]) for _ in range(depth))
            w = rng.choice([1.0, 1.0, 3.0])
            assert a.template_path("h", "GET", p, w) == b.template_path("h", "GET", p, w)
        a.maintain()
        b._full_next = True                             # force a full traversal every pass
        b.maintain()
    assert json.dumps(a.to_dict(), sort_keys=True) == json.dumps(b.to_dict(), sort_keys=True)
    root = _root(a, "h")
    assert root.is_var or "{var}" in root.children      # the stream did crowd the root


def test_forced_pass_when_node_budget_is_near(monkeypatch):
    monkeypatch.setattr(T, "_SOFT_NODES", 50)
    monkeypatch.setattr(T, "_MAX_NODES", 60)
    t = T.Templater(_force_at=50)
    for i in range(10):                                 # uncrowded: 10 rare children per node
        for j in range(5):
            t.template_path("h", "GET", f"/s{i}/x{j}")
    assert t._n_nodes >= 50
    t.maintain()
    root = t.trees[("h", "GET")]
    assert len(root.children) == 10                     # s0..s9 have count 5: established
    for c in root.children.values():                    # rare x0..x4 merged although uncrowded
        assert set(c.children) == {"{var}"} and c.children["{var}"].count == 5
    assert t._n_nodes == 21 and t._force_at == 50
    # without the budget pressure the same uncrowded tree is left alone
    t3 = T.Templater()
    for i in range(10):
        for j in range(5):
            t3.template_path("h", "GET", f"/s{i}/x{j}")
    t3.maintain()
    assert all(len(c.children) == 5 for c in t3.trees[("h", "GET")].children.values())


# ------------------------------------------------------------------ vocabulary
def test_intern_ids_counts_and_rare():
    t = T.Templater()
    a = t.intern("GET h /a|2xx")
    b = t.intern("GET h /b|2xx", 2.5)
    assert (a, b) == (1, 2)
    assert t.intern("GET h /a|2xx", 3) == 1
    assert t.vocab["GET h /a|2xx"] == [1, 4.0, 0.0]
    assert t.token_of(1) == "GET h /a|2xx" and t.token_of(2) == "GET h /b|2xx"
    assert t.intern(T.RARE_TOKEN) == 0 and t.intern("{rare:http}") == 0
    assert t.token_of(0) == T.RARE_TOKEN and t.token_of(999) == T.RARE_TOKEN
    assert t.token_of("junk") == T.RARE_TOKEN
    # an unseen token with an invalid weight is not admitted
    for w in (0, -1, NAN, INF):
        assert t.intern("new", w) == 0
    assert "new" not in t.vocab and t.next_id == 3
    # a known token with an invalid weight is a lookup
    assert t.intern("GET h /a|2xx", NAN) == 1 and t.vocab["GET h /a|2xx"][1] == 4.0


def test_space_saving_eviction_semantics(monkeypatch):
    monkeypatch.setattr(T, "VOCAB_CAP", 4)
    t = T.Templater()
    for tok, w in (("a", 5), ("b", 2), ("c", 2), ("d", 9)):
        t.intern(tok, w)
    tid = t.intern("e", 1.0)
    # argmin (count, id) = b (count 2, id 2) is evicted; e inherits 2 + 1, error 2
    assert "b" not in t.vocab and t.vocab["e"] == [5, 3.0, 2.0]
    assert tid == 5 and t.token_of(2) == T.RARE_TOKEN and 2 not in t.id_to_token
    # re-admission gets a NEW id (ids never reused)
    t.intern("b", 1.0)
    assert t.vocab["b"][0] == 6 and t.token_of(2) == T.RARE_TOKEN
    assert len(t.vocab) == 4 and len(t.id_to_token) == 4


def test_space_saving_matches_bruteforce_reference(monkeypatch):
    cap = 25
    monkeypatch.setattr(T, "VOCAB_CAP", cap)
    rng = random.Random(2024)
    heavy = [f"h{i}" for i in range(8)]
    stream = []
    for i in range(4000):
        r = rng.random()
        tok = rng.choice(heavy) if r < 0.5 else (f"m{rng.randrange(60)}" if r < 0.8 else f"u{i}")
        stream.append((tok, rng.choice([1.0, 1.0, 0.5, 4.0])))
    t = T.Templater()
    for k, (tok, w) in enumerate(stream):
        t.intern(tok, w)
        if k % 500 == 499:
            ref, ref_next = _ref_space_saving(stream[: k + 1], cap)
            assert t.vocab == ref and t.next_id == ref_next
    ref, _ = _ref_space_saving(stream, cap)
    assert t.vocab == ref
    true = Counter()
    for tok, w in stream:
        true[tok] += w
    for tok, (tid, c, err) in t.vocab.items():
        assert c - err <= true[tok] + 1e-9 <= c + 2e-9          # Space-Saving guarantee
        assert t.token_of(tid) == tok
    for h in heavy:
        assert h in t.vocab                                     # heavy hitters survive


def test_vocab_cap_default_and_flood():
    t = T.Templater()
    for i in range(T.VOCAB_CAP + 1500):
        t.intern(f"tok{i}")
        if i % 7 == 0:
            t.intern("hot", 10)
    assert len(t.vocab) == T.VOCAB_CAP == len(t.id_to_token)
    assert "hot" in t.vocab
    assert t.next_id == T.VOCAB_CAP + 1500 + 2


# ------------------------------------------------------------------ serialisation
def _build(seed: int = 0) -> T.Templater:
    rng = random.Random(seed)
    t = T.Templater()
    for i in range(3000):
        r = rng.random()
        if r < 0.5:
            p = f"/orders/view/{rng.randrange(10 ** 6)}"
        elif r < 0.7:
            p = f"/users/{_rand_str(rng, string.ascii_lowercase, 3)}/profile?tab=1"
        else:
            p = "/" + _rand_str(rng, string.ascii_lowercase, 8)
        tok, _ = t.http_token(rng.choice(["GET", "POST"]), "erp.corp", p, rng.choice([200, 404]))
        t.intern(tok, rng.choice([1.0, 2.0]))
        if i % 500 == 499:
            t.maintain()
    return t


def test_to_dict_is_json_safe_and_round_trips():
    t = _build()
    d = t.to_dict()
    s = json.dumps(d, allow_nan=False)
    t2 = T.Templater.from_dict(json.loads(s))
    assert json.dumps(t2.to_dict(), sort_keys=True) == json.dumps(d, sort_keys=True)
    assert t2.vocab == t.vocab and t2.id_to_token == t.id_to_token and t2.next_id == t.next_id
    # both continue identically (templates, ids, evictions)
    rng1, rng2 = random.Random(5), random.Random(5)
    for _ in range(500):
        p = f"/users/{_rand_str(rng1, string.ascii_lowercase, 3)}/x"
        _rand_str(rng2, string.ascii_lowercase, 3)
        a = t.http_token("GET", "erp.corp", p, 200)
        b = t2.http_token("GET", "erp.corp", p, 200)
        assert a == b
        assert t.intern(a[0]) == t2.intern(b[0])
    t.maintain()
    t2.maintain()
    assert json.dumps(t.to_dict(), sort_keys=True) == json.dumps(t2.to_dict(), sort_keys=True)


def test_round_trip_preserves_collapse_and_eviction_order(monkeypatch):
    monkeypatch.setattr(T, "VOCAB_CAP", 10)
    t = T.Templater()
    for u in range(50):
        _warm(t, f"/u/name{chr(97 + u % 26)}{chr(97 + u // 26)}x/p", n=3)
    t.maintain()
    assert _root(t).children["u"].is_var
    for i in range(30):
        t.intern(f"t{i % 13}", 1 + (i % 3))
    t2 = T.Templater.from_dict(copy.deepcopy(t.to_dict()))
    assert _root(t2).children["u"].is_var
    assert t2.template_path("erp.corp", "GET", "/u/zzz/p")[0] == "/u/{var}/p"
    for i in range(40):
        assert t.intern(f"n{i}") == t2.intern(f"n{i}")
        assert t.vocab == t2.vocab


def test_from_dict_edge_cases():
    assert T.Templater.from_dict(None).vocab == {}
    assert T.Templater.from_dict({}).trees == {}
    t = T.Templater()
    assert T.Templater.from_dict(t) is t
    bad = {"trees": [["h", "GET", [3, 0, {"a": "junk", "b": [NAN, 0, {}]}]], ["short"]],
           "vocab": [["x", 5, 2.0, 0.0], ["bad"], ["y", "z", 1, 0], ["", 3, 1, 0],
                     ["neg", 7, -1, 0]],
           "next_id": 2}
    t2 = T.Templater.from_dict(bad)
    assert set(t2.vocab) == {"x"} and t2.next_id == 6          # never below max id + 1
    root = t2.trees[("h", "GET")]
    assert root.count == 3 and root.children["b"].count == 0.0
    t2.intern("new")
    assert t2.vocab["new"][0] == 6


# ------------------------------------------------------------------ performance
def test_per_request_cost():
    """R2 budget is ~2.6 us per request (measured); assert a generous bound."""
    rng = random.Random(0)
    paths = ([f"/orders/view/{rng.randrange(10 ** 6)}" for _ in range(6000)]
             + [f"/api/v1/items/{rng.randrange(1000)}/detail?page=1" for _ in range(3000)]
             + ["/static/app/main.bundle.js"] * 3000)
    rng.shuffle(paths)
    best = INF
    for _ in range(3):
        t = T.Templater()
        t0 = time.perf_counter()
        for p in paths:
            tok, _ = t.http_token("GET", "erp.corp", p, 200)
            t.intern(tok)
        best = min(best, (time.perf_counter() - t0) / len(paths))
    assert best < 12e-6, best
    t0 = time.perf_counter()
    t.maintain()
    assert time.perf_counter() - t0 < 4e-3


def test_memos_skip_long_keys():
    """A flood of long unique paths must not bloat the module memos."""
    long_path = "/" + "x" * 5000 + "/y"
    tpl, _ = T.Templater().template_path("h", "GET", long_path)
    assert tpl == "/{var}/{var}"
    assert long_path not in T._PATH and ("/" + "x" * 5000) not in T._PRE
    assert ("x" * 5000) not in T._LIT
    assert T.mask_path(long_path)[0] == ["x" * 61 + "...", "y"]


def test_partial_passes_keep_node_count_exact(monkeypatch):
    """Regression: incremental maintain() passes merged nodes away without
    decrementing _n_nodes, so a steady 404 scanner inflated the count until
    it spuriously hit the node budget: a forced pass then merged rare
    literals everywhere (resetting a sparse endpoint in an uncrowded node)
    and new literals were routed to {var} by the hard cap."""
    monkeypatch.setattr(T, "_SOFT_NODES", 500)
    monkeypatch.setattr(T, "_MAX_NODES", 600)
    rng = random.Random(3)
    t = T.Templater(_force_at=500)
    hits = {0, 5, 9}
    tpl = None
    for tick in range(10):
        for _ in range(200):                            # crowded root: junk merged each pass
            t.template_path("h", "GET", "/" + _rand_str(rng, string.ascii_lowercase, 8))
        _warm(t, "/api/list", host="h")                # /api stays established
        if tick in hits:
            tpl = t.template_path("h", "GET", "/api/rarely")[0]
        t.maintain()
        assert t._n_nodes == sum(_n_nodes(r) for r in t.trees.values())
    assert tpl == "/api/rarely"                         # 3rd hit: established, never reset
    # a collapse inside a partial pass is counted too
    t2 = T.Templater()
    for i in range(45):
        _warm(t2, f"/c/p{i:02d}x", host="h")
    t2.maintain()
    t2.template_path("h", "GET", "/c/new")
    t2.maintain()                                       # partial pass (dirty node only)
    assert _root(t2, "h").children["c"].is_var
    assert t2._n_nodes == _n_nodes(_root(t2, "h"))
