"""Tests for engines/behavior/lib/stack.py: passive client-stack tokens
(R3 client.stack_set / stack_events, R2 act.stream stack_id, B09 consistency).

ja3n and stack_id are checked against independent re-implementations written
from the contract text (GREASE mask test, extension sort, md5 of the re-joined
string; blake2b-8 read big-endian & 0x7FFFFFFF), plus golden values: stack
tokens and ids are persisted in client.stack_set, act.stream and model.client,
so a silent change of either formula would orphan every learned stack.
"""
from __future__ import annotations

import hashlib
import math
import os
import random
import subprocess
import sys
import time

import numpy as np
import pytest

BACKEND = os.path.join(os.path.dirname(__file__), "..", "..", "backend")
sys.path.insert(0, BACKEND)

from app.engines.behavior.lib import stack as S  # noqa: E402

NAN = float("nan")
INF = float("inf")
FAMILIES = {"chrome", "edge", "firefox", "safari", "python-requests", "curl",
            "go-http-client", "java", "okhttp", "wget", "postman", "other", "none"}
OSES = {"win", "mac", "linux", "android", "ios"}

UA_CHROME_WIN = ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                 "(KHTML, like Gecko) Chrome/126.0.0.0 Safari/537.36")
UA_EDGE_WIN = UA_CHROME_WIN + " Edg/126.0.2592.87"
UA_FIREFOX_WIN = "Mozilla/5.0 (Windows NT 10.0; Win64; x64; rv:127.0) Gecko/20100101 Firefox/127.0"
UA_SAFARI_MAC = ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 "
                 "(KHTML, like Gecko) Version/17.4 Safari/605.1.15")
UA_REQUESTS = "python-requests/2.31.0"

# Chrome 126-style ClientHello: ciphers, extensions, curves, point formats (no GREASE).
CH_CIPHERS = [4865, 4866, 4867, 49195, 49199, 49196, 49200, 52393, 52392, 49171, 49172,
              156, 157, 47, 53]
CH_EXTS = [0, 23, 65281, 10, 11, 35, 16, 5, 13, 18, 51, 45, 43, 27, 17513, 21]
CH_CURVES = [29, 23, 24]
CH_PF = [0]
GREASE_ALL = [0x0A0A + 0x1010 * i for i in range(16)]     # 0x0a0a, 0x1a1a, ..., 0xfafa


def j(*lists) -> str:
    return ",".join("-".join(map(str, x)) if isinstance(x, (list, tuple)) else str(x)
                    for x in lists)


# ------------------------------------------------------------- references
def ref_ja3n(version, ciphers, exts, curves, pfs) -> str:
    """Spec text: drop GREASE from ciphers/ext/curves, sort ext, md5-hex."""
    g = lambda v: (v & 0x0F0F) == 0x0A0A  # noqa: E731
    s = ",".join([str(version),
                  "-".join(str(v) for v in ciphers if not g(v)),
                  "-".join(str(v) for v in sorted(v for v in exts if not g(v))),
                  "-".join(str(v) for v in curves if not g(v)),
                  "-".join(str(v) for v in pfs)])
    return hashlib.md5(s.encode()).hexdigest()


def ref_stack_id(token: str) -> int:
    b = token.encode("utf-8", "surrogatepass")
    x = int.from_bytes(hashlib.blake2b(b, digest_size=8).digest(), "big")
    return (x & 0x7FFFFFFF) or 1


def chrome_ja3(rng: random.Random) -> str:
    """A Chrome-like JA3 with fresh GREASE values and a random extension order."""
    exts = CH_EXTS[:]
    rng.shuffle(exts)
    ga, gb, gc = (rng.choice(GREASE_ALL) for _ in range(3))
    return j(771, [ga] + CH_CIPHERS, [gb] + exts + [rng.choice(GREASE_ALL)], [gc] + CH_CURVES, CH_PF)


# =============================================================== ja3n
class TestJa3n:
    def test_extension_order_invariant(self):
        a = j(771, CH_CIPHERS, CH_EXTS, CH_CURVES, CH_PF)
        b = j(771, CH_CIPHERS, list(reversed(CH_EXTS)), CH_CURVES, CH_PF)
        assert a != b
        assert S.ja3n(a) == S.ja3n(b)
        assert len(S.ja3n(a)) == 32 and all(c in "0123456789abcdef" for c in S.ja3n(a))

    def test_chrome_randomisation_and_grease_collapse(self):
        rng = random.Random(7)
        raws = {chrome_ja3(rng) for _ in range(200)}
        assert len(raws) > 150                       # the raw JA3 really churns
        hashes = {S.ja3n(r) for r in raws}
        assert hashes == {ref_ja3n(771, CH_CIPHERS, CH_EXTS, CH_CURVES, CH_PF)}

    def test_matches_reference_on_random_hellos(self):
        rng = random.Random(11)
        for _ in range(300):
            ver = rng.choice([769, 770, 771, 772])
            pool = list(range(0, 70000, 7)) + GREASE_ALL
            pool = [v for v in pool if v <= 0xFFFF]
            ciphers = rng.sample(pool, rng.randint(0, 20))
            exts = rng.sample(pool, rng.randint(0, 20))
            curves = rng.sample(pool, rng.randint(0, 6))
            pfs = [rng.randint(0, 2) for _ in range(rng.randint(0, 3))]
            assert S.ja3n(j(ver, ciphers, exts, curves, pfs)) == ref_ja3n(ver, ciphers, exts, curves, pfs)

    def test_golden(self):
        # Pinned: persisted stack tokens depend on this exact normalisation.
        ja3 = j(771, [0x3A3A] + CH_CIPHERS, [0x8A8A] + CH_EXTS + [0xDADA], [0x2A2A] + CH_CURVES, CH_PF)
        norm = ("771,4865-4866-4867-49195-49199-49196-49200-52393-52392-49171-49172-156-157-47-53,"
                "0-5-10-11-13-16-18-21-23-27-35-43-45-51-17513-65281,29-23-24,0")
        assert S.ja3n(ja3) == hashlib.md5(norm.encode()).hexdigest() == "aa56c057ad164ec4fdcb7a5a283be9fc"

    def test_grease_mask_semantics(self):
        # every value with (v & 0x0f0f) == 0x0a0a is GREASE for ciphers/ext/curves ...
        base = S.ja3n(j(771, [47], [0], [29], [0]))
        for g in GREASE_ALL + [0x0A1A, 0x1A0A]:
            assert S.ja3n(j(771, [g, 47], [g, 0, g], [29, g], [0])) == base
        # ... but not for point formats, and not for near-misses
        assert S.ja3n(j(771, [47], [0], [29], [0, 0x0A0A])) != base
        assert S.ja3n(j(771, [47, 0x0A0B], [0], [29], [0])) != base
        assert S.ja3n(j(771, [47], [0, 0x0B0A], [29], [0])) != base

    def test_cipher_and_curve_order_are_kept(self):
        a = S.ja3n(j(771, [47, 53], [0, 10], [29, 23], [0]))
        assert S.ja3n(j(771, [53, 47], [0, 10], [29, 23], [0])) != a
        assert S.ja3n(j(771, [47, 53], [0, 10], [23, 29], [0])) != a
        assert S.ja3n(j(771, [47, 53], [10, 0], [29, 23], [0])) == a

    def test_distinct_stacks_stay_distinct(self):
        chrome = j(771, CH_CIPHERS, CH_EXTS, CH_CURVES, CH_PF)
        requests = j(771, [4866, 4867, 4865, 49196, 49200], [0, 11, 10, 35, 22, 23, 13, 43, 45, 51],
                     [29, 23, 30, 25, 24], [0, 1, 2])
        tls12 = j(769, CH_CIPHERS, CH_EXTS, CH_CURVES, CH_PF)
        assert len({S.ja3n(chrome), S.ja3n(requests), S.ja3n(tls12)}) == 3

    def test_missing_trailing_fields_read_as_empty(self):
        assert S.ja3n("769,47-53,0-11") == S.ja3n("769,47-53,0-11,,")
        assert S.ja3n("769,47-53,0-11") == ref_ja3n(769, [47, 53], [0, 11], [], [])
        assert S.ja3n("769,47-53,0-11,23") == S.ja3n("769,47-53,0-11,23,")
        assert S.ja3n("771,,,,") == ref_ja3n(771, [], [], [], [])

    def test_canonical_items_and_whitespace(self):
        a = S.ja3n("771,47-53,0-10,29,0")
        assert S.ja3n("  771, 47 - 53 ,0-10, 29 ,0\n") == a
        assert S.ja3n("771,0047-53,000-10,29,0") == a

    def test_hash_passthrough(self):
        h = "E7D705A3286E19EA42F587B344EE6865"
        assert S.ja3n(h) == h.lower()
        assert S.ja3n(" " + h.lower() + " ") == h.lower()
        assert S.ja3n(h[:31]) == "-"                  # not 32 hex, not a JA3 string either

    @pytest.mark.parametrize("bad", [
        None, "", "   ", "-", "abc", "771", "771,47", "771,47,0,29,0,1",
        "771,47-x,0,29,0", "771,47,0-65536,29,0", "771-772,47,0,29,0", ",47,0,29,0",
        "771,47--53,0,29,0", "771,47-,0,29,0", "771,4²7,0,29,0", "771,-47,0,29,0",
        "771,1e3,0,29,0", "771,47,0,29,0.5", "771," + "1" * 20 + ",0,29,0",
        NAN, 771, 3.5, b"\xff\xfe", ["771"], "771," + "-".join(["47"] * 3000) + ",0,29,0",
    ])
    def test_malformed(self, bad):
        assert S.ja3n(bad) == "-"

    def test_bytes_input_decoded(self):
        assert S.ja3n(b"771,47-53,0-10,29,0") == S.ja3n("771,47-53,0-10,29,0")


# =============================================================== ua_parse
UA_CASES = [
    # browsers
    (UA_CHROME_WIN, ("chrome", "126", "win")),
    (UA_EDGE_WIN, ("edge", "126", "win")),
    (UA_FIREFOX_WIN, ("firefox", "127", "win")),
    (UA_SAFARI_MAC, ("safari", "17", "mac")),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/537.36 (KHTML, like Gecko) "
     "Chrome/127.0.0.0 Safari/537.36", ("chrome", "127", "mac")),
    ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 "
     "Safari/537.36", ("chrome", "126", "linux")),
    ("Mozilla/5.0 (X11; Ubuntu; Linux x86_64; rv:127.0) Gecko/20100101 Firefox/127.0",
     ("firefox", "127", "linux")),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 14.5; rv:127.0) Gecko/20100101 Firefox/127.0",
     ("firefox", "127", "mac")),
    ("Mozilla/5.0 (Linux; Android 14; Pixel 8) AppleWebKit/537.36 (KHTML, like Gecko) "
     "Chrome/126.0.6478.122 Mobile Safari/537.36", ("chrome", "126", "android")),
    ("Mozilla/5.0 (Linux; Android 10; K) AppleWebKit/537.36 (KHTML, like Gecko) Chrome/126.0.0.0 "
     "Mobile Safari/537.36 EdgA/126.0.2592.80", ("edge", "126", "android")),
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
     "Version/17.4 Mobile/15E148 Safari/604.1", ("safari", "17", "ios")),
    ("Mozilla/5.0 (iPad; CPU OS 16_6 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
     "Version/16.6 Mobile/15E148 Safari/604.1", ("safari", "16", "ios")),
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
     "CriOS/126.0.6478.54 Mobile/15E148 Safari/604.1", ("chrome", "126", "ios")),
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
     "FxiOS/127.0 Mobile/15E148 Safari/605.1.15", ("firefox", "127", "ios")),
    ("Mozilla/5.0 (iPhone; CPU iPhone OS 17_4 like Mac OS X) AppleWebKit/605.1.15 (KHTML, like Gecko) "
     "Version/17.0 EdgiOS/126.2592.86 Mobile/15E148 Safari/605.1.15", ("edge", "126", "ios")),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
     "Chrome/70.0.3538.102 Safari/537.36 Edge/18.19045", ("edge", "18", "win")),
    ("Mozilla/5.0 (X11; CrOS x86_64 14541.0.0) AppleWebKit/537.36 (KHTML, like Gecko) "
     "Chrome/126.0.0.0 Safari/537.36", ("chrome", "126", "linux")),
    ("Mozilla/5.0 (Macintosh; Intel Mac OS X 10_15_7) AppleWebKit/605.1.15 (KHTML, like Gecko) "
     "Safari/605.1.15", ("safari", "0", "mac")),          # no Version/ -> unknown major
    # libraries
    (UA_REQUESTS, ("python-requests", "2", None)),
    ("python-requests/2.28.1", ("python-requests", "2", None)),
    ("okhttp/4.12.0", ("okhttp", "4", None)),
    ("okhttp/3.12.1", ("okhttp", "3", None)),
    ("Go-http-client/1.1", ("go-http-client", "1", None)),
    ("Go-http-client/2.0", ("go-http-client", "2", None)),
    ("curl/8.4.0", ("curl", "8", None)),
    ("curl/7.29.0 (x86_64-redhat-linux-gnu) libcurl/7.29.0 NSS/3.53.1", ("curl", "7", "linux")),
    ("PycURL/7.45.2 libcurl/8.4.0 OpenSSL/3.0.13", ("curl", "8", None)),
    ("Wget/1.21.4", ("wget", "1", None)),
    ("Wget/1.20.3 (linux-gnu)", ("wget", "1", "linux")),
    ("PostmanRuntime/7.36.0", ("postman", "7", None)),
    ("Java/1.8.0_292", ("java", "1", None)),
    ("Java/17.0.2", ("java", "17", None)),
    ("Java-http-client/21.0.1", ("java", "21", None)),
    ("Apache-HttpClient/4.5.13 (Java/11.0.20)", ("java", "11", None)),
    ("okhttp/4.9.0 Dalvik/2.1.0 (Linux; U; Android 12)", ("okhttp", "4", "android")),
    # unlisted browsers / shells / unknowns -> other
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
     "Chrome/125.0.0.0 Safari/537.36 OPR/111.0.0.0", ("other", "111", "win")),
    ("Mozilla/5.0 (Linux; Android 14; SM-S918B) AppleWebKit/537.36 (KHTML, like Gecko) "
     "SamsungBrowser/24.0 Chrome/117.0.0.0 Mobile Safari/537.36", ("other", "24", "android")),
    ("Mozilla/5.0 (X11; Linux x86_64) AppleWebKit/537.36 (KHTML, like Gecko) HeadlessChrome/126.0.0.0 "
     "Safari/537.36", ("other", "126", "linux")),
    ("Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) Slack/4.38.125 "
     "Chrome/124.0.6367.243 Electron/30.1.0 Safari/537.36 Sonic Slack_SSB/4.38.125", ("other", "30", "win")),
    ("Mozilla/5.0 (compatible; Googlebot/2.1; +http://www.google.com/bot.html)", ("other", "2", None)),
    ("Dalvik/2.1.0 (Linux; U; Android 11; SM-G991B Build/RP1A.200720.012)", ("other", "2", "android")),
    ("Mozilla/5.0 (Windows NT 6.1; Trident/7.0; rv:11.0) like Gecko", ("other", "7", "win")),
    ("MyApp/3.2 CFNetwork/1490.0.4 Darwin/23.2.0", ("other", "3", None)),
    ("Microsoft-CryptoAPI/10.0", ("other", "10", None)),
    ("Windows-Update-Agent/10.0.10011.16384 Client-Protocol/2.0", ("other", "10", "win")),
    ("Mozilla/5.0 (X11; FreeBSD amd64; rv:127.0) Gecko/20100101 Firefox/127.0", ("firefox", "127", None)),
    ("foo", ("other", "0", None)),
    ("Mozilla/5.0", ("other", "0", None)),
    # empty-ish
    ("", ("none", "0", None)),
    ("   ", ("none", "0", None)),
    ("-", ("none", "0", None)),
    (None, ("none", "0", None)),
    (NAN, ("none", "0", None)),
    (42, ("none", "0", None)),
]


class TestUaParse:
    @pytest.mark.parametrize("ua,expected", UA_CASES)
    def test_table(self, ua, expected):
        assert S.ua_parse(ua) == expected

    def test_major_edge_cases(self):
        assert S.ua_parse("python-requests/")[:2] == ("python-requests", "0")
        assert S.ua_parse("python-requests/x.y")[:2] == ("python-requests", "0")
        assert S.ua_parse("curl/008.1")[:2] == ("curl", "8")
        assert S.ua_parse("curl/0.9")[:2] == ("curl", "0")
        assert S.ua_parse("curl/" + "9" * 40)[:2] == ("curl", "0")     # implausible -> unknown
        assert S.ua_parse("curl/" + "9" * 5000)[:2] == ("curl", "0")   # no int-digit-limit crash
        assert S.ua_parse("PYTHON-REQUESTS/2.31")[:2] == ("python-requests", "2")

    def test_major_is_ascii_for_unicode_digits(self):
        # regression: regex \d matches Arabic-Indic / fullwidth digits; the token
        # field must stay canonical ASCII so 'chrome/١٢٦' == 'chrome/126'.
        assert S.ua_parse("Mozilla/5.0 (Windows NT 10.0) Chrome/١٢٦.0") == \
            ("chrome", "126", "win")
        assert S.ua_parse("Foo/٠٠١٢")[:2] == ("other", "12")
        assert S.ua_parse("curl/８.4")[:2] == ("curl", "8")                 # fullwidth 8
        tok = S.stack_token(None, "Chrome/١٢٦", 120, 64240)
        assert tok.isascii() and "|chrome/126|" in tok

    @pytest.mark.parametrize("ua,expected", [
        ("Dalvik/2.1.0 (Linux; U; Android 11)", ("other", "2")),
        ("360SE/12.0", ("other", "12")),                 # name starts at the first letter
        ("Mozilla/5abc/7", ("other", "7")),              # next run starts right after '/'
        ("Mozilla/5.0 (X) 1/2 Foo/3", ("other", "3")),   # letter-less run is skipped
        (" ./1_a/", ("other", "0")),
        ("Mozilla/5.0 (compatible)", ("other", "0")),
    ])
    def test_fallback_product(self, ua, expected):
        assert S.ua_parse(ua)[:2] == expected

    def test_fallback_is_linear_time(self):
        # regression: the fallback regex tried a match at every letter of a
        # slash-less run -> O(n^2), ~1.6 ms for one 512-char junk UA (R3: 1 ms/tick).
        # Compare 512 vs 128 chars on the uncached parser: quadratic gives ~16x.
        parse = S._ua_parse_cached.__wrapped__

        def best(n):
            ua = "Mozilla/5.0 (" + "a" * n
            ts = []
            for _ in range(7):
                t = time.perf_counter()
                parse(ua)
                ts.append(time.perf_counter() - t)
            return min(ts)
        small, big = best(128), best(512)
        assert big < 8 * small, (small, big)
        assert big < 1e-3, big

    def test_library_boundaries(self):
        # 'curl' inside another word is not curl; 'java' inside a word is not Java.
        assert S.ua_parse("Securl/1.0")[0] == "other"
        assert S.ua_parse("javascript-client/3.0")[0] == "other"
        assert S.ua_parse("Myjava/3.0")[0] == "other"

    def test_os_precedence(self):
        # iOS UAs say 'like Mac OS X'; Android UAs say 'Linux'; Darwin is undeclared.
        assert S.ua_parse("X (iPhone; CPU iPhone OS 17_4 like Mac OS X)")[2] == "ios"
        assert S.ua_parse("X (Linux; Android 14)")[2] == "android"
        assert S.ua_parse("Mozilla/5.0 (Mobile; Windows Phone 8.1; Android 4.0; ARM) like iPhone OS 7_0_3 "
                          "Mac OS X")[2] == "win"
        assert S.ua_parse("App/1 Darwin/23.2.0")[2] is None

    def test_long_ua_is_bounded_and_fast(self):
        ua = UA_CHROME_WIN + " " + "a" * 100_000
        t = time.perf_counter()
        assert S.ua_parse(ua) == ("chrome", "126", "win")
        assert S.ua_parse("x" * 100_000 + " python-requests/2.31")[0] == "other"   # past the cut
        assert time.perf_counter() - t < 0.2

    def test_fuzz_closed_alphabet(self):
        rng = random.Random(3)
        alphabet = "abcXYZ019/.;()_- |Chrome Safari Edg Firefox curl okhttp Java Windows Linux iPhone"
        for _ in range(2000):
            ua = "".join(rng.choice(alphabet) for _ in range(rng.randint(0, 80)))
            fam, major, os_name = S.ua_parse(ua)
            assert fam in FAMILIES
            assert major.isdigit() and (major == "0" or not major.startswith("0"))
            assert os_name is None or os_name in OSES


# =============================================================== ttl / window
class TestTtlClass:
    @pytest.mark.parametrize("ttl,cls", [
        (57, 64), (1, 32), (0.5, 32), (31, 32), (32, 32), (33, 64), (63.9, 64), (64, 64),
        (64.1, 128), (65, 128), (113, 128), (128, 128), (129, 255), (240, 255), (255, 255),
        (256, 255), (1e9, 255), (INF, 255), (0, 0), (-1, 0), (-INF, 0), (None, 0), (NAN, 0),
        ("57", 64), (" 120 ", 128), ("abc", 0), ([57], 0), (np.float32(57), 64), (np.int64(128), 128),
        (np.nan, 0),
    ])
    def test_table(self, ttl, cls):
        out = S.ttl_class(ttl)
        assert out == cls and type(out) is int

    def test_monotone_and_closed(self):
        prev = 0
        for t in np.linspace(-5, 400, 4001):
            c = S.ttl_class(float(t))
            assert c in (0,) + S.TTL_CLASSES
            assert c >= prev
            prev = c
            if 0 < t <= 255:
                assert c >= t


class TestWinClass:
    @pytest.mark.parametrize("win,cls", [
        (None, "w0"), (0, "w0"), (-5, "w0"), (NAN, "w0"), ("abc", "w0"), (-INF, "w0"),
        (0.5, "w10"), (1, "w10"), (1023, "w10"), (1024, "w10"), (1025, "w11"),
        (8192, "w13"), (14600, "w14"), (29200, "w15"), (32768, "w15"), (32769, "w16"),
        (64240, "w16"), (65535, "w16"), (65536, "w16"), (65537, "w17"),
        (2 ** 20, "w20"), (2 ** 20 + 1, "w20"), (2 ** 40, "w20"), (1e300, "w20"), (INF, "w20"),
        ("65535", "w16"), (np.int32(29200), "w15"), (np.float64(1025.0), "w11"),
    ])
    def test_table(self, win, cls):
        assert S.win_class(win) == cls

    def test_exact_ceil_log2(self):
        lo, hi = S.WIN_CLASS_RANGE
        for n in list(range(1, 5000)) + [2 ** k + d for k in range(10, 22) for d in (-1, 0, 1)]:
            k = (n - 1).bit_length()                    # exact ceil(log2(n)) for ints >= 1
            assert S.win_class(n) == f"w{min(max(k, lo), hi)}", n
        # float just above a power of two rounds up, exact power does not
        assert S.win_class(math.nextafter(4096.0, INF)) == "w13"
        assert S.win_class(4096.0) == "w12"


# =============================================================== stack_token
CHROME_JA3 = j(771, CH_CIPHERS, CH_EXTS, CH_CURVES, CH_PF)
REQ_JA3 = j(771, [4866, 4867, 4865, 49196, 49200], [0, 11, 10, 35, 22, 23, 13, 43, 45, 51],
            [29, 23, 30, 25, 24], [0, 1, 2])


class TestStackToken:
    def test_r3_spec_python_requests_ttl57(self):
        tok = S.stack_token(REQ_JA3, UA_REQUESTS, 57, 29200)
        assert tok.endswith("|python-requests/2|linux|64|w15")
        assert tok == f"{S.ja3n(REQ_JA3)}|python-requests/2|linux|64|w15"
        # whatever the window, the '...|python-requests/2|linux|64|...' part holds
        for win in (None, 1, 5840, 64240, 1 << 30):
            assert "|python-requests/2|linux|64|w" in S.stack_token(REQ_JA3, UA_REQUESTS, 57, win)

    def test_chrome_windows(self):
        tok = S.stack_token(CHROME_JA3, UA_CHROME_WIN, 117, 64240)
        assert tok == f"{S.ja3n(CHROME_JA3)}|chrome/126|win|128|w16"

    def test_extension_randomisation_keeps_token(self):
        rng = random.Random(5)
        toks = {S.stack_token(chrome_ja3(rng), UA_CHROME_WIN, 120, 64240) for _ in range(50)}
        assert len(toks) == 1

    def test_forged_ua_separates_stacks(self):
        # B09 (c): Chrome UA copied onto a Linux client -> different token, OS contradiction
        real = S.stack_token(CHROME_JA3, UA_CHROME_WIN, 120, 64240)
        forged = S.stack_token(REQ_JA3, UA_CHROME_WIN, 60, 29200)
        assert real != forged
        p = S.parse_stack_token(forged)
        assert p["os"] == "win" and p["ttl_class"] == "64"
        assert S.os_ttl_consistent(p["os"], int(p["ttl_class"])) is False

    def test_os_fallbacks(self):
        assert S.stack_token(None, "okhttp/4.12.0", 110, 65535).split("|")[2] == "win"   # TTL 128
        assert S.stack_token(None, "okhttp/4.12.0", 20, 65535).split("|")[2] == "win"    # TTL 32
        assert S.stack_token(None, "okhttp/4.12.0", 250, 65535).split("|")[2] == "net"   # TTL 255
        assert S.stack_token(None, "okhttp/4.12.0", None, 65535).split("|")[2] == "?"
        assert S.stack_token(None, None, None, None) == "-|none/0|?|0|w0"
        # a declared OS always wins over the TTL-implied one
        assert S.stack_token(None, UA_SAFARI_MAC, 120, None).split("|")[2] == "mac"

    def test_ja4_preferred(self):
        ja4 = "t13d1516h2_8daaf6152771_02713d6af862"
        tok = S.stack_token(CHROME_JA3, UA_CHROME_WIN, 120, 64240, ja4=ja4)
        assert tok == f"ja4:{ja4}|chrome/126|win|128|w16"
        assert S.stack_token(None, UA_CHROME_WIN, 120, 64240, ja4=f"  {ja4}\n").startswith(f"ja4:{ja4}|")

    @pytest.mark.parametrize("bad", [None, "", "  ", "-", "a|b", "t13d 1516", NAN, 17, "x" * 600])
    def test_bad_ja4_falls_back_to_ja3n(self, bad):
        tok = S.stack_token(CHROME_JA3, UA_CHROME_WIN, 120, 64240, ja4=bad)
        assert tok.split("|")[0] == S.ja3n(CHROME_JA3)

    def test_always_five_fields_fuzz(self):
        rng = random.Random(9)
        junk = "ab|/ \t,-0123456789é\ud800"
        for _ in range(1000):
            s = lambda: "".join(rng.choice(junk) for _ in range(rng.randint(0, 20)))  # noqa: E731
            tok = S.stack_token(s(), s(), rng.choice([None, NAN, rng.uniform(-10, 400)]),
                                rng.choice([None, s(), rng.uniform(-10, 1e7)]), ja4=s())
            parts = tok.split("|")
            assert len(parts) == 5
            p = S.parse_stack_token(tok)
            assert "|".join([p["ja3n"], f'{p["ua_family"]}/{p["ua_major"]}', p["os"],
                             p["ttl_class"], p["win_class"]]) == tok
            assert S.stack_id(tok) == ref_stack_id(tok)

    def test_perf(self):
        rows = [(chrome_ja3(random.Random(i % 40)), UA_CHROME_WIN if i % 3 else UA_REQUESTS,
                 57 + i % 70, 29200 + i % 5000) for i in range(3000)]
        for r in rows[:100]:
            S.stack_token(*r)
        t = time.perf_counter()
        for r in rows:
            S.stack_id(S.stack_token(*r))
        per = (time.perf_counter() - t) / len(rows)
        assert per < 50e-6, per                         # ~5 us warm; R3 budget is 1 ms / tick


# =============================================================== stack_id
class TestStackId:
    def test_formula_and_range(self):
        rng = random.Random(1)
        for i in range(2000):
            tok = f"{rng.getrandbits(128):032x}|chrome/{i}|win|128|w16"
            sid = S.stack_id(tok)
            assert sid == ref_stack_id(tok)
            assert 0 < sid <= 0x7FFFFFFF and type(sid) is int

    def test_golden(self):
        tok = "-|none/0|?|0|w0"
        x = int.from_bytes(hashlib.blake2b(tok.encode(), digest_size=8).digest(), "big")
        assert S.stack_id(tok) == (x & 0x7FFFFFFF) == 2081287222
        assert S.stack_id("") == ref_stack_id("")

    def test_stable_across_processes(self):
        toks = [S.stack_token(CHROME_JA3, UA_CHROME_WIN, 120, 64240),
                S.stack_token(REQ_JA3, UA_REQUESTS, 57, 29200)]
        code = ("import sys; sys.path.insert(0, %r); from app.engines.behavior.lib import stack as S; "
                "print(S.stack_id(S.stack_token(%r, %r, 120, 64240)), "
                "S.stack_id(S.stack_token(%r, %r, 57, 29200)))"
                % (BACKEND, CHROME_JA3, UA_CHROME_WIN, REQ_JA3, UA_REQUESTS))
        env = dict(os.environ, PYTHONHASHSEED="12345")
        out = subprocess.run([sys.executable, "-c", code], capture_output=True, text=True,
                             env=env, check=True).stdout.split()
        assert [int(x) for x in out] == [S.stack_id(t) for t in toks]

    def test_zero_is_never_returned(self, monkeypatch):
        class Zero:
            def __init__(self, *a, **k):
                pass

            def digest(self):
                return b"\x80\x00\x00\x00\x00\x00\x00\x00"     # low 31 bits all zero

        S._stack_id_cached.cache_clear()
        monkeypatch.setattr(S.hashlib, "blake2b", Zero)
        try:
            assert S.stack_id("anything-zero") == 1
        finally:
            monkeypatch.undo()
            S._stack_id_cached.cache_clear()
        assert S.stack_id("anything-zero") == ref_stack_id("anything-zero")

    def test_non_str_and_surrogates(self):
        assert S.stack_id(123) == ref_stack_id("123")
        assert S.stack_id("a\ud800b") == ref_stack_id("a\ud800b")   # lone surrogate still hashes


# =============================================================== parse_stack_token
class TestParseStackToken:
    def test_roundtrip(self):
        tok = S.stack_token(REQ_JA3, UA_REQUESTS, 57, 29200)
        p = S.parse_stack_token(tok)
        assert p == {"ja3n": S.ja3n(REQ_JA3), "ua_family": "python-requests", "ua_major": "2",
                     "os": "linux", "ttl_class": "64", "win_class": "w15"}
        assert set(p) == {"ja3n", "ua_family", "ua_major", "os", "ttl_class", "win_class"}

    def test_ja4_and_go(self):
        p = S.parse_stack_token(S.stack_token(None, "Go-http-client/2.0", 255, 1 << 14, ja4="t13d_ab_cd"))
        assert p == {"ja3n": "ja4:t13d_ab_cd", "ua_family": "go-http-client", "ua_major": "2",
                     "os": "net", "ttl_class": "255", "win_class": "w14"}

    @pytest.mark.parametrize("tok,exp", [
        (None, {}), ("", {}), (17, {}),
        ("abc", {"ja3n": "abc"}),
        ("abc|chrome/126", {"ja3n": "abc", "ua_family": "chrome", "ua_major": "126"}),
        ("abc|chrome|win", {"ja3n": "abc", "ua_family": "chrome", "os": "win"}),
        ("a|b|c|x/1|win|64|w12", {"ja3n": "a|b|c", "ua_family": "x", "ua_major": "1", "os": "win",
                                  "ttl_class": "64", "win_class": "w12"}),
        ("||||", {}),
    ])
    def test_best_effort(self, tok, exp):
        defaults = {"ja3n": "-", "ua_family": "none", "ua_major": "0", "os": "?",
                    "ttl_class": "0", "win_class": "w0"}
        assert S.parse_stack_token(tok) == {**defaults, **exp}


# =============================================================== os_ttl_consistent
class TestOsTtlConsistent:
    @pytest.mark.parametrize("os_name,ttl_cls,exp", [
        ("win", 64, False), ("win", 128, True), ("win", 32, True),
        ("linux", 128, False), ("android", 128, False), ("mac", 128, False), ("ios", 128, False),
        ("linux", 64, True), ("android", 64, True), ("mac", 64, True), ("ios", 64, True),
        ("linux", 32, True),
        ("win", 255, None), ("linux", 255, None),          # re-originated by a network device
        (None, 64, None), ("?", 64, None), ("", 128, None), ("net", 64, None), ("beos", 64, None),
        ("win", 0, None), ("linux", None, None), ("win", NAN, None),
        ("WIN", 64, False), (" Linux ", 128, False),
        ("win", "64", False), ("linux", "128", False),     # string field of a parsed token
        ("win", 57, False), ("mac", 113, False),           # raw TTL goes through ttl_class
    ])
    def test_table(self, os_name, ttl_cls, exp):
        assert S.os_ttl_consistent(os_name, ttl_cls) is exp

    def test_generator_device_tuples(self):
        # generator.md: org standard Chrome/Windows/TTL 128 and Safari-mac TTL 64 are consistent;
        # the impersonation pack (Chrome/Windows UA, Linux JA3, TTL 64) is not.
        for ua, ttl, exp in [(UA_CHROME_WIN, 120, True), (UA_SAFARI_MAC, 60, True),
                             (UA_FIREFOX_WIN, 124, True), (UA_EDGE_WIN, 126, True),
                             (UA_CHROME_WIN, 61, False), (UA_REQUESTS, 57, None)]:
            _, _, dos = S.ua_parse(ua)
            assert S.os_ttl_consistent(dos, S.ttl_class(ttl)) is exp, ua


def test_constants_frozen():
    assert S.TTL_CLASSES == (32, 64, 128, 255)
    assert S.OS_FROM_TTL == {32: "win", 64: "linux", 128: "win", 255: "net"}
    assert S.WIN_CLASS_RANGE == (10, 20)
    assert (S.GREASE_MASK, S.GREASE_VALUE, S.UNKNOWN) == (0x0F0F, 0x0A0A, "?")
