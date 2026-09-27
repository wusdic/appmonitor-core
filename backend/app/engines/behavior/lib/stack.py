"""Passive client-stack fingerprint tokens (R3 writes, R2 stamps stack_id, B09/B16/B17 read).

STATUS: implemented. Token format, constants and signatures are frozen
(docs/lib3/helpers_api.md).

Why a composite token: no single passive attribute identifies a client.
JA3 changes with Chrome's extension-order randomisation (hence ja3n: the
extension list sorted and GREASE removed), UA strings are trivially copied,
TTL / TCP window betray the real OS. Concatenating the normalised parts
gives a stack that survives benign churn but separates 'python-requests on
Linux' from 'Chrome on Windows' even when the UA is forged.

Token format (exact): 'ja3n|ua_family/major|os|ttl_class|win_class'
    ja3n       md5 hex (32) of the normalised JA3 string, or 'ja4:<ja4>' when a
               JA4 is available, or '-' when neither is present
    ua_family  lower-case family ('chrome', 'edge', 'firefox', 'safari',
               'python-requests', 'curl', 'go-http-client', 'java', 'okhttp',
               'wget', 'postman', 'other', 'none'); major = leading integer of
               the version ('0' if unknown)
    os         UA-declared OS ('win', 'mac', 'linux', 'android', 'ios') or, when
               the UA declares none, the TTL-implied OS (OS_FROM_TTL); '?' if both unknown
    ttl_class  observed TTL rounded UP to one of 32, 64, 128, 255 ('0' if unknown)
    win_class  'w{ceil(log2(win))}' clipped to w10..w20 ('w0' if unknown)
R3 test: a python-requests/2.31 UA with TTL 57 -> '...|python-requests/2|linux|64|...'.

Every field is drawn from a closed alphabet without '|' (md5 hex, the family
list, digits, 'wNN', and a JA4 that is only accepted when it holds no '|' or
whitespace), so parse_stack_token is an exact inverse of stack_token.

Implementation notes (the choices the frozen contract leaves open):
  * ja3n is lenient about shape but strict about content. 3..5 comma fields
    are accepted and missing trailing fields read as empty ('769,47-53,0-11'
    == '769,47-53,0-11,,': a ClientHello without supported_groups has an empty
    curves field anyway); every item must be an ASCII decimal in 0..65535, the
    version a single one. Items are re-emitted canonically (str(int)), so
    '0047' == '47'. Anything else is malformed -> '-'. A bare 32-hex value is
    taken to be a JA3 *hash* (what Zeek-style sensors log) and passed through
    lower-cased: it cannot be normalised, but collapsing it to '-' would
    erase the only TLS signal and blind B09's p(ja3n | UA family) check. (The
    md5 of an already-normalised string equals its ja3n, so the two agree.)
  * GREASE is the frozen mask test (v & 0x0f0f) == 0x0a0a (a superset of the
    RFC 8701 values 0x?a?a; the extra matches are unassigned code points).
    It applies to ciphers, extensions and curves; point formats are 8-bit and
    never GREASEd. Only the extension list is sorted: cipher and curve order
    are stable per TLS library and discriminate between them.
  * ua_parse tries ordered rules and the first match wins:
      1) libraries anywhere in the UA (case-insensitive): python-requests,
         go-http-client, okhttp, (lib)curl, wget, PostmanRuntime, then Java/
         and Java-http-client/ last, because Java/ also appears as a comment
         in other Java clients ('Apache-HttpClient/4.5 (Java/11)' -> java/11);
      2) browsers that are not one of the listed families but carry a
         Chrome/ or Safari/ token (Opera, Samsung, Yandex, UC, Vivaldi,
         Electron apps, HeadlessChrome, crawlers) -> 'other' with their own
         major, so an automation shell or an app does not hide behind
         'chrome/126';
      3) Edg/ EdgA/ EdgiOS/ Edge/ -> edge, Chrome/ CriOS/ Chromium/ -> chrome,
         Firefox/ FxiOS/ -> firefox, Safari/ -> safari with the major taken
         from Version/ (Safari/605.1.15 is the WebKit build, not the release;
         no Version/ -> '0');
      4) otherwise 'other' with the major of the first product token that is
         not 'Mozilla' ('Dalvik/2.1.0' -> other/2).
    Declared OS, first match: Windows Phone -> win, iPhone/iPad/iPod/iOS ->
    ios (before mac: iOS UAs say 'like Mac OS X'), Android -> android (before
    linux), Windows/Win32/Win64 -> win, Macintosh/Mac OS -> mac, Linux/Ubuntu/
    Fedora/Debian/CrOS -> linux. 'Darwin' is deliberately undeclared (CFNetwork
    apps send it on both macOS and iOS), as are the BSDs. Majors longer than 9
    digits are treated as unknown ('0'). A missing, blank or '-' UA is 'none'.
    UAs are truncated to 512 characters before matching.
  * ttl_class / win_class accept anything float() accepts (numpy scalars,
    numeric strings); NaN, <= 0 and junk are unknown. win_class computes
    ceil(log2(w)) exactly from frexp (2^k -> k, 2^k + 1 -> k + 1).
  * os_ttl_consistent: TTL class 255 returns None rather than True/False. No
    listed OS originates TCP at 255, so a 255 path means a network device
    re-originated the packets (proxy, firewall) and the TTL says nothing
    about the client OS; flagging it would mark everyone behind such a proxy
    inconsistent. Class 32 is consistent with every OS (legacy Windows, or a
    64-host far away). ttl_cls is passed through ttl_class first, so a raw TTL
    (57) or the string field of a parsed token ('64') also works.
  * stack_id reads the 8-byte blake2b digest big-endian (the int.from_bytes
    default, as in sketch / combine) and maps the one-in-2^31 zero to 1, so
    0 stays free to mean "no stack" in act.stream.
  * ja3n, ua_parse and stack_id are memoised (bounded LRU): the inputs repeat
    heavily across observations and ticks (a few dozen distinct stacks per
    system), md5 / blake2b / regex cost ~1-5 us cold and ~0.1 us warm, and
    R3 must stay under 1 ms per tick.
"""
from __future__ import annotations

import hashlib
import math
import re
from functools import lru_cache
from typing import Dict, Optional, Pattern, Tuple

TTL_CLASSES: Tuple[int, ...] = (32, 64, 128, 255)
OS_FROM_TTL: Dict[int, str] = {32: "win", 64: "linux", 128: "win", 255: "net"}
WIN_CLASS_RANGE: Tuple[int, int] = (10, 20)
GREASE_MASK = 0x0F0F
GREASE_VALUE = 0x0A0A          # (v & 0x0f0f) == 0x0a0a marks a GREASE value
UNKNOWN = "?"

_NO_FP = "-"                   # ja3n field when neither JA3 nor JA4 is usable
_JA4_PREFIX = "ja4:"
_TOKEN_DEFAULTS: Dict[str, str] = {"ja3n": _NO_FP, "ua_family": "none", "ua_major": "0",
                                   "os": UNKNOWN, "ttl_class": "0", "win_class": "w0"}
_DECLARED_OS = frozenset({"win", "mac", "linux", "android", "ios"})

_JA3_MAX_LEN = 4096            # a real JA3 string is < 1 KB; longer is junk (and cache-hostile)
_JA3_U16_MAX = 0xFFFF
_JA4_MAX_LEN = 512             # room for JA4_r; the hashed form is 36 chars
_UA_MAX_LEN = 512
_MAJOR_MAX_DIGITS = 9
_CACHE_SIZE = 4096
_ID_MASK = 0x7FFFFFFF

_HEX32 = re.compile(r"[0-9a-fA-F]{32}")
_JA4_OK = re.compile(r"[\x21-\x7b\x7d\x7e]+")      # printable ASCII, no space, no '|'
_MISSING_STR = frozenset({"", "-", "none", "null", "nan"})


# ------------------------------------------------------------------ helpers
def _num(x) -> float:
    """float(x), NaN when x is None or not numeric (never raises)."""
    if x is None:
        return math.nan
    try:
        return float(x)
    except (TypeError, ValueError, OverflowError):
        return math.nan


def _clean_str(x) -> Optional[str]:
    """Stripped str, or None for None / non-str / blank / log placeholders ('-')."""
    if isinstance(x, bytes):
        x = x.decode("utf-8", "replace")
    if not isinstance(x, str):
        return None
    s = x.strip()
    return None if s.lower() in _MISSING_STR else s


def _is_grease(v: int) -> bool:
    return (v & GREASE_MASK) == GREASE_VALUE


def _major(digits: Optional[str]) -> str:
    """Leading-integer major as a canonical decimal string ('0' if unknown)."""
    if not digits:
        return "0"
    # regex \d is Unicode (e.g. Arabic-Indic '١٢٦'); int() maps every Nd digit, so
    # str(int()) keeps the token field ASCII ('126'). Inputs are <= 512 chars.
    v = int(digits)
    return str(v) if v < 10 ** _MAJOR_MAX_DIGITS else "0"


# ------------------------------------------------------------------ ja3n
def _ja3_list(field: str) -> Optional[list]:
    """'4865-4866-49195' -> [4865, 4866, 49195]; '' -> []; None if malformed."""
    if not field:
        return []
    out = []
    for item in field.split("-"):
        item = item.strip()
        if not (item.isascii() and item.isdigit()) or len(item) > 10:
            return None
        v = int(item)
        if v > _JA3_U16_MAX:
            return None
        out.append(v)
    return out


@lru_cache(maxsize=_CACHE_SIZE)
def _ja3n_cached(s: str) -> str:
    if _HEX32.fullmatch(s):
        return s.lower()                      # already a JA3 hash: pass through
    fields = [f.strip() for f in s.split(",")]
    if not 3 <= len(fields) <= 5:
        return _NO_FP
    fields += [""] * (5 - len(fields))
    parsed = [_ja3_list(f) for f in fields]
    if any(p is None for p in parsed):
        return _NO_FP
    version, ciphers, exts, curves, pfs = parsed
    if len(version) != 1:
        return _NO_FP
    ciphers = [v for v in ciphers if not _is_grease(v)]
    exts = sorted(v for v in exts if not _is_grease(v))
    curves = [v for v in curves if not _is_grease(v)]
    norm = ",".join([str(version[0])] + ["-".join(map(str, lst))
                                         for lst in (ciphers, exts, curves, pfs)])
    return hashlib.md5(norm.encode("ascii"), usedforsecurity=False).hexdigest()


def ja3n(ja3: Optional[str]) -> str:
    """Normalised JA3 hash.

    JA3 = 'version,ciphers,extensions,curves,point_formats' ('-'-separated ints).
    Drop GREASE values from ciphers / extensions / curves, sort the extension
    list numerically (other lists keep order), re-join and md5-hex it.
    Two JA3s differing only in extension order map to the same ja3n.
    Empty / malformed -> '-'.

    Also (see module notes): 3..5 fields accepted (missing trailing fields are
    empty); a bare 32-hex JA3 hash is passed through lower-cased.
    """
    s = _clean_str(ja3)
    if s is None or len(s) > _JA3_MAX_LEN:
        return _NO_FP
    return _ja3n_cached(s)


def _clean_ja4(ja4) -> Optional[str]:
    """A JA4 usable as 'ja4:<ja4>' (no '|' / whitespace, bounded), else None."""
    s = _clean_str(ja4)
    if s is None or len(s) > _JA4_MAX_LEN or not _JA4_OK.fullmatch(s):
        return None
    return s


# ------------------------------------------------------------------ UA
_I = re.IGNORECASE

# 1) client libraries: anywhere in the UA, case-insensitive, first match wins.
_LIB_RULES: Tuple[Tuple[str, Pattern[str]], ...] = (
    ("python-requests", re.compile(r"(?<![\w.-])python-requests/(\d*)", _I)),
    ("go-http-client", re.compile(r"(?<![\w.-])go-http-client/(\d*)", _I)),
    ("okhttp", re.compile(r"(?<![\w.-])okhttp/(\d*)", _I)),
    ("curl", re.compile(r"(?<![\w.-])(?:lib)?curl/(\d*)", _I)),
    ("wget", re.compile(r"(?<![\w.-])wget/(\d*)", _I)),
    ("postman", re.compile(r"(?<![\w.-])postmanruntime/(\d*)", _I)),
    ("java", re.compile(r"(?<![\w.-])java(?:-http-client)?/(\d*)", _I)),
)

# 2) non-listed browsers / shells that also carry Chrome/ or Safari/ tokens.
_OTHER_BROWSER = re.compile(
    r"(?:\b(?:OPR|OPiOS|OPT|Opera|SamsungBrowser|YaBrowser|UCBrowser|Vivaldi|Electron"
    r"|HeadlessChrome|PhantomJS)|bot|spider|crawler)/(\d*)", _I)

# 3) listed browsers, in precedence order (Edg before Chrome before Safari).
_BROWSER_RULES: Tuple[Tuple[str, Pattern[str]], ...] = (
    ("edge", re.compile(r"\bEdg(?:e|A|iOS)?/(\d*)")),
    ("chrome", re.compile(r"\b(?:Chrome|CriOS|Chromium)/(\d*)")),
    ("firefox", re.compile(r"\b(?:Firefox|FxiOS)/(\d*)")),
)
_SAFARI = re.compile(r"\bSafari/")
_SAFARI_VERSION = re.compile(r"\bVersion/(\d*)")

# 4) fallback: first product token that is not 'Mozilla'. A match may only start
# at the beginning of a token-char run (lookbehind): starting at every letter
# made a long slash-less run quadratic (~1.6 ms for a 512-char junk UA, over
# R3's whole per-tick budget). The product name is the run from its first
# letter, as before ('360SE/12' -> 'SE'); the version digits sit in a lookahead
# so the next run may start right after the '/' ('Mozilla/5abc/7' -> abc/7).
_TOKEN_CH = r"[\w.+!#$%&'*^`~-]"
_PRODUCT = re.compile(rf"(?<!{_TOKEN_CH})({_TOKEN_CH}+)/(?=(\d*))")
_ALPHA = re.compile(r"[A-Za-z]")

_OS_RULES: Tuple[Tuple[str, Pattern[str]], ...] = (
    ("win", re.compile(r"\bWindows Phone\b", _I)),
    ("ios", re.compile(r"\b(?:iPhone|iPad|iPod|iOS)\b|\bCPU OS \d", _I)),
    ("android", re.compile(r"\bAndroid\b", _I)),
    ("win", re.compile(r"\bWin(?:dows|32|64|NT|9[58x])?\b", _I)),
    ("mac", re.compile(r"\b(?:Macintosh|Mac[ _]OS|macOS|Mac_PowerPC)\b", _I)),
    ("linux", re.compile(r"\b(?:Linux|Ubuntu|Fedora|Debian|CrOS)\b", _I)),
)


def _declared_os(ua: str) -> Optional[str]:
    for os_name, rx in _OS_RULES:
        if rx.search(ua):
            return os_name
    return None


def _family_major(ua: str) -> Tuple[str, str]:
    for fam, rx in _LIB_RULES:
        m = rx.search(ua)
        if m:
            return fam, _major(m.group(1))
    m = _OTHER_BROWSER.search(ua)
    if m:
        return "other", _major(m.group(1))
    for fam, rx in _BROWSER_RULES:
        m = rx.search(ua)
        if m:
            return fam, _major(m.group(1))
    if _SAFARI.search(ua):
        m = _SAFARI_VERSION.search(ua)
        return "safari", _major(m.group(1) if m else None)
    for m in _PRODUCT.finditer(ua):
        run = m.group(1)
        a = _ALPHA.search(run)
        if a is not None and run[a.start():].lower() != "mozilla":
            return "other", _major(m.group(2))
    return "other", "0"


@lru_cache(maxsize=_CACHE_SIZE)
def _ua_parse_cached(ua: str) -> Tuple[str, str, Optional[str]]:
    fam, major = _family_major(ua)
    return fam, major, _declared_os(ua)


def ua_parse(ua: Optional[str]) -> Tuple[str, str, Optional[str]]:
    """(family, major, declared_os or None) via ordered regexes (Edg/ before
    Chrome/, Chrome/ before Safari/, libraries: python-requests/, curl/,
    Go-http-client/, Java/, okhttp/, Wget/, PostmanRuntime/). Empty -> ('none', '0', None).

    Unlisted browsers (Opera, Samsung, HeadlessChrome, Electron, crawlers ...)
    and unknown clients are 'other'; see the module notes for the full order.
    """
    s = _clean_str(ua)
    if s is None:
        return "none", "0", None
    return _ua_parse_cached(s[:_UA_MAX_LEN])


# ------------------------------------------------------------------ TTL / window
def ttl_class(ttl: Optional[float]) -> int:
    """Smallest class in TTL_CLASSES >= ttl; ttl > 255 -> 255; None / <= 0 -> 0."""
    t = _num(ttl)
    if not t > 0:                             # also catches NaN
        return 0
    for c in TTL_CLASSES:
        if t <= c:
            return c
    return TTL_CLASSES[-1]


def win_class(win_size: Optional[float]) -> str:
    """'w{ceil(log2(win))}' clipped to WIN_CLASS_RANGE; None / <= 0 -> 'w0'."""
    w = _num(win_size)
    if not w > 0:
        return "w0"
    lo, hi = WIN_CLASS_RANGE
    if math.isinf(w):
        return f"w{hi}"
    m, e = math.frexp(w)                      # w = m * 2^e, 0.5 <= m < 1
    k = e - 1 if m == 0.5 else e              # exact ceil(log2(w))
    return f"w{min(max(k, lo), hi)}"


# ------------------------------------------------------------------ token
def stack_token(ja3: Optional[str], ua: Optional[str], ttl: Optional[float],
                win_size: Optional[float], ja4: Optional[str] = None) -> str:
    """Build 'ja3n|ua_family/major|os|ttl_class|win_class' (see module doc)."""
    j4 = _clean_ja4(ja4)
    fp = _JA4_PREFIX + j4 if j4 is not None else ja3n(ja3)
    fam, major, declared = ua_parse(ua)
    tc = ttl_class(ttl)
    os_name = declared or OS_FROM_TTL.get(tc) or UNKNOWN
    return f"{fp}|{fam}/{major}|{os_name}|{tc}|{win_class(win_size)}"


@lru_cache(maxsize=4 * _CACHE_SIZE)
def _stack_id_cached(token: str) -> int:
    d = hashlib.blake2b(token.encode("utf-8", "surrogatepass"), digest_size=8).digest()
    return (int.from_bytes(d, "big") & _ID_MASK) or 1


def stack_id(token: str) -> int:
    """Stable 31-bit id of a stack token: int.from_bytes(blake2b(token, 8)) & 0x7FFFFFFF.
    Shared by R2 (act.stream stack_id) and R3 (client.stack_events); 0 is never returned
    (a zero digest maps to 1). The digest is read big-endian; UTF-8 with surrogatepass."""
    return _stack_id_cached(token if isinstance(token, str) else str(token))


def parse_stack_token(token: str) -> Dict[str, str]:
    """Inverse view {'ja3n', 'ua_family', 'ua_major', 'os', 'ttl_class', 'win_class'}.

    Best effort on foreign input (never raises): with more than 5 fields the
    extra '|' are kept in the ja3n field (the only one that could hold one);
    with fewer, the missing trailing fields take their unknown values ('-',
    'none', '0', '?', '0', 'w0'); an empty or non-str token gives all defaults.
    """
    out = dict(_TOKEN_DEFAULTS)
    if not isinstance(token, str) or not token:
        return out
    parts = token.split("|")
    if len(parts) > 5:                        # a foreign '|' can only sit in the fp field
        parts = ["|".join(parts[:-4])] + parts[-4:]
    fp, ua_part, os_name, ttl_s, win_s = parts + [None] * (5 - len(parts))
    out["ja3n"] = fp or _NO_FP
    if ua_part:
        fam, sep, major = ua_part.partition("/")
        out["ua_family"] = fam or "none"
        out["ua_major"] = major if sep and major else "0"
    if os_name:
        out["os"] = os_name
    if ttl_s:
        out["ttl_class"] = ttl_s
    if win_s:
        out["win_class"] = win_s
    return out


def os_ttl_consistent(declared_os: Optional[str], ttl_cls: int) -> Optional[bool]:
    """B09 inconsistency input: False when a UA-declared OS contradicts the TTL
    class (win vs 64, linux/android/mac/ios vs 128); None when either is unknown.

    True otherwise, except TTL class 255 -> None (a network device re-originated
    the path, so the TTL says nothing about the client OS). ttl_cls goes through
    ttl_class first, so a raw TTL or a token's '64' string is accepted too.
    """
    d = _clean_str(declared_os)
    if d is None:
        return None
    d = d.lower()
    tc = ttl_class(ttl_cls)
    if d not in _DECLARED_OS or tc == 0 or tc == 255:
        return None
    if d == "win":
        return tc != 64
    return tc != 128                          # linux / android / mac / ios originate at 64
