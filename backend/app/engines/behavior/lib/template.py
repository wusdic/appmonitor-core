"""Drain-lite templating and action-token formats (R2; consumed by B08, B10, B13, B16, B22).

STATUS: implemented. Token formats, constants, dataclass layout and
signatures are frozen (docs/lib3/helpers_api.md). DEVIATION (additive): the
Templater dataclass gained private bookkeeping fields (`_heap`, `_dirty`,
`_n_nodes`, `_passes`, `_full_next`, `_force_at`, `_walk`; repr/compare off,
all defaulted, never serialised), so positional and keyword construction are
unchanged.

Why: raw paths / qnames carry ids that make every request unique. Masking
plus a small prefix tree turns them into a stable per-system vocabulary of
(channel, op, resource, outcome) tokens that sequence, novelty, identity and
budget models can count, while the masked values (object ids) are kept
separately for breadth budgets. Privacy: only templates and query parameter
NAMES survive; values never enter a token.

Token formats (exact, single spaces):
    HTTP : '{METHOD} {host} {template}|{status_class}'
           e.g. 'GET erp.corp /orders/view/{num}|2xx'
           template = masked path, then '?' + '&'.join(sorted(param names)) if a
           query string exists: '/search?page&q'; status_class in
           {'1xx','2xx','3xx','4xx','5xx','0xx'} ('0xx' = no response)
    TLS  : '{sni_etld1}:{dport} u{a}/d{b}'          (TLS without L7)
           e.g. 'example.com:443 u10/d14', a = floor(log2(max(1, bytes_up))),
           b likewise for bytes_down
    DNS  : '{QTYPE} {templated_qname}'
           e.g. 'TXT {rnd}.x.evil.com'; labels left of the eTLD+1 become
           '{rnd}' when char entropy >= 3.2 bits, length >= 20 or digit ratio >= 0.3
    L4   : '{proto}/{service} u{a}/d{b}'
           e.g. 'tcp/ssh u12/d9'; service from SERVICE_PORTS else the port number
    Eviction token: '{rare}'.  Sequence tokens append outcome: token + '|' + outcome
    where HTTP tokens already carry it.

Mask rules (applied per path segment, first match wins, in this order):
    UUID                               -> {uuid}
    e-mail                             -> {email}
    date (YYYY-MM-DD, YYYYMMDD, ...)   -> {date}
    hex of 16+ chars                   -> {hex}
    all digits                         -> {num}
    alnum 8+ chars with >= 3 digits    -> {id}
    base64url 16+ chars or char-entropy >= 3.5 bits -> {tok}
Digits inside an otherwise literal segment ('v2') are kept.

Prefix tree per (system, host, method), depth <= 6 segments: a node with
> 40 children whose top child has < 0.5 share collapses to {var}; siblings
with count <= 2 merge lazily into their parent's {var} on the next
maintenance pass. Vocabulary <= 4000 tokens per system; Space-Saving evicts
the minimum-count token to '{rare}'.

Implementation notes (the choices the rules above leave open):
  * Masks, precisely.
      - uuid: 8-4-4-4-12 hex with dashes (32 bare hex chars are {hex}).
      - email: local@domain.tld, tested on the percent-decoded segment
        ('a%40b.com' is an e-mail).
      - date: YYYY-MM-DD with an optional ISO time ('2024-01-15T10:30:00Z'),
        YYYY[-_.]M[M][-_.]D[D] (one separator used consistently), YYYY-MM,
        D[D][-.]M[M][-.]YYYY (either day/month order), YYYYMMDD and
        YYYYMMDDTHHMM[SS][Z]. Month 1-12, day 1-31, year 1900-2199; the bare
        8-digit form only for 1970-2099, so most 8-digit order numbers stay
        {num}. 12/14-digit timestamps are {num} (timestamp-shaped order ids
        are object ids).
      - hex: 16+ hex chars with at least one digit AND one letter a-f. An
        all-digit string is a number, not a digest: snowflake / 18-digit
        order ids must stay {num} (an OBJ mask, kept for act.objs breadth).
      - {id}: ASCII alnum, 8+ chars, >= 3 digits (verbatim).
      - {tok}: the spec condition (base64url charset and 16+ chars, or
        char entropy >= 3.5 bits) AND the string looks random, not like an
        identifier. Raw entropy alone would mask ordinary names:
        'getUserProfileByEmail' has 3.88 bits and
        'customer-relationship-management' 3.84. "Looks random" = ASCII, the
        alnum chunks of 8+ chars (split at non-alnum) mix at least two of
        {upper, lower, digit}, and they fragment into >= 0.4 identifier
        parts per char (parts = regex [A-Z]+(?![a-z]) | [A-Z]?[a-z]+ | digits,
        i.e. a camelCase / acronym / number splitter). Measured: random
        strings of 16 / 22 / 32 chars are masked ({tok}, or {id} when alnum
        with >= 3 digits) 88 / 94 / 97 % (base64url) and 92 / 96 / 98 %
        (base62), while camelCase, PascalCase, acronyms, kebab / snake slugs
        and versioned file names ('jquery-3.6.0.min.js') stay literal. A
        missed token is harmless: it is a literal that the tree reports as
        {var} until it repeats (and crowding merges / collapses it after).
      - A segment whose whole value is not masked but which ends in a short
        extension ('.jpg', 1-5 alnum chars, not all digits) is re-tested on
        its stem: '12345.jpg' -> '{num}.jpg' (value '12345').
      - Matrix parameters (';jsessionid=...') are dropped from a segment,
        empty segments are dropped ('/a//b/' == '/a/b'), a fragment is
        dropped, an absolute-form URL is reduced to its path.
      - Literal segments are kept verbatim (case preserved) except that
        whitespace, control chars and '|{}?#&' are percent-encoded (a
        literal can never alias a mask token or break the token grammar)
        and they are capped at 64 chars. Query parameter names get the same
        masking (a '?1699999999' cache-buster is '?{num}'), are deduplicated,
        sorted and capped at 16; an empty query string adds no '?'.
      - Masked values are returned for every mask (mask_segment's value is
        the percent-decoded original); callers keep OBJ_MASKS for act.objs.
  * Tree.
      - Every node on the walk (root included) gains w. The masks and {var}
        are "stable" children. A literal child (or a mask with an extension)
        is PROVISIONAL while its count <= MERGE_MAX_COUNT and is reported as
        {var}: the template of a request therefore does not depend on when
        the lazy merge runs (before or after it, a count-2 sibling reads
        {var}), a 404 / scanner flood of never-repeating paths is one
        template from the first request on, and a new endpoint gets its
        literal name from its 3rd hit (the same "system count >= 3" floor
        B10 uses for {rare:<channel>}).
      - maintain() merges count <= 2 literal siblings into the parent's {var}
        child only in CROWDED nodes (> VAR_MIN_CHILDREN children). Because
        such siblings are already reported as {var}, the merge is a pure
        memory reclaim; doing it unconditionally would, with a pass every
        tick at dt = 60 s, reset every sparse endpoint each minute so that it
        could never reach 3 hits. After the merge a node that still has
        > 40 children with top share < 0.5 collapses: all subtrees merge
        into one {var} child and is_var marks the position variable, so
        every later segment there (masks included) is {var}.
      - Hard caps between passes: <= 256 children per node, <= 100k nodes
        and <= 512 (host, method) trees per Templater; overflow routes new
        literals to {var} (no node is created). When the tree nears the node
        cap (90k), a pass merges rare literals everywhere, crowded or not,
        and is not forced again before another 5k nodes exist; with more
        than 256 trees, trees whose root count is <= 2 are dropped.
      - Segments below MAX_DEPTH are not modelled: a deeper path ends in one
        '{var}' ('/a/b/c/d/e/f/{var}'), whatever its length.
      - maintain() visits only nodes that gained a child since the previous
        pass (plus merged subtrees); every 64th pass is a full traversal so
        a slow drift of the top-child share is still seen. O(changed nodes).
      - Weights w <= 0 or non-finite are lookups: nothing is counted or
        created (an unseen literal reads {var}).
  * Methods are upper-cased; HTTP_METHODS and the common WebDAV verbs are
    kept, anything else is 'OTHER' (bounds the (host, method) keys). Hosts go
    through names.normalize_host (lower case, no port / scheme); '' -> '-'.
  * DNS: runs of consecutive {rnd} labels collapse into one, so a tunnel
    that varies its label count per query keeps one token. The three
    randomness thresholds are applied verbatim (so 's3' -> {rnd}); an IP
    literal or a bare registrable domain is kept as is; '' -> '.'. Numeric
    qtypes map to their names ('16' -> 'TXT', else 'TYPE<n>'); an empty or
    malformed qtype is 'UNK'.
  * size_class: floor(log2) via frexp (exact at powers of two); NaN, inf,
    negative or unparsable byte counts count as 0 (u0). TLS: IPv6 literals
    are bracketed ('[2001:db8::1]:443'); an empty SNI is '-'.
  * Space-Saving (Metwally et al. 2005): a new token when the vocabulary is
    full replaces the argmin of (count, token_id), inherits count c_min + w
    and error c_min. The minimum is found with a lazy min-heap (stale
    entries are re-pushed on pop), so an eviction is amortised O(log V)
    instead of a 4000-entry scan; the argmin, hence the state, is the same
    after a to_dict / from_dict round trip. intern('{rare...}') and
    intern of an unseen token with w <= 0 return 0 and change nothing.
    Evicted ids leave id_to_token, so token_of(evicted id) == '{rare}'.
  * channel_of / token_family parse the token shape (they also accept a
    trailing '|outcome' and B10's '{rare:<channel>}'). Families:
        http: 'http|<op>|<host>|<first path segment>', op in read / write /
              other ('read' = GET/HEAD/OPTIONS, 'write' = WRITE_METHODS)
        dns : 'dns|<op>|<etld1>|<label next to the etld1>', op in addr / txt /
              other ('addr' = A/AAAA/CNAME/HTTPS/SVCB, 'txt' = TXT/NULL)
        tls : 'tls|conn|<etld1>|<dport>'
        l4  : 'l4|<proto>||<service>'
        rare: '<channel of {rare:<channel>} or rare>|||'
    e.g. 'http|read|erp.corp|orders', 'dns|txt|evil.com|x', 'tls|conn|example.com|443'.
  * Cost (CPython 3.11, measured): http_token + intern ~2.3 us for
    '/orders/view/<id>', ~1.4 us for a static path, ~2.5 us on a mixed
    stream. Memos: unmasked literal segments, whole paths without a masked
    value, the directory part of a path, query strings, (host, method)
    pairs, and per Templater the node walk of a fully established path
    ((key, segments, params) -> nodes + template; cleared by maintain(), the
    only place the structure of existing nodes changes). Module memos hold
    <= 32768 entries each (cleared when full) and never string keys longer
    than 256 chars, so a flood of long unique paths cannot bloat them.
    Numeric ids take a str.isdigit fast path. maintain() is O(changed nodes).
"""
from __future__ import annotations

import heapq
import math
import re
from collections import Counter
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple
from urllib.parse import unquote

from . import names as _names

MASK_TOKENS: Tuple[str, ...] = ("{uuid}", "{email}", "{date}", "{hex}", "{num}", "{id}", "{tok}")
OBJ_MASKS = frozenset({"{num}", "{id}", "{uuid}"})     # values kept for act.objs
VAR_TOKEN = "{var}"
RND_TOKEN = "{rnd}"
RARE_TOKEN = "{rare}"
MAX_DEPTH = 6
VAR_MIN_CHILDREN = 40
VAR_MAX_TOP_SHARE = 0.5
MERGE_MAX_COUNT = 2
VOCAB_CAP = 4000
DNS_RND_ENTROPY = 3.2
DNS_RND_LEN = 20
DNS_RND_DIGIT_RATIO = 0.3
TOK_ENTROPY = 3.5
HTTP_METHODS = frozenset({"GET", "POST", "PUT", "PATCH", "DELETE", "HEAD", "OPTIONS",
                          "CONNECT", "TRACE"})
WRITE_METHODS = frozenset({"POST", "PUT", "PATCH", "DELETE"})
DNS_QTYPES = frozenset({"A", "AAAA", "CNAME", "MX", "NS", "PTR", "SOA", "SRV", "TXT",
                        "ANY", "HTTPS", "SVCB", "CAA", "DS", "DNSKEY", "NULL"})
SERVICE_PORTS: Dict[int, str] = {
    20: "ftp-data", 21: "ftp", 22: "ssh", 23: "telnet", 25: "smtp", 53: "dns", 80: "http",
    88: "kerberos", 110: "pop3", 123: "ntp", 135: "msrpc", 139: "netbios", 143: "imap",
    161: "snmp", 389: "ldap", 443: "https", 445: "smb", 465: "smtps", 514: "syslog",
    587: "submission", 636: "ldaps", 993: "imaps", 995: "pop3s", 1433: "mssql",
    1521: "oracle", 2049: "nfs", 3306: "mysql", 3389: "rdp", 5432: "postgres",
    5900: "vnc", 5985: "winrm", 5986: "winrm-s", 6379: "redis", 8080: "http-alt",
    8443: "https-alt", 9200: "elastic", 27017: "mongo",
}

# ------------------------------------------------------------ private constants
_STABLE = frozenset(MASK_TOKENS) | {VAR_TOKEN}      # children never provisional / merged
_OTHER_METHOD = "OTHER"
_EXTRA_METHODS = frozenset({"PROPFIND", "PROPPATCH", "MKCOL", "COPY", "MOVE", "LOCK",
                            "UNLOCK", "REPORT", "SEARCH", "MKCALENDAR", "PURGE"})
_READ_METHODS = frozenset({"GET", "HEAD", "OPTIONS"})
_DNS_ADDR = frozenset({"A", "AAAA", "CNAME", "HTTPS", "SVCB"})
_DNS_TXT = frozenset({"TXT", "NULL"})
_QTYPE_NUM = {1: "A", 2: "NS", 5: "CNAME", 6: "SOA", 10: "NULL", 12: "PTR", 15: "MX",
              16: "TXT", 28: "AAAA", 33: "SRV", 43: "DS", 48: "DNSKEY", 64: "SVCB",
              65: "HTTPS", 255: "ANY", 257: "CAA"}
_CHANNELS = frozenset({"http", "tls", "dns", "l4"})
_SC = ("0xx", "1xx", "2xx", "3xx", "4xx", "5xx")

_MAX_SEGS = 32              # path segments masked per request (deeper ones are the tail anyway)
_MAX_LITERAL = 64           # chars kept of an unmasked segment
_MAX_PARAMS = 16            # query parameter names kept
_MAX_CHILDREN = 256         # children per node between maintenance passes
_MAX_NODES = 100_000        # nodes per Templater (all trees)
_SOFT_NODES = 90_000        # a pass above this merges rare literals everywhere
_MAX_TREES = 512            # (host, method) trees per Templater
_FULL_PASS_EVERY = 64       # every n-th maintain() traverses every tree
_CACHE_MAX = 1 << 15        # entries per memo dict (cleared when full)
_MEMO_KEY_MAX = 256         # longer string keys are never memoised (bounds memo memory)
_FMT = 1                    # to_dict format version
_WALK_MAX = 4096            # memoised (key, segments, params) walks per Templater

_TOK_FRAG = 0.4             # identifier parts per char above which a chunk looks random
_TOK_CHUNK = 8              # alnum chunks shorter than this are ignored by the test

_UUID_RE = re.compile(r"[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}"
                      r"-[0-9a-fA-F]{12}")
_EMAIL_RE = re.compile(r"[A-Za-z0-9._%+\-]+@[A-Za-z0-9\-]+(?:\.[A-Za-z0-9\-]+)*\.[A-Za-z]{2,}")
_HEX_RE = re.compile(r"[0-9a-fA-F]{16,}")
_B64URL_RE = re.compile(r"[A-Za-z0-9_\-]{16,}={0,2}")
_ISO_RE = re.compile(r"(\d{4})-(\d{2})-(\d{2})(?:[T ]\d{2}:\d{2}(?::\d{2}(?:\.\d{1,9})?)?"
                     r"(?:Z|[+-]\d{2}:?\d{2})?)?", re.ASCII)
_YMD_RE = re.compile(r"(\d{4})([-_.])(\d{1,2})\2(\d{1,2})", re.ASCII)
_YM_RE = re.compile(r"(\d{4})[-_.](\d{2})", re.ASCII)
_DMY_RE = re.compile(r"(\d{1,2})([-.])(\d{1,2})\2(\d{4})", re.ASCII)
_COMPACT_DT_RE = re.compile(r"(\d{4})(\d{2})(\d{2})T\d{4}(?:\d{2})?Z?", re.ASCII)
_PART_RE = re.compile(r"[A-Z]+(?![a-z])|[A-Z]?[a-z]+|[0-9]+")
_NONALNUM_RE = re.compile(r"[^A-Za-z0-9]+")
_UNSAFE_RE = re.compile(r"[\x00-\x20\x7f|{}?#&]")
_SIZE_RE = re.compile(r"u\d+/d\d+", re.ASCII)

# memo dicts (plain dicts: they sit in R2's per-request loop)
_LIT: Dict[str, str] = {}          # raw segment -> literal output (unmasked segments only)
# whole paths without a masked value -> (segments, param names, [])
_PATH: Dict[Any, Tuple[Tuple[str, ...], Tuple[str, ...], List[Tuple[str, str]]]] = {}
_PRE: Dict[str, Tuple[str, ...]] = {}   # directory part of a path -> masked segments (no-mask only)
_QRY: Dict[str, Tuple[str, ...]] = {}   # raw query string -> sorted masked names
_KEY: Dict[Tuple[Any, Any], Tuple[str, str]] = {}   # (host, method) -> normalised pair
_SCM: Dict[Any, str] = {}          # status -> status class
_DNS_T: Dict[str, str] = {}
_CHAN: Dict[str, str] = {}
_FAM: Dict[str, str] = {}


def _memo_put(d: Dict, k: Any, v: Any) -> None:
    if k.__class__ is str and len(k) > _MEMO_KEY_MAX:
        return
    if len(d) >= _CACHE_MAX:
        d.clear()
    d[k] = v


# ================================================================ masking
def _entropy_bits(s: str) -> float:
    """Shannon entropy of the character distribution of s (bits per char)."""
    n = len(s)
    if n < 2:
        return 0.0
    acc = 0.0
    for c in Counter(s).values():
        acc += c * math.log2(c)
    return max(0.0, math.log2(n) - acc / n)


def _ymd_ok(y: int, m: int, d: int, y0: int = 1900, y1: int = 2199) -> bool:
    return y0 <= y <= y1 and 1 <= m <= 12 and 1 <= d <= 31


def _is_date(s: str) -> bool:
    """Non-bare date shapes (the bare 8-digit form is tested on the digit path)."""
    m = _ISO_RE.fullmatch(s)
    if m:
        return _ymd_ok(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    m = _YMD_RE.fullmatch(s)
    if m:
        return _ymd_ok(int(m.group(1)), int(m.group(3)), int(m.group(4)))
    m = _YM_RE.fullmatch(s)
    if m:
        return 1900 <= int(m.group(1)) <= 2199 and 1 <= int(m.group(2)) <= 12
    m = _DMY_RE.fullmatch(s)
    if m:
        a, b, y = int(m.group(1)), int(m.group(3)), int(m.group(4))
        return _ymd_ok(y, b, a) or _ymd_ok(y, a, b)
    m = _COMPACT_DT_RE.fullmatch(s)
    if m:
        return _ymd_ok(int(m.group(1)), int(m.group(2)), int(m.group(3)))
    return False


def _looks_random(s: str) -> bool:
    """Identifier-splitter fragmentation of the long alnum chunks (see module notes)."""
    n = parts = 0
    up = lo = dg = False
    for chunk in _NONALNUM_RE.split(s):
        if len(chunk) < _TOK_CHUNK:
            continue
        n += len(chunk)
        parts += len(_PART_RE.findall(chunk))
        if not up and not chunk.islower() and any("A" <= c <= "Z" for c in chunk):
            up = True
        if not lo and any("a" <= c <= "z" for c in chunk):
            lo = True
        if not dg and any("0" <= c <= "9" for c in chunk):
            dg = True
    if n == 0 or (up + lo + dg) < 2:
        return False
    return parts / n >= _TOK_FRAG


def _mask_core(s: str) -> Optional[str]:
    """Mask token for a whole (decoded) segment, rules in spec order; None = literal."""
    n = len(s)
    if not n:
        return None
    if s.isdigit() and s.isascii():
        # all digits: never uuid / email / hex (a digest needs a letter); maybe a bare date
        if n == 8 and _ymd_ok(int(s[:4]), int(s[4:6]), int(s[6:]), 1970, 2099):
            return "{date}"
        return "{num}"
    if n == 36 and _UUID_RE.fullmatch(s):
        return "{uuid}"
    if "@" in s and _EMAIL_RE.fullmatch(s):
        return "{email}"
    if n >= 7 and "0" <= s[0] <= "9" and _is_date(s):
        return "{date}"
    if n >= 16 and _HEX_RE.fullmatch(s) and not s.isalpha():
        return "{hex}"
    if n >= 8 and s.isascii() and s.isalnum():
        nd = 0
        for c in s:
            if "0" <= c <= "9":
                nd += 1
                if nd >= 3:
                    return "{id}"
    if n >= 12 and s.isascii():
        if ((n >= 16 and _B64URL_RE.fullmatch(s)) or _entropy_bits(s) >= TOK_ENTROPY) \
                and _looks_random(s):
            return "{tok}"
    return None


def _pct(m: "re.Match[str]") -> str:
    return "%%%02X" % ord(m.group())


def _literal(seg: str) -> str:
    if _UNSAFE_RE.search(seg):
        seg = _UNSAFE_RE.sub(_pct, seg)
    if len(seg) > _MAX_LITERAL:
        seg = seg[:_MAX_LITERAL - 3] + "..."
    return seg


def mask_segment(seg: str) -> Tuple[str, Optional[str]]:
    """(masked, original value if it was masked by an OBJ/other rule else None)."""
    if seg.__class__ is not str:
        seg = "" if seg is None else str(seg)
    lit = _LIT.get(seg)
    if lit is not None:
        return lit, None
    if not seg:
        return "", None
    d = seg
    if "%" in seg:
        try:
            d = unquote(seg)
        except Exception:                     # pragma: no cover - unquote is lenient
            d = seg
    m = _mask_core(d)
    if m is not None:
        return m, d
    dot = d.rfind(".")
    if 0 < dot < len(d) - 1 and len(d) - dot <= 6:
        ext = d[dot + 1:]
        if ext.isascii() and ext.isalnum() and not ext.isdigit():
            stem = d[:dot]
            m = _mask_core(stem)
            if m is not None:
                return m + "." + ext, stem
    lit = _literal(seg)
    if ";" not in seg:              # mask_path strips matrix params before its memo lookup
        _memo_put(_LIT, seg, lit)
    return lit, None


def _query_names(query: str) -> Tuple[str, ...]:
    hit = _QRY.get(query)
    if hit is None:
        names = set()
        for part in query.split("&"):
            nm = part.split("=", 1)[0] if part else ""
            if nm:
                names.add(mask_segment(nm)[0])
        hit = tuple(sorted(names)[:_MAX_PARAMS])
        _memo_put(_QRY, query, hit)
    return hit


def _split_mask(p: str, segs: List[str], masked: List[Tuple[str, str]]) -> None:
    """Mask every '/'-separated segment of p into segs / masked (in place, capped)."""
    lit_get = _LIT.get
    for raw in p.split("/"):
        if len(segs) >= _MAX_SEGS:
            return
        if not raw:
            continue
        lit = lit_get(raw)
        if lit is not None:                            # known literal (memo)
            segs.append(lit)
        elif raw.isdigit() and raw.isascii() and len(raw) != 8:
            segs.append("{num}")               # numeric id fast path (8 digits: maybe a date)
            masked.append(("{num}", raw))
        else:
            if ";" in raw:                             # matrix params (jsessionid, ...)
                raw = raw[:raw.index(";")]
                if not raw:
                    continue
            m, v = mask_segment(raw)
            if v is not None:
                masked.append((m if m in _STABLE else m[:m.index("}") + 1], v))
            segs.append(m)


def _mask_path(path: Any) -> Tuple[Tuple[str, ...], Tuple[str, ...], List[Tuple[str, str]]]:
    """mask_path without defensive copies (the Templater hot path). A path with
    no masked value is memoised whole and its `masked` list is SHARED: callers
    copy it before handing it out. The directory part of a path (all but the
    last segment) is memoised when it has no masked value, since REST ids
    mostly sit in the last segment ('/orders/view/<id>')."""
    try:
        hit = _PATH.get(path)
    except TypeError:                                  # unhashable garbage
        hit, path = None, str(path)
    if hit is not None:
        return hit
    if not path:
        return (), (), []
    key = path
    if not isinstance(path, str):
        path = str(path)
    h = path.find("#")
    if h >= 0:
        path = path[:h]
    q = path.find("?")
    if q >= 0:
        query, p = path[q + 1:], path[:q]
    else:
        query, p = "", path
    if p[:1] != "/":
        j = p.find("://")
        if 0 < j <= 10 and p[:j].isalpha():           # absolute-form 'http://host/x'
            p = p[j + 3:]
            k = p.find("/")
            p = p[k:] if k >= 0 else ""
    masked: List[Tuple[str, str]] = []
    j = p.rfind("/")
    if j > 0:
        pre = p[:j]
        segs_pre = _PRE.get(pre)
        if segs_pre is None:
            lst: List[str] = []
            _split_mask(pre, lst, masked)
            segs_pre = tuple(lst)
            if not masked:
                _memo_put(_PRE, pre, segs_pre)
        last = p[j + 1:]
    else:
        segs_pre = ()
        last = p[j + 1:]                               # j in (-1, 0)
    if not last or len(segs_pre) >= _MAX_SEGS:
        segs = segs_pre
    elif last.isdigit() and last.isascii() and len(last) != 8:
        segs = segs_pre + ("{num}",)                   # numeric id fast path
        masked.append(("{num}", last))
    else:
        lit = _LIT.get(last)
        if lit is not None:
            segs = segs_pre + (lit,)
        else:
            lst = []
            _split_mask(last, lst, masked)
            segs = segs_pre + tuple(lst)
    params = _query_names(query) if query else ()
    out = (segs, params, masked)
    if not masked:
        _memo_put(_PATH, key, out)
    return out


def mask_path(path: str) -> Tuple[List[str], List[str], List[Tuple[str, str]]]:
    """(masked segments, sorted query parameter names, [(mask, value)] of masked values)."""
    segs, params, masked = _mask_path(path)
    return list(segs), list(params), list(masked)


# ============================================================ small tokens
def status_class(status: Optional[int]) -> str:
    """'2xx' etc.; None / 0 / out of range -> '0xx'."""
    if status.__class__ is int:
        hit = _SCM.get(status)
        if hit is not None:
            return hit
    if status is None or isinstance(status, bool):
        return "0xx"
    try:
        s = int(status)
    except (TypeError, ValueError, OverflowError):
        return "0xx"
    out = _SC[s // 100] if 100 <= s <= 599 else "0xx"
    if status.__class__ is int and len(_SCM) < 4096:
        _SCM[status] = out
    return out


def _log2_floor(x: Any) -> int:
    try:
        v = float(x)
    except (TypeError, ValueError, OverflowError):
        return 0
    if not (v > 1.0) or v == math.inf:          # NaN, <= 1, inf -> 0
        return 0
    return math.frexp(v)[1] - 1                  # exact floor(log2 v)


def size_class(bytes_up: float, bytes_down: float) -> str:
    """'u{floor(log2(max(1, up)))}/d{floor(log2(max(1, down)))}'."""
    return f"u{_log2_floor(bytes_up)}/d{_log2_floor(bytes_down)}"


def dns_label_is_random(label: str) -> bool:
    """entropy >= 3.2 bits or len >= 20 or digit ratio >= 0.3."""
    n = len(label) if label else 0
    if n == 0:
        return False
    if n >= DNS_RND_LEN:
        return True
    nd = 0
    for c in label:
        if "0" <= c <= "9":
            nd += 1
    if nd / n >= DNS_RND_DIGIT_RATIO:            # a ratio, not nd >= 0.3 * n (0.3 * 10 > 3.0)
        return True
    # entropy >= 3.2 needs >= 10 distinct chars (log2(9) = 3.17)
    return n >= 10 and _entropy_bits(label) >= DNS_RND_ENTROPY


def dns_template(qname: str) -> str:
    """Random labels left of the eTLD+1 -> '{rnd}' (names.split_host); lower-case."""
    key = qname if isinstance(qname, str) else str(qname or "")
    hit = _DNS_T.get(key)
    if hit is not None:
        return hit
    left, reg = _names.split_host(key)
    if not reg:
        out = "."
    else:
        parts: List[str] = []
        for lb in left:
            if dns_label_is_random(lb):
                if not parts or parts[-1] != RND_TOKEN:
                    parts.append(RND_TOKEN)
            else:
                parts.append(_literal(lb))
        parts.append(_literal(reg))
        out = ".".join(parts)
    _memo_put(_DNS_T, key, out)
    return out


def _port(dport: Any) -> int:
    try:
        p = int(dport)
    except (TypeError, ValueError, OverflowError):
        return 0
    return p if 0 <= p <= 65535 else 0


def tls_token(sni: str, dport: int, bytes_up: float, bytes_down: float) -> str:
    e = _literal(_names.etld1(sni or "")) if sni else ""
    if not e:
        e = "-"
    elif ":" in e:                               # IPv6 literal
        e = f"[{e}]"
    return f"{e}:{_port(dport)} {size_class(bytes_up, bytes_down)}"


def _norm_qtype(qtype: Any) -> str:
    q = str(qtype if qtype is not None else "").strip().upper()
    if not q:
        return "UNK"
    if q.isdigit():
        n = int(q)
        return _QTYPE_NUM.get(n, f"TYPE{n}")
    if q in DNS_QTYPES:
        return q
    return q if (q.isascii() and q.isalnum() and len(q) <= 12) else "UNK"


def dns_token(qtype: str, qname: str) -> str:
    return f"{_norm_qtype(qtype)} {dns_template(qname)}"


def l4_token(proto: str, dport: int, bytes_up: float, bytes_down: float) -> str:
    p = str(proto or "").strip().lower()
    p = "".join(c for c in p if c.isalnum()) or "ip"
    port = _port(dport)
    return f"{p}/{SERVICE_PORTS.get(port, str(port))} {size_class(bytes_up, bytes_down)}"


# ======================================================= token inspection
def _channel(token: str) -> str:
    if not token:
        return "rare"
    if token[0] == "{":
        if token.startswith("{rare:"):
            ch = token[6:].split("}", 1)[0]
            return ch if ch in _CHANNELS else "rare"
        return "rare"
    parts = token.split(" ")
    if len(parts) >= 3:
        if parts[0] in HTTP_METHODS or parts[2][:1] in ("/", "*"):
            return "http"
        return "rare"
    if len(parts) == 2:
        a, b = parts
        if _SIZE_RE.fullmatch(b.split("|", 1)[0]):
            if "/" in a:
                return "l4"
            if ":" in a:
                return "tls"
            return "rare"
        if a in DNS_QTYPES or (a.isascii() and a.isalnum() and a.upper() == a):
            return "dns"
    return "rare"


def channel_of(token: str) -> str:
    """'http' | 'tls' | 'dns' | 'l4' | 'rare' inferred from the token format
    (first word an HTTP method / a DNS qtype / contains 'host:port' / 'proto/svc')."""
    hit = _CHAN.get(token)
    if hit is None:
        hit = _channel(token if isinstance(token, str) else "")
        _memo_put(_CHAN, token, hit)
    return hit


def _family(token: str) -> str:
    ch = _channel(token)
    if ch == "rare" or token[:1] == "{":
        return f"{ch}|||"
    parts = token.split(" ")
    if ch == "http":
        m, host = parts[0], parts[1]
        op = "read" if m in _READ_METHODS else ("write" if m in WRITE_METHODS else "other")
        tpl = parts[2].split("|", 1)[0].split("?", 1)[0]
        seg0 = tpl.lstrip("/").split("/", 1)[0]
        return f"http|{op}|{host}|{seg0}"
    if ch == "dns":
        qt, qn = parts[0], parts[1].split("|", 1)[0]
        op = "addr" if qt in _DNS_ADDR else ("txt" if qt in _DNS_TXT else "other")
        left, reg = _names.split_host(qn)
        return f"dns|{op}|{reg}|{left[-1] if left else ''}"
    if ch == "tls":
        hp = parts[0]
        host, _, port = hp.rpartition(":")
        return f"tls|conn|{host}|{port}"
    # l4
    proto, _, svc = parts[0].partition("/")
    return f"l4|{proto}||{svc}"


def token_family(token: str) -> str:
    """Coarse family 'channel|method class|host|first path segment' (B02 role
    descriptor, B10 dwell families); e.g. 'http|read|erp.corp|orders'."""
    hit = _FAM.get(token)
    if hit is None:
        hit = _family(token if isinstance(token, str) else "")
        _memo_put(_FAM, token, hit)
    return hit


# ================================================================ templater
@dataclass
class _Node:
    count: float = 0.0
    children: Dict[str, "_Node"] = field(default_factory=dict)
    is_var: bool = False


def _norm_key(host: Any, method: Any) -> Tuple[str, str]:
    """(normalised host, normalised method), memoised per raw pair."""
    try:
        hit = _KEY.get((host, method))
    except TypeError:                                  # unhashable garbage
        hit, host, method = None, str(host), str(method)
    if hit is None:
        h = _names.normalize_host(host if isinstance(host, str) else str(host or ""))
        m = str(method or "").strip().upper()
        hit = (_literal(h) if h else "-",
               m if (m in HTTP_METHODS or m in _EXTRA_METHODS) else _OTHER_METHOD)
        if len(str(host)) <= _MEMO_KEY_MAX and len(str(method)) <= 32:
            _memo_put(_KEY, (host, method), hit)
    return hit


def _merge_into(dst: _Node, src: _Node) -> int:
    """Add src's counts and subtree into dst; src is left empty (a detached node
    that is maintained later is then a no-op). Returns the number of nodes that
    disappear (src plus every descendant that folded into an existing node)."""
    if src is dst:
        return 0
    gone = 1
    dst.count += src.count
    dch = dst.children
    for k, sc in src.children.items():
        dc = dch.get(k)
        if dc is None:
            dch[k] = sc
        else:
            gone += _merge_into(dc, sc)
    src.children = {}
    src.count = 0.0
    return gone


def _maintain_node(node: _Node, recurse: bool, force: bool, gone: List[int]) -> int:
    """Merge / collapse at one node. recurse=True maintains the whole subtree
    and returns its node count; otherwise only subtrees that received merged
    nodes are revisited (return value unused). gone[0] accumulates the net
    number of nodes removed, so a partial pass keeps Templater._n_nodes exact
    (the node caps and the forced pass must see the real tree size)."""
    ch = node.children
    targets: List[_Node] = []
    if ch:
        if node.is_var:
            if len(ch) > 1 or VAR_TOKEN not in ch:        # fold strays into the single {var}
                var = ch.get(VAR_TOKEN)
                if var is None:
                    var = _Node()
                    gone[0] -= 1
                for k in [k for k in ch if k != VAR_TOKEN]:
                    gone[0] += _merge_into(var, ch.pop(k))
                ch[VAR_TOKEN] = var
                targets.append(var)
        else:
            if force or len(ch) > VAR_MIN_CHILDREN:
                rare = [k for k, c in ch.items()
                        if k not in _STABLE and c.count <= MERGE_MAX_COUNT]
                if rare:
                    var = ch.get(VAR_TOKEN)
                    if var is None:
                        var = ch[VAR_TOKEN] = _Node()
                        gone[0] -= 1
                    for k in rare:
                        gone[0] += _merge_into(var, ch.pop(k))
                    targets.append(var)
            if len(ch) > VAR_MIN_CHILDREN:
                tot = 0.0
                top = 0.0
                for c in ch.values():
                    tot += c.count
                    if c.count > top:
                        top = c.count
                if top < VAR_MAX_TOP_SHARE * tot:
                    var = _Node()
                    gone[0] -= 1
                    for c in list(ch.values()):
                        gone[0] += _merge_into(var, c)
                    node.children = {VAR_TOKEN: var}
                    node.is_var = True
                    targets = [var]
    if recurse:
        total = 1
        for c in node.children.values():
            total += _maintain_node(c, True, force, gone)
        return total
    for t in targets:
        _maintain_node(t, True, force, gone)
    return 0


def _enc_node(n: _Node) -> List[Any]:
    c = float(n.count)
    return [c if math.isfinite(c) else 0.0, 1 if n.is_var else 0,
            {k: _enc_node(v) for k, v in n.children.items()}]


def _dec_node(x: Any, depth: int = 0) -> Tuple[_Node, int]:
    """(node, node count); malformed entries become empty nodes."""
    if isinstance(x, _Node):
        return x, 1
    node = _Node()
    total = 1
    if isinstance(x, (list, tuple)) and len(x) >= 3:
        try:
            c = float(x[0])
        except (TypeError, ValueError):
            c = 0.0
        node.count = c if math.isfinite(c) and c > 0.0 else 0.0
        node.is_var = bool(x[1])
        if isinstance(x[2], dict) and depth <= MAX_DEPTH + 1:
            for k, v in x[2].items():
                child, n = _dec_node(v, depth + 1)
                node.children[str(k)] = child
                total += n
    return node, total


@dataclass
class Templater:
    """Per-system Drain-lite state (persisted as model.template@(s, '__system__')).

    trees      {(host, method): root _Node}
    vocab      {token: [token_id, count, error]} Space-Saving counters (<= VOCAB_CAP)
    next_id    next token id; ids are never reused, so act.stream token_ids stay
               valid; an evicted token's id maps to RARE_TOKEN thereafter
    """
    trees: Dict[Tuple[str, str], _Node] = field(default_factory=dict)
    vocab: Dict[str, List[float]] = field(default_factory=dict)
    id_to_token: Dict[int, str] = field(default_factory=dict)
    next_id: int = 1          # 0 is reserved for RARE_TOKEN
    # private bookkeeping (rebuilt by from_dict; never serialised)
    _heap: List[Tuple[float, int, str]] = field(default_factory=list, repr=False, compare=False)
    _dirty: Dict[int, Tuple[int, _Node]] = field(default_factory=dict, repr=False, compare=False)
    _n_nodes: int = field(default=0, repr=False, compare=False)
    _passes: int = field(default=0, repr=False, compare=False)
    _full_next: bool = field(default=True, repr=False, compare=False)
    _force_at: int = field(default=_SOFT_NODES, repr=False, compare=False)
    _walk: Dict[Tuple[Any, ...], Tuple[Tuple[_Node, ...], str]] = field(
        default_factory=dict, repr=False, compare=False)

    # ----------------------------------------------------------- templating
    def template_path(self, host: str, method: str, path: str, w: float = 1.0
                      ) -> Tuple[str, List[Tuple[str, str]]]:
        """Learn-and-apply: (template string incl. '?params', masked (mask, value) pairs).
        Updates node counts by w. O(depth)."""
        tpl, masked = self._template(_norm_key(host, method), path, w)
        return tpl, list(masked)

    def _template(self, key: Tuple[str, str], path: Any, w: Any
                  ) -> Tuple[str, List[Tuple[str, str]]]:
        """template_path on a normalised (host, method) key; `masked` may be shared
        (memoised no-mask path), callers copy it before handing it out."""
        segs, params, masked = _mask_path(path)
        if w.__class__ is not float:
            try:
                w = float(w)
            except (TypeError, ValueError):
                w = 0.0
        learn = 0.0 < w < math.inf
        wk = (key, segs, params)
        hit = self._walk.get(wk)
        if hit is not None:                  # every node on the path is established
            if learn:
                for nd in hit[0]:
                    nd.count += w
            return hit[1], masked
        node: Optional[_Node] = self.trees.get(key)
        if node is None and learn and len(self.trees) < _MAX_TREES:
            node = self.trees[key] = _Node()
            self._n_nodes += 1
        if node is not None and learn:
            node.count += w
        path_nodes = [node]
        settled = node is not None           # memoisable: no provisional literal on the walk
        if len(segs) > MAX_DEPTH:
            segs = segs[:MAX_DEPTH] + (VAR_TOKEN,)              # tail below MAX_DEPTH
        out: List[str] = []
        for i, seg in enumerate(segs):
            if node is None:
                out.append(seg if seg in _STABLE else VAR_TOKEN)
                continue
            k = VAR_TOKEN if node.is_var else seg
            child = node.children.get(k)
            if child is None:
                if not learn:
                    node = None
                    settled = False
                    out.append(k if k in _STABLE else VAR_TOKEN)
                    continue
                ch = node.children
                if k not in _STABLE and (len(ch) >= _MAX_CHILDREN
                                         or self._n_nodes >= _MAX_NODES):
                    k = VAR_TOKEN
                    child = ch.get(k)
                if child is None:
                    child = ch[k] = _Node()
                    self._n_nodes += 1
                    self._dirty[id(node)] = (i, node)
            if learn:
                child.count += w
            # a literal is provisional ({var}) until it has more than MERGE_MAX_COUNT
            if child.count > MERGE_MAX_COUNT or k in _STABLE:
                out.append(k)
            else:
                out.append(VAR_TOKEN)
                settled = False
            path_nodes.append(child)
            node = child
        tpl = "/" + "/".join(out)
        if params:
            tpl += "?" + "&".join(params)
        if settled:
            walk = self._walk
            if len(walk) >= _WALK_MAX:
                walk.clear()
            walk[wk] = (tuple(path_nodes), tpl)
        return tpl, masked

    def apply_path(self, host: str, method: str, path: str) -> str:
        """Read-only route template '{METHOD} {host} {template}' (no status
        class) for the progressive core's http.route (docs/lib3/progressive.md
        §2): the same walk as template_path with weight 0, so no count,
        node or vocabulary changes; unknown segments render as {var}."""
        key = _norm_key(host, method)
        tpl, _ = self._template(key, path, 0.0)
        return f"{key[1]} {key[0]} {tpl}"

    def http_token(self, method: str, host: str, path: str, status: Optional[int],
                   w: float = 1.0) -> Tuple[str, List[Tuple[str, str]]]:
        """(token in the HTTP format, masked values)."""
        key = _norm_key(host, method)
        tpl, masked = self._template(key, path, w)
        return f"{key[1]} {key[0]} {tpl}|{status_class(status)}", list(masked)

    # ----------------------------------------------------------- vocabulary
    def intern(self, token: str, w: float = 1.0) -> int:
        """Space-Saving count of token; returns its id (0 if it maps to RARE_TOKEN)."""
        if not token or token[:5] == "{rare":
            return 0
        try:
            w = float(w)
        except (TypeError, ValueError):
            w = 0.0
        ok = 0.0 < w < math.inf
        e = self.vocab.get(token)
        if e is not None:
            if ok:
                e[1] += w
            return int(e[0])
        if not ok:
            return 0
        c_min = 0.0
        while len(self.vocab) >= VOCAB_CAP:
            c_min = self._evict_min()
        tid = int(self.next_id)
        self.next_id = tid + 1
        c = c_min + w
        self.vocab[token] = [tid, c, c_min]
        self.id_to_token[tid] = token
        heapq.heappush(self._heap, (c, tid, token))
        return tid

    def _rebuild_heap(self) -> None:
        self._heap = [(float(e[1]), int(e[0]), t) for t, e in self.vocab.items()]
        heapq.heapify(self._heap)

    def _evict_min(self) -> float:
        """Remove the argmin of (count, id); returns its count."""
        vocab = self.vocab
        h = self._heap
        if len(h) < len(vocab):
            self._rebuild_heap()
            h = self._heap
        while h:
            c, tid, tok = heapq.heappop(h)
            e = vocab.get(tok)
            if e is None or int(e[0]) != tid:
                continue                               # entry of an evicted token
            cur = float(e[1])
            if cur > c:
                heapq.heappush(h, (cur, tid, tok))     # stale: counted since pushed
                continue
            del vocab[tok]
            self.id_to_token.pop(tid, None)
            return cur
        # heap exhausted but vocab non-empty (external edits): rebuild and retry once
        if vocab:
            self._rebuild_heap()
            return self._evict_min()
        return 0.0

    def token_of(self, token_id: int) -> str:
        try:
            tid = int(token_id)
        except (TypeError, ValueError, OverflowError):
            return RARE_TOKEN
        return self.id_to_token.get(tid, RARE_TOKEN) if tid else RARE_TOKEN

    # ---------------------------------------------------------- maintenance
    def maintain(self) -> None:
        """Lazy maintenance pass: {var} collapse and sibling merge (call once per tick)."""
        self._passes += 1
        self._walk.clear()                   # structure may change below
        force = self._n_nodes >= self._force_at
        if force or self._full_next or self._passes % _FULL_PASS_EVERY == 0:
            self._dirty.clear()
            total = 0
            for root in self.trees.values():
                total += _maintain_node(root, True, force, [0])
            self._n_nodes = total
            self._full_next = False
            # after a forced pass only established nodes remain: do not force again
            # until another (_MAX_NODES - _SOFT_NODES) / 2 nodes have been created
            self._force_at = (max(_SOFT_NODES, total + (_MAX_NODES - _SOFT_NODES) // 2)
                              if force else _SOFT_NODES)
        elif self._dirty:
            items = sorted(self._dirty.values(), key=lambda t: t[0])
            self._dirty.clear()
            gone = [0]
            for _, node in items:
                _maintain_node(node, False, False, gone)
            self._n_nodes = max(0, self._n_nodes - gone[0])
        if len(self.trees) > _MAX_TREES // 2:
            drop = [k for k, r in self.trees.items() if r.count <= MERGE_MAX_COUNT]
            for k in drop:
                del self.trees[k]
            if drop:
                self._full_next = True

    # -------------------------------------------------------- serialisation
    def to_dict(self) -> Dict[str, Any]:
        """JSON-safe snapshot: trees as [host, method, [count, is_var, {seg: node}]],
        vocab as [token, id, count, error]; id_to_token is derived from vocab."""
        return {
            "fmt": _FMT,
            "next_id": int(self.next_id),
            "trees": [[h, m, _enc_node(r)] for (h, m), r in self.trees.items()],
            "vocab": [[t, int(e[0]), float(e[1]), float(e[2])] for t, e in self.vocab.items()],
        }

    @classmethod
    def from_dict(cls, d: Optional[Dict[str, Any]]) -> "Templater":
        """Inverse of to_dict. None / {} -> a fresh Templater; a Templater passes
        through unchanged; malformed rows are skipped."""
        if isinstance(d, Templater):
            return d
        t = cls()
        if not d:
            return t
        n_nodes = 0
        for row in d.get("trees") or ():
            try:
                h, m, enc = row
            except (TypeError, ValueError):
                continue
            node, n = _dec_node(enc)
            t.trees[(str(h), str(m))] = node
            n_nodes += n
        max_id = 0
        for row in d.get("vocab") or ():
            try:
                tok, tid, c, err = row
                tid, c, err = int(tid), float(c), float(err)
            except (TypeError, ValueError):
                continue
            if not tok or tid <= 0 or not math.isfinite(c) or c < 0.0:
                continue
            err = err if math.isfinite(err) and err >= 0.0 else 0.0
            t.vocab[str(tok)] = [tid, c, err]
            t.id_to_token[tid] = str(tok)
            max_id = max(max_id, tid)
        try:
            nid = int(d.get("next_id", 1))
        except (TypeError, ValueError):
            nid = 1
        t.next_id = max(nid, max_id + 1, 1)
        t._n_nodes = n_nodes
        t._full_next = True
        t._rebuild_heap()
        return t
