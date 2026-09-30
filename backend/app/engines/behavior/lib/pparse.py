"""Payload parsing, value retention policy, who resolution and session keys
for the progressive core's event builder (docs/lib3/progressive.md §5.1, §6.2.1).

STATUS: implemented (W-P1 support for P00). Pure functions.

parse_l7(l7, ...) -> {attr: value}
    body: form (x-www-form-urlencoded or sniffed k=v&k=v), JSON (first 64
    leaves, depth <= 4), multipart (part names + sizes; file parts give
    body.kv.<name>.len / .ctype), XML (element path set), otherwise a
    stateless Drain-style template of the printable prefix (body.tpl).
    At most K_BODY keys per event (the rest counted in body.keys_extra); keys
    lower-cased and templated (items[3].name -> items[].name).
    query: q.keys / q.kv.<key>; headers: hdr.<name> (cookie / authorization
    are never kept in clear, see the value policy); sess.key from a session
    cookie value: HMAC-SHA256(deployment key)[:12].
ValuePolicy.apply(name, value) -> (stored value, mode)
    clear | hmac ('h:' + 12 hex) | shape (phier.shape; long values also keep
    '<name>.len'). Defaults (§5.1.3): secret globs -> shape; random-looking
    or high-entropy long strings -> shape; strings longer than v_len -> shape;
    other payload values clear. Applies to payload namespaces (body, q, hdr,
    meta, sess) only; structural attributes (routes, stacks, JA3, ...) are
    never rewritten.
resolve_who(src, headers, trusted, header_names) -> (client ip, transport src | None)
"""
from __future__ import annotations

import fnmatch
import hashlib
import hmac as _hmac
import ipaddress
import json
import math
import re
from typing import Any, Dict, Iterable, List, Mapping, Optional, Sequence, Tuple
from urllib.parse import parse_qsl, unquote_plus

from . import pevent as EV
from .phier import Shaped, shape
from .template import _entropy_bits, _looks_random, mask_segment

K_BODY = 64
BODY_CAP = 4096
PAYLOAD_NS = ("body.", "q.", "hdr.", "meta.")          # sess.key is already an HMAC
_FORM_RE = re.compile(r"^[^=&\s]{1,128}=[^&]*(?:&[^=&\s]{1,128}=[^&]*)*$")
_BOUNDARY_RE = re.compile(r'boundary="?([^";]+)"?', re.I)
_CD_NAME_RE = re.compile(r'name="([^"]*)"', re.I)
_CD_FILE_RE = re.compile(r'filename="([^"]*)"', re.I)
_XML_TAG_RE = re.compile(r"<([A-Za-z_][\w\-.:]*)[^>]*?(/?)>|</([A-Za-z_][\w\-.:]*)>")
_WS_RE = re.compile(r"\s+")
NEVER_CLEAR = ("hdr.cookie", "hdr.authorization", "hdr.proxy-authorization", "hdr.set-cookie")


# ================================================================== policy
def hmac12(key: str, value: str) -> str:
    return _hmac.new(key.encode("utf-8"), str(value).encode("utf-8", "surrogatepass"),
                     hashlib.sha256).hexdigest()[:12]


class ValuePolicy:
    """Value retention policy (§5.1.3), built from pconfig()['value_policy']."""

    def __init__(self, cfg: Optional[Mapping[str, Any]] = None) -> None:
        c = dict(cfg or {})
        self.rules: List[Tuple[str, str]] = [(str(g), str(m)) for g, m in (c.get("rules") or [])]
        self.secret_globs = tuple(c.get("secret_globs") or ())
        self.v_len = int(c.get("v_len", 64))
        self.random_bits = float(c.get("random_bits", 3.5))
        self.random_len = int(c.get("random_len", 16))
        self.key = str(c.get("hmac_key", "appmon-default-deployment-key"))
        self._memo: Dict[str, str] = {}

    def mode_for(self, name: str) -> str:
        """Configured mode for an attribute name (before value tests)."""
        hit = self._memo.get(name)
        if hit is not None:
            return hit
        mode = ""
        for g, m in self.rules:
            if fnmatch.fnmatchcase(name, g):
                mode = m
                break
        if not mode and (name in NEVER_CLEAR
                         or any(fnmatch.fnmatchcase(name, g) for g in self.secret_globs)):
            mode = "shape"
        if not mode:
            mode = "clear"
        if len(self._memo) > 8192:
            self._memo.clear()
        self._memo[name] = mode
        return mode

    def randomish(self, v: str) -> bool:
        """A secret / nonce-like value: one whitespace-free token that
        lib/template masks as {hex} / {tok} / {uuid} / {id}, or that is
        identifier-fragmented (_looks_random) with >= random_bits bits/char at
        >= random_len characters. Measured: the literal §5.1.3 rule (entropy
        >= 3.5 bits/char at length >= 16 alone) shaped ordinary header values
        ('application/x-www-form-urlencoded', user agents, referers), whose
        character entropy is 3.5-4.2 bits/char."""
        if any(c.isspace() for c in v):
            return False
        m, _ = mask_segment(v)
        if m in ("{hex}", "{tok}", "{uuid}", "{id}"):
            return True
        return len(v) >= self.random_len and _entropy_bits(v) >= self.random_bits \
            and _looks_random(v)

    def apply(self, name: str, value: Any) -> Tuple[Any, str]:
        """(stored value, mode). Non-payload names and non-string values pass."""
        if not name.startswith(PAYLOAD_NS):
            return value, "clear"
        mode = self.mode_for(name)
        if isinstance(value, frozenset):
            if mode == "shape":
                return frozenset(Shaped(shape(str(x))) for x in value), "shape"
            if mode == "hmac":
                return frozenset("h:" + hmac12(self.key, str(x)) for x in value), "hmac"
            return value, "clear"
        if not isinstance(value, str):
            return value, "clear"
        if mode == "clear":
            n = len(value)
            if n > self.v_len:
                mode = "shape"
            elif n >= 12 and self.randomish(value):
                mode = "shape"
        if mode == "shape":
            return Shaped(shape(value)), "shape"
        if mode == "hmac":
            return "h:" + hmac12(self.key, value), "hmac"
        return value, "clear"

    def apply_all(self, attrs: Dict[str, Any], modes: Optional[Dict[str, str]] = None) -> Dict[str, Any]:
        """Apply the policy to every attribute in place; long clear values
        replaced by shape also keep '<name>.len'. modes collects non-clear modes."""
        extra: Dict[str, Any] = {}
        for nm in list(attrs.keys()):
            v = attrs[nm]
            nv, mode = self.apply(nm, v)
            if mode != "clear":
                attrs[nm] = nv
                if modes is not None:
                    modes[nm] = mode
                if isinstance(v, str) and mode == "shape" and not nm.endswith(".len"):
                    extra[nm + ".len"] = len(v)
        attrs.update(extra)
        return attrs


# ================================================================== parsing
def _key(k: Any) -> str:
    return EV.template_key(str(k).strip().lower())


def _decode_body(body: Any, cap: int) -> str:
    if isinstance(body, (bytes, bytearray)):
        return bytes(body[:cap]).decode("utf-8", "replace")
    return str(body)[:cap]


def sniff_format(body: str, ctype: str = "", hint: str = "") -> str:
    h = (hint or "").lower()
    if h in ("form", "json", "multipart", "xml", "text"):
        return h
    ct = (ctype or "").lower()
    if "multipart/form-data" in ct:
        return "multipart"
    if "json" in ct:
        return "json"
    if "x-www-form-urlencoded" in ct:
        return "form"
    if "xml" in ct:
        return "xml"
    s = body.lstrip()
    if s[:1] in "{[":
        try:
            json.loads(body)
            return "json"
        except ValueError:
            pass
    if s[:1] == "<":
        return "xml"
    if "=" in s and _FORM_RE.match(s.strip()):
        return "form"
    return "text"


def _parse_form(s: str, prefix: str, out: Dict[str, Any], k_max: int) -> Tuple[frozenset, int]:
    keys: List[str] = []
    extra = 0
    for k, v in parse_qsl(s, keep_blank_values=True):
        kk = _key(k)
        if not kk:
            continue
        if kk in keys:
            continue
        if len(keys) >= k_max:
            extra += 1
            continue
        keys.append(kk)
        out[EV.attr_name(prefix, "kv", kk)] = v
    return frozenset(keys), extra


def _parse_json(s: str, prefix: str, out: Dict[str, Any], k_max: int) -> Tuple[frozenset, int]:
    try:
        obj = json.loads(s)
    except ValueError:
        return frozenset(), 0
    flat: Dict[str, Any] = {}
    root = prefix + ".kv"
    EV.flatten(root, obj, flat, max_leaves=k_max)
    keys = frozenset(n[len(root) + 1:] for n in flat if not n.endswith("[].n"))
    out.update(flat)
    total = _count_leaves(obj)
    return keys, max(0, total - len(keys))


def _count_leaves(obj: Any, depth: int = 0) -> int:
    if depth > 6:
        return 1
    if isinstance(obj, Mapping):
        return sum(_count_leaves(v, depth + 1) for v in obj.values()) or 1
    if isinstance(obj, (list, tuple)):
        return 1
    return 1


def _parse_multipart(s: str, ctype: str, prefix: str, out: Dict[str, Any], k_max: int
                     ) -> Tuple[frozenset, int]:
    m = _BOUNDARY_RE.search(ctype or "")
    if not m:
        first = s.split("\r\n", 1)[0].split("\n", 1)[0]
        if not first.startswith("--"):
            return frozenset(), 0
        boundary = first[2:].strip()
    else:
        boundary = m.group(1)
    keys: List[str] = []
    extra = 0
    for part in s.split("--" + boundary):
        if not part.strip() or part.strip() == "--":
            continue
        head, _, body = part.partition("\r\n\r\n")
        if not _:
            head, _, body = part.partition("\n\n")
        nm = _CD_NAME_RE.search(head)
        if not nm:
            continue
        kk = _key(nm.group(1))
        if len(keys) >= k_max:
            extra += 1
            continue
        keys.append(kk)
        body = body.rstrip("\r\n")
        if _CD_FILE_RE.search(head):
            out[EV.attr_name(prefix, "kv", kk, "len")] = len(body)
            ct = re.search(r"content-type:\s*([^\r\n;]+)", head, re.I)
            if ct:
                out[EV.attr_name(prefix, "kv", kk, "ctype")] = ct.group(1).strip().lower()
        else:
            out[EV.attr_name(prefix, "kv", kk)] = body
    return frozenset(keys), extra


def _parse_xml(s: str) -> frozenset:
    stack: List[str] = []
    paths = set()
    for m in _XML_TAG_RE.finditer(s[:BODY_CAP]):
        open_tag, selfclose, close_tag = m.group(1), m.group(2), m.group(3)
        if open_tag:
            stack.append(open_tag.lower())
            paths.add("/".join(stack))
            if selfclose:
                stack.pop()
        elif close_tag and stack:
            if stack[-1] == close_tag.lower():
                stack.pop()
        if len(paths) >= K_BODY:
            break
    return frozenset(paths)


def text_template(s: str, max_tokens: int = 32) -> str:
    """Stateless Drain-style template of the printable prefix: whitespace
    tokens masked by lib/template.mask_segment ({num}, {id}, {hex}, ...)."""
    toks = _WS_RE.split(s.strip())[:max_tokens]
    return " ".join(mask_segment(t)[0] if t else t for t in toks)


def parse_body(body: Any, ctype: str = "", hint: str = "", prefix: str = "body",
               k_max: int = K_BODY, cap: int = BODY_CAP) -> Dict[str, Any]:
    s = _decode_body(body, cap)
    out: Dict[str, Any] = {}
    fmt = sniff_format(s, ctype, hint)
    out[prefix + ".fmt"] = fmt
    keys: frozenset = frozenset()
    extra = 0
    if fmt == "form":
        keys, extra = _parse_form(s, prefix, out, k_max)
    elif fmt == "json":
        keys, extra = _parse_json(s, prefix, out, k_max)
    elif fmt == "multipart":
        keys, extra = _parse_multipart(s, ctype, prefix, out, k_max)
    elif fmt == "xml":
        out[prefix + ".paths"] = _parse_xml(s)
    else:
        printable = "".join(ch if ch.isprintable() else " " for ch in s[:256])
        if printable.strip():
            out[prefix + ".tpl"] = text_template(printable)
    if keys:
        out[prefix + ".keys"] = keys
    if extra:
        out[prefix + ".keys_extra"] = extra
    return out


def parse_l7(l7: Mapping[str, Any], session_cookies: Sequence[str] = (),
             hmac_key: str = "appmon-default-deployment-key", k_max: int = K_BODY,
             cap: int = BODY_CAP, parse_body_enabled: bool = True) -> Tuple[Dict[str, Any], bool]:
    """Attributes of an extra['l7'] view (§5.1.1) and whether the body was truncated."""
    out: Dict[str, Any] = {}
    headers = {str(k).lower(): v for k, v in (l7.get("headers") or {}).items()}
    ctype = str(headers.get("content-type", "") or "")
    body = l7.get("body")
    trunc = bool(l7.get("body_trunc", False))
    if l7.get("body_len") is not None:
        try:
            out["body.len"] = int(l7["body_len"])
        except (TypeError, ValueError):
            pass
    if body is not None and parse_body_enabled:
        if "body.len" not in out:
            out["body.len"] = len(body)
        out.update(parse_body(body, ctype, str(l7.get("body_type") or ""), "body", k_max, cap))
    q = l7.get("query")
    if q:
        keys, extra = _parse_form(str(q), "q", out, k_max)
        if keys:
            out["q.keys"] = keys
        if extra:
            out["q.keys_extra"] = extra
    for hk, hv in headers.items():
        if hk == "cookie":
            continue
        if isinstance(hv, (str, int, float)):
            out[EV.attr_name("hdr", hk)] = hv if not isinstance(hv, str) else hv[:BODY_CAP]
    if l7.get("resp_len") is not None:
        try:
            out["net.resp_len"] = int(l7["resp_len"])
        except (TypeError, ValueError):
            pass
    sess = l7.get("sess")
    if sess is None and session_cookies and headers.get("cookie"):
        sess = session_from_cookie(str(headers["cookie"]), session_cookies)
    if sess:
        out["sess.key"] = hmac12(hmac_key, str(sess))
    return out, trunc


def session_from_cookie(cookie: str, names: Sequence[str]) -> Optional[str]:
    wanted = {n.lower() for n in names}
    for part in cookie.split(";"):
        k, _, v = part.strip().partition("=")
        if k.strip().lower() in wanted and v:
            return v.strip()
    return None


def parse_structured_text(name: str, value: str, parse_as: str, k_max: int = K_BODY) -> Dict[str, Any]:
    """Registry hint `parse_as` (§5.3): a text attribute found to be k=v or
    JSON structured emits '<name>.keys' and '<name>.kv.<key>'."""
    out: Dict[str, Any] = {}
    if parse_as == "form":
        keys, _ = _parse_form(value.replace(";", "&"), name, out, k_max)
    elif parse_as == "json":
        keys, _ = _parse_json(value, name, out, k_max)
    else:
        return out
    if keys:
        out[name + ".keys"] = keys
    return out


# ============================================================== who / nets
class TrustedNets:
    def __init__(self, cidrs: Iterable[str] = ()) -> None:
        self.nets = []
        for c in cidrs or ():
            try:
                self.nets.append(ipaddress.ip_network(str(c), strict=False))
            except ValueError:
                continue

    def __bool__(self) -> bool:
        return bool(self.nets)

    def contains(self, ip: str) -> bool:
        try:
            a = ipaddress.ip_address(ip.strip().strip("[]"))
        except ValueError:
            return False
        return any(a.version == n.version and a in n for n in self.nets)


def _forwarded_for(v: str) -> List[str]:
    out = []
    for part in v.split(","):
        for kv in part.split(";"):
            k, _, x = kv.strip().partition("=")
            if k.lower() == "for" and x:
                x = x.strip('"').strip()
                if x.startswith("["):
                    x = x[1:].split("]", 1)[0]
                elif x.count(":") == 1:
                    x = x.split(":", 1)[0]
                out.append(x)
    return out


def resolve_who(src: str, headers: Optional[Mapping[str, Any]], trusted: TrustedNets,
                header_names: Sequence[str] = ("x-forwarded-for", "forwarded", "x-real-ip")
                ) -> Tuple[str, Optional[str]]:
    """(client ip, transport source when different) (§5.1.4): for a request
    whose transport source is a trusted proxy, the right-most address of the
    forwarding chain that is not trusted; otherwise the transport source."""
    if not trusted or not headers or not trusted.contains(src):
        return src, None
    h = {str(k).lower(): v for k, v in headers.items()}
    for name in header_names:
        v = h.get(name.lower())
        if not v:
            continue
        v = str(v)
        if name.lower() == "forwarded":
            chain = _forwarded_for(v)
        else:
            chain = [x.strip() for x in v.split(",") if x.strip()]
        for ip in reversed(chain):
            if not trusted.contains(ip):
                try:
                    ipaddress.ip_address(ip)
                except ValueError:
                    continue
                return ip, src
    return src, None
