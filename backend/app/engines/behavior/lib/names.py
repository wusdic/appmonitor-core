"""Host-name helpers: registrable domain (eTLD+1) and internal/external tests.

SNI and DNS sets are aggregated at eTLD+1 so that CDN shards and random
sub-labels of one organisation count as one destination (R1, B01 sketch,
B08 novelty). A full Public Suffix List is overkill for the passive-sensor
setting; the subset below covers the multi-label suffixes seen in the target
deployments (mainland China, UK, AU, JP, ...). Unknown two-label suffixes
simply fall back to "last two labels", which is the right answer for every
generic TLD.
"""
from __future__ import annotations

import ipaddress
from typing import Iterable, List, Optional, Tuple, Union

# Multi-label public suffixes (subset of the PSL).
PUBLIC_SUFFIXES_2L = frozenset({
    "com.cn", "gov.cn", "edu.cn", "org.cn", "net.cn", "ac.cn", "mil.cn",
    "co.uk", "org.uk", "ac.uk", "gov.uk", "net.uk", "ltd.uk", "plc.uk",
    "com.au", "net.au", "org.au", "edu.au", "gov.au",
    "co.jp", "ne.jp", "or.jp", "ac.jp", "go.jp",
    "com.hk", "org.hk", "edu.hk", "gov.hk", "com.tw", "org.tw", "edu.tw", "gov.tw",
    "co.kr", "or.kr", "com.sg", "edu.sg", "gov.sg", "co.in", "com.br", "com.mx",
    "co.nz", "co.za", "com.tr", "com.ru",
})

# TLDs that are never on the public internet.
INTERNAL_TLDS = frozenset({"local", "lan", "internal", "corp", "home", "intranet",
                           "localdomain", "localhost", "test", "invalid", "arpa"})


def normalize_host(host: str) -> str:
    """Lower-case, strip scheme/port/trailing dot/brackets. '' for empty input."""
    h = (host or "").strip().lower()
    if "://" in h:
        h = h.split("://", 1)[1]
    h = h.split("/", 1)[0]
    if h.startswith("["):                     # [v6]:port
        return h[1:].split("]", 1)[0]
    if h.count(":") == 1:                     # host:port (not bare IPv6)
        h = h.split(":", 1)[0]
    return h.rstrip(".")


def _ip(h: str) -> Optional[Union[ipaddress.IPv4Address, ipaddress.IPv6Address]]:
    try:
        return ipaddress.ip_address(h)
    except ValueError:
        return None


def etld1(host: str) -> str:
    """Registrable domain of `host`.

    'a.b.example.com.cn' -> 'example.com.cn'; 'x.cdn.example.com' ->
    'example.com'; IP literals and single-label names are returned
    normalised and unchanged; a bare public suffix is returned as is.
    """
    h = normalize_host(host)
    if not h or _ip(h) is not None:
        return h
    labels = [lb for lb in h.split(".") if lb]
    if len(labels) <= 2:
        return ".".join(labels)
    if ".".join(labels[-2:]) in PUBLIC_SUFFIXES_2L:
        return ".".join(labels[-3:])
    return ".".join(labels[-2:])


def split_host(host: str) -> Tuple[List[str], str]:
    """(labels left of the registrable domain, eTLD+1). Used by DNS templating."""
    h = normalize_host(host)
    reg = etld1(h)
    if not reg or h == reg or _ip(h) is not None:
        return [], reg
    left = h[: -len(reg)].rstrip(".")
    return ([lb for lb in left.split(".") if lb] if left else []), reg


def is_external(host: str, org_domains: Optional[Iterable[str]] = None) -> bool:
    """True if `host` is outside the organisation.

    Internal: IP literals that are private/loopback/link-local, single-label
    names, internal TLDs (.local, .corp, ...), and any host equal to or under
    one of `org_domains` (compared at eTLD+1 as well as by suffix).
    Empty host -> False (unknown is not evidence of external).
    """
    h = normalize_host(host)
    if not h:
        return False
    ip = _ip(h)
    if ip is not None:
        return not (ip.is_private or ip.is_loopback or ip.is_link_local
                    or ip.is_reserved or ip.is_unspecified)
    labels = h.split(".")
    if len(labels) == 1 or labels[-1] in INTERNAL_TLDS:
        return False
    reg = etld1(h)
    for d in org_domains or ():
        d = normalize_host(d)
        if not d:
            continue
        if h == d or h.endswith("." + d) or reg == etld1(d):
            return False
    return True
