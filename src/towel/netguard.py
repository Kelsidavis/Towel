"""SSRF guard for outbound, model-controlled URLs.

A rogue model with both a secret-read tool and an HTTP/webhook tool can
chain them into exfiltration, and can also point requests at internal
infrastructure the host can reach but the model never should — cloud
metadata endpoints, link-local services, private LAN hosts, loopback.

``check_url(url)`` returns a refusal reason string when a URL should be
blocked, or ``None`` when it is allowed. It is intentionally fail-closed
for the cases that matter: non-HTTP schemes, missing hosts, and any host
that resolves to a loopback / private / link-local / reserved address —
including the cloud metadata IP ``169.254.169.254``.

This blocks the *destination*, complementing the gating layer (which
governs *whether* a tool may run at all) and the audit log (which records
that it did). DNS rebinding is partially mitigated by resolving here, but
a caller that re-resolves at connect time can still be raced; for full
safety the resolved IP should be pinned. Documented as a known gap.
"""

from __future__ import annotations

import ipaddress
import logging
import socket
from urllib.parse import urlparse

log = logging.getLogger("towel.netguard")

_ALLOWED_SCHEMES = frozenset({"http", "https"})


def _ip_is_blocked(ip: str) -> bool:
    try:
        addr = ipaddress.ip_address(ip)
    except ValueError:
        return True  # unparseable → fail closed
    return (
        addr.is_loopback
        or addr.is_private
        or addr.is_link_local      # 169.254.0.0/16 incl. cloud metadata
        or addr.is_reserved
        or addr.is_multicast
        or addr.is_unspecified
    )


def check_url(url: str) -> str | None:
    """Return why an outbound URL should be refused, or None to allow it.

    Blocks non-HTTP(S) schemes, hostless URLs, and any host that resolves
    to a loopback/private/link-local/reserved address.
    """
    if not url or not isinstance(url, str):
        return "refused: empty or non-string URL."

    parsed = urlparse(url.strip())
    scheme = (parsed.scheme or "").lower()
    if scheme not in _ALLOWED_SCHEMES:
        return (
            f"refused: scheme {scheme or '(none)'!r} is not allowed; only "
            "http/https outbound requests are permitted."
        )

    host = parsed.hostname
    if not host:
        return "refused: URL has no host."

    # Resolve every address the host maps to; block if ANY is internal
    # (defeats a name that returns one public and one private record).
    try:
        infos = socket.getaddrinfo(host, parsed.port or None, proto=socket.IPPROTO_TCP)
    except socket.gaierror as exc:
        return f"refused: could not resolve host {host!r} ({exc})."

    for info in infos:
        ip = info[4][0]
        if _ip_is_blocked(ip):
            return (
                f"refused: host {host!r} resolves to internal address {ip} "
                "(loopback/private/link-local/metadata). Outbound requests "
                "to internal infrastructure are blocked."
            )

    return None
