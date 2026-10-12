"""Find the Olares Router of this machine.

Every Olares runs exactly one Router (the AI gateway). An installed app
reaches it through the platform edge, which stamps the caller identity, so
no credential is needed. The only machine-specific piece is the base domain
of the Olares; it is read from the platform-managed mesh-in hosts file and,
as a fallback, from the Host header of the request that reached us.
"""

import hashlib
import logging
import os
import re

import aiohttp

log = logging.getLogger("aimrealtime.router")

HOSTS_FILE = "/var/run/olares/mesh-in-shared-hosts/shared-hosts.txt"
GATEWAY_HOST = os.environ.get("OLARES_GATEWAY_HOST", "app-gateway-data.os-gateway.svc")
ROUTER_APPID = hashlib.md5(b"router").hexdigest()[:8]

_LINE = re.compile(r"^([a-z0-9]+)\.([a-z0-9][a-z0-9.-]*[a-z])$", re.I)


def _domains_from_hosts_file() -> list[str]:
    """Base domains of this Olares, taken from the platform hosts file."""
    try:
        with open(HOSTS_FILE, encoding="utf-8") as fh:
            lines = fh.read().splitlines()
    except OSError:
        return []
    bases: list[str] = []
    for line in lines:
        m = _LINE.match(line.strip().lower())
        if not m:
            continue
        label, base = m.group(1), m.group(2)
        if label in ("router", ROUTER_APPID) and base not in bases:
            bases.append(base)
    return bases


def _domain_from_host_header(host: str) -> str:
    """Base domain behind our own entrance route: <route>.<base> -> <base>."""
    host = (host or "").split(":")[0].strip().lower()
    m = _LINE.match(host)
    if not m or m.group(2) in ("localhost",):
        return ""
    return m.group(2)


class RouterEndpoint:
    """Cached Router base URL for this machine, probed once and re-probed on
    transport failure. Returns (base_url, extra_headers)."""

    def __init__(self) -> None:
        self.override = os.environ.get("ROUTER_BASE_URL", "").strip().rstrip("/")
        self._resolved: tuple[str, dict[str, str]] | None = None

    @property
    def resolved(self) -> bool:
        return self._resolved is not None

    def candidates(self, request_host: str = "") -> list[tuple[str, dict[str, str]]]:
        if self.override:
            return [(self.override, {})]
        bases = _domains_from_hosts_file()
        seen = set(bases)
        extra = _domain_from_host_header(request_host)
        if extra and extra not in seen:
            bases.append(extra)
        out: list[tuple[str, dict[str, str]]] = []
        for base in bases:
            for label in ("router", ROUTER_APPID):
                out.append((f"http://{GATEWAY_HOST}/v1", {"Host": f"{label}.{base}"}))
            for label in ("router", ROUTER_APPID):
                out.append((f"https://{label}.{base}/v1", {}))
        return out

    async def get(self, request_host: str = "") -> tuple[str, dict[str, str]]:
        if self._resolved:
            return self._resolved
        for url, headers in self.candidates(request_host):
            if await self.probe(url, headers):
                self._resolved = (url, headers)
                via = headers.get("Host", "direct")
                log.info("router resolved: %s (host %s)", url, via)
                return self._resolved
        return "", {}

    def invalidate(self) -> None:
        self._resolved = None

    async def probe(self, url: str, headers: dict[str, str]) -> bool:
        timeout = aiohttp.ClientTimeout(total=8, connect=4)
        try:
            async with aiohttp.ClientSession(timeout=timeout) as s:
                async with s.get(f"{url}/models", headers=headers) as resp:
                    return resp.status == 200
        except (aiohttp.ClientError, TimeoutError, OSError):
            return False
