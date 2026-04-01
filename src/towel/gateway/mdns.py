"""mDNS service advertisement and discovery for Towel.

The coordinator advertises ``_towel._tcp.local.`` so workers on the LAN
can find it without any configuration.  Workers browse for the service
and connect to the first coordinator they see.

Uses the ``zeroconf`` library which talks multicast DNS natively — no
Avahi daemon or system service required.
"""

from __future__ import annotations

import asyncio
import logging
import socket
from dataclasses import dataclass, field

from zeroconf import IPVersion, ServiceBrowser, ServiceStateChange, Zeroconf
from zeroconf.asyncio import AsyncServiceInfo, AsyncZeroconf

log = logging.getLogger("towel.gateway.mdns")

SERVICE_TYPE = "_towel._tcp.local."
DEFAULT_TIMEOUT_S = 30.0


def _local_ip() -> str:
    """Best-effort LAN IP address."""
    try:
        with socket.socket(socket.AF_INET, socket.SOCK_DGRAM) as s:
            s.connect(("8.8.8.8", 80))
            return s.getsockname()[0]
    except OSError:
        return "127.0.0.1"


class TowelServiceAdvertiser:
    """Advertise this coordinator via mDNS so workers can find it."""

    def __init__(self, port: int, hostname: str | None = None) -> None:
        self.port = port
        self.hostname = hostname or socket.gethostname()
        self._zc: AsyncZeroconf | None = None
        self._info: AsyncServiceInfo | None = None

    async def start(self) -> None:
        """Register the service on the network."""
        ip = _local_ip()
        self._info = AsyncServiceInfo(
            SERVICE_TYPE,
            f"towel-controller-{self.hostname}.{SERVICE_TYPE}",
            addresses=[socket.inet_aton(ip)],
            port=self.port,
            properties={
                "version": "1",
                "hostname": self.hostname,
            },
        )
        self._zc = AsyncZeroconf(ip_version=IPVersion.V4Only)
        await self._zc.async_register_service(self._info)
        log.info("mDNS: advertising %s on %s:%d", SERVICE_TYPE, ip, self.port)

    async def stop(self) -> None:
        """Unregister the service."""
        if self._zc and self._info:
            await self._zc.async_unregister_service(self._info)
            await self._zc.async_close()
            self._zc = None
            log.info("mDNS: stopped advertising")


@dataclass
class DiscoveredController:
    """A controller found via mDNS."""

    host: str
    port: int
    hostname: str
    ws_url: str


@dataclass
class _BrowseState:
    """Mutable state shared between the browse callback and the waiter."""

    result: DiscoveredController | None = None
    event: asyncio.Event = field(default_factory=asyncio.Event)


async def discover_controller(timeout: float = DEFAULT_TIMEOUT_S) -> DiscoveredController | None:
    """Browse mDNS for a Towel coordinator.  Returns the first one found."""
    state = _BrowseState()
    zc = Zeroconf(ip_version=IPVersion.V4Only)

    def on_change(
        zeroconf: Zeroconf,
        service_type: str,
        name: str,
        state_change: ServiceStateChange,
    ) -> None:
        if state_change != ServiceStateChange.Added:
            return
        info = zeroconf.get_service_info(service_type, name)
        if not info:
            return
        addresses = info.parsed_addresses(IPVersion.V4Only)
        if not addresses:
            return
        host = addresses[0]
        port = info.port
        props = {
            k.decode() if isinstance(k, bytes) else k: v.decode() if isinstance(v, bytes) else v
            for k, v in info.properties.items()
        }
        ctrl = DiscoveredController(
            host=host,
            port=port,
            hostname=props.get("hostname", "unknown"),
            ws_url=f"ws://{host}:{port}",
        )
        log.info("mDNS: found controller %s at %s", ctrl.hostname, ctrl.ws_url)
        state.result = ctrl
        state.event.set()

    browser = ServiceBrowser(zc, SERVICE_TYPE, handlers=[on_change])

    try:
        await asyncio.wait_for(state.event.wait(), timeout=timeout)
    except TimeoutError:
        log.warning("mDNS: no controller found within %.0fs", timeout)
    finally:
        browser.cancel()
        zc.close()

    return state.result
