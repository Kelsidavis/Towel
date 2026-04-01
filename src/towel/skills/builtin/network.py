"""Network skill — DNS lookup, port checking, and connectivity tests."""

from __future__ import annotations

import asyncio
import socket
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class NetworkSkill(Skill):
    @property
    def name(self) -> str:
        return "network"

    @property
    def description(self) -> str:
        return "Network diagnostics: DNS lookup, port check, HTTP ping"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="dns_lookup",
                description="Resolve a hostname to IP addresses (A and AAAA records)",
                parameters={
                    "type": "object",
                    "properties": {
                        "hostname": {"type": "string", "description": "Hostname to resolve"},
                    },
                    "required": ["hostname"],
                },
            ),
            ToolDefinition(
                name="port_check",
                description="Check if a TCP port is open on a host",
                parameters={
                    "type": "object",
                    "properties": {
                        "host": {"type": "string", "description": "Hostname or IP"},
                        "port": {"type": "integer", "description": "Port number"},
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds (default: 3)",
                        },
                    },
                    "required": ["host", "port"],
                },
            ),
            ToolDefinition(
                name="http_ping",
                description="Check if a URL is reachable and measure response time",
                parameters={
                    "type": "object",
                    "properties": {
                        "url": {
                            "type": "string",
                            "description": "URL to ping (e.g., https://example.com)",
                        },
                        "timeout": {
                            "type": "number",
                            "description": "Timeout in seconds (default: 5)",
                        },
                    },
                    "required": ["url"],
                },
            ),
            ToolDefinition(
                name="whois_lookup",
                description="Get basic IP geolocation and ownership info",
                parameters={
                    "type": "object",
                    "properties": {
                        "target": {"type": "string", "description": "IP address or hostname"},
                    },
                    "required": ["target"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "dns_lookup":
                return await self._dns_lookup(arguments["hostname"])
            case "port_check":
                return await self._port_check(
                    arguments["host"],
                    arguments["port"],
                    arguments.get("timeout", 3),
                )
            case "http_ping":
                return await self._http_ping(
                    arguments["url"],
                    arguments.get("timeout", 5),
                )
            case "whois_lookup":
                return await self._whois(arguments["target"])
            case _:
                return f"Unknown tool: {tool_name}"

    async def _dns_lookup(self, hostname: str) -> str:
        loop = asyncio.get_event_loop()
        try:
            results = await loop.run_in_executor(
                None,
                lambda: socket.getaddrinfo(hostname, None, socket.AF_UNSPEC, socket.SOCK_STREAM),
            )
        except socket.gaierror as e:
            return f"DNS lookup failed for {hostname}: {e}"

        seen: set[str] = set()
        lines = [f"DNS records for {hostname}:"]
        for family, _, _, _, addr in results:
            ip = addr[0]
            if ip in seen:
                continue
            seen.add(ip)
            label = "AAAA" if family == socket.AF_INET6 else "A"
            lines.append(f"  {label}: {ip}")

        if len(seen) == 0:
            return f"No DNS records found for {hostname}"

        return "\n".join(lines)

    async def _port_check(self, host: str, port: int, timeout: float) -> str:
        loop = asyncio.get_event_loop()
        try:
            sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
            sock.settimeout(timeout)
            result = await loop.run_in_executor(None, lambda: sock.connect_ex((host, port)))
            sock.close()

            if result == 0:
                # Try to identify the service
                try:
                    service = socket.getservbyport(port)
                except OSError:
                    service = "unknown"
                return f"Port {port} on {host}: OPEN ({service})"
            else:
                return f"Port {port} on {host}: CLOSED/FILTERED"
        except TimeoutError:
            return f"Port {port} on {host}: TIMEOUT after {timeout}s"
        except socket.gaierror as e:
            return f"Cannot resolve {host}: {e}"
        except Exception as e:
            return f"Error checking {host}:{port}: {e}"

    async def _http_ping(self, url: str, timeout: float) -> str:
        import time

        try:
            import httpx
        except ImportError:
            return "httpx not installed — cannot ping URLs"

        try:
            start = time.perf_counter()
            async with httpx.AsyncClient(timeout=timeout, follow_redirects=True) as client:
                resp = await client.get(url, headers={"User-Agent": "Towel/1.0"})
            elapsed = (time.perf_counter() - start) * 1000  # ms

            lines = [
                f"HTTP ping {url}:",
                f"  Status: {resp.status_code} {resp.reason_phrase}",
                f"  Time:   {elapsed:.0f} ms",
                f"  Size:   {len(resp.content):,} bytes",
            ]

            # Show server header if present
            server = resp.headers.get("server")
            if server:
                lines.append(f"  Server: {server}")

            ct = resp.headers.get("content-type", "")
            if ct:
                lines.append(f"  Type:   {ct.split(';')[0]}")

            return "\n".join(lines)

        except httpx.TimeoutException:
            return f"HTTP ping {url}: TIMEOUT after {timeout}s"
        except httpx.ConnectError as e:
            return f"HTTP ping {url}: CONNECTION FAILED ({e})"
        except Exception as e:
            return f"HTTP ping {url}: ERROR ({e})"

    async def _whois(self, target: str) -> str:
        """Basic IP info via ip-api.com (free, no key needed)."""
        try:
            import httpx
        except ImportError:
            return "httpx not installed"

        # Resolve hostname to IP first if needed
        ip = target
        try:
            info = socket.getaddrinfo(target, None, socket.AF_INET, socket.SOCK_STREAM)
            if info:
                ip = info[0][4][0]
        except socket.gaierror:
            pass

        try:
            async with httpx.AsyncClient(timeout=5) as client:
                resp = await client.get(f"http://ip-api.com/json/{ip}")
                data = resp.json()

            if data.get("status") == "fail":
                return f"Lookup failed for {target}: {data.get('message', 'unknown')}"

            lines = [f"IP info for {target} ({ip}):"]
            for key, label in [
                ("org", "Org"),
                ("isp", "ISP"),
                ("city", "City"),
                ("regionName", "Region"),
                ("country", "Country"),
                ("timezone", "Timezone"),
                ("as", "AS"),
            ]:
                val = data.get(key)
                if val:
                    lines.append(f"  {label}: {val}")
            return "\n".join(lines)

        except Exception as e:
            return f"Lookup failed: {e}"
