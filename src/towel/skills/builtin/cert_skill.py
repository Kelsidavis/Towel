"""SSL certificate skill — inspect TLS certificates for any domain."""

from __future__ import annotations

import asyncio
import socket
import ssl
from datetime import datetime
from typing import Any

from towel.skills.base import Skill, ToolDefinition


class CertSkill(Skill):
    @property
    def name(self) -> str:
        return "cert"

    @property
    def description(self) -> str:
        return "Inspect SSL/TLS certificates for any domain"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="cert_check",
                description="Get SSL certificate details for a domain",
                parameters={
                    "type": "object",
                    "properties": {
                        "domain": {
                            "type": "string",
                            "description": "Domain to check (e.g., github.com)",
                        },
                        "port": {"type": "integer", "description": "Port (default: 443)"},
                    },
                    "required": ["domain"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "cert_check":
            return f"Unknown: {tool_name}"
        domain = arguments["domain"]
        port = arguments.get("port", 443)
        loop = asyncio.get_event_loop()
        try:

            def _check():
                ctx = ssl.create_default_context()
                with ctx.wrap_socket(socket.socket(), server_hostname=domain) as s:
                    s.settimeout(10)
                    s.connect((domain, port))
                    cert = s.getpeercert()
                    return cert

            cert = await loop.run_in_executor(None, _check)
            subject = dict(x[0] for x in cert.get("subject", []))
            issuer = dict(x[0] for x in cert.get("issuer", []))
            not_before = cert.get("notBefore", "?")
            not_after = cert.get("notAfter", "?")
            # Parse expiry
            try:
                exp = datetime.strptime(not_after, "%b %d %H:%M:%S %Y %Z")
                days_left = (exp - datetime.utcnow()).days
                expiry_status = f"{days_left} days remaining" if days_left > 0 else "EXPIRED"
            except Exception:
                expiry_status = "?"
            sans = [v for _, v in cert.get("subjectAltName", [])]
            lines = [
                f"SSL Certificate: {domain}:{port}",
                f"  Subject: {subject.get('commonName', '?')}",
                f"  Issuer:  {issuer.get('organizationName', issuer.get('commonName', '?'))}",
                f"  Valid:   {not_before} → {not_after}",
                f"  Status:  {expiry_status}",
                f"  SANs:    {', '.join(sans[:5])}"
                + (f" (+{len(sans) - 5})" if len(sans) > 5 else ""),
                f"  Serial:  {cert.get('serialNumber', '?')}",
            ]
            return "\n".join(lines)
        except ssl.SSLCertVerificationError as e:
            return f"Certificate verification failed: {e}"
        except Exception as e:
            return f"Error: {e}"
