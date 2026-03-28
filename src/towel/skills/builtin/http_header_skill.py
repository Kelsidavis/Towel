"""HTTP header skill — analyze, explain, and generate HTTP headers."""

from __future__ import annotations
from typing import Any
from towel.skills.base import Skill, ToolDefinition

_HEADER_DB: dict[str, str] = {
    "content-type": "MIME type of the body (e.g., application/json, text/html)",
    "authorization": "Credentials for authentication (Bearer token, Basic auth)",
    "cache-control": "Caching directives (no-cache, max-age, public, private)",
    "content-length": "Size of the body in bytes",
    "content-encoding": "Compression (gzip, br, deflate)",
    "accept": "Media types the client accepts",
    "accept-encoding": "Compression algorithms the client supports",
    "accept-language": "Preferred languages",
    "access-control-allow-origin": "CORS: which origins can access the resource",
    "access-control-allow-methods": "CORS: allowed HTTP methods",
    "access-control-allow-headers": "CORS: allowed request headers",
    "cookie": "Cookies sent from client to server",
    "set-cookie": "Server sets a cookie on the client",
    "location": "URL to redirect to (with 301/302/307/308)",
    "user-agent": "Client software identification",
    "referer": "URL of the page that linked to this request",
    "x-forwarded-for": "Client IP when behind a proxy/load balancer",
    "x-request-id": "Unique request identifier for tracing",
    "x-frame-options": "Clickjacking protection (DENY, SAMEORIGIN)",
    "x-content-type-options": "Prevent MIME type sniffing (nosniff)",
    "strict-transport-security": "Force HTTPS (HSTS) with max-age",
    "content-security-policy": "XSS/injection protection policy",
    "x-xss-protection": "Legacy XSS filter (deprecated, use CSP)",
    "etag": "Version identifier for caching",
    "if-none-match": "Conditional request using ETag",
    "if-modified-since": "Conditional request using timestamp",
    "last-modified": "When the resource was last changed",
    "transfer-encoding": "How the body is transferred (chunked)",
    "vary": "Which headers affect caching",
    "www-authenticate": "Authentication method required (401 response)",
    "retry-after": "When to retry after rate limiting (429/503)",
    "x-ratelimit-limit": "Rate limit ceiling",
    "x-ratelimit-remaining": "Requests remaining in window",
    "server": "Server software identification",
}


class HttpHeaderSkill(Skill):
    @property
    def name(self) -> str: return "headers"
    @property
    def description(self) -> str: return "Analyze, explain, and generate HTTP headers"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="header_explain", description="Explain what HTTP headers do",
                parameters={"type":"object","properties":{
                    "headers":{"type":"string","description":"Header names (comma-separated) or raw header block"},
                },"required":["headers"]}),
            ToolDefinition(name="header_security", description="Analyze response headers for security issues",
                parameters={"type":"object","properties":{
                    "headers":{"type":"string","description":"Raw HTTP response headers"},
                },"required":["headers"]}),
            ToolDefinition(name="header_cors", description="Generate CORS headers for a given configuration",
                parameters={"type":"object","properties":{
                    "origins":{"type":"string","description":"Allowed origins (comma-separated, or * for all)"},
                    "methods":{"type":"string","description":"Allowed methods (default: GET,POST,OPTIONS)"},
                    "credentials":{"type":"boolean","description":"Allow credentials (default: false)"},
                },"required":["origins"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "header_explain": return self._explain(arguments["headers"])
            case "header_security": return self._security(arguments["headers"])
            case "header_cors": return self._cors(arguments["origins"], arguments.get("methods","GET,POST,OPTIONS"), arguments.get("credentials",False))
            case _: return f"Unknown tool: {tool_name}"

    def _explain(self, headers: str) -> str:
        lines = []
        # Parse as raw headers or comma-separated names
        if ":" in headers:
            for line in headers.strip().splitlines():
                if ":" not in line: continue
                name, _, val = line.partition(":")
                name = name.strip().lower()
                desc = _HEADER_DB.get(name, "Custom/unknown header")
                lines.append(f"  {name}: {val.strip()}")
                lines.append(f"    → {desc}")
        else:
            for name in headers.split(","):
                name = name.strip().lower()
                desc = _HEADER_DB.get(name, "Unknown header")
                lines.append(f"  {name}: {desc}")
        return "Header reference:\n" + "\n".join(lines) if lines else "No headers to explain."

    def _security(self, headers: str) -> str:
        parsed = {}
        for line in headers.strip().splitlines():
            if ":" not in line: continue
            name, _, val = line.partition(":")
            parsed[name.strip().lower()] = val.strip()

        checks = []
        if "strict-transport-security" not in parsed:
            checks.append("  [!] Missing Strict-Transport-Security (HSTS)")
        if "x-content-type-options" not in parsed:
            checks.append("  [!] Missing X-Content-Type-Options (set to nosniff)")
        if "x-frame-options" not in parsed and "content-security-policy" not in parsed:
            checks.append("  [!] Missing clickjacking protection (X-Frame-Options or CSP frame-ancestors)")
        if "content-security-policy" not in parsed:
            checks.append("  [!] Missing Content-Security-Policy")
        server = parsed.get("server", "")
        if server and any(v in server.lower() for v in ["apache","nginx","iis"]):
            checks.append(f"  [!] Server header reveals software: {server}")
        if "x-powered-by" in parsed:
            checks.append(f"  [!] X-Powered-By reveals technology: {parsed['x-powered-by']}")

        # Good things
        good = []
        if "strict-transport-security" in parsed: good.append("  [+] HSTS enabled")
        if "x-content-type-options" in parsed: good.append("  [+] MIME sniffing prevented")
        if "content-security-policy" in parsed: good.append("  [+] CSP configured")

        result = "Security header analysis:\n"
        if good: result += "\n".join(good) + "\n"
        if checks: result += "\n".join(checks)
        elif not good: result += "  No headers to analyze."
        else: result += "  No issues found."
        return result

    def _cors(self, origins: str, methods: str, credentials: bool) -> str:
        lines = ["CORS headers:\n"]
        lines.append(f"  Access-Control-Allow-Origin: {origins}")
        lines.append(f"  Access-Control-Allow-Methods: {methods}")
        lines.append(f"  Access-Control-Allow-Headers: Content-Type, Authorization")
        if credentials:
            lines.append(f"  Access-Control-Allow-Credentials: true")
            if origins == "*":
                lines.append("\n  [!] Warning: credentials=true is incompatible with origin=*")
        lines.append(f"  Access-Control-Max-Age: 86400")
        return "\n".join(lines)
