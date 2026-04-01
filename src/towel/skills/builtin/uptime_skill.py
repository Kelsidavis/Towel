"""Uptime monitor skill — check if URLs are up and measure response time."""
from __future__ import annotations

import time as _time
from typing import Any

from towel.skills.base import Skill, ToolDefinition

_history: list[dict] = []

class UptimeSkill(Skill):
    @property
    def name(self) -> str: return "uptime_monitor"
    @property
    def description(self) -> str: return "Check if URLs are up and track response times"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="uptime_check", description="Check if a URL is up and measure response time",
                parameters={"type":"object","properties":{
                    "url":{"type":"string","description":"URL to check"},
                    "expected_status":{"type":"integer","description":"Expected HTTP status (default: 200)"},
                },"required":["url"]}),
            ToolDefinition(name="uptime_batch", description="Check multiple URLs at once",
                parameters={"type":"object","properties":{
                    "urls":{"type":"array","items":{"type":"string"},"description":"URLs to check"},
                },"required":["urls"]}),
            ToolDefinition(name="uptime_history", description="Show recent check history",
                parameters={"type":"object","properties":{
                    "limit":{"type":"integer","description":"Max entries (default: 20)"},
                }}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "uptime_check": return await self._check(arguments["url"], arguments.get("expected_status", 200))
            case "uptime_batch": return await self._batch(arguments["urls"])
            case "uptime_history": return self._hist(arguments.get("limit", 20))
            case _: return f"Unknown: {tool_name}"

    async def _check(self, url: str, expected: int) -> str:
        import httpx
        start = _time.perf_counter()
        try:
            async with httpx.AsyncClient(timeout=10, follow_redirects=True) as c:
                resp = await c.get(url, headers={"User-Agent": "Towel-Uptime/1.0"})
            ms = (_time.perf_counter() - start) * 1000
            ok = resp.status_code == expected
            status = "UP" if ok else "DOWN"
            icon = "+" if ok else "!"
            entry = {"url": url, "status": status, "code": resp.status_code, "ms": round(ms),
                     "time": _time.strftime("%H:%M:%S")}
            _history.insert(0, entry)
            if len(_history) > 100: _history.pop()
            return f"[{icon}] {url}: {status} (HTTP {resp.status_code}, {ms:.0f}ms)"
        except Exception as e:
            entry = {"url": url, "status": "DOWN", "code": 0, "ms": 0, "time": _time.strftime("%H:%M:%S")}
            _history.insert(0, entry)
            return f"[!] {url}: DOWN ({e})"

    async def _batch(self, urls: list[str]) -> str:
        import asyncio
        results = await asyncio.gather(*[self._check(u, 200) for u in urls[:20]])
        return "\n".join(results)

    def _hist(self, limit: int) -> str:
        if not _history: return "No check history."
        lines = [f"Recent checks ({min(limit, len(_history))}):"]
        for e in _history[:limit]:
            icon = "+" if e["status"] == "UP" else "!"
            lines.append(f"  [{icon}] {e['time']} {e['url'][:40]} {e['status']} {e['code']} {e['ms']}ms")
        return "\n".join(lines)
