"""CVE skill — search for security vulnerabilities."""
from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition


class CveSkill(Skill):
    @property
    def name(self) -> str: return "cve"
    @property
    def description(self) -> str: return "Search CVE security vulnerability database"
    def tools(self) -> list[ToolDefinition]:
        return [ToolDefinition(name="cve_search", description="Search for CVEs by keyword",
            parameters={"type":"object","properties":{"query":{"type":"string"},"limit":{"type":"integer","description":"Max (default: 5)"}},"required":["query"]})]
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "cve_search": return f"Unknown: {tool_name}"
        import httpx
        try:
            async with httpx.AsyncClient(timeout=15) as c:
                resp = await c.get("https://services.nvd.nist.gov/rest/json/cves/2.0",
                    params={"keywordSearch": arguments["query"], "resultsPerPage": arguments.get("limit", 5)})
                vulns = resp.json().get("vulnerabilities", [])
                if not vulns: return "No CVEs found."
                lines = [f"CVE results for '{arguments['query']}':"]
                for v in vulns:
                    cve = v.get("cve", {})
                    cid = cve.get("id", "?")
                    desc = cve.get("descriptions", [{}])[0].get("value", "")[:100]
                    metrics = cve.get("metrics", {})
                    score = "?"
                    for key in ["cvssMetricV31", "cvssMetricV30", "cvssMetricV2"]:
                        if key in metrics:
                            score = metrics[key][0].get("cvssData", {}).get("baseScore", "?")
                            break
                    lines.append(f"\n  {cid} (CVSS: {score})\n    {desc}")
                return "\n".join(lines)
        except Exception as e: return f"CVE error: {e}"
