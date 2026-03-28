"""Country skill — look up country info, codes, currencies."""
from __future__ import annotations
from typing import Any
from towel.skills.base import Skill, ToolDefinition

class CountrySkill(Skill):
    @property
    def name(self) -> str: return "country"
    @property
    def description(self) -> str: return "Look up country info — capital, currency, population, codes"
    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="country_info", description="Get info about a country",
                parameters={"type":"object","properties":{"name":{"type":"string","description":"Country name"}},"required":["name"]}),
        ]
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "country_info": return f"Unknown: {tool_name}"
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                resp = await client.get(f"https://restcountries.com/v3.1/name/{arguments['name']}")
                if resp.status_code == 404: return f"Country not found: {arguments['name']}"
                c = resp.json()[0]
                currencies = ", ".join(f"{v['name']} ({v.get('symbol','')})" for v in c.get("currencies",{}).values())
                langs = ", ".join(c.get("languages",{}).values())
                return (f"{c['name']['common']} ({c['name']['official']})\n"
                        f"  Capital: {', '.join(c.get('capital',[]))}\n"
                        f"  Region: {c.get('region','')} / {c.get('subregion','')}\n"
                        f"  Population: {c.get('population',0):,}\n"
                        f"  Currency: {currencies}\n"
                        f"  Languages: {langs}\n"
                        f"  Codes: {c.get('cca2','')} / {c.get('cca3','')}\n"
                        f"  Timezone: {', '.join(c.get('timezones',[]))}")
        except Exception as e: return f"Error: {e}"
