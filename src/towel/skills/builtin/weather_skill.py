"""Weather skill — get weather forecasts using wttr.in (no API key)."""
from __future__ import annotations
from typing import Any
from towel.skills.base import Skill, ToolDefinition

class WeatherSkill(Skill):
    @property
    def name(self) -> str: return "weather"
    @property
    def description(self) -> str: return "Get weather forecasts by city"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="weather_now", description="Get current weather for a city",
                parameters={"type":"object","properties":{
                    "city":{"type":"string","description":"City name"},
                },"required":["city"]}),
            ToolDefinition(name="weather_forecast", description="Get 3-day weather forecast",
                parameters={"type":"object","properties":{
                    "city":{"type":"string","description":"City name"},
                },"required":["city"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        import httpx
        city = arguments["city"].replace(" ", "+")
        try:
            async with httpx.AsyncClient(timeout=10) as client:
                if tool_name == "weather_now":
                    resp = await client.get(f"https://wttr.in/{city}?format=%l:+%c+%t+%h+%w")
                    return resp.text.strip()
                elif tool_name == "weather_forecast":
                    resp = await client.get(f"https://wttr.in/{city}?format=3")
                    return resp.text.strip()
                return f"Unknown tool: {tool_name}"
        except Exception as e: return f"Weather error: {e}"
