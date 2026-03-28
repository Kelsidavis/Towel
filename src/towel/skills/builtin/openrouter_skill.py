"""OpenRouter skill — query available AI models and pricing."""
from __future__ import annotations
from typing import Any
from towel.skills.base import Skill, ToolDefinition

class OpenRouterSkill(Skill):
    @property
    def name(self) -> str: return "openrouter"
    @property
    def description(self) -> str: return "Browse AI models on OpenRouter — pricing and capabilities"
    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="or_models", description="List available models on OpenRouter with pricing",
                parameters={"type":"object","properties":{
                    "query":{"type":"string","description":"Filter by name (optional)"},
                    "limit":{"type":"integer","description":"Max results (default: 10)"},
                }}),
        ]
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "or_models": return f"Unknown: {tool_name}"
        import httpx
        try:
            async with httpx.AsyncClient(timeout=10) as c:
                resp = await c.get("https://openrouter.ai/api/v1/models")
                models = resp.json().get("data",[])
                q = arguments.get("query","").lower()
                if q: models = [m for m in models if q in m.get("id","").lower() or q in m.get("name","").lower()]
                models = models[:arguments.get("limit",10)]
                if not models: return "No models found."
                lines = [f"OpenRouter models ({len(models)}):"]
                for m in models:
                    price = m.get("pricing",{})
                    prompt_cost = float(price.get("prompt","0")) * 1_000_000
                    lines.append(f"  {m['id']}")
                    lines.append(f"    Context: {m.get('context_length','?')} · ${prompt_cost:.2f}/1M tokens")
                return "\n".join(lines)
        except Exception as e: return f"Error: {e}"
