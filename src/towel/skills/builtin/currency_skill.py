"""Currency skill — exchange rates via frankfurter.app (free, no key)."""

from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition


class CurrencySkill(Skill):
    @property
    def name(self) -> str:
        return "currency"

    @property
    def description(self) -> str:
        return "Currency exchange rates and conversion"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="currency_convert",
                description="Convert between currencies",
                parameters={
                    "type": "object",
                    "properties": {
                        "amount": {"type": "number", "description": "Amount to convert"},
                        "from_currency": {
                            "type": "string",
                            "description": "Source currency (e.g., USD)",
                        },
                        "to_currency": {
                            "type": "string",
                            "description": "Target currency (e.g., EUR)",
                        },
                    },
                    "required": ["amount", "from_currency", "to_currency"],
                },
            ),
            ToolDefinition(
                name="currency_rates",
                description="Get exchange rates for a base currency",
                parameters={
                    "type": "object",
                    "properties": {
                        "base": {"type": "string", "description": "Base currency (default: USD)"},
                    },
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        import httpx

        try:
            async with httpx.AsyncClient(timeout=10) as client:
                if tool_name == "currency_convert":
                    amt = arguments["amount"]
                    fr = arguments["from_currency"].upper()
                    to = arguments["to_currency"].upper()
                    resp = await client.get(
                        f"https://api.frankfurter.app/latest?amount={amt}&from={fr}&to={to}"
                    )
                    data = resp.json()
                    rate = data.get("rates", {}).get(to)
                    if rate:
                        return f"{amt} {fr} = {rate} {to}"
                    return f"Cannot convert {fr} to {to}"
                elif tool_name == "currency_rates":
                    base = arguments.get("base", "USD").upper()
                    resp = await client.get(f"https://api.frankfurter.app/latest?from={base}")
                    data = resp.json()
                    rates = data.get("rates", {})
                    lines = [f"Exchange rates for {base} ({data.get('date', '?')}):"]
                    for cur, rate in sorted(rates.items()):
                        lines.append(f"  {cur}: {rate}")
                    return "\n".join(lines)
        except Exception as e:
            return f"Currency error: {e}"
        return f"Unknown tool: {tool_name}"
