"""Random skill — dice, coin flips, random choices, shuffles."""
from __future__ import annotations
import random
from typing import Any
from towel.skills.base import Skill, ToolDefinition

class RandomSkill(Skill):
    @property
    def name(self) -> str: return "random"
    @property
    def description(self) -> str: return "Random generators — dice, coins, choices, shuffles"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="roll_dice", description="Roll dice (e.g., 2d6, 1d20, 3d8+5)",
                parameters={"type":"object","properties":{
                    "notation":{"type":"string","description":"Dice notation (NdS or NdS+M)"},
                },"required":["notation"]}),
            ToolDefinition(name="flip_coin", description="Flip one or more coins",
                parameters={"type":"object","properties":{
                    "count":{"type":"integer","description":"Number of flips (default: 1)"},
                }}),
            ToolDefinition(name="random_choice", description="Pick randomly from a list of options",
                parameters={"type":"object","properties":{
                    "options":{"type":"array","items":{"type":"string"},"description":"Options to choose from"},
                    "count":{"type":"integer","description":"How many to pick (default: 1)"},
                },"required":["options"]}),
            ToolDefinition(name="random_number", description="Generate a random number in a range",
                parameters={"type":"object","properties":{
                    "min":{"type":"number","description":"Minimum (default: 1)"},
                    "max":{"type":"number","description":"Maximum (default: 100)"},
                    "count":{"type":"integer","description":"How many (default: 1)"},
                }}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "roll_dice": return self._dice(arguments["notation"])
            case "flip_coin": return self._coin(arguments.get("count", 1))
            case "random_choice": return self._choice(arguments["options"], arguments.get("count", 1))
            case "random_number": return self._number(arguments.get("min",1), arguments.get("max",100), arguments.get("count",1))
            case _: return f"Unknown: {tool_name}"

    def _dice(self, notation: str) -> str:
        import re
        m = re.match(r"(\d+)d(\d+)(?:\+(\d+))?", notation.lower().strip())
        if not m: return f"Invalid notation: {notation} (use NdS or NdS+M)"
        n, sides, mod = int(m[1]), int(m[2]), int(m[3] or 0)
        if n > 100 or sides > 1000: return "Too many dice or sides."
        rolls = [random.randint(1, sides) for _ in range(n)]
        total = sum(rolls) + mod
        result = f"Rolling {notation}: [{', '.join(str(r) for r in rolls)}]"
        if mod: result += f" + {mod}"
        result += f" = {total}"
        return result

    def _coin(self, count: int) -> str:
        count = min(count, 100)
        flips = [random.choice(["Heads", "Tails"]) for _ in range(count)]
        if count == 1: return f"Coin flip: {flips[0]}"
        h = flips.count("Heads")
        return f"Flipped {count}: {h} Heads, {count-h} Tails\n  {', '.join(flips)}"

    def _choice(self, options: list[str], count: int) -> str:
        if not options: return "No options given."
        count = min(count, len(options))
        if count == 1: return f"Picked: {random.choice(options)}"
        picked = random.sample(options, count)
        return f"Picked {count}: {', '.join(picked)}"

    def _number(self, lo: float, hi: float, count: int) -> str:
        count = min(count, 100)
        if lo == int(lo) and hi == int(hi):
            nums = [random.randint(int(lo), int(hi)) for _ in range(count)]
        else:
            nums = [round(random.uniform(lo, hi), 2) for _ in range(count)]
        if count == 1: return str(nums[0])
        return ", ".join(str(n) for n in nums)
