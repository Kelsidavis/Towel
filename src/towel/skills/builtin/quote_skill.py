"""Quote skill — random inspirational quotes."""
from __future__ import annotations

import random
from typing import Any

from towel.skills.base import Skill, ToolDefinition

_QUOTES = [
    ("Don't Panic.", "Douglas Adams"),
    ("The best way to predict the future is to invent it.", "Alan Kay"),
    ("Talk is cheap. Show me the code.", "Linus Torvalds"),
    ("Simplicity is the ultimate sophistication.", "Leonardo da Vinci"),
    ("First, solve the problem. Then, write the code.", "John Johnson"),
    ("Code is like humor. When you have to explain it, it's bad.", "Cory House"),
    ("Any fool can write code that a computer can understand.", "Martin Fowler"),
    ("The only way to do great work is to love what you do.", "Steve Jobs"),
    ("Perfection is achieved not when there is nothing more to add, but when there is nothing left to take away.", "Antoine de Saint-Exupéry"),
    ("It works on my machine.", "Every developer"),
    ("There are only two hard things in CS: cache invalidation and naming things.", "Phil Karlton"),
    ("The best error message is the one that never shows up.", "Thomas Fuchs"),
    ("Debugging is twice as hard as writing the code in the first place.", "Brian Kernighan"),
    ("A towel is about the most massively useful thing an interstellar hitchhiker can have.", "Douglas Adams"),
    ("So long, and thanks for all the fish.", "Douglas Adams"),
]

class QuoteSkill(Skill):
    @property
    def name(self) -> str: return "quotes"
    @property
    def description(self) -> str: return "Random inspirational and programming quotes"
    def tools(self) -> list[ToolDefinition]:
        return [ToolDefinition(name="random_quote", description="Get a random quote",
            parameters={"type":"object","properties":{}})]
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        q, a = random.choice(_QUOTES)
        return f'"{q}"\n  — {a}'
