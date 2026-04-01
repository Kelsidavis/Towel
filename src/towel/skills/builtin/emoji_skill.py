"""Emoji skill — search and look up emoji by name or keyword."""
from __future__ import annotations

from typing import Any

from towel.skills.base import Skill, ToolDefinition

_EMOJI: dict[str, str] = {
    "smile":"😊","laugh":"😂","heart":"❤️","fire":"🔥","thumbsup":"👍","thumbsdown":"👎",
    "clap":"👏","wave":"👋","think":"🤔","cry":"😢","angry":"😠","cool":"😎",
    "rocket":"🚀","star":"⭐","check":"✅","cross":"❌","warning":"⚠️","lock":"🔒",
    "key":"🔑","bug":"🐛","gear":"⚙️","bulb":"💡","pin":"📌","book":"📖",
    "clock":"🕐","calendar":"📅","mail":"📧","phone":"📱","laptop":"💻","cloud":"☁️",
    "sun":"☀️","moon":"🌙","rain":"🌧️","snow":"❄️","lightning":"⚡","rainbow":"🌈",
    "tree":"🌳","flower":"🌸","dog":"🐶","cat":"🐱","bird":"🐦","fish":"🐟",
    "coffee":"☕","pizza":"🍕","cake":"🎂","beer":"🍺","party":"🎉","music":"🎵",
    "art":"🎨","movie":"🎬","game":"🎮","trophy":"🏆","medal":"🏅","crown":"👑",
    "money":"💰","chart":"📊","link":"🔗","search":"🔍","tools":"🔧","shield":"🛡️",
    "100":"💯","ok":"👌","pray":"🙏","muscle":"💪","brain":"🧠","eyes":"👀",
    "sparkles":"✨","boom":"💥","zzz":"💤","poop":"💩","skull":"💀","ghost":"👻",
    "alien":"👽","robot":"🤖","unicorn":"🦄","dragon":"🐉","snake":"🐍","penguin":"🐧",
}

class EmojiSkill(Skill):
    @property
    def name(self) -> str: return "emoji"
    @property
    def description(self) -> str: return "Search and look up emoji by name"
    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="emoji_search", description="Search for an emoji by keyword",
                parameters={"type":"object","properties":{"query":{"type":"string"}},"required":["query"]}),
            ToolDefinition(name="emoji_list", description="List all available emoji",
                parameters={"type":"object","properties":{}}),
        ]
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "emoji_search":
            q = arguments["query"].lower()
            matches = [(k,v) for k,v in _EMOJI.items() if q in k]
            if not matches: return f"No emoji matching: {q}"
            return "\n".join(f"  {v}  :{k}:" for k,v in matches)
        elif tool_name == "emoji_list":
            return "\n".join(f"  {v}  :{k}:" for k,v in sorted(_EMOJI.items()))
        return f"Unknown: {tool_name}"
