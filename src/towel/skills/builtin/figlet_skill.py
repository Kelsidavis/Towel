"""FIGlet skill — large ASCII art text using built-in font."""
from __future__ import annotations
from typing import Any
from towel.skills.base import Skill, ToolDefinition

# Simple block font
_F = {
    'A':["  #  ","# # #","#####","# # #","# # #"],'B':["#### ","#   #","#### ","#   #","#### "],
    'C':[" ####","#    ","#    ","#    "," ####"],'D':["#### ","#   #","#   #","#   #","#### "],
    'E':["#####","#    ","###  ","#    ","#####"],'F':["#####","#    ","###  ","#    ","#    "],
    'G':[" ####","#    ","# ###","#   #"," ####"],'H':["#   #","#   #","#####","#   #","#   #"],
    'I':["#####","  #  ","  #  ","  #  ","#####"],'J':["  ###","    #","    #","#   #"," ### "],
    'K':["#  # ","# #  ","##   ","# #  ","#  # "],'L':["#    ","#    ","#    ","#    ","#####"],
    'M':["#   #","## ##","# # #","#   #","#   #"],'N':["#   #","##  #","# # #","#  ##","#   #"],
    'O':[" ### ","#   #","#   #","#   #"," ### "],'P':["#### ","#   #","#### ","#    ","#    "],
    'R':["#### ","#   #","#### ","# #  ","#  # "],'S':[" ####","#    "," ### ","    #","#### "],
    'T':["#####","  #  ","  #  ","  #  ","  #  "],'U':["#   #","#   #","#   #","#   #"," ### "],
    'V':["#   #","#   #","#   #"," # # ","  #  "],'W':["#   #","#   #","# # #","## ##","#   #"],
    'X':["#   #"," # # ","  #  "," # # ","#   #"],'Y':["#   #"," # # ","  #  ","  #  ","  #  "],
    'Z':["#####","   # ","  #  "," #   ","#####"],' ':["     ","     ","     ","     ","     "],
    '0':[" ### ","#  ##","# # #","##  #"," ### "],'1':["  #  "," ##  ","  #  ","  #  ","#####"],
    '!':["  #  ","  #  ","  #  ","     ","  #  "],
}

class FigletSkill(Skill):
    @property
    def name(self) -> str: return "figlet"
    @property
    def description(self) -> str: return "Generate large ASCII art text banners"
    def tools(self) -> list[ToolDefinition]:
        return [ToolDefinition(name="figlet_text", description="Render text as large ASCII art",
            parameters={"type":"object","properties":{"text":{"type":"string"},"char":{"type":"string","description":"Fill character (default: #)"}},"required":["text"]})]
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name != "figlet_text": return f"Unknown: {tool_name}"
        text = arguments["text"].upper()[:40]
        ch = arguments.get("char", "#")
        lines = [""] * 5
        for c in text:
            g = _F.get(c, _F.get(" "))
            if g:
                for i in range(5): lines[i] += g[i].replace("#", ch) + " "
        return "\n".join(lines)
