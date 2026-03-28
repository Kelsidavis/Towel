"""XML skill — parse and extract data from XML."""
from __future__ import annotations
import json, re
from typing import Any
from towel.skills.base import Skill, ToolDefinition

def _xml_to_dict(xml: str) -> dict | list | str:
    """Simple XML to dict (handles basic cases without lxml)."""
    xml = re.sub(r'<\?.*?\?>', '', xml).strip()
    xml = re.sub(r'<!--.*?-->', '', xml, flags=re.DOTALL).strip()
    def parse_element(s):
        m = re.match(r'<(\w+)([^>]*)>(.*?)</\1>', s, re.DOTALL)
        if not m: return s.strip()
        tag, attrs, inner = m.group(1), m.group(2), m.group(3).strip()
        children = re.findall(r'<(\w+)[^>]*>.*?</\1>', inner, re.DOTALL)
        if children:
            result = {}
            for child_match in re.finditer(r'(<(\w+)[^>]*>.*?</\2>)', inner, re.DOTALL):
                key = child_match.group(2)
                val = parse_element(child_match.group(1))
                if key in result:
                    if not isinstance(result[key], list): result[key] = [result[key]]
                    result[key].append(val)
                else: result[key] = val
            return {tag: result}
        return {tag: inner}
    return parse_element(xml)

class XmlSkill(Skill):
    @property
    def name(self) -> str: return "xml_tools"
    @property
    def description(self) -> str: return "Parse XML to JSON and extract data"
    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="xml_to_json", description="Convert XML text to JSON",
                parameters={"type":"object","properties":{"xml":{"type":"string"}},"required":["xml"]}),
            ToolDefinition(name="xml_extract", description="Extract values matching an XPath-like pattern from XML",
                parameters={"type":"object","properties":{"xml":{"type":"string"},
                    "tag":{"type":"string","description":"Tag name to extract"}},"required":["xml","tag"]}),
        ]
    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        if tool_name == "xml_to_json":
            result = _xml_to_dict(arguments["xml"])
            return json.dumps(result, indent=2, ensure_ascii=False)
        elif tool_name == "xml_extract":
            tag = arguments["tag"]
            matches = re.findall(rf'<{tag}[^>]*>(.*?)</{tag}>', arguments["xml"], re.DOTALL)
            if not matches: return f"No <{tag}> elements found."
            return "\n".join(f"  {i+1}. {m.strip()}" for i, m in enumerate(matches[:50]))
        return f"Unknown: {tool_name}"
