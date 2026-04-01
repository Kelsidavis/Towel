"""Knowledge skill — personal knowledge base for notes, links, and facts."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from towel.config import TOWEL_HOME
from towel.skills.base import Skill, ToolDefinition

KB_FILE = TOWEL_HOME / "knowledge.json"


def _load_kb() -> list[dict]:
    if not KB_FILE.exists(): return []
    try: return json.loads(KB_FILE.read_text(encoding="utf-8"))
    except: return []

def _save_kb(entries: list[dict]) -> None:
    KB_FILE.parent.mkdir(parents=True, exist_ok=True)
    KB_FILE.write_text(json.dumps(entries, indent=2, ensure_ascii=False), encoding="utf-8")


class KnowledgeSkill(Skill):
    @property
    def name(self) -> str: return "knowledge"
    @property
    def description(self) -> str: return "Personal knowledge base — save, search, and recall notes, links, and facts"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="kb_add", description="Save a note, link, or fact to the knowledge base",
                parameters={"type":"object","properties":{
                    "content":{"type":"string","description":"The note/fact/link to save"},
                    "tags":{"type":"array","items":{"type":"string"},"description":"Tags for categorization"},
                    "title":{"type":"string","description":"Optional title"},
                },"required":["content"]}),
            ToolDefinition(name="kb_search", description="Search the knowledge base by keyword or tag",
                parameters={"type":"object","properties":{
                    "query":{"type":"string","description":"Search query"},
                    "tag":{"type":"string","description":"Filter by tag"},
                },"required":["query"]}),
            ToolDefinition(name="kb_list", description="List recent entries in the knowledge base",
                parameters={"type":"object","properties":{
                    "limit":{"type":"integer","description":"Max entries (default: 20)"},
                    "tag":{"type":"string","description":"Filter by tag"},
                }}),
            ToolDefinition(name="kb_delete", description="Delete a knowledge base entry by index",
                parameters={"type":"object","properties":{
                    "index":{"type":"integer","description":"Entry index (from kb_list)"},
                },"required":["index"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "kb_add": return self._add(arguments["content"], arguments.get("tags",[]), arguments.get("title",""))
            case "kb_search": return self._search(arguments["query"], arguments.get("tag"))
            case "kb_list": return self._list(arguments.get("limit",20), arguments.get("tag"))
            case "kb_delete": return self._delete(arguments["index"])
            case _: return f"Unknown tool: {tool_name}"

    def _add(self, content: str, tags: list[str], title: str) -> str:
        entries = _load_kb()
        entry = {"content": content, "tags": [t.lower() for t in tags], "title": title,
                 "created": datetime.now(UTC).isoformat()}
        entries.append(entry)
        _save_kb(entries)
        return f"Saved to knowledge base ({len(entries)} total). Tags: {', '.join(tags) or 'none'}"

    def _search(self, query: str, tag: str|None) -> str:
        entries = _load_kb()
        q = query.lower()
        results = []
        for i, e in enumerate(entries):
            if tag and tag.lower() not in e.get("tags",[]): continue
            if q in e["content"].lower() or q in e.get("title","").lower() or q in str(e.get("tags",[])):
                results.append((i, e))
        if not results: return f"No matches for '{query}'"
        lines = [f"Found {len(results)} match(es):"]
        for idx, e in results[:20]:
            title = e.get("title") or e["content"][:50]
            tags = " ".join(f"#{t}" for t in e.get("tags",[]))
            lines.append(f"  [{idx}] {title} {tags}")
        return "\n".join(lines)

    def _list(self, limit: int, tag: str|None) -> str:
        entries = _load_kb()
        if tag: entries = [(i,e) for i,e in enumerate(entries) if tag.lower() in e.get("tags",[])]
        else: entries = list(enumerate(entries))
        if not entries: return "Knowledge base is empty."
        lines = [f"Knowledge base ({len(entries)} entries):"]
        for idx, e in entries[-limit:]:
            title = e.get("title") or e["content"][:50]
            tags = " ".join(f"#{t}" for t in e.get("tags",[]))
            lines.append(f"  [{idx}] {title} {tags}")
        return "\n".join(lines)

    def _delete(self, index: int) -> str:
        entries = _load_kb()
        if index < 0 or index >= len(entries): return f"Invalid index: {index}"
        removed = entries.pop(index)
        _save_kb(entries)
        return f"Deleted: {removed.get('title') or removed['content'][:40]}"
