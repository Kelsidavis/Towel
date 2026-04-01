"""Bookmark skill — save and organize URLs with tags and notes."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from towel.config import TOWEL_HOME
from towel.skills.base import Skill, ToolDefinition

BM_FILE = TOWEL_HOME / "bookmarks.json"

def _load() -> list[dict]:
    if not BM_FILE.exists(): return []
    try: return json.loads(BM_FILE.read_text(encoding="utf-8"))
    except: return []

def _save(bms: list[dict]) -> None:
    BM_FILE.parent.mkdir(parents=True, exist_ok=True)
    BM_FILE.write_text(json.dumps(bms, indent=2, ensure_ascii=False), encoding="utf-8")


class BookmarkSkill(Skill):
    @property
    def name(self) -> str: return "bookmarks"
    @property
    def description(self) -> str: return "Save, search, and organize URL bookmarks"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(name="bookmark_add", description="Save a URL bookmark",
                parameters={"type":"object","properties":{
                    "url":{"type":"string","description":"URL to bookmark"},
                    "title":{"type":"string","description":"Title/description"},
                    "tags":{"type":"array","items":{"type":"string"},"description":"Tags"},
                },"required":["url"]}),
            ToolDefinition(name="bookmark_search", description="Search bookmarks by keyword or tag",
                parameters={"type":"object","properties":{
                    "query":{"type":"string","description":"Search term"},
                    "tag":{"type":"string","description":"Filter by tag"},
                },"required":[]}),
            ToolDefinition(name="bookmark_list", description="List all bookmarks",
                parameters={"type":"object","properties":{
                    "limit":{"type":"integer","description":"Max entries (default: 20)"},
                }}),
            ToolDefinition(name="bookmark_delete", description="Delete a bookmark by index",
                parameters={"type":"object","properties":{
                    "index":{"type":"integer","description":"Bookmark index"},
                },"required":["index"]}),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "bookmark_add": return self._add(arguments["url"], arguments.get("title",""), arguments.get("tags",[]))
            case "bookmark_search": return self._search(arguments.get("query",""), arguments.get("tag"))
            case "bookmark_list": return self._list(arguments.get("limit",20))
            case "bookmark_delete": return self._delete(arguments["index"])
            case _: return f"Unknown tool: {tool_name}"

    def _add(self, url: str, title: str, tags: list[str]) -> str:
        bms = _load()
        bms.append({"url": url, "title": title or url, "tags": [t.lower() for t in tags],
                     "added": datetime.now(UTC).isoformat()})
        _save(bms)
        return f"Bookmarked: {title or url}"

    def _search(self, query: str, tag: str|None) -> str:
        bms = _load()
        q = query.lower()
        results = []
        for i, b in enumerate(bms):
            if tag and tag.lower() not in b.get("tags",[]): continue
            if q and q not in b["url"].lower() and q not in b.get("title","").lower(): continue
            results.append((i, b))
        if not results: return "No bookmarks found."
        lines = [f"Found {len(results)} bookmark(s):"]
        for idx, b in results[:20]:
            tags = " ".join(f"#{t}" for t in b.get("tags",[]))
            lines.append(f"  [{idx}] {b.get('title',b['url'])} {tags}")
            lines.append(f"       {b['url']}")
        return "\n".join(lines)

    def _list(self, limit: int) -> str:
        bms = _load()
        if not bms: return "No bookmarks saved."
        lines = [f"Bookmarks ({len(bms)} total):"]
        for i, b in enumerate(bms[-limit:]):
            idx = len(bms) - limit + i if limit < len(bms) else i
            tags = " ".join(f"#{t}" for t in b.get("tags",[]))
            lines.append(f"  [{idx}] {b.get('title',b['url'][:50])} {tags}")
        return "\n".join(lines)

    def _delete(self, index: int) -> str:
        bms = _load()
        if index < 0 or index >= len(bms): return f"Invalid index: {index}"
        removed = bms.pop(index)
        _save(bms)
        return f"Deleted: {removed.get('title', removed['url'])}"
