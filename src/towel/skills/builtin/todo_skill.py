"""Todo skill — task management with priorities and due dates."""

from __future__ import annotations

import json
from datetime import UTC, datetime
from typing import Any

from towel.config import TOWEL_HOME
from towel.skills.base import Skill, ToolDefinition

TODO_FILE = TOWEL_HOME / "todos.json"


def _load() -> list[dict]:
    if not TODO_FILE.exists():
        return []
    try:
        return json.loads(TODO_FILE.read_text(encoding="utf-8"))
    except Exception:
        return []


def _save(todos: list[dict]) -> None:
    TODO_FILE.parent.mkdir(parents=True, exist_ok=True)
    TODO_FILE.write_text(json.dumps(todos, indent=2, ensure_ascii=False), encoding="utf-8")


class TodoSkill(Skill):
    @property
    def name(self) -> str:
        return "todo"

    @property
    def description(self) -> str:
        return "Task management — add, complete, list, and prioritize todos"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="todo_add",
                description="Add a new todo item",
                parameters={
                    "type": "object",
                    "properties": {
                        "task": {"type": "string", "description": "Task description"},
                        "priority": {
                            "type": "string",
                            "enum": ["high", "medium", "low"],
                            "description": "Priority (default: medium)",
                        },
                        "due": {"type": "string", "description": "Due date (YYYY-MM-DD, optional)"},
                        "tags": {
                            "type": "array",
                            "items": {"type": "string"},
                            "description": "Tags",
                        },
                    },
                    "required": ["task"],
                },
            ),
            ToolDefinition(
                name="todo_list",
                description="List all todos, optionally filtered",
                parameters={
                    "type": "object",
                    "properties": {
                        "show_done": {
                            "type": "boolean",
                            "description": "Include completed (default: false)",
                        },
                        "priority": {"type": "string", "description": "Filter by priority"},
                        "tag": {"type": "string", "description": "Filter by tag"},
                    },
                },
            ),
            ToolDefinition(
                name="todo_done",
                description="Mark a todo as completed by index",
                parameters={
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "Todo index"},
                    },
                    "required": ["index"],
                },
            ),
            ToolDefinition(
                name="todo_remove",
                description="Delete a todo by index",
                parameters={
                    "type": "object",
                    "properties": {
                        "index": {"type": "integer", "description": "Todo index"},
                    },
                    "required": ["index"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "todo_add":
                return self._add(
                    arguments["task"],
                    arguments.get("priority", "medium"),
                    arguments.get("due"),
                    arguments.get("tags", []),
                )
            case "todo_list":
                return self._list(
                    arguments.get("show_done", False),
                    arguments.get("priority"),
                    arguments.get("tag"),
                )
            case "todo_done":
                return self._done(arguments["index"])
            case "todo_remove":
                return self._remove(arguments["index"])
            case _:
                return f"Unknown tool: {tool_name}"

    def _add(self, task: str, priority: str, due: str | None, tags: list[str]) -> str:
        todos = _load()
        entry = {
            "task": task,
            "priority": priority,
            "done": False,
            "tags": [t.lower() for t in tags],
            "created": datetime.now(UTC).isoformat(),
        }
        if due:
            entry["due"] = due
        todos.append(entry)
        _save(todos)
        icon = {"high": "!", "medium": "-", "low": "."}[priority]
        return f"[{icon}] Added: {task}" + (f" (due {due})" if due else "")

    def _list(self, show_done: bool, priority: str | None, tag: str | None) -> str:
        todos = _load()
        lines = []
        pri_order = {"high": 0, "medium": 1, "low": 2}
        filtered = []
        for i, t in enumerate(todos):
            if not show_done and t.get("done"):
                continue
            if priority and t.get("priority") != priority:
                continue
            if tag and tag.lower() not in t.get("tags", []):
                continue
            filtered.append((i, t))
        filtered.sort(key=lambda x: pri_order.get(x[1].get("priority", "medium"), 1))
        if not filtered:
            return "No todos." + (" (use show_done=true to see completed)" if not show_done else "")
        for idx, t in filtered:
            check = "x" if t.get("done") else " "
            pri = {"high": "!!!", "medium": "--", "low": ".."}[t.get("priority", "medium")]
            due = f" (due {t['due']})" if t.get("due") else ""
            tags = " ".join(f"#{tg}" for tg in t.get("tags", []))
            lines.append(f"  [{check}] {idx}. {pri} {t['task']}{due} {tags}")
        return f"{len(filtered)} todo(s):\n" + "\n".join(lines)

    def _done(self, index: int) -> str:
        todos = _load()
        if index < 0 or index >= len(todos):
            return f"Invalid index: {index}"
        todos[index]["done"] = True
        todos[index]["completed_at"] = datetime.now(UTC).isoformat()
        _save(todos)
        return f"Completed: {todos[index]['task']}"

    def _remove(self, index: int) -> str:
        todos = _load()
        if index < 0 or index >= len(todos):
            return f"Invalid index: {index}"
        removed = todos.pop(index)
        _save(todos)
        return f"Removed: {removed['task']}"
