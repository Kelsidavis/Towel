"""Memory skill — lets the agent remember and recall across sessions."""

from __future__ import annotations

from typing import Any

from towel.memory.guard import reject_reason
from towel.memory.store import MEMORY_TYPES, MemoryStore
from towel.skills.base import Skill, ToolDefinition


class MemorySkill(Skill):
    """Gives the agent persistent memory across conversations."""

    def __init__(self, store: MemoryStore | None = None, sync: Any = None) -> None:
        self._store = store or MemoryStore()
        # Optional ClusterMemorySync. When set, remember/forget route
        # through it so writes AND removals replicate to the cluster
        # (a raw store.forget() never broadcasts a removal). Typed as
        # Any to avoid importing the cluster module into the worker
        # registry path that constructs MemorySkill without sync.
        self._sync = sync

    def attach_sync(self, sync: Any) -> None:
        """Wire this skill to a ClusterMemorySync after construction.

        The gateway builds the sync object after the agent (and its
        skills) already exist, so it attaches it here. Once set,
        tool-driven writes/removals propagate cluster-wide instead of
        only riding the next full snapshot.
        """
        self._sync = sync

    @property
    def name(self) -> str:
        return "memory"

    @property
    def description(self) -> str:
        return "Remember and recall facts across sessions"

    def tools(self) -> list[ToolDefinition]:
        return [
            ToolDefinition(
                name="remember",
                description=(
                    "Store a fact in persistent memory. Use for "
                    "user preferences, project details, or "
                    "anything worth remembering across sessions."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {
                            "type": "string",
                            "description": (
                                "Short identifier (e.g., "
                                "'user_name', 'preferred_language')"
                            ),
                        },
                        "content": {"type": "string", "description": "The fact to remember"},
                        "type": {
                            "type": "string",
                            "description": (
                                "Memory type: "
                                f"{', '.join(MEMORY_TYPES)} "
                                "(default: fact)"
                            ),
                        },
                    },
                    "required": ["key", "content"],
                },
            ),
            ToolDefinition(
                name="forget",
                description="Remove a memory by key.",
                parameters={
                    "type": "object",
                    "properties": {
                        "key": {"type": "string", "description": "The memory key to forget"},
                    },
                    "required": ["key"],
                },
            ),
            ToolDefinition(
                name="recall",
                description=(
                    "Search your memories. Use when the user "
                    "references something from a past session. Pass "
                    "a tag to restrict the search to a labeled subset."
                ),
                parameters={
                    "type": "object",
                    "properties": {
                        "query": {
                            "type": "string",
                            "description": "Search term (searches keys and content)",
                        },
                        "tag": {
                            "type": "string",
                            "description": "Optional: restrict to memories carrying this tag.",
                        },
                    },
                    "required": ["query"],
                },
            ),
        ]

    async def execute(self, tool_name: str, arguments: dict[str, Any]) -> Any:
        match tool_name:
            case "remember":
                key = arguments["key"]
                content = arguments["content"]
                mtype = arguments.get("type", "fact")
                if mtype not in MEMORY_TYPES:
                    mtype = "fact"
                # Write-time guard: refuse judgments/safety-flags about the
                # user. The store is for facts the user stated, not triage
                # notes a model invents about them — and refused writes
                # never reach the cluster sync queue either.
                refusal = reject_reason(key, content)
                if refusal is not None:
                    return refusal
                # Tag tool-driven writes so 'memory stats' / 'memory tidy'
                # can distinguish them from operator-CLI and auto-capture
                # entries. Existing keys keep their original source. Route
                # through the cluster sync when present so the write
                # replicates incrementally instead of only via snapshot.
                if self._sync is not None:
                    entry = self._sync.remember(
                        key, content, memory_type=mtype, source="tool:remember"
                    )
                else:
                    entry = self._store.remember(
                        key, content, memory_type=mtype, source="tool:remember"
                    )
                return f"Remembered [{entry.memory_type}] {entry.key}: {entry.content}"

            case "forget":
                key = arguments["key"]
                # Route through cluster sync so the removal broadcasts a
                # forget mutation to every node; a raw store.forget() only
                # deletes locally and the entry returns on the next
                # snapshot from a peer that still holds it.
                removed = (
                    self._sync.forget(key)
                    if self._sync is not None
                    else self._store.forget(key)
                )
                if removed:
                    return f"Forgot: {key}"
                return f"No memory found for key: {key}"

            case "recall":
                query = arguments["query"]
                tag = arguments.get("tag") or None
                # fused_search = BM25 + vector (when embeddings extra
                # installed), degrades to BM25 alone otherwise. Same
                # retrieval the prompt-block injection uses, so the
                # LLM-driven recall and the automatic injection don't
                # disagree about what's relevant.
                results = self._store.fused_search(query, limit=5, tag=tag)
                if not results:
                    return "No matching memories found."
                lines = [f"Found {len(results)} memory(ies):"]
                for e in results:
                    lines.append(f"  [{e.memory_type}] {e.key}: {e.content}")
                # Surface the top graph neighbor of the best hit so
                # the agent gets adjacent context even when the query
                # didn't lexically or semantically reach it directly.
                neighbors = self._store.recall_related(results[0].key, limit=2)
                if neighbors:
                    lines.append("Related:")
                    for rel, weight in neighbors:
                        lines.append(
                            f"  [{rel.memory_type}] {rel.key} (co-recall ×{weight}): {rel.content}"
                        )
                return "\n".join(lines)

            case _:
                return f"Unknown tool: {tool_name}"
