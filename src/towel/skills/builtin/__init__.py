"""Built-in skills that ship with Towel."""

from towel.skills.builtin.filesystem import FileSystemSkill
from towel.skills.builtin.shell import ShellSkill
from towel.skills.builtin.web import WebFetchSkill
from towel.skills.builtin.memory_skill import MemorySkill
from towel.skills.builtin.git import GitSkill
from towel.skills.builtin.search import SearchSkill
from towel.skills.builtin.clipboard import ClipboardSkill
from towel.skills.builtin.data import DataSkill
from towel.skills.builtin.system import SystemSkill

__all__ = [
    "FileSystemSkill", "ShellSkill", "WebFetchSkill", "MemorySkill",
    "GitSkill", "SearchSkill", "ClipboardSkill", "DataSkill", "SystemSkill",
]


def register_builtins(
    registry: "towel.skills.registry.SkillRegistry",
    memory_store: "towel.memory.store.MemoryStore | None" = None,
) -> None:
    """Register all built-in skills."""
    from towel.skills.registry import SkillRegistry

    registry.register(FileSystemSkill())
    registry.register(ShellSkill())
    registry.register(WebFetchSkill())
    registry.register(MemorySkill(store=memory_store))
    registry.register(GitSkill())
    registry.register(SearchSkill())
    registry.register(ClipboardSkill())
    registry.register(DataSkill())
    registry.register(SystemSkill())
