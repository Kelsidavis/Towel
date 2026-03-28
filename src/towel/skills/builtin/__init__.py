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
from towel.skills.builtin.time_skill import TimeSkill
from towel.skills.builtin.network import NetworkSkill
from towel.skills.builtin.hash_skill import HashSkill
from towel.skills.builtin.env_skill import EnvSkill
from towel.skills.builtin.regex_skill import RegexSkill
from towel.skills.builtin.convert_skill import ConvertSkill
from towel.skills.builtin.json_skill import JsonSkill
from towel.skills.builtin.diff_skill import DiffSkill
from towel.skills.builtin.archive_skill import ArchiveSkill
from towel.skills.builtin.cron_skill import CronSkill
from towel.skills.builtin.markdown_skill import MarkdownSkill
from towel.skills.builtin.http_skill import HttpSkill
from towel.skills.builtin.sql_skill import SqlSkill
from towel.skills.builtin.image_skill import ImageSkill
from towel.skills.builtin.process_skill import ProcessSkill
from towel.skills.builtin.text_skill import TextSkill
from towel.skills.builtin.knowledge_skill import KnowledgeSkill
from towel.skills.builtin.translate_skill import TranslateSkill
from towel.skills.builtin.security_skill import SecuritySkill
from towel.skills.builtin.todo_skill import TodoSkill
from towel.skills.builtin.template_gen_skill import TemplateGenSkill
from towel.skills.builtin.math_skill import MathSkill
from towel.skills.builtin.docker_skill import DockerSkill
from towel.skills.builtin.calendar_skill import CalendarSkill
from towel.skills.builtin.qr_skill import QrSkill

__all__ = [
    "FileSystemSkill", "ShellSkill", "WebFetchSkill", "MemorySkill",
    "GitSkill", "SearchSkill", "ClipboardSkill", "DataSkill", "SystemSkill",
    "TimeSkill", "NetworkSkill", "HashSkill", "EnvSkill", "RegexSkill",
    "ConvertSkill", "JsonSkill", "DiffSkill", "ArchiveSkill", "CronSkill", "MarkdownSkill", "HttpSkill", "SqlSkill", "ImageSkill", "ProcessSkill", "TextSkill", "KnowledgeSkill", "TranslateSkill", "SecuritySkill", "TodoSkill", "TemplateGenSkill", "MathSkill", "DockerSkill", "CalendarSkill", "QrSkill",
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
    registry.register(TimeSkill())
    registry.register(NetworkSkill())
    registry.register(HashSkill())
    registry.register(EnvSkill())
    registry.register(RegexSkill())
    registry.register(ConvertSkill())
    registry.register(JsonSkill())
    registry.register(DiffSkill())
    registry.register(ArchiveSkill())
    registry.register(CronSkill())
    registry.register(MarkdownSkill())
    registry.register(HttpSkill())
    registry.register(SqlSkill())
    registry.register(ImageSkill())
    registry.register(ProcessSkill())
    registry.register(TextSkill())
    registry.register(KnowledgeSkill())
    registry.register(TranslateSkill())
    registry.register(SecuritySkill())
    registry.register(TodoSkill())
    registry.register(TemplateGenSkill())
    registry.register(MathSkill())
    registry.register(DockerSkill())
    registry.register(CalendarSkill())
    registry.register(QrSkill())
