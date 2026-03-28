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
from towel.skills.builtin.jwt_skill import JwtSkill
from towel.skills.builtin.color_skill import ColorSkill
from towel.skills.builtin.uuid_skill import UuidSkill
from towel.skills.builtin.yaml_skill import YamlSkill
from towel.skills.builtin.snippet_gen_skill import SnippetGenSkill
from towel.skills.builtin.csv_skill import CsvSkill
from towel.skills.builtin.semver_skill import SemverSkill
from towel.skills.builtin.ip_calc_skill import IpCalcSkill
from towel.skills.builtin.dotenv_skill import DotenvSkill
from towel.skills.builtin.log_analyzer_skill import LogAnalyzerSkill
from towel.skills.builtin.http_header_skill import HttpHeaderSkill
from towel.skills.builtin.ascii_skill import AsciiSkill
from towel.skills.builtin.string_skill import StringSkill
from towel.skills.builtin.ssh_skill import SshSkill
from towel.skills.builtin.npm_skill import NpmSkill
from towel.skills.builtin.pip_skill import PipSkill
from towel.skills.builtin.metrics_skill import MetricsSkill
from towel.skills.builtin.pdf_skill import PdfSkill
from towel.skills.builtin.placeholder_skill import PlaceholderSkill
from towel.skills.builtin.webhook_trigger_skill import WebhookTriggerSkill
from towel.skills.builtin.gitignore_skill import GitignoreSkill
from towel.skills.builtin.lint_skill import LintSkill
from towel.skills.builtin.diagram_skill import DiagramSkill
from towel.skills.builtin.changelog_gen_skill import ChangelogGenSkill
from towel.skills.builtin.note_skill import NoteSkill
from towel.skills.builtin.clipboard_history_skill import ClipboardHistorySkill
from towel.skills.builtin.crontab_skill import CrontabSkill
from towel.skills.builtin.bookmark_skill import BookmarkSkill
from towel.skills.builtin.keychain_skill import KeychainSkill
from towel.skills.builtin.openapi_skill import OpenApiSkill
from towel.skills.builtin.typo_skill import TypoSkill
from towel.skills.builtin.make_skill import MakeSkill
from towel.skills.builtin.man_skill import ManSkill
from towel.skills.builtin.github_skill import GithubSkill
from towel.skills.builtin.pypi_skill import PypiSkill
from towel.skills.builtin.cert_skill import CertSkill
from towel.skills.builtin.whois_skill import WhoisSkill
from towel.skills.builtin.dns_skill import DnsSkill
from towel.skills.builtin.stackoverflow_skill import StackOverflowSkill
from towel.skills.builtin.reddit_skill import RedditSkill
from towel.skills.builtin.currency_skill import CurrencySkill
from towel.skills.builtin.hackernews_skill import HackerNewsSkill
from towel.skills.builtin.wikipedia_skill import WikipediaSkill
from towel.skills.builtin.weather_skill import WeatherSkill

__all__ = [
    "FileSystemSkill", "ShellSkill", "WebFetchSkill", "MemorySkill",
    "GitSkill", "SearchSkill", "ClipboardSkill", "DataSkill", "SystemSkill",
    "TimeSkill", "NetworkSkill", "HashSkill", "EnvSkill", "RegexSkill",
    "ConvertSkill", "JsonSkill", "DiffSkill", "ArchiveSkill", "CronSkill", "MarkdownSkill", "HttpSkill", "SqlSkill", "ImageSkill", "ProcessSkill", "TextSkill", "KnowledgeSkill", "TranslateSkill", "SecuritySkill", "TodoSkill", "TemplateGenSkill", "MathSkill", "DockerSkill", "CalendarSkill", "QrSkill", "JwtSkill", "ColorSkill", "UuidSkill", "YamlSkill", "SnippetGenSkill", "CsvSkill", "SemverSkill", "IpCalcSkill", "DotenvSkill", "LogAnalyzerSkill", "HttpHeaderSkill", "AsciiSkill", "StringSkill", "SshSkill", "NpmSkill", "PipSkill", "MetricsSkill", "PdfSkill", "PlaceholderSkill", "WebhookTriggerSkill", "GitignoreSkill", "LintSkill", "DiagramSkill", "ChangelogGenSkill", "NoteSkill", "ClipboardHistorySkill", "CrontabSkill", "BookmarkSkill", "KeychainSkill", "OpenApiSkill", "TypoSkill", "MakeSkill", "ManSkill", "GithubSkill", "PypiSkill", "CertSkill", "WhoisSkill", "DnsSkill", "StackOverflowSkill", "RedditSkill", "CurrencySkill", "HackerNewsSkill", "WikipediaSkill", "WeatherSkill",
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
    registry.register(JwtSkill())
    registry.register(ColorSkill())
    registry.register(UuidSkill())
    registry.register(YamlSkill())
    registry.register(SnippetGenSkill())
    registry.register(CsvSkill())
    registry.register(SemverSkill())
    registry.register(IpCalcSkill())
    registry.register(DotenvSkill())
    registry.register(LogAnalyzerSkill())
    registry.register(HttpHeaderSkill())
    registry.register(AsciiSkill())
    registry.register(StringSkill())
    registry.register(SshSkill())
    registry.register(NpmSkill())
    registry.register(PipSkill())
    registry.register(MetricsSkill())
    registry.register(PdfSkill())
    registry.register(PlaceholderSkill())
    registry.register(WebhookTriggerSkill())
    registry.register(GitignoreSkill())
    registry.register(LintSkill())
    registry.register(DiagramSkill())
    registry.register(ChangelogGenSkill())
    registry.register(NoteSkill())
    registry.register(ClipboardHistorySkill())
    registry.register(CrontabSkill())
    registry.register(BookmarkSkill())
    registry.register(KeychainSkill())
    registry.register(OpenApiSkill())
    registry.register(TypoSkill())
    registry.register(MakeSkill())
    registry.register(ManSkill())
    registry.register(GithubSkill())
    registry.register(PypiSkill())
    registry.register(CertSkill())
    registry.register(WhoisSkill())
    registry.register(DnsSkill())
    registry.register(StackOverflowSkill())
    registry.register(RedditSkill())
    registry.register(CurrencySkill())
    registry.register(HackerNewsSkill())
    registry.register(WikipediaSkill())
    registry.register(WeatherSkill())
