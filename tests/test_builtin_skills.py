"""Tests for built-in skills."""

import asyncio
import tempfile
from pathlib import Path

import pytest

from towel.skills.builtin.filesystem import FileSystemSkill
from towel.skills.builtin.shell import ShellSkill
from towel.skills.builtin.web import WebFetchSkill
from towel.skills.builtin import register_builtins
from towel.skills.registry import SkillRegistry


@pytest.fixture
def fs_skill():
    return FileSystemSkill()


@pytest.fixture
def shell_skill():
    return ShellSkill()


class TestFileSystemSkill:
    def test_tools_defined(self, fs_skill):
        tools = fs_skill.tools()
        names = {t.name for t in tools}
        assert names == {"read_file", "write_file", "list_directory"}

    @pytest.mark.asyncio
    async def test_read_file(self, fs_skill, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello towel")
        result = await fs_skill.execute("read_file", {"path": str(f)})
        assert result == "hello towel"

    @pytest.mark.asyncio
    async def test_read_missing_file(self, fs_skill):
        result = await fs_skill.execute("read_file", {"path": "/nonexistent/file.txt"})
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_write_file(self, fs_skill, tmp_path):
        f = tmp_path / "out.txt"
        result = await fs_skill.execute("write_file", {"path": str(f), "content": "42"})
        assert "Written" in result
        assert f.read_text() == "42"

    @pytest.mark.asyncio
    async def test_list_directory(self, fs_skill, tmp_path):
        (tmp_path / "a.txt").touch()
        (tmp_path / "b.txt").touch()
        (tmp_path / "subdir").mkdir()
        result = await fs_skill.execute("list_directory", {"path": str(tmp_path)})
        assert "a.txt" in result
        assert "b.txt" in result
        assert "subdir" in result


class TestShellSkill:
    def test_tools_defined(self, shell_skill):
        tools = shell_skill.tools()
        assert len(tools) == 1
        assert tools[0].name == "run_command"

    @pytest.mark.asyncio
    async def test_echo(self, shell_skill):
        result = await shell_skill.execute("run_command", {"command": "echo hello"})
        assert "hello" in result
        assert "[exit: 0]" in result

    @pytest.mark.asyncio
    async def test_blocked_command(self, shell_skill):
        result = await shell_skill.execute("run_command", {"command": "sudo rm -rf /"})
        assert "Blocked" in result

    @pytest.mark.asyncio
    async def test_timeout(self, shell_skill):
        result = await shell_skill.execute(
            "run_command", {"command": "sleep 10", "timeout": 1}
        )
        assert "timed out" in result.lower()


class TestSystemSkill:
    @pytest.fixture
    def sys_skill(self):
        from towel.skills.builtin.system import SystemSkill
        return SystemSkill()

    def test_tools_defined(self, sys_skill):
        tools = sys_skill.tools()
        names = {t.name for t in tools}
        assert names == {"system_info", "system_processes", "system_disk"}

    @pytest.mark.asyncio
    async def test_system_info(self, sys_skill):
        result = await sys_skill.execute("system_info", {})
        assert "System:" in result
        assert "CPU" in result or "cores" in result.lower()

    @pytest.mark.asyncio
    async def test_system_processes(self, sys_skill):
        result = await sys_skill.execute("system_processes", {"sort_by": "cpu", "limit": 5})
        assert "PID" in result or "pid" in result.lower()

    @pytest.mark.asyncio
    async def test_system_disk(self, sys_skill):
        result = await sys_skill.execute("system_disk", {})
        assert "Filesystem" in result or "filesystem" in result.lower() or "/" in result


class TestTimeSkill:
    @pytest.fixture
    def time_skill(self):
        from towel.skills.builtin.time_skill import TimeSkill
        return TimeSkill()

    def test_tools_defined(self, time_skill):
        tools = time_skill.tools()
        names = {t.name for t in tools}
        assert names == {"current_time", "time_between", "unix_timestamp"}

    @pytest.mark.asyncio
    async def test_current_time_local(self, time_skill):
        result = await time_skill.execute("current_time", {})
        assert "Date:" in result
        assert "Time:" in result
        assert "local" in result

    @pytest.mark.asyncio
    async def test_current_time_utc(self, time_skill):
        result = await time_skill.execute("current_time", {"timezone": "UTC"})
        assert "UTC" in result

    @pytest.mark.asyncio
    async def test_current_time_unknown_tz(self, time_skill):
        result = await time_skill.execute("current_time", {"timezone": "FAKE"})
        assert "Unknown timezone" in result

    @pytest.mark.asyncio
    async def test_time_between(self, time_skill):
        result = await time_skill.execute("time_between", {"start": "2026-01-01", "end": "2026-03-27"})
        assert "85 days" in result

    @pytest.mark.asyncio
    async def test_time_between_invalid(self, time_skill):
        result = await time_skill.execute("time_between", {"start": "nope", "end": "2026-01-01"})
        assert "Invalid date" in result

    @pytest.mark.asyncio
    async def test_unix_timestamp_current(self, time_skill):
        result = await time_skill.execute("unix_timestamp", {})
        assert "Current Unix timestamp" in result

    @pytest.mark.asyncio
    async def test_unix_timestamp_convert(self, time_skill):
        result = await time_skill.execute("unix_timestamp", {"timestamp": 0})
        assert "1970" in result


class TestNetworkSkill:
    @pytest.fixture
    def net_skill(self):
        from towel.skills.builtin.network import NetworkSkill
        return NetworkSkill()

    def test_tools_defined(self, net_skill):
        tools = net_skill.tools()
        names = {t.name for t in tools}
        assert names == {"dns_lookup", "port_check", "http_ping", "whois_lookup"}

    @pytest.mark.asyncio
    async def test_dns_lookup_localhost(self, net_skill):
        result = await net_skill.execute("dns_lookup", {"hostname": "localhost"})
        assert "127.0.0.1" in result or "::1" in result

    @pytest.mark.asyncio
    async def test_dns_lookup_invalid(self, net_skill):
        result = await net_skill.execute("dns_lookup", {"hostname": "this.does.not.exist.invalid"})
        assert "failed" in result.lower()

    @pytest.mark.asyncio
    async def test_port_check_closed(self, net_skill):
        # Port 1 is almost certainly closed
        result = await net_skill.execute("port_check", {"host": "127.0.0.1", "port": 1, "timeout": 1})
        assert "CLOSED" in result or "TIMEOUT" in result

    @pytest.mark.asyncio
    async def test_http_ping(self, net_skill):
        # Ping a reliable host
        result = await net_skill.execute("http_ping", {"url": "https://httpbin.org/status/200", "timeout": 5})
        assert "200" in result or "TIMEOUT" in result or "ERROR" in result


class TestHashSkill:
    @pytest.fixture
    def hash_skill(self):
        from towel.skills.builtin.hash_skill import HashSkill
        return HashSkill()

    def test_tools_defined(self, hash_skill):
        tools = hash_skill.tools()
        names = {t.name for t in tools}
        assert names == {"hash_text", "hash_file", "base64_encode", "base64_decode", "url_encode"}

    @pytest.mark.asyncio
    async def test_hash_sha256(self, hash_skill):
        result = await hash_skill.execute("hash_text", {"text": "hello"})
        assert "sha256" in result
        assert "2cf24dba" in result  # known sha256 prefix for "hello"

    @pytest.mark.asyncio
    async def test_hash_md5(self, hash_skill):
        result = await hash_skill.execute("hash_text", {"text": "hello", "algorithm": "md5"})
        assert "md5" in result
        assert "5d41402a" in result

    @pytest.mark.asyncio
    async def test_hash_file(self, hash_skill, tmp_path):
        f = tmp_path / "test.txt"
        f.write_text("hello")
        result = await hash_skill.execute("hash_file", {"path": str(f)})
        assert "sha256" in result
        assert "test.txt" in result

    @pytest.mark.asyncio
    async def test_hash_file_missing(self, hash_skill):
        result = await hash_skill.execute("hash_file", {"path": "/nonexistent"})
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_base64_roundtrip(self, hash_skill):
        encoded = await hash_skill.execute("base64_encode", {"text": "Don't Panic"})
        assert "Base64:" in encoded
        b64_val = encoded.split(": ", 1)[1]
        decoded = await hash_skill.execute("base64_decode", {"data": b64_val})
        assert "Don't Panic" in decoded

    @pytest.mark.asyncio
    async def test_url_encode(self, hash_skill):
        result = await hash_skill.execute("url_encode", {"text": "hello world&foo=bar"})
        assert "hello%20world" in result

    @pytest.mark.asyncio
    async def test_url_decode(self, hash_skill):
        result = await hash_skill.execute("url_encode", {"text": "hello%20world", "decode": True})
        assert "hello world" in result


class TestEnvSkill:
    @pytest.fixture
    def env_skill(self):
        from towel.skills.builtin.env_skill import EnvSkill
        return EnvSkill()

    def test_tools_defined(self, env_skill):
        tools = env_skill.tools()
        names = {t.name for t in tools}
        assert names == {"env_get", "env_list", "env_path", "env_which"}

    @pytest.mark.asyncio
    async def test_env_get_home(self, env_skill):
        result = await env_skill.execute("env_get", {"name": "HOME"})
        assert "HOME=" in result

    @pytest.mark.asyncio
    async def test_env_get_missing(self, env_skill):
        result = await env_skill.execute("env_get", {"name": "TOWEL_NONEXISTENT_VAR_XYZ"})
        assert "not set" in result

    @pytest.mark.asyncio
    async def test_env_list(self, env_skill):
        result = await env_skill.execute("env_list", {})
        assert "Environment" in result
        assert "HOME" in result

    @pytest.mark.asyncio
    async def test_env_list_prefix(self, env_skill):
        result = await env_skill.execute("env_list", {"prefix": "HOME"})
        assert "HOME" in result

    @pytest.mark.asyncio
    async def test_env_path(self, env_skill):
        result = await env_skill.execute("env_path", {})
        assert "PATH entries" in result
        assert "/usr" in result or "/bin" in result

    @pytest.mark.asyncio
    async def test_env_which_python(self, env_skill):
        result = await env_skill.execute("env_which", {"command": "python3"})
        assert "python3:" in result
        assert "not found" not in result

    @pytest.mark.asyncio
    async def test_env_which_missing(self, env_skill):
        result = await env_skill.execute("env_which", {"command": "nonexistent_binary_xyz"})
        assert "not found" in result

    @pytest.mark.asyncio
    async def test_env_list_redacts_secrets(self, env_skill, monkeypatch):
        monkeypatch.setenv("MY_SECRET_TOKEN", "super-secret-value")
        result = await env_skill.execute("env_list", {"prefix": "MY_SECRET"})
        assert "****" in result
        assert "super-secret-value" not in result


class TestRegexSkill:
    @pytest.fixture
    def rx(self):
        from towel.skills.builtin.regex_skill import RegexSkill
        return RegexSkill()

    def test_tools_defined(self, rx):
        names = {t.name for t in rx.tools()}
        assert names == {"regex_test", "regex_findall", "regex_replace", "regex_split"}

    @pytest.mark.asyncio
    async def test_match(self, rx):
        result = await rx.execute("regex_test", {"pattern": r"\d+", "text": "abc 42 def"})
        assert "42" in result
        assert "Position" in result

    @pytest.mark.asyncio
    async def test_no_match(self, rx):
        result = await rx.execute("regex_test", {"pattern": r"\d+", "text": "no numbers"})
        assert "No match" in result

    @pytest.mark.asyncio
    async def test_groups(self, rx):
        result = await rx.execute("regex_test", {"pattern": r"(\w+)@(\w+)", "text": "user@host"})
        assert "Group 1" in result
        assert "user" in result

    @pytest.mark.asyncio
    async def test_findall(self, rx):
        result = await rx.execute("regex_findall", {"pattern": r"\b\w{3}\b", "text": "the cat sat"})
        assert "3 match" in result
        assert "the" in result

    @pytest.mark.asyncio
    async def test_replace(self, rx):
        result = await rx.execute("regex_replace", {
            "pattern": r"\d+", "replacement": "N", "text": "item 1, item 2, item 3"
        })
        assert "item N, item N, item N" in result
        assert "3 match" in result

    @pytest.mark.asyncio
    async def test_split(self, rx):
        result = await rx.execute("regex_split", {"pattern": r"[,;]\s*", "text": "a, b; c, d"})
        assert "4 part" in result

    @pytest.mark.asyncio
    async def test_invalid_pattern(self, rx):
        result = await rx.execute("regex_test", {"pattern": r"[invalid", "text": "test"})
        assert "Invalid regex" in result

    @pytest.mark.asyncio
    async def test_case_insensitive(self, rx):
        result = await rx.execute("regex_test", {"pattern": "hello", "text": "HELLO", "flags": "i"})
        assert "HELLO" in result


class TestConvertSkill:
    @pytest.fixture
    def conv(self):
        from towel.skills.builtin.convert_skill import ConvertSkill
        return ConvertSkill()

    def test_tools_defined(self, conv):
        names = {t.name for t in conv.tools()}
        assert names == {"convert_units", "list_units"}

    @pytest.mark.asyncio
    async def test_km_to_mi(self, conv):
        result = await conv.execute("convert_units", {"value": 10, "from_unit": "km", "to_unit": "mi"})
        assert "6.21" in result

    @pytest.mark.asyncio
    async def test_f_to_c(self, conv):
        result = await conv.execute("convert_units", {"value": 212, "from_unit": "F", "to_unit": "C"})
        assert "100" in result

    @pytest.mark.asyncio
    async def test_gb_to_mb(self, conv):
        result = await conv.execute("convert_units", {"value": 1, "from_unit": "GB", "to_unit": "MB"})
        assert "1024" in result

    @pytest.mark.asyncio
    async def test_lb_to_kg(self, conv):
        result = await conv.execute("convert_units", {"value": 100, "from_unit": "lb", "to_unit": "kg"})
        assert "45" in result

    @pytest.mark.asyncio
    async def test_cross_category_error(self, conv):
        result = await conv.execute("convert_units", {"value": 1, "from_unit": "km", "to_unit": "kg"})
        assert "Cannot convert" in result

    @pytest.mark.asyncio
    async def test_unknown_unit(self, conv):
        result = await conv.execute("convert_units", {"value": 1, "from_unit": "zorps", "to_unit": "km"})
        assert "Unknown unit" in result

    @pytest.mark.asyncio
    async def test_list_units(self, conv):
        result = await conv.execute("list_units", {})
        assert "Length" in result
        assert "Temperature" in result


class TestRegisterBuiltins:
    def test_registers_all(self):
        reg = SkillRegistry()
        register_builtins(reg)
        names = set(reg.list_skills())
        assert "filesystem" in names
        assert "shell" in names
        assert "web" in names
        assert "convert" in names
        assert "time" in names
        assert "network" in names
        assert "hash" in names
        assert "env" in names
        assert "regex" in names
        assert "system" in names


class TestDiffSkill:
    @pytest.fixture
    def diff_skill(self):
        from towel.skills.builtin.diff_skill import DiffSkill
        return DiffSkill()

    def test_tools_defined(self, diff_skill):
        names = {t.name for t in diff_skill.tools()}
        assert names == {"diff_files", "diff_text", "diff_stats"}

    @pytest.mark.asyncio
    async def test_diff_files_identical(self, diff_skill, tmp_path):
        a = tmp_path / "a.txt"; b = tmp_path / "b.txt"
        a.write_text("hello\n"); b.write_text("hello\n")
        result = await diff_skill.execute("diff_files", {"file_a": str(a), "file_b": str(b)})
        assert "identical" in result.lower()

    @pytest.mark.asyncio
    async def test_diff_files_different(self, diff_skill, tmp_path):
        a = tmp_path / "a.txt"; b = tmp_path / "b.txt"
        a.write_text("line1\nline2\n"); b.write_text("line1\nchanged\n")
        result = await diff_skill.execute("diff_files", {"file_a": str(a), "file_b": str(b)})
        assert "-line2" in result
        assert "+changed" in result

    @pytest.mark.asyncio
    async def test_diff_files_missing(self, diff_skill):
        result = await diff_skill.execute("diff_files", {"file_a": "/nonexistent", "file_b": "/also-no"})
        assert "not found" in result.lower()

    @pytest.mark.asyncio
    async def test_diff_text(self, diff_skill):
        result = await diff_skill.execute("diff_text", {"text_a": "foo\nbar", "text_b": "foo\nbaz"})
        assert "-bar" in result
        assert "+baz" in result

    @pytest.mark.asyncio
    async def test_diff_stats(self, diff_skill, tmp_path):
        a = tmp_path / "a.txt"; b = tmp_path / "b.txt"
        a.write_text("a\nb\nc\n"); b.write_text("a\nx\nc\nd\n")
        result = await diff_skill.execute("diff_stats", {"file_a": str(a), "file_b": str(b)})
        assert "Similarity" in result
        assert "Additions" in result


class TestArchiveSkill:
    @pytest.fixture
    def arc(self):
        from towel.skills.builtin.archive_skill import ArchiveSkill
        return ArchiveSkill()

    def test_tools_defined(self, arc):
        names = {t.name for t in arc.tools()}
        assert names == {"archive_list", "archive_create", "archive_extract"}

    @pytest.mark.asyncio
    async def test_create_and_list(self, arc, tmp_path):
        (tmp_path / "a.txt").write_text("aaa")
        (tmp_path / "b.txt").write_text("bbb")
        out = str(tmp_path / "test.zip")
        result = await arc.execute("archive_create", {
            "output": out, "sources": [str(tmp_path / "a.txt"), str(tmp_path / "b.txt")]
        })
        assert "Created" in result
        assert "2 files" in result

        ls = await arc.execute("archive_list", {"path": out})
        assert "a.txt" in ls
        assert "b.txt" in ls

    @pytest.mark.asyncio
    async def test_create_directory(self, arc, tmp_path):
        d = tmp_path / "mydir"; d.mkdir()
        (d / "x.txt").write_text("x")
        (d / "y.txt").write_text("y")
        out = str(tmp_path / "dir.zip")
        result = await arc.execute("archive_create", {"output": out, "sources": [str(d)]})
        assert "2 files" in result

    @pytest.mark.asyncio
    async def test_extract(self, arc, tmp_path):
        src = tmp_path / "f.txt"; src.write_text("content")
        zp = str(tmp_path / "e.zip")
        await arc.execute("archive_create", {"output": zp, "sources": [str(src)]})
        dest = tmp_path / "out"
        result = await arc.execute("archive_extract", {"path": zp, "dest": str(dest)})
        assert "Extracted" in result
        assert (dest / "f.txt").read_text() == "content"

    @pytest.mark.asyncio
    async def test_list_not_found(self, arc):
        result = await arc.execute("archive_list", {"path": "/nonexistent.zip"})
        assert "Not found" in result


class TestCronSkill:
    @pytest.fixture
    def cron(self):
        from towel.skills.builtin.cron_skill import CronSkill
        return CronSkill()

    def test_tools_defined(self, cron):
        names = {t.name for t in cron.tools()}
        assert names == {"cron_explain", "cron_next", "cron_build"}

    @pytest.mark.asyncio
    async def test_explain_every_5(self, cron):
        result = await cron.execute("cron_explain", {"expression": "*/5 * * * *"})
        assert "5 minutes" in result

    @pytest.mark.asyncio
    async def test_explain_daily(self, cron):
        result = await cron.execute("cron_explain", {"expression": "0 9 * * *"})
        assert "9:00" in result

    @pytest.mark.asyncio
    async def test_explain_weekdays(self, cron):
        result = await cron.execute("cron_explain", {"expression": "30 8 * * 1-5"})
        assert "Mon" in result or "1" in result

    @pytest.mark.asyncio
    async def test_next_runs(self, cron):
        result = await cron.execute("cron_next", {"expression": "*/5 * * * *", "count": 3})
        assert "Next runs" in result
        lines = [l for l in result.splitlines() if l.strip().startswith("20")]
        assert len(lines) == 3

    @pytest.mark.asyncio
    async def test_build_every_5(self, cron):
        result = await cron.execute("cron_build", {"description": "every 5 minutes"})
        assert "*/5" in result

    @pytest.mark.asyncio
    async def test_build_daily(self, cron):
        result = await cron.execute("cron_build", {"description": "daily at 9am"})
        assert "0 9" in result

    @pytest.mark.asyncio
    async def test_build_weekdays(self, cron):
        result = await cron.execute("cron_build", {"description": "weekdays at noon"})
        assert "1-5" in result

    @pytest.mark.asyncio
    async def test_invalid(self, cron):
        result = await cron.execute("cron_explain", {"expression": "bad"})
        assert "Invalid" in result


class TestMarkdownSkill:
    @pytest.fixture
    def md(self):
        from towel.skills.builtin.markdown_skill import MarkdownSkill
        return MarkdownSkill()

    def test_tools_defined(self, md):
        names = {t.name for t in md.tools()}
        assert names == {"md_table", "md_toc", "md_checklist", "json_to_md"}

    @pytest.mark.asyncio
    async def test_table_from_json(self, md):
        data = '[{"name": "Alice", "age": 30}, {"name": "Bob", "age": 25}]'
        result = await md.execute("md_table", {"data": data})
        assert "| name | age |" in result
        assert "Alice" in result

    @pytest.mark.asyncio
    async def test_table_from_csv(self, md):
        data = "name,score\nAlice,95\nBob,87"
        result = await md.execute("md_table", {"data": data})
        assert "| name | score |" in result

    @pytest.mark.asyncio
    async def test_toc(self, md):
        text = "# Intro\n## Setup\n### Config\n## Usage"
        result = await md.execute("md_toc", {"markdown": text})
        assert "Table of Contents" in result
        assert "[Setup]" in result

    @pytest.mark.asyncio
    async def test_checklist(self, md):
        result = await md.execute("md_checklist", {"items": ["a", "b", "c"], "checked": [1]})
        assert "- [ ] a" in result
        assert "- [x] b" in result
        assert "- [ ] c" in result


class TestHttpSkill:
    @pytest.fixture
    def http(self):
        from towel.skills.builtin.http_skill import HttpSkill
        return HttpSkill()

    def test_tools_defined(self, http):
        names = {t.name for t in http.tools()}
        assert names == {"http_request", "http_head"}

    @pytest.mark.asyncio
    async def test_get(self, http):
        result = await http.execute("http_request", {"url": "https://httpbin.org/get", "timeout": 5})
        assert "200" in result or "TIMEOUT" in result or "Error" in result

    @pytest.mark.asyncio
    async def test_head(self, http):
        result = await http.execute("http_head", {"url": "https://httpbin.org/get"})
        assert "200" in result or "Error" in result


class TestHeartbeat:
    def test_heartbeat_lifecycle(self):
        from towel.agent.heartbeat import Heartbeat
        hb = Heartbeat(interval=0.1)
        hb.start()
        hb.on_model_loaded()
        hb.on_generation_start()
        hb.on_generation_complete()
        status = hb.status()
        assert status.alive
        assert status.model_loaded
        assert status.total_generations == 1
        hb.stop()

    def test_heartbeat_errors(self):
        from towel.agent.heartbeat import Heartbeat
        triggered = []
        hb = Heartbeat(interval=60, max_consecutive_errors=3,
                       on_unhealthy=lambda s: triggered.append(True))
        hb.start()
        for _ in range(3):
            hb.on_error(RuntimeError("test"))
        status = hb.status()
        assert not status.alive
        assert status.consecutive_errors == 3
        assert len(triggered) == 1
        hb.stop()

    def test_heartbeat_error_reset(self):
        from towel.agent.heartbeat import Heartbeat
        hb = Heartbeat(interval=60, max_consecutive_errors=5)
        hb.start()
        hb.on_error(RuntimeError("e1"))
        hb.on_error(RuntimeError("e2"))
        assert hb.status().consecutive_errors == 2
        hb.on_generation_complete()  # resets consecutive
        assert hb.status().consecutive_errors == 0
        assert hb.status().total_errors == 2  # total preserved
        hb.stop()


class TestSqlSkill:
    @pytest.fixture
    def sql(self):
        from towel.skills.builtin.sql_skill import SqlSkill
        return SqlSkill()

    @pytest.fixture
    def db(self, tmp_path):
        import sqlite3
        path = tmp_path / "test.db"
        conn = sqlite3.connect(str(path))
        conn.execute("CREATE TABLE users (id INTEGER PRIMARY KEY, name TEXT, age INTEGER)")
        conn.execute("INSERT INTO users VALUES (1, 'Alice', 30)")
        conn.execute("INSERT INTO users VALUES (2, 'Bob', 25)")
        conn.execute("INSERT INTO users VALUES (3, 'Charlie', 35)")
        conn.commit()
        conn.close()
        return str(path)

    def test_tools_defined(self, sql):
        names = {t.name for t in sql.tools()}
        assert names == {"sql_query", "sql_schema", "sql_explain"}

    @pytest.mark.asyncio
    async def test_query(self, sql, db):
        result = await sql.execute("sql_query", {"database": db, "query": "SELECT * FROM users"})
        assert "Alice" in result
        assert "Bob" in result
        assert "3 row" in result

    @pytest.mark.asyncio
    async def test_query_where(self, sql, db):
        result = await sql.execute("sql_query", {"database": db, "query": "SELECT name FROM users WHERE age > 28"})
        assert "Alice" in result
        assert "Charlie" in result
        assert "Bob" not in result

    @pytest.mark.asyncio
    async def test_schema(self, sql, db):
        result = await sql.execute("sql_schema", {"database": db})
        assert "users" in result
        assert "3 cols" in result

    @pytest.mark.asyncio
    async def test_schema_table(self, sql, db):
        result = await sql.execute("sql_schema", {"database": db, "table": "users"})
        assert "name" in result
        assert "INTEGER" in result

    @pytest.mark.asyncio
    async def test_explain(self, sql, db):
        result = await sql.execute("sql_explain", {"database": db, "query": "SELECT * FROM users"})
        assert "plan" in result.lower() or "SCAN" in result

    @pytest.mark.asyncio
    async def test_readonly(self, sql, db):
        result = await sql.execute("sql_query", {"database": db, "query": "DROP TABLE users"})
        assert "only SELECT" in result.lower() or "Only SELECT" in result

    @pytest.mark.asyncio
    async def test_not_found(self, sql):
        result = await sql.execute("sql_query", {"database": "/nonexistent.db", "query": "SELECT 1"})
        assert "not found" in result.lower()


class TestImageSkill:
    @pytest.fixture
    def img(self):
        from towel.skills.builtin.image_skill import ImageSkill
        return ImageSkill()

    def test_tools_defined(self, img):
        names = {t.name for t in img.tools()}
        assert names == {"image_info"}

    @pytest.mark.asyncio
    async def test_png(self, img, tmp_path):
        # Minimal valid PNG: 1x1 pixel
        import struct
        png = b'\x89PNG\r\n\x1a\n'
        # IHDR chunk
        ihdr_data = struct.pack('>IIBBBBB', 100, 200, 8, 2, 0, 0, 0)
        import zlib
        crc = struct.pack('>I', zlib.crc32(b'IHDR' + ihdr_data) & 0xffffffff)
        ihdr = struct.pack('>I', len(ihdr_data)) + b'IHDR' + ihdr_data + crc
        f = tmp_path / "test.png"
        f.write_bytes(png + ihdr)
        result = await img.execute("image_info", {"path": str(f)})
        assert "100x200" in result
        assert "PNG" in result

    @pytest.mark.asyncio
    async def test_not_found(self, img):
        result = await img.execute("image_info", {"path": "/nonexistent.png"})
        assert "Not found" in result


class TestProcessSkill:
    @pytest.fixture
    def proc(self):
        from towel.skills.builtin.process_skill import ProcessSkill
        return ProcessSkill()

    def test_tools_defined(self, proc):
        names = {t.name for t in proc.tools()}
        assert names == {"process_find", "process_info", "process_tree", "process_ports"}

    @pytest.mark.asyncio
    async def test_find(self, proc):
        import os
        result = await proc.execute("process_find", {"name": "python"})
        assert "python" in result.lower() or "No processes" in result

    @pytest.mark.asyncio
    async def test_info_self(self, proc):
        import os
        result = await proc.execute("process_info", {"pid": os.getpid()})
        assert "python" in result.lower() or str(os.getpid()) in result


class TestTextSkill:
    @pytest.fixture
    def txt(self):
        from towel.skills.builtin.text_skill import TextSkill
        return TextSkill()

    def test_tools_defined(self, txt):
        names = {t.name for t in txt.tools()}
        assert names == {"text_stats", "text_transform", "text_frequency"}

    @pytest.mark.asyncio
    async def test_stats(self, txt):
        result = await txt.execute("text_stats", {"text": "Hello world. This is a test."})
        assert "Words: 6" in result
        assert "Sentences: 2" in result

    @pytest.mark.asyncio
    async def test_transform_upper(self, txt):
        result = await txt.execute("text_transform", {"text": "hello", "transform": "upper"})
        assert result == "HELLO"

    @pytest.mark.asyncio
    async def test_transform_snake(self, txt):
        result = await txt.execute("text_transform", {"text": "helloWorld", "transform": "snake"})
        assert result == "hello_world"

    @pytest.mark.asyncio
    async def test_transform_camel(self, txt):
        result = await txt.execute("text_transform", {"text": "hello_world", "transform": "camel"})
        assert result == "helloWorld"

    @pytest.mark.asyncio
    async def test_transform_number_lines(self, txt):
        result = await txt.execute("text_transform", {"text": "a\nb\nc", "transform": "number_lines"})
        assert "   1  a" in result
        assert "   3  c" in result

    @pytest.mark.asyncio
    async def test_frequency(self, txt):
        result = await txt.execute("text_frequency", {"text": "the cat sat on the mat the cat"})
        assert "the" in result
        assert "cat" in result

    @pytest.mark.asyncio
    async def test_unique_lines(self, txt):
        result = await txt.execute("text_transform", {"text": "a\nb\na\nc\nb", "transform": "unique_lines"})
        assert result == "a\nb\nc"


class TestKnowledgeSkill:
    @pytest.fixture
    def kb(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.skills.builtin.knowledge_skill.KB_FILE", tmp_path / "kb.json")
        from towel.skills.builtin.knowledge_skill import KnowledgeSkill
        return KnowledgeSkill()

    @pytest.mark.asyncio
    async def test_add_and_search(self, kb):
        await kb.execute("kb_add", {"content": "Python uses indentation", "tags": ["python", "syntax"]})
        result = await kb.execute("kb_search", {"query": "indentation"})
        assert "indentation" in result

    @pytest.mark.asyncio
    async def test_list(self, kb):
        await kb.execute("kb_add", {"content": "note 1"})
        await kb.execute("kb_add", {"content": "note 2"})
        result = await kb.execute("kb_list", {})
        assert "2 entries" in result

    @pytest.mark.asyncio
    async def test_delete(self, kb):
        await kb.execute("kb_add", {"content": "temp note"})
        result = await kb.execute("kb_delete", {"index": 0})
        assert "Deleted" in result


class TestTranslateSkill:
    @pytest.fixture
    def tr(self):
        from towel.skills.builtin.translate_skill import TranslateSkill
        return TranslateSkill()

    @pytest.mark.asyncio
    async def test_detect_spanish(self, tr):
        result = await tr.execute("detect_language", {"text": "el gato está en la mesa de la cocina"})
        assert "Spanish" in result

    @pytest.mark.asyncio
    async def test_translation_prompt(self, tr):
        result = await tr.execute("translation_prompt", {"text": "Hello world", "target_language": "French"})
        assert "French" in result
        assert "Hello world" in result


class TestSecuritySkill:
    @pytest.fixture
    def sec(self):
        from towel.skills.builtin.security_skill import SecuritySkill
        return SecuritySkill()

    @pytest.mark.asyncio
    async def test_scan_clean(self, sec, tmp_path):
        (tmp_path / "clean.py").write_text("x = 42\n")
        result = await sec.execute("scan_secrets", {"path": str(tmp_path)})
        assert "No secrets" in result

    @pytest.mark.asyncio
    async def test_scan_finds_key(self, sec, tmp_path):
        (tmp_path / "bad.py").write_text('API_KEY = "sk-abc123def456ghi789jkl012mno345pqr"\n')
        result = await sec.execute("scan_secrets", {"path": str(tmp_path)})
        assert "API key" in result or "secret" in result.lower() or "OpenAI" in result

    @pytest.mark.asyncio
    async def test_deps_scan(self, sec, tmp_path):
        (tmp_path / "requirements.txt").write_text("flask\nrequests\nnumpy==1.24.0\n")
        result = await sec.execute("scan_dependencies", {"path": str(tmp_path)})
        assert "unpinned" in result


class TestTodoSkill:
    @pytest.fixture
    def todo(self, tmp_path, monkeypatch):
        monkeypatch.setattr("towel.skills.builtin.todo_skill.TODO_FILE", tmp_path / "todos.json")
        from towel.skills.builtin.todo_skill import TodoSkill
        return TodoSkill()

    @pytest.mark.asyncio
    async def test_add_and_list(self, todo):
        await todo.execute("todo_add", {"task": "Buy milk", "priority": "high"})
        await todo.execute("todo_add", {"task": "Read docs", "priority": "low"})
        result = await todo.execute("todo_list", {})
        assert "Buy milk" in result
        assert "Read docs" in result
        assert "2 todo" in result

    @pytest.mark.asyncio
    async def test_complete(self, todo):
        await todo.execute("todo_add", {"task": "Finish PR"})
        result = await todo.execute("todo_done", {"index": 0})
        assert "Completed" in result

    @pytest.mark.asyncio
    async def test_remove(self, todo):
        await todo.execute("todo_add", {"task": "Temp task"})
        result = await todo.execute("todo_remove", {"index": 0})
        assert "Removed" in result


class TestTemplateGenSkill:
    @pytest.fixture
    def scaffold(self):
        from towel.skills.builtin.template_gen_skill import TemplateGenSkill
        return TemplateGenSkill()

    @pytest.mark.asyncio
    async def test_list(self, scaffold):
        result = await scaffold.execute("scaffold_list", {})
        assert "python-script" in result
        assert "dockerfile" in result

    @pytest.mark.asyncio
    async def test_generate(self, scaffold, tmp_path):
        result = await scaffold.execute("scaffold_generate", {
            "template": "readme", "name": "myapp", "description": "A cool app",
            "output_dir": str(tmp_path),
        })
        assert "Created" in result
        content = (tmp_path / "README.md").read_text()
        assert "myapp" in content
        assert "A cool app" in content

    @pytest.mark.asyncio
    async def test_unknown_template(self, scaffold):
        result = await scaffold.execute("scaffold_generate", {"template": "nonexistent"})
        assert "Unknown template" in result


class TestMathSkill:
    @pytest.fixture
    def m(self):
        from towel.skills.builtin.math_skill import MathSkill
        return MathSkill()

    @pytest.mark.asyncio
    async def test_stats(self, m):
        result = await m.execute("math_stats", {"numbers": [1, 2, 3, 4, 5]})
        assert "Mean:" in result
        assert "3" in result

    @pytest.mark.asyncio
    async def test_format_bytes(self, m):
        result = await m.execute("math_format", {"number": 1536, "format": "bytes"})
        assert "1.5 KB" in result

    @pytest.mark.asyncio
    async def test_format_roman(self, m):
        result = await m.execute("math_format", {"number": 42, "format": "roman"})
        assert result == "XLII"

    @pytest.mark.asyncio
    async def test_fibonacci(self, m):
        result = await m.execute("math_sequence", {"type": "fibonacci", "count": 8})
        assert "0, 1, 1, 2, 3, 5, 8, 13" == result

    @pytest.mark.asyncio
    async def test_primes(self, m):
        result = await m.execute("math_sequence", {"type": "primes", "count": 5})
        assert "2, 3, 5, 7, 11" == result


class TestDockerSkill:
    @pytest.fixture
    def dock(self):
        from towel.skills.builtin.docker_skill import DockerSkill
        return DockerSkill()

    def test_tools_defined(self, dock):
        names = {t.name for t in dock.tools()}
        assert names == {"docker_ps", "docker_images", "docker_logs", "docker_inspect", "docker_stats"}

    @pytest.mark.asyncio
    async def test_ps(self, dock):
        result = await dock.execute("docker_ps", {})
        # Either shows containers or says docker isn't running
        assert "NAMES" in result or "Docker" in result or "not" in result.lower()


class TestCalendarSkill:
    @pytest.fixture
    def cal(self):
        from towel.skills.builtin.calendar_skill import CalendarSkill
        return CalendarSkill()

    def test_tools_defined(self, cal):
        names = {t.name for t in cal.tools()}
        assert names == {"cal_month", "cal_business_days", "cal_add_days", "cal_countdown"}

    @pytest.mark.asyncio
    async def test_month(self, cal):
        result = await cal.execute("cal_month", {"year": 2026, "month": 1})
        assert "January 2026" in result
        assert "Mo" in result or "Mon" in result

    @pytest.mark.asyncio
    async def test_business_days(self, cal):
        result = await cal.execute("cal_business_days", {"start": "2026-03-23", "end": "2026-03-27"})
        assert "Business days: 5" in result

    @pytest.mark.asyncio
    async def test_add_days(self, cal):
        result = await cal.execute("cal_add_days", {"date": "2026-01-01", "days": 10})
        assert "2026-01-11" in result

    @pytest.mark.asyncio
    async def test_countdown(self, cal):
        result = await cal.execute("cal_countdown", {"target": "2030-01-01", "label": "New Decade"})
        assert "New Decade" in result
        assert "days" in result


class TestQrSkill:
    @pytest.fixture
    def qr(self):
        from towel.skills.builtin.qr_skill import QrSkill
        return QrSkill()

    def test_tools_defined(self, qr):
        names = {t.name for t in qr.tools()}
        assert names == {"qr_generate"}

    @pytest.mark.asyncio
    async def test_generate(self, qr):
        result = await qr.execute("qr_generate", {"data": "https://towel.dev"})
        assert "█" in result or "▀" in result
        assert "towel.dev" in result
