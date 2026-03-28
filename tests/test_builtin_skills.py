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


class TestRegisterBuiltins:
    def test_registers_all(self):
        reg = SkillRegistry()
        register_builtins(reg)
        names = set(reg.list_skills())
        assert "filesystem" in names
        assert "shell" in names
        assert "web" in names
        assert "time" in names
        assert "network" in names
        assert "hash" in names
        assert "env" in names
        assert "system" in names
