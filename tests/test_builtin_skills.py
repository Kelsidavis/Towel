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


class TestRegisterBuiltins:
    def test_registers_all(self):
        reg = SkillRegistry()
        register_builtins(reg)
        names = set(reg.list_skills())
        assert "filesystem" in names
        assert "shell" in names
        assert "web" in names
        assert "system" in names
