"""Tests for the git skill."""

import os
from pathlib import Path

import pytest

from towel.skills.builtin.git import GitSkill


@pytest.fixture
def git(tmp_path):
    """Create a temporary git repo for testing."""
    os.system(f"git init {tmp_path} > /dev/null 2>&1")
    os.system(f"git -C {tmp_path} config user.email test@test.com")
    os.system(f"git -C {tmp_path} config user.name Test")
    # Initial commit
    (tmp_path / "README.md").write_text("# Test")
    os.system(f"git -C {tmp_path} add -A > /dev/null 2>&1")
    os.system(f'git -C {tmp_path} commit -m "init" > /dev/null 2>&1')
    return GitSkill(), str(tmp_path)


class TestGitSkillTools:
    def test_tools_defined(self):
        s = GitSkill()
        names = {t.name for t in s.tools()}
        assert names == {"git_status", "git_diff", "git_log", "git_commit", "git_branch"}

    def test_name(self):
        assert GitSkill().name == "git"


class TestGitStatus:
    @pytest.mark.asyncio
    async def test_clean_repo(self, git):
        skill, path = git
        result = await skill.execute("git_status", {"path": path})
        assert "clean" in result.lower()

    @pytest.mark.asyncio
    async def test_unstaged_changes(self, git):
        skill, path = git
        (Path(path) / "README.md").write_text("# Modified")
        result = await skill.execute("git_status", {"path": path})
        assert "Unstaged" in result
        assert "README.md" in result

    @pytest.mark.asyncio
    async def test_untracked_files(self, git):
        skill, path = git
        (Path(path) / "new.txt").write_text("new")
        result = await skill.execute("git_status", {"path": path})
        assert "Untracked" in result
        assert "new.txt" in result

    @pytest.mark.asyncio
    async def test_shows_branch(self, git):
        skill, path = git
        result = await skill.execute("git_status", {"path": path})
        assert "Branch:" in result


class TestGitDiff:
    @pytest.mark.asyncio
    async def test_no_changes(self, git):
        skill, path = git
        result = await skill.execute("git_diff", {"path": path})
        assert "No changes" in result

    @pytest.mark.asyncio
    async def test_unstaged_diff(self, git):
        skill, path = git
        (Path(path) / "README.md").write_text("# Changed\nNew line")
        result = await skill.execute("git_diff", {"path": path})
        assert "Changed" in result or "+" in result

    @pytest.mark.asyncio
    async def test_staged_diff(self, git):
        skill, path = git
        (Path(path) / "README.md").write_text("# Staged")
        os.system(f"git -C {path} add -A > /dev/null 2>&1")
        result = await skill.execute("git_diff", {"path": path, "staged": True})
        assert "Staged" in result


class TestGitLog:
    @pytest.mark.asyncio
    async def test_log(self, git):
        skill, path = git
        result = await skill.execute("git_log", {"path": path, "limit": 5})
        assert "init" in result

    @pytest.mark.asyncio
    async def test_oneline(self, git):
        skill, path = git
        result = await skill.execute("git_log", {"path": path, "oneline": True})
        assert "init" in result


class TestGitCommit:
    @pytest.mark.asyncio
    async def test_commit(self, git):
        skill, path = git
        (Path(path) / "file.txt").write_text("content")
        result = await skill.execute("git_commit", {"path": path, "message": "add file"})
        assert "add file" in result or "1 file" in result

    @pytest.mark.asyncio
    async def test_nothing_to_commit(self, git):
        skill, path = git
        result = await skill.execute("git_commit", {"path": path, "message": "empty"})
        assert "nothing" in result.lower() or "clean" in result.lower()

    @pytest.mark.asyncio
    async def test_commit_specific_files(self, git):
        skill, path = git
        (Path(path) / "a.txt").write_text("a")
        (Path(path) / "b.txt").write_text("b")
        result = await skill.execute(
            "git_commit", {"path": path, "message": "add a", "files": "a.txt"}
        )
        assert "add a" in result or "1 file" in result


class TestGitBranch:
    @pytest.mark.asyncio
    async def test_list_branches(self, git):
        skill, path = git
        result = await skill.execute("git_branch", {"path": path})
        assert "main" in result or "master" in result

    @pytest.mark.asyncio
    async def test_create_branch(self, git):
        skill, path = git
        result = await skill.execute("git_branch", {"path": path, "create": "feature-x"})
        assert "feature-x" in result

    @pytest.mark.asyncio
    async def test_switch_branch(self, git):
        skill, path = git
        os.system(f"git -C {path} branch feature-y > /dev/null 2>&1")
        result = await skill.execute("git_branch", {"path": path, "switch": "feature-y"})
        assert "feature-y" in result
