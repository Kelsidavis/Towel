"""Tests for the search skill."""

import pytest

from towel.skills.builtin.search import SearchSkill


@pytest.fixture
def skill():
    return SearchSkill()


class TestSearchSkillTools:
    def test_tools_defined(self, skill):
        names = {t.name for t in skill.tools()}
        assert names == {"search_files", "find_files"}


class TestSearchFiles:
    @pytest.mark.asyncio
    async def test_basic_search(self, skill, tmp_path):
        (tmp_path / "hello.py").write_text("def greet():\n    print('hello world')\n")
        (tmp_path / "other.py").write_text("def bye():\n    pass\n")

        result = await skill.execute("search_files", {"pattern": "hello", "path": str(tmp_path)})
        assert "hello" in result
        assert "hello.py" in result
        assert "other.py" not in result

    @pytest.mark.asyncio
    async def test_regex_search(self, skill, tmp_path):
        (tmp_path / "code.py").write_text("x = 42\ny = 100\nz = 7\n")
        result = await skill.execute("search_files", {"pattern": r"\d{3}", "path": str(tmp_path)})
        assert "100" in result

    @pytest.mark.asyncio
    async def test_case_insensitive(self, skill, tmp_path):
        (tmp_path / "f.txt").write_text("Hello World\n")
        result = await skill.execute("search_files", {"pattern": "hello", "path": str(tmp_path)})
        assert "Hello" in result

    @pytest.mark.asyncio
    async def test_glob_filter(self, skill, tmp_path):
        (tmp_path / "a.py").write_text("target\n")
        (tmp_path / "b.txt").write_text("target\n")
        result = await skill.execute(
            "search_files", {"pattern": "target", "path": str(tmp_path), "glob": "*.py"}
        )
        assert "a.py" in result
        assert "b.txt" not in result

    @pytest.mark.asyncio
    async def test_no_matches(self, skill, tmp_path):
        (tmp_path / "f.txt").write_text("nothing here\n")
        result = await skill.execute("search_files", {"pattern": "zzzzz", "path": str(tmp_path)})
        assert "No matches" in result

    @pytest.mark.asyncio
    async def test_skips_git_dir(self, skill, tmp_path):
        git = tmp_path / ".git"
        git.mkdir()
        (git / "config").write_text("target\n")
        (tmp_path / "real.txt").write_text("other\n")
        result = await skill.execute("search_files", {"pattern": "target", "path": str(tmp_path)})
        assert "No matches" in result


class TestFindFiles:
    def test_find(self, skill, tmp_path):
        (tmp_path / "a.py").touch()
        (tmp_path / "b.py").touch()
        (tmp_path / "c.txt").touch()
        result = skill._find("*.py", str(tmp_path))
        assert "a.py" in result
        assert "b.py" in result
        assert "c.txt" not in result

    def test_find_no_match(self, skill, tmp_path):
        result = skill._find("*.rs", str(tmp_path))
        assert "No files" in result
